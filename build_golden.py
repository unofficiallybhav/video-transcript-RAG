"""Builds the two-tier golden dataset for offline evaluation.

Each tier exists because a different kind of ground truth is trustworthy for a different
metric, and mixing them would mean every number carried the weakest tier's caveat.

  Tier A  83 course quiz questions, with answer keys written by the course authors.
          Scores correctness, faithfulness, relevancy and citation validity. It has no
          chunk-level labels yet, so nothing here scores retrieval.

  Tier C  ~12 hand-written questions about topics the corpus does not contain, verified
          absent rather than assumed. Scores the refusal path.

A third tier generated its questions from one chunk each, so that chunk was the gold
label by construction, and it carried every retrieval metric. A human review of 15 of
its 40 questions rejected 2 as not answerable from the passage they were built from, and
it was retired rather than regenerated: construction guarantees the label is right while
guaranteeing nothing about whether the question is answerable or worth asking. See
`docs/adr/0003-retiring-the-generated-retrieval-tier.md`. Retrieval gold is to be
hand-labelled onto tier A instead, which is the one admissible label source this project
has not used.

  uv run build_golden.py parse    quizzes      -> golden/tier_a.json
  uv run build_golden.py tierc    verify       -> checks golden/tier_c.json
  uv run build_golden.py build    merge all    -> golden/golden.json
"""
import json
import re
import sys
from pathlib import Path

EVALS_DIR = Path("evals")
GOLDEN_DIR = Path("golden")
TIER_A_PATH = GOLDEN_DIR / "tier_a.json"
TIER_C_PATH = GOLDEN_DIR / "tier_c.json"
GOLDEN_PATH = GOLDEN_DIR / "golden.json"

# quiz folder -> the course value stored in the qdrant payload
COURSES = {
    "04 - Machine Learning with Big Data": "machine learning",
    "05 - Graph Analytics for Big Data": "graph analytics",
}


# "### 1. text", "#### 1. text" or a bare "1. text"
QUESTION_RE = re.compile(r"^#{0,4}\s*(\d+)\.\s+(.*)$")
OPTION_RE = re.compile(r"^\s*-\s+(.*)$")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")

# a multiple choice question whose options have been stripped is unanswerable, and its
# gold answer is often a deliberate distractor that appears nowhere in the lectures
OPTION_DEPENDENT_RE = re.compile(
    r"which (one )?of the (following|these)|all of the following|of the above"
    r"|all of these|none of the|\bEXCEPT\b|^what is NOT\b",
    re.I,
)
NUMERIC_ANSWER_RE = re.compile(r"[\d.,%\s]+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "to",
    "for", "with", "and", "or", "but", "if", "then", "that", "this", "these", "those",
    "it", "its", "as", "at", "by", "from", "we", "you", "they", "he", "she", "do",
    "does", "did", "can", "could", "would", "should", "will", "what", "which", "who",
    "when", "where", "how", "why", "not", "no", "yes", "so", "there", "their", "has",
    "have", "had", "about", "into", "than", "more", "most", "some", "any", "all",
}


def clean(text):
    text = BOLD_RE.sub(r"\1", text)
    text = text.replace("[X]", "")
    return " ".join(text.split())


def is_correct(option):
    return "**" in option or "[X]" in option


# Mechanical overlap signals. Nothing calls them since the generated tier was retired;
# they are kept because hand-labelling gold chunks onto tier A needs exactly these to
# rank and annotate candidate chunks, and a model may not be asked to do it instead.
def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOPWORDS and len(w) > 2}


def jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def tokenise(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------- tier A: the quizzes

def parse_quiz(path, course):
    """Pulls (question, correct answers) pairs out of one quiz file."""
    lines = path.read_text(encoding="utf-8").splitlines()

    questions = []
    current = None

    for line in lines:
        heading = QUESTION_RE.match(line)
        # a numbered line that is not an option starts a new question
        if heading and not OPTION_RE.match(line):
            if current:
                questions.append(current)
            current = {"number": int(heading.group(1)), "stem": [heading.group(2)], "options": []}
            continue

        if current is None:
            continue

        option = OPTION_RE.match(line)
        if option:
            current["options"].append(option.group(1))
        elif not current["options"]:
            # still in the question body, which may run over several lines or hold code
            current["stem"].append(line)

    if current:
        questions.append(current)

    parsed = []
    for q in questions:
        question = " ".join(" ".join(q["stem"]).split())
        correct = [clean(o) for o in q["options"] if is_correct(o)]
        answer = " ; ".join(c for c in correct if c)

        # an image-only question or answer cannot be evaluated against transcripts
        if IMAGE_RE.search(question) or any(IMAGE_RE.search(o) for o in q["options"] if is_correct(o)):
            continue
        # no marked answer means no ground truth to score against
        if not answer:
            continue
        if not keep_question(question, answer):
            continue

        parsed.append({
            "id": f"A-{path.stem}-{q['number']}",
            "tier": "A",
            "course": course,
            "quiz": path.stem,
            "question": question,
            "answer": answer,
            "multi_answer": len(correct) > 1,
            # hands-on questions ask about clicking through knime or spark rather than
            # about lecture content; kept, but broken out separately in the report
            "hands_on": bool(re.search(r"\b(KNIME|Spark|GraphX|Hands On|code snippet)\b",
                                       question, re.I)),
        })

    return parsed


def keep_question(question, answer):
    """Drops questions that cannot stand on their own once the options are gone."""
    if OPTION_DEPENDENT_RE.search(question):
        return False
    if answer.strip().lower() in ("true", "false"):
        return False
    # bare numbers are answers to knime click-through exercises, not lecture content
    if NUMERIC_ANSWER_RE.fullmatch(answer.strip()):
        return False
    return True


def parse_all():
    rows = []
    skipped = []
    for folder, course in COURSES.items():
        for path in sorted((EVALS_DIR / folder).glob("Quiz*.md")):
            found = parse_quiz(path, course)
            if not found:
                skipped.append(path.name)
            rows.extend(found)

    GOLDEN_DIR.mkdir(exist_ok=True)
    TIER_A_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tier A: {len(rows)} questions -> {TIER_A_PATH}")
    for course in sorted(COURSES.values()):
        print(f"  {course}: {sum(1 for r in rows if r['course'] == course)}")
    print(f"  hands-on flagged: {sum(1 for r in rows if r['hands_on'])}")
    print(f"  multi-answer:     {sum(1 for r in rows if r['multi_answer'])}")
    for name in skipped:
        print(f"  skipped (no usable questions): {name}")
    return rows


# --------------------------------------------------- the chunk corpus, rebuilt offline

def load_chunks():
    """Rebuilds the chunk corpus from the cached transcripts, ids matching qdrant."""
    import video_transcript_rag as rag

    chunks = []
    for stem, entry in rag.load_corpus().items():
        transcript_path = rag.CACHE_DIR / f"{stem}.json"
        if not transcript_path.exists():
            continue
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        chunks.extend(rag.chunking(segments, entry["source"], entry["course"]))

    for chunk in chunks:
        chunk["chunk_id"] = f"{chunk['source']}:{chunk['chunk_index']}"
    return chunks


# ------------------------------------------- tier C: questions the corpus cannot answer

def verify_tier_c():
    """Confirms every tier C topic really is absent from the transcripts."""
    if not TIER_C_PATH.exists():
        sys.exit(f"{TIER_C_PATH} not found; it is hand-written")

    rows = json.loads(TIER_C_PATH.read_text(encoding="utf-8"))
    corpus = " ".join(c["text"].lower() for c in load_chunks())

    failures = []
    for row in rows:
        # word boundaries matter: a plain substring test matched "lora" inside
        # "exploration" and failed a question that was genuinely unanswerable
        present = [term for term in row["absent_terms"]
                   if re.search(r"\b" + re.escape(term.lower()), corpus)]
        if present:
            failures.append((row["id"], present))

    print(f"Tier C: {len(rows)} questions, {len(failures)} with terms present in the corpus")
    for qid, terms in failures:
        print(f"  FAIL {qid}: found {terms}")
    if failures:
        sys.exit("tier C questions must be unanswerable; fix the ones above")
    print("  all tier C topics confirmed absent")
    return rows


# ------------------------------------------------------------------------ merge tiers

def build():
    tiers = []
    for path, name in ((TIER_A_PATH, "A"), (TIER_C_PATH, "C")):
        if not path.exists():
            print(f"  warning: {path} missing, tier {name} omitted")
            continue
        tiers += json.loads(path.read_text(encoding="utf-8"))

    GOLDEN_DIR.mkdir(exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(tiers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}: {len(tiers)} questions")
    for tier in ("A", "C"):
        count = sum(1 for row in tiers if row["tier"] == tier)
        print(f"  tier {tier}: {count}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "parse"
    if stage == "parse":
        parse_all()
    elif stage == "tierc":
        verify_tier_c()
    elif stage == "build":
        build()
    else:
        sys.exit("usage: build_golden.py [parse|tierc|build]")
