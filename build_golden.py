"""Builds the three-tier golden dataset for offline evaluation.

Each tier exists because a different kind of ground truth is trustworthy for a different
metric, and mixing them would mean every number carried the weakest tier's caveat.

  Tier A  85 course quiz questions, with answer keys written by the course authors.
          Scores correctness, faithfulness, relevancy and citation accuracy. It has no
          chunk-level labels: labelling which chunk "should" have been retrieved for a
          quiz question needs a judge, and a judge that shares a blind spot with the
          generator can mark a question unanswerable and then score the resulting
          refusal as correct.

  Tier B  ~40 questions generated from one chunk each, so that chunk IS the gold label,
          exactly and by construction. Carries precision@k and recall@k. Generated
          questions share vocabulary with their source and so flatter retrieval, which
          is what the overlap guard and the recorded bm25 rank are for.

  Tier C  ~12 hand-written questions about topics the corpus does not contain, verified
          absent rather than assumed. Scores the refusal path.

  uv run build_golden.py parse    quizzes      -> golden/tier_a.json
  uv run build_golden.py tierb    chunks + llm -> golden/tier_b.json
  uv run build_golden.py tierc    verify       -> checks golden/tier_c.json
  uv run build_golden.py build    merge all    -> golden/golden.json
"""
import json
import random
import re
import sys
from pathlib import Path

from llm import complete, extract_json

EVALS_DIR = Path("evals")
GOLDEN_DIR = Path("golden")
TIER_A_PATH = GOLDEN_DIR / "tier_a.json"
TIER_B_PATH = GOLDEN_DIR / "tier_b.json"
TIER_C_PATH = GOLDEN_DIR / "tier_c.json"
GOLDEN_PATH = GOLDEN_DIR / "golden.json"

# quiz folder -> the course value stored in the qdrant payload
COURSES = {
    "04 - Machine Learning with Big Data": "machine learning",
    "05 - Graph Analytics for Big Data": "graph analytics",
}

GENERATOR_MODEL = "openai/gpt-oss-120b"
TIER_B_SIZE = 40
TIER_B_SEED = 11
# a generated question that reuses the passage's wording measures lexical overlap
# rather than retrieval, so anything above this is regenerated once and then dropped
MAX_JACCARD = 0.35

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


# ------------------------------------------------- tier B: questions built from chunks

GENERATE_PROMPT = """You are writing one exam-style question to test whether a search
system can find this passage from a lecture transcript.

PASSAGE (from the video "{source}"):
{text}

Write one question that this passage answers, and the answer.

Rules:
- Ask the question the way a student would ask it, in their own words.
- Do NOT reuse the distinctive wording or phrasing of the passage. Paraphrase.
- The question must make sense on its own, without the passage in front of you.
- The answer must be stated in the passage, and must be one or two sentences.
- Do not refer to "the passage", "the video" or "this lecture" in the question.

Reply with only JSON: {{"question": "...", "answer": "..."}}"""


def generate_question(chunk, attempt=0):
    """Asks the llm for one question answered by this chunk."""
    prompt = GENERATE_PROMPT.format(source=chunk["source"], text=chunk["text"])
    if attempt:
        prompt += "\n\nYour previous attempt copied the passage's wording. Paraphrase much more."

    payload = extract_json(complete(GENERATOR_MODEL, prompt))
    if not isinstance(payload, dict):
        return None
    question = (payload.get("question") or "").strip()
    answer = (payload.get("answer") or "").strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer}


def gold_for(chunk, answer, chunks):
    """The source chunk, plus any other chunk holding the same supporting sentence.

    chunking() carries CHUNK_OVERLAP segments between consecutive chunks, so when the
    answer sits in that shared region the neighbouring chunk answers the question just
    as well and must not be scored as a miss.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.?!]) +", chunk["text"]) if len(s.strip()) > 25]
    answer_words = content_words(answer)
    best = max(sentences, key=lambda s: jaccard(content_words(s), answer_words), default="")

    gold = {chunk["chunk_id"]}
    if best:
        gold |= {c["chunk_id"] for c in chunks if best in c["text"]}
    return sorted(gold)


def sample_chunks(chunks, tier_a):
    """Picks chunks to build questions from, covering videos the quizzes never touch."""
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenise(c["text"]) for c in chunks])
    covered = set()
    for q in tier_a:
        scores = bm25.get_scores(tokenise(f"{q['question']} {q['answer']}"))
        top = sorted(range(len(chunks)), key=lambda j: scores[j], reverse=True)[:3]
        covered |= {chunks[j]["source"] for j in top}

    uncovered = sorted({c["source"] for c in chunks} - covered)
    print(f"  videos the quizzes never touch: {len(uncovered)}")

    random.seed(TIER_B_SEED)
    picked = []
    # one chunk from each unmeasured video first, so tier B fills the coverage gap
    for source in uncovered:
        picked.append(random.choice([c for c in chunks if c["source"] == source]))

    remaining = [c for c in chunks if c["chunk_id"] not in {p["chunk_id"] for p in picked}]
    picked += random.sample(remaining, max(0, min(len(remaining), TIER_B_SIZE - len(picked))))
    return picked[:TIER_B_SIZE], bm25


def build_tier_b():
    if not TIER_A_PATH.exists():
        sys.exit(f"{TIER_A_PATH} not found; run `parse` first")

    tier_a = json.loads(TIER_A_PATH.read_text(encoding="utf-8"))
    chunks = load_chunks()
    print(f"{len(chunks)} chunks in the corpus")

    picked, bm25 = sample_chunks(chunks, tier_a)
    print(f"  generating from {len(picked)} chunks")

    rows = []
    dropped = 0
    for i, chunk in enumerate(picked, 1):
        chunk_words = content_words(chunk["text"])

        for attempt in range(2):
            generated = generate_question(chunk, attempt)
            if not generated:
                continue
            overlap = jaccard(content_words(generated["question"]), chunk_words)
            if overlap <= MAX_JACCARD:
                break
            print(f"  overlap {overlap:.2f} too high, regenerating")
        else:
            generated = None

        if not generated:
            dropped += 1
            print(f"[{i}/{len(picked)}] {chunk['chunk_id']}: dropped")
            continue

        # where the source chunk ranks lexically for its own question: mostly rank 1
        # would mean the set is too easy, so it is recorded rather than assumed
        scores = bm25.get_scores(tokenise(generated["question"]))
        ranked = sorted(range(len(chunks)), key=lambda j: scores[j], reverse=True)
        rank = next((r + 1 for r, j in enumerate(ranked)
                     if chunks[j]["chunk_id"] == chunk["chunk_id"]), None)

        rows.append({
            "id": f"B-{chunk['chunk_id']}",
            "tier": "B",
            "course": chunk["course"],
            "source_chunk": chunk["chunk_id"],
            "question": generated["question"],
            "answer": generated["answer"],
            "gold_chunks": gold_for(chunk, generated["answer"], chunks),
            "jaccard": round(jaccard(content_words(generated["question"]), chunk_words), 3),
            "bm25_rank": rank,
            "hands_on": False,
        })
        print(f"[{i}/{len(picked)}] {chunk['chunk_id']}: "
              f"jaccard={rows[-1]['jaccard']} bm25_rank={rank} "
              f"gold={len(rows[-1]['gold_chunks'])}")

    GOLDEN_DIR.mkdir(exist_ok=True)
    TIER_B_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    ranks = [r["bm25_rank"] for r in rows if r["bm25_rank"]]
    print(f"\nTier B: {len(rows)} questions -> {TIER_B_PATH} ({dropped} dropped)")
    print(f"  mean jaccard with source: "
          f"{sum(r['jaccard'] for r in rows)/len(rows):.3f}" if rows else "  no rows")
    if ranks:
        print(f"  source chunk at bm25 rank 1: {sum(1 for r in ranks if r == 1)}/{len(ranks)}"
              f"  (high means the questions are lexically easy)")
    print(f"  questions with more than one gold chunk: "
          f"{sum(1 for r in rows if len(r['gold_chunks']) > 1)}")


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
    for path, name in ((TIER_A_PATH, "A"), (TIER_B_PATH, "B"), (TIER_C_PATH, "C")):
        if not path.exists():
            print(f"  warning: {path} missing, tier {name} omitted")
            continue
        tiers += json.loads(path.read_text(encoding="utf-8"))

    GOLDEN_DIR.mkdir(exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(tiers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}: {len(tiers)} questions")
    for tier in ("A", "B", "C"):
        count = sum(1 for row in tiers if row["tier"] == tier)
        print(f"  tier {tier}: {count}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "parse"
    if stage == "parse":
        parse_all()
    elif stage == "tierb":
        build_tier_b()
    elif stage == "tierc":
        verify_tier_c()
    elif stage == "build":
        build()
    else:
        sys.exit("usage: build_golden.py [parse|tierb|tierc|build]")
