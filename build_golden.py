"""Builds the two-tier golden dataset for offline evaluation.

Each tier exists because a different kind of ground truth is trustworthy for a different
metric, and mixing them would mean every number carried the weakest tier's caveat.

  Tier A  83 course quiz questions, with answer keys written by the course authors.
          Scores correctness, faithfulness, relevancy and citation validity. Its gold
          chunks are labelled by hand, question by question, from a shortlist two
          mechanical signals rank: that is the `label` and `gold` stages below, and it
          is what carries the retrieval metrics.

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
  uv run build_golden.py label    shortlists   -> golden/labelsheet.md, for a human to
                                                  mark; rerunning keeps the marks
  uv run build_golden.py gold     the marks    -> gold chunks into golden/tier_a.json
                                                  and golden/golden.json
  uv run build_golden.py build    merge all    -> golden/golden.json

Labelling tier A is more than one sitting's work, and every stage here supports being
part-way through it: an unmarked question is counted as unlabelled and left out of every
retrieval figure, so the report is correct at every point along the way.
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
LABELSHEET_PATH = GOLDEN_DIR / "labelsheet.md"

# How many candidates each signal contributes to a shortlist. The two overlap heavily,
# so five each lands around eight distinct chunks — one screen per question, which is
# what keeps 83 questions markable at all.
BM25_N = 5
DENSE_N = 5

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

# An answer key that is itself an "all of the above" option. `OPTION_DEPENDENT_RE` above
# catches these when the giveaway is in the question, but on three quiz questions it sat
# only in the key — "All of these statements are true." — so the question survived parsing
# as a bare stem with an answer naming options that are no longer there. Anchored at the
# start, so an answer that merely mentions all of several things is untouched.
ANSWER_IS_ALL_OPTIONS_RE = re.compile(r"^all of (these|the above|them)\b", re.I)

# Quizzes that walk through a KNIME or Spark exercise rather than lecture content. Every
# answer in them is a figure from the exercise's own dataset — a percentage, a cluster
# number, a column name — so no transcript chunk can answer them and no retrieval metric
# over them would mean anything. They are dropped at parse rather than labelled: tier A's
# ground truth is that the corpus can answer the question, per ADR 0001, and these do not
# meet it. Removed on the labeller's instruction, 2026-10-23.
EXCLUDED_QUIZZES = {
    "Quiz 7 - Classification in KNIME and Spark":
        "every question asks for a figure produced by running the KNIME or Spark "
        "exercise, not for anything the lectures state",
    "Quiz 9 - Model Evaluation in KNIME and Spark":
        "as Quiz 7: the answers are accuracy figures and confusion matrix cells from "
        "the exercise's own run",
}

# The same kind of question, inside quizzes that are otherwise lecture-based. Each was
# marked `[-]` on the label sheet by a human who looked through its candidate chunks for
# a gold chunk and found the corpus holds none; `golden/labelsheet.md` is that evidence.
# Listed by id with its reason rather than caught by a rule, because "the answer is a
# figure from the exercise dataset" has no reliable spelling.
#
# One entry is evidenced differently and says so, because excluding a question deletes
# its entry from the sheet on the next `label` run: the note instructing its removal goes
# in the reason here, quoted, since the sheet it was written on no longer holds it. ADR
# 0005 records why that is the only durable place for it.
EXCLUDED_QUESTIONS = {
    "A-Quiz 11 - Cluster Analysis in Spark-1":
        "answer is a proportion computed from the weather dataset (99.4%)",
    "A-Quiz 11 - Cluster Analysis in Spark-5":
        "answer is a cluster number from the labeller's own k-means run",
    "A-Quiz 11 - Cluster Analysis in Spark-6":
        "answer describes the contents of specific clusters from that run",
    "A-Quiz 11 - Cluster Analysis in Spark-7":
        "answer is a cluster number from a 20-cluster run with a given seed",
    "A-Quiz 5 - Handling Missing Values in KNIME and Spark-4":
        "answer is the name of a column in the exercise's dataset",
    "A-Quiz 4 - Graph Analytics with Neo4j-2":
        "answer is what a Cypher query returns for the exercise's own graph, which no "
        "lecture states; removed 2026-10-27 on the labeller's note, which read "
        "\"Remove this as its an hands on question\" and was written on the sheet entry "
        "that this exclusion then deleted",
}

# Questions restated in the tier because the authors' wording cannot stand without the
# options that parsing drops — "which of the statements below is true" names a list that
# is no longer there. `keep_question` catches most of these and drops them; one got past
# it and the labeller chose to restate it rather than lose the question, so the deviation
# is declared here instead of being edited into the quiz file under `evals/`.
#
# This moves correctness off the authors' own answer key for the question named, which is
# the anchor the tier A figures rest on, so each entry carries its reason and the original
# wording is kept on the row by `apply_rephrasings`. A rephrasing whose id no longer
# parses stops the stage rather than silently lapsing: the sheet would go on showing the
# restated question while the tier scored the original.
REPHRASED_QUESTIONS = {
    "A-Quiz 2 - Graph Analytics Applications-16": {
        "question": "Is Dijsktra's algorithm computationally inefficient?",
        "answer": "Yes. Dijsktra's algorithm has high computational complexity.",
        "reason": "the original was a 'which of the statements below is true' stem whose "
                  "options parsing had already dropped, leaving nothing to answer; "
                  "restated by the labeller, 2026-10-27",
    },
}


def apply_rephrasings(rows, table=None):
    """Parsed rows with the declared rephrasings applied, originals kept beside them.

    The authors' own question and answer key ride along under `original` so that a row
    restated here is distinguishable from one carrying their words, wherever the row is
    read. An id in the table that matches no row exits rather than returning quietly,
    because a rephrasing that stops applying leaves the label sheet and the tier stating
    two different questions, with nothing to show which was scored.
    """
    table = REPHRASED_QUESTIONS if table is None else table
    by_id = {row["id"]: row for row in rows}

    missing = sorted(qid for qid in table if qid not in by_id)
    if missing:
        sys.exit("rephrased question id(s) match no parsed question: "
                 + ", ".join(missing)
                 + "\nA rephrasing is pinned to an id. Remove the entry from "
                   "REPHRASED_QUESTIONS, or correct the id.")

    for qid, restated in table.items():
        row = by_id[qid]
        row["original"] = {"question": row["question"],
                           "answer": row["answer"],
                           "reason": restated["reason"]}
        row["question"] = restated["question"]
        row["answer"] = restated["answer"]

    return rows


def is_excluded(qid):
    """Whether a question id was deliberately dropped from tier A, and not merely absent.

    A question removed on purpose is not a lost label, so reading a mark for one is not
    an error — which is the difference `label_rows`' orphan check needs to be told about.
    """
    return (qid in EXCLUDED_QUESTIONS
            or any(qid.startswith(f"A-{quiz}-") for quiz in EXCLUDED_QUIZZES))

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


# Mechanical overlap signals, all three of them live again in the labelling path:
# `tokenise` feeds the bm25 index `bm25_ranker` builds, and `content_words` and `jaccard`
# annotate each candidate with its overlap against the question and its answer key. They
# are mechanical on purpose — a model may not be asked to rank the candidates instead.
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
    if ANSWER_IS_ALL_OPTIONS_RE.match(answer.strip()):
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
    excluded = []
    for folder, course in COURSES.items():
        for path in sorted((EVALS_DIR / folder).glob("Quiz*.md")):
            if path.stem in EXCLUDED_QUIZZES:
                excluded.append(f"{path.stem} (whole quiz): {EXCLUDED_QUIZZES[path.stem]}")
                continue
            found = parse_quiz(path, course)
            if not found:
                skipped.append(path.name)
            for row in found:
                if row["id"] in EXCLUDED_QUESTIONS:
                    excluded.append(f"{row['id']}: {EXCLUDED_QUESTIONS[row['id']]}")
                else:
                    rows.append(row)

    rows = apply_rephrasings(rows)

    GOLDEN_DIR.mkdir(exist_ok=True)
    TIER_A_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tier A: {len(rows)} questions -> {TIER_A_PATH}")
    for course in sorted(COURSES.values()):
        print(f"  {course}: {sum(1 for r in rows if r['course'] == course)}")
    print(f"  hands-on flagged: {sum(1 for r in rows if r['hands_on'])}")
    print(f"  multi-answer:     {sum(1 for r in rows if r['multi_answer'])}")
    for name in skipped:
        print(f"  skipped (no usable questions): {name}")
    for line in excluded:
        print(f"  excluded: {line}")
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


# ------------------------------------- tier A: hand-labelled gold chunks (issue #12)

# The two marked lines, each naming its own id so that no verdict depends on the line's
# position in the file or on how an editor reflowed the indentation around it. The word
# `question` or `chunk` is what tells the two apart.
#
# Whatever sits between the brackets is captured, however long, and is then stripped and
# looked up. Matching only a single character was worse than strict: `[x ]`, which is
# what several editors leave behind, matched neither pattern, so the line was read by
# nobody and a tick on it disappeared without a word — the exact failure the vocabulary
# is total to prevent. Surrounding whitespace is therefore insignificant (`[]`, `[ ]` and
# `[ x ]` all mean what they look like) and anything else raises.
QUESTION_MARK_RE = re.compile(r"^- \[([^\]]*)\] question `([^`]+)`")
# A line the human wrote themselves: why several chunks are ticked, why one that looks
# right was rejected, why a question should go. It is reasoning no signal on the sheet
# carries, so it is read back, written back on regeneration, and stored on the row.
NOTE_RE = re.compile(r"^\*\*Note:\*\*\s*(.+)$")
# `## `the question id`` opens a question's section. It is what tells a note written
# above a question line from one written below the previous question's candidates.
SECTION_RE = re.compile(r"^## `([^`]+)`")
CANDIDATE_MARK_RE = re.compile(r"^\s*- \[([^\]]*)\] chunk `([^`]+)`")

# Read, never written: ADR 0001 keeps a model out of every labelling decision, and which
# chunk a question needed is one. The vocabulary is total — an unlisted mark raises — so
# the question marked with it stays in view instead of joining the unlabelled count.
QUESTION_MARKS = {"": "unlabelled", "x": "labelled", "-": "no_gold_in_corpus"}
CANDIDATE_MARKS = {"": False, "x": True}


def labelsheet_marks(sheet):
    """Each question's verdict and its ticked gold chunks, read back from the sheet.

    Ticking a candidate is itself the statement that a question is labelled, so marking
    83 questions costs one tick per gold chunk and nothing else. The question's own mark
    is for the one verdict no tick can express: `[-]`, no chunk in the corpus answers
    this question. `[x]` is accepted as an explicit "labelled" for anyone who prefers to
    write it, and means exactly what the ticks below it already say.

    Anything else raises, with the offending ids named. The sheet is the only copy of a
    human's labels and cannot be regenerated from another artifact, so a mark this
    function cannot read is a reason to stop rather than to carry on over a quietly
    smaller denominator. A verdict that contradicts its ticks raises for the same reason:
    `[-]` beside a ticked chunk means either the verdict or the tick is a slip, and
    guessing which would silently discard a judgement nobody made.
    """
    verdicts = {}
    unknown = []
    current = None
    pending_notes = []

    for line in sheet.splitlines():
        if SECTION_RE.match(line):
            # a new section closes the previous question, so a note below this line
            # belongs to the question this section is about and not to the last one
            current, pending_notes = None, []
            continue

        found = NOTE_RE.match(line)
        if found:
            # notes are written above the question line in a generated sheet and below
            # it by hand just as readily; both are kept, in the order they were written
            note = found.group(1).strip()
            (current["notes"] if current else pending_notes).append(note)
            continue

        found = QUESTION_MARK_RE.match(line)
        if found:
            mark, qid = found.group(1).strip().lower(), found.group(2)
            if mark not in QUESTION_MARKS:
                unknown.append(f"question {qid} marked [{found.group(1)}]")
                current, pending_notes = None, []
                continue
            current = {"status": QUESTION_MARKS[mark], "gold": [],
                       "notes": pending_notes}
            pending_notes = []
            verdicts[qid] = current
            continue

        found = CANDIDATE_MARK_RE.match(line)
        if not found or current is None:
            continue
        mark, cid = found.group(1).strip().lower(), found.group(2)
        if mark not in CANDIDATE_MARKS:
            unknown.append(f"chunk {cid} marked [{found.group(1)}]")
            continue
        if CANDIDATE_MARKS[mark]:
            current["gold"].append(cid)

    if unknown:
        raise ValueError(
            "the label sheet carries marks this stage does not recognise, so the "
            "retrieval figures would be taken over fewer questions than were labelled: "
            + "; ".join(unknown)
            + f". Use one of {sorted(QUESTION_MARKS)} on a question line and "
              f"{sorted(CANDIDATE_MARKS)} on a chunk line.")

    contradictions = []
    for qid, verdict in verdicts.items():
        if verdict["gold"] and verdict["status"] == "no_gold_in_corpus":
            contradictions.append(
                f"{qid} is marked [-], no gold chunk in the corpus, but has "
                f"{len(verdict['gold'])} chunk(s) ticked below it")
        elif verdict["gold"]:
            verdict["status"] = "labelled"
        elif verdict["status"] == "labelled":
            contradictions.append(
                f"{qid} is marked [x], labelled, but ticks no chunk as gold")
    if contradictions:
        raise ValueError(
            "the label sheet carries verdicts that contradict their ticks, and which of "
            "the two is the slip cannot be guessed: " + "; ".join(contradictions))

    return verdicts


SHEET_HEADER = """# Tier A retrieval labels

Each question below is followed by a shortlist of candidate chunks. Tick the chunks the
question needed, then run `uv run build_golden.py gold` to write the labels into
`golden/tier_a.json` and `uv run evaluate.py rescore` to report the figures.

  - tick `- [x] chunk ...`   this chunk is a gold chunk for the question above it
  - tick as many as apply    a question whose answer spans chunks has several
  - `- [-] question ...`      no chunk in the corpus answers this question
  - leave a question alone   not yet labelled, and left out of every figure

Ticking a chunk is itself the statement that a question is labelled; the question's own
mark is only for `[-]`. Leaving a question untouched is not a verdict: it is counted as
unlabelled and excluded, so the report is correct part-way through. There is no need to
finish in one sitting.

The shortlist is mechanical. bm25 and embedding similarity order it and print their
ranks, and nothing else decides anything: which chunk is gold is yours to say, per
`docs/adr/0001-three-tier-golden-dataset.md`. If the chunk a question needed is not in
its shortlist, add a line of your own beneath it and tick that —

      - [x] chunk `some video.mp4:4`

— the id is what is read, and a regenerated sheet keeps it.

An unrecognised mark stops the stage rather than being skipped, naming the id, because
this file is the only copy of these labels.
"""

LABEL_FIELDS = ("gold_chunks", "no_gold_in_corpus", "label_note")


def render_labelsheet(entries, marks):
    """The label sheet, with every mark the previous sheet carried written back on.

    Preservation is by id, both question and chunk, so the shortlist may be widened,
    narrowed or reordered between runs without touching a verdict. A chunk that was
    ticked and has since left the shortlist is still rendered, and still ticked, below
    the current candidates: a shortlist is mechanical and may change, while a tick on it
    is a judgement that cannot be recovered from anywhere else.
    """
    lines = [SHEET_HEADER]
    for entry in entries:
        verdict = marks.get(entry["id"], {"status": "unlabelled", "gold": [],
                                          "notes": []})
        gold = set(verdict["gold"])
        question_mark = "-" if verdict["status"] == "no_gold_in_corpus" else " "

        lines += [
            f"## `{entry['id']}`",
            "",
            f"**Q:** {entry['question']}",
            "",
            f"**Answer key:** {entry['answer']}",
            "",
        ]
        # the human's own lines, written back above the marks exactly where they put
        # them: a note is work the sheet holds the only copy of, like a tick
        for note in verdict.get("notes", ()):
            lines += [f"**Note:** {note}", ""]
        lines += [
            f"- [{question_mark}] question `{entry['id']}` — "
            f"{entry['quiz']}{', hands-on' if entry['hands_on'] else ''}",
        ]

        shown = []
        for candidate in entry["candidates"]:
            cid = candidate["chunk_id"]
            shown.append(cid)
            if candidate["bm25_rank"] is None:
                signals = "bm25 outside its shortlist"
            else:
                signals = (f"bm25 #{candidate['bm25_rank']} "
                           f"({candidate['bm25_score']:.2f})")
            if candidate["dense_rank"] is not None:
                signals += f" · dense #{candidate['dense_rank']}"
            if candidate["jaccard"] is not None:
                signals += f" · jaccard {candidate['jaccard']}"
            lines += [
                f"  - [{'x' if cid in gold else ' '}] chunk `{cid}` · {signals}",
                f"    > {candidate['text']}",
            ]

        # a tick the current shortlist no longer offers, kept so that regeneration
        # cannot erase a label the shortlist has moved away from
        for cid in verdict["gold"]:
            if cid not in shown:
                lines.append(f"  - [x] chunk `{cid}` · kept from an earlier sheet")
        lines.append("")

    return "\n".join(lines) + "\n"


def label_rows(rows, marks, known_chunks=None):
    """Tier A rows with the sheet's verdicts written on, replacing any earlier label.

    Both fields are cleared before either is written, so a question relabelled from gold
    chunks to no-gold-in-corpus does not keep scoring recall against chunks it no longer
    claims. Labelling 83 questions takes more than one sitting and verdicts are revisited
    inside it, which makes a stale label a thing that happens rather than a thing to
    assume away.

    A mark naming a question the tier no longer holds raises, and so does a gold chunk id
    that is not in the corpus when the corpus is passed. Both are judgements that would
    otherwise be stored where nothing can read them.
    """
    tier_ids = {row["id"] for row in rows}
    orphans = sorted(set(marks) - tier_ids)
    if orphans:
        raise ValueError(
            "the label sheet marks questions tier A no longer holds, so those labels "
            f"would be lost: {', '.join(orphans)}. Re-run `label` to rebuild the sheet "
            "against the current tier, which preserves the marks of the questions that "
            "remain.")

    if known_chunks is not None:
        missing = sorted({cid for verdict in marks.values() for cid in verdict["gold"]
                          if cid not in known_chunks})
        if missing:
            raise ValueError(
                "the label sheet ticks chunk ids that are not in the corpus, which "
                f"retrieval could never return: {', '.join(missing)}")

    labelled = []
    for row in rows:
        row = {key: value for key, value in row.items() if key not in LABEL_FIELDS}
        verdict = marks.get(row["id"])
        if verdict and verdict["gold"]:
            row["gold_chunks"] = list(verdict["gold"])
        elif verdict and verdict["status"] == "no_gold_in_corpus":
            row["no_gold_in_corpus"] = True
        # the reasoning behind a label, stored beside it so it outlives the sheet
        if verdict and verdict.get("notes"):
            row["label_note"] = " ".join(verdict["notes"])
        labelled.append(row)
    return labelled


def bm25_ranker(chunks):
    """A function from a query to its top-ranked chunk ids, over one index built here.

    The index is built once for all 83 questions rather than per question: `load_chunks`
    and the tokenising are the slow part, and the ranking itself is cheap.

    This is the opposite direction from the bm25 the retired tier used, which drew a
    seeded random sample of chunks to generate questions *from*. That went with the tier;
    `docs/adr/0003-retiring-the-generated-retrieval-tier.md` records the deviation. What
    is needed here is: given one question, rank the chunks.
    """
    from rank_bm25 import BM25Okapi

    index = BM25Okapi([tokenise(chunk["text"]) for chunk in chunks])

    def top(query, n=BM25_N):
        scores = index.get_scores(tokenise(query))
        order = sorted(range(len(chunks)), key=lambda j: scores[j], reverse=True)[:n]
        return [(chunks[j]["chunk_id"], scores[j], rank)
                for rank, j in enumerate(order, 1)]

    return top


def shortlist(bm25_hits, dense_ids, by_id, query_words):
    """The candidates for one question: the two signals' top-N, merged and annotated.

    Ordered by bm25 rank, with the chunks only the dense half found appended. Both ranks
    are printed beside each candidate because they disagree often, and a candidate the
    embedding ranked first and bm25 did not rank at all is exactly the one worth reading.
    Jaccard overlap with the question and its answer key is printed as a third signal: it
    is the one that reads the answer key, so a chunk that shares the key's wording is
    worth looking at first even when neither ranking favours it.

    No signal decides anything. They order and annotate the shortlist so that labelling a
    question means reading about eight chunks instead of all 406; which of them is gold is
    the human's to say.
    """
    ranks = {cid: (score, rank) for cid, score, rank in bm25_hits}
    dense = {cid: rank for rank, cid in enumerate(dense_ids, 1)}

    candidates = []
    for cid in list(ranks) + [cid for cid in dense if cid not in ranks]:
        score, rank = ranks.get(cid, (None, None))
        text = by_id[cid]["text"] if cid in by_id else "(chunk missing)"
        candidates.append({
            "chunk_id": cid,
            "text": text,
            "bm25_score": score,
            "bm25_rank": rank,
            "dense_rank": dense.get(cid),
            "jaccard": round(jaccard(query_words, content_words(text)), 3),
        })
    return candidates


def label():
    """Writes the sheet a human marks gold chunks on, keeping any marks already made."""
    if not TIER_A_PATH.exists():
        sys.exit(f"{TIER_A_PATH} not found; run `parse` first")

    from evaluate import retrieve

    rows = json.loads(TIER_A_PATH.read_text(encoding="utf-8"))
    chunks = load_chunks()
    if not chunks:
        sys.exit("no chunks were rebuilt; corpus.json and cache/ must both be present")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ranker = bm25_ranker(chunks)

    marks = {}
    if LABELSHEET_PATH.exists():
        marks = labelsheet_marks(LABELSHEET_PATH.read_text(encoding="utf-8"))
        print(f"preserving marks on {sum(1 for v in marks.values() if v['gold'])} "
              f"labelled and "
              f"{sum(1 for v in marks.values() if v['status'] == 'no_gold_in_corpus')} "
              f"no-gold questions already on the sheet")

    entries = []
    for i, row in enumerate(rows, 1):
        # bm25 is given the answer key as well as the question: it is the signal that
        # has to surface a chunk the embedding missed, and the key names the material.
        # The dense half is given the question alone, which is what the pipeline
        # retrieves on, so the sheet shows the shortlist the pipeline itself would see.
        query = f"{row['question']} {row['answer']}"
        bm25_hits = ranker(query)
        dense_ids = [cid for cid, _payload in retrieve(row["question"], limit=DENSE_N)]
        entries.append({**row, "candidates": shortlist(bm25_hits, dense_ids, by_id,
                                                       content_words(query))})
        print(f"[{i}/{len(rows)}] {row['id'][:50]} "
              f"{len(entries[-1]['candidates'])} candidates")

    GOLDEN_DIR.mkdir(exist_ok=True)
    LABELSHEET_PATH.write_text(render_labelsheet(entries, marks), encoding="utf-8")
    sizes = [len(e["candidates"]) for e in entries]
    print(f"\nWrote {LABELSHEET_PATH}: {len(entries)} questions, "
          f"{sum(sizes)} candidate chunks "
          f"({min(sizes)}-{max(sizes)} per question)")
    print("  mark it, then run `uv run build_golden.py gold`")


def write_gold():
    """Reads the marked sheet back and writes the labels into the tier and golden set."""
    if not LABELSHEET_PATH.exists():
        sys.exit(f"{LABELSHEET_PATH} not found; run `label` first")
    if not TIER_A_PATH.exists():
        sys.exit(f"{TIER_A_PATH} not found; run `parse` first")

    marks = labelsheet_marks(LABELSHEET_PATH.read_text(encoding="utf-8"))

    # a mark on a question that was deliberately excluded is an honoured judgement, not
    # a lost one, so it is dropped quietly here rather than raising the orphan check
    dropped = sorted(qid for qid in marks if is_excluded(qid))
    marks = {qid: verdict for qid, verdict in marks.items() if qid not in dropped}
    if dropped:
        print(f"ignoring marks on {len(dropped)} question(s) excluded from tier A:")
        for qid in dropped:
            print(f"  {qid}")

    rows = json.loads(TIER_A_PATH.read_text(encoding="utf-8"))
    chunks = load_chunks()
    rows = label_rows(rows, marks,
                      known_chunks={c["chunk_id"] for c in chunks} if chunks else None)

    TIER_A_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    # the report's own count, so that what this stage prints and what the report prints
    # cannot drift apart on which rows are labelled
    from evaluate import label_counts

    counts = label_counts(rows)
    labelled = [r for r in rows if r.get("gold_chunks")]
    print(f"Tier A labels -> {TIER_A_PATH}")
    print(f"  labelled with gold chunks:   {counts['labelled']} of {counts['questions']}")
    print(f"  gold chunks per labelled question: "
          f"{sum(len(r['gold_chunks']) for r in labelled) / len(labelled):.2f}"
          if labelled else "  gold chunks per labelled question: n/a")
    print(f"  no gold chunk in the corpus: {counts['no_gold_in_corpus']}")
    print(f"  not yet labelled:            {counts['unlabelled']}")

    # the golden set is what `evaluate.py` reads, so it is refreshed here rather than
    # left for a separate `build` the labels would otherwise be invisible without
    build()
    print("\nRun `uv run evaluate.py rescore` to report retrieval over the labelled "
          "questions.")


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
    elif stage == "label":
        label()
    elif stage == "gold":
        write_gold()
    elif stage == "build":
        build()
    else:
        sys.exit("usage: build_golden.py [parse|tierc|label|gold|build]")
