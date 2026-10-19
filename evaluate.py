"""Offline evaluation of the video transcript rag pipeline.

Run before deploying a change: it replays the golden dataset through the real retrieval
and generation path and scores each tier on what that tier's ground truth can support.

  Tier A (quiz questions, human answer keys)
      correctness, faithfulness, relevancy, citation accuracy
  Tier B (questions generated from one chunk each, so gold labels are exact)
      precision@k, recall@k, plus the generation metrics as secondary
  Tier C (questions the corpus cannot answer, verified absent)
      refusal rate

Tier metrics are reported separately and never averaged: tier B's retrieval numbers are
exact but its questions are synthetic, tier A's questions are real but have no chunk
labels, and averaging would hide both facts.

  uv run evaluate.py run         score the pipeline, write golden/report.{json,md}
                                 resumable: rerun it after a daily limit to continue
  uv run evaluate.py report      rebuild the report from whatever is checkpointed
  uv run evaluate.py rescore     re-derive the judge-free metrics from the checkpoint
                                 offline and rewrite the report: no provider calls, no
                                 new answers, judge scores passed through untouched
  uv run evaluate.py spotcheck   emit generated tier B questions for human review
  uv run evaluate.py review      summarise the spot-check verdicts
  uv run evaluate.py calibrate   re-judge a seeded sample twice to put error bars on
                                 the judge's scores, write golden/judge_calibration.json
                                 resumable: rerun it after a daily limit to continue
"""
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from build_golden import GOLDEN_PATH, load_chunks
from llm import DailyLimitReached, complete, extract_json

GOLDEN_DIR = Path("golden")
REPORT_JSON = GOLDEN_DIR / "report.json"
REPORT_MD = GOLDEN_DIR / "report.md"
SPOTCHECK_MD = GOLDEN_DIR / "spotcheck.md"
# each scored question is appended here as it finishes, so hitting the daily token limit
# costs nothing: the next run skips what is already done
RESULTS_PATH = GOLDEN_DIR / "results.jsonl"

K_VALUES = (1, 3, 5, 10)
RETRIEVE_K = max(K_VALUES)
# the deployed chat path asks for 5 chunks, so generation is scored on 5
GENERATION_K = 5
JUDGE_MODEL = "openai/gpt-oss-20b"
SPOTCHECK_SIZE = 15
SPOTCHECK_SEED = 7

# --- judge calibration (issue #8) -------------------------------------------------
# A stratified sample of the already-scored questions, re-judged to put error bars on
# the judge-derived scores. Seed and sizes are constants so the sample is reproducible.
CALIBRATION_SEED = 11
CALIBRATION_STRATA = (("A", 20), ("B", 10))
CALIBRATION_PATH = GOLDEN_DIR / "judge_calibration.json"
# one line per (question, pass) as it finishes, so a daily limit costs no re-judging
CALIBRATION_RAW_PATH = GOLDEN_DIR / "judge_calibration.jsonl"
# Both thresholds were fixed in issue #1 and #8 before any of this was run. They are
# written into the artifact with the outcome each one triggers, so a result that lands
# the wrong way cannot afterwards be re-read as small-sample noise.
CALIBRATION_THRESHOLD = 0.10
CALIBRATION_RULES = (
    {
        "key": "between_run_spread",
        "name": "between-run spread",
        "measure": "mean absolute between-run difference, worst metric",
        "threshold": CALIBRATION_THRESHOLD,
        "comparison": "greater than",
        "if_exceeded": "two-decimal continuous scoring is retired in favour of one "
                       "decimal",
        "if_not_exceeded": "two-decimal continuous scoring is kept",
    },
    {
        "key": "faithfulness_without_reference",
        "name": "faithfulness dependence on the reference answer",
        "measure": "mean absolute faithfulness difference with the reference withheld",
        "threshold": CALIBRATION_THRESHOLD,
        "comparison": "greater than",
        "if_exceeded": "faithfulness is split into its own judge call, permanently",
        "if_not_exceeded": "faithfulness stays in the combined judge call",
    },
)
JUDGE_METRICS = ("faithfulness", "relevancy", "correctness")

# the system prompt tells the model to cite the video and timestamp, so citations are
# parsed out of the answer and checked against what was actually retrieved.
# video names are matched literally against the real filenames: a generic regex swallowed
# the words in front of the name ("In modulariy.mp4") and failed every lookup
TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2})")
# A bare name — the filename without its extension — is only a citation when a timestamp
# sits beside it, inside this many characters and without a clause boundary in between.
# The corpus forces the restriction: `classification.mp4` and `cluster analysis.mp4` are
# real videos, and across the scored answers bare mentions split 14 in citation position
# against 35 in ordinary prose, so unrestricted stem matching would manufacture more
# false citations than the defect it repairs.
# ponytail: a character window and one boundary class is the whole rule. If answers ever
# cite in a form this misses, widen the window before reaching for a sentence splitter.
CITATION_WINDOW = 20
CLAUSE_BOUNDARY_RE = re.compile(r"[.;\n]")
_video_names = []


def video_names():
    """Every video filename in the corpus, longest first so the longest match wins."""
    if not _video_names:
        import video_transcript_rag as rag

        _video_names.extend(sorted(
            (p.name for p in Path("data").rglob("*") if p.suffix.lower() in rag.VIDEO_SUFFIXES),
            key=len,
            reverse=True,
        ))
    return _video_names

JUDGE_PROMPT = """You are scoring one answer produced by a lecture-transcript question
answering system. Score each metric on a continuous scale from 0.00 to 1.00, to two
decimal places. Use the whole range: judge how much of the answer satisfies the
criterion, not just whether it passes or fails.

QUESTION: {question}

REFERENCE ANSWER (ground truth): {reference}

RETRIEVED CONTEXT GIVEN TO THE SYSTEM:
{context}

THE SYSTEM'S ANSWER: {answer}

Score these three, strictly:
- faithfulness: what proportion of the factual claims in the system's answer is
  supported by the retrieved context. 1.00 means every claim is supported; 0.00 means
  none are. A single unsupported claim among several supported ones should land high but
  below 1.00.
- relevancy: how directly the answer addresses the question asked. 1.00 means it answers
  exactly what was asked; lower it for padding, hedging or unrelated material around a
  correct answer; 0.00 means it answers something else entirely.
- correctness: how closely the answer agrees with the REFERENCE ANSWER. 1.00 means full
  agreement; partial credit for an incomplete answer or one that misses part of a
  multi-part reference; 0.00 means it contradicts the reference or declines to answer.

Also report whether the system refused to answer (said it has no information).

Reply with only JSON, with scores as decimals between 0.00 and 1.00:
{{"faithfulness": 0.00, "relevancy": 0.00, "correctness": 0.00, "refused": true|false, "note": "one short sentence explaining the scores"}}"""

REFUSAL_PROMPT = """A question answering system was asked about a topic its document
collection does not cover. The only correct behaviour is to decline.

QUESTION: {question}
THE SYSTEM'S ANSWER: {answer}

Did the system decline to answer, rather than making up an answer from unrelated material?

Reply with only JSON: {{"refused": true|false, "note": "one short sentence"}}"""


def load_golden():
    if not GOLDEN_PATH.exists():
        sys.exit(f"{GOLDEN_PATH} not found; run build_golden.py parse, tierb, build first")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def retrieve(query, limit=RETRIEVE_K):
    """The dense retrieval under test, returning ordered chunk ids and payloads."""
    import video_transcript_rag as rag

    points = rag.qdrant_client.query_points(
        collection_name=rag.COLLECTION_NAME,
        query=rag.embeddings_client.encode(query).tolist(),
        limit=limit,
        with_payload=True,
    ).points
    return [(f"{p.payload['source']}:{p.payload['chunk_index']}", p.payload) for p in points]


def precision_recall(retrieved_ids, gold_ids, k):
    gold = set(gold_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in gold)
    return hits / k, (hits / len(gold) if gold else 0.0)


def format_context(hits):
    lines = []
    for _cid, payload in hits:
        minutes, seconds = divmod(int(payload["start"]), 60)
        lines.append(f"[{payload['source']} @ {minutes:02d}:{seconds:02d}] {payload['text']}")
    return "\n".join(lines)


def _same_clause(answer, start, end):
    """Whether two positions in the answer sit in the same clause."""
    return CLAUSE_BOUNDARY_RE.search(answer, start, end) is None


def cited_mentions(answer, names, stamp_positions):
    """Every video the answer cites, as (position, filename), in order of appearance.

    This is the one definition of a citation that all three citation metrics read, so
    that they can be compared side by side. A name carrying its file extension is a
    citation wherever it appears. A bare name is a citation only when a timestamp follows
    it inside `CITATION_WINDOW` characters and in the same clause, which is what keeps
    `cluster analysis.mp4` cited at a time apart from prose about cluster analysis. A
    bare name lying inside a longer mention already accepted is not a second citation:
    naming `building and applying a classification model` does not also cite
    `classification.mp4`.
    """
    lowered = answer.lower()
    spelled, bare = [], []
    for name in names:
        at = lowered.find(name.lower())
        while at != -1:
            spelled.append((at, name, at + len(name)))
            at = lowered.find(name.lower(), at + 1)
    # names arrive longest first, so a longer mention is already accepted by the time a
    # name nested inside it is tried, and `accepted` suppresses the nested one
    accepted = [(s, e) for s, _name, e in spelled]
    for name in names:
        stem = name.rsplit(".", 1)[0].lower()
        at = lowered.find(stem)
        while at != -1:
            end = at + len(stem)
            inside_accepted = any(s <= at < e for s, e in accepted)
            if not inside_accepted and any(
                end <= pos < end + CITATION_WINDOW and _same_clause(answer, end, pos)
                for pos in stamp_positions
            ):
                bare.append((at, name))
                accepted.append((at, end))
            at = lowered.find(stem, at + 1)
    return sorted([(at, name) for at, name, _end in spelled] + bare)


def check_citations(answer, hits, known_videos=None):
    """Exact, judge-free check of the video and timestamp the answer cites.

    What counts as a citation is decided once, by `cited_mentions`, and the
    citation-present, video-valid and time-valid metrics all read that one answer.

    A timestamp is attributed to the nearest cited video before it; failing that, to the
    nearest cited video after it in the same clause, so a model that names its source
    after the time is not penalised for word order. A timestamp with no cited video
    either side of it is uncheckable and counts invalid: the metric never credits a
    citation it cannot verify, and never charges one against a video the answer did not
    name.
    """
    retrieved = {}
    for _cid, payload in hits:
        retrieved.setdefault(payload["source"], []).append((payload["start"], payload["end"]))

    stamps = [(m.start(), int(m.group(1)) * 60 + int(m.group(2)))
              for m in TIMESTAMP_RE.finditer(answer)]
    mentions = cited_mentions(
        answer,
        known_videos if known_videos is not None else video_names(),
        [pos for pos, _seconds in stamps],
    )

    cited_videos = [name for _pos, name in mentions]
    videos_valid = [name in retrieved for name in cited_videos]

    times_valid = []
    for pos, seconds in stamps:
        earlier = [name for at, name in mentions if at < pos]
        later = [name for at, name in mentions if at > pos and _same_clause(answer, pos, at)]
        if earlier:
            spans = retrieved.get(earlier[-1], [])
        elif later:
            spans = retrieved.get(later[0], [])
        else:
            times_valid.append(False)
            continue
        # The cited time should fall inside a chunk that was actually retrieved. The
        # lower bound is floored to whole seconds because that is the only start time
        # `format_context` ever shows the generator; the upper bound is unchanged, so
        # this adds no tolerance beyond the display rounding.
        times_valid.append(any(int(start) <= seconds <= end for start, end in spans))

    return {
        "citation_present": 1.0 if cited_videos and stamps else 0.0,
        "citation_video_valid": (sum(videos_valid) / len(videos_valid)) if videos_valid else None,
        "citation_time_valid": (sum(times_valid) / len(times_valid)) if times_valid else None,
        "cited_videos": sorted(set(cited_videos)),
    }


def score(value):
    """Clamps a judge score to 0.00-1.00, or None when it cannot be read as a number.

    Out-of-range values are clamped and reported rather than rescaled: trying to guess
    whether a 1.4 meant 1.0 or 1.4% turned a near-perfect score into a near-zero one.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= number <= 1.0:
        print(f"  warning: judge returned {value!r}, outside 0.00-1.00; clamping")
    return round(min(1.0, max(0.0, number)), 2)


def judge(question, answer, context, prompt=JUDGE_PROMPT, complete_fn=complete):
    """Scores one answer with the judge model, which is not the generator.

    The prompt and the client are arguments so that the calibration stage can re-judge
    with the reference answer withheld, and so that both can be exercised against a stub
    judge without a provider call. The defaults are the scoring path `run` uses.
    """
    payload = extract_json(complete_fn(JUDGE_MODEL, prompt.format(
        question=question["question"],
        reference=question["answer"],
        context=context,
        answer=answer,
    )))
    if not isinstance(payload, dict):
        return {"faithfulness": None, "relevancy": None, "correctness": None,
                "refused": None, "note": "unparseable judge reply"}
    return {
        "faithfulness": score(payload.get("faithfulness")),
        "relevancy": score(payload.get("relevancy")),
        "correctness": score(payload.get("correctness")),
        "refused": payload.get("refused"),
        "note": payload.get("note"),
    }


def judge_refusal(question, answer):
    """Tier C only: did the pipeline decline, as it should have."""
    payload = extract_json(complete(JUDGE_MODEL, REFUSAL_PROMPT.format(
        question=question["question"], answer=answer,
    )))
    if not isinstance(payload, dict):
        return {"refused": None, "note": "unparseable judge reply"}
    return payload


def numeric(value):
    """Whether a value counts towards a metric: a real number, not an absent None.

    Absence is spelled None throughout — an unscored judge field, a citation validity
    metric on an answer that cited nothing — and every mean, count and difference in
    this module has to tell absence from zero, so the test is written down once.
    """
    return isinstance(value, (int, float))


def mean(values):
    values = [v for v in values if numeric(v)]
    return round(sum(values) / len(values), 3) if values else None


def refusal_verdicts(rows):
    """Each row's refusal verdict as 1.0 or 0.0, over the rows the judge left one on.

    A row with no verdict is excluded rather than read as "did not refuse". The two
    metrics built from this — tier C's refusal rate and tiers A and B's answerable
    refusal rate — are different claims about the same verdict, and each names itself
    where it is recorded.
    """
    return [1.0 if r.get("refused") else 0.0 for r in rows if r.get("refused") is not None]


def record_with_denominator(block, key, values):
    """Records a metric's value under `key` and the questions it covers under `key_n`.

    A question that carries no value for a metric is absent from that metric, not a zero
    in it. `check_citations` returns None for citation validity when the answer cited
    nothing, and averaging that None as 0.0 would collapse "did not cite" into "cited
    wrongly" — two different failures with two different remedies. So the mean drops
    those questions, and the count travels beside the value to say how many were left,
    because a mean whose denominator is invisible cannot be read honestly.
    """
    present = [v for v in values if numeric(v)]
    block[key] = mean(present)
    block[f"{key}_n"] = len(present)


def load_checkpoint():
    """Rows scored by earlier runs, keyed by question id."""
    if not RESULTS_PATH.exists():
        return {}
    rows = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def generate(question, context):
    """The deployed answer path, replayed through the shared rate limiting.

    rag.ask_llm() talks to groq directly with no pacing, so calling it here let an
    unhandled 429 kill a run 60 questions in. The messages are the deployed ones.
    """
    import video_transcript_rag as rag

    return complete(rag.llm, messages=rag.answer_messages(question, context))


def run():
    golden = load_golden()
    done = load_checkpoint()
    if done:
        print(f"resuming: {len(done)} of {len(golden)} already scored")

    GOLDEN_DIR.mkdir(exist_ok=True)
    results = []

    for i, q in enumerate(golden, 1):
        if q["id"] in done:
            results.append(done[q["id"]])
            continue

        hits = retrieve(q["question"])
        retrieved_ids = [cid for cid, _ in hits]
        context = format_context(hits[:GENERATION_K])

        try:
            answer = generate(q["question"], context)
        except DailyLimitReached as e:
            print(f"\nDaily token limit reached after {len(results)} of {len(golden)} "
                  f"questions. Progress is saved in {RESULTS_PATH}; rerun "
                  f"`uv run evaluate.py run` once the limit resets to continue, or "
                  f"`uv run evaluate.py report` to score what is done.\n  {e}")
            return

        row = {
            "id": q["id"],
            "tier": q["tier"],
            "course": q.get("course", ""),
            "hands_on": q.get("hands_on", False),
            "question": q["question"],
            "answer": answer,
            "retrieved": retrieved_ids,
            **check_citations(answer, hits[:GENERATION_K]),
        }

        # retrieval metrics only where gold chunks exist, which is tier B by construction
        gold = q.get("gold_chunks")
        if gold:
            row["gold_chunks"] = gold
            for k in K_VALUES:
                precision, recall = precision_recall(retrieved_ids, gold, k)
                row[f"precision@{k}"] = precision
                row[f"recall@{k}"] = recall

        try:
            if q["tier"] == "C":
                verdict = judge_refusal(q, answer)
                row["refused"] = verdict.get("refused")
                row["note"] = verdict.get("note")
            else:
                scores = judge(q, answer, context)
                row.update({
                    "faithfulness": scores.get("faithfulness"),
                    "relevancy": scores.get("relevancy"),
                    "correctness": scores.get("correctness"),
                    "refused": scores.get("refused"),
                    "note": scores.get("note"),
                })
        except DailyLimitReached as e:
            print(f"\nJudge hit its daily token limit after {len(results)} of "
                  f"{len(golden)} questions. Progress is saved in {RESULTS_PATH}.\n  {e}")
            return

        results.append(row)
        # written as we go, so an interrupted run loses at most one question
        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = (f"refused={row.get('refused')}" if q["tier"] == "C"
                   else f"corr={row.get('correctness')} faith={row.get('faithfulness')}")
        if gold:
            summary = f"r@5={row['recall@5']:.2f} " + summary
        print(f"[{i}/{len(golden)}] {q['id'][:45]} {summary}")

    write_report(results)


def rescore_rows(rows, chunks):
    """The judge-free metrics of a stored run, re-derived; judge fields passed through.

    Each checkpoint row carries the answer that was generated and the ordered ids it
    retrieved, and the chunk corpus supplies those ids' spans, so the citation check is
    handed exactly the input the original run handed it: the first GENERATION_K retrieved
    chunks, in retrieval order. Nothing is generated and nothing is judged again, so the
    rewritten report describes the same system behaviour that was first measured.
    """
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    rescored = []
    for row in rows:
        hits = [(cid, by_id[cid]) for cid in row["retrieved"][:GENERATION_K]]
        fresh = check_citations(row["answer"], hits)
        gold = row.get("gold_chunks")
        for k in K_VALUES if gold else ():
            precision, recall = precision_recall(row["retrieved"], gold, k)
            fresh[f"precision@{k}"] = precision
            fresh[f"recall@{k}"] = recall
        rescored.append({**row, **fresh})
    return rescored


def rescore():
    """Rewrites both report artifacts from the checkpoint, without calling a provider."""
    rows = list(load_checkpoint().values())
    if not rows:
        sys.exit(f"{RESULTS_PATH} is empty; run `run` first")
    chunks = load_chunks()
    if not chunks:
        sys.exit("no chunks were rebuilt; data/ and cache/ must both be present")
    print(f"re-deriving judge-free metrics for {len(rows)} checkpointed questions "
          f"from {len(chunks)} chunks")
    write_report(rescore_rows(rows, chunks))


def write_report(results):
    report = summarise(results)
    GOLDEN_DIR.mkdir(exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps({"summary": report, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nWrote {REPORT_JSON} and {REPORT_MD}\n")
    print(render_markdown(report))


def tier_c_block(rows):
    """Tier C's refusal rate, which is the one tier where declining is correct.

    It is deliberately not called an answerable refusal rate: tier C's ground truth is
    that no correct answer exists, so a refusal there is the pipeline working. The two
    are rendered side by side so that neither can be read as success on its own.
    """
    out = {"questions": len(rows)}
    record_with_denominator(out, "refusal_rate", refusal_verdicts(rows))
    return out


def summarise(results):
    def block(rows, retrieval=False, answerable_refusal=False):
        # An empty tier reports the one thing that is true of it — that it holds no
        # questions — rather than nothing at all, so every block in this summary has the
        # same shape and the renderer never has to guess whether `questions` is there.
        if not rows:
            return {"questions": 0}
        out = {"questions": len(rows)}
        if retrieval:
            for k in K_VALUES:
                record_with_denominator(out, f"precision@{k}",
                    [r.get(f"precision@{k}") for r in rows])
                # precision@k cannot exceed len(gold)/k, and most tier B questions have a
                # single gold chunk, so the raw number is meaningless without its ceiling
                record_with_denominator(out, f"precision@{k}_max",
                    [min(1.0, len(r.get("gold_chunks", [])) / k) for r in rows])
                record_with_denominator(out, f"recall@{k}",
                    [r.get(f"recall@{k}") for r in rows])
        for metric in ("faithfulness", "relevancy", "correctness",
                       "citation_present", "citation_video_valid", "citation_time_valid"):
            record_with_denominator(out, metric, [r.get(metric) for r in rows])
        # An answerable refusal is a declined answer to a question whose tier's ground
        # truth says the corpus can answer it, so it is a claim about a tier and is
        # derived only for the tier-level calls. A per-course or hands-on slice of it
        # would be a different and much smaller claim that nothing reports. It reuses
        # the refusal verdict the judge already recorded; no question is judged again to
        # obtain it.
        if answerable_refusal:
            record_with_denominator(out, "answerable_refusal_rate", refusal_verdicts(rows))
        return out

    tier_a = [r for r in results if r["tier"] == "A"]
    tier_b = [r for r in results if r["tier"] == "B"]
    tier_c = [r for r in results if r["tier"] == "C"]

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "generation_k": GENERATION_K,
        "judge_model": JUDGE_MODEL,
        "tier_a": block(tier_a, answerable_refusal=True),
        "tier_a_by_course": {
            course: block([r for r in tier_a if r["course"] == course])
            for course in sorted({r["course"] for r in tier_a})
        },
        "tier_a_hands_on": block([r for r in tier_a if r["hands_on"]]),
        "tier_b": block(tier_b, retrieval=True, answerable_refusal=True),
        "tier_c": tier_c_block(tier_c),
        "unscored": sum(1 for r in results
                        if r["tier"] != "C" and r.get("correctness") is None),
    }


def table(block, keys):
    """Each metric as a row of value and the questions that value was taken over.

    The count column is what stops a mean being read as covering the whole tier: a
    citation validity value drawn from 70 of 83 questions is a different claim from the
    same number drawn from all 83, and the difference is only visible if the denominator
    is printed beside the value.
    """
    lines = ["| metric | value | questions |", "| --- | --- | --- |"]
    for key in keys:
        if key in block:
            lines.append(f"| {key} | {block[key]} | {block.get(f'{key}_n', '')} |")
    return lines


GENERATION_METRICS = ("faithfulness", "relevancy", "correctness",
                      "citation_present", "citation_video_valid", "citation_time_valid")
RETRIEVAL_METRICS = tuple(
    name for k in K_VALUES
    for name in (f"precision@{k}", f"precision@{k}_max", f"recall@{k}")
)


def render_markdown(summary):
    lines = [
        "# Offline evaluation report",
        "",
        f"Generated {summary['generated']} · judge `{summary['judge_model']}` · "
        f"generation uses top {summary['generation_k']} chunks",
        "",
        "Faithfulness, relevancy and correctness are judge scores on a continuous "
        "0.00-1.00 scale; each question's per-score reasoning is kept in the `note` "
        "field of `report.json`.",
        "",
        "Tiers are scored on what their ground truth supports and are never averaged "
        "together.",
        "",
        "Every value carries the number of questions it was taken over. A question that "
        "carries no value for a metric is left out of that metric rather than counted as "
        "a zero, so a citation validity value covers only the answers that cited "
        "something: \"did not cite\" and \"cited wrongly\" stay distinguishable, and the "
        "count column is how a reader tells which is which.",
        "",
        "Provenance: the citation check was invalid in the run this report supersedes. "
        "It compared a cited timestamp, which the generator only ever saw floored to a "
        "whole second, against unfloored chunk spans, and it charged a timestamp against "
        "the nearest video name appearing before it even when the answer never cited "
        "that video. That citation defect was corrected 2026-10-19. Citation values here "
        "therefore measure a repaired ruler, not a changed pipeline, and are not "
        "comparable with any earlier copy of this report; the superseded values are "
        "discarded rather than kept beside them, so that a corrected measurement is "
        "never plotted as a trend.",
        "",
        "## Tier A — quiz questions with human answer keys",
        "",
        "Correctness is anchored to the course authors' own answer key. No chunk labels, "
        "so no retrieval metrics.",
        "",
        f"{summary['tier_a']['questions']} questions in the tier; each value below "
        f"covers the ones in its own `questions` column.",
        "",
    ]
    lines += table(summary["tier_a"], GENERATION_METRICS)

    lines += ["", "### By course", "",
              "| course | questions | correctness | faithfulness | citation_time_valid "
              "| cited a time |",
              "| --- | --- | --- | --- | --- | --- |"]
    for course, block in summary["tier_a_by_course"].items():
        lines.append(f"| {course} | {block.get('questions')} | {block.get('correctness')} "
                     f"| {block.get('faithfulness')} | {block.get('citation_time_valid')} "
                     f"| {block.get('citation_time_valid_n')} |")

    hands_on = summary["tier_a_hands_on"]
    if hands_on["questions"]:
        lines += ["", f"Hands-on questions ({hands_on['questions']} of tier A) ask about "
                      f"clicking through KNIME or Spark rather than lecture content: "
                      f"correctness {hands_on.get('correctness')} over "
                      f"{hands_on.get('correctness_n')} of them."]

    lines += ["", "## Tier B — generated from single chunks, exact gold labels", "",
              "Retrieval metrics are exact here. The questions are synthetic, so read "
              "them as a retrieval measurement, not as user behaviour.", "",
              f"{summary['tier_b']['questions']} questions in the tier; each value "
              f"below covers the ones in its own `questions` column.", "",
              "`precision@k_max` is the ceiling that k and the number of gold chunks "
              "allow: a question with one gold chunk can score at most 1/k, so compare "
              "each precision against its own maximum, not against 1.0.", ""]
    lines += table(summary["tier_b"], RETRIEVAL_METRICS)
    lines += ["", "Generation metrics on the same questions:", ""]
    lines += table(summary["tier_b"], GENERATION_METRICS)

    tier_a, tier_b, tier_c = (summary["tier_a"], summary["tier_b"], summary["tier_c"])
    lines += [
        "", "## Refusal — all three tiers together", "",
        "An answerable refusal is a declined answer to a question whose tier's ground "
        "truth says the corpus can answer it, so it is a failure of the same pipeline "
        "whose refusals on tier C are a success. Both rates are derived from the "
        "refusal verdict the judge already recorded for each answer; nothing was judged "
        "again to obtain them.",
        "",
        "Read the three rows as one measurement. A pipeline that declined every "
        "question would score 1.0 on tier C and look perfect there, so tier C's "
        "refusal rate is not a success on its own; nor is a low answerable refusal rate "
        "on a pipeline that never declines anything.",
        "",
        "| tier | ground truth | metric | value | questions |",
        "| --- | --- | --- | --- | --- |",
        f"| tier A | the corpus can answer | answerable refusal rate "
        f"| {tier_a.get('answerable_refusal_rate')} "
        f"| {tier_a.get('answerable_refusal_rate_n')} |",
        f"| tier B | the corpus can answer | answerable refusal rate "
        f"| {tier_b.get('answerable_refusal_rate')} "
        f"| {tier_b.get('answerable_refusal_rate_n')} |",
        f"| tier C | no correct answer exists | refusal rate "
        f"| {tier_c.get('refusal_rate')} | {tier_c.get('refusal_rate_n')} |",
        "",
        f"Tier C's {tier_c['questions']} questions are on topics verified absent from "
        f"the corpus; the pipeline should decline all of them. Every answerable refusal "
        f"counted above scored correctness 0.0, so without this rate the failure is "
        f"hidden inside a low correctness value instead of named.",
    ]

    if summary["unscored"]:
        lines += ["", f"{summary['unscored']} answers could not be scored by the judge."]
    return "\n".join(lines) + "\n"


def spotcheck():
    """Writes the generated tier B questions out for a human to confirm or reject."""
    golden = [q for q in load_golden() if q["tier"] == "B"]
    if not golden:
        sys.exit("no tier B questions found; run build_golden.py tierb then build")

    by_id = {c["chunk_id"]: c for c in load_chunks()}
    random.seed(SPOTCHECK_SEED)
    sample = random.sample(golden, min(SPOTCHECK_SIZE, len(golden)))

    lines = [
        "# Tier B question review",
        "",
        "These questions were generated from the passage shown beneath each one.",
        "Untick a question that is NOT answerable from its passage, or that merely",
        "copies the passage's wording, then run `uv run evaluate.py review`.",
        "",
    ]
    for q in sample:
        chunk = by_id.get(q["source_chunk"])
        lines += [
            f"- [x] `{q['id']}` (jaccard {q['jaccard']}, bm25 rank {q['bm25_rank']})",
            f"    - **Q:** {q['question']}",
            f"    - **A:** {q['answer']}",
            f"    - **Source passage ({q['source_chunk']}):** "
            f"{chunk['text'] if chunk else '(chunk missing)'}",
            "",
        ]

    GOLDEN_DIR.mkdir(exist_ok=True)
    SPOTCHECK_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SPOTCHECK_MD}: {len(sample)} generated questions to review")


CHECKBOX_RE = re.compile(r"^- \[([ xX])\] `([^`]+)`")


def review():
    """Reports how many reviewed tier B questions a human rejected."""
    if not SPOTCHECK_MD.exists():
        sys.exit(f"{SPOTCHECK_MD} not found; run spotcheck first")

    verdicts = {m.group(2): m.group(1).lower() == "x"
                for m in (CHECKBOX_RE.match(line)
                          for line in SPOTCHECK_MD.read_text(encoding="utf-8").splitlines())
                if m}
    if not verdicts:
        sys.exit("no checkboxes found in the spot-check sheet")

    rejected = [qid for qid, kept in verdicts.items() if not kept]
    print(f"reviewed: {len(verdicts)}")
    print(f"accepted: {len(verdicts)-len(rejected)}")
    print(f"rejected: {len(rejected)}")
    for qid in rejected:
        print(f"  {qid}")
    if rejected:
        share = len(rejected) / len(verdicts)
        print(f"\n{share:.0%} of the sample was rejected. Tier B metrics carry that error "
              f"rate; regenerate those items or tighten MAX_JACCARD if it is high.")


# ------------------------------------------------- judge calibration (issue #8)

def no_reference_prompt():
    """JUDGE_PROMPT with the reference answer, and the metric that needs it, removed.

    Derived from the real judge prompt rather than copied, so the two calibration passes
    cannot drift apart: an edit to JUDGE_PROMPT lands in both, and the only difference
    between the passes stays the reference answer. Each edit is guarded, so a reworded
    prompt fails here instead of quietly sending the reference through anyway.
    """
    prompt = JUDGE_PROMPT
    for old, new in (("REFERENCE ANSWER (ground truth): {reference}\n\n", ""),
                     ("Score these three, strictly:", "Score these two, strictly:"),
                     ('"correctness": 0.00, ', "")):
        if old not in prompt:
            raise RuntimeError(
                f"JUDGE_PROMPT no longer contains {old!r}; the reference-withheld "
                "variant used by `calibrate` has to be rederived")
        prompt = prompt.replace(old, new)
    prompt, removed = re.subn(r"\n- correctness:.*?(?=\n\n)", "", prompt, flags=re.S)
    if removed != 1:
        raise RuntimeError("could not remove the correctness criterion from JUDGE_PROMPT")
    return prompt


def calibration_sample(rows):
    """The seeded, stratified sample of already-scored questions, reproducibly.

    Candidates are sorted by id first, so the sample depends on the seed and the set of
    scored questions and not on the order the checkpoint happens to be in. A row with no
    judge score is not a candidate: there is no first run to compare a re-run against.
    """
    sample = []
    for tier, size in CALIBRATION_STRATA:
        candidates = sorted(
            (r for r in rows
             if r.get("tier") == tier
             and any(numeric(r.get(m)) for m in JUDGE_METRICS)),
            key=lambda r: r["id"],
        )
        sample.extend(random.Random(CALIBRATION_SEED).sample(
            candidates, min(size, len(candidates))))
    return sample


def _gap(left, right, metric):
    """Signed differences on one metric, over the ids both sides scored numerically."""
    gaps = {}
    for qid, scores in left.items():
        theirs, ours = right.get(qid, {}).get(metric), scores.get(metric)
        if numeric(theirs) and numeric(ours):
            gaps[qid] = ours - theirs
    return gaps


def _spread(gaps):
    """How far apart two judged passes landed on one metric, and in which direction.

    The mean absolute difference is the number the pre-committed thresholds are read
    against. The signed shift says whether the second pass scored systematically higher
    or lower rather than merely differently, and the maximum says whether the mean is
    hiding one large disagreement. `n` travels with them because a spread over 2 ids and
    a spread over 20 are different claims.
    """
    values = list(gaps.values())
    return {
        "n": len(values),
        "mean_absolute_difference": mean([abs(v) for v in values]),
        "mean_signed_shift": mean(values),
        "max_absolute_difference": (round(max(abs(v) for v in values), 3)
                                     if values else None),
    }


def calibration_summary(baseline_rows, results):
    """Turns the two judged passes into the numbers and the pre-committed outcomes.

    `baseline_rows` are the checkpoint rows carrying the original run's judge scores;
    `results` are the rows this stage judged, each tagged with its pass. The thresholds
    come from CALIBRATION_RULES, which were fixed before any of this was run, and are
    written out alongside the verdict so the outcome cannot later be renegotiated.
    """
    baseline = {r["id"]: r for r in baseline_rows}
    rerun = {r["id"]: r for r in results if r.get("pass") == "rerun"}
    no_reference = {r["id"]: r for r in results if r.get("pass") == "no_reference"}

    between_run = {m: _spread(_gap(rerun, baseline, m)) for m in JUDGE_METRICS}
    # the reference-withheld pass is compared against the fresh unchanged-prompt pass,
    # so the only thing that differs between the two is the reference answer
    withheld = {m: _spread(_gap(no_reference, rerun, m))
                for m in ("faithfulness", "relevancy")}

    measured = [(m, between_run[m]["mean_absolute_difference"]) for m in JUDGE_METRICS
                if between_run[m]["mean_absolute_difference"] is not None]
    worst_metric, worst = max(measured, key=lambda pair: pair[1], default=(None, None))
    faith_shift = withheld["faithfulness"]["mean_absolute_difference"]

    # Each rule is matched to its measurement by key, never by position. A third rule
    # added to CALIBRATION_RULES must not be able to inherit another's number silently:
    # the whole credibility of this stage rests on these thresholds being the ones that
    # were pre-committed, and on each verdict being read against its own measurement.
    measurements = {
        "between_run_spread": worst,
        "faithfulness_without_reference": faith_shift,
    }
    outcomes = []
    for rule in CALIBRATION_RULES:
        value = measurements[rule["key"]]
        triggered = value is not None and value > rule["threshold"]
        outcome = {
            **rule,
            "value": value,
            "triggered": triggered,
            "outcome": rule["if_exceeded"] if triggered else rule["if_not_exceeded"],
        }
        if rule["key"] == "between_run_spread":
            outcome["worst_metric"] = worst_metric
        outcomes.append(outcome)

    return {
        "stage": "calibrate",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "judge_model": JUDGE_MODEL,
        "seed": CALIBRATION_SEED,
        "strata": {tier: size for tier, size in CALIBRATION_STRATA},
        "sample": sorted(rerun),
        "passes": {
            "rerun": "the judge prompt unchanged, to measure between-run spread "
                     "against the score already in golden/results.jsonl",
            "no_reference": "the same prompt with the reference answer withheld, to "
                            "measure how far faithfulness depends on seeing it",
        },
        "pre_committed": True,
        "pre_committed_note": "Both thresholds and the outcome each one triggers were "
                              "fixed in issues #1 and #8 before this stage was ever "
                              "run, so a result cannot be re-read as small-sample "
                              "noise after the fact.",
        "between_run": between_run,
        "no_reference": withheld,
        "outcomes": outcomes,
    }


def calibration_progress():
    """Judged passes from earlier attempts, keyed by (question id, pass)."""
    if not CALIBRATION_RAW_PATH.exists():
        return {}
    done = {}
    for line in CALIBRATION_RAW_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[(row["id"], row["pass"])] = row
    return done


def calibrate(complete_fn=complete):
    """Re-judges a seeded sample twice, to put error bars on the judge-derived scores.

    Pass one repeats the judge prompt unchanged, so the gap against the score already in
    the checkpoint is between-run spread on an unchanged input. Pass two withholds the
    reference answer, so the gap against pass one is how far faithfulness leans on
    seeing it. Nothing is generated and no existing score is overwritten: the stage
    reuses the stored answers and writes only its own artifact.

    `complete_fn` is the provider call, so the whole stage is exercisable against a stub
    judge. The default is the shared paced client, and the real run needs no change here.
    Each judged pass is appended as it finishes, so a daily limit resumes for free.
    """
    checkpoint = list(load_checkpoint().values())
    if not checkpoint:
        sys.exit(f"{RESULTS_PATH} is empty; run `run` first")
    sample = calibration_sample(checkpoint)
    print(f"calibrating on {len(sample)} questions "
          f"(seed {CALIBRATION_SEED}, {dict(CALIBRATION_STRATA)})")

    golden = {q["id"]: q for q in load_golden()}
    missing = [r["id"] for r in sample if r["id"] not in golden]
    if missing:
        sys.exit(f"{len(missing)} sampled questions are not in {GOLDEN_PATH}, so their "
                 f"reference answer is unavailable: {missing[0]}")
    chunks = {c["chunk_id"]: c for c in load_chunks()}

    done = calibration_progress()
    passes = (("rerun", JUDGE_PROMPT), ("no_reference", no_reference_prompt()))
    GOLDEN_DIR.mkdir(exist_ok=True)

    for i, row in enumerate(sample, 1):
        # the context the judge saw originally, rebuilt from the checkpoint's ordered
        # chunk ids and the local cache, so neither retrieval nor generation runs again
        hits = [(cid, chunks[cid]) for cid in row["retrieved"][:GENERATION_K]
                if cid in chunks]
        context = format_context(hits)
        for which, prompt in passes:
            if (row["id"], which) in done:
                continue
            try:
                scores = judge(golden[row["id"]], row["answer"], context,
                               prompt=prompt, complete_fn=complete_fn)
            except DailyLimitReached as e:
                print(f"\nJudge hit its daily token limit after {len(done)} of "
                      f"{len(sample) * len(passes)} passes. Progress is saved in "
                      f"{CALIBRATION_RAW_PATH}; rerun `uv run evaluate.py calibrate` "
                      f"once the limit resets to continue.\n  {e}")
                return
            judged = {"id": row["id"], "tier": row["tier"], "pass": which, **scores}
            done[(row["id"], which)] = judged
            with CALIBRATION_RAW_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(sample)}] {row['id'][:40]} {which}: "
                  f"faith={judged.get('faithfulness')}")

    summary = calibration_summary(checkpoint, list(done.values()))
    CALIBRATION_PATH.write_text(
        json.dumps({"summary": summary, "judged": list(done.values())},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nWrote {CALIBRATION_PATH}")
    print(report_calibration(summary))


def report_calibration(summary):
    """The outcome in plain words, stated against the thresholds whichever way it fell."""
    lines = ["Between-run difference, judge prompt unchanged:"]
    for metric in JUDGE_METRICS:
        spread = summary["between_run"][metric]
        lines.append(f"  {metric:<13} mean abs diff {spread['mean_absolute_difference']} "
                     f"(max {spread['max_absolute_difference']}, n={spread['n']})")
    withheld = summary["no_reference"]["faithfulness"]
    lines += [
        "",
        f"Faithfulness with the reference answer withheld: mean abs diff "
        f"{withheld['mean_absolute_difference']}, mean shift "
        f"{withheld['mean_signed_shift']} (n={withheld['n']})",
        "",
        "Pre-committed decision rules (fixed in #1/#8 before this was run):",
    ]
    for rule in summary["outcomes"]:
        verdict = "EXCEEDED" if rule["triggered"] else "not exceeded"
        lines += [
            f"  {rule['name']}: {rule['measure']} = {rule['value']}, threshold "
            f"{rule['threshold']} -> {verdict}",
            f"    outcome: {rule['outcome']}",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "run":
        run()
    elif stage == "spotcheck":
        spotcheck()
    elif stage == "review":
        review()
    elif stage == "calibrate":
        calibrate()
    elif stage == "report":
        rows = list(load_checkpoint().values())
        if not rows:
            sys.exit(f"{RESULTS_PATH} is empty; run `run` first")
        print(f"scoring {len(rows)} checkpointed questions")
        write_report(rows)
    elif stage == "rescore":
        rescore()
    else:
        sys.exit("usage: evaluate.py [run|report|rescore|spotcheck|review|calibrate]")
