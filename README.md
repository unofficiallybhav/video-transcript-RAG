# video-transcript-rag

Retrieval-augmented question answering over 83 course video transcripts, and an offline
evaluation layer that decides whether a change to the pipeline is an improvement.

The pipeline is the ordinary part: Whisper transcripts, 406 chunks, dense retrieval out
of qdrant, an answer generated from the top 5 chunks with a cited video and timestamp.
**The evaluation layer is the part worth reading.** Most of this repo's design effort went
into making its own numbers hard to fool itself with.

## Start here

| | |
|---|---|
| [`golden/report.md`](golden/report.md) | the current measurement, every value carrying the number of questions it was taken over |
| [`docs/adr/0001-three-tier-golden-dataset.md`](docs/adr/0001-three-tier-golden-dataset.md) | why there are three tiers, why their metrics are never averaged, and why no model takes part in a labelling decision |
| [`docs/adr/0002-faithfulness-judged-without-the-reference-answer.md`](docs/adr/0002-faithfulness-judged-without-the-reference-answer.md) | a pre-registered experiment that forced a change to how faithfulness is scored |
| [`golden/judge_calibration.json`](golden/judge_calibration.json) | that experiment's raw result, thresholds and all |
| [`GLOSSARY.md`](GLOSSARY.md) | the project's own vocabulary, including which words it refuses to use for which metric |

## The idea the project is built around

A judge model scoring its own pipeline's answers will flatter them, and an evaluation
layer that cannot tell a repaired ruler from a changed pipeline will report noise as
progress. Three things follow, and they are most of the design:

**Ground truth is partitioned by what kind of ground truth it is.** Tier A is 83 course
quiz questions with answer keys the course authors wrote. Tier B is 40 questions each
generated from one known chunk, so retrieval has an exact target. Tier C is 12 questions
on topics verified absent from the corpus, where declining to answer is the only correct
behaviour. Each tier is scored only on what its own ground truth supports, and the tiers
are never averaged — a blended number would cancel opposing failures, since refusing is
correct on Tier C and a defect on the other two.

**No model participates in a labelling decision.** Not in deciding a question's gold
chunk, not in deciding whether a question is answerable. Where a label needs a judgement,
a human makes it: `golden/spotcheck.md` is a human's verdicts on a sample of the generated
Tier B questions, and 13% of that sample was rejected, so Tier B's retrieval figures are
read as carrying that error rate.

**A measurement change is never plotted as a trend.** Twice now a metric has been found
invalid and repaired — the citation check, then faithfulness. Both times the superseded
values were discarded rather than kept beside the new ones, and the report says which
metric was repaired, when, and why its old values are not comparable. `GLOSSARY.md`
separates a metric's *validity* from its *value* precisely so those two can be discussed
without being confused.

### The faithfulness experiment

The clearest example of the above, and the thing to read if you only read one: faithfulness
claims to measure how much of an answer the retrieved context supports. It was being scored
in the same judge call as correctness, which meant the judge could see the answer key.

Two thresholds and the consequence each would trigger were fixed **before** the experiment
ran, so the result could not afterwards be renegotiated as small-sample noise. Thirty
questions were re-judged with the reference answer withheld. Faithfulness moved 0.164
against a threshold of 0.10 — a signed shift of +0.109, meaning the judge had been
*penalising* answers that diverged from the answer key, so faithfulness was partly grading
correctness, the one thing it names itself as not measuring. The pre-committed consequence
was a permanent split into two judge calls, and it was carried out.

Re-judging all 123 scored answers then moved the tier means almost not at all (Tier A
0.793 → 0.787) while moving individual answers by 0.093 on average. That is recorded as
it happened, including why it does not weaken the decision: a score taken with the answer
key in view was not a claim about contextual support whatever number it landed on. The
repair was to the metric's validity, not its value.

## Current measurement

Judge is `openai/gpt-oss-20b`, which is not the generator. Generation sees the top 5 chunks.

**Tier A** — 83 quiz questions, human answer keys

| correctness | faithfulness | relevancy | cited a video+time | timestamp valid | answerable refusal |
|---|---|---|---|---|---|
| 0.761 | 0.787 | 0.792 | 0.831 | 0.936 (of 70) | **0.157** |

**Tier B** — 40 questions, exact chunk labels

| recall@1 | recall@5 | recall@10 | precision@5 (max 0.225) | correctness | faithfulness |
|---|---|---|---|---|---|
| 0.588 | 0.912 | 0.925 | 0.200 | 0.889 | 0.962 |

**Tier C** — 12 unanswerable questions: refusal rate **1.0**. Read beside Tier A's
answerable refusal rate, never on its own — a pipeline that declined everything would
score 1.0 here and look perfect.

## What is weak, stated plainly

- **15.7% of answerable Tier A questions were refused.** Every one scored correctness 0.0,
  so without naming this rate the failure hides inside a low correctness score.
- **Hands-on questions are the worst area**: the 10 Tier A questions about clicking through
  KNIME or Spark score 0.40 correctness, and 6 of 10 were refused. 4 of the 15 reviewed
  Tier B questions are hands-on too, which bears on how Tier B's stronger figures read.
- **Machine learning trails graph analytics** on correctness, 0.733 against 0.81.
- **3 Tier B gold chunks are never retrieved at any k**, so recall@10 of 0.925 has a
  ceiling below 1.0 that is a property of the index, not the k.
- **Tier B's questions are synthetic.** Its retrieval numbers are exact and its questions
  are not user behaviour. 13% of a reviewed sample was rejected outright.

## Running it

The offline stages need no network, no API credentials and no GPU. A clone can reproduce
the whole report from the stored run:

```sh
uv run pytest                  # 73 tests, no network, no credentials
uv run evaluate.py rescore     # re-derive the judge-free metrics and rewrite the report
uv run evaluate.py report      # rebuild the report from the checkpoint as-is
```

Stages that cost provider tokens, and are not needed to read the results:

```sh
uv run evaluate.py run         # score the pipeline end to end, resumable
uv run evaluate.py rejudge     # re-measure faithfulness with the reference withheld
uv run evaluate.py calibrate   # the judge calibration experiment (~110k tokens)
```

Dependencies are managed with `uv`, never pip. The video files themselves are gone and are
not required: `corpus.json` lists the 83 ingested videos and `cache/` holds one transcript
each, which together reproduce the 406 chunks and the chunk ids already in qdrant.

## Layout

```
evaluate.py                 the evaluation layer: retrieval, judging, reporting, calibration
build_golden.py             builds the three tiers of the golden dataset
video_transcript_rag.py     ingestion, chunking, embedding, the deployed answer path
llm.py                      one rate-limit-paced provider call, shared by both
golden/                     the dataset, the stored run, the reports, the calibration record
cache/                      83 Whisper transcripts
docs/adr/                   decisions whose reasoning is not recoverable from the code
```
