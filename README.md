# video-transcript-rag

Ask questions about a library of course lecture videos and get an answer that cites the
video and the timestamp it came from.

The corpus is 83 lecture videos from two Big Data courses — 43 on machine learning, 40 on
graph analytics. They are transcribed locally with Whisper, split into 406 timestamped
chunks, embedded and indexed in qdrant, and answered out of the top 5 retrieved chunks by
a hosted model that is instructed to cite its source and to decline anything the corpus
does not cover.

Alongside it sits an offline evaluation layer that decides whether a change to the
pipeline is actually an improvement. Most of this repo's design effort went there, and
it is the part worth reading.

## The pipeline, end to end

All of it lives in [`video_transcript_rag.py`](video_transcript_rag.py).

### 1. Ingestion

`ingest_all(root)` walks a media directory and picks up anything with a recognised
extension — `.mp4`, `.mkv`, `.webm`, `.m4a`, `.mp3`, `.wav`, `.flac`. The folder a video
sits in names its course, which becomes a payload field and later a retrieval filter.
Each file then runs the full transcript → chunks → vectors path, and a failure on one
file is reported and skipped rather than killing the batch.

### 2. Transcription

Whisper runs **locally**, not through an API: `faster-whisper` with `large-v3-turbo`, on
the GPU in `float16` where CUDA is available and falling back to CPU `int8` where it is
not. On Windows the CUDA DLLs that ship inside the `nvidia-*` wheels have to be on the
DLL path before `ctranslate2` is imported, which is what `enable_cuda_libs()` does.

Whisper decodes the video directly and windows long audio itself, so there is no upload
limit to split around. Transcription is forced to English with `beam_size=5` and
`condition_on_previous_text=False`, the last of which stops one bad segment's wording
from steering everything after it.

Every transcript is cached as JSON in `cache/`, one file per video, holding each segment's
`start`, `end` and `text`. Re-ingesting a video reuses the cache instead of re-transcribing.

### 3. Chunking

Segments are accumulated until a chunk reaches **1200 characters**, then flushed, with the
last **2 segments** carried into the next chunk as overlap. The tail is skipped when it
contains nothing but that overlap.

Chunks are split on segment boundaries rather than mid-sentence, and each one keeps the
`start` of its first segment and the `end` of its last. Those timestamps are what makes a
citation checkable, so chunking is not a purely lexical step here — it is what the
accuracy of every later citation claim rests on.

### 4. Embedding

`sentence-transformers` with `all-MiniLM-L6-v2`, 384 dimensions, encoded in a batch per
video. Local, no API call.

### 5. Indexing

One qdrant collection, `video transcript`, cosine distance, with keyword payload indexes
on `source` and `course` so retrieval can be filtered to one video or one course.

Point ids are **deterministic**: `uuid5(NAMESPACE_URL, "<source>:<chunk_index>")`. That
makes re-ingesting a video idempotent, and it is why a chunk id like
`building a graph.mp4:3` means the same thing in the index, in the golden dataset and in
the stored evaluation run. Before upserting, every existing point for that `source` is
deleted, so a re-ingest that produces fewer chunks does not leave stale ones behind.

### 6. Retrieval

Dense retrieval only. The query is encoded with the same embedding model and
`query_points` returns the top 5 by cosine similarity, optionally filtered by `source` or
`course`.

Each hit is rendered into the context as:

```
[<video name> @ MM:SS] <chunk text>
```

The video name travels with the timestamp because a bare timestamp means nothing across a
library this size. Note that `MM:SS` **floors the chunk's start to a whole second** — the
model can only ever cite the rounded value it was shown, and a citation checker that
compared against the unfloored start would mark correct behaviour as wrong. It did, for a
while; see the evaluation section.

### 7. Generation

Groq, `openai/gpt-oss-120b`, `temperature=0.0`. The system prompt carries the retrieved
context and asks for a short summary of what was discussed plus the video name and
timestamp, and instructs the model to decline rather than answer from outside the context.

`answer_messages()` builds the exact message list the deployed path sends, and the
evaluator replays *that function*, so the measured prompt and the shipped prompt cannot
drift apart.

### Rate limiting

Every hosted call goes through [`llm.py`](llm.py), which paces requests against the free
tier's 8k tokens-per-minute budget using the response's rate-limit headers rather than
retrying blindly after a failure. The per-day budget is not reported in headers at all, so
a daily exhaustion raises `DailyLimitReached` instead of sleeping for hours.

## Using it

```sh
uv run video_transcript_rag.py            # interactive chat
uv run video_transcript_rag.py ingest     # batch-ingest data/, then exit
```

In the chat REPL:

| command | effect |
|---|---|
| `ingest all` / `ingest <path>` | ingest a directory or a single file |
| `source <file>` | restrict retrieval to one video (`source` alone clears it) |
| `course <name>` | restrict retrieval to one course (`course` alone clears it) |
| `exit` | quit |

Credentials live in `.env`: `QDRANT_URL`, `QDRANT_API_KEY`, and numbered Groq keys
`GROQ_API_KEY_1` / `GROQ_API_KEY_2`, where `GROQ_KEY` selects which one to spend and
defaults to `2`. Each key carries its own daily token budget and one full evaluation pass
costs most of one, which is why they are numbered.

## The evaluation layer

A judge model scoring its own pipeline's answers will flatter them, and an evaluation
layer that cannot tell a repaired ruler from a changed pipeline will report noise as
progress. Three ideas follow, and they are most of the design.

**Ground truth is partitioned by what kind of ground truth it is.** Tier A is 66 course
quiz questions with answer keys the course authors wrote. Tier C is 12 questions on
topics verified absent from the corpus, where declining to answer is the only correct
behaviour. Each tier is scored only on what its own ground truth supports, and the tiers
are never averaged — a blended number would cancel opposing failures, since refusing is
correct on Tier C and a defect on Tier A.

**No model participates in a labelling decision.** Not in deciding a question's gold
chunk, not in deciding whether a question is answerable. A third tier used to generate
its own questions from the chunk that then served as their gold label, which made the
label exact and the question unverified. A human reviewed 15 of its 40 questions and
rejected 2, and the tier was retired rather than tightened — no threshold on lexical
overlap separated the rejected questions from the sound ones. Its verdicts survive in
`golden/spotcheck.md`, and
[`docs/adr/0003-retiring-the-generated-retrieval-tier.md`](docs/adr/0003-retiring-the-generated-retrieval-tier.md)
records the decision.

Gold chunks are therefore labelled by hand. `uv run build_golden.py label` writes
`golden/labelsheet.md`: every Tier A question beside a shortlist of candidate chunks, each
shown with its transcript text, its bm25 and embedding ranks, and its lexical overlap with
the answer key. Those signals choose what a human reads; they never decide what is gold.
Marking is incremental and the report is correct part-way through — an unmarked question
is counted as unlabelled and left out of every retrieval figure rather than scored zero in
one.

**A measurement change is never plotted as a trend.** Twice now a metric has been found
invalid and repaired — the citation check, then faithfulness. Both times the superseded
values were discarded rather than kept beside the new ones, and the report says which
metric was repaired, when, and why its old values are not comparable. `GLOSSARY.md`
separates a metric's *validity* from its *value* precisely so those two can be discussed
without being confused.

### The faithfulness experiment

The clearest example, and the thing to read if you only read one. Faithfulness claims to
measure how much of an answer the retrieved context supports. It was being scored in the
same judge call as correctness, which meant the judge could see the answer key.

Two thresholds and the consequence each would trigger were fixed **before** the experiment
ran, so the result could not afterwards be renegotiated as small-sample noise. Thirty
questions were re-judged with the reference answer withheld. Faithfulness moved 0.164
against a threshold of 0.10 — a signed shift of +0.109, meaning the judge had been
*penalising* answers that diverged from the answer key, so faithfulness was partly grading
correctness, the one thing it names itself as not measuring. The pre-committed consequence
was a permanent split into two judge calls, and it was carried out.

Re-judging all 123 scored answers then moved the tier means almost not at all (Tier A
0.793 → 0.787) while moving individual answers by 0.093 on average. That is recorded as it
happened, including why it does not weaken the decision: a score taken with the answer key
in view was not a claim about contextual support whatever number it landed on. The repair
was to the metric's validity, not its value.

### Where to read further

| | |
|---|---|
| [`golden/report.md`](golden/report.md) | the current measurement, every value carrying the number of questions it was taken over |
| [`docs/adr/0001-three-tier-golden-dataset.md`](docs/adr/0001-three-tier-golden-dataset.md) | why ground truth is partitioned at all, why tier metrics are never averaged, and why no model takes part in a labelling decision |
| [`docs/adr/0003-retiring-the-generated-retrieval-tier.md`](docs/adr/0003-retiring-the-generated-retrieval-tier.md) | why the tier that measured retrieval was retired, and why no threshold would have saved it |
| [`docs/adr/0004-questions-the-corpus-cannot-answer-leave-tier-a.md`](docs/adr/0004-questions-the-corpus-cannot-answer-leave-tier-a.md) | why 17 questions left Tier A, what every figure it moved, and why the hands-on finding was dissolved rather than answered |
| [`docs/adr/0002-faithfulness-judged-without-the-reference-answer.md`](docs/adr/0002-faithfulness-judged-without-the-reference-answer.md) | the pre-registered experiment above |
| [`golden/judge_calibration.json`](golden/judge_calibration.json) | that experiment's raw result, thresholds and all |
| [`GLOSSARY.md`](GLOSSARY.md) | the project's own vocabulary, including which words it refuses to use for which metric |

## Current measurement

Judge is `openai/gpt-oss-20b`, which is deliberately not the generator. Generation sees the
top 5 chunks.

**Tier A** — 66 quiz questions, human answer keys

| correctness | faithfulness | relevancy | cited a video+time | timestamp valid | answerable refusal |
|---|---|---|---|---|---|
| 0.889 | 0.861 | 0.912 | 0.924 | 0.935 (of 62) | **0.061** |

> These are not comparable with any earlier copy of this table. The tier held 83
> questions until 17 were removed on 2026-10-23, because their answers were figures
> produced by running a KNIME or Spark exercise — a percentage of a dataset, a cluster
> number, a column name — which no lecture transcript states, so Tier A's ground truth
> that the corpus can answer the question never held for them. Those 17 had scored
> correctness 0.265, so every figure above rises partly for that reason. What changed is
> which questions are measured, not the pipeline. See
> [`docs/adr/0004-questions-the-corpus-cannot-answer-leave-tier-a.md`](docs/adr/0004-questions-the-corpus-cannot-answer-leave-tier-a.md).

**Tier C** — 12 unanswerable questions: refusal rate **1.0**. Read beside Tier A's
answerable refusal rate, never on its own — a pipeline that declined everything would
score 1.0 here and look perfect.

**Retrieval** — measured for the first time, over the **33 of 66** Tier A questions whose
gold chunks are labelled by hand so far.

| k | recall@k | precision@k | ceiling (`precision@k_max`) |
|---|---|---|---|
| 1 | 0.265 | 0.545 | 1.0 |
| 3 | 0.629 | 0.465 | 0.727 |
| 5 | **0.778** | 0.345 | 0.461 |
| 10 | 0.917 | 0.206 | 0.230 |

Generation is scored on the top 5, so `recall@5` 0.778 is the figure that bears on the
answers: about a fifth of the material the labeller judged necessary never reaches the
generator. Read each precision against the ceiling beside it, not against 1.0 — the
labelled questions average 2.3 gold chunks, so `precision@5` cannot exceed 0.461. At that
reading retrieval is using 75% of the precision available to it at k=5, and ranks a gold
chunk first 54% of the time.

The remaining 33 questions are unlabelled and absent from every figure above rather than
scored zero in it; the report prints the count beside each one.

## What is weak, stated plainly

- **A fifth of the material an answer needed never reaches the generator.** `recall@5` is
  0.778 over the 33 labelled questions, and generation runs on the top 5. This is the
  first measured retrieval figure this project has had, and it is the clearest target
  left: `recall@10` is 0.917, so most of the missing material is ranked 6th to 10th
  rather than absent from the index.
- **The top-ranked chunk is a gold chunk 54% of the time** (`precision@1` 0.545 against a
  ceiling of 1.0), which is the figure a reranker would move.
- **6.1% of answerable Tier A questions are still refused** — 4 of 66. Every one scored
  correctness 0.0, so without naming this rate the failure hides inside a low correctness
  score. It was 15.7% of 83 before the 17 removals, and 9 of those 17 were refusals the
  pipeline was arguably right to make.
- **Graph analytics now trails machine learning** on correctness, 0.81 against 0.96. The
  direction has reversed: machine learning was the weaker course at 0.733, and its weak
  questions were disproportionately the exercise questions now removed. Graph analytics is
  also the half that is **entirely unlabelled**, so none of its retrieval is measured.
- **The hands-on finding was dissolved, not answered.** The earlier 0.40 correctness over
  10 hands-on questions is gone: 9 of those 10 were asking for exercise output the corpus
  does not contain. One hands-on question remains, which supports no claim. Whether the
  pipeline is genuinely weak at hands-on material is now untested, and testing it needs
  hands-on questions whose answers the lectures actually state.
- **Retrieval is dense-only.** There is no hybrid or lexical fallback and no reranking.

## Running the evaluation

The offline stages need no network, no API credentials and no GPU. A clone can reproduce
the whole report from the stored run:

```sh
uv run pytest                  # 112 tests, no network, no credentials
uv run evaluate.py rescore     # re-derive the judge-free metrics and rewrite the report
```

> **Use `rescore`, not `report`.** `uv run evaluate.py report` rebuilds the report from the
> values stored in the checkpoint rather than re-deriving them, which silently reverts
> Tier A `citation_time_valid` from 0.936 to the superseded 0.488. After any change that
> regenerates the report, run `rescore`.

Labelling gold chunks, which also needs no provider tokens — `label` needs the qdrant
index for the shortlist's dense half, `gold` and `rescore` need nothing but the repo:

```sh
uv run build_golden.py label   # write golden/labelsheet.md, keeping any marks on it
                               # ... then tick the gold chunks in that file by hand
uv run build_golden.py gold    # read the marks back into the golden set
uv run evaluate.py rescore     # report retrieval over the questions labelled so far
```

Stages that cost provider tokens, and are not needed to read the results:

```sh
uv run evaluate.py run         # score the pipeline end to end, resumable
uv run evaluate.py rejudge     # re-measure faithfulness with the reference withheld
uv run evaluate.py calibrate   # the judge calibration experiment (~73k tokens)
```

Set `PYTHONIOENCODING=utf-8` whenever an answer or report is printed: stored answers
contain characters cp1252 cannot encode.

Dependencies are managed with `uv`, never pip. The video files themselves are gone and are
not required — `corpus.json` lists the 83 ingested videos and `cache/` holds one transcript
each, which together reproduce the 406 chunks and the chunk ids already in qdrant.

## Layout

```
video_transcript_rag.py     the pipeline: ingestion, transcription, chunking,
                            embedding, indexing, retrieval, generation, chat
llm.py                      one rate-limit-paced provider call, shared by both layers
evaluate.py                 the evaluation layer: scoring, judging, reporting, calibration
build_golden.py             builds the tiers of the golden dataset
corpus.json                 the 83 ingested videos, ordered as they were first walked
cache/                      83 Whisper transcripts, one per video
golden/                     the dataset, the stored run, the reports, the calibration record
golden/labelsheet.md        the gold chunk labels, marked by hand; the only copy of them
docs/adr/                   decisions whose reasoning is not recoverable from the code
tests/                      112 offline tests
```
