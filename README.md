# VIDEO-TRANSCRIPT-RAG

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

## THE PIPELINE, END TO END

All of it lives in [`video_transcript_rag.py`](video_transcript_rag.py).

### 1. INGESTION

`ingest_all(root)` walks a media directory and picks up anything with a recognised
extension — `.mp4`, `.mkv`, `.webm`, `.m4a`, `.mp3`, `.wav`, `.flac`. The folder a video
sits in names its course, which becomes a payload field and later a retrieval filter.
Each file then runs the full transcript → chunks → vectors path, and a failure on one
file is reported and skipped rather than killing the batch.

### 2. TRANSCRIPTION

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

### 3. CHUNKING

Segments are accumulated until a chunk reaches **1200 characters**, then flushed, with the
last **2 segments** carried into the next chunk as overlap. The tail is skipped when it
contains nothing but that overlap.

Chunks are split on segment boundaries rather than mid-sentence, and each one keeps the
`start` of its first segment and the `end` of its last. Those timestamps are what makes a
citation checkable, so chunking is not a purely lexical step here — it is what the
accuracy of every later citation claim rests on.

### 4. EMBEDDING

`sentence-transformers` with `all-MiniLM-L6-v2`, 384 dimensions, encoded in a batch per
video. Local, no API call.

### 5. INDEXING

One qdrant collection, `video transcript`, cosine distance, with keyword payload indexes
on `source` and `course` so retrieval can be filtered to one video or one course.

Point ids are **deterministic**: `uuid5(NAMESPACE_URL, "<source>:<chunk_index>")`. That
makes re-ingesting a video idempotent, and it is why a chunk id like
`building a graph.mp4:3` means the same thing in the index, in the golden dataset and in
the stored evaluation run. Before upserting, every existing point for that `source` is
deleted, so a re-ingest that produces fewer chunks does not leave stale ones behind.

### 6. RETRIEVAL

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
compared against the unfloored start would mark correct behaviour as wrong.

### 7. GENERATION

Groq, `openai/gpt-oss-120b`, `temperature=0.0`. The system prompt carries the retrieved
context and asks for a short summary of what was discussed plus the video name and
timestamp, and instructs the model to decline rather than answer from outside the context.

`answer_messages()` builds the exact message list the deployed path sends, and the
evaluator replays *that function*, so the measured prompt and the shipped prompt cannot
drift apart.

## USING IT

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

## THE EVALUATION LAYER

### HOW IT WORKS

**Ground truth is partitioned by what kind of ground truth it is.** Tier A is 65 course
quiz questions with answer keys the course authors wrote. Tier C is 12 questions on topics
verified absent from the corpus, where declining to answer is the only correct behaviour.
Each tier is scored only on what its own ground truth supports, and the tiers are never
averaged — a blended number would cancel opposing failures, since refusing is correct on
Tier C and a defect on Tier A.

**No model participates in a labelling decision.** Not in deciding a question's gold
chunk, not in deciding whether a question is answerable. `uv run build_golden.py label`
writes `golden/labelsheet.md`: every Tier A question beside a shortlist of candidate
chunks, each shown with its transcript text, its bm25 and embedding ranks, and its lexical
overlap with the answer key. Those signals choose what a human reads; they never decide
what is gold. `uv run build_golden.py gold` reads the marks back into the golden set.
Marking is incremental and the report is correct part-way through — an unmarked question
is counted as unlabelled and left out of every retrieval figure rather than scored zero in
one.

**Generation is judged by a model that is not the generator**, `openai/gpt-oss-20b`.
Faithfulness is scored in its own judge call that never sees the reference answer, so it
can only be a claim about whether the retrieved context supports the answer; correctness
and relevancy come from the call that does see the key.

**Retrieval costs nothing to re-measure.** Every stored run keeps the 10 chunk ids it
retrieved, so precision and recall at every k re-derive offline from a run already paid
for. `uv run evaluate.py rescore` re-derives the judge-free metrics and rewrites
`golden/report.md`.

**A measurement change is never plotted as a trend.** Where a metric has been found
invalid and repaired, the superseded values are discarded rather than kept beside the new
ones, and the report says which metric was repaired, when, and why its old values are not
comparable. Figures are also always reported with the number of questions they were taken
over, because the tier's size has changed across sessions.

### RESULTS

**Tier A** — 65 quiz questions, human answer keys. Each value covers the questions that
carry it, which is why the counts differ: a citation validity figure covers only the
answers that cited something.

| metric | value | questions |
|---|---|---|
| correctness | 0.903 | 65 |
| faithfulness | 0.879 | 65 |
| relevancy | 0.926 | 65 |
| citation_present | 0.938 | 65 |
| citation_video_valid | 0.986 | 61 |
| citation_time_valid | 0.935 | 62 |
| answerable refusal rate | 0.046 | 65 |

**Tier C** — 12 unanswerable questions: refusal rate **1.0**. Read beside Tier A's
answerable refusal rate, never on its own — a pipeline that declined everything would
score 1.0 here and look perfect.

**Retrieval** — over the **60 of 65** Tier A questions carrying hand-labelled gold chunks.
The other 5 are labelled as having no gold chunk in the corpus and are excluded from recall
rather than scored zero in it, because an index gap and a retrieval miss are different
findings with different remedies. No question in the tier is unlabelled.

| k | recall@k | precision@k | ceiling (`precision@k_max`) |
|---|---|---|---|
| 1 | 0.282 | 0.533 | 1.0 |
| 3 | 0.585 | 0.389 | 0.661 |
| 5 | **0.703** | 0.287 | 0.410 |
| 10 | 0.810 | 0.168 | 0.205 |

Generation is scored on the top 5, so `recall@5` 0.703 is the figure that bears on the
answers: about a third of the material the labeller judged necessary never reaches the
generator. Read each precision against the ceiling beside it, not against 1.0 — the
labelled questions average 2.05 gold chunks, so `precision@5` cannot exceed 0.410.

**The two courses fail differently**, which is the measurement's most useful result:

| course | n | recall@5 | recall@10 | gold entirely outside top 10 | gold chunks per question |
|---|---|---|---|---|---|
| machine learning | 35 | 0.762 | 0.902 | 0 of 35 | 2.40 |
| graph analytics | 25 | 0.620 | 0.680 | 5 of 25 | 1.56 |

Machine learning's misses sit at ranks 6–10 — a ranking problem, reachable by re-ranking.
Graph analytics gains only 6 points from k=5 to k=10, so re-ranking buys it almost nothing:
its gold is absent from the top 10 altogether, which is a coverage problem. Whether that is
chunking, embedding or a genuine index gap is unmeasured. Because the two courses differ in
gold chunks per question their `precision@k_max` ceilings differ, so their precision columns
are not directly comparable.

Generation over the same 60 labelled questions, so retrieval and generation can be read
against each other rather than over two differently sized sets: correctness 0.928,
faithfulness 0.920, relevancy 0.954, citation_present 0.967.

## RUNNING THE EVALUATION

The offline stages need no network, no API credentials and no GPU. A clone can reproduce
the whole report from the stored run:

```sh
uv run pytest                  # 134 tests, no network, no credentials
uv run evaluate.py rescore     # re-derive the judge-free metrics and rewrite the report
```

> **Use `rescore`, not `report`.** `uv run evaluate.py report` rebuilds the report from the
> values stored in the checkpoint rather than re-deriving them, which silently reverts
> Tier A `citation_time_valid` to the superseded 0.488. After any change that regenerates
> the report, run `rescore`.

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
uv run evaluate.py run "<id>"  # score one question again, when its wording or key changed
uv run evaluate.py rejudge     # re-measure faithfulness with the reference withheld
uv run evaluate.py calibrate   # the judge calibration experiment (~73k tokens)
```

`run <id>` overrides the checkpoint for that one question and appends the fresh row, which
supersedes the earlier one by id. It does not render the report — one row cannot state
figures about the whole golden set — so follow it with `rescore`.

Set `PYTHONIOENCODING=utf-8` whenever an answer or report is printed: stored answers
contain characters cp1252 cannot encode.

Dependencies are managed with `uv`, never pip. The video files themselves are gone and are
not required — `corpus.json` lists the 83 ingested videos and `cache/` holds one transcript
each, which together reproduce the 406 chunks and the chunk ids already in qdrant.

## LAYOUT

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
tests/                      134 offline tests
```
