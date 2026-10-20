# Retiring the generated retrieval tier, and moving gold chunks to hand labelling

Status: accepted

Amends `0001-three-tier-golden-dataset.md`, which remains in force. This record retires
one of the three label sources that ADR allowed; it does not touch the constraint that
produced them.

## Decision

Tier B is removed from the golden dataset, from the stored run and from the report, with
its retrieval figures and its generation figures alike. The report states that retrieval
is unmeasured rather than omitting the section. Chunk-level gold labels will instead be
hand-labelled onto Tier A's author-written quiz questions.

## Why the tier was retired rather than regenerated

ADR 0001 admitted three label sources: course authors' answer keys, construction, and
hand verification. Tier B took the construction route — a question generated from one
chunk has that chunk as its label by definition — because it was the only route that
produced chunk labels for free.

The review ADR 0001 required found the flaw in that route. **Construction guarantees the
label is right. It guarantees nothing about whether the question is answerable, or
whether it is a question anyone would ask.** Of 15 questions reviewed against the passage
each was built from, 2 were rejected as not answerable from it: a 13% error rate carried
by every figure in the tier.

A threshold would not have fixed it. The overlap guard `MAX_JACCARD` stood at 0.35, and
tightening it was the obvious repair:

- `B-building a degree histogram.mp4:3`, rejected, sat at jaccard 0.286 — a tighter
  threshold would have caught this one.
- `B-measuring graph evolution.mp4:0`, rejected, sat at jaccard 0.101 with bm25 rank 1.
  No mechanical signal flagged it at all.
- `B-djikstra algotrithm.mp4:1`, marked sound but hands-on, sat at jaccard 0.099 with
  bm25 rank 2 — indistinguishable by signal from questions that held up.

Lexical overlap measures whether a question copies its source's wording. The defect is
that a question asks about something only visible on screen, or asks for a figure the
passage states without the context that makes it a question. Those are unrelated
properties, so no setting of the guard separates them.

Regenerating the tier under a tighter threshold would also have invalidated the stored
135-row run and the 15 human verdicts just recorded, in exchange for questions with the
same defect at a lower rate.

## Why the generation figures went too

Tier B's correctness (0.889) and faithfulness (0.962) were markedly stronger than Tier
A's (0.761 and 0.787), and it was tempting to keep them as a second generation sample.

They are scores against reference answers that were generated alongside the questions, by
the same call, from the same passage. A question that is not sound does not yield a sound
answer key, and 27% of the reviewed sample asked about a hands-on demonstration — the
pipeline's weakest category on Tier A, where it scores 0.40. Keeping figures that are
both untrusted at their source and plausibly flattered by their composition, while
discarding the retrieval figures from the same questions, would have been selective.

**What this protects against:** quoting the comfortable half of a retired measurement.

## Why the report names the gap

Retrieval is unmeasured now, and an absent table reads as either an oversight or as
nothing to report. The report carries a named section saying retrieval is unmeasured,
why, and what would restore it. A reader who scans for recall and finds nothing must land
on the reason.

**What this protects against:** a gap in the evidence being mistaken for an absence of
problems. This is the same instinct as ADR 0001's refusal to publish a headline score.

## Why hand labelling, and why Tier A

ADR 0001's consequences already name the remedy: "A metric that needs a label no tier has
is not added; a tier is extended, or the metric is not reported." Tier A is extended.

Tier A's questions come from real course quizzes with answer keys the course authors
wrote — labels from outside the system under test, which is the whole constraint. ADR
0001 names their limitation exactly: they "carry no chunk labels". Supplying those labels
by hand is the third admissible source, hand verification, which this project had not
used for retrieval.

Two alternatives were considered and rejected:

- **A public retrieval benchmark.** BEIR, MS MARCO and the like ship their own corpora,
  so their relevance judgements cannot score retrieval over these 406 video chunks. Run
  against a second index they would measure the embedding and chunking stack on a foreign
  corpus, which is a different question from whether retrieval works on this one.
- **A model proposing the gold chunk.** This is precisely the circularity ADR 0001
  forbids, and it is the route that just failed. A mechanical shortlist by bm25 and
  embedding similarity is not a labelling decision and is allowed; choosing from it is,
  and stays with a human.

Hand labelling also closes a gap ADR 0001 recorded and could not act on: Tier B rows
carry no `hands_on` field, so nothing measured whether its stronger generation figures
came from its question construction. Labelled onto Tier A, retrieval and generation are
measured over one question set with one `hands_on` field, and that becomes testable.

## Consequences

- **No retrieval metric is reported until Tier A is labelled.** `precision@k` and
  `recall@k` are not estimated, approximated or carried forward from the retired tier.
- **The retrieval helpers stay**, and they divide into two kinds.

  `precision_recall` and `K_VALUES` in `evaluate.py` are still *called*, by `run()` and
  `rescore_rows()`, inside a branch guarded on a row carrying `gold_chunks`. No row
  carries them, so the branch does not execute — the code is live and the data is
  absent. Scoring a labelled row is therefore a question of supplying labels, not of
  restoring code.

  `RETRIEVAL_METRICS` in `evaluate.py`, and `content_words`, `jaccard` and `tokenise` in
  `build_golden.py`, are genuinely unreferenced. They are kept because hand labelling
  needs exactly them — a table of retrieval columns to render, and mechanical signals to
  rank and annotate candidate chunks — and because a model may not be asked to rank the
  candidates instead. This is recorded at each of them.

- **The bm25 index construction went, against the ticket's own instruction to keep it.**
  The ticket said the bm25 machinery should stay because hand labelling would reuse it.
  On reading the code that turned out to describe something that did not exist: bm25
  appeared only inside `sample_chunks`, which built an index over every chunk, scored the
  tier A questions against it, and then drew a seeded random sample of chunks to generate
  questions *from*. That is tier B's sampling procedure, parameterised by `TIER_B_SIZE`
  and `TIER_B_SEED`, not a reusable way to rank candidates for a question. Keeping it
  would have meant keeping the retired tier's sampling logic to preserve the two lines
  that build the index.

  What hand labelling needs is the opposite direction — given one question, rank the
  chunks — so it is a new function either way. `tokenise` is kept because it is the part
  that genuinely carries over, and `rank-bm25` stays in `pyproject.toml` so that
  labelling does not open with a dependency change.
- **`golden/judge_calibration.json` is not regenerated.** It was measured under strata of
  20 Tier A and 10 Tier B questions. It is the record of the experiment #9 acted on and
  is cited by name in ADR 0002, the report's provenance note and the glossary.
  `CALIBRATION_STRATA` now reads 30 Tier A for any future run; the stored artifact
  describes the run that happened, not the one the constant would now produce.
- **`golden/spotcheck.md` stays tracked** although the stage that wrote it is gone. It is
  the only record of the 15 human verdicts this decision rests on, and deleting the
  evidence along with the thing it condemned would leave this ADR unfalsifiable.
- **The 3 Tier B gold chunks never retrieved at any k** were the only evidence of an
  index-imposed ceiling on recall. That finding goes with the tier and is not carried
  forward as a claim; whether such a ceiling exists is open again.
- **Reopening this means re-reading ADR 0001 first.** The no-model-in-labelling rule is
  unchanged and is what motivates the replacement, not an obstacle to it.
