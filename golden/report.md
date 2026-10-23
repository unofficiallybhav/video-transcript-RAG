# Offline evaluation report

Generated 2026-10-23T16:14:38 · judge `openai/gpt-oss-20b` · generation uses top 5 chunks

Faithfulness, relevancy and correctness are judge scores on a continuous 0.00-1.00 scale; each question's per-score reasoning is kept in the `note` field of `report.json`.

Faithfulness is scored by a second judge call that is shown the question, the retrieved context and the answer, and never the reference answer, so that it can only be a claim about whether the context supports the answer. Relevancy and correctness come from the call that does see the reference answer, with both criteria worded exactly as before. The faithfulness call's own reasoning is kept separately, in `faithfulness_note`.

Tiers are scored on what their ground truth supports and are never averaged together.

Every value carries the number of questions it was taken over. A question that carries no value for a metric is left out of that metric rather than counted as a zero, so a citation validity value covers only the answers that cited something: "did not cite" and "cited wrongly" stay distinguishable, and the count column is how a reader tells which is which.

Provenance: the citation check was invalid in the run this report supersedes. It compared a cited timestamp, which the generator only ever saw floored to a whole second, against unfloored chunk spans, and it charged a timestamp against the nearest video name appearing before it even when the answer never cited that video. That citation defect was corrected 2026-10-19. Citation values here therefore measure a repaired ruler, not a changed pipeline, and are not comparable with any earlier copy of this report; the superseded values are discarded rather than kept beside them, so that a corrected measurement is never plotted as a trend.

Provenance: faithfulness values produced before 2026-10-20 came from a single judge call that also scored correctness, and are not comparable with the ones here. The calibration experiment recorded in `golden/judge_calibration.json` re-judged 30 questions with the reference answer withheld and measured a mean absolute faithfulness difference of 0.164 against a threshold of 0.10 fixed before the experiment was run, shifted +0.109: withholding the reference answer raises faithfulness, because with it visible the judge penalised answers that diverge from it, so faithfulness was partly grading correctness instead of support by the retrieved context. The pre-committed consequence was to split faithfulness into its own call permanently. `docs/adr/0002-faithfulness-judged-without-the-reference-answer.md` records the decision in full.

Provenance: that calibration was measured on 2026-10-20 over a stratified sample of 20 tier A and 10 tier B questions. Tier B has since been retired, so the sample is not reproducible and the artifact is read as of that date rather than as a figure the current code would produce. It is kept unchanged because it is the record of the experiment the faithfulness split acted on; re-running calibration under the present strata would be a different measurement, not a refresh of this one.

The choice of what to do with the superseded values was between re-judging the golden set and discarding the combined call's figures as not comparable. They are being re-judged, not discarded: faithfulness is re-measured over the stored answers by `uv run evaluate.py rejudge`, which retrieves and generates nothing, so the fresh score describes the same pipeline behaviour that was first measured. Only faithfulness is re-judged. Correctness and relevancy keep the values the combined call gave them and are unaffected by the split: their call is the same prompt it has always been, and re-judging them would move them for no reason this report could name. Until a row has been re-judged its old faithfulness value is left out of every figure here rather than averaged in beside a new one.

What the split did to the numbers, measured over the 123 re-judged answers: the tier means barely moved. Tier A's combined-call faithfulness mean was 0.793 and tier B's was 0.963, over the same answers the tables above now score with the split call. Individual answers moved much more than the means did — 0.093 mean absolute movement on tier A, 0.013 on tier B — so the split changed which answers score what, while leaving the tier aggregate close to where it was. That is not a reason to doubt the split. Faithfulness scored with the answer key in view was not a claim about support by the retrieved context whatever number it landed on, so this is a repair to the metric's validity and not a correction to its value, in the sense those two words carry in `GLOSSARY.md`. Nor is the +0.109 signed shift the calibration experiment measured this measurement: that experiment withheld the reference answer but still asked the judge for relevancy in the same call, over a 30-question sample, so its figure and this one are not the same quantity. The superseded values are discarded rather than kept beside these.

Cost: the faithfulness call is about 1,783 tokens per question, measured over the 123 scored tier A and B questions at the four-characters-per-token estimate the rate limiter paces on. It brings the cost of scoring one answer from about 1,970 tokens to about 3,753. Re-judging faithfulness alone over all 123 is about 227,000 tokens, which exceeds the 200,000 per day one API key allows — hence two keys, and hence `rejudge` resuming where it stopped.

## Tier A — quiz questions with human answer keys

Correctness is anchored to the course authors' own answer key. The chunk labels that carry retrieval are hand-written onto this tier and reported in their own section below, over the questions that have them.

66 questions in the tier; each value below covers the ones in its own `questions` column.

| metric | value | questions |
| --- | --- | --- |
| faithfulness | 0.861 | 66 |
| relevancy | 0.912 | 66 |
| correctness | 0.889 | 66 |
| citation_present | 0.924 | 66 |
| citation_video_valid | 0.986 | 61 |
| citation_time_valid | 0.935 | 62 |

### By course

| course | questions | correctness | faithfulness | citation_time_valid | cited a time |
| --- | --- | --- | --- | --- | --- |
| graph analytics | 31 | 0.81 | 0.781 | 0.963 | 27 |
| machine learning | 35 | 0.96 | 0.931 | 0.914 | 35 |

Hands-on questions are no longer a reportable slice of this tier: 1 remain, against the 10 the earlier 0.40 correctness figure was taken over. The exercise quizzes those came from were removed because their answers are figures from an exercise's dataset rather than anything the lectures state — see the provenance note at the end.

## Retrieval — tier A, hand-labelled

**33 of 66** tier A questions carry hand-labelled gold chunks. Every figure below is taken over those alone; the remaining 33 are not yet labelled and are absent from the metric rather than scored zero in it.

These labels were made by a human choosing from a shortlist that bm25 and embedding similarity ranked and annotated. No model took part in the decision, which is what separates them from the retired generated tier's labels: those were correct by construction and said nothing about whether the question was worth asking. See `docs/adr/0001-three-tier-golden-dataset.md` for the admissible label sources and `docs/adr/0003-retiring-the-generated-retrieval-tier.md` for why this is the only one left.

`precision@k_max` is the highest `precision@k` these questions could reach: a question with one gold chunk cannot score above 1/k however well retrieval works. Read each precision against the ceiling beside it, not against 1.0.

| metric | value | questions |
| --- | --- | --- |
| precision@1 | 0.545 | 33 |
| precision@1_max | 1.0 | 33 |
| recall@1 | 0.265 | 33 |
| precision@3 | 0.465 | 33 |
| precision@3_max | 0.727 | 33 |
| recall@3 | 0.629 | 33 |
| precision@5 | 0.345 | 33 |
| precision@5_max | 0.461 | 33 |
| recall@5 | 0.778 | 33 |
| precision@10 | 0.206 | 33 |
| precision@10_max | 0.23 | 33 |
| recall@10 | 0.917 | 33 |

Generation over the same 33 questions, so the two can be read against each other rather than over two differently sized sets:

| metric | value | questions |
| --- | --- | --- |
| faithfulness | 0.977 | 33 |
| relevancy | 0.981 | 33 |
| correctness | 0.958 | 33 |
| citation_present | 1.0 | 33 |
| citation_video_valid | 0.985 | 33 |
| citation_time_valid | 0.924 | 33 |

### Hands-on — not reportable

1 of the 33 labelled questions carry the hands-on flag, which supports no claim and is not tabled. The breakout was added to test whether the pipeline's weakness on hands-on questions was a retrieval failure or a generation failure. That weakness was measured over 10 questions, and the exercise quizzes those came from have since left tier A because the corpus cannot answer them — so the hypothesis is no longer answerable from this tier, and was dissolved rather than settled. Any future test of it needs hands-on questions whose answers the lectures actually state.

## Refusal — both tiers together

An answerable refusal is a declined answer to a question whose tier's ground truth says the corpus can answer it, so it is a failure of the same pipeline whose refusals on tier C are a success. Both rates are derived from the refusal verdict the judge already recorded for each answer; nothing was judged again to obtain them.

Read the two rows as one measurement. A pipeline that declined every question would score 1.0 on tier C and look perfect there, so tier C's refusal rate is not a success on its own; nor is a low answerable refusal rate on a pipeline that never declines anything.

| tier | ground truth | metric | value | questions |
| --- | --- | --- | --- | --- |
| tier A | the corpus can answer | answerable refusal rate | 0.061 | 66 |
| tier C | no correct answer exists | refusal rate | 1.0 | 12 |

Tier C's 12 questions are on topics verified absent from the corpus; the pipeline should decline all of them. Every answerable refusal counted above scored correctness 0.0, so without this rate the failure is hidden inside a low correctness value instead of named.

Provenance: 17 questions that this run scored have since been removed from the golden set, and are left out of every figure above. They asked for figures produced by running a KNIME or Spark exercise — a percentage of a dataset, a cluster number, the name of a column — which no lecture transcript states, so tier A's ground truth that the corpus can answer the question did not hold for them. While they were in the tier they scored correctness 0.265 over 17 of them, 12 of them scoring 0.0 and 9 refused, so every tier A mean above is higher than in any earlier copy of this report partly for that reason. That is a change in which questions are measured, not a change in the pipeline, so the superseded figures are discarded rather than kept beside these. `build_golden.py`'s `EXCLUDED_QUIZZES` and `EXCLUDED_QUESTIONS` name each removed question with its reason.
