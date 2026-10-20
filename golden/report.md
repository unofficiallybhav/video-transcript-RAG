# Offline evaluation report

Generated 2026-10-20T15:27:54 · judge `openai/gpt-oss-20b` · generation uses top 5 chunks

Faithfulness, relevancy and correctness are judge scores on a continuous 0.00-1.00 scale; each question's per-score reasoning is kept in the `note` field of `report.json`.

Faithfulness is scored by a second judge call that is shown the question, the retrieved context and the answer, and never the reference answer, so that it can only be a claim about whether the context supports the answer. Relevancy and correctness come from the call that does see the reference answer, with both criteria worded exactly as before. The faithfulness call's own reasoning is kept separately, in `faithfulness_note`.

Tiers are scored on what their ground truth supports and are never averaged together.

Every value carries the number of questions it was taken over. A question that carries no value for a metric is left out of that metric rather than counted as a zero, so a citation validity value covers only the answers that cited something: "did not cite" and "cited wrongly" stay distinguishable, and the count column is how a reader tells which is which.

Provenance: the citation check was invalid in the run this report supersedes. It compared a cited timestamp, which the generator only ever saw floored to a whole second, against unfloored chunk spans, and it charged a timestamp against the nearest video name appearing before it even when the answer never cited that video. That citation defect was corrected 2026-10-19. Citation values here therefore measure a repaired ruler, not a changed pipeline, and are not comparable with any earlier copy of this report; the superseded values are discarded rather than kept beside them, so that a corrected measurement is never plotted as a trend.

Provenance: faithfulness values produced before 2026-10-20 came from a single judge call that also scored correctness, and are not comparable with the ones here. The calibration experiment recorded in `golden/judge_calibration.json` re-judged 30 questions with the reference answer withheld and measured a mean absolute faithfulness difference of 0.164 against a threshold of 0.10 fixed before the experiment was run, shifted +0.109: withholding the reference answer raises faithfulness, because with it visible the judge penalised answers that diverge from it, so faithfulness was partly grading correctness instead of support by the retrieved context. The pre-committed consequence was to split faithfulness into its own call permanently. `docs/adr/0002-faithfulness-judged-without-the-reference-answer.md` records the decision in full.

The choice of what to do with the superseded values was between re-judging the golden set and discarding the combined call's figures as not comparable. They are being re-judged, not discarded: faithfulness is re-measured over the stored answers by `uv run evaluate.py rejudge`, which retrieves and generates nothing, so the fresh score describes the same pipeline behaviour that was first measured. Only faithfulness is re-judged. Correctness and relevancy keep the values the combined call gave them and are unaffected by the split: their call is the same prompt it has always been, and re-judging them would move them for no reason this report could name. Until a row has been re-judged its old faithfulness value is left out of every figure here rather than averaged in beside a new one.

What the split did to the numbers, measured over the 123 re-judged answers: the tier means barely moved. Tier A's combined-call faithfulness mean was 0.793 and tier B's was 0.963, over the same answers the tables above now score with the split call. Individual answers moved much more than the means did — 0.093 mean absolute movement on tier A, 0.013 on tier B — so the split changed which answers score what, while leaving the tier aggregate close to where it was. That is not a reason to doubt the split. Faithfulness scored with the answer key in view was not a claim about support by the retrieved context whatever number it landed on, so this is a repair to the metric's validity and not a correction to its value, in the sense those two words carry in `GLOSSARY.md`. Nor is the +0.109 signed shift the calibration experiment measured this measurement: that experiment withheld the reference answer but still asked the judge for relevancy in the same call, over a 30-question sample, so its figure and this one are not the same quantity. The superseded values are discarded rather than kept beside these.

Cost: the faithfulness call is about 1,783 tokens per question, measured over the 123 scored tier A and B questions at the four-characters-per-token estimate the rate limiter paces on. It brings the cost of scoring one answer from about 1,970 tokens to about 3,753. Re-judging faithfulness alone over all 123 is about 227,000 tokens, which exceeds the 200,000 per day one API key allows — hence two keys, and hence `rejudge` resuming where it stopped.

## Tier A — quiz questions with human answer keys

Correctness is anchored to the course authors' own answer key. No chunk labels, so no retrieval metrics.

83 questions in the tier; each value below covers the ones in its own `questions` column.

| metric | value | questions |
| --- | --- | --- |
| faithfulness | 0.787 | 83 |
| relevancy | 0.792 | 83 |
| correctness | 0.761 | 83 |
| citation_present | 0.831 | 83 |
| citation_video_valid | 0.981 | 69 |
| citation_time_valid | 0.936 | 70 |

### By course

| course | questions | correctness | faithfulness | citation_time_valid | cited a time |
| --- | --- | --- | --- | --- | --- |
| graph analytics | 31 | 0.81 | 0.781 | 0.963 | 27 |
| machine learning | 52 | 0.733 | 0.79 | 0.919 | 43 |

Hands-on questions (10 of tier A) ask about clicking through KNIME or Spark rather than lecture content: correctness 0.4 over 10 of them.

## Retrieval — unmeasured

There are no retrieval figures in this report. `precision@k` and `recall@k` need chunk-level labels, and the only tier that carried them was tier B, whose questions were generated from the chunk that then served as their own gold label. A human review of 15 of those 40 questions rejected 2 as not answerable from the passage they were built from, and the tier was retired rather than regenerated: no threshold on lexical overlap separated the rejected questions from the sound ones. `docs/adr/0003-retiring-the-generated-retrieval-tier.md` records that decision and `golden/spotcheck.md` keeps the verdicts it rested on.

This is a gap, not a result. Retrieval runs on every question in this report — each answer below was generated from the top 5 chunks — and nothing here checks whether those were the right chunks. Until gold chunks are labelled by hand onto tier A's quiz questions, no claim about retrieval quality is supported either way.


## Refusal — both tiers together

An answerable refusal is a declined answer to a question whose tier's ground truth says the corpus can answer it, so it is a failure of the same pipeline whose refusals on tier C are a success. Both rates are derived from the refusal verdict the judge already recorded for each answer; nothing was judged again to obtain them.

Read the two rows as one measurement. A pipeline that declined every question would score 1.0 on tier C and look perfect there, so tier C's refusal rate is not a success on its own; nor is a low answerable refusal rate on a pipeline that never declines anything.

| tier | ground truth | metric | value | questions |
| --- | --- | --- | --- | --- |
| tier A | the corpus can answer | answerable refusal rate | 0.157 | 83 |
| tier C | no correct answer exists | refusal rate | 1.0 | 12 |

Tier C's 12 questions are on topics verified absent from the corpus; the pipeline should decline all of them. Every answerable refusal counted above scored correctness 0.0, so without this rate the failure is hidden inside a low correctness value instead of named.
