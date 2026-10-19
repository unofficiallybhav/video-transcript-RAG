# Offline evaluation report

Generated 2026-10-19T11:38:43 · judge `openai/gpt-oss-20b` · generation uses top 5 chunks

Faithfulness, relevancy and correctness are judge scores on a continuous 0.00-1.00 scale; each question's per-score reasoning is kept in the `note` field of `report.json`.

Tiers are scored on what their ground truth supports and are never averaged together.

Every value carries the number of questions it was taken over. A question that carries no value for a metric is left out of that metric rather than counted as a zero, so a citation validity value covers only the answers that cited something: "did not cite" and "cited wrongly" stay distinguishable, and the count column is how a reader tells which is which.

Provenance: the citation check was invalid in the run this report supersedes. It compared a cited timestamp, which the generator only ever saw floored to a whole second, against unfloored chunk spans, and it charged a timestamp against the nearest video name appearing before it even when the answer never cited that video. That citation defect was corrected 2026-10-19. Citation values here therefore measure a repaired ruler, not a changed pipeline, and are not comparable with any earlier copy of this report; the superseded values are discarded rather than kept beside them, so that a corrected measurement is never plotted as a trend.

## Tier A — quiz questions with human answer keys

Correctness is anchored to the course authors' own answer key. No chunk labels, so no retrieval metrics.

83 questions in the tier; each value below covers the ones in its own `questions` column.

| metric | value | questions |
| --- | --- | --- |
| faithfulness | 0.793 | 83 |
| relevancy | 0.792 | 83 |
| correctness | 0.761 | 83 |
| citation_present | 0.831 | 83 |
| citation_video_valid | 0.981 | 69 |
| citation_time_valid | 0.936 | 70 |

### By course

| course | questions | correctness | faithfulness | citation_time_valid | cited a time |
| --- | --- | --- | --- | --- | --- |
| graph analytics | 31 | 0.81 | 0.789 | 0.963 | 27 |
| machine learning | 52 | 0.733 | 0.795 | 0.919 | 43 |

Hands-on questions (10 of tier A) ask about clicking through KNIME or Spark rather than lecture content: correctness 0.4 over 10 of them.

## Tier B — generated from single chunks, exact gold labels

Retrieval metrics are exact here. The questions are synthetic, so read them as a retrieval measurement, not as user behaviour.

40 questions in the tier; each value below covers the ones in its own `questions` column.

`precision@k_max` is the ceiling that k and the number of gold chunks allow: a question with one gold chunk can score at most 1/k, so compare each precision against its own maximum, not against 1.0.

| metric | value | questions |
| --- | --- | --- |
| precision@1 | 0.625 | 40 |
| precision@1_max | 1.0 | 40 |
| recall@1 | 0.588 | 40 |
| precision@3 | 0.3 | 40 |
| precision@3_max | 0.375 | 40 |
| recall@3 | 0.838 | 40 |
| precision@5 | 0.2 | 40 |
| precision@5_max | 0.225 | 40 |
| recall@5 | 0.912 | 40 |
| precision@10 | 0.103 | 40 |
| precision@10_max | 0.113 | 40 |
| recall@10 | 0.925 | 40 |

Generation metrics on the same questions:

| metric | value | questions |
| --- | --- | --- |
| faithfulness | 0.963 | 40 |
| relevancy | 0.939 | 40 |
| correctness | 0.889 | 40 |
| citation_present | 0.975 | 40 |
| citation_video_valid | 0.987 | 39 |
| citation_time_valid | 0.974 | 39 |

## Refusal — all three tiers together

An answerable refusal is a declined answer to a question whose tier's ground truth says the corpus can answer it, so it is a failure of the same pipeline whose refusals on tier C are a success. Both rates are derived from the refusal verdict the judge already recorded for each answer; nothing was judged again to obtain them.

Read the three rows as one measurement. A pipeline that declined every question would score 1.0 on tier C and look perfect there, so tier C's refusal rate is not a success on its own; nor is a low answerable refusal rate on a pipeline that never declines anything.

| tier | ground truth | metric | value | questions |
| --- | --- | --- | --- | --- |
| tier A | the corpus can answer | answerable refusal rate | 0.157 | 83 |
| tier B | the corpus can answer | answerable refusal rate | 0.025 | 40 |
| tier C | no correct answer exists | refusal rate | 1.0 | 12 |

Tier C's 12 questions are on topics verified absent from the corpus; the pipeline should decline all of them. Every answerable refusal counted above scored correctness 0.0, so without this rate the failure is hidden inside a low correctness value instead of named.
