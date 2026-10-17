# Offline evaluation report

Generated 2026-10-17T13:15:30 · judge `openai/gpt-oss-20b` · generation uses top 5 chunks

Faithfulness, relevancy and correctness are judge scores on a continuous 0.00-1.00 scale; each question's per-score reasoning is kept in the `note` field of `report.json`.

Tiers are scored on what their ground truth supports and are never averaged together.

## Tier A — quiz questions with human answer keys

Correctness is anchored to the course authors' own answer key. No chunk labels, so no retrieval metrics.

| metric | value |
| --- | --- |
| questions | 83 |
| faithfulness | 0.793 |
| relevancy | 0.792 |
| correctness | 0.761 |
| citation_present | 0.831 |
| citation_video_valid | 0.988 |
| citation_time_valid | 0.488 |

### By course

| course | questions | correctness | faithfulness | citation_time_valid |
| --- | --- | --- | --- | --- |
| graph analytics | 31 | 0.81 | 0.789 | 0.469 |
| machine learning | 52 | 0.733 | 0.795 | 0.5 |

Hands-on questions (10 of tier A) ask about clicking through KNIME or Spark rather than lecture content: correctness 0.4.

## Tier B — generated from single chunks, exact gold labels

Retrieval metrics are exact here. The questions are synthetic, so read them as a retrieval measurement, not as user behaviour.

`precision@k_max` is the ceiling that k and the number of gold chunks allow: a question with one gold chunk can score at most 1/k, so compare each precision against its own maximum, not against 1.0.

| metric | value |
| --- | --- |
| questions | 40 |
| precision@1 | 0.625 |
| precision@1_max | 1.0 |
| recall@1 | 0.588 |
| precision@3 | 0.3 |
| precision@3_max | 0.375 |
| recall@3 | 0.838 |
| precision@5 | 0.2 |
| precision@5_max | 0.225 |
| recall@5 | 0.912 |
| precision@10 | 0.103 |
| precision@10_max | 0.113 |
| recall@10 | 0.925 |

Generation metrics on the same questions:

| metric | value |
| --- | --- |
| faithfulness | 0.963 |
| relevancy | 0.939 |
| correctness | 0.889 |
| citation_present | 0.975 |
| citation_video_valid | 0.987 |
| citation_time_valid | 0.62 |

## Tier C — refusal test

12 questions on topics verified absent from the corpus. The pipeline should decline all of them.

- refusal rate: **1.0**
