# Restating a question, and rescoring one answer

Status: accepted

Amends `0004-questions-the-corpus-cannot-answer-leave-tier-a.md` and
`0001-three-tier-golden-dataset.md`, both of which remain in force. ADR 0004 removed 17
Tier A questions and recorded that Tier A holds 66; this record removes an 18th and
restates one, so **Tier A holds 65 questions**. It also records a second route to a
question's answer key, which ADR 0001 did not contemplate.

Both decisions here were taken while finishing the hand-labelling that issue #12 exists
for. Neither was asked for by that issue, and the second contradicts a line in it. That
is the reason this record exists rather than a code comment.

## Decision

Two things, and the second is the one that costs something.

**1. `A-Quiz 4 - Graph Analytics with Neo4j-2` leaves Tier A.** Its answer is what a
Cypher query returns for the exercise's own graph — "All nodes except root nodes" — which
no lecture transcript states. It is the same kind as ADR 0004's fourteen and is excluded
by the same mechanism, `EXCLUDED_QUESTIONS` in `build_golden.py`, with its reason beside
it.

**2. `A-Quiz 2 - Graph Analytics Applications-16` is restated rather than removed**, and
its answer key is restated with it:

| | question | answer key |
| --- | --- | --- |
| authors' | Which of the statements below is true? | Dijsktra's algorithm is computationally inefficient (has high computational complexity). |
| restated | Is Dijsktra's algorithm computationally inefficient? | Yes. Dijsktra's algorithm has high computational complexity. |

The restatement is declared in `REPHRASED_QUESTIONS` and applied by `apply_rephrasings`.
The authors' wording and key are kept on the row under `original`, with the reason.

**3. `uv run evaluate.py run <id>` scores one question again**, regenerating its answer.
It was added to make decision 2 measurable and was used once, on that question.

## Why restating, when ADR 0004 removed questions of the same shape

ADR 0004 removed three questions because "the question stem survived parsing without its
options". That is exactly what was wrong with this one, so the precedent points at
removal and this record departs from it. The departure is narrow and the difference is in
the answer key.

ADR 0004's three were "all of the above" questions: the key itself named the options, so
with the options gone there was nothing left to be true *about*, and `keep_question` drops
them by `ANSWER_IS_ALL_OPTIONS_RE`. This question's key is substantive — it states a
property of Dijkstra's algorithm that the lectures do teach. Only the stem was
option-dependent, and `OPTION_DEPENDENT_RE` did not catch "the statements below". So the
material was answerable and the question was not; removal would have discarded a
measurable question for a defect in one sentence of it.

The restatement is the labeller's, written on the sheet, and the measurement bears it out:
the question went from having its gold chunk outside the retrieved top 10 to recall@5 of
1.00, with correctness and faithfulness both 1.00. It had been scored as a retrieval
failure when it was a question that could not be answered as written.

## What it costs, stated plainly

**Correctness is no longer anchored to the course authors' key for this one question.**
ADR 0001 admits Tier A because its labels come "from outside the system under test"; for
this question the key was rewritten inside the repo. That is a real weakening of the
anchor, affecting 1 of 65 questions, and it is why `original` rides on the row: any figure
drawn from Tier A can be checked against which questions are the authors' own.

**Issue #12 said the opposite about regenerating answers.** It promised "This costs no API
quota" and "No provider calls, no regenerated answers, judge fields copied through
unmodified". That holds for `rescore`, which is what the sentence was about, and it still
holds — `rescore` regenerates nothing. `run <id>` is a separate stage and does regenerate,
so the ticket delivered one thing it said it would not. It was used on one question and
cost one retrieval and two judge calls.

**A published figure moved.** `tests/test_rescore.py` pins Tier A's citation aggregate
over the stored run as a check that the offline re-derivation is faithful. Regenerating an
answer changes its citations, so `citation_present` moved 0.831 → 0.843 and
`citation_time_valid` 0.936 → 0.937. The pin was updated to the derived values, 82 of the
83 rows being untouched. `rejudge` had superseded rows there before without moving these,
because it re-judges a stored answer and never regenerates one; `run <id>` is the first
stage that does, and any future stage that regenerates an answer will move this pin again.

## Consequences

- Tier A holds 65 questions. A count of 66 or 83 in an earlier record is that record's own
  date, not a contradiction to fix.
- A restated question is the exception and must stay visible as one. `REPHRASED_QUESTIONS`
  exits when an id in it no longer parses, because a lapsed restatement would leave the
  sheet and the tier stating two different questions with nothing to show which was
  scored.
- Prefer fixing `keep_question` over restating. An option-dependent stem that reaches the
  tier is a hole in `OPTION_DEPENDENT_RE` first and a candidate for restatement second;
  this one was left restated because its key is sound and the question is worth measuring.
- `run <id>` is for a question whose wording or key changed. It is not a refresh: it
  supersedes a stored answer and moves any figure taken over it.
- Figures either side of this record are over different question sets — 66, then 65, and
  one of the 65 answered against a restated question. They are not a trend.
