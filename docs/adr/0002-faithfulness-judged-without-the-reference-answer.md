# Faithfulness is judged by its own call, which never sees the reference answer

Status: accepted

## Decision

Faithfulness is scored by a judge call that is shown the question, the retrieved context
and the system's answer, and nothing else. Correctness and relevancy continue to be
scored by the call that is shown the reference answer, in the prompt that call has always
used.

That the faithfulness judge is blind to the reference answer is part of the metric's
definition, not a detail of how it happens to be implemented. A faithfulness value
obtained with the reference answer visible is a different metric and is not comparable
with one obtained without it.

## Why

Faithfulness claims to measure what proportion of an answer's factual claims the
retrieved context supports. While it shared a call with correctness, the judge scoring it
could see the answer key, and the calibration experiment in #8 measured what that did.

Thirty questions were re-judged with the reference answer withheld. Faithfulness moved
0.164 mean absolute difference, against a threshold of 0.10. The signed shift was +0.109:
withholding the reference answer *raises* faithfulness, because with it visible the judge
penalised answers that diverge from it. Faithfulness was partly grading correctness —
the one thing it names itself as not measuring.

The threshold and the consequence it triggers were both fixed in #1 and #8 before the
experiment was run, precisely so that a result this size could not afterwards be re-read
as small-sample noise. `golden/judge_calibration.json` is the record, and is left in
place as that record.

The experiment's other threshold was not exceeded: between-run spread across two fresh
judge passes came in at 0.018 on its worst metric, so two-decimal continuous scoring is
kept.

## Why the other call was not tidied up

The obvious tidy-up is to delete the faithfulness criterion from the combined prompt,
since its answer is no longer used. It is not done, and should not be.

Correctness and relevancy have values measured over 135 questions. A prompt that no
longer mentions faithfulness is a different call: the criteria would be worded the same
but the request around them would not, and those two scores would move for a reason
nothing in the report could name. So the combined call keeps asking for faithfulness and
the answer is discarded at the call site. It costs about 70 tokens a question and it
keeps two metrics comparable across the split.

## What was done with the superseded values

The combined call's faithfulness values are re-judged, not discarded. Only faithfulness
is re-judged; correctness and relevancy keep the values they were measured with, for the
reason above. The `rejudge` stage re-measures faithfulness over the stored answers, one
call per answer, without retrieving or generating anything again.

Until that has run for a row, the row carries no faithfulness value the report will use:
a row records which call scored its faithfulness, and the report drops any value that did
not come from the split call. This matters in practice and not only in principle, because
re-judging the whole set costs more tokens than one API key's daily budget allows, so a
part-finished re-judge is the normal state for a while.

## What the re-judge measured

All 123 scored tier A and B answers were re-judged on 2026-10-20. The tier means barely
moved: tier A 0.793 to 0.787, tier B 0.963 to 0.962. Individual answers moved much more
than the means did — 0.093 mean absolute movement on tier A, 0.013 on tier B — so the
split changed which answers score what while leaving the tier aggregate close to where it
was.

This does not weaken the decision, and the decision was pre-committed so that a result of
any size could not reopen it. Faithfulness scored with the answer key in view was not a
claim about support by the retrieved context whatever number it landed on: this is a
repair to the metric's validity, not a correction to its value, in the sense `GLOSSARY.md`
gives those two words. The same distinction covered the citation repair in #1.

The +0.109 signed shift the calibration experiment measured is not this quantity. That
experiment withheld the reference answer but still asked the judge for relevancy in the
same call, over a 30-question stratified sample. The shipped faithfulness call asks for
faithfulness alone, over all 123. Both are recorded; neither is the other's replication.

## Consequences

- Scoring one answer costs two judge calls, about 3,700 tokens rather than 1,970. The
  extra call is about 1,783 tokens; re-judging faithfulness alone over the 123 scored
  tier A and B questions is about 227,000, more than one key's 200,000 per day.
- Faithfulness is reported for tiers A and B, whose ground truth is an answer the
  pipeline was expected to produce. It needs no reference answer and no chunk label, only
  the context that was retrieved, so the tier boundary does not restrict it further.
- Two prompts now derive from `JUDGE_PROMPT`, each through guarded edits that fail loudly
  if it is reworded. Rewording `JUDGE_PROMPT` is therefore a deliberate act with a
  blast radius, and the error message says what to rederive.
- Faithfulness figures from before 2026-10-20 are not comparable with later ones and are
  not retained beside them.
