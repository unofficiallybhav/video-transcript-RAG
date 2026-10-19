# Three tiers of ground truth, reported separately, with no model in a labelling decision

Status: accepted

## Decision

The golden dataset is split into three tiers by the kind of ground truth its questions
carry. Each tier is scored only on the metrics its own ground truth supports; tier
metrics are reported separately and are never averaged into a single headline number;
and no model participates in any labelling decision, including the decision of what a
question's gold chunks are and the decision of whether a question is answerable at all.

The third of these is the constraint, and the first two are what it forces. The tiers are
not a difficulty ladder and not an organisational convenience: they exist because the
only labels available for free are model-produced, and model-produced labels are not
admissible here.

## Why no model in a labelling decision

A judge that shares a blind spot with the generator will score that blind spot as
success. Concretely: labelling which chunk a quiz question "should" have retrieved needs
a judgement call. Ask a model for it, and when the model cannot see the answer in the
corpus it labels the question unanswerable, the generator then refuses, and the refusal
is scored as correct. The pipeline's worst failure mode — declining a question the corpus
answers — becomes invisible, and the number that should expose it instead endorses it.
The same circularity reaches any metric built on model-assigned labels.

So labels come from outside the system under test: from the course authors who wrote the
quiz answer keys, from construction (a question generated from one chunk has that chunk
as its label by definition), or from hand verification that a topic is absent. A model is
allowed to score an answer against a label it did not create. It is never allowed to
create the label.

**What this protects against:** the evaluation certifying its own blind spots. Every
cheaper labelling route is some form of asking a model what the right answer is, which is
the question the evaluation exists to answer.

## Why three tiers rather than one dataset

No single source of ground truth supports every metric. Author-written answer keys
support correctness and citation checking but carry no chunk labels. Chunk labels obtained
by construction support precision and recall exactly, but the questions built that way
are synthetic and share vocabulary with their source, so they flatter retrieval. Verified
absence supports the refusal path and nothing else.

Merging them does not produce a dataset with all three strengths. It produces one where
every number carries the weakest contributing tier's caveat, and where the caveat is no
longer attached to anything a reader can see.

**What this protects against:** the collapse back into one dataset and one judge, which
is the obvious tidy-up and which reintroduces model-produced labels by the back door.

## Why tier metrics are never averaged

An average across tiers is arithmetically available and semantically empty, because the
tiers do not measure the same thing on comparable populations. Averaging also cancels
opposing failures: refusal is the desired behaviour on Tier C and a defect on Tiers A and
B, so a pipeline that refuses everything scores perfectly on one tier and merely poorly
on the others, and a blended number reads as mediocre rather than as broken. A reader who
sees only the blend cannot recover which tier moved.

Metrics defined only for one tier are reported only for that tier. Metrics defined for
several are reported per tier, side by side, and the report never offers a single overall
figure.

**What this protects against:** a dashboard-shaped summary. The pressure to produce one
comparable number is constant, and it is exactly the pressure the split exists to resist.

## Consequences

- Any new metric must name the tier whose ground truth justifies it. A metric that needs
  a label no tier has is not added; a tier is extended, or the metric is not reported.
- The report is longer and has no headline score. This is deliberate and should not be
  "simplified".
- Tier B's retrieval figures remain contingent on a human spot-check of its generated
  questions until that review is run. The tier boundary is what keeps that caveat
  attached to the retrieval numbers and off the Tier A numbers.
- Reopening any part of this decision means re-reading this record first: the reasoning
  is not reconstructable from the metrics or from the module docstrings, which state the
  design without stating what it is defending.
