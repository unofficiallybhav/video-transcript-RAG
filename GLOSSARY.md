# Video Transcript RAG

A retrieval-augmented question answering pipeline over course video transcripts, and an
offline evaluation layer that decides whether a change to that pipeline is an
improvement. Most of the vocabulary below belongs to the evaluation layer, because that
is where the project's own invented language lives.

## Language

### The golden dataset

**Tier**:
A partition of the golden dataset by the kind of ground truth its questions carry, and
therefore by which metrics can be trusted about them.
_Avoid_: split, bucket, category, difficulty level, tranche

**Tier A**:
The tier whose questions come from course quizzes with answer keys written by the course
authors. Its ground truth is a reference answer, not a chunk label.
_Avoid_: the quiz tier, the human tier

**Tier B**:
Retired on 2026-10-20. The tier whose questions were each generated from a single chunk,
so that chunk was the retrieval target by construction. The term is kept here because
older reports, commits and the calibration record still use it; it names nothing in the
current dataset. See `docs/adr/0003-retiring-the-generated-retrieval-tier.md`.
_Avoid_: using it for any current tier; writing as though three tiers exist

**Tier C**:
The tier whose questions are about topics verified absent from the corpus. Its ground
truth is that no correct answer exists.
_Avoid_: the negative tier, the adversarial tier, the unanswerable tier

**Gold chunk**:
The chunk a question is scored as having needed, plus any chunk that duplicates the
material that makes it answerable. A question has gold chunks only when they were
established without a model's opinion. The construction route that once supplied them —
generating the question from the chunk — is retired, so the only remaining route is a
human choosing from a mechanically ranked shortlist. No question currently carries gold
chunks.
_Avoid_: relevant chunk, correct chunk, ground-truth chunk, target chunk, expected chunk

### Measurement

**Judge-free metric**:
A metric whose value follows from the recorded run and the corpus alone, with no model
asked for an opinion at any point in deriving it. Such a metric can be recomputed from
stored artifacts, and recomputing it yields the same number every time.
_Avoid_: objective metric, deterministic metric, automatic metric, hard metric

**Faithfulness**:
What proportion of the factual claims an answer makes is supported by the context that
was retrieved for it. A claim about support, not about truth: an answer can be faithful
to a passage that is itself wrong, and an answer that is right about the world but
absent from the context is unfaithful.
_Avoid_: groundedness, hallucination rate, factuality, answer support

Reported for Tier A. That the judge scoring it is never shown the reference
answer belongs to the definition and is not an implementation detail: a value obtained
with the reference answer visible is a different metric and is not comparable with one
obtained without it. See `docs/adr/0002-faithfulness-judged-without-the-reference-answer.md`.

**Citation validity**:
Whether a citation an answer makes — a named video together with a timestamp, offered as
the source of a claim — points at material that was actually retrieved for that
question. It is a property of the reference, not of the claim: an answer can be correct
and cite invalidly, or cite validly and be wrong.
_Avoid_: citation accuracy, grounding, attribution, citation correctness

The avoid-line is about the name of the metric: it is citation validity, never
"attribution". The verb stays available for the internal mechanic — a timestamp is
attributed to the video name it sits beside — which is a step in deriving the metric,
not another name for it.

**Answerable refusal**:
A declined answer to a question whose tier's ground truth says the corpus can answer it.
It is a failure of the same pipeline whose refusals on Tier C are a success, which is
why it is named and counted rather than left to depress a correctness score.
_Avoid_: false refusal, over-refusal, unnecessary refusal, abstention, false negative

**Validity (of a metric)**:
Whether a metric measures the thing it claims to measure. A property of the metric's
definition and of the way it is computed, independent of any run.
_Avoid_: soundness, trustworthiness, correctness of a metric

**Value (of a metric)**:
The number a metric reports for a particular run. Movement in a value is only meaningful
once that metric's validity is settled; a value produced by an invalid metric describes
the ruler, not the pipeline.
_Avoid_: score, result, reading, figure

A value may be obtained without validity, and the two are discussed separately: "the
value fell" and "the metric is invalid" are different claims with different remedies,
and conflating them is how a repaired measurement gets read as a changed pipeline.
