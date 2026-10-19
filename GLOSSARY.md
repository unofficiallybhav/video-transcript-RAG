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
The tier whose questions were each generated from a single chunk, so that chunk is the
retrieval target by construction. Its ground truth is a chunk label, not a reference
answer a human stands behind.
_Avoid_: the synthetic tier, the generated tier

**Tier C**:
The tier whose questions are about topics verified absent from the corpus. Its ground
truth is that no correct answer exists.
_Avoid_: the negative tier, the adversarial tier, the unanswerable tier

**Gold chunk**:
The chunk a question is scored as having needed: the one it was constructed from, plus
any chunk that duplicates the material that makes it answerable. A question has gold
chunks only when they were established without a model's opinion.
_Avoid_: relevant chunk, correct chunk, ground-truth chunk, target chunk, expected chunk

### Measurement

**Judge-free metric**:
A metric whose value follows from the recorded run and the corpus alone, with no model
asked for an opinion at any point in deriving it. Such a metric can be recomputed from
stored artifacts, and recomputing it yields the same number every time.
_Avoid_: objective metric, deterministic metric, automatic metric, hard metric

**Citation validity**:
Whether a citation an answer makes — a named video together with a timestamp, offered as
the source of a claim — points at material that was actually retrieved for that
question. It is a property of the reference, not of the claim: an answer can be correct
and cite invalidly, or cite validly and be wrong.
_Avoid_: citation accuracy, grounding, attribution, citation correctness

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
