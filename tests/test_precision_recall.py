"""Behaviour of the retrieval arithmetic, stated on hand-built id lists.

`precision_recall` is the whole of what this project means by retrieval quality, and it
is the one piece of that path that survived the retirement of the generated tier. It is
live code reached by both `run()` and `rescore_rows()` through a branch guarded on a row
carrying `gold_chunks`; no row carries them at present, so the branch does not execute
and nothing else here exercises it. These tests cover it directly, so that hand-labelling
gold chunks onto tier A is a question of supplying labels rather than of trusting
untested arithmetic.

Vocabulary follows `GLOSSARY.md`: a gold chunk is one a question is scored as having
needed, established without a model's opinion.

  uv run pytest tests/test_precision_recall.py
"""
from dataclasses import dataclass

import pytest

from evaluate import apply_labels, precision_recall, retrieval_metrics


@dataclass(frozen=True)
class Case:
    """One retrieval outcome and the pair of values it should produce."""

    behaviour: str
    retrieved: list
    gold: list
    k: int
    precision: float
    recall: float


CASES = [
    Case(
        behaviour="a single gold chunk ranked first scores full recall and 1/k precision",
        retrieved=["a.mp4:0", "b.mp4:1", "c.mp4:2"],
        gold=["a.mp4:0"],
        k=3,
        precision=1 / 3,
        recall=1.0,
    ),
    Case(
        behaviour="precision counts only the first k retrieved, so a gold chunk below "
                  "the cut does not score",
        retrieved=["b.mp4:1", "c.mp4:2", "a.mp4:0"],
        gold=["a.mp4:0"],
        k=2,
        precision=0.0,
        recall=0.0,
    ),
    Case(
        behaviour="recall is the share of gold chunks found, not the share of slots "
                  "filled",
        retrieved=["a.mp4:0", "b.mp4:1", "x.mp4:9"],
        gold=["a.mp4:0", "b.mp4:1", "c.mp4:2", "d.mp4:3"],
        k=3,
        precision=2 / 3,
        recall=0.5,
    ),
    Case(
        behaviour="precision is divided by k and not by how many were retrieved, so a "
                  "short retrieval cannot inflate it",
        retrieved=["a.mp4:0"],
        gold=["a.mp4:0"],
        k=5,
        precision=0.2,
        recall=1.0,
    ),
    Case(
        behaviour="a question with no gold chunk scores zero recall rather than "
                  "dividing by zero",
        retrieved=["a.mp4:0"],
        gold=[],
        k=5,
        precision=0.0,
        recall=0.0,
    ),
    Case(
        behaviour="a gold chunk retrieved twice is counted once, so a duplicate cannot "
                  "push recall above 1.0 nor precision above its own ceiling",
        retrieved=["a.mp4:0", "a.mp4:0"],
        gold=["a.mp4:0"],
        k=2,
        # one gold chunk at k=2 caps precision at len(gold)/k = 0.5, the same ceiling
        # the report prints as `precision@k_max`; the repeated slot delivers nothing new
        precision=0.5,
        recall=1.0,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.behaviour for c in CASES])
def test_precision_and_recall(case):
    precision, recall = precision_recall(case.retrieved, case.gold, case.k)
    assert precision == pytest.approx(case.precision)
    assert recall == pytest.approx(case.recall)


# ------------------------------------------------- the metrics one row contributes


@dataclass(frozen=True)
class MetricCase:
    """One row's retrieved ids and gold labels, and the metric entries they produce."""

    behaviour: str
    retrieved: list
    gold: list
    expects: dict


METRIC_CASES = [
    MetricCase(
        behaviour="a gold chunk ranked first scores full precision at k=1 and full "
                  "recall at every k",
        retrieved=["a.mp4:0", "b.mp4:1", "c.mp4:2"],
        gold=["a.mp4:0"],
        expects={"precision@1": 1.0, "recall@1": 1.0, "precision@5": 0.2, "recall@5": 1.0},
    ),
    MetricCase(
        behaviour="precision@k_max is the ceiling k leaves, so one gold chunk at k=5 "
                  "cannot be read as a retrieval failure",
        retrieved=["a.mp4:0"],
        gold=["a.mp4:0"],
        expects={"precision@5": 0.2, "precision@5_max": 0.2},
    ),
    MetricCase(
        behaviour="the ceiling is capped by k as well as by the gold count, so two gold "
                  "chunks at k=1 leave a ceiling of 1.0",
        retrieved=["a.mp4:0", "b.mp4:1"],
        gold=["a.mp4:0", "b.mp4:1"],
        expects={"precision@1": 1.0, "precision@1_max": 1.0, "recall@1": 0.5},
    ),
]


@pytest.mark.parametrize("case", METRIC_CASES, ids=[c.behaviour for c in METRIC_CASES])
def test_retrieval_metrics(case):
    metrics = retrieval_metrics(case.retrieved, case.gold)
    assert {k: metrics[k] for k in case.expects} == pytest.approx(case.expects)


def test_a_row_without_gold_chunks_carries_no_retrieval_metrics():
    """An unlabelled question leaves every retrieval denominator rather than scoring 0.

    `record_with_denominator` drops a metric a row carries no value for, so the way a
    question stays out of a retrieval mean is by carrying no entry at all. Returning
    zeros here would put the 83 - labelled questions into the denominator at the worst
    score the metric has, which is the opposite of what partial labelling means.
    """
    assert retrieval_metrics(["a.mp4:0"], []) == {}


# ------------------------------------- joining the hand-written labels onto a stored run


def stored_row(qid, retrieved):
    """A checkpoint row, carrying only what the retrieval join reads and passes through."""
    return {"id": qid, "tier": "A", "retrieved": retrieved, "correctness": 1.0}


def test_a_labelled_question_gains_its_gold_chunks_and_metrics():
    rows = apply_labels([stored_row("a1", ["a.mp4:0", "b.mp4:1"])],
                        {"a1": {"gold_chunks": ["a.mp4:0"]}})
    assert rows[0]["gold_chunks"] == ["a.mp4:0"]
    assert rows[0]["recall@1"] == 1.0
    # nothing the run measured is disturbed by the join
    assert rows[0]["correctness"] == 1.0


def test_a_no_gold_question_is_recorded_as_such_and_scores_no_recall():
    """The verdict travels, but it is not a gold chunk list, so no recall is derived.

    An index gap and a retrieval miss are different findings. Writing this verdict as an
    empty `gold_chunks` would make it score recall 0.0 and be read as the second.
    """
    rows = apply_labels([stored_row("a1", ["a.mp4:0"])],
                        {"a1": {"no_gold_in_corpus": True}})
    assert rows[0]["no_gold_in_corpus"] is True
    assert "gold_chunks" not in rows[0]
    assert "recall@1" not in rows[0]


def test_an_unlabelled_question_is_left_exactly_as_the_run_recorded_it():
    row = stored_row("a1", ["a.mp4:0"])
    assert apply_labels([row], {})[0] == row


def test_a_label_withdrawn_since_the_run_is_cleared_rather_than_passed_through():
    """A stored row carries the labels it was scored with, and they can go stale.

    `run()` writes `gold_chunks` and the retrieval metrics into the checkpoint. A
    question relabelled afterwards — a tick removed, or the verdict changed to no gold
    chunk in the corpus — would otherwise keep scoring recall against chunks it no longer
    claims, and would keep being counted among the labelled. `label_rows` clears both
    fields for this reason; the join has to do the same on the rows it reads.
    """
    stale = {**stored_row("a1", ["a.mp4:0"]), "gold_chunks": ["a.mp4:0"],
             "recall@5": 1.0, "precision@5": 0.2, "precision@5_max": 0.2}

    withdrawn = apply_labels([stale], {})[0]
    assert "gold_chunks" not in withdrawn
    assert "recall@5" not in withdrawn

    no_gold = apply_labels([stale], {"a1": {"no_gold_in_corpus": True}})[0]
    assert no_gold["no_gold_in_corpus"] is True
    assert "gold_chunks" not in no_gold
    assert "recall@5" not in no_gold


def test_a_relabelled_question_scores_against_its_new_gold_chunks_only():
    stale = {**stored_row("a1", ["a.mp4:0", "b.mp4:1"]), "gold_chunks": ["a.mp4:0"],
             "recall@1": 1.0}
    relabelled = apply_labels([stale], {"a1": {"gold_chunks": ["b.mp4:1"]}})[0]
    assert relabelled["gold_chunks"] == ["b.mp4:1"]
    assert relabelled["recall@1"] == 0.0


def test_joining_labels_twice_changes_nothing_further():
    labels = {"a1": {"gold_chunks": ["a.mp4:0"]}}
    once = apply_labels([stored_row("a1", ["a.mp4:0"])], labels)
    assert apply_labels(once, labels) == once
