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

from evaluate import precision_recall


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
