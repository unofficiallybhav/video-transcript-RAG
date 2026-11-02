"""Behaviour of the rephrasing table, which restates a quiz question in the tier.

A tier A question carries the course authors' own wording and their own answer key, and
correctness is anchored to that key — so replacing either is a deviation from the anchor,
not a tidy-up. The quiz files under `evals/` stay untouched; a rephrasing is declared here
instead, by id, with the original wording kept on the row. That way the deviation is
visible in `golden/tier_a.json` and in any report drawn from it, rather than being a
silent edit to the authors' file that nothing downstream could notice.

The one promise these tests state: a declared rephrasing reaches the tier, every other
question is left exactly as parsed, and a rephrasing that no longer matches a question
stops the stage instead of quietly doing nothing. The last one matters most. A rephrasing
is pinned to an id, and the ids move when a quiz is excluded or re-parsed — a rephrasing
that silently stopped applying would leave the authors' original wording in the tier while
the label sheet still showed the restated one, which is exactly the kind of rot this repo
has twice been bitten by.

  uv run pytest tests/test_rephrasings.py
"""
from dataclasses import dataclass, field

import pytest

from build_golden import apply_rephrasings


def row(qid="A-q1", question="Which of the statements below is true?",
        answer="Dijkstra's algorithm is computationally inefficient."):
    """One row as `parse_quiz` emits it, with only the fields a rephrasing touches."""
    return {"id": qid, "tier": "A", "question": question, "answer": answer}


@dataclass(frozen=True)
class Case:
    """One rephrasing table applied to one set of parsed rows."""

    behaviour: str
    rows: list
    table: dict
    expects: dict = field(default_factory=dict)


CASES = [
    Case(
        behaviour="a declared rephrasing replaces the question and the answer key, so "
                  "the tier carries the wording the labeller actually labelled",
        rows=[row()],
        table={"A-q1": {"question": "Is Dijkstra's algorithm computationally inefficient?",
                        "answer": "Yes. Dijkstra's algorithm has high computational complexity.",
                        "reason": "the original named options that parsing had dropped"}},
        expects={"question": "Is Dijkstra's algorithm computationally inefficient?",
                 "answer": "Yes. Dijkstra's algorithm has high computational complexity."},
    ),
    Case(
        behaviour="the authors' wording and the reason ride along on the row, so the "
                  "deviation from their key is readable wherever the row goes",
        rows=[row()],
        table={"A-q1": {"question": "Is Dijkstra's algorithm computationally inefficient?",
                        "answer": "Yes. Dijkstra's algorithm has high computational complexity.",
                        "reason": "the original named options that parsing had dropped"}},
        expects={"original": {
            "question": "Which of the statements below is true?",
            "answer": "Dijkstra's algorithm is computationally inefficient.",
            "reason": "the original named options that parsing had dropped"}},
    ),
    Case(
        behaviour="a question absent from the table is passed through untouched, and "
                  "carries no restatement marker to be mistaken for one",
        rows=[row(qid="A-q2"), row(qid="A-q1")],
        table={"A-q1": {"question": "q", "answer": "a", "reason": "r"}},
        expects={"question": "Which of the statements below is true?",
                 "answer": "Dijkstra's algorithm is computationally inefficient.",
                 "original": None},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.behaviour for c in CASES])
def test_rephrasing(case):
    rows = apply_rephrasings([dict(r) for r in case.rows], case.table)
    assert len(rows) == len(case.rows)
    got = rows[0]
    for key, want in case.expects.items():
        if want is None:
            assert key not in got or got[key] is None
        else:
            assert got[key] == want


def test_a_rephrasing_matching_no_question_stops_the_stage():
    """Its id left the tier, so the sheet and the tier would disagree in silence."""
    with pytest.raises(SystemExit) as raised:
        apply_rephrasings([row(qid="A-q1")],
                          {"A-gone": {"question": "q", "answer": "a", "reason": "r"}})
    assert "A-gone" in str(raised.value)


def test_nothing_is_rephrased_when_the_table_is_empty():
    rows = apply_rephrasings([row(), row(qid="A-q2")], {})
    assert [r["question"] for r in rows] == ["Which of the statements below is true?"] * 2
    assert not any("original" in r for r in rows)


def test_the_real_table_names_only_ids_the_real_parse_keeps():
    """The guard above, against the table this repo actually ships."""
    from build_golden import REPHRASED_QUESTIONS, is_excluded

    assert not [qid for qid in REPHRASED_QUESTIONS if is_excluded(qid)]
