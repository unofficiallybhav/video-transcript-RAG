"""Which questions a run scores, and what `run <id>` is allowed to touch.

A full run is resumable: it scores what the checkpoint does not already hold, so an
interrupted run continues rather than paying for the questions it already answered.
`run <id>` is the opposite instruction — score this one again, because something about it
changed — so it must override the checkpoint for that id and nothing else.

Two promises are stated here. The first is the selection itself. The second is that
`run <id>` never writes the report: `golden/report.md` is a document about the whole
golden set, and rendering it from one row would replace every figure in it with that
row's own. One question's fresh row belongs in `golden/results.jsonl`, which is
append-only and keyed last-line-wins, and reaches the report through `rescore`. That
second promise is one condition inside `run`, so it is asserted over `run`'s source
rather than by calling a function written to be called by nothing but this test.

  uv run pytest tests/test_run_one.py
"""
from dataclasses import dataclass

import pytest

import inspect

import evaluate
from evaluate import questions_to_score


def q(qid, tier="A"):
    return {"id": qid, "tier": tier, "question": "?", "answer": "!"}


GOLDEN = [q("A-1"), q("A-2"), q("C-1", tier="C")]


@dataclass(frozen=True)
class Case:
    behaviour: str
    done: tuple
    only: object
    expects: list


CASES = [
    Case(
        behaviour="a run with nothing checkpointed scores the whole golden set, in the "
                  "order the golden set is in",
        done=(),
        only=None,
        expects=["A-1", "A-2", "C-1"],
    ),
    Case(
        behaviour="a resumed run scores only what the checkpoint does not hold, so the "
                  "questions already paid for are not paid for twice",
        done=("A-1",),
        only=None,
        expects=["A-2", "C-1"],
    ),
    Case(
        behaviour="a finished run scores nothing rather than rescoring the set",
        done=("A-1", "A-2", "C-1"),
        only=None,
        expects=[],
    ),
    Case(
        behaviour="`run <id>` scores exactly that question and no other",
        done=(),
        only="A-2",
        expects=["A-2"],
    ),
    Case(
        behaviour="`run <id>` scores it even though the checkpoint already holds it, "
                  "which is the whole point of naming an id",
        done=("A-1", "A-2", "C-1"),
        only="A-2",
        expects=["A-2"],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.behaviour for c in CASES])
def test_selection(case):
    chosen = questions_to_score(GOLDEN, {qid: {} for qid in case.done}, case.only)
    assert [x["id"] for x in chosen] == case.expects


def test_an_id_outside_the_golden_set_stops_the_stage():
    """Naming a question that is not there is a typo, not an instruction to score none."""
    with pytest.raises(SystemExit) as raised:
        questions_to_score(GOLDEN, {}, "A-nope")
    assert "A-nope" in str(raised.value)


def test_the_report_is_written_only_on_the_full_run_path():
    """One row cannot render a report about the whole golden set without replacing it.

    Asserted over `run`'s source because the decision is one condition inside it rather
    than a function of its own, and because what matters is that the single-id path
    cannot reach `write_report` at all. A reader tempted to lift the guard out of the
    `if` will fail here.
    """
    source = inspect.getsource(evaluate.run)
    calls = [line.strip() for line in source.splitlines() if "write_report(" in line]
    assert calls == ["write_report(results)"], calls
    before = source.split("write_report(results)")[0].splitlines()
    guard = [line.strip() for line in before if line.strip()][-1]
    assert guard == "if only is None:", guard
