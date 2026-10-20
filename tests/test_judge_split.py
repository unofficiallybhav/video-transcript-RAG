"""Tests for the split judge call (issue #9, docs/adr/0002).

Faithfulness is judged by a call that never receives the reference answer. The split is
the pre-committed consequence of the calibration experiment in #8, where faithfulness
with the reference withheld moved 0.164 against a threshold of 0.10: with the reference
visible the judge penalised divergence from it, so faithfulness was partly grading
correctness instead of support by the retrieved context.

Nothing here makes a provider call. The judge is a stub passed in at the call site,
which is the same seam the real run uses.
"""
import json
from dataclasses import dataclass

import pytest

import evaluate


# --------------------------------------------------------------- the two prompts

@dataclass
class Case:
    """One claim about a derived prompt. `absent` is checked case-insensitively."""
    name: str
    prompt: str
    present: tuple = ()
    absent: tuple = ()


PROMPT_CASES = [
    Case("the faithfulness call is shown no reference answer",
         prompt="faithfulness", absent=("REFERENCE", "{reference}")),
    Case("the faithfulness call asks for faithfulness and nothing else",
         prompt="faithfulness", present=("faithfulness",),
         absent=("relevancy", "correctness")),
    Case("the reference-bearing call keeps the reference and both its metrics",
         prompt="combined", present=("{reference}", "relevancy", "correctness")),
    Case("the reference-bearing call still asks for the refusal verdict",
         prompt="combined", present=("refused",)),
]
PROMPTS = {"faithfulness": evaluate.faithfulness_prompt, "combined": lambda: evaluate.JUDGE_PROMPT}


@pytest.mark.parametrize("case", PROMPT_CASES, ids=lambda c: c.name)
def test_derived_prompt(case):
    prompt = PROMPTS[case.prompt]()
    for text in case.present:
        assert text in prompt
    for text in case.absent:
        assert text.lower() not in prompt.lower()


def test_the_faithfulness_criterion_ships_in_the_wording_it_was_calibrated_in():
    criterion = evaluate.JUDGE_PROMPT.split("- faithfulness:")[1].split("- relevancy:")[0]
    assert criterion in evaluate.faithfulness_prompt()


def test_the_faithfulness_prompt_formats_without_a_reference():
    # a leftover {reference} would raise KeyError here rather than reach the judge
    filled = evaluate.faithfulness_prompt().format(question="q", context="c", answer="a")
    assert "reference" not in filled.lower()


def test_rewording_the_judge_prompt_fails_loudly_instead_of_leaking_the_reference(
        monkeypatch):
    """The guard is the whole reason the prompt is derived rather than written out."""
    monkeypatch.setattr(evaluate, "JUDGE_PROMPT", "Score it. REFERENCE ANSWER: {reference}")
    with pytest.raises(RuntimeError, match="no longer contains"):
        evaluate.faithfulness_prompt()


# ------------------------------------------------------------ the two-call judge

def recording_judge(by_metric):
    """A judge that answers each prompt with the scores keyed to the call it came from.

    It records every prompt it was sent, so a test can assert what the faithfulness call
    was and was not shown.
    """
    sent = []

    def complete_fn(model, prompt=None, messages=None):
        sent.append(prompt)
        return "scores: " + json.dumps(
            by_metric["combined" if "REFERENCE" in prompt else "faithfulness"])

    complete_fn.sent = sent
    return complete_fn


def split_judge(faithfulness=0.42):
    return recording_judge({
        "faithfulness": {"faithfulness": faithfulness, "note": "support only"},
        # the combined call still returns a faithfulness score; this is the contaminated
        # one, and judge_split must not let it reach the row
        "combined": {"faithfulness": 0.99, "relevancy": 0.90, "correctness": 0.75,
                     "refused": False, "note": "agrees with the key"},
    })


QUESTION = {"question": "what is a confusion matrix?", "answer": "THE-REFERENCE"}


def test_faithfulness_comes_from_a_call_that_never_sees_the_reference():
    fake = split_judge()
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["faithfulness"] == 0.42
    without_reference = [p for p in fake.sent if "REFERENCE" not in p]
    assert len(without_reference) == 1
    assert "THE-REFERENCE" not in without_reference[0]


def test_the_combined_calls_own_faithfulness_score_is_discarded():
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=split_judge())
    assert scores["faithfulness"] != 0.99


def test_correctness_and_relevancy_come_from_the_call_that_sees_the_reference():
    fake = split_judge()
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["correctness"] == 0.75
    assert scores["relevancy"] == 0.90
    assert "THE-REFERENCE" in [p for p in fake.sent if "REFERENCE" in p][0]


def test_correctness_and_relevancy_are_judged_by_the_unchanged_prompt():
    """#9 requires them unaffected in how they are judged, so their call is untouched."""
    fake = split_judge()
    evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    combined = [p for p in fake.sent if "REFERENCE" in p][0]
    assert combined == evaluate.JUDGE_PROMPT.format(
        question=QUESTION["question"], reference=QUESTION["answer"],
        context="ctx", answer="ans")


def test_the_refusal_verdict_and_note_come_from_the_reference_bearing_call():
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=split_judge())
    assert scores["refused"] is False
    assert scores["note"] == "agrees with the key"
    # the faithfulness call's own reasoning is kept, under its own key, because
    # faithfulness is now the only thing that call explains
    assert scores["faithfulness_note"] == "support only"


def test_scoring_one_answer_costs_exactly_two_judge_calls():
    fake = split_judge()
    evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert len(fake.sent) == 2


def test_a_split_judged_row_says_so():
    """Faithfulness from the combined call is not comparable, so the row records which
    call produced it: a part-way re-judge must not average the two together."""
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=split_judge())
    assert scores["faithfulness_judge"] == evaluate.SPLIT_JUDGE


def test_an_unparseable_faithfulness_reply_leaves_faithfulness_absent_and_unmarked():
    """An absent score must not be recorded as measured, or `rejudge` skips it forever."""
    def fake(model, prompt=None, messages=None):
        if "REFERENCE" in prompt:
            return json.dumps({"relevancy": 0.9, "correctness": 0.7, "refused": False})
        return "the judge rambled instead of replying with json"

    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["faithfulness"] is None
    assert "faithfulness_judge" not in scores
    assert scores["correctness"] == 0.7


# ------------------------------------- what the report does with the two kinds

def row(qid, faithfulness, judged_by_split):
    out = {"id": qid, "tier": "A", "faithfulness": faithfulness, "correctness": 0.7}
    if judged_by_split:
        out["faithfulness_judge"] = evaluate.SPLIT_JUDGE
    return out


def test_the_report_drops_faithfulness_the_combined_call_produced():
    rows, superseded = evaluate.drop_superseded_faithfulness(
        [row("a", 0.8, False), row("b", 0.5, True)])
    assert superseded == 1
    assert [r["faithfulness"] for r in rows] == [None, 0.5]


def test_a_row_the_split_call_scored_is_left_alone():
    rows, superseded = evaluate.drop_superseded_faithfulness([row("b", 0.5, True)])
    assert superseded == 0
    assert rows[0]["faithfulness"] == 0.5


def test_dropping_leaves_every_other_judge_field_untouched():
    rows, _ = evaluate.drop_superseded_faithfulness([row("a", 0.8, False)])
    assert rows[0]["correctness"] == 0.7
