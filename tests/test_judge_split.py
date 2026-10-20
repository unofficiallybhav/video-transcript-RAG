"""Tests for the split judge call (issue #9).

Faithfulness is judged by a call that never receives the reference answer. The split is
the pre-committed consequence of the calibration experiment in #8, where faithfulness
with the reference withheld moved 0.164 against a threshold of 0.10: with the reference
visible the judge penalised divergence from it, so faithfulness was partly grading
correctness instead of support by the retrieved context.

Nothing here makes a provider call. The judge is a stub passed in at the call site,
which is the same seam the real run uses.
"""
import json

import evaluate


# --------------------------------------------------------------- the two prompts

def test_faithfulness_prompt_never_mentions_the_reference_answer():
    prompt = evaluate.faithfulness_prompt()
    assert "REFERENCE" not in prompt
    assert "{reference}" not in prompt


def test_faithfulness_prompt_asks_for_faithfulness_alone():
    prompt = evaluate.faithfulness_prompt()
    assert "faithfulness" in prompt
    assert "relevancy" not in prompt
    assert "correctness" not in prompt


def test_faithfulness_prompt_keeps_the_criterion_word_for_word():
    """The calibrated criterion is the one that ships, down to its wording."""
    criterion = evaluate.JUDGE_PROMPT.split("- faithfulness:")[1].split("- relevancy:")[0]
    assert criterion in evaluate.faithfulness_prompt()


def test_faithfulness_prompt_formats_without_a_reference():
    # a leftover {reference} would raise KeyError here rather than reach the judge
    filled = evaluate.faithfulness_prompt().format(question="q", context="c", answer="a")
    assert "reference" not in filled.lower()


def test_correctness_prompt_keeps_the_reference_and_drops_faithfulness():
    prompt = evaluate.correctness_prompt()
    assert "{reference}" in prompt
    assert "correctness" in prompt
    assert "relevancy" in prompt
    assert "faithfulness" not in prompt


def test_correctness_prompt_keeps_both_criteria_word_for_word():
    """Correctness and relevancy are unaffected in how they are judged."""
    for metric, nxt in (("- relevancy:", "- correctness:"), ("- correctness:", "\n\n")):
        criterion = evaluate.JUDGE_PROMPT.split(metric)[1].split(nxt)[0]
        assert criterion in evaluate.correctness_prompt()


def test_correctness_prompt_still_asks_for_the_refusal_verdict():
    assert "refused" in evaluate.correctness_prompt()


# ------------------------------------------------------------ the two-call judge

def recording_judge(by_metric):
    """A judge that answers each prompt with the scores keyed to the metric it asks for.

    It records every prompt it was sent, so a test can assert what the faithfulness call
    was and was not shown.
    """
    sent = []

    def complete_fn(model, prompt=None, messages=None):
        sent.append(prompt)
        which = "faithfulness" if "REFERENCE" not in prompt else "correctness"
        return "scores: " + json.dumps(by_metric[which])

    complete_fn.sent = sent
    return complete_fn


def split_judge():
    return recording_judge({
        "faithfulness": {"faithfulness": 0.42, "note": "support only"},
        "correctness": {"relevancy": 0.90, "correctness": 0.75, "refused": False,
                        "note": "agrees with the key"},
    })


QUESTION = {"question": "what is a confusion matrix?", "answer": "THE-REFERENCE"}


def test_faithfulness_comes_from_a_call_that_never_sees_the_reference():
    fake = split_judge()
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["faithfulness"] == 0.42
    without_reference = [p for p in fake.sent if "REFERENCE" not in p]
    assert len(without_reference) == 1
    assert "THE-REFERENCE" not in without_reference[0]


def test_correctness_and_relevancy_come_from_the_call_that_sees_the_reference():
    fake = split_judge()
    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["correctness"] == 0.75
    assert scores["relevancy"] == 0.90
    assert "THE-REFERENCE" in [p for p in fake.sent if "REFERENCE" in p][0]


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


def test_an_unparseable_faithfulness_reply_leaves_faithfulness_absent():
    def fake(model, prompt=None, messages=None):
        if "REFERENCE" in prompt:
            return json.dumps({"relevancy": 0.9, "correctness": 0.7, "refused": False})
        return "the judge rambled instead of replying with json"

    scores = evaluate.judge_split(QUESTION, "ans", "ctx", complete_fn=fake)
    assert scores["faithfulness"] is None
    assert scores["correctness"] == 0.7
