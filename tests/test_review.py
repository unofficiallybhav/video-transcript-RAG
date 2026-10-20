"""Tests for the tier B credibility review's verdict sheet.

The verdicts are a human's, and the sheet is the only place they exist, so parsing it is
the one step where a reviewer's mark can be lost. A mark this module does not recognise
has to stop the stage rather than quietly shrink the denominator.

ADR 0001 keeps a model out of every labelling decision, so nothing here judges a
question; it only reads marks back.
"""
import pytest

import evaluate

SHEET = """# Tier B question review

Untick a question that is NOT answerable from its passage.

- [x] `B-kept.mp4:0` (jaccard 0.1, bm25 rank 1)
    - **Q:** answerable from its passage
- [ ] `B-rejected.mp4:1` (jaccard 0.2, bm25 rank 3)
    - **Q:** not answerable from its passage
- [?] `B-hands-on.mp4:2` (jaccard 0.3, bm25 rank 2)
    - **Q:** answerable, but about a hands-on exercise
"""


def test_each_mark_is_read_back_as_its_own_verdict():
    assert evaluate.spotcheck_verdicts(SHEET) == {
        "B-kept.mp4:0": "accepted",
        "B-rejected.mp4:1": "rejected",
        "B-hands-on.mp4:2": "hands_on",
    }


def test_an_upper_case_tick_is_still_a_tick():
    assert evaluate.spotcheck_verdicts("- [X] `B-q.mp4:0` (jaccard 0.1, bm25 rank 1)") == {
        "B-q.mp4:0": "accepted"}


def test_prose_and_question_lines_are_not_verdicts():
    verdicts = evaluate.spotcheck_verdicts(SHEET)
    assert all(qid.startswith("B-") for qid in verdicts)
    assert len(verdicts) == 3


def test_a_mark_this_module_does_not_know_stops_the_stage():
    """Silently dropping it would shrink the denominator with nothing said."""
    with pytest.raises(ValueError, match="B-mystery.mp4:0"):
        evaluate.spotcheck_verdicts("- [~] `B-mystery.mp4:0` (jaccard 0.1, bm25 rank 1)")


def test_a_hands_on_question_is_sound_so_it_is_not_a_rejection():
    """A `?` says what the question is about, not that it is unanswerable."""
    counts = evaluate.review_counts(evaluate.spotcheck_verdicts(SHEET))
    assert counts["reviewed"] == 3
    assert counts["rejected"] == 1
    assert counts["accepted"] == 2
    assert counts["hands_on"] == 1
    assert counts["rejected_share"] == pytest.approx(1 / 3)


def test_the_hands_on_count_is_a_subset_of_the_accepted_ones():
    counts = evaluate.review_counts(evaluate.spotcheck_verdicts(SHEET))
    assert counts["hands_on"] <= counts["accepted"]


def test_an_empty_checkbox_is_an_untick():
    """Editors strip the space out of `[ ]`; it is the same mark, not an unknown one."""
    assert evaluate.spotcheck_verdicts("- [] `B-q.mp4:0` (jaccard 0.1, bm25 rank 1)") == {
        "B-q.mp4:0": "rejected"}


def test_an_empty_checkbox_counts_as_a_rejection():
    counts = evaluate.review_counts(evaluate.spotcheck_verdicts(
        "- [] `B-a.mp4:0` (jaccard 0.1, bm25 rank 1)\n"
        "- [x] `B-b.mp4:0` (jaccard 0.1, bm25 rank 1)"))
    assert counts == {"reviewed": 2, "accepted": 1, "rejected": 1, "hands_on": 0,
                      "rejected_share": 0.5, "rejected_ids": ["B-a.mp4:0"],
                      "hands_on_ids": []}


# ------------------------------- the review's result reaching the report

def test_the_report_carries_the_review_when_a_sheet_has_been_filled_in():
    summary = evaluate.summarise(
        [{"id": "B-q.mp4:0", "tier": "B", "course": "", "hands_on": False,
          "retrieved": []}],
        tier_b_review=evaluate.review_counts(evaluate.spotcheck_verdicts(SHEET)))
    assert summary["tier_b_review"]["rejected_share"] == pytest.approx(1 / 3)
    assert "33% of a reviewed sample" in evaluate.render_markdown(summary)


def test_the_report_says_the_review_is_outstanding_when_no_sheet_exists():
    summary = evaluate.summarise(
        [{"id": "B-q.mp4:0", "tier": "B", "course": "", "hands_on": False,
          "retrieved": []}])
    assert summary["tier_b_review"] is None
    assert "has not been run" in evaluate.render_markdown(summary)
