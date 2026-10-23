"""Behaviour of the report path, stated on hand-built rows.

`summarise()` turns scored rows into the dict the two report artifacts are rendered
from, so this is where the reader-facing promises live: that every metric value arrives
with the number of questions it was taken over, that a question carrying no value for a
metric is left out of that metric rather than counted as a zero, and that an answerable
refusal is counted on the tiers whose ground truth says the corpus can answer.

The rows here are hand-built and minimal: each case carries only the keys the behaviour
it states actually names. No citation value is asserted as a literal, because the
citation check's own values move as its validity is repaired; what is asserted is which
rows a value was taken over. Nothing reaches the vector store, a provider or the
filesystem.

Vocabulary follows `GLOSSARY.md`: a tier is a partition by kind of ground truth, an
answerable refusal is a declined answer to a question the corpus can answer, and a
metric's value is distinct from its validity.

  uv run pytest
"""
from dataclasses import dataclass, field

import pytest

from evaluate import (MIN_BREAKOUT, drop_questions_not_in_golden, render_markdown,
                      summarise)


def row(tier, qid="q", **metrics):
    """One scored row, carrying only the keys a behaviour statement names."""
    return {"id": qid, "tier": tier, "course": "machine learning",
            "hands_on": False, **metrics}


@dataclass(frozen=True)
class Case:
    """One set of rows, one tier block, and the summary entries it should produce."""

    behaviour: str
    rows: list
    tier_key: str
    expects: dict = field(default_factory=dict)


CASES = [
    Case(
        behaviour="a metric's count is the number of rows that carry a value for it",
        rows=[
            row("A", "a1", correctness=1.0, citation_time_valid=1.0),
            row("A", "a2", correctness=0.0, citation_time_valid=None),
            row("A", "a3", correctness=0.0),
        ],
        tier_key="tier_a",
        expects={"questions": 3, "correctness_n": 3, "citation_time_valid_n": 1},
    ),
    Case(
        behaviour="a row carrying no value for a metric is excluded from its mean, "
                  "not counted as zero",
        rows=[
            row("A", "a1", citation_time_valid=1.0),
            row("A", "a2", citation_time_valid=None),
            row("A", "a3"),
        ],
        tier_key="tier_a",
        expects={"questions": 3, "citation_time_valid": 1.0,
                 "citation_time_valid_n": 1},
    ),
    Case(
        behaviour="a metric no row carries has a count of zero and no value",
        rows=[row("A", "a1", correctness=0.5)],
        tier_key="tier_a",
        expects={"citation_present": None, "citation_present_n": 0},
    ),
    Case(
        behaviour="tier A reports an answerable refusal rate over its own questions",
        rows=[
            row("A", "a1", refused=True),
            row("A", "a2", refused=False),
            row("A", "a3", refused=False),
            row("A", "a4", refused=False),
        ],
        tier_key="tier_a",
        expects={"answerable_refusal_rate": 0.25, "answerable_refusal_rate_n": 4},
    ),
    Case(
        behaviour="a row the judge left no refusal verdict on is excluded from the "
                  "answerable refusal rate",
        rows=[
            row("A", "a1", refused=True),
            row("A", "a2", refused=False),
            row("A", "a3", refused=None),
        ],
        tier_key="tier_a",
        expects={"answerable_refusal_rate": 0.5, "answerable_refusal_rate_n": 2},
    ),
    Case(
        behaviour="tier C reports a refusal rate, which is the correct behaviour there "
                  "and is never called an answerable refusal",
        rows=[row("C", "c1", refused=True), row("C", "c2", refused=False)],
        tier_key="tier_c",
        expects={"questions": 2, "refusal_rate": 0.5,
                 "answerable_refusal_rate": None},
    ),
    Case(
        behaviour="retrieval is reported over the labelled questions only, with its "
                  "own denominator, so partial labelling is visible rather than "
                  "inflating or deflating the figure",
        rows=[
            row("A", "a1", gold_chunks=["a.mp4:0"], **{"recall@5": 1.0}),
            row("A", "a2"),
            row("A", "a3", no_gold_in_corpus=True),
        ],
        tier_key="tier_a_retrieval",
        expects={"questions": 1, "recall@5": 1.0, "recall@5_n": 1},
    ),
    Case(
        behaviour="a question labelled as having no gold chunk leaves recall rather "
                  "than scoring zero in it, so an index gap and a retrieval miss stay "
                  "distinguishable",
        rows=[
            row("A", "a1", gold_chunks=["a.mp4:0"], **{"recall@5": 1.0}),
            row("A", "a2", no_gold_in_corpus=True),
        ],
        tier_key="tier_a_retrieval",
        expects={"recall@5": 1.0, "recall@5_n": 1},
    ),
    Case(
        behaviour="generation is reported beside retrieval over that same labelled "
                  "set, so the two can be read against each other",
        rows=[
            row("A", "a1", correctness=1.0, gold_chunks=["a.mp4:0"], **{"recall@5": 0.0}),
            row("A", "a2", correctness=0.0),
        ],
        tier_key="tier_a_retrieval",
        expects={"questions": 1, "correctness": 1.0, "correctness_n": 1,
                 "recall@5": 0.0},
    ),
    Case(
        behaviour="the three label states are counted separately, so unfinished work "
                  "is never read as an absent answer",
        rows=[
            row("A", "a1", gold_chunks=["a.mp4:0"]),
            row("A", "a2"),
            row("A", "a3", no_gold_in_corpus=True),
            row("A", "a4"),
        ],
        tier_key="tier_a_labels",
        expects={"questions": 4, "labelled": 1, "no_gold_in_corpus": 1,
                 "unlabelled": 2},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.behaviour for c in CASES])
def test_summary_block(case):
    block = summarise(case.rows)[case.tier_key]
    assert {k: block.get(k) for k in case.expects} == case.expects


def test_rendered_report_shows_each_metric_count():
    """A reader of the markdown can see how many questions a value covers."""
    rows = [
        row("A", "a1", correctness=1.0, citation_time_valid=1.0),
        row("A", "a2", correctness=0.0, citation_time_valid=None),
    ]
    rendered = render_markdown(summarise(rows))
    assert "| citation_time_valid | 1.0 | 1 |" in rendered
    assert "| correctness | 0.5 | 2 |" in rendered


def test_rendered_report_presents_both_refusal_rates_together():
    """Neither refusal rate may be read as success on its own."""
    rows = [row("A", "a1", refused=True), row("C", "c1", refused=True)]
    rendered = render_markdown(summarise(rows))
    section = rendered.split("## Refusal")[1]
    assert "answerable refusal rate" in section
    assert "tier A" in section and "tier C" in section


def test_rendered_report_states_that_retrieval_is_unmeasured():
    """The retired tier leaves a gap, and a gap has to be said rather than shown.

    Tier B carried every retrieval metric, so retiring it removes them all. An absent
    table reads as an oversight or as nothing to report; a named section saying
    retrieval is unmeasured, and why, is the only form of this that a reader cannot
    mistake for a passing grade.
    """
    rendered = render_markdown(summarise([row("A", "a1", correctness=1.0)]))
    assert "## Retrieval — unmeasured" in rendered
    section = rendered.split("## Retrieval — unmeasured")[1]
    assert "recall" in section.lower()
    assert "0003" in section, "the section does not point at the ADR that retired it"


def test_a_question_removed_from_the_golden_set_leaves_the_report():
    """Removing a question from the dataset has to remove it from the figures too.

    `golden/results.jsonl` keeps every question ever scored and is never rewritten, so a
    question dropped from the golden set still has a stored row. Reporting it would
    describe a dataset that no longer exists — and since the questions removed here were
    removed for scoring 0.0 on a corpus that cannot answer them, leaving them in would
    hold every tier A mean down on their account.
    """
    rows = [row("A", "a1", correctness=1.0), row("A", "gone", correctness=0.0)]
    kept, removed = drop_questions_not_in_golden(rows, {"a1": {"id": "a1"}})
    assert [r["id"] for r in kept] == ["a1"]
    assert [r["id"] for r in removed] == ["gone"]


def test_an_absent_golden_set_removes_nothing():
    """A checkout with no golden set still renders a report, rather than an empty one."""
    rows = [row("A", "a1", correctness=1.0)]
    assert drop_questions_not_in_golden(rows, {}) == (rows, [])


def test_the_report_states_what_the_removed_questions_had_scored():
    """The provenance line reports the removed set's own figures, it does not assert them.

    Saying "every one scored 0.0" was wrong the first time it was written: 5 of 17 had
    scored above zero. A sentence that states what moved the means has to be derived from
    the rows it describes, or it is one more number nobody can check.
    """
    removed = [row("A", "g1", correctness=0.0), row("A", "g2", correctness=1.0),
               row("A", "g3", correctness=0.0, refused=True)]
    rendered = render_markdown(summarise([row("A", "a1", correctness=1.0)],
                                         removed_from_golden=removed))
    line = next(l for l in rendered.splitlines() if "removed from the golden set" in l)
    assert "3 questions" in line
    assert "0.333" in line, "the removed set's own correctness mean is not stated"
    assert "2 of them scoring 0.0" in line


def test_rendered_report_measures_retrieval_once_a_question_is_labelled():
    """One hand-labelled question replaces the gap with a figure and its denominator.

    The section has to say how much of the tier is labelled, because 1 of 83 and 83 of
    83 are the same table with very different weight, and the difference is only in the
    count.
    """
    rows = [
        row("A", "a1", gold_chunks=["a.mp4:0"], **{"recall@5": 1.0, "precision@5": 0.2}),
        row("A", "a2"),
        row("A", "a3", no_gold_in_corpus=True),
    ]
    rendered = render_markdown(summarise(rows))
    assert "## Retrieval — unmeasured" not in rendered
    section = rendered.split("## Retrieval")[1]
    assert "| recall@5 | 1.0 | 1 |" in section
    assert "1 of 3" in section, "the section does not say how much of the tier is labelled"
    assert "no gold chunk" in section.lower()


def test_a_hands_on_breakout_too_small_to_support_a_claim_is_not_tabled():
    """One question cannot answer a hypothesis, and a table invites reading it as if it can.

    The hands-on category was all but emptied when the KNIME and Spark exercise quizzes
    left tier A, because their answers were figures from an exercise's dataset rather
    than anything the lectures state. What remains is too small to break out, and saying
    so is the only honest form of it: a 12-row table over a single question reads as a
    measurement.
    """
    rows = [
        dict(row("A", "a1", gold_chunks=["a.mp4:0"], **{"recall@5": 1.0}), hands_on=True),
        row("A", "a2", gold_chunks=["b.mp4:1"], **{"recall@5": 0.5}),
    ]
    section = render_markdown(summarise(rows)).split("## Retrieval")[1]
    assert "| recall@5 | 1.0 | 1 |" not in section, "a one-question table was rendered"
    assert "1 of the 2" in section
    assert "supports no claim" in section


def test_rendered_report_breaks_retrieval_out_by_hands_on():
    """Whether hands-on weakness is a retrieval failure has to be answerable here.

    Enough hands-on questions to clear the breakout threshold, so this exercises the
    table rather than the too-small-to-report path beside it.
    """
    rows = [dict(row("A", f"h{i}", gold_chunks=["a.mp4:0"], **{"recall@5": 0.0}),
                 hands_on=True) for i in range(MIN_BREAKOUT)]
    rows.append(row("A", "a1", gold_chunks=["b.mp4:1"], **{"recall@5": 1.0}))
    section = render_markdown(summarise(rows)).split("## Retrieval")[1]
    assert "hands-on" in section.lower()
    assert f"| recall@5 | 0.0 | {MIN_BREAKOUT} |" in section


def test_rendered_report_carries_the_citation_provenance_note():
    """A reader comparing with an older copy learns why a citation value moved."""
    rendered = render_markdown(summarise([row("A", "a1", correctness=1.0)]))
    note = next((line for line in rendered.splitlines()
                 if "corrected 2026-10-19" in line.lower()), None)
    assert note is not None, "no line names the date the citation defect was corrected"
    assert "timestamp" in note.lower()
    assert "citation" in note.lower()
