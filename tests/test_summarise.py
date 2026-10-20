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

from evaluate import render_markdown, summarise


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


def test_rendered_report_carries_the_citation_provenance_note():
    """A reader comparing with an older copy learns why a citation value moved."""
    rendered = render_markdown(summarise([row("A", "a1", correctness=1.0)]))
    note = next((line for line in rendered.splitlines()
                 if "corrected 2026-10-19" in line.lower()), None)
    assert note is not None, "no line names the date the citation defect was corrected"
    assert "timestamp" in note.lower()
    assert "citation" in note.lower()
