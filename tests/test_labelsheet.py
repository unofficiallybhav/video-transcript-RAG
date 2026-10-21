"""Behaviour of the hand-labelling sheet, stated on hand-written sheets.

The sheet is the only copy of the human's retrieval labels, so these tests are about one
promise: a mark a person wrote is either read as what they meant or raises. Nothing is
skipped, because a skipped mark leaves its question out of the denominator with nothing
said about it — and `golden/labelsheet.md` cannot be regenerated from anything, unlike
every other artifact in this repo.

Tests cover the parser and the writer's preservation of marks, not the sheet's
appearance. Nothing here touches the vector store, a provider, the chunk corpus or the
filesystem: the candidates are hand-built.

Vocabulary follows `GLOSSARY.md`: a gold chunk is one a question is scored as having
needed, established without a model's opinion, and the shortlist the marks are made over
is mechanical — bm25 and embedding similarity order it and nothing decides with it.

  uv run pytest tests/test_labelsheet.py
"""
from dataclasses import dataclass

import pytest

from build_golden import label_rows, labelsheet_marks, render_labelsheet


def sheet(*lines):
    """A sheet fragment, written the way a person marking one would leave it."""
    return "\n".join(lines)


QUESTION = "- [{mark}] question `A-q1` — tick the gold chunks below"
CANDIDATE = "  - [{mark}] chunk `{cid}` · bm25 #1 (12.40) · dense #2"


@dataclass(frozen=True)
class Case:
    """One marked sheet and the verdict it should be read as carrying."""

    behaviour: str
    sheet: str
    expects: dict


CASES = [
    Case(
        behaviour="a ticked candidate is a gold chunk, and ticking one is itself the "
                  "statement that the question is labelled",
        sheet=sheet(QUESTION.format(mark=" "),
                    CANDIDATE.format(mark="x", cid="a.mp4:0"),
                    CANDIDATE.format(mark=" ", cid="b.mp4:1")),
        expects={"status": "labelled", "gold": ["a.mp4:0"]},
    ),
    Case(
        behaviour="several candidates can be ticked, so a question whose answer spans "
                  "chunks is labelled truthfully rather than forced to one",
        sheet=sheet(QUESTION.format(mark="x"),
                    CANDIDATE.format(mark="x", cid="a.mp4:0"),
                    CANDIDATE.format(mark="x", cid="b.mp4:1")),
        expects={"status": "labelled", "gold": ["a.mp4:0", "b.mp4:1"]},
    ),
    Case(
        behaviour="a question marked [-] with nothing ticked says the corpus holds no "
                  "gold chunk for it, which is a finding and not an absent label",
        sheet=sheet(QUESTION.format(mark="-"),
                    CANDIDATE.format(mark=" ", cid="a.mp4:0")),
        expects={"status": "no_gold_in_corpus", "gold": []},
    ),
    Case(
        behaviour="an untouched question is not yet labelled, which is distinct from "
                  "having no gold chunk",
        sheet=sheet(QUESTION.format(mark=" "),
                    CANDIDATE.format(mark=" ", cid="a.mp4:0")),
        expects={"status": "unlabelled", "gold": []},
    ),
    Case(
        behaviour="an empty [] is the same untick as [ ], because editors strip the "
                  "space out and stopping over whitespace would cost a sitting's work",
        sheet=sheet(QUESTION.format(mark=""),
                    CANDIDATE.format(mark="", cid="a.mp4:0")),
        expects={"status": "unlabelled", "gold": []},
    ),
    Case(
        behaviour="an upper case X is a tick, since a person marking 83 questions is "
                  "not holding a case convention in mind",
        sheet=sheet(QUESTION.format(mark=" "),
                    CANDIDATE.format(mark="X", cid="a.mp4:0")),
        expects={"status": "labelled", "gold": ["a.mp4:0"]},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.behaviour for c in CASES])
def test_marks_are_read_back(case):
    verdict = labelsheet_marks(case.sheet)["A-q1"]
    assert {key: verdict[key] for key in case.expects} == case.expects


@dataclass(frozen=True)
class RaiseCase:
    """One sheet a person could plausibly leave, and what the refusal must name."""

    behaviour: str
    sheet: str
    names: str


RAISE_CASES = [
    RaiseCase(
        behaviour="a question mark the vocabulary does not define raises, naming the "
                  "question, rather than being skipped into the unlabelled count",
        sheet=sheet(QUESTION.format(mark="?"),
                    CANDIDATE.format(mark=" ", cid="a.mp4:0")),
        names="A-q1",
    ),
    RaiseCase(
        behaviour="a candidate mark the vocabulary does not define raises, naming the "
                  "chunk, rather than dropping a gold chunk out of recall",
        sheet=sheet(QUESTION.format(mark=" "),
                    CANDIDATE.format(mark="~", cid="a.mp4:0")),
        names="a.mp4:0",
    ),
    RaiseCase(
        behaviour="[-] with a candidate ticked is a contradiction and raises, because "
                  "either the verdict or the tick is a slip and neither may be guessed",
        sheet=sheet(QUESTION.format(mark="-"),
                    CANDIDATE.format(mark="x", cid="a.mp4:0")),
        names="A-q1",
    ),
    RaiseCase(
        behaviour="[x] with nothing ticked is a contradiction and raises, because a "
                  "question labelled as having gold chunks has to name them",
        sheet=sheet(QUESTION.format(mark="x"),
                    CANDIDATE.format(mark=" ", cid="a.mp4:0")),
        names="A-q1",
    ),
]


@pytest.mark.parametrize("case", RAISE_CASES, ids=[c.behaviour for c in RAISE_CASES])
def test_a_mark_that_cannot_be_read_raises_with_its_id_named(case):
    with pytest.raises(ValueError, match=case.names):
        labelsheet_marks(case.sheet)


# ----------------------------------------- the labels as they are written to the tier


def question(qid="A-q1", **fields):
    """A tier A row, carrying only what the label write reads or preserves."""
    return {"id": qid, "tier": "A", "question": "why?", "answer": "because",
            "hands_on": False, **fields}


def test_a_labelled_question_is_written_with_its_gold_chunks():
    rows = label_rows([question()], {"A-q1": {"status": "labelled",
                                              "gold": ["a.mp4:0", "b.mp4:1"]}})
    assert rows[0]["gold_chunks"] == ["a.mp4:0", "b.mp4:1"]
    assert "no_gold_in_corpus" not in rows[0]
    # the question itself is untouched; this stage adds labels and nothing else
    assert rows[0]["question"] == "why?"


def test_a_no_gold_question_is_written_as_a_verdict_not_an_empty_label():
    rows = label_rows([question()], {"A-q1": {"status": "no_gold_in_corpus", "gold": []}})
    assert rows[0]["no_gold_in_corpus"] is True
    assert "gold_chunks" not in rows[0]


def test_an_unlabelled_question_is_written_with_neither_field():
    rows = label_rows([question()], {"A-q1": {"status": "unlabelled", "gold": []}})
    assert "gold_chunks" not in rows[0]
    assert "no_gold_in_corpus" not in rows[0]


def test_relabelling_replaces_the_previous_label_rather_than_accumulating():
    """Partial labelling means labels are revisited, and a stale one would be read.

    A question labelled with gold chunks and later judged to have none in the corpus
    would otherwise keep scoring recall against the chunks it no longer claims.
    """
    already = question(gold_chunks=["a.mp4:0"])
    rows = label_rows([already], {"A-q1": {"status": "no_gold_in_corpus", "gold": []}})
    assert "gold_chunks" not in rows[0]
    assert rows[0]["no_gold_in_corpus"] is True


def test_a_mark_for_a_question_the_tier_no_longer_holds_raises():
    """A label with nowhere to go is a lost judgement, so it stops the stage.

    Re-running `parse` can drop a question, and dropping its label silently would leave
    the sheet and the tier disagreeing about what was labelled.
    """
    with pytest.raises(ValueError, match="A-gone"):
        label_rows([question()], {"A-gone": {"status": "labelled", "gold": ["a.mp4:0"]}})


def test_a_gold_chunk_id_absent_from_the_corpus_raises():
    """A hand-typed id is the one route into this sheet that is not machine-generated.

    The sheet invites adding a candidate line by hand when the right chunk is not in the
    shortlist, so a typo in a chunk id is reachable, and it would otherwise be stored as
    a gold chunk that retrieval can never return.
    """
    with pytest.raises(ValueError, match="typo.mp4:9"):
        label_rows([question()], {"A-q1": {"status": "labelled", "gold": ["typo.mp4:9"]}},
                   known_chunks={"a.mp4:0"})


# ------------------------------------------ regeneration, which may not lose a mark


def candidate(cid, bm25_rank=1, dense_rank=None):
    """One shortlisted chunk, shaped the way the shortlist builder yields them."""
    return {"chunk_id": cid, "text": f"transcript text of {cid}",
            "bm25_rank": bm25_rank, "bm25_score": 12.4, "dense_rank": dense_rank}


def entry(qid="A-q1", candidates=None):
    """One question and its shortlist, as the sheet writer takes them."""
    return {"id": qid, "question": "why?", "answer": "because", "quiz": "Quiz 1",
            "hands_on": False, "candidates": candidates or [candidate("a.mp4:0")]}


def test_a_regenerated_sheet_still_carries_the_marks_it_was_given():
    """Marks survive regeneration, read back through the parser rather than by eye.

    Asserting on the rendered text would pin the sheet's appearance, which is the one
    thing about it that is free to change. What is promised is that the human's work
    comes back out of the sheet the writer produced.
    """
    marks = {"A-q1": {"status": "labelled", "gold": ["a.mp4:0"]}}
    rendered = render_labelsheet([entry()], marks)
    assert labelsheet_marks(rendered)["A-q1"]["gold"] == ["a.mp4:0"]


def test_a_no_gold_verdict_survives_regeneration():
    marks = {"A-q1": {"status": "no_gold_in_corpus", "gold": []}}
    rendered = render_labelsheet([entry()], marks)
    assert labelsheet_marks(rendered)["A-q1"]["status"] == "no_gold_in_corpus"


def test_a_candidate_added_by_regeneration_arrives_unticked():
    marks = {"A-q1": {"status": "labelled", "gold": ["a.mp4:0"]}}
    rendered = render_labelsheet(
        [entry(candidates=[candidate("a.mp4:0"), candidate("new.mp4:7", bm25_rank=2)])],
        marks)
    assert labelsheet_marks(rendered)["A-q1"]["gold"] == ["a.mp4:0"]


def test_a_ticked_chunk_dropped_from_the_shortlist_is_kept_as_a_candidate():
    """A shortlist is mechanical and may change; a tick on it is a human judgement.

    Widening or reordering the shortlist can push a chunk out of it. Rendering only the
    current candidates would then erase a gold label that was already made, which is the
    one failure mode this sheet cannot recover from.
    """
    marks = {"A-q1": {"status": "labelled", "gold": ["dropped.mp4:3"]}}
    rendered = render_labelsheet([entry(candidates=[candidate("a.mp4:0")])], marks)
    assert labelsheet_marks(rendered)["A-q1"]["gold"] == ["dropped.mp4:3"]


def test_an_unmarked_sheet_reads_back_as_wholly_unlabelled():
    """The first sheet asserts nothing, so the report's denominator starts at zero."""
    verdicts = labelsheet_marks(render_labelsheet([entry("A-q1"), entry("A-q2")], {}))
    assert [v["status"] for v in verdicts.values()] == ["unlabelled", "unlabelled"]


def test_every_question_reaches_the_sheet_with_its_candidate_text():
    rendered = render_labelsheet([entry("A-q1"), entry("A-q2")], {})
    assert set(labelsheet_marks(rendered)) == {"A-q1", "A-q2"}
    assert "transcript text of a.mp4:0" in rendered
