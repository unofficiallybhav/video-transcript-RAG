"""Behaviour of the judge-free citation check, stated the way the report reads it.

Each case hands `check_citations` one hand-written answer and one hand-built set of
retrieved chunks, then asserts the citation metrics that come back. Nothing here reaches
into how names are matched or stamps are parsed, and nothing touches the vector store,
the provider or the filesystem: the corpus is injected through `known_videos`, which is
what that argument is for.

To add a behaviour, append a `Case` to `CASES`. Keep `expects` to the metrics the
behaviour statement actually names, so a red case names exactly one reason, and state
the behaviour in the vocabulary of `GLOSSARY.md` — a citation is a named video together
with a timestamp, and citation validity is a property of that reference alone.

  uv run pytest
"""
from dataclasses import dataclass

import pytest

from evaluate import check_citations

# Real filenames from the corpus, longest first, as `video_names()` would supply them.
CORPUS = (
    "cluster analysis.mp4",
    "classification.mp4",
)


def chunk(source, start, end, index):
    """One retrieved hit, shaped the way `retrieve()` returns them."""
    return (
        f"{source}:{index}",
        {
            "source": source,
            "start": start,
            "end": end,
            "chunk_index": index,
            "text": "transcript text, which the citation check never reads",
        },
    )


@dataclass(frozen=True)
class Case:
    """One answer, one retrieved set, and the metrics that behaviour should produce."""

    behaviour: str
    answer: str
    hits: list
    expects: dict
    corpus: tuple = CORPUS


CASES = [
    Case(
        behaviour="an invented timestamp outside every retrieved span is not credited",
        answer="The decision boundary is covered in classification.mp4 at 12:00.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_time_valid": 0.0},
    ),
    Case(
        behaviour="a timestamp from a chunk that was not retrieved is not credited, "
                  "though the video it names was",
        answer="classification.mp4 defines the confusion matrix at 02:00.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_video_valid": 1.0, "citation_time_valid": 0.0},
    ),
    Case(
        behaviour="an answer that cites nothing reports no citation, and leaves both "
                  "validity metrics absent rather than zero",
        answer="Clustering groups unlabelled records by similarity, "
               "while classification assigns a known label.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 0.0,
            "citation_video_valid": None,
            "citation_time_valid": None,
            "cited_videos": [],
        },
    ),
    Case(
        behaviour="a refusal reports no citation, and leaves both validity metrics "
                  "absent rather than zero",
        answer="The provided transcripts do not cover that topic, "
               "so I cannot answer the question.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 0.0,
            "citation_video_valid": None,
            "citation_time_valid": None,
            "cited_videos": [],
        },
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.behaviour)
def test_citation_metrics(case):
    metrics = check_citations(case.answer, case.hits, known_videos=case.corpus)
    assert {key: metrics[key] for key in case.expects} == case.expects
