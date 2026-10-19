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
# Also real: one filename contains another, so a bare match on the inner name sits inside
# a mention of the outer one.
NESTED_CORPUS = (
    "building and applying a classification model.mp4",
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
        behaviour="a timestamp echoing a chunk's floored start is credited, because the "
                  "floored start is the only time the context ever displayed",
        answer="Precision and recall are defined in classification.mp4 at 03:02.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_time_valid": 1.0},
    ),
    Case(
        behaviour="a timestamp inside a span but not at its start is credited",
        answer="classification.mp4 covers the confusion matrix at 03:20.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_time_valid": 1.0},
    ),
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
        behaviour="a bare video name sitting next to a timestamp is a citation, so the "
                  "metric does not depend on the model writing the file extension",
        answer="Precision and recall are defined in classification at 03:02.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 1.0,
            "citation_video_valid": 1.0,
            "citation_time_valid": 1.0,
            "cited_videos": ["classification.mp4"],
        },
    ),
    Case(
        behaviour="an answer mixing an extension-bearing and a bare name attributes "
                  "each timestamp to the video it was written next to",
        answer="classification.mp4 defines recall at 03:02, "
               "and cluster analysis at 05:10 covers k-means.",
        hits=[
            chunk("classification.mp4", 182.4, 214.9, 3),
            chunk("cluster analysis.mp4", 309.6, 341.2, 5),
        ],
        expects={
            "citation_time_valid": 1.0,
            "cited_videos": ["classification.mp4", "cluster analysis.mp4"],
        },
    ),
    Case(
        behaviour="a timestamp written before its video name is attributed to that "
                  "name when it falls in the same clause",
        answer="At 03:02, classification.mp4 defines recall.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_time_valid": 1.0},
    ),
    Case(
        behaviour="a timestamp with no video named either side of it is uncheckable, "
                  "and counts invalid rather than valid",
        answer="The confusion matrix is covered at 03:02.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 0.0,
            "citation_video_valid": None,
            "citation_time_valid": 0.0,
            "cited_videos": [],
        },
    ),
    Case(
        behaviour="an answer that cites nothing reports no citation, and leaves both "
                  "validity metrics absent rather than zero",
        answer="Clustering groups unlabelled records by similarity.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 0.0,
            "citation_video_valid": None,
            "citation_time_valid": None,
            "cited_videos": [],
        },
    ),
    Case(
        behaviour="a bare video name in ordinary prose, with no timestamp beside it, is "
                  "not a citation of the video that shares its name",
        answer="Clustering groups unlabelled records by similarity, "
               "while classification assigns a known label.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={"citation_present": 0.0, "cited_videos": []},
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
    Case(
        behaviour="one video name that happens to contain another is cited once, as the "
                  "video the answer actually named",
        answer="This is covered in building and applying a classification model "
               "at 03:02.",
        hits=[chunk("building and applying a classification model.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_video_valid": 1.0,
            "citation_time_valid": 1.0,
            "cited_videos": ["building and applying a classification model.mp4"],
        },
        corpus=NESTED_CORPUS,
    ),
    # Deliberate, and the one place the spec argues with itself. Story 3 asks for a
    # timestamp written before its video name to be attributed to that name in the same
    # clause, and the case above does that for `classification.mp4`. But the
    # Implementation Decision says a bare name counts "only when a timestamp follows
    # it", so a bare name after its timestamp is never recognised at all, and the
    # forward attribution has nothing to reach. The Implementation Decision governs and
    # the behaviour is left as it is; this case exists so the asymmetry is on record
    # rather than discovered again as a defect. It awaits the user's ruling.
    Case(
        behaviour="a bare video name written after its timestamp is not recognised, so "
                  "the timestamp stays uncheckable and counts invalid",
        answer="At 03:02, classification defines recall.",
        hits=[chunk("classification.mp4", 182.4, 214.9, 3)],
        expects={
            "citation_present": 0.0,
            "citation_video_valid": None,
            "citation_time_valid": 0.0,
            "cited_videos": [],
        },
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.behaviour)
def test_citation_metrics(case):
    metrics = check_citations(case.answer, case.hits, known_videos=case.corpus)
    assert {key: metrics[key] for key in case.expects} == case.expects
