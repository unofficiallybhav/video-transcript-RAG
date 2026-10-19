"""The offline re-derivation must reproduce the run that was already published.

The expected figures below are the ones in the report published on 2026-10-17, copied
in as literals rather than read back from `golden/report.json`, which the stage rewrites.
`citation_time_valid` of 0.488 is a known defect in `check_citations` and is corrected by
later tickets; reproducing it here is what proves the re-derivation path faithful.
"""
import pytest

import evaluate
from build_golden import load_chunks

PUBLISHED_CITATIONS = {
    "tier_a": {"questions": 83, "citation_present": 0.831,
               "citation_video_valid": 0.988, "citation_time_valid": 0.488},
    "tier_b": {"questions": 40, "citation_present": 0.975,
               "citation_video_valid": 0.987, "citation_time_valid": 0.62,
               "precision@1": 0.625, "recall@1": 0.588,
               "precision@5": 0.2, "recall@5": 0.912,
               "precision@10": 0.103, "recall@10": 0.925},
}
JUDGE_FIELDS = ("faithfulness", "relevancy", "correctness", "refused", "note")


@pytest.fixture(scope="module")
def checkpoint():
    rows = list(evaluate.load_checkpoint().values())
    if not rows:
        pytest.skip("golden/results.jsonl is empty")
    return rows


@pytest.fixture(scope="module")
def chunks():
    corpus = load_chunks()
    if not corpus:
        pytest.skip("data/ is absent, so the chunk corpus cannot be rebuilt")
    return corpus


@pytest.fixture(scope="module")
def rescored(checkpoint, chunks):
    return evaluate.rescore_rows(checkpoint, chunks)


@pytest.mark.parametrize("tier", sorted(PUBLISHED_CITATIONS))
def test_reproduces_published_citation_figures(rescored, tier):
    block = evaluate.summarise(rescored)[tier]
    expected = PUBLISHED_CITATIONS[tier]
    assert {k: block[k] for k in expected} == expected


def test_judge_fields_pass_through_unmodified(checkpoint, rescored):
    before = [{k: r.get(k) for k in JUDGE_FIELDS} for r in checkpoint]
    after = [{k: r.get(k) for k in JUDGE_FIELDS} for r in rescored]
    assert after == before


def test_rescoring_is_idempotent(rescored, chunks):
    assert evaluate.rescore_rows(rescored, chunks) == rescored
