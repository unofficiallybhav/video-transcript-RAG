"""Tests for the judge calibration stage (issue #8).

Every test here runs with no network, no credentials and no provider call: the judge is
a stub passed in at the call site, which is the same seam the real run uses.
"""
import json
import random

import pytest

import evaluate


def checkpoint_rows(a=50, c=12):
    """Checkpoint-shaped rows: scored questions, as results.jsonl has them."""
    rows = []
    for tier, count in (("A", a), ("C", c)):
        for i in range(count):
            rows.append({
                "id": f"tier{tier}-q{i:03d}",
                "tier": tier,
                "question": f"question {tier}{i}",
                "answer": f"system answer {tier}{i}",
                "retrieved": [f"video {i}.mp4:{j}" for j in range(5)],
                "faithfulness": 0.80,
                "relevancy": 0.90,
                "correctness": 0.70,
                "refused": False,
            })
    return rows


# ----------------------------------------------------------------- the seeded sample

def test_sample_is_thirty_tier_a_questions():
    """The strata follow the surviving tiers: tier C's ground truth carries no score
    for the judge to disagree with itself about, so calibration draws from tier A."""
    sample = evaluate.calibration_sample(checkpoint_rows())
    assert len(sample) == 30
    assert sum(1 for r in sample if r["tier"] == "A") == 30


def test_sample_is_reproducible_from_the_seed():
    first = evaluate.calibration_sample(checkpoint_rows())
    second = evaluate.calibration_sample(checkpoint_rows())
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_sample_does_not_depend_on_checkpoint_order():
    rows = checkpoint_rows()
    ordered = evaluate.calibration_sample(rows)
    shuffled_rows = list(rows)
    random.Random(99).shuffle(shuffled_rows)
    assert ([r["id"] for r in evaluate.calibration_sample(shuffled_rows)]
            == [r["id"] for r in ordered])


def test_sample_ignores_tier_c_and_unscored_rows():
    rows = checkpoint_rows()
    rows.append({"id": "tierC-q000", "tier": "C", "question": "q", "answer": "a",
                 "retrieved": [], "refused": True})
    rows.append({"id": "tierA-unscored", "tier": "A", "question": "q", "answer": "a",
                 "retrieved": [], "faithfulness": None, "relevancy": None,
                 "correctness": None})
    ids = [r["id"] for r in evaluate.calibration_sample(rows)]
    assert "tierC-q000" not in ids
    assert "tierA-unscored" not in ids


# ------------------------------------------- the pass with the reference withheld

def test_withheld_prompt_contains_no_reference_answer():
    prompt = evaluate.no_reference_prompt()
    assert "REFERENCE" not in prompt
    assert "{reference}" not in prompt


def test_withheld_prompt_does_not_ask_for_correctness():
    prompt = evaluate.no_reference_prompt()
    assert "correctness" not in prompt


def test_withheld_prompt_keeps_the_faithfulness_criterion_word_for_word():
    criterion = evaluate.JUDGE_PROMPT.split("- faithfulness:")[1].split("- relevancy:")[0]
    assert criterion in evaluate.no_reference_prompt()


def test_withheld_prompt_formats_from_question_context_and_answer_alone():
    # a leftover {reference} would raise KeyError here rather than reach the judge
    filled = evaluate.no_reference_prompt().format(
        question="q", context="c", answer="a")
    assert "reference" not in filled.lower()


# --------------------------------------------- the judge call site is injectable

def stub_judge(scores):
    """A judge that replies with fixed scores and records the prompts it was sent."""
    sent = []

    def complete_fn(model, prompt=None, messages=None):
        sent.append({"model": model, "prompt": prompt})
        return "here are the scores: " + json.dumps(scores)

    complete_fn.sent = sent
    return complete_fn


def test_judge_scores_an_answer_through_an_injected_client():
    fake = stub_judge({"faithfulness": 0.75, "relevancy": 0.5, "correctness": 0.25,
                       "refused": False, "note": "stub"})
    scores = evaluate.judge({"question": "q", "answer": "ref"}, "ans", "ctx",
                            complete_fn=fake)
    assert scores["faithfulness"] == 0.75
    assert scores["correctness"] == 0.25
    assert len(fake.sent) == 1
    assert fake.sent[0]["model"] == evaluate.JUDGE_MODEL
    assert "ref" in fake.sent[0]["prompt"]


def test_judge_with_the_reference_withheld_never_sends_it():
    fake = stub_judge({"faithfulness": 0.75, "relevancy": 0.5, "refused": False})
    scores = evaluate.judge({"question": "q", "answer": "THE-REFERENCE"}, "ans", "ctx",
                            prompt=evaluate.no_reference_prompt(), complete_fn=fake)
    assert "THE-REFERENCE" not in fake.sent[0]["prompt"]
    assert scores["faithfulness"] == 0.75
    assert scores["correctness"] is None


# ------------------------------------- the numbers, and the pre-committed outcomes

def judged(qid, which, faith, relevancy=0.9, correctness=0.7):
    return {"id": qid, "pass": which, "faithfulness": faith,
            "relevancy": relevancy, "correctness": correctness, "refused": False}


def test_between_run_difference_is_the_mean_absolute_gap_per_metric():
    baseline = [dict(r, faithfulness=0.80) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "rerun_2", 1.00), judged("tierA-q001", "rerun_2", 0.80)]
    summary = evaluate.calibration_summary(baseline, results)
    # |1.00-0.80| and |0.80-0.80| -> mean 0.10
    assert summary["between_run"]["faithfulness"]["mean_absolute_difference"] == 0.1
    assert summary["between_run"]["faithfulness"]["n"] == 2
    # 1.00-0.80 in binary floating point is 0.19999999999999996; a reader of the
    # artifact should not have to see that
    assert summary["between_run"]["faithfulness"]["max_absolute_difference"] == 0.2


def test_between_run_spread_ignores_the_stored_score_and_reports_its_drift_apart():
    """Two identical fresh passes agree, however far the stored run has drifted.

    The threshold must read judge nondeterminism only. A score stored on an earlier day
    carries provider-side model drift as well, so it is reported separately and is not
    allowed to move the verdict.
    """
    baseline = [dict(r, faithfulness=0.20) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "rerun_2", 0.80), judged("tierA-q001", "rerun_2", 0.80)]
    summary = evaluate.calibration_summary(baseline, results)

    assert summary["between_run"]["faithfulness"]["mean_absolute_difference"] == 0.0
    assert summary["outcomes"][0]["triggered"] is False
    # the drift against the stored run is still visible, just not on the threshold
    assert summary["versus_stored_run"]["faithfulness"]["mean_absolute_difference"] == 0.6


def test_a_mean_difference_exactly_at_the_threshold_does_not_retire_two_decimals():
    baseline = [dict(r, faithfulness=0.80) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "rerun_2", 1.00), judged("tierA-q001", "rerun_2", 0.80)]
    outcome = evaluate.calibration_summary(baseline, results)["outcomes"][0]
    assert outcome["triggered"] is False
    assert outcome["outcome"] == "two-decimal continuous scoring is kept"


def test_a_mean_difference_above_the_threshold_retires_two_decimals():
    baseline = [dict(r, faithfulness=0.80) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "rerun_2", 0.50), judged("tierA-q001", "rerun_2", 0.80)]
    outcome = evaluate.calibration_summary(baseline, results)["outcomes"][0]
    assert outcome["triggered"] is True
    assert outcome["outcome"] == ("two-decimal continuous scoring is retired in favour "
                                  "of one decimal")
    assert outcome["worst_metric"] == "faithfulness"


def test_faithfulness_shift_without_the_reference_is_measured_against_the_rerun():
    baseline = [dict(r, faithfulness=0.80) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "no_reference", 0.50),
               judged("tierA-q001", "no_reference", 0.80)]
    summary = evaluate.calibration_summary(baseline, results)
    without = summary["no_reference"]["faithfulness"]
    assert without["mean_absolute_difference"] == 0.15
    assert without["mean_signed_shift"] == -0.15
    assert summary["outcomes"][1]["triggered"] is True
    assert summary["outcomes"][1]["outcome"] == ("faithfulness is split into its own "
                                                 "judge call, permanently")


def test_the_thresholds_are_recorded_as_pre_committed_either_way():
    baseline = [dict(r, faithfulness=0.80) for r in checkpoint_rows(a=2, c=0)]
    results = [judged("tierA-q000", "rerun", 0.80), judged("tierA-q001", "rerun", 0.80),
               judged("tierA-q000", "no_reference", 0.80),
               judged("tierA-q001", "no_reference", 0.80)]
    summary = evaluate.calibration_summary(baseline, results)
    assert summary["pre_committed"] is True
    for outcome in summary["outcomes"]:
        assert outcome["threshold"] == 0.10
        assert outcome["triggered"] is False
        assert outcome["if_exceeded"] and outcome["if_not_exceeded"]
        assert outcome["outcome"] == outcome["if_not_exceeded"]


# ---------------------------------------------------- the stage, end to end

def calibration_workspace(monkeypatch, tmp_path, rows):
    """A throwaway golden/ holding a checkpoint, a golden set and a main report.

    Nothing here touches the real corpus or the real report, and no credential is read.
    """
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (golden_dir / "golden.json").write_text(json.dumps(
        [{"id": r["id"], "tier": r["tier"], "question": r["question"],
          "answer": f"reference for {r['id']}"} for r in rows]), encoding="utf-8")
    (golden_dir / "report.json").write_text('{"summary": "untouched"}', encoding="utf-8")
    (golden_dir / "report.md").write_text("# untouched\n", encoding="utf-8")

    chunks = [{"chunk_id": cid, "source": cid.split(":")[0],
               "chunk_index": int(cid.split(":")[1]), "start": 12.5, "end": 40.0,
               "text": f"transcript text of {cid}"}
              for r in rows for cid in r["retrieved"]]

    monkeypatch.setattr(evaluate, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(evaluate, "RESULTS_PATH", golden_dir / "results.jsonl")
    monkeypatch.setattr(evaluate, "GOLDEN_PATH", golden_dir / "golden.json")
    monkeypatch.setattr(evaluate, "CALIBRATION_PATH", golden_dir / "judge_calibration.json")
    monkeypatch.setattr(evaluate, "CALIBRATION_RAW_PATH",
                        golden_dir / "judge_calibration.jsonl")
    monkeypatch.setattr(evaluate, "load_chunks", lambda: chunks)
    return golden_dir


def test_calibrate_judges_the_sample_three_times_and_writes_its_own_artifact(
        monkeypatch, tmp_path):
    rows = checkpoint_rows(a=40, c=15)
    golden_dir = calibration_workspace(monkeypatch, tmp_path, rows)
    fake = stub_judge({"faithfulness": 0.8, "relevancy": 0.9, "correctness": 0.7,
                       "refused": False, "note": "stub"})

    evaluate.calibrate(complete_fn=fake)

    assert len(fake.sent) == 90
    artifact = json.loads(
        (golden_dir / "judge_calibration.json").read_text("utf-8"))["summary"]
    assert len(artifact["sample"]) == 30
    assert artifact["seed"] == evaluate.CALIBRATION_SEED
    assert artifact["judge_model"] == evaluate.JUDGE_MODEL
    assert [o["threshold"] for o in artifact["outcomes"]] == [0.10, 0.10]
    assert all(o["outcome"] for o in artifact["outcomes"])


def test_calibrate_leaves_the_main_report_untouched(monkeypatch, tmp_path):
    rows = checkpoint_rows(a=40, c=15)
    golden_dir = calibration_workspace(monkeypatch, tmp_path, rows)
    evaluate.calibrate(complete_fn=stub_judge({"faithfulness": 0.8, "relevancy": 0.9,
                                               "correctness": 0.7, "refused": False}))
    assert (golden_dir / "report.json").read_text("utf-8") == '{"summary": "untouched"}'
    assert (golden_dir / "report.md").read_text("utf-8") == "# untouched\n"


def test_calibrate_resumes_after_the_daily_limit_without_rejudging(monkeypatch, tmp_path):
    rows = checkpoint_rows(a=40, c=15)
    golden_dir = calibration_workspace(monkeypatch, tmp_path, rows)

    good = stub_judge({"faithfulness": 0.8, "relevancy": 0.9, "correctness": 0.7,
                       "refused": False})

    def exhausted(model, prompt=None, messages=None):
        if len(good.sent) >= 6:
            raise evaluate.DailyLimitReached("tokens per day exceeded")
        return good(model, prompt=prompt)

    evaluate.calibrate(complete_fn=exhausted)
    assert not (golden_dir / "judge_calibration.json").exists()
    assert len((golden_dir / "judge_calibration.jsonl").read_text("utf-8")
               .strip().splitlines()) == 6

    rest = stub_judge({"faithfulness": 0.8, "relevancy": 0.9, "correctness": 0.7,
                       "refused": False})
    evaluate.calibrate(complete_fn=rest)
    assert len(rest.sent) == 84
    assert (golden_dir / "judge_calibration.json").exists()


@pytest.fixture(scope="module")
def checkpoint():
    rows = list(evaluate.load_checkpoint().values())
    if not rows:
        pytest.skip("golden/results.jsonl is empty")
    return rows


def test_sample_is_reproducible_from_the_real_checkpoint(checkpoint):
    """The 30 questions the real run will judge, derived without any provider call."""
    rows = checkpoint
    sample = evaluate.calibration_sample(rows)
    assert sum(1 for r in sample if r["tier"] == "A") == 30
    assert [r["id"] for r in sample] == [
        r["id"] for r in evaluate.calibration_sample(list(reversed(rows)))]
