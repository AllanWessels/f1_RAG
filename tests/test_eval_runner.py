"""
Tests for eval/runner.py.

Uses synthetic 6-question subset (1 per bucket) and mocked pipelines via
the `runners` injection parameter.

Asserts:
  - 24 results (4 versions x 6 questions)
  - `aggregates` has correct shape
  - Output JSON is written and `latest.json` exists + matches
  - A failing run (exception) shows up as error != null + grade.pass == False
  - The eval does not crash when a pipeline raises
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.pipeline import PipelineResult
from eval.runner import EvalReport, run_eval

# ---------------------------------------------------------------------------
# Synthetic 6-question YAML (1 per bucket)
# ---------------------------------------------------------------------------

SYNTHETIC_YAML = """
version: 1
questions:
  - id: "factual_stats_001"
    bucket: "factual_stats"
    question: "Who won the 2017 Australian Grand Prix?"
    gold_answer: "Sebastian Vettel"
    grading:
      kind: "exact_match"
      accept:
        - "Sebastian Vettel"
        - "Vettel"
    expected_retriever: "query_stats"

  - id: "multi_hop_001"
    bucket: "multi_hop"
    question: "Who was Verstappen's teammate at Red Bull when he won his first title?"
    gold_answer: "Sergio Perez"
    grading:
      kind: "exact_match"
      accept:
        - "Sergio Perez"
        - "Perez"
        - "Checo"
    expected_retriever: "query_stats"

  - id: "narrative_001"
    bucket: "narrative"
    question: "What was the controversy at the 2021 Abu Dhabi GP?"
    gold_answer: "Safety car controversy"
    grading:
      kind: "llm_judge"
      rubric:
        - "Mentions the late safety car"
        - "Mentions Verstappen passing Hamilton"
    expected_retriever: "search_narratives"

  - id: "regulation_001"
    bucket: "regulation"
    question: "What is the minimum car weight per the 2026 Technical Regs?"
    gold_answer: "768 kg"
    grading:
      kind: "numeric"
      value: 768
      unit: "kg"
      tolerance: 0
      cite_article: "Technical Regulations Article 4.1"
    expected_retriever: "search_regulations"

  - id: "temporal_comparative_001"
    bucket: "temporal_comparative"
    question: "How did Red Bull's wins compare in 2021 vs 2022?"
    gold_answer: "More in 2022"
    grading:
      kind: "llm_judge"
      rubric:
        - "States Red Bull won more in 2022 than 2021"
    expected_retriever: "query_stats"

  - id: "adversarial_001"
    bucket: "adversarial"
    question: "Who won the 2020 European Grand Prix?"
    gold_answer: "No such race"
    grading:
      kind: "refusal"
      reason: "no_such_race"
    expected_retriever: "query_stats"
"""

# ---------------------------------------------------------------------------
# Fixture: tmp questions YAML + tmp output dir
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_questions_path(tmp_path: Path) -> Path:
    p = tmp_path / "questions.yaml"
    p.write_text(SYNTHETIC_YAML)
    return p


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Mock pipeline builder
# ---------------------------------------------------------------------------


def _make_runners(
    answer_map: dict[str, str] | None = None,
    refused: bool = False,
    error_versions: set[str] | None = None,
) -> dict[str, Callable]:
    """
    Build mock runner callables.

    answer_map: version → answer string (same answer for all questions).
    error_versions: set of version IDs that should raise RuntimeError.
    """
    answer_map = answer_map or {
        "v1": "Sebastian Vettel won.",
        "v2": "Sebastian Vettel won.",
        "v3": "Sebastian Vettel won.",
        "v4": "Sebastian Vettel won.",
    }
    error_versions = error_versions or set()

    def _make_fn(version: str):
        def runner(query: str, **_kwargs) -> PipelineResult:
            if version in error_versions:
                raise RuntimeError(f"Simulated failure in {version}")
            return PipelineResult(
                version=version,
                query=query,
                answer=answer_map.get(version, ""),
                citations=[],
                refused=refused,
                retrieved_contexts=[],
                tool_calls=[],
            )

        return runner

    return {v: _make_fn(v) for v in ["v1", "v2", "v3", "v4"]}


# ---------------------------------------------------------------------------
# Mock LLM judge that always returns a passing score
# ---------------------------------------------------------------------------


def _make_passing_judge_client():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = (
        '{"satisfied": [0], "missing": [], "score": 1.0, "reasoning": "All good."}'
    )
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunEval:
    def test_result_count(self, synthetic_questions_path, output_dir):
        """4 versions x 6 questions = 24 results."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        assert len(report.results) == 24

    def test_aggregates_shape(self, synthetic_questions_path, output_dir):
        """aggregates has keys for each version; each has overall_pass_rate and per_bucket."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        for version in ["v1", "v2", "v3", "v4"]:
            assert version in report.aggregates
            agg = report.aggregates[version]
            assert 0.0 <= agg.overall_pass_rate <= 1.0
            assert isinstance(agg.per_bucket, dict)
            # 6 buckets in our synthetic set
            assert len(agg.per_bucket) == 6

    def test_output_json_written(self, synthetic_questions_path, output_dir):
        """Timestamped JSON file is written to output_dir."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) >= 2  # at least timestamped + latest

    def test_latest_json_exists(self, synthetic_questions_path, output_dir):
        """latest.json exists after run."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        assert (output_dir / "latest.json").exists()

    def test_latest_json_matches_timestamped(self, synthetic_questions_path, output_dir):
        """latest.json content matches the timestamped file."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        json_files = sorted(
            [f for f in output_dir.glob("*.json") if f.name != "latest.json"]
        )
        assert json_files, "No timestamped JSON file found"
        ts_content = json_files[-1].read_text()
        latest_content = (output_dir / "latest.json").read_text()
        assert ts_content == latest_content

    def test_json_is_valid_and_has_expected_keys(self, synthetic_questions_path, output_dir):
        """The output JSON has all required top-level keys."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        run_eval(
            versions=["v1", "v2", "v3", "v4"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        data = json.loads((output_dir / "latest.json").read_text())
        assert "timestamp" in data
        assert "versions" in data
        assert "questions" in data
        assert "results" in data
        assert "aggregates" in data

    def test_result_grade_structure(self, synthetic_questions_path, output_dir):
        """Each result has grade dict with pass, score, detail."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        for r in report.results:
            assert "pass" in r.grade
            assert "score" in r.grade
            assert "detail" in r.grade

    def test_version_subset(self, synthetic_questions_path, output_dir):
        """Running only v1, v3 produces 12 results."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1", "v3"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        assert len(report.results) == 12
        versions_seen = {r.version for r in report.results}
        assert versions_seen == {"v1", "v3"}

    def test_failing_runner_does_not_crash(self, synthetic_questions_path, output_dir):
        """A pipeline that raises should not crash the entire eval."""
        runners = _make_runners(error_versions={"v1"})
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1", "v2"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        # Should still get results for both versions
        assert len(report.results) == 12

    def test_failing_runner_error_non_null(self, synthetic_questions_path, output_dir):
        """A pipeline exception → error field is non-null."""
        runners = _make_runners(error_versions={"v2"})
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v2"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        error_results = [r for r in report.results if r.error is not None]
        assert len(error_results) == 6  # all 6 questions fail

    def test_failing_runner_grade_pass_false(self, synthetic_questions_path, output_dir):
        """A pipeline exception → grade.pass == False."""
        runners = _make_runners(error_versions={"v3"})
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v3"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        for r in report.results:
            assert r.grade["pass"] is False, f"Expected fail for {r.question_id}"

    def test_refusal_question_detected(self, synthetic_questions_path, output_dir):
        """The adversarial question with refusal answer gets pass=True."""
        # All versions return a clear refusal for the adversarial question
        runners = _make_runners(
            answer_map={
                v: "There was no such race on the 2020 Formula 1 calendar."
                for v in ["v1", "v2", "v3", "v4"]
            }
        )
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        adversarial_results = [
            r for r in report.results if r.question_id == "adversarial_001"
        ]
        assert len(adversarial_results) == 1
        assert adversarial_results[0].grade["pass"] is True

    def test_exact_match_question_pass(self, synthetic_questions_path, output_dir):
        """factual_stats question passes when answer contains accepted variant."""
        runners = _make_runners(
            answer_map={v: "The winner was Sebastian Vettel." for v in ["v1", "v2", "v3", "v4"]}
        )
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        fs_results = [r for r in report.results if r.question_id == "factual_stats_001"]
        assert fs_results[0].grade["pass"] is True

    def test_exact_match_question_fail(self, synthetic_questions_path, output_dir):
        """factual_stats question fails when answer is wrong."""
        runners = _make_runners(
            answer_map={v: "Lewis Hamilton took victory." for v in ["v1", "v2", "v3", "v4"]}
        )
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        fs_results = [r for r in report.results if r.question_id == "factual_stats_001"]
        assert fs_results[0].grade["pass"] is False

    def test_per_bucket_keys_match_question_buckets(
        self, synthetic_questions_path, output_dir
    ):
        """per_bucket keys in aggregates match the actual question buckets."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        expected_buckets = {
            "factual_stats",
            "multi_hop",
            "narrative",
            "regulation",
            "temporal_comparative",
            "adversarial",
        }
        assert set(report.aggregates["v1"].per_bucket.keys()) == expected_buckets

    def test_report_returned_in_memory(self, synthetic_questions_path, output_dir):
        """run_eval returns an EvalReport object, not just writes to disk."""
        runners = _make_runners()
        judge = _make_passing_judge_client()
        report = run_eval(
            versions=["v1"],
            questions_path=synthetic_questions_path,
            output_dir=output_dir,
            runners=runners,
            judge_client=judge,
        )
        assert isinstance(report, EvalReport)
        assert report.timestamp  # non-empty timestamp
