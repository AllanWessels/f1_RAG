"""
Eval runner — runs all pipeline versions x all questions, grades results,
aggregates per-version per-bucket statistics, and materialises results to JSON.

CLI:
    python -m eval.runner               # all 4 versions, 30 questions
    python -m eval.runner --versions v1 v3
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.pipeline import PipelineResult
from eval.graders import GradeResult, grade
from eval.schema import Question, QuestionSet, load_questions

logger = logging.getLogger(__name__)

VersionId = Literal["v1", "v2", "v3", "v4"]
ALL_VERSIONS: list[VersionId] = ["v1", "v2", "v3", "v4"]

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class RunError(BaseModel):
    """Captured exception from a pipeline run."""

    exc_type: str
    message: str
    traceback: str


class QuestionResult(BaseModel):
    question_id: str
    bucket: str
    version: str
    question: str
    gold_answer: str
    answer: str
    citations: list[str]
    refused: bool
    grade: dict  # GradeResult.to_dict()
    tool_calls: list[dict]
    trace_url: str | None = None
    error: str | None = None


class VersionAggregate(BaseModel):
    overall_pass_rate: float
    overall_mean_score: float
    per_bucket: dict[str, float]  # bucket → pass_rate


class EvalReport(BaseModel):
    timestamp: str
    versions: list[str]
    questions: list[str]  # question IDs
    results: list[QuestionResult]
    aggregates: dict[str, VersionAggregate]
    ragas: dict[str, dict[str, float | None]] | None = None


# ---------------------------------------------------------------------------
# Default pipeline runners (imported lazily to avoid heavy deps at import time)
# ---------------------------------------------------------------------------


def _default_runners() -> dict[str, Callable]:
    from app.pipelines.v1_naive import run_v1
    from app.pipelines.v2_hybrid import run_v2
    from app.pipelines.v3_rerank import run_v3
    from app.pipelines.v4_router import run_v4

    return {"v1": run_v1, "v2": run_v2, "v3": run_v3, "v4": run_v4}


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def _aggregate(results: list[QuestionResult]) -> VersionAggregate:
    """Compute pass_rates from a list of results (all same version)."""
    if not results:
        return VersionAggregate(
            overall_pass_rate=0.0, overall_mean_score=0.0, per_bucket={}
        )

    passes = [r.grade.get("pass", False) for r in results]
    scores = [r.grade.get("score", 0.0) for r in results]
    overall_pass = sum(passes) / len(passes)
    overall_score = sum(scores) / len(scores)

    # Per-bucket
    buckets: dict[str, list[bool]] = {}
    for r in results:
        buckets.setdefault(r.bucket, []).append(r.grade.get("pass", False))

    per_bucket = {b: sum(ps) / len(ps) for b, ps in buckets.items()}

    return VersionAggregate(
        overall_pass_rate=overall_pass,
        overall_mean_score=overall_score,
        per_bucket=per_bucket,
    )


# ---------------------------------------------------------------------------
# Core eval function
# ---------------------------------------------------------------------------


def run_eval(
    versions: list[VersionId] | None = None,
    questions_path: Path = Path("eval/questions.yaml"),
    output_dir: Path = Path("eval/results"),
    *,
    runners: dict[str, Callable] | None = None,
    judge_client=None,
) -> EvalReport:
    """
    Run the full eval harness.

    Parameters
    ----------
    versions:
        Which pipeline versions to evaluate. Defaults to all 4.
    questions_path:
        Path to questions.yaml.
    output_dir:
        Directory to write timestamped JSON results to.
    runners:
        Override pipeline callables (for testing). Keys are version IDs.
    judge_client:
        Override anthropic.Anthropic client for LLM judge (for testing).
    """
    if versions is None:
        versions = list(ALL_VERSIONS)

    # Resolve runners
    effective_runners = runners if runners is not None else _default_runners()

    # Load questions
    question_set: QuestionSet = load_questions(questions_path)
    all_questions: list[Question] = question_set.questions
    logger.info("Loaded %d questions from %s", len(all_questions), questions_path)

    results: list[QuestionResult] = []

    for version in versions:
        run_fn = effective_runners.get(version)
        if run_fn is None:
            logger.warning("No runner for version %s — skipping", version)
            continue

        logger.info("Running version %s over %d questions …", version, len(all_questions))

        for question in all_questions:
            error_str: str | None = None
            pipeline_result: PipelineResult | None = None
            answer = ""
            citations: list[str] = []
            refused = False
            tool_calls: list[dict] = []

            # ---- call pipeline ----
            try:
                pipeline_result = run_fn(query=question.question)
                answer = pipeline_result.answer
                citations = pipeline_result.citations
                refused = pipeline_result.refused
                tool_calls = [tc.model_dump() for tc in pipeline_result.tool_calls]
            except Exception as exc:
                logger.error(
                    "Pipeline %s raised on %s: %s", version, question.id, exc
                )
                error_str = f"{type(exc).__name__}: {exc}"
                # Synthesise a dummy PipelineResult so we can still grade
                from app.pipeline import PipelineResult as PR

                pipeline_result = PR(
                    version=version,  # type: ignore[arg-type]
                    query=question.question,
                    answer="",
                    refused=False,
                )

            # ---- grade ----
            try:
                grade_result: GradeResult = grade(question, pipeline_result, judge_client)
            except Exception as exc:
                logger.error(
                    "Grader raised on %s/%s: %s", version, question.id, exc
                )
                grade_result = GradeResult(
                    pass_=False,
                    score=0.0,
                    detail=f"grader error: {exc}",
                )
                if error_str is None:
                    error_str = f"grader error: {type(exc).__name__}: {exc}"

            # If there was a run error, override grade to fail
            if error_str is not None and pipeline_result.answer == "":
                grade_result = GradeResult(
                    pass_=False,
                    score=0.0,
                    detail=f"run error: {error_str}",
                )

            results.append(
                QuestionResult(
                    question_id=question.id,
                    bucket=question.bucket,
                    version=version,
                    question=question.question,
                    gold_answer=question.gold_answer,
                    answer=answer,
                    citations=citations,
                    refused=refused,
                    grade=grade_result.to_dict(),
                    tool_calls=tool_calls,
                    trace_url=None,
                    error=error_str,
                )
            )

    # ---- aggregate per version ----
    aggregates: dict[str, VersionAggregate] = {}
    for version in versions:
        version_results = [r for r in results if r.version == version]
        aggregates[version] = _aggregate(version_results)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    report = EvalReport(
        timestamp=timestamp,
        versions=list(versions),
        questions=[q.id for q in all_questions],
        results=results,
        aggregates=aggregates,
        ragas=None,
    )

    # ---- persist ----
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = timestamp.replace(":", "-")
    out_path = output_dir / f"{ts_safe}.json"
    latest_path = output_dir / "latest.json"

    payload = report.model_dump()
    # Serialise aggregates (Pydantic models → dicts already via model_dump)
    json_str = json.dumps(payload, indent=2, default=str)
    out_path.write_text(json_str)
    # latest.json is a regular copy (not a symlink — portable on Docker volumes)
    shutil.copy2(out_path, latest_path)

    logger.info("Results written to %s (+ latest.json)", out_path)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Run F1 RAG eval harness")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["v1", "v2", "v3", "v4"],
        default=["v1", "v2", "v3", "v4"],
        help="Pipeline versions to evaluate",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions.yaml"),
        help="Path to questions YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to write results JSON",
    )
    args = parser.parse_args()

    report = run_eval(
        versions=args.versions,
        questions_path=args.questions,
        output_dir=args.output_dir,
    )

    print(f"\nEval complete — {len(report.results)} results")
    for version, agg in report.aggregates.items():
        print(
            f"  {version}: pass_rate={agg.overall_pass_rate:.1%}"
            f"  mean_score={agg.overall_mean_score:.2f}"
        )


if __name__ == "__main__":
    main()
