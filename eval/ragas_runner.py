"""
Thin RAGAS wrapper for computing faithfulness, context_precision,
context_recall, and answer_relevancy over a batch of RAG generations.

Usage::

    from eval.ragas_runner import run_ragas_metrics

    tuples = [
        ("Who won the 2017 Australian GP?", "Sebastian Vettel", "Sebastian Vettel won.", ["chunk text..."]),
        ...
    ]
    metrics = run_ragas_metrics(tuples)
    # → {"faithfulness": 0.85, "context_precision": 0.9, ...}  or None values on error

The function is tolerant of RAGAS being slow or raising init errors — it logs
and continues, returning None for any metric that fails.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Metric names we care about
_METRIC_NAMES = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def run_ragas_metrics(
    tuples: list[tuple[str, str, str, list[str]]],
    llm=None,
    embeddings=None,
) -> dict[str, float | None]:
    """
    Compute RAGAS metrics over a list of (question, gold_answer, answer, contexts) tuples.

    Parameters
    ----------
    tuples:
        Each element is (question, gold_answer, system_answer, list_of_context_strings).
    llm:
        Optional RAGAS-compatible LLM wrapper. If None, defaults to the RAGAS default
        (which requires OPENAI_API_KEY or similar env config).
    embeddings:
        Optional RAGAS-compatible embeddings wrapper.

    Returns
    -------
    dict mapping metric name → mean float score across all samples, or None if a metric
    failed or returned NaN.
    """
    if not tuples:
        logger.warning("ragas_runner: no samples provided — returning empty metrics")
        return {name: None for name in _METRIC_NAMES}

    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        logger.warning("ragas_runner: RAGAS not available (%s) — skipping metrics", exc)
        return {name: None for name in _METRIC_NAMES}

    # Build the RAGAS dataset
    samples: list[SingleTurnSample] = []
    for question, gold_answer, answer, contexts in tuples:
        samples.append(
            SingleTurnSample(
                user_input=question,
                reference=gold_answer,
                response=answer,
                retrieved_contexts=list(contexts),
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # Build metric instances — they require an LLM at construction time
    # in RAGAS 0.4+.  If no llm is provided we fall back to trying the
    # default (requires OPENAI_API_KEY), and log clearly if it fails.
    try:
        metric_instances: list[Any] = []

        if llm is not None:
            metric_instances = [
                Faithfulness(llm=llm),
                ContextPrecisionWithReference(llm=llm),
                ContextRecall(llm=llm),
                AnswerRelevancy(llm=llm),
            ]
        else:
            # RAGAS will try to use OpenAI by default; warn if key missing
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning(
                    "ragas_runner: OPENAI_API_KEY not set and no LLM provided — "
                    "RAGAS metrics may fail"
                )
            metric_instances = [
                Faithfulness(llm=None),
                ContextPrecisionWithReference(llm=None),
                ContextRecall(llm=None),
                AnswerRelevancy(llm=None),
            ]
    except Exception as exc:
        logger.warning("ragas_runner: could not instantiate metrics: %s", exc)
        return {name: None for name in _METRIC_NAMES}

    # Run evaluation
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metric_instances,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=False,
        )
    except Exception as exc:
        logger.warning("ragas_runner: evaluate() raised: %s", exc)
        return {name: None for name in _METRIC_NAMES}

    # Extract scalar scores — result is an EvaluationResult
    scores: dict[str, float | None] = {}
    result_dict = dict(result) if hasattr(result, "__iter__") else {}

    # Map friendly names to the keys RAGAS uses
    name_map = {
        "faithfulness": "faithfulness",
        "context_precision": "context_precision_with_reference",
        "context_recall": "context_recall",
        "answer_relevancy": "answer_relevancy",
    }

    for our_name, ragas_key in name_map.items():
        val = result_dict.get(ragas_key) or result_dict.get(our_name)
        if val is None:
            # Try numeric fallback via pandas
            try:
                import math

                df = result.to_pandas()
                col_candidates = [c for c in df.columns if ragas_key in c or our_name in c]
                if col_candidates:
                    col_val = df[col_candidates[0]].mean()
                    val = None if math.isnan(col_val) else float(col_val)
            except Exception:
                val = None
        if val is not None:
            try:
                import math

                val = None if math.isnan(float(val)) else float(val)
            except (TypeError, ValueError):
                val = None
        scores[our_name] = val

    return scores


def run_ragas_for_eval(
    version_results: dict[str, list[tuple[str, str, str, list[str]]]],
    llm=None,
    embeddings=None,
) -> dict[str, dict[str, float | None]]:
    """
    Run RAGAS metrics per version.

    Parameters
    ----------
    version_results:
        Mapping version_id → list of (question, gold_answer, answer, contexts).
    llm, embeddings:
        Optional RAGAS wrappers.

    Returns
    -------
    Mapping version_id → metric dict.
    """
    out: dict[str, dict[str, float | None]] = {}
    for version, tuples in version_results.items():
        logger.info("ragas_runner: computing metrics for version %s …", version)
        try:
            out[version] = run_ragas_metrics(tuples, llm=llm, embeddings=embeddings)
        except Exception as exc:
            logger.warning(
                "ragas_runner: unexpected error for version %s: %s", version, exc
            )
            out[version] = {name: None for name in _METRIC_NAMES}
    return out
