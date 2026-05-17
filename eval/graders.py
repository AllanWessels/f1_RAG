"""
Graders for the F1 RAG eval harness.

Four grader types:
  - ExactMatchGrader  — normalized substring match against accepted variants
  - NumericGrader     — extract a number from the answer, compare with tolerance + optional citation check
  - LLMJudgeGrader    — Claude Sonnet 4.6 structured rubric scoring
  - RefusalGrader     — detects abstention / refusal

Dispatch function `grade(question, pipeline_result) -> GradeResult` reads
`question.grading.kind` and instantiates the right grader.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass

import anthropic

from app.pipeline import PipelineResult
from eval.schema import (
    ExactMatchGrading,
    LLMJudgeGrading,
    NumericGrading,
    Question,
    RefusalGrading,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GradeResult
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    pass_: bool  # 'pass' is a reserved keyword
    score: float  # 0.0 - 1.0
    detail: str

    def to_dict(self) -> dict:
        return {"pass": self.pass_, "score": self.score, "detail": self.detail}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _normalise(text: str) -> str:
    """Lowercase, strip diacritics, strip punctuation, collapse whitespace."""
    # NFKD decomposition → strip combining characters (diacritics)
    nfkd = unicodedata.normalize("NFKD", text)
    no_diacritics = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = no_diacritics.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    collapsed = " ".join(no_punct.split())
    return collapsed


# ---------------------------------------------------------------------------
# Unit normalisation for NumericGrader
# ---------------------------------------------------------------------------

# Map aliases → canonical unit string (lowercase)
_UNIT_ALIASES: dict[str, str] = {
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilo": "kg",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "$": "usd",
    "m": "million",
    "million": "million",
    "millions": "million",
    "b": "billion",
    "billion": "billion",
    "ices": "ices",
    "ice": "ices",
    "km": "km",
    "kilometre": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometers": "km",
    "litre": "l",
    "litres": "l",
    "liter": "l",
    "liters": "l",
    "l": "l",
}


def _canonical_unit(unit: str) -> str:
    return _UNIT_ALIASES.get(unit.lower().strip(), unit.lower().strip())


# Regex to find a number (int or float, with optional commas) optionally
# followed by a unit-like word.
_NUMBER_RE = re.compile(
    r"(?<!\w)"  # not preceded by word char
    r"(\d[\d,]*(?:\.\d+)?)"  # number with optional commas and decimal
    r"(?:\s*([a-zA-Z$]+))?"  # optional unit
    r"(?!\w)",  # not followed by word char
    re.IGNORECASE,
)

# Scale suffixes embedded in the number itself (e.g. "$135 million")
_SCALE_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(million|billion|m|b)\b", re.IGNORECASE)
_SCALE_MAP = {"million": 1_000_000, "m": 1_000_000, "billion": 1_000_000_000, "b": 1_000_000_000}


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """
    Return list of (value, canonical_unit) pairs found in *text*.

    Handles:
      - "768 kg", "768kg", "768 kilograms"
      - "$135 million" → (135_000_000, "usd")
      - "4 ICEs"
    """
    results: list[tuple[float, str]] = []

    # First pass: scaled numbers like "$135 million"
    for m in _SCALE_RE.finditer(text):
        raw = float(m.group(1).replace(",", ""))
        scale_word = m.group(2).lower()
        scale = _SCALE_MAP.get(scale_word, 1)
        value = raw * scale
        # look for currency symbol just before
        start = m.start()
        prefix = text[max(0, start - 2) : start].strip()
        unit = "usd" if "$" in prefix else "million"
        results.append((value, unit))

    # Second pass: plain numbers
    for m in _NUMBER_RE.finditer(text):
        raw_num = float(m.group(1).replace(",", ""))
        raw_unit = m.group(2) or ""
        canonical = _canonical_unit(raw_unit) if raw_unit else ""
        # skip if already captured by scale pass (approximate match)
        if not any(abs(v - raw_num) < 1e-6 * max(1, abs(v)) for v, _ in results):
            results.append((raw_num, canonical))

    return results


# ---------------------------------------------------------------------------
# ExactMatchGrader
# ---------------------------------------------------------------------------


class ExactMatchGrader:
    """Pass if the normalised answer contains any accepted variant."""

    def __init__(self, accept: list[str]) -> None:
        self.accept = accept
        self._normalised_accept = [_normalise(a) for a in accept]

    def grade(self, answer: str) -> GradeResult:
        norm_answer = _normalise(answer)
        for variant, norm_variant in zip(self.accept, self._normalised_accept, strict=True):
            if norm_variant in norm_answer:
                return GradeResult(
                    pass_=True,
                    score=1.0,
                    detail=f"exact match: {variant}",
                )
        return GradeResult(
            pass_=False,
            score=0.0,
            detail=f"no match found; accepted: {self.accept}",
        )


# ---------------------------------------------------------------------------
# NumericGrader
# ---------------------------------------------------------------------------


class NumericGrader:
    """
    Extract a numeric value from the answer and compare with the gold value.

    If *cite_article* is set, also verify that the answer references that
    article string (case-insensitive substring match).
    """

    def __init__(
        self,
        value: float,
        unit: str,
        tolerance: float = 0.0,
        cite_article: str | None = None,
    ) -> None:
        self.gold_value = value
        self.gold_unit = _canonical_unit(unit)
        self.tolerance = tolerance
        self.cite_article = cite_article

    def grade(self, answer: str, citations: list[str] | None = None) -> GradeResult:
        # 1. Extract numbers
        pairs = _extract_numbers(answer)

        matched_value: float | None = None
        for val, unit in pairs:
            # Check unit if gold unit is meaningful
            if self.gold_unit and unit and unit != self.gold_unit:
                continue
            # Check numeric match within tolerance
            if self.tolerance == 0:
                if abs(val - self.gold_value) < 1e-6:
                    matched_value = val
                    break
            else:
                if abs(val - self.gold_value) <= self.tolerance:
                    matched_value = val
                    break

        if matched_value is None:
            # Last-chance: try ignoring unit (some answers omit it)
            for val, _ in pairs:
                if self.tolerance == 0:
                    if abs(val - self.gold_value) < 1e-6:
                        matched_value = val
                        break
                else:
                    if abs(val - self.gold_value) <= self.tolerance:
                        matched_value = val
                        break

        if matched_value is None:
            return GradeResult(
                pass_=False,
                score=0.0,
                detail=f"numeric {self.gold_value} {self.gold_unit} not found; extracted={pairs}",
            )

        # 2. Citation check
        if self.cite_article:
            cite_lower = self.cite_article.lower()
            answer_lower = answer.lower()
            citation_hit = cite_lower in answer_lower or any(
                cite_lower in c.lower() for c in (citations or [])
            )
            if not citation_hit:
                return GradeResult(
                    pass_=False,
                    score=0.5,
                    detail=(
                        f"numeric match ({matched_value}) OK but missing citation"
                        f" '{self.cite_article}'"
                    ),
                )

        return GradeResult(
            pass_=True,
            score=1.0,
            detail=f"numeric match: {matched_value} {self.gold_unit}",
        )


# ---------------------------------------------------------------------------
# LLMJudgeGrader
# ---------------------------------------------------------------------------

_JUDGE_MODEL = "claude-sonnet-4-6"

_JUDGE_SYSTEM = """\
You are an impartial grader for a Formula 1 RAG system. You will receive:
1. The original question.
2. A list of gold rubric bullets that a correct answer should satisfy.
3. The system's answer to grade.

Respond with ONLY a JSON object (no markdown fences) with exactly these keys:
{
  "satisfied": [list of 0-based bullet indexes that the answer satisfies],
  "missing":   [list of 0-based bullet indexes the answer fails to satisfy],
  "score":     float between 0.0 and 1.0 (fraction of bullets satisfied),
  "reasoning": "one or two sentence explanation"
}

Be strict but fair. Partial credit is not allowed per bullet — a bullet is either satisfied or not.
"""


class LLMJudgeGrader:
    """Call Claude Sonnet 4.6 to score an answer against a rubric."""

    PASS_THRESHOLD = 0.6

    def __init__(
        self,
        rubric: list[str],
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.rubric = rubric
        self.client = client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def grade(self, question: str, answer: str) -> GradeResult:
        rubric_text = "\n".join(
            f"  [{i}] {bullet}" for i, bullet in enumerate(self.rubric)
        )
        user_msg = (
            f"Question: {question}\n\n"
            f"Rubric bullets:\n{rubric_text}\n\n"
            f"System answer:\n{answer}"
        )

        try:
            response = self.client.messages.create(
                model=_JUDGE_MODEL,
                max_tokens=512,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if the model adds them despite instructions
            if raw.startswith("```"):
                raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
            parsed = json.loads(raw)

            satisfied = parsed.get("satisfied", [])
            missing = parsed.get("missing", [])
            score = float(parsed.get("score", len(satisfied) / max(len(self.rubric), 1)))
            reasoning = parsed.get("reasoning", "")

            pass_ = score >= self.PASS_THRESHOLD
            detail = (
                f"score={score:.2f}; satisfied={satisfied}; missing={missing}; {reasoning}"
            )
            return GradeResult(pass_=pass_, score=score, detail=detail)

        except (json.JSONDecodeError, KeyError, IndexError, Exception) as exc:
            logger.warning("LLMJudgeGrader: error calling/parsing judge: %s", exc)
            return GradeResult(
                pass_=False,
                score=0.0,
                detail=f"judge error: {exc}",
            )


# ---------------------------------------------------------------------------
# RefusalGrader
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = re.compile(
    r"i don'?t have|i do not have|no such (race|event|grand prix)|"
    r"did not (take place|happen|exist)|was not held|out of scope|"
    r"not available|cannot (provide|answer|access)|can'?t (provide|answer|access)|"
    r"outside (the|my) (coverage|scope|data)|not in (the|my) (data|corpus|scope)|"
    r"i'?m unable|does not exist|never (took place|happened)|"
    r"not (on )?the (2020|2021|2022|2023|2024) (f1 |formula.?1 )?calendar|"
    r"premise is (wrong|incorrect)|data (does )?not cover|no (data|information) (on|about)",
    re.IGNORECASE,
)


class RefusalGrader:
    """Pass if the answer is a clear refusal / abstention."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def grade(self, pipeline_result: PipelineResult) -> GradeResult:
        # Explicit refused flag set by the pipeline
        if pipeline_result.refused:
            return GradeResult(
                pass_=True,
                score=1.0,
                detail="PipelineResult.refused=True",
            )

        # Keyword / pattern match in the answer text
        if _REFUSAL_PATTERNS.search(pipeline_result.answer):
            return GradeResult(
                pass_=True,
                score=1.0,
                detail=f"refusal pattern matched (reason={self.reason})",
            )

        return GradeResult(
            pass_=False,
            score=0.0,
            detail="answer does not appear to refuse or abstain",
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def grade(question: Question, pipeline_result: PipelineResult, judge_client=None) -> GradeResult:
    """
    Dispatch to the correct grader based on *question.grading.kind*.

    *judge_client* may be an anthropic.Anthropic instance for test injection.
    """
    grading = question.grading
    answer = pipeline_result.answer
    citations = pipeline_result.citations

    if isinstance(grading, ExactMatchGrading):
        return ExactMatchGrader(grading.accept).grade(answer)

    if isinstance(grading, NumericGrading):
        return NumericGrader(
            value=grading.value,
            unit=grading.unit,
            tolerance=grading.tolerance,
            cite_article=grading.cite_article,
        ).grade(answer, citations)

    if isinstance(grading, LLMJudgeGrading):
        return LLMJudgeGrader(grading.rubric, client=judge_client).grade(
            question.question, answer
        )

    if isinstance(grading, RefusalGrading):
        return RefusalGrader(grading.reason).grade(pipeline_result)

    raise ValueError(f"Unknown grading kind: {grading.kind}")
