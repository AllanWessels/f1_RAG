"""Pydantic v2 schema for eval/questions.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Grading config — discriminated union on `kind`
# ---------------------------------------------------------------------------


class ExactMatchGrading(BaseModel):
    kind: Literal["exact_match"]
    accept: list[str]


class NumericGrading(BaseModel):
    kind: Literal["numeric"]
    value: float
    unit: str
    tolerance: float = 0.0
    cite_article: str | None = None


class LLMJudgeGrading(BaseModel):
    kind: Literal["llm_judge"]
    rubric: list[str]


class RefusalGrading(BaseModel):
    kind: Literal["refusal"]
    reason: Literal["no_such_race", "out_of_scope"]


GradingConfig = Annotated[
    ExactMatchGrading | NumericGrading | LLMJudgeGrading | RefusalGrading,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

BucketName = Literal[
    "factual_stats",
    "multi_hop",
    "narrative",
    "regulation",
    "temporal_comparative",
    "adversarial",
]

RetrieverName = Literal["query_stats", "search_narratives", "search_regulations"]


class Question(BaseModel):
    id: str
    bucket: BucketName
    question: str
    gold_answer: str
    grading: GradingConfig
    expected_retriever: RetrieverName
    notes: str | None = None


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


class QuestionSet(BaseModel):
    version: int
    questions: list[Question]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> QuestionSet:
    """Parse *path* as YAML and validate against QuestionSet."""
    raw = yaml.safe_load(path.read_text())
    return QuestionSet.model_validate(raw)
