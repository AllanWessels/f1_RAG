"""Tests for eval/questions.yaml and eval/schema.py."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from eval.schema import (
    ExactMatchGrading,
    LLMJudgeGrading,
    NumericGrading,
    QuestionSet,
    RefusalGrading,
    load_questions,
)

QUESTIONS_PATH = Path(__file__).parent.parent / "eval" / "questions.yaml"

EXPECTED_BUCKETS = {
    "factual_stats",
    "multi_hop",
    "narrative",
    "regulation",
    "temporal_comparative",
    "adversarial",
}

GRADING_KIND_TO_MODEL = {
    "exact_match": ExactMatchGrading,
    "numeric": NumericGrading,
    "llm_judge": LLMJudgeGrading,
    "refusal": RefusalGrading,
}


@pytest.fixture(scope="module")
def question_set() -> QuestionSet:
    """Load and validate the question set once for all tests in this module."""
    return load_questions(QUESTIONS_PATH)


class TestSchemaValidation:
    """The YAML file parses and validates against the Pydantic schema."""

    def test_loads_without_error(self) -> None:
        qs = load_questions(QUESTIONS_PATH)
        assert isinstance(qs, QuestionSet)

    def test_version_field(self, question_set: QuestionSet) -> None:
        assert question_set.version == 1


class TestQuestionCount:
    """Exactly 30 questions, 5 per bucket."""

    def test_total_count(self, question_set: QuestionSet) -> None:
        assert len(question_set.questions) == 30

    def test_five_per_bucket(self, question_set: QuestionSet) -> None:
        from collections import Counter

        counts = Counter(q.bucket for q in question_set.questions)
        for bucket in EXPECTED_BUCKETS:
            assert counts[bucket] == 5, (
                f"Expected 5 questions for bucket '{bucket}', got {counts[bucket]}"
            )


class TestBucketPresence:
    """All 6 bucket values are present."""

    def test_all_buckets_present(self, question_set: QuestionSet) -> None:
        present = {q.bucket for q in question_set.questions}
        assert present == EXPECTED_BUCKETS


class TestUniqueIDs:
    """No duplicate question IDs."""

    def test_no_duplicate_ids(self, question_set: QuestionSet) -> None:
        ids = [q.id for q in question_set.questions]
        duplicates = [qid for qid in ids if ids.count(qid) > 1]
        assert duplicates == [], f"Duplicate IDs found: {duplicates}"


class TestGradingDiscriminator:
    """Each question's grading config matches its `kind` discriminator."""

    def test_grading_type_matches_kind(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            expected_cls = GRADING_KIND_TO_MODEL[q.grading.kind]
            assert isinstance(q.grading, expected_cls), (
                f"Question '{q.id}': grading.kind='{q.grading.kind}' but "
                f"grading object is {type(q.grading).__name__}"
            )

    def test_exact_match_has_accept_list(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            if q.grading.kind == "exact_match":
                assert isinstance(q.grading, ExactMatchGrading)
                assert len(q.grading.accept) >= 1, (
                    f"Question '{q.id}': exact_match grading must have at least one accept value"
                )

    def test_numeric_has_value_and_unit(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            if q.grading.kind == "numeric":
                assert isinstance(q.grading, NumericGrading)
                assert q.grading.unit, f"Question '{q.id}': numeric grading missing unit"

    def test_llm_judge_has_rubric(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            if q.grading.kind == "llm_judge":
                assert isinstance(q.grading, LLMJudgeGrading)
                assert len(q.grading.rubric) >= 1, (
                    f"Question '{q.id}': llm_judge grading must have at least one rubric item"
                )

    def test_refusal_has_reason(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            if q.grading.kind == "refusal":
                assert isinstance(q.grading, RefusalGrading)
                assert q.grading.reason in {"no_such_race", "out_of_scope"}, (
                    f"Question '{q.id}': refusal reason must be 'no_such_race' or 'out_of_scope'"
                )


class TestRetrieverAssignment:
    """Every question has a valid expected_retriever."""

    VALID_RETRIEVERS: ClassVar[set[str]] = {"query_stats", "search_narratives", "search_regulations"}

    def test_all_retrievers_valid(self, question_set: QuestionSet) -> None:
        for q in question_set.questions:
            assert q.expected_retriever in self.VALID_RETRIEVERS, (
                f"Question '{q.id}': invalid expected_retriever '{q.expected_retriever}'"
            )


class TestBucketRetrieverAlignment:
    """Sanity-check: regulation questions use search_regulations; adversarial/stats use query_stats."""

    def test_regulation_questions_use_regulation_retriever(
        self, question_set: QuestionSet
    ) -> None:
        for q in question_set.questions:
            if q.bucket == "regulation":
                assert q.expected_retriever == "search_regulations", (
                    f"Question '{q.id}': regulation question should use search_regulations, "
                    f"got '{q.expected_retriever}'"
                )

    def test_narrative_questions_use_narrative_retriever(
        self, question_set: QuestionSet
    ) -> None:
        for q in question_set.questions:
            if q.bucket == "narrative":
                assert q.expected_retriever == "search_narratives", (
                    f"Question '{q.id}': narrative question should use search_narratives, "
                    f"got '{q.expected_retriever}'"
                )
