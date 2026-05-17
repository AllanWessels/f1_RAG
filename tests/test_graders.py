"""
Tests for eval/graders.py.

Covers:
  - ExactMatchGrader: matches normalised variants, rejects misses, handles accents/case
  - NumericGrader: number extraction, unit normalisation, tolerance, cite_article
  - LLMJudgeGrader: mocked anthropic client, JSON parsing, score threshold logic
  - RefusalGrader: refusal phrase detection + PipelineResult.refused=True
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.pipeline import PipelineResult
from eval.graders import (
    ExactMatchGrader,
    LLMJudgeGrader,
    NumericGrader,
    RefusalGrader,
    grade,
)
from eval.schema import (
    ExactMatchGrading,
    LLMJudgeGrading,
    NumericGrading,
    Question,
    RefusalGrading,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_result(answer: str = "", refused: bool = False) -> PipelineResult:
    return PipelineResult(
        version="v1",
        query="test question",
        answer=answer,
        refused=refused,
    )


def _make_question(grading, question_text: str = "Test question?") -> Question:
    return Question(
        id="test_001",
        bucket="factual_stats",
        question=question_text,
        gold_answer="gold",
        grading=grading,
        expected_retriever="query_stats",
    )


# ---------------------------------------------------------------------------
# ExactMatchGrader
# ---------------------------------------------------------------------------


class TestExactMatchGrader:
    def test_exact_match_simple(self):
        grader = ExactMatchGrader(["Sebastian Vettel", "Vettel"])
        result = grader.grade("Sebastian Vettel won the race.")
        assert result.pass_ is True
        assert result.score == 1.0
        assert "Vettel" in result.detail

    def test_exact_match_short_variant(self):
        grader = ExactMatchGrader(["Sebastian Vettel", "Vettel"])
        result = grader.grade("The winner was Vettel from Ferrari.")
        assert result.pass_ is True

    def test_case_insensitive(self):
        grader = ExactMatchGrader(["Sebastian Vettel"])
        result = grader.grade("SEBASTIAN VETTEL drove brilliantly.")
        assert result.pass_ is True

    def test_accent_stripping(self):
        # "Räikkönen" normalised → "raikkonen"
        grader = ExactMatchGrader(["Räikkönen", "Raikkonen"])
        result = grader.grade("Kimi Raikkonen was the winner.")
        assert result.pass_ is True

    def test_accent_in_answer(self):
        grader = ExactMatchGrader(["Pérez", "Perez"])
        result = grader.grade("Sergio Pérez finished second.")
        assert result.pass_ is True

    def test_punctuation_stripped(self):
        grader = ExactMatchGrader(["Mercedes"])
        result = grader.grade("The winner: Mercedes-AMG Petronas!")
        assert result.pass_ is True

    def test_miss(self):
        grader = ExactMatchGrader(["Sebastian Vettel", "Vettel"])
        result = grader.grade("Lewis Hamilton took the chequered flag.")
        assert result.pass_ is False
        assert result.score == 0.0
        assert "no match" in result.detail

    def test_numeric_exact_match(self):
        grader = ExactMatchGrader(["19", "nineteen"])
        result = grader.grade("Verstappen scored 19 wins in 2023.")
        assert result.pass_ is True

    def test_numeric_word_form(self):
        grader = ExactMatchGrader(["19", "nineteen"])
        result = grader.grade("He won nineteen races that season.")
        assert result.pass_ is True

    def test_empty_answer(self):
        grader = ExactMatchGrader(["Mercedes"])
        result = grader.grade("")
        assert result.pass_ is False

    def test_multiple_accepts_first_found(self):
        grader = ExactMatchGrader(["Sergio Perez", "Perez", "Checo"])
        result = grader.grade("Checo was the teammate.")
        assert result.pass_ is True


# ---------------------------------------------------------------------------
# NumericGrader
# ---------------------------------------------------------------------------


class TestNumericGrader:
    def test_basic_kg(self):
        grader = NumericGrader(value=768, unit="kg")
        result = grader.grade("The minimum weight is 768 kg.")
        assert result.pass_ is True
        assert "768" in result.detail

    def test_kg_no_space(self):
        grader = NumericGrader(value=768, unit="kg")
        result = grader.grade("Minimum weight: 768kg.")
        assert result.pass_ is True

    def test_kilograms_alias(self):
        grader = NumericGrader(value=768, unit="kg")
        result = grader.grade("The car must weigh at least 768 kilograms.")
        assert result.pass_ is True

    def test_wrong_value(self):
        grader = NumericGrader(value=768, unit="kg")
        result = grader.grade("The minimum weight is 800 kg.")
        assert result.pass_ is False
        assert result.score == 0.0

    def test_tolerance(self):
        grader = NumericGrader(value=768, unit="kg", tolerance=5)
        result = grader.grade("The weight is approximately 770 kg.")
        assert result.pass_ is True

    def test_tolerance_exceeded(self):
        grader = NumericGrader(value=768, unit="kg", tolerance=5)
        result = grader.grade("The weight is 800 kg.")
        assert result.pass_ is False

    def test_zero_tolerance_exact(self):
        grader = NumericGrader(value=4, unit="ICEs", tolerance=0)
        result = grader.grade("A driver may use up to 4 ICEs before a penalty.")
        assert result.pass_ is True

    def test_usd_millions(self):
        grader = NumericGrader(value=135_000_000, unit="USD", tolerance=0)
        result = grader.grade("The cost cap is $135 million for 2026.")
        assert result.pass_ is True

    def test_cite_article_present(self):
        grader = NumericGrader(
            value=768,
            unit="kg",
            tolerance=0,
            cite_article="Technical Regulations Article 4.1",
        )
        result = grader.grade(
            "The minimum weight is 768 kg per Technical Regulations Article 4.1."
        )
        assert result.pass_ is True

    def test_cite_article_missing(self):
        grader = NumericGrader(
            value=768,
            unit="kg",
            tolerance=0,
            cite_article="Technical Regulations Article 4.1",
        )
        result = grader.grade("The minimum weight is 768 kg.")
        assert result.pass_ is False
        assert result.score == 0.5  # partial credit
        assert "citation" in result.detail

    def test_cite_article_in_citations_list(self):
        grader = NumericGrader(
            value=768,
            unit="kg",
            tolerance=0,
            cite_article="Article 4.1",
        )
        result = grader.grade("768 kg is required.", citations=["Technical Regulations Article 4.1"])
        assert result.pass_ is True

    def test_no_number_in_answer(self):
        grader = NumericGrader(value=768, unit="kg")
        result = grader.grade("I don't know the minimum weight.")
        assert result.pass_ is False

    def test_unit_agnostic_fallback(self):
        # If the answer has the right number but a different or no unit, still pass
        grader = NumericGrader(value=4, unit="ICEs")
        result = grader.grade("The limit is 4 per season.")
        assert result.pass_ is True


# ---------------------------------------------------------------------------
# LLMJudgeGrader
# ---------------------------------------------------------------------------


def _make_mock_client(json_payload: str):
    """Return a mock anthropic.Anthropic client that returns the given JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json_payload
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestLLMJudgeGrader:
    def test_all_satisfied_passes(self):
        rubric = ["Mentions the safety car", "Mentions Verstappen overtaking Hamilton"]
        payload = '{"satisfied": [0, 1], "missing": [], "score": 1.0, "reasoning": "Great answer."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Test question?", "Some answer about safety car and Verstappen.")
        assert result.pass_ is True
        assert result.score == 1.0

    def test_partial_score_below_threshold_fails(self):
        rubric = ["A", "B", "C", "D", "E"]
        payload = '{"satisfied": [0], "missing": [1, 2, 3, 4], "score": 0.2, "reasoning": "Only one satisfied."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Question?", "Poor answer.")
        assert result.pass_ is False
        assert result.score == pytest.approx(0.2)

    def test_score_exactly_at_threshold_passes(self):
        rubric = ["A", "B", "C"]
        payload = '{"satisfied": [0, 1], "missing": [2], "score": 0.6, "reasoning": "Two of three."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Question?", "Answer hitting 2 of 3 bullets.")
        assert result.pass_ is True

    def test_score_just_below_threshold_fails(self):
        rubric = ["A", "B", "C"]
        payload = '{"satisfied": [0], "missing": [1, 2], "score": 0.33, "reasoning": "Only one."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Question?", "Weak answer.")
        assert result.pass_ is False

    def test_json_parsing_correct(self):
        rubric = ["A"]
        payload = '{"satisfied": [0], "missing": [], "score": 1.0, "reasoning": "Perfect."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Q?", "Perfect answer.")
        assert "satisfied=[0]" in result.detail

    def test_markdown_fence_stripped(self):
        rubric = ["A"]
        payload = '```json\n{"satisfied": [0], "missing": [], "score": 1.0, "reasoning": "OK."}\n```'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Q?", "Good answer.")
        assert result.pass_ is True

    def test_malformed_json_returns_failure(self):
        rubric = ["A"]
        grader = LLMJudgeGrader(rubric, client=_make_mock_client("NOT JSON"))
        result = grader.grade("Q?", "Answer.")
        assert result.pass_ is False
        assert "error" in result.detail.lower()

    def test_reasoning_in_detail(self):
        rubric = ["Mentions crash"]
        payload = '{"satisfied": [0], "missing": [], "score": 1.0, "reasoning": "The crash is mentioned."}'
        grader = LLMJudgeGrader(rubric, client=_make_mock_client(payload))
        result = grader.grade("Q?", "There was a crash.")
        assert "crash is mentioned" in result.detail


# ---------------------------------------------------------------------------
# RefusalGrader
# ---------------------------------------------------------------------------


class TestRefusalGrader:
    def test_refused_flag_true(self):
        grader = RefusalGrader("no_such_race")
        pr = _make_pipeline_result(answer="Some answer", refused=True)
        result = grader.grade(pr)
        assert result.pass_ is True
        assert "refused=True" in result.detail

    def test_refused_flag_false_no_pattern(self):
        grader = RefusalGrader("no_such_race")
        pr = _make_pipeline_result(answer="Lewis Hamilton won the race.", refused=False)
        result = grader.grade(pr)
        assert result.pass_ is False

    def test_no_such_race_pattern(self):
        grader = RefusalGrader("no_such_race")
        pr = _make_pipeline_result(
            answer="There was no such race on the 2020 calendar.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_no_such_grand_prix_pattern(self):
        grader = RefusalGrader("no_such_race")
        pr = _make_pipeline_result(
            answer="There was no such Grand Prix in 2020.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_out_of_scope_pattern(self):
        grader = RefusalGrader("out_of_scope")
        pr = _make_pipeline_result(
            answer="This information is not available in the data sources.",
            refused=False,
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_i_dont_have_pattern(self):
        grader = RefusalGrader("out_of_scope")
        pr = _make_pipeline_result(
            answer="I don't have enough information to answer that.",
            refused=False,
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_cannot_provide_pattern(self):
        grader = RefusalGrader("out_of_scope")
        pr = _make_pipeline_result(
            answer="I cannot provide telemetry data.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_data_not_cover_pattern(self):
        grader = RefusalGrader("out_of_scope")
        pr = _make_pipeline_result(
            answer="The data does not cover pre-2014 races.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_outside_scope_pattern(self):
        grader = RefusalGrader("out_of_scope")
        pr = _make_pipeline_result(
            answer="This falls outside the coverage of my data sources.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is True

    def test_false_answer_no_refusal(self):
        grader = RefusalGrader("no_such_race")
        pr = _make_pipeline_result(
            answer="Lewis Hamilton won the 2020 European Grand Prix.", refused=False
        )
        result = grader.grade(pr)
        assert result.pass_ is False
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Dispatch function: grade()
# ---------------------------------------------------------------------------


class TestGradeDispatch:
    def test_dispatch_exact_match(self):
        grading = ExactMatchGrading(kind="exact_match", accept=["Vettel"])
        q = _make_question(grading)
        pr = _make_pipeline_result("Sebastian Vettel won.")
        result = grade(q, pr)
        assert result.pass_ is True

    def test_dispatch_numeric(self):
        grading = NumericGrading(kind="numeric", value=768, unit="kg", tolerance=0)
        q = _make_question(grading)
        pr = _make_pipeline_result("The minimum weight is 768 kg.")
        result = grade(q, pr)
        assert result.pass_ is True

    def test_dispatch_refusal(self):
        grading = RefusalGrading(kind="refusal", reason="no_such_race")
        q = _make_question(grading)
        pr = _make_pipeline_result("There was no such race.", refused=False)
        result = grade(q, pr)
        assert result.pass_ is True

    def test_dispatch_llm_judge(self):
        rubric = ["States 2016", "States 9 wins"]
        payload = '{"satisfied": [0, 1], "missing": [], "score": 1.0, "reasoning": "Both covered."}'
        mock_client = _make_mock_client(payload)
        grading = LLMJudgeGrading(kind="llm_judge", rubric=rubric)
        q = _make_question(grading, "When did Rosberg win and how many wins?")
        pr = _make_pipeline_result("Rosberg won in 2016 with 9 wins.")
        result = grade(q, pr, judge_client=mock_client)
        assert result.pass_ is True
