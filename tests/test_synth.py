from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.synth import (
    CITE_PATTERN,
    ContextChunk,
    Synthesizer,
    SynthResult,
    _format_context,
)


@pytest.fixture
def sample_contexts() -> list[ContextChunk]:
    return [
        ContextChunk(
            cite_id="A",
            source="stats",
            text="2021 Abu Dhabi GP — P1 Verstappen, P2 Hamilton, P3 Sainz.",
            metadata={"year": 2021},
        ),
        ContextChunk(
            cite_id="B",
            source="narratives",
            text="The race ended in controversy after Michael Masi allowed only some lapped cars to unlap.",
            metadata={"gp_name": "Abu Dhabi"},
        ),
    ]


def _mock_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_format_context_includes_cite_id_and_source(sample_contexts) -> None:
    formatted = _format_context(sample_contexts)
    assert "[A]" in formatted and "source=stats" in formatted
    assert "[B]" in formatted and "source=narratives" in formatted
    assert "Verstappen" in formatted
    assert "Masi" in formatted


def test_synth_returns_refusal_on_empty_contexts() -> None:
    synth = Synthesizer(client=MagicMock())
    result = synth.synthesize("anything", contexts=[])
    assert result.refused
    assert result.citations == []
    synth.client.messages.create.assert_not_called()


def test_synth_extracts_citations_in_order(sample_contexts) -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        "Verstappen won [cite:A]. The finish was controversial [cite:B]. Hamilton was P2 [cite:A]."
    )
    synth = Synthesizer(client=client)
    result = synth.synthesize("Who won?", sample_contexts)
    assert result.answer.startswith("Verstappen won")
    assert result.citations == ["A", "B"]  # ordered, deduped
    assert not result.refused


def test_synth_detects_refusal_language(sample_contexts) -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response(
        "I don't have enough information to answer that."
    )
    synth = Synthesizer(client=client)
    result = synth.synthesize("What's the weather?", sample_contexts)
    assert result.refused
    assert result.citations == []


def test_synth_invokes_correct_model(sample_contexts) -> None:
    client = MagicMock()
    client.messages.create.return_value = _mock_response("Anything [cite:A].")
    synth = Synthesizer(client=client, model="claude-sonnet-4-6")
    synth.synthesize("q", sample_contexts)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert "system" in kwargs
    assert kwargs["messages"][0]["role"] == "user"


def test_cite_pattern_matches_alphanum_underscore_dot() -> None:
    text = "Saw [cite:race__2021_22] and [cite:reg-C4.1] and [cite:A_B]."
    assert CITE_PATTERN.findall(text) == ["race__2021_22", "reg-C4.1", "A_B"]


def test_synth_result_serializable() -> None:
    r = SynthResult(answer="x", citations=["A"])
    dumped = r.model_dump()
    assert dumped["answer"] == "x"
    assert dumped["citations"] == ["A"]
    assert dumped["refused"] is False
