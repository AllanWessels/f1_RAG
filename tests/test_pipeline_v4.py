"""
Unit tests for the v4 router pipeline (RouterPipeline).

All external dependencies — Anthropic client, three retrievers, synthesizer —
are fully mocked with unittest.mock so no network or model calls happen.

The Anthropic API is simulated by constructing objects that match the SDK's
response shape:
  - content blocks with .type == "tool_use" or "text"
  - tool_use blocks with .name, .input (dict), .id attributes
  - the response object has a .content list and .stop_reason attribute

The mock for client.messages.create uses side_effect to return a sequence of
responses, simulating the tool-use → text flow.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.pipeline import PipelineInput
from app.pipelines.v4_router import RouterPipeline
from app.retrievers.narratives import NarrativeHit
from app.retrievers.regulations import RegulationHit
from app.retrievers.stats import StatsResult
from app.synth import SynthResult

# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------


def _tool_use_block(name: str, input_dict: dict, tool_id: str = "tu_001"):
    """Build a mock tool_use content block."""
    block = SimpleNamespace(
        type="tool_use",
        name=name,
        input=input_dict,
        id=tool_id,
    )
    return block


def _text_block(text: str):
    """Build a mock text content block."""
    return SimpleNamespace(type="text", text=text)


def _haiku_response(content: list, stop_reason: str = "tool_use"):
    """Build a mock Anthropic Message response."""
    return SimpleNamespace(content=content, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_stats_result(sql: str = "SELECT 1", error: str | None = None) -> StatsResult:
    if error:
        return StatsResult(sql=sql, columns=[], rows=[], row_count=0, truncated=False, error=error)
    return StatsResult(
        sql=sql,
        columns=["name", "season"],
        rows=[{"name": "Lewis Hamilton", "season": 2021}],
        row_count=1,
        truncated=False,
        error=None,
    )


def _make_narrative_hits() -> list[NarrativeHit]:
    return [
        NarrativeHit(
            chunk_id="n001",
            score=0.9,
            text="The 2021 Abu Dhabi GP ended controversially when the safety car ...",
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
        ),
        NarrativeHit(
            chunk_id="n002",
            score=0.8,
            text="Verstappen overtook Hamilton on the final lap to win the championship.",
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
        ),
    ]


def _make_regulation_hits() -> list[RegulationHit]:
    return [
        RegulationHit(
            chunk_id="r001",
            score=0.95,
            text="The minimum weight of the car, without fuel, shall be 798 kg.",
            regulation_type="technical",
            article_number="4.1",
        ),
    ]


def _make_synth_result(answer: str = "Synthesized answer.") -> SynthResult:
    return SynthResult(answer=answer, citations=["n_0"], refused=False, raw_response=answer)


def _make_pipeline(
    client_side_effect,
    stats_result=None,
    narrative_hits=None,
    regulation_hits=None,
    synth_result=None,
    max_steps: int = 4,
) -> RouterPipeline:
    """Build a RouterPipeline with all mocked dependencies."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = client_side_effect

    mock_stats = MagicMock()
    mock_stats.run.return_value = stats_result or _make_stats_result()

    mock_narratives = MagicMock()
    mock_narratives.search.return_value = narrative_hits if narrative_hits is not None else _make_narrative_hits()

    mock_regulations = MagicMock()
    mock_regulations.search.return_value = regulation_hits if regulation_hits is not None else _make_regulation_hits()

    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = synth_result or _make_synth_result()

    return RouterPipeline(
        anthropic_client=mock_client,
        stats_retriever=mock_stats,
        narratives_retriever=mock_narratives,
        regulations_retriever=mock_regulations,
        synthesizer=mock_synth,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Test 1: Haiku calls search_narratives → text response
# ---------------------------------------------------------------------------


def test_narratives_tool_called_once():
    """Haiku calls search_narratives once, then returns text. Narratives retriever
    is invoked; synth receives the narrative contexts; result has one ToolCall."""
    narrative_hits = _make_narrative_hits()
    synth_result = _make_synth_result("The 2021 Abu Dhabi GP was controversial.")

    responses = [
        _haiku_response(
            [_tool_use_block("search_narratives", {"query": "2021 Abu Dhabi controversy"}, "tu_1")],
            stop_reason="tool_use",
        ),
        _haiku_response(
            [_text_block("I found information about the 2021 Abu Dhabi GP.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        narrative_hits=narrative_hits,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query="What happened at the 2021 Abu Dhabi GP?"))

    # Narratives retriever called once with the haiku-chosen query
    pipeline.narratives_retriever.search.assert_called_once_with(
        "2021 Abu Dhabi controversy", top_k=5
    )
    # Stats and regulations NOT called
    pipeline.stats_retriever.run.assert_not_called()
    pipeline.regulations_retriever.search.assert_not_called()

    # Synth receives narrative contexts
    called_contexts = pipeline.synthesizer.synthesize.call_args[0][1]
    assert all(c.source == "narratives" for c in called_contexts)
    assert len(called_contexts) == 2

    # Result shape
    assert result.version == "v4"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_narratives"
    assert result.tool_calls[0].arguments == {"query": "2021 Abu Dhabi controversy"}
    assert result.answer == synth_result.answer


# ---------------------------------------------------------------------------
# Test 2: Haiku calls query_stats → text response
# ---------------------------------------------------------------------------


def test_stats_tool_called_with_sql():
    """Haiku calls query_stats with SQL; stats retriever runs; result has 1 ToolCall
    named 'query_stats' with arguments containing 'sql'."""
    sql = "SELECT drivers.surname, results.points FROM results JOIN drivers USING (driverId) WHERE raceId = 1"
    stats_result = _make_stats_result(sql=sql)
    synth_result = _make_synth_result("Hamilton won with 25 points.")

    responses = [
        _haiku_response(
            [_tool_use_block("query_stats", {"sql": sql}, "tu_2")],
            stop_reason="tool_use",
        ),
        _haiku_response(
            [_text_block("I found the stats for that race.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        stats_result=stats_result,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query="Who won the 2021 Australian GP?"))

    pipeline.stats_retriever.run.assert_called_once_with(sql)

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "query_stats"
    assert "sql" in tc.arguments
    assert tc.arguments["sql"] == sql

    # Synth context is stats
    called_contexts = pipeline.synthesizer.synthesize.call_args[0][1]
    assert len(called_contexts) == 1
    assert called_contexts[0].source == "stats"


# ---------------------------------------------------------------------------
# Test 3: Multi-tool — query_stats then search_narratives
# ---------------------------------------------------------------------------


def test_multi_tool_stats_then_narratives():
    """Haiku calls query_stats first, then search_narratives; both retrievers
    are called; synth receives both context types; result has 2 tool_calls."""
    sql = "SELECT name FROM races WHERE season=2021 AND round=1"
    stats_result = _make_stats_result(sql=sql)
    narrative_hits = _make_narrative_hits()
    synth_result = _make_synth_result("Combined answer from stats and narrative.")

    responses = [
        # Step 1: Haiku calls query_stats
        _haiku_response(
            [_tool_use_block("query_stats", {"sql": sql}, "tu_3a")],
            stop_reason="tool_use",
        ),
        # Step 2: After stats result fed back, Haiku calls search_narratives
        _haiku_response(
            [_tool_use_block("search_narratives", {"query": "2021 Bahrain GP"}, "tu_3b")],
            stop_reason="tool_use",
        ),
        # Step 3: Final text response
        _haiku_response(
            [_text_block("I now have both stats and narrative context.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        stats_result=stats_result,
        narrative_hits=narrative_hits,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query="Tell me about the 2021 Bahrain GP with stats."))

    pipeline.stats_retriever.run.assert_called_once_with(sql)
    pipeline.narratives_retriever.search.assert_called_once_with("2021 Bahrain GP", top_k=5)

    assert len(result.tool_calls) == 2
    tool_names = [tc.name for tc in result.tool_calls]
    assert "query_stats" in tool_names
    assert "search_narratives" in tool_names

    # Synth should receive contexts from both sources
    called_contexts = pipeline.synthesizer.synthesize.call_args[0][1]
    sources = {c.source for c in called_contexts}
    assert "stats" in sources
    assert "narratives" in sources


# ---------------------------------------------------------------------------
# Test 4: Max-step cap terminates the loop
# ---------------------------------------------------------------------------


def test_max_step_cap_terminates():
    """If Haiku keeps returning tool_use blocks, the loop terminates at max_steps.
    The result includes the tool_calls gathered so far, and synth is called."""
    sql = "SELECT 1"
    stats_result = _make_stats_result(sql=sql)
    synth_result = _make_synth_result("Capped answer.")

    # All 4 responses are tool_use (Haiku never stops)
    tool_response = _haiku_response(
        [_tool_use_block("query_stats", {"sql": sql}, "tu_cap")],
        stop_reason="tool_use",
    )

    responses = [tool_response, tool_response, tool_response, tool_response]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        stats_result=stats_result,
        synth_result=synth_result,
        max_steps=4,
    )

    result = pipeline.run(PipelineInput(query="Keep calling tools."))

    # Loop ran exactly max_steps times
    assert pipeline.client.messages.create.call_count == 4
    # Tool calls collected equal max_steps
    assert len(result.tool_calls) == 4
    # Synth was still called despite hitting cap
    pipeline.synthesizer.synthesize.assert_called_once()
    assert result.version == "v4"


# ---------------------------------------------------------------------------
# Test 5: Tool error — stats retriever returns an error
# ---------------------------------------------------------------------------


def test_stats_error_fed_back():
    """When stats retriever returns StatsResult(error='invalid SQL'), the tool
    result text fed back to Haiku says the SQL was invalid; the failed ToolCall
    is still included in the result."""
    bad_sql = "DROP TABLE races"
    error_result = _make_stats_result(sql=bad_sql, error="invalid SQL syntax")
    synth_result = _make_synth_result("Could not get stats due to error.")

    responses = [
        _haiku_response(
            [_tool_use_block("query_stats", {"sql": bad_sql}, "tu_err")],
            stop_reason="tool_use",
        ),
        _haiku_response(
            [_text_block("The SQL query failed.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        stats_result=error_result,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query="What are the race results?"))

    # Tool call is recorded
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "query_stats"
    # Summary mentions the error
    assert "error" in tc.result_summary.lower() or "failed" in tc.result_summary.lower()

    # Verify the tool result message sent back to Haiku contains the error text
    # (second call to messages.create gets the tool_result in its messages)
    second_call_messages = pipeline.client.messages.create.call_args_list[1][1]["messages"]
    # The last user message should be the tool_result
    last_user_msg = second_call_messages[-1]
    assert last_user_msg["role"] == "user"
    tool_result_content = last_user_msg["content"]
    # content is a list of tool_result dicts
    combined_text = " ".join(
        item.get("content", "") for item in tool_result_content if isinstance(item, dict)
    )
    assert "error" in combined_text.lower() or "failed" in combined_text.lower()


# ---------------------------------------------------------------------------
# Test 6: Synthesizer receives the ORIGINAL user query
# ---------------------------------------------------------------------------


def test_synth_receives_original_query():
    """The synthesizer must be called with the original user query, not Haiku's
    final text response."""
    original_query = "Who won the 2022 championship?"
    synth_result = _make_synth_result("Verstappen won the 2022 championship.")

    responses = [
        _haiku_response(
            [_tool_use_block("search_narratives", {"query": "2022 championship winner"}, "tu_q")],
            stop_reason="tool_use",
        ),
        _haiku_response(
            [_text_block("Haiku's own summary: the answer is Verstappen.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query=original_query))

    # Synth called with the exact original query string
    synth_call_args = pipeline.synthesizer.synthesize.call_args[0]
    assert synth_call_args[0] == original_query
    # Result also carries the original query
    assert result.query == original_query


# ---------------------------------------------------------------------------
# Test 7: Version is "v4"
# ---------------------------------------------------------------------------


def test_result_version_is_v4():
    """PipelineResult.version must be 'v4'."""
    synth_result = _make_synth_result()

    responses = [
        _haiku_response(
            [_text_block("No tools needed.")],
            stop_reason="end_turn",
        ),
    ]

    pipeline = _make_pipeline(
        client_side_effect=responses,
        synth_result=synth_result,
    )

    result = pipeline.run(PipelineInput(query="Hello."))

    assert result.version == "v4"
