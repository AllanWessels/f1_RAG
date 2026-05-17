"""
Pipeline v4 — Haiku 4.5 router with tool-use.

Claude Haiku 4.5 selects which retriever(s) to call (query_stats,
search_narratives, search_regulations) via Anthropic tool-use. Results from
each retriever are collected as ContextChunks and then passed to the Sonnet
4.6 Synthesizer for the final answer.

Design:
  1. Build tool definitions with explicit JSON-schema input shapes.
  2. Loop up to max_steps: send messages to Haiku; if it returns tool_use
     blocks, dispatch each to the correct retriever, capture the result, feed
     it back as a tool_result message, and continue.
  3. Once Haiku returns a plain-text response (no tool_use), exit the loop.
  4. Pass all collected ContextChunks + the original query to Synthesizer.
  5. Return a PipelineResult(version="v4", ...).
"""

from __future__ import annotations

import hashlib
import logging
import os

import anthropic

from app.pipeline import PipelineInput, PipelineResult, ToolCall
from app.synth import ContextChunk, Synthesizer
from app.tracing import traced

logger = logging.getLogger(__name__)

ROUTER_MODEL = "claude-haiku-4-5"
MAX_STEPS = 8

# ---------------------------------------------------------------------------
# Tool definitions sent to Haiku
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "query_stats",
        "description": (
            "Execute a read-only SQL SELECT query against the Jolpica/Ergast-style F1 "
            "SQLite database. Use this tool for hard, factual statistics: race winners, "
            "championship points, lap times, podium finishes, qualifying positions, pit-stop "
            "counts, etc.\n\n"
            "Schema (snake_case columns — NOT Ergast-style camelCase):\n"
            "  races(race_id, season, round, circuit_id, name, date)\n"
            "  results(result_id, race_id, driver_id, constructor_id, grid, position, "
            "position_text, position_order, points, laps, status)\n"
            "  qualifying(qualifying_id, race_id, driver_id, constructor_id, number, "
            "position, q1, q2, q3)\n"
            "  pitstops(pitstop_id, race_id, driver_id, stop, lap, time_of_day, duration)\n"
            "  laptimes(laptime_id, race_id, driver_id, lap, position, time)\n"
            "  drivers(driver_id, code, number, forename, surname, dob, nationality)\n"
            "  constructors(constructor_id, name, nationality)\n"
            "  circuits(circuit_id, name, locality, country, lat, lng)\n\n"
            "Example: SELECT d.forename || ' ' || d.surname AS winner "
            "FROM results r JOIN races ra ON ra.race_id = r.race_id "
            "JOIN drivers d ON d.driver_id = r.driver_id "
            "WHERE ra.season = 2021 AND ra.name LIKE '%Brazil%' AND r.position = 1;\n\n"
            "Only SELECT or WITH statements are allowed. No INSERT/UPDATE/DELETE/DROP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A single read-only SELECT or WITH SQL statement. "
                        "Must not contain semicolons mid-statement."
                    ),
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "search_narratives",
        "description": (
            "Semantic search over Wikipedia race-report chunks (2014+ modern era). "
            "Use this tool for narrative questions: controversies, race stories, weather "
            "conditions, dramatic moments, contextual background. Examples: 'crash at "
            "2021 Abu Dhabi safety car restart', 'rain at 2021 Belgian GP', "
            "'controversy Hamilton Verstappen collision'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query for the narrative database.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_regulations",
        "description": (
            "Semantic search over 2026 FIA Sporting, Technical, and Financial Regulations. "
            "Use this tool for rules, limits, penalties, technical specifications, and "
            "governance questions. Examples: 'minimum car weight 2026', 'budget cap limit', "
            "'penalty for causing a collision', 'DRS activation zone rules'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query for the FIA regulations.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# Haiku system prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are an F1 (Formula 1) research routing agent. Your job is to decide which \
retrieval tools to call in order to gather the information needed to answer the \
user's question. You have three tools available:

1. query_stats(sql) — Hard statistics from a Jolpica/Ergast SQLite database.
   Use this for: race winners, championship standings, points, lap times, \
qualifying positions, podiums, pit-stop data.
   Database tables: races, results, qualifying, pitstops, laptimes, drivers, \
constructors, circuits.
   Example questions: "Who won the 2017 Australian GP?", "How many points did \
Hamilton score in 2020?", "What was Verstappen's qualifying position at Monza 2021?"

2. search_narratives(query, top_k) — Semantic search over Wikipedia race reports.
   Use this for: race stories, controversies, crashes, weather, dramatic moments, \
contextual background, multi-hop narrative reasoning.
   Example questions: "What was the controversy at 2021 Abu Dhabi?", "What happened \
at the 2021 Belgian GP?", "Describe the safety car situation at the 2020 Bahrain GP."

3. search_regulations(query, top_k) — Semantic search over 2026 FIA regulations.
   Use this for: rules, technical limits, penalties, budget caps, technical specs, \
governance.
   Example questions: "What is the minimum car weight?", "What is the penalty for \
causing a collision?", "What are the DRS rules for 2026?"

Guidelines:
- Call as many tools as the question needs — often just one, sometimes two or three.
- For factual stats questions, always prefer query_stats with targeted SQL.
- For narrative or contextual questions, use search_narratives.
- For rule/regulation questions, use search_regulations.
- For multi-part questions, call multiple tools.
- Write correct SQL for the stats database; use table aliases and JOINs as needed.
- After reviewing tool results, you do NOT need to produce the final answer — \
a separate synthesis model will do that. Simply call the tools you need, then \
give a brief (1-2 sentence) summary of what you found so the synthesizer has context.
"""


# ---------------------------------------------------------------------------
# Helpers: convert retriever results → ContextChunk(s)
# ---------------------------------------------------------------------------


def _hash6(text: str) -> str:
    """Return 6-char hex hash of text."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:6]


def _stats_result_to_chunk(stats_result) -> ContextChunk:
    """Convert a StatsResult to a single ContextChunk."""
    sql = stats_result.sql
    rows = stats_result.rows
    columns = stats_result.columns
    row_count = stats_result.row_count

    # Build compact text: SQL on first line, then up to 20 CSV rows
    lines = [f"SQL: {sql}"]
    if stats_result.error:
        lines.append(f"Error: {stats_result.error}")
    elif columns:
        lines.append(",".join(columns))
        for row in rows[:20]:
            lines.append(",".join(str(row.get(c, "")) for c in columns))
        if stats_result.truncated:
            lines.append(f"... (truncated, showing {len(rows[:20])} of {row_count})")

    text = "\n".join(lines)
    cite_id = f"sql_{_hash6(sql)}"
    return ContextChunk(
        cite_id=cite_id,
        source="stats",
        text=text,
        metadata={"sql": sql, "row_count": row_count},
    )


def _stats_result_to_summary(stats_result, max_rows: int = 5) -> str:
    """Short text summary of a StatsResult to feed back to Haiku."""
    if stats_result.error:
        return f"SQL query failed. Error: {stats_result.error}"
    columns = stats_result.columns
    rows = stats_result.rows
    sample_lines = []
    if columns:
        sample_lines.append(",".join(columns))
        for row in rows[:max_rows]:
            sample_lines.append(",".join(str(row.get(c, "")) for c in columns))
    sample = "\n".join(sample_lines)
    return (
        f"Query returned {stats_result.row_count} rows. "
        f"Columns: {', '.join(columns)}.\n"
        f"Sample:\n{sample}"
    )


def _narratives_to_chunks(hits: list, top_k: int = 3) -> list[ContextChunk]:
    """Convert a list of NarrativeHit to ContextChunks."""
    chunks = []
    for i, hit in enumerate(hits):
        metadata: dict = {}
        if hit.year is not None:
            metadata["year"] = hit.year
        if hit.gp_name:
            metadata["gp_name"] = hit.gp_name
        if hit.section_path:
            metadata["section_path"] = "/".join(hit.section_path)
        chunks.append(
            ContextChunk(
                cite_id=f"n_{i}",
                source="narratives",
                text=hit.text,
                metadata=metadata,
            )
        )
    return chunks


def _narratives_to_summary(hits: list, top_k: int = 3) -> str:
    """Short text summary of narrative hits to feed back to Haiku."""
    if not hits:
        return "No narrative hits found."
    lines = [f"Top {min(top_k, len(hits))} narrative results:"]
    for i, hit in enumerate(hits[:top_k]):
        label = f"{hit.year} {hit.gp_name}" if hit.year and hit.gp_name else f"hit {i}"
        snippet = hit.text[:200].replace("\n", " ")
        lines.append(f"  [{i}] {label}: {snippet}...")
    return "\n".join(lines)


def _regulations_to_chunks(hits: list) -> list[ContextChunk]:
    """Convert a list of RegulationHit to ContextChunks."""
    chunks = []
    for i, hit in enumerate(hits):
        metadata: dict = {}
        if hit.article_number:
            metadata["article_number"] = hit.article_number
        if hit.regulation_type:
            metadata["regulation_type"] = hit.regulation_type
        chunks.append(
            ContextChunk(
                cite_id=f"r_{i}",
                source="regulations",
                text=hit.text,
                metadata=metadata,
            )
        )
    return chunks


def _regulations_to_summary(hits: list, top_k: int = 3) -> str:
    """Short text summary of regulation hits to feed back to Haiku."""
    if not hits:
        return "No regulation hits found."
    lines = [f"Top {min(top_k, len(hits))} regulation results:"]
    for i, hit in enumerate(hits[:top_k]):
        label = hit.article_number or f"hit {i}"
        reg_type = hit.regulation_type or ""
        snippet = hit.text[:200].replace("\n", " ")
        lines.append(f"  [{i}] Article {label} ({reg_type}): {snippet}...")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RouterPipeline class
# ---------------------------------------------------------------------------


class RouterPipeline:
    """v4 pipeline: Haiku 4.5 tool-use router + Sonnet 4.6 synthesis."""

    def __init__(
        self,
        *,
        anthropic_client: anthropic.Anthropic | None = None,
        stats_retriever=None,
        narratives_retriever=None,
        regulations_retriever=None,
        synthesizer: Synthesizer | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.client = anthropic_client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.stats_retriever = stats_retriever
        self.narratives_retriever = narratives_retriever
        self.regulations_retriever = regulations_retriever
        self.synthesizer = synthesizer or Synthesizer()
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict,
        top_k: int,
    ) -> tuple[str, list[ContextChunk], str | None]:
        """
        Dispatch a tool call to the appropriate retriever.

        Returns:
            (summary_for_haiku, context_chunks, error_or_none)
        """
        if tool_name == "query_stats":
            sql = tool_input.get("sql", "")
            result = self.stats_retriever.run(sql)
            chunk = _stats_result_to_chunk(result)
            summary = _stats_result_to_summary(result)
            error = result.error
            return summary, [chunk], error

        elif tool_name == "search_narratives":
            query = tool_input.get("query", "")
            k = tool_input.get("top_k", top_k)
            hits = self.narratives_retriever.search(query, top_k=k)
            chunks = _narratives_to_chunks(hits)
            summary = _narratives_to_summary(hits)
            return summary, chunks, None

        elif tool_name == "search_regulations":
            query = tool_input.get("query", "")
            k = tool_input.get("top_k", top_k)
            hits = self.regulations_retriever.search(query, top_k=k)
            chunks = _regulations_to_chunks(hits)
            summary = _regulations_to_summary(hits)
            return summary, chunks, None

        else:
            msg = f"Unknown tool: {tool_name}"
            logger.warning(msg)
            return msg, [], msg

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    @traced(name="v4_router_run")
    def run(self, input: PipelineInput) -> PipelineResult:
        query = input.query
        top_k = input.top_k

        # Conversation history sent to Haiku
        messages: list[dict] = [{"role": "user", "content": query}]

        collected_chunks: list[ContextChunk] = []
        collected_tool_calls: list[ToolCall] = []

        for step in range(self.max_steps):
            logger.debug("Router step %d/%d", step + 1, self.max_steps)

            response = self.client.messages.create(
                model=ROUTER_MODEL,
                max_tokens=1024,
                system=ROUTER_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Collect any tool_use blocks in this response
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                # Haiku gave a text response — exit loop
                break

            # Append Haiku's assistant message (with tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch each tool call; accumulate results
            tool_results_content = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                summary, chunks, _error = self._dispatch_tool(tool_name, tool_input, top_k)

                collected_chunks.extend(chunks)
                collected_tool_calls.append(
                    ToolCall(
                        name=tool_name,
                        arguments=dict(tool_input),
                        result_summary=summary,
                    )
                )

                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": summary,
                    }
                )

            # Feed tool results back to Haiku
            messages.append({"role": "user", "content": tool_results_content})

        # Synthesize using the ORIGINAL query and all collected context
        synth_result = self.synthesizer.synthesize(query, collected_chunks)

        return PipelineResult(
            version="v4",
            query=query,
            answer=synth_result.answer,
            citations=synth_result.citations,
            refused=synth_result.refused,
            retrieved_contexts=collected_chunks,
            tool_calls=collected_tool_calls,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_v4(
    query: str,
    top_k: int = 5,
    *,
    anthropic_client: anthropic.Anthropic | None = None,
    stats_retriever=None,
    narratives_retriever=None,
    regulations_retriever=None,
    synthesizer: Synthesizer | None = None,
    max_steps: int = MAX_STEPS,
    **kwargs,
) -> PipelineResult:
    """
    Convenience factory for the v4 router pipeline.

    Builds real retrievers from env-vars if not provided.
    """
    if stats_retriever is None:
        from app.retrievers.stats import StatsRetriever

        stats_retriever = StatsRetriever()

    if narratives_retriever is None:
        from app.retrievers.narratives import NarrativesRetriever

        narratives_retriever = NarrativesRetriever()

    if regulations_retriever is None:
        import os as _os

        from qdrant_client import QdrantClient

        from app.embedding import Embedders, Reranker
        from app.retrievers.regulations import RegulationsRetriever

        qdrant_url = _os.environ.get("QDRANT_URL", "http://localhost:6333")
        regulations_retriever = RegulationsRetriever(
            qdrant_client=QdrantClient(url=qdrant_url),
            embedders=Embedders(),
            reranker=Reranker(),
        )

    if synthesizer is None:
        synthesizer = Synthesizer()

    pipeline = RouterPipeline(
        anthropic_client=anthropic_client,
        stats_retriever=stats_retriever,
        narratives_retriever=narratives_retriever,
        regulations_retriever=regulations_retriever,
        synthesizer=synthesizer,
        max_steps=max_steps,
    )
    return pipeline.run(PipelineInput(query=query, top_k=top_k))
