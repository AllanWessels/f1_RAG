"""
Shared types for the four pipeline versions.

All pipelines accept a `PipelineInput` and return a `PipelineResult` so the
FastAPI UI, the eval runner, and Langfuse tracing can treat them uniformly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.synth import ContextChunk

VersionId = Literal["v1", "v2", "v3", "v4"]


class PipelineInput(BaseModel):
    query: str
    top_k: int = 5


class ToolCall(BaseModel):
    """One tool invocation by the v4 router (Haiku tool-use)."""

    name: str  # "query_stats" | "search_narratives" | "search_regulations"
    arguments: dict = Field(default_factory=dict)
    result_summary: str = ""  # short text describing what came back


class PipelineResult(BaseModel):
    version: VersionId
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    retrieved_contexts: list[ContextChunk] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)  # v4 only
