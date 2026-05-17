"""
Sonnet 4.6 answer synthesis with explicit, structured citations.

All four pipelines call `Synthesizer.synthesize(query, contexts)`. The model
is instructed to cite each fact by the `cite_id` of the context chunk that
supports it. Citations are extracted from the answer and returned separately.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SYNTH_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


class ContextChunk(BaseModel):
    """A retrieved chunk passed to the synthesizer."""

    cite_id: str  # short id used by the model in [cite:cite_id] markers
    source: str  # "stats" | "narratives" | "regulations"
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SynthResult(BaseModel):
    answer: str
    citations: list[str]  # cite_ids referenced in the answer, in order
    refused: bool = False
    raw_response: str = ""


SYSTEM_PROMPT = """You are an F1 (Formula 1) research assistant. You answer questions strictly from the provided context chunks. Each chunk has a short identifier you must cite when you use it.

Rules:
1. Cite every factual claim by appending [cite:CITE_ID] right after the sentence that uses it, where CITE_ID matches the chunk's identifier exactly. You may cite multiple sources: [cite:A][cite:B].
2. If the context does not answer the question, say so plainly: "I don't have enough information to answer that." Do not invent facts. Do not use outside knowledge.
3. Adversarial / nonsensical questions (e.g., asking about a race that didn't happen): say the premise is wrong. Cite a context chunk that contradicts the premise if one is available; otherwise just decline.
4. Keep answers concise: 1-3 short paragraphs unless the question genuinely needs more.
5. Numbers and proper nouns must come from the context, not your prior knowledge.
"""


CITE_PATTERN = re.compile(r"\[cite:([^\]]+)\]")
REFUSAL_PATTERNS = (
    "i don't have enough information",
    "i do not have enough information",
    "i can't answer",
    "i cannot answer",
    "premise is wrong",
    "no such race",
    "did not happen",
    "no such event",
)


def _format_context(contexts: Iterable[ContextChunk]) -> str:
    blocks = []
    for c in contexts:
        meta = (
            ", ".join(f"{k}={v}" for k, v in c.metadata.items()) if c.metadata else ""
        )
        header = f"[{c.cite_id}] source={c.source}"
        if meta:
            header += f"  {meta}"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)


class Synthesizer:
    def __init__(
        self,
        *,
        client: anthropic.Anthropic | None = None,
        model: str = SYNTH_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens

    def synthesize(
        self, query: str, contexts: list[ContextChunk]
    ) -> SynthResult:
        if not contexts:
            return SynthResult(
                answer="I don't have enough information to answer that.",
                citations=[],
                refused=True,
                raw_response="",
            )

        user_msg = (
            f"Question: {query}\n\n"
            f"Context:\n\n{_format_context(contexts)}\n\n"
            "Answer the question using ONLY the context above. Cite as [cite:ID] for each claim."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        citations = list(dict.fromkeys(CITE_PATTERN.findall(text)))
        refused = any(p in text.lower() for p in REFUSAL_PATTERNS)
        return SynthResult(
            answer=text.strip(),
            citations=citations,
            refused=refused,
            raw_response=text,
        )
