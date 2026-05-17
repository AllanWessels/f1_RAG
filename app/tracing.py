"""
Langfuse tracing wiring for the F1 RAG project.

Provides:
  - get_langfuse() -> Langfuse | None
      Returns a Langfuse client when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
      are set in the environment; returns None otherwise.

  - traced(name=None, observation_type="span") decorator
      Wraps a function with a Langfuse span/observation when Langfuse is
      configured; is a transparent no-op (zero overhead) when it is not.

Implementation note:
  We use langfuse.decorators.observe under the hood.  That decorator already
  gracefully no-ops when no public key is configured (the Langfuse SDK checks
  for LANGFUSE_PUBLIC_KEY at import time and disables itself).  This means the
  @traced decorator is safe to use everywhere — unit tests that run without
  Langfuse env vars will never make network calls.

  Sensitive information (API keys) is never captured because:
    a) observe() captures *args / **kwargs via repr(), which for Anthropic
       client objects simply shows the class name, not the key.
    b) We never explicitly log ANTHROPIC_API_KEY or LANGFUSE_SECRET_KEY.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Literal, TypeVar, overload

from langfuse.decorators import observe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public client accessor
# ---------------------------------------------------------------------------


def get_langfuse():
    """Return a Langfuse client if configured, else None.

    A client is considered configured when both LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are present as non-empty environment variables.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    # Lazy import so that importing this module without keys never errors.
    from langfuse import Langfuse

    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        return client
    except Exception:
        logger.exception("Failed to initialise Langfuse client")
        return None


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])

ObservationType = Literal["span", "generation"]


@overload
def traced(__fn: _F) -> _F: ...


@overload
def traced(
    __fn: None = None,
    *,
    name: str | None = None,
    observation_type: ObservationType = "span",
) -> Callable[[_F], _F]: ...


def traced(
    __fn: _F | None = None,
    *,
    name: str | None = None,
    observation_type: ObservationType = "span",
):
    """Decorator that wraps a function with a Langfuse span when tracing is enabled.

    When LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set the decorated
    function is returned unchanged — zero overhead, no imports triggered.

    Parameters
    ----------
    name:
        Optional display name for the span.  Defaults to the function's
        ``__name__``.
    observation_type:
        "span" (default) for generic spans; "generation" for LLM calls.

    Usage::

        @traced
        def my_func(x):
            ...

        @traced(name="my-span", observation_type="generation")
        def my_llm_func(prompt):
            ...
    """
    as_type = "generation" if observation_type == "generation" else None

    def decorator(fn: _F) -> _F:
        span_name = name or fn.__name__
        # observe() already no-ops gracefully when Langfuse is disabled.
        return observe(fn, name=span_name, as_type=as_type)  # type: ignore[return-value]

    if __fn is not None:
        # Called as @traced (no parentheses)
        return decorator(__fn)

    # Called as @traced(...) with keyword args
    return decorator
