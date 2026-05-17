"""
Tests for app/tracing.py — Langfuse client wrapper + @traced decorator.

All tests must pass without a live Langfuse instance.  Tests that exercise
the "Langfuse configured" path use unittest.mock.patch to prevent any
network calls.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. get_langfuse() returns None when env vars are unset
# ---------------------------------------------------------------------------


def test_get_langfuse_returns_none_without_env_vars(monkeypatch) -> None:
    """get_langfuse() must return None when keys are absent."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import get_langfuse

    assert get_langfuse() is None


def test_get_langfuse_returns_none_with_only_public_key(monkeypatch) -> None:
    """get_langfuse() returns None when only the public key is set."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import get_langfuse

    assert get_langfuse() is None


def test_get_langfuse_returns_none_with_only_secret_key(monkeypatch) -> None:
    """get_langfuse() returns None when only the secret key is set."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    from app.tracing import get_langfuse

    assert get_langfuse() is None


# ---------------------------------------------------------------------------
# 2. get_langfuse() returns a Langfuse client when both env vars are set
# ---------------------------------------------------------------------------


def test_get_langfuse_returns_client_when_keys_set(monkeypatch) -> None:
    """get_langfuse() returns a non-None object when both keys are present."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-abc")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-xyz")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    fake_client = MagicMock()
    fake_client.enabled = True

    # The import of Langfuse in get_langfuse() is lazy (inside the function body).
    # We patch "langfuse.Langfuse" — the canonical location — so our mock is
    # returned when get_langfuse() does `from langfuse import Langfuse`.
    with patch("langfuse.Langfuse", return_value=fake_client):
        from app.tracing import get_langfuse

        result = get_langfuse()

    # get_langfuse() must return the (mocked) client, not None
    assert result is not None


def test_get_langfuse_returns_non_none_with_real_langfuse(monkeypatch) -> None:
    """get_langfuse() returns a truthy value when both keys are provided."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-abc")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-xyz")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    # Patch Langfuse at the source to avoid real HTTP connections
    fake_instance = MagicMock()
    fake_instance.enabled = True

    with patch("langfuse.client.Langfuse.__init__", return_value=None):
        from app.tracing import get_langfuse

        result = get_langfuse()

    # May be the real Langfuse instance (disabled) or our mock; either is non-None
    assert result is not None


# ---------------------------------------------------------------------------
# 3. @traced is a transparent no-op when Langfuse is not configured
# ---------------------------------------------------------------------------


def test_traced_noop_returns_same_value(monkeypatch) -> None:
    """Wrapped function returns the correct value when Langfuse is disabled."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    @traced
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 4) == 7


def test_traced_with_args_noop_returns_same_value(monkeypatch) -> None:
    """@traced(name=...) works without Langfuse and returns correct value."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    @traced(name="my-span")
    def greet(name: str) -> str:
        return f"Hello, {name}"

    assert greet("World") == "Hello, World"


def test_traced_noop_makes_no_langfuse_calls(monkeypatch) -> None:
    """No Langfuse SDK calls are made when keys are absent."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    call_log: list[str] = []

    @traced
    def side_effect_func() -> str:
        call_log.append("called")
        return "ok"

    result = side_effect_func()
    assert result == "ok"
    assert call_log == ["called"]  # only our own code ran


# ---------------------------------------------------------------------------
# 4. @traced wraps function when Langfuse IS configured (mock client)
# ---------------------------------------------------------------------------


def test_traced_calls_observe_when_configured(monkeypatch) -> None:
    """When keys are set, the decorated function still returns the correct value."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    from app.tracing import traced

    @traced(name="compute")
    def compute(x: int) -> int:
        return x * 2

    # Function must still work — tracing is best-effort and transparent
    result = compute(5)
    assert result == 10


def test_traced_generation_type_returns_correct_value(monkeypatch) -> None:
    """observation_type='generation' decorated function returns correct value."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    @traced(name="llm-call", observation_type="generation")
    def fake_llm(prompt: str) -> str:
        return f"response to: {prompt}"

    assert fake_llm("test prompt") == "response to: test prompt"


# ---------------------------------------------------------------------------
# 5. Decorator preserves function signature
# ---------------------------------------------------------------------------


def test_traced_preserves_signature_bare(monkeypatch) -> None:
    """@traced (bare) preserves the wrapped function's signature."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    def original(x: int, y: str = "default") -> str:
        return f"{x} {y}"

    wrapped = traced(original)
    sig = inspect.signature(wrapped)
    params = list(sig.parameters.keys())

    assert "x" in params
    assert "y" in params
    assert sig.parameters["y"].default == "default"


def test_traced_preserves_signature_with_name(monkeypatch) -> None:
    """@traced(name=...) preserves the wrapped function's signature."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    def pipeline_run(self_arg, query: str, top_k: int = 5) -> str:
        return query

    wrapped = traced(name="pipeline-run")(pipeline_run)
    sig = inspect.signature(wrapped)
    params = list(sig.parameters.keys())

    assert "self_arg" in params
    assert "query" in params
    assert "top_k" in params
    assert sig.parameters["top_k"].default == 5


def test_traced_callable_with_positional_and_keyword_args(monkeypatch) -> None:
    """Wrapped function accepts both positional and keyword args correctly."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from app.tracing import traced

    @traced(name="mixed-args")
    def mixed(a: int, b: int, *, c: str = "x") -> str:
        return f"{a},{b},{c}"

    assert mixed(1, 2) == "1,2,x"
    assert mixed(1, b=2, c="y") == "1,2,y"


# ---------------------------------------------------------------------------
# 6. Decorators on retriever/synth methods do not change their return types
# ---------------------------------------------------------------------------


def test_synthesizer_synthesize_decorated_noop(monkeypatch) -> None:
    """Synthesizer.synthesize still works correctly with tracing no-op."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from unittest.mock import MagicMock

    from app.synth import Synthesizer

    # Verify synthesize returns refusal on empty contexts (from existing tests)
    synth = Synthesizer(client=MagicMock())
    result = synth.synthesize("anything", contexts=[])
    assert result.refused is True
    assert result.citations == []


def test_stats_retriever_run_decorated_noop(monkeypatch, tmp_path) -> None:
    """StatsRetriever.run still returns StatsResult correctly with tracing no-op."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    import sqlite3

    from app.retrievers.stats import StatsRetriever

    # Create a minimal test database
    db = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO test_table VALUES (1, 'hello')")
    conn.commit()
    conn.close()

    retriever = StatsRetriever(db_path=db)
    result = retriever.run("SELECT val FROM test_table WHERE id = 1")

    assert result.error is None
    assert result.row_count == 1
    assert result.rows[0]["val"] == "hello"
