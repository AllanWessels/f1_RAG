# F1 RAG

A Formula 1 question-answering RAG with router-based tool-use over three corpora (statistics, race narratives, FIA regulations) and a real per-bucket evaluation harness comparing four progressively more sophisticated retrieval architectures.

See [docs/DESIGN.md](docs/DESIGN.md) for the system design and [docs/EXECUTION.md](docs/EXECUTION.md) for the milestone plan.

## Quickstart

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY at minimum
docker compose up --build
```

Then open:
- `http://localhost:8000` — chat + eval dashboard
- `http://localhost:3000` — Langfuse traces

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```
