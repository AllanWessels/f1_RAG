from fastapi import FastAPI

app = FastAPI(title="F1 RAG", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
