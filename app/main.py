from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.session import init_db
from app.search.bm25 import build_bm25_index
from app.api.ask import router as ask_router
from app.api.ingest import router as ingest_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    build_bm25_index()   # BM25 index at startup
    yield

app = FastAPI(
    title="RAG Stack E2E",
    description="Ingestion → Hybrid Search (BM25 + pgvector) → Rerank → Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ask_router)
app.include_router(ingest_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
