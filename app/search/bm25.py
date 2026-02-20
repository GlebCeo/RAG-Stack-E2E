from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text
from app.config import settings

_bm25_index = None
_corpus_chunks = []

def build_bm25_index():
    """Build BM25 index from all chunks (sync, called at startup)."""
    global _bm25_index, _corpus_chunks
    engine = create_engine(settings.DATABASE_SYNC_URL)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, content FROM chunks")).fetchall()

    _corpus_chunks = [{"id": str(r.id), "content": r.content} for r in rows]
    tokenized = [c["content"].lower().split() for c in _corpus_chunks]

    if tokenized:
        _bm25_index = BM25Okapi(tokenized)
        print(f"✅ BM25 index built: {len(_corpus_chunks)} chunks")
    else:
        print("⚠️  BM25 index empty (no chunks yet)")

def bm25_search(query: str, top_k: int = None) -> list[dict]:
    from app.config import settings
    top_k = top_k or settings.TOP_K

    if _bm25_index is None or not _corpus_chunks:
        return []

    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)

    ranked = sorted(
        zip(scores, _corpus_chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {"id": item["id"], "content": item["content"], "score": float(score), "source": "bm25"}
        for score, item in ranked
        if score > 0
    ]
