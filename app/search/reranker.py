from sentence_transformers import CrossEncoder
import numpy as np

_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("⏳ Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Reranker loaded")
    return _reranker

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {**chunk, "rerank_score": float(score), "source": "hybrid+rerank"}
        for score, chunk in ranked
    ]
