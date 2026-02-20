from app.search.vector import vector_search
from app.search.bm25 import bm25_search

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists using RRF."""
    scores = {}
    contents = {}

    for results in results_list:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            contents[doc_id] = item["content"]

    merged = [
        {"id": doc_id, "content": contents[doc_id], "score": score, "source": "hybrid"}
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return merged

async def hybrid_search(db, query: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    # Параллельно: vector + BM25
    vector_results = await vector_search(db, query_embedding, top_k=top_k * 2)
    bm25_results = bm25_search(query, top_k=top_k * 2)

    merged = reciprocal_rank_fusion([vector_results, bm25_results])
    return merged[:top_k]
