from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.config import settings

async def vector_search(db: AsyncSession, query_embedding: list[float], top_k: int = None) -> list[dict]:
    top_k = top_k or settings.TOP_K
    emb_literal = "[" + ",".join("%.8f" % x for x in query_embedding) + "]"
    sql = (
        "SELECT id, content, "
        "1 - (embedding <=> '" + emb_literal + "'::vector) AS score "
        "FROM chunks "
        "ORDER BY embedding <=> '" + emb_literal + "'::vector "
        "LIMIT " + str(top_k)
    )
    result = await db.execute(text(sql))
    rows = result.fetchall()
    return [
        {"id": str(r.id), "content": r.content, "score": float(r.score), "source": "vector"}
        for r in rows
    ]
