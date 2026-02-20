from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI
import time

from app.db.session import get_db
from app.config import settings
from app.ingestion.embedder import embed_query
from app.search.hybrid import hybrid_search
from app.search.reranker import rerank

router = APIRouter()
llm = AsyncOpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

class AskRequest(BaseModel):
    question: str
    mode: str = "hybrid+rerank"   # "vector" | "hybrid" | "hybrid+rerank"
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    mode: str

@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, db: AsyncSession = Depends(get_db)):
    t0 = time.time()

    # 1. Embed query
    q_emb = embed_query(body.question)

    # 2. Retrieve
    if body.mode == "vector":
        from app.search.vector import vector_search
        chunks = await vector_search(db, q_emb, top_k=body.top_k)
    else:
        chunks = await hybrid_search(db, body.question, q_emb, top_k=body.top_k * 2)

    # 3. Rerank
    if body.mode == "hybrid+rerank" and chunks:
        chunks = rerank(body.question, chunks, top_k=body.top_k)
    else:
        chunks = chunks[:body.top_k]

    # 4. Generate answer
    context = "\n\n---\n\n".join([f"[{i+1}] {c['content']}" for i, c in enumerate(chunks)])

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {body.question}

Answer:"""

    response = await llm.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()
    latency_ms = (time.time() - t0) * 1000

    return AskResponse(
        answer=answer,
        sources=[{"id": c["id"], "content": c["content"][:200] + "...", "score": c.get("score", 0)} for c in chunks],
        latency_ms=round(latency_ms, 2),
        mode=body.mode,
    )
