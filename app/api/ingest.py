from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.ingestion.pipeline import ingest_document
from app.search.bm25 import build_bm25_index

router = APIRouter()

class IngestRequest(BaseModel):
    title: str
    content: str
    source: str = None

@router.post("/ingest")
async def ingest(body: IngestRequest, db: AsyncSession = Depends(get_db)):
    result = await ingest_document(db, title=body.title, content=body.content, source=body.source)
    build_bm25_index()   # перестраиваем BM25 индекс
    return result

@router.post("/ingest/rebuild-index")
async def rebuild_index():
    build_bm25_index()
    return {"status": "ok"}
