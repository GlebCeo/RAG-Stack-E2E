import hashlib
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import Document, Chunk
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts

def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def ingest_document(
    db: AsyncSession,
    title: str,
    content: str,
    source: str = None,
) -> dict:
    content_hash = hash_content(content)

    # Дедупликация
    existing = await db.execute(select(Document).where(Document.content_hash == content_hash))
    if existing.scalar_one_or_none():
        return {"status": "skipped", "reason": "duplicate", "title": title}

    doc = Document(id=uuid.uuid4(), title=title, source=source, content_hash=content_hash)
    db.add(doc)
    await db.flush()

    # Chunking
    chunks_text = chunk_text(content)

    # Embeddings
    embeddings = embed_texts(chunks_text)

    # Сохраняем чанки
    for i, (text, emb) in enumerate(zip(chunks_text, embeddings)):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            content=text,
            chunk_index=i,
            embedding=emb,
            token_count=len(text.split()),
        )
        db.add(chunk)

    await db.commit()
    return {"status": "ingested", "title": title, "chunks": len(chunks_text)}
