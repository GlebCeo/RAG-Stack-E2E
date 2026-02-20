import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.session import AsyncSessionLocal, init_db
from app.ingestion.pipeline import ingest_document
from app.search.bm25 import build_bm25_index

async def main():
    print("🔧 Initializing DB...")
    await init_db()

    with open("data/documents.jsonl") as f:
        docs = [json.loads(l) for l in f if l.strip()]

    print(f"📥 Ingesting {len(docs)} documents...")
    async with AsyncSessionLocal() as db:
        for doc in docs:
            result = await ingest_document(db, title=doc["title"], content=doc["content"], source=doc.get("source"))
            print(f"  {'✅' if result['status'] == 'ingested' else '⏭️ '} {result['title']} — {result['status']}")

    print("\n🔍 Building BM25 index...")
    build_bm25_index()
    print("✅ Done! Ready to query.")

if __name__ == "__main__":
    asyncio.run(main())
