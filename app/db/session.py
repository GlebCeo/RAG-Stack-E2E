from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine, text
from app.config import settings
from app.db.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
sync_engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)

async def init_db():
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)
        # IVFFlat только если достаточно данных (> 100 строк)
        try:
            count = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
            n = count.scalar()
            if n >= 100:
                lists = min(n // 10, 100)
                await conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks "
                    f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
                ))
        except Exception:
            pass

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
