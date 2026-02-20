from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"⏳ Loading embedding model: {settings.EMBED_MODEL}")
        _model = SentenceTransformer(settings.EMBED_MODEL)
        print("✅ Embedding model loaded")
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
