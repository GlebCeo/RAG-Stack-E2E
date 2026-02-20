from sentence_transformers import SentenceTransformer, util
import numpy as np

_sim_model = None

def get_sim_model():
    global _sim_model
    if _sim_model is None:
        _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sim_model

def answer_similarity(generated: str, expected: str) -> float:
    """Cosine similarity between generated and expected answer embeddings."""
    model = get_sim_model()
    embs = model.encode([generated, expected])
    score = float(util.cos_sim(embs[0], embs[1]))
    return round(score, 4)

def faithfulness_score(answer: str, chunks: list[str]) -> float:
    """How much of the answer is grounded in retrieved chunks."""
    if not chunks:
        return 0.0
    model = get_sim_model()
    answer_emb = model.encode([answer])[0]
    chunk_embs = model.encode(chunks)
    scores = [float(util.cos_sim(answer_emb, c)) for c in chunk_embs]
    return round(max(scores), 4)
