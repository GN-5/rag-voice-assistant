from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = None
        for device in ["cuda", "cpu"]:
            try:
                self.model = CrossEncoder(model_name, device=device)
                logger.info(f"Reranker loaded on {device.upper()}")
                return
            except Exception as e:
                logger.warning(f"Reranker failed on {device}: {e}")
        raise RuntimeError("Reranker failed to load on any device")
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)
        
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for chunk, score in ranked:
            c = chunk.copy()
            c["rerank_score"] = float(score)
            results.append(c)
        return results
