from sentence_transformers import CrossEncoder
from typing import List, Dict


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name, device="cuda")
    
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
