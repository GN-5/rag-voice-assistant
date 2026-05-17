import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional


class VectorStore:
    def __init__(self, embedder, index_dir: Path):
        self.embedder = embedder
        self.index_dir = index_dir
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_store: List[Dict] = []
    
    def _init_index(self):
        self.index = faiss.IndexFlatIP(self.embedder.dim)
    
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
    
    def add_chunks(self, chunks: List[Dict]):
        if self.index is None:
            self._init_index()
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings)
        self.chunk_store.extend(chunks)
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.is_ready():
            return []
        q_emb = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunk_store[idx].copy()
            chunk["dense_score"] = float(score)
            chunk["rank"] = len(results) + 1
            results.append(chunk)
        return results
    
    def clear(self):
        self.index = None
        self.chunk_store = []
    
    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "faiss_chunks.pkl", "wb") as f:
            pickle.dump(self.chunk_store, f)
    
    def load_if_exists(self):
        index_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "faiss_chunks.pkl"
        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                self.chunk_store = pickle.load(f)
