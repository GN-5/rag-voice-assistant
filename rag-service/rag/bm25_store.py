from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import re


class BM25Store:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_store: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s]", "", text.lower()).split()
    
    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.chunk_store) > 0
    
    def add_chunks(self, chunks: List[Dict]):
        for chunk in chunks:
            self.tokenized_corpus.append(self._tokenize(chunk["text"]))
            self.chunk_store.append(chunk)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.is_ready():
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == 0:
                continue
            chunk = self.chunk_store[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunk["rank"] = rank + 1
            results.append(chunk)
        return results
    
    def clear(self):
        self.bm25 = None
        self.chunk_store = []
        self.tokenized_corpus = []
    
    def save(self):
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump({
                "chunk_store": self.chunk_store,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
    
    def load_if_exists(self):
        path = self.index_dir / "bm25.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.chunk_store = data["chunk_store"]
            self.tokenized_corpus = data["tokenized_corpus"]
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
