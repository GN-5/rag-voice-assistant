from typing import List, Dict
from .vector_store import VectorStore
from .bm25_store import BM25Store
from .reranker import Reranker


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        reranker: Reranker,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k
    
    def retrieve(self, query: str) -> List[Dict]:
        dense_results = self.vector_store.search(query, top_k=self.top_k_retrieve)
        sparse_results = self.bm25_store.search(query, top_k=self.top_k_retrieve)
        
        fused = self._rrf_fuse(dense_results, sparse_results)
        
        candidates = fused[:self.top_k_retrieve]
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
        
        return reranked
    
    def _rrf_fuse(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        chunk_by_id: Dict[str, Dict] = {}
        
        for results in [dense_results, sparse_results]:
            for rank_0, chunk in enumerate(results):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank_0 + 1)
                chunk_by_id[cid] = chunk
        
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused = []
        for cid in sorted_ids:
            c = chunk_by_id[cid].copy()
            c["rrf_score"] = rrf_scores[cid]
            fused.append(c)
        return fused
