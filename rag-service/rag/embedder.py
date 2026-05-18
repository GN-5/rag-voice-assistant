from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = None
        self.dim = None
        for device in ["cuda", "cpu"]:
            try:
                self.model = SentenceTransformer(model_name, device=device)
                self.model.eval()
                self.dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Embedder loaded on {device.upper()}")
                return
            except Exception as e:
                logger.warning(f"Embedder failed on {device}: {e}")
        raise RuntimeError("Embedder failed to load on any device")
    
    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]
