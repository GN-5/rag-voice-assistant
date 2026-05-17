import nltk
from typing import List, Dict

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class SemanticChunker:
    def __init__(self, chunk_size: int = 256, overlap_tokens: int = 50):
        self.chunk_size = chunk_size
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0
        
        for sent in sentences:
            sent_tokens = len(sent.split())
            
            if current_len + sent_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": f"{source}::chunk_{chunk_idx}",
                })
                chunk_idx += 1
                
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_len + s_len > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += s_len
                
                current_chunk = overlap_sentences
                current_len = overlap_len
            
            current_chunk.append(sent)
            current_len += sent_tokens
        
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": source,
                "chunk_id": f"{source}::chunk_{chunk_idx}",
            })
        
        return chunks
