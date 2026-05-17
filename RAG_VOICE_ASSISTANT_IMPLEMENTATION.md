# RAG Voice Assistant — Complete Implementation Specification
### Document Version: 1.0 | Target Agent: Claude Code (or any capable coding agent)

---

## PREAMBLE FOR THE CODING AGENT

You are building a **privacy-first, self-hosted, real-time voice assistant** that answers questions from uploaded documents using hybrid RAG (Retrieval-Augmented Generation). Everything runs on the user's own hardware. No external cloud services touch the audio or document data. Read this document in full before writing a single line of code. Every architectural decision in here has been deliberated — do not deviate without flagging it.

---

## 1. PROJECT OVERVIEW

### 1.1 What We Are Building

A voice assistant that a user operates from a **Mac browser** (the client), where all inference, retrieval, and audio processing runs on a **remote Ubuntu server** (the backend). The user speaks into their microphone; the system transcribes the speech, retrieves relevant context from uploaded documents, constructs a prompt, generates an answer via a local LLM, and streams synthesized audio back to the browser — all in under 2 seconds for the first audible word.

### 1.2 Core Capabilities (V0 / MVP Scope)

1. Real-time voice input and output via WebRTC (browser mic → server → browser speaker)
2. Speech-to-text using faster-whisper running on GPU
3. Hybrid RAG: Dense (FAISS vector search) + Sparse (BM25 keyword search) + RRF fusion + Cross-encoder reranking
4. Local LLM inference (Gemma 4 via llama.cpp with CUDA)
5. Text-to-speech using Kokoro running on GPU, streaming audio chunks
6. Document management: upload, index, list, delete from a simple web UI
7. Indexing progress feedback via SSE (Server-Sent Events) — UI blocks queries during indexing

### 1.3 What Is Explicitly Out of Scope for V0

- Multi-user sessions
- Notion or external database integrations
- Voice activity detection fine-tuning
- Authentication/authorization
- Persistent session history across restarts
- Any external API calls (all inference is local)

---

## 2. INFRASTRUCTURE OVERVIEW

### 2.1 Physical Machines

| Machine | Role | OS | Network |
|---|---|---|---|
| Ubuntu Server | Backend / Inference Engine | Ubuntu Server 24.04 LTS | Tailscale (100.x.x.x) |
| MacBook | Client / Developer Machine | macOS | Tailscale (100.y.y.y) |

Both machines are connected over **Tailscale** (WireGuard-based VPN). Tailscale handles encryption of all traffic in transit — no additional TLS/nginx required for V0.

### 2.2 Ubuntu Server Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| CUDA | Must be CUDA 12.x (verify with `nvidia-smi`) |
| nvidia-container-toolkit | Already installed |

### 2.3 VRAM Budget (Critical — Do Not Exceed)

The GPU must run all GPU-accelerated services simultaneously. Budget is tight. Stick to this allocation:

| Service | Model | VRAM |
|---|---|---|
| llama.cpp (bare metal) | Gemma 4 E4B Q4_K_M | ~6.5 GB |
| Speaches (STT) | faster-whisper large-v3-turbo | ~1.5 GB |
| RAG Service (embeddings) | BAAI/bge-small-en-v1.5 | ~0.1 GB |
| RAG Service (reranker) | BAAI/bge-reranker-base | ~0.3 GB |
| Kokoro (TTS) | kokoro-82M | ~0.3 GB |
| **Total** | | **~8.7 GB** |

Headroom: ~3.3 GB. If headroom is violated at runtime, the first swap is to `faster-whisper medium.en` (~0.8 GB, saving ~0.7 GB). The WHISPER_MODEL env var controls this without code changes.

---

## 3. SYSTEM ARCHITECTURE

### 3.1 Full Request Lifecycle

```
[User speaks into Mac browser]
         |
         | WebRTC audio stream
         v
[LiveKit Server] ← (self-hosted Docker container on Ubuntu)
         |
         | forwards audio frames to Agent
         v
[LiveKit Agent] (Python process, Docker container on Ubuntu)
         |
         | 1. Silero VAD detects end-of-speech
         | 2. Audio frames sent to Speaches (STT)
         v
[Speaches / faster-whisper] → returns transcript text
         |
         | transcript text
         v
[LiveKit Agent: on_user_turn_completed hook]
         |
         | HTTP POST /retrieve (query=transcript)
         v
[RAG Service / FastAPI]
         ├── BM25 sparse search (in-memory)
         ├── FAISS dense search (GPU-accelerated)
         └── RRF fusion → cross-encoder rerank → top-k chunks returned
         |
         | augmented context injected into chat context
         v
[LiveKit Agent: sends prompt to LLM]
         |
         | HTTP to llama.cpp (bare metal, already running at 100.x.x.x:8080)
         | Uses OpenAI-compatible /v1/chat/completions endpoint
         | Streaming enabled (stream=True)
         v
[llama.cpp → Gemma 4 E4B] → streams token chunks
         |
         | token stream
         v
[LiveKit Agent: feeds tokens to Kokoro TTS]
         |
         | sentence-boundary streaming: Kokoro synthesizes as full sentences arrive
         v
[Kokoro TTS] → streams PCM audio chunks
         |
         | audio chunks via WebRTC
         v
[Mac browser speaker — user hears response]
```

### 3.2 Docker Service Map

Everything runs in a **single `docker-compose.yml`** on the Ubuntu server, with the exception of llama.cpp which is already running bare-metal.

| Container Name | Image / Source | Ports (host:container) | GPU | Purpose |
|---|---|---|---|---|
| `livekit` | `livekit/livekit-server:latest` | 7880:7880, 7881:7881, 7882:7882/udp | No | WebRTC media routing |
| `speaches` | `ghcr.io/speaches-ai/speaches:latest-gpu` | 8000:8000 | Yes | STT (faster-whisper, OpenAI-compatible API) |
| `kokoro` | `ghcr.io/remsky/kokoro-fastapi-gpu:latest` | 8880:8880 | Yes | TTS (Kokoro-82M, OpenAI-compatible API) |
| `rag-service` | `./rag-service` (custom Dockerfile) | 8100:8100 | Yes | RAG pipeline + document management + web frontend |
| `agent` | `./agent` (custom Dockerfile) | — (no public port) | Yes | LiveKit Agent orchestrator |

**llama.cpp** is NOT in docker-compose. It is already running at `http://100.x.x.x:8080` (Tailscale IP) on the Ubuntu host. The agent container reaches it via this address.

---

## 4. REPOSITORY STRUCTURE

The agent writes code to this directory layout on the Ubuntu server. All development happens on Ubuntu (via Remote SSH from Mac/Cursor).

```
/project-root/
├── docker-compose.yml
├── .env                          # All environment variables (not committed)
├── .env.example                  # Template for .env (committed)
├── .gitignore
├── README.md
│
├── livekit/
│   └── livekit.yaml              # LiveKit server config
│
├── agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── agent.py                  # Main LiveKit Agent entrypoint
│   └── plugins/
│       └── __init__.py
│
├── rag-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   # FastAPI app: RAG API + file upload + frontend serving
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Semantic chunker with sentence-boundary overlap
│   │   ├── embedder.py           # BAAI/bge-small-en-v1.5 embedding wrapper
│   │   ├── vector_store.py       # FAISS IVF-PQ index management
│   │   ├── bm25_store.py         # BM25 index (rank_bm25)
│   │   ├── reranker.py           # Cross-encoder BAAI/bge-reranker-base
│   │   ├── retriever.py          # Hybrid retrieval: RRF fusion + reranker orchestration
│   │   └── document_processor.py # PDF/DOCX/TXT/MD → text extraction + normalization
│   ├── storage/
│   │   ├── documents/            # Uploaded raw files (volume-mounted)
│   │   ├── index/                # FAISS index files (volume-mounted)
│   │   └── metadata.json         # File registry: name, path, timestamp, chunk_ids
│   └── static/
│       └── index.html            # Single-page frontend (mic button, file upload, status)
│
└── volumes/                      # Docker volume mount point reference (not committed)
```

---

## 5. ENVIRONMENT VARIABLES

Create `.env` in the project root. The docker-compose.yml reads this file. Never commit it.

```env
# ── LiveKit ───────────────────────────────────────────────
LIVEKIT_URL=ws://100.x.x.x:7880          # Tailscale IP of Ubuntu server
LIVEKIT_API_KEY=devkey                    # Must match livekit.yaml
LIVEKIT_API_SECRET=devsecret             # Must match livekit.yaml

# ── STT (Speaches / faster-whisper) ──────────────────────
WHISPER_MODEL=large-v3-turbo             # Swap to medium.en if VRAM pressure

# ── LLM (llama.cpp bare-metal) ───────────────────────────
LLAMA_CPP_BASE_URL=http://100.x.x.x:8080/v1   # Tailscale IP, already running

# ── TTS (Kokoro) ─────────────────────────────────────────
KOKORO_BASE_URL=http://kokoro:8880/v1

# ── RAG Service ──────────────────────────────────────────
RAG_SERVICE_URL=http://rag-service:8100
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
TOP_K_RETRIEVE=20                        # Candidates before reranking
TOP_K_RERANK=5                           # Final chunks injected into prompt

# ── Agent ─────────────────────────────────────────────────
AGENT_NAME=rag-assistant
```

---

## 6. SERVICE CONFIGURATIONS

### 6.1 LiveKit Server (`livekit/livekit.yaml`)

```yaml
port: 7880
rtc:
  tcp_port: 7881
  udp_port: 7882
  use_external_ip: false        # Tailscale handles addressing
keys:
  devkey: devsecret             # Must match .env LIVEKIT_API_KEY / SECRET
logging:
  level: info
```

LiveKit is a media router only. It moves audio frames between the browser and the agent. It performs no AI inference and consumes minimal CPU/RAM. CAAL (the reference project) runs it identically.

### 6.2 `docker-compose.yml`

```yaml
version: "3.9"

services:

  livekit:
    image: livekit/livekit-server:latest
    command: --config /etc/livekit.yaml
    ports:
      - "7880:7880"
      - "7881:7881"
      - "7882:7882/udp"
    volumes:
      - ./livekit/livekit.yaml:/etc/livekit.yaml:ro
    restart: unless-stopped

  speaches:
    image: ghcr.io/speaches-ai/speaches:latest-gpu
    ports:
      - "8000:8000"
    environment:
      - WHISPER_MODEL=${WHISPER_MODEL}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-gpu:latest
    ports:
      - "8880:8880"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  rag-service:
    build: ./rag-service
    ports:
      - "8100:8100"
    environment:
      - EMBEDDING_MODEL=${EMBEDDING_MODEL}
      - RERANKER_MODEL=${RERANKER_MODEL}
      - TOP_K_RETRIEVE=${TOP_K_RETRIEVE}
      - TOP_K_RERANK=${TOP_K_RERANK}
    volumes:
      - ./rag-service/storage:/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - livekit

  agent:
    build: ./agent
    environment:
      - LIVEKIT_URL=${LIVEKIT_URL}
      - LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
      - LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
      - LLAMA_CPP_BASE_URL=${LLAMA_CPP_BASE_URL}
      - RAG_SERVICE_URL=${RAG_SERVICE_URL}
      - KOKORO_BASE_URL=${KOKORO_BASE_URL}
      - AGENT_NAME=${AGENT_NAME}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - livekit
      - speaches
      - kokoro
      - rag-service
    extra_hosts:
      - "host.docker.internal:host-gateway"   # Allows reaching bare-metal llama.cpp
```

**Important note on llama.cpp networking:** The LLAMA_CPP_BASE_URL uses the Tailscale IP (`100.x.x.x`). From inside a Docker container on the same host, Tailscale IPs on the host's `tailscale0` interface are reachable over the Docker bridge network. This works without any extra configuration. The `extra_hosts` entry also ensures `host.docker.internal` is available as a fallback if needed.

---

## 7. RAG SERVICE — DETAILED SPECIFICATION

This is the most complex component. It is a FastAPI application that serves three concerns: (1) document ingestion and indexing, (2) hybrid retrieval API for the agent, and (3) the web frontend (static HTML + SSE for progress).

### 7.1 `rag-service/Dockerfile`

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-dev \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
```

### 7.2 `rag-service/requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
aiofiles==23.2.1
sse-starlette==2.1.0

# Document processing
pymupdf==1.24.0               # PDF extraction (fitz)
python-docx==1.1.0            # DOCX extraction
markdown==3.6                 # MD handling

# Embeddings + Vector search
sentence-transformers==3.0.0  # BGE models (embedding + reranker)
faiss-gpu==1.7.2              # FAISS with CUDA — if unavailable use faiss-cpu as fallback
torch==2.3.0                  # CUDA-enabled torch (auto-selected by sentence-transformers)

# Sparse search
rank-bm25==0.2.2

# Utilities
numpy==1.26.4
nltk==3.8.1
```

**Note on faiss-gpu:** The pip package `faiss-gpu` is for older CUDA versions. For CUDA 12.x, build from source OR use `faiss-cpu` with explicit GPU indexing via PyTorch tensors. Simplest working approach: install `faiss-cpu` and rely on sentence-transformers for GPU-accelerated embedding, keeping FAISS operations on CPU for V0. FAISS CPU search is still fast enough for small-to-medium document sets.

### 7.3 `rag-service/main.py`

This is the FastAPI application. It must implement the following routes:

```
GET  /                      → serves static/index.html
POST /upload                → accepts multipart file upload, triggers indexing
GET  /index-progress        → SSE stream: emits {"progress": 0.0-1.0, "stage": "..."} events
GET  /documents             → returns list of indexed documents (name, size, timestamp)
DELETE /documents/{filename} → removes document from disk + triggers background reindex
POST /retrieve              → accepts {"query": str}, returns {"chunks": [...], "sources": [...]}
GET  /health                → returns {"status": "ok"}
```

The `/retrieve` endpoint is the hot path called by the LiveKit Agent on every user turn. It must be fast. Target: under 200ms for a document set under 50 files.

**Startup behavior:** On app startup, the RAG service must:
1. Load the embedding model (BAAI/bge-small-en-v1.5) onto GPU and keep it warm
2. Load the reranker model (BAAI/bge-reranker-base) onto GPU and keep it warm
3. Load existing FAISS index from `storage/index/faiss.index` (if it exists)
4. Load existing BM25 index from `storage/index/bm25.pkl` (if it exists)
5. Load `storage/metadata.json` (if it exists)
6. Begin accepting requests

Models must be loaded once at startup (use FastAPI `lifespan` context manager), not per-request.

**Full `main.py` structure:**

```python
import asyncio, os, json, pickle, logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.retriever import HybridRetriever
from rag.document_processor import DocumentProcessor
from rag.chunker import SemanticChunker

STORAGE_DIR = Path("/app/storage")
DOCUMENTS_DIR = STORAGE_DIR / "documents"
INDEX_DIR = STORAGE_DIR / "index"
METADATA_FILE = STORAGE_DIR / "metadata.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Global state (loaded once at startup, shared across requests)
embedder: Embedder = None
reranker: Reranker = None
vector_store: VectorStore = None
bm25_store: BM25Store = None
retriever: HybridRetriever = None

# SSE progress tracking
indexing_progress: dict = {"progress": 1.0, "stage": "idle"}
indexing_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, reranker, vector_store, bm25_store, retriever
    
    logging.info("Loading models...")
    embedder = Embedder(model_name=os.environ["EMBEDDING_MODEL"])
    reranker = Reranker(model_name=os.environ["RERANKER_MODEL"])
    
    logging.info("Loading indexes...")
    vector_store = VectorStore(embedder=embedder, index_dir=INDEX_DIR)
    vector_store.load_if_exists()
    
    bm25_store = BM25Store(index_dir=INDEX_DIR)
    bm25_store.load_if_exists()
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        top_k_retrieve=int(os.environ.get("TOP_K_RETRIEVE", 20)),
        top_k_rerank=int(os.environ.get("TOP_K_RERANK", 5)),
    )
    
    logging.info("RAG service ready.")
    yield
    # Cleanup on shutdown (none required for V0)

app = FastAPI(lifespan=lifespan)

# --- Health ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Frontend ---
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("/app/static/index.html")

# --- Document Management ---
@app.get("/documents")
def list_documents():
    meta = load_metadata()
    return {"documents": list(meta.values())}

@app.delete("/documents/{filename}")
async def delete_document(filename: str, background_tasks: BackgroundTasks):
    meta = load_metadata()
    if filename not in meta:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = DOCUMENTS_DIR / filename
    if file_path.exists():
        file_path.unlink()
    
    del meta[filename]
    save_metadata(meta)
    
    # Rebuild index silently in background
    background_tasks.add_task(rebuild_index_task, meta)
    return {"status": "deleted", "message": f"{filename} removed. Index rebuilding in background."}

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    
    dest_path = DOCUMENTS_DIR / file.filename
    async with aiofiles.open(dest_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    meta = load_metadata()
    meta[file.filename] = {
        "filename": file.filename,
        "path": str(dest_path),
        "size_bytes": len(content),
        "uploaded_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    save_metadata(meta)
    
    # Trigger indexing with progress tracking
    background_tasks.add_task(index_document_task, file.filename, dest_path, meta)
    return {"status": "uploaded", "filename": file.filename}

# --- SSE Progress ---
@app.get("/index-progress")
async def index_progress():
    async def event_generator() -> AsyncGenerator:
        last_sent = None
        while True:
            current = indexing_progress.copy()
            if current != last_sent:
                yield {"data": json.dumps(current)}
                last_sent = current
            if current["progress"] >= 1.0:
                break
            await asyncio.sleep(0.2)
    return EventSourceResponse(event_generator())

# --- Retrieve ---
class RetrieveRequest(BaseModel):
    query: str

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG service not ready")
    if not vector_store.is_ready() and not bm25_store.is_ready():
        return {"chunks": [], "sources": [], "note": "No documents indexed yet"}
    
    results = retriever.retrieve(request.query)
    return {
        "chunks": [r["text"] for r in results],
        "sources": list({r["source"] for r in results}),
    }

# --- Internal helpers ---
def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}

def save_metadata(meta: dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2))

async def index_document_task(filename: str, path: Path, meta: dict):
    """Index a single newly uploaded document, then save indexes."""
    global indexing_progress
    async with indexing_lock:
        try:
            indexing_progress = {"progress": 0.05, "stage": "extracting"}
            processor = DocumentProcessor()
            text = processor.extract(path)
            
            indexing_progress = {"progress": 0.2, "stage": "chunking"}
            chunker = SemanticChunker(chunk_size=256, overlap_tokens=50)
            chunks = chunker.chunk(text, source=filename)
            
            indexing_progress = {"progress": 0.4, "stage": "embedding"}
            vector_store.add_chunks(chunks)
            
            indexing_progress = {"progress": 0.7, "stage": "bm25_indexing"}
            bm25_store.add_chunks(chunks)
            
            indexing_progress = {"progress": 0.9, "stage": "saving"}
            vector_store.save()
            bm25_store.save()
            
            indexing_progress = {"progress": 1.0, "stage": "done"}
        except Exception as e:
            logging.error(f"Indexing failed for {filename}: {e}")
            indexing_progress = {"progress": 1.0, "stage": "error", "error": str(e)}

async def rebuild_index_task(meta: dict):
    """Full index rebuild from all remaining documents. Called after deletion."""
    global indexing_progress
    async with indexing_lock:
        try:
            indexing_progress = {"progress": 0.05, "stage": "rebuilding"}
            vector_store.clear()
            bm25_store.clear()
            
            processor = DocumentProcessor()
            chunker = SemanticChunker(chunk_size=256, overlap_tokens=50)
            
            files = list(meta.values())
            for i, doc_meta in enumerate(files):
                path = Path(doc_meta["path"])
                progress = 0.1 + (0.8 * (i / max(len(files), 1)))
                indexing_progress = {"progress": progress, "stage": f"reindexing_{doc_meta['filename']}"}
                
                text = processor.extract(path)
                chunks = chunker.chunk(text, source=doc_meta["filename"])
                vector_store.add_chunks(chunks)
                bm25_store.add_chunks(chunks)
            
            indexing_progress = {"progress": 0.95, "stage": "saving"}
            vector_store.save()
            bm25_store.save()
            indexing_progress = {"progress": 1.0, "stage": "done"}
        except Exception as e:
            logging.error(f"Rebuild failed: {e}")
            indexing_progress = {"progress": 1.0, "stage": "error", "error": str(e)}
```

### 7.4 `rag/document_processor.py`

Handles extraction of raw text from uploaded file formats. Uses `fitz` (PyMuPDF) for PDFs, `python-docx` for DOCX, and direct file read for TXT/MD.

```python
from pathlib import Path
import fitz  # PyMuPDF
import docx

class DocumentProcessor:
    def extract(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".docx":
            return self._extract_docx(path)
        elif suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _extract_pdf(self, path: Path) -> str:
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return self._normalize("\n".join(pages))
    
    def _extract_docx(self, path: Path) -> str:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        return self._normalize("\n".join(paragraphs))
    
    def _normalize(self, text: str) -> str:
        import unicodedata, re
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
```

### 7.5 `rag/chunker.py`

Sentence-boundary aware chunker with token overlap. Uses NLTK sentence tokenizer.

```python
import nltk
from typing import List, Dict

# Download required NLTK data on first run
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

class SemanticChunker:
    def __init__(self, chunk_size: int = 256, overlap_tokens: int = 50):
        self.chunk_size = chunk_size  # approximate token count per chunk
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str, source: str) -> List[Dict]:
        """
        Split text into sentence-aware chunks.
        Each chunk dict: {"text": str, "source": str, "chunk_id": str}
        """
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0
        
        for sent in sentences:
            sent_tokens = len(sent.split())  # rough token estimate
            
            if current_len + sent_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": f"{source}::chunk_{chunk_idx}",
                })
                chunk_idx += 1
                
                # Carry forward last ~overlap_tokens worth of sentences
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
        
        # Flush remaining
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": source,
                "chunk_id": f"{source}::chunk_{chunk_idx}",
            })
        
        return chunks
```

### 7.6 `rag/embedder.py`

Wraps sentence-transformers BGE embedding model. Loads once, stays on GPU.

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # Force GPU if available
        self.model = SentenceTransformer(model_name, device="cuda")
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Returns float32 numpy array of shape (len(texts), dim)"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # Required for cosine similarity via inner product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding. BGE models prepend 'Represent this sentence...' prefix for retrieval."""
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]
```

### 7.7 `rag/vector_store.py`

FAISS index management. For V0, use IndexFlatIP (inner product = cosine similarity after normalization). IVF-PQ is more scalable but requires training on a corpus; for MVP document counts, Flat is fast enough and simpler.

```python
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
        self.chunk_store: List[Dict] = []  # Parallel list: chunk_store[i] = chunk dict for faiss index id i
    
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
            chunk["rank"] = len(results) + 1  # 1-indexed rank for RRF
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
```

### 7.8 `rag/bm25_store.py`

BM25 sparse keyword index using rank_bm25.

```python
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
        
        # Get top_k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == 0:
                continue
            chunk = self.chunk_store[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunk["rank"] = rank + 1  # 1-indexed for RRF
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
```

### 7.9 `rag/reranker.py`

Cross-encoder reranker using BGE reranker base.

```python
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
```

### 7.10 `rag/retriever.py`

Orchestrates hybrid retrieval: BM25 + FAISS → RRF fusion → reranker.

```python
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
        rrf_k: int = 60,  # RRF constant — standard value
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k
    
    def retrieve(self, query: str) -> List[Dict]:
        # 1. Parallel retrieval from both indexes
        dense_results = self.vector_store.search(query, top_k=self.top_k_retrieve)
        sparse_results = self.bm25_store.search(query, top_k=self.top_k_retrieve)
        
        # 2. Reciprocal Rank Fusion
        fused = self._rrf_fuse(dense_results, sparse_results)
        
        # 3. Rerank top candidates
        candidates = fused[:self.top_k_retrieve]
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k_rerank)
        
        return reranked
    
    def _rrf_fuse(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion formula:
        score(d) = sum(1 / (k + rank_i(d))) for each ranked list i
        """
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
```

---

## 8. LIVEKIT AGENT — DETAILED SPECIFICATION

### 8.1 `agent/Dockerfile`

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download VAD model files at build time (not at runtime)
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

COPY . .

CMD ["python", "agent.py", "start"]
```

### 8.2 `agent/requirements.txt`

```
livekit-agents[openai,silero]>=0.12.0
livekit-plugins-openai>=0.8.0
livekit-plugins-silero>=0.7.0
httpx>=0.27.0
```

**Note on plugin choices:**
- `livekit-plugins-openai` is used for both STT (pointing at Speaches) and LLM (pointing at llama.cpp) and TTS (pointing at Kokoro) — all three expose OpenAI-compatible APIs so the same plugin handles all three with just a `base_url` override.
- Silero VAD is the end-of-speech detector. It runs locally inside the agent container and consumes negligible GPU.

### 8.3 `agent/agent.py`

This is the full agent implementation. Read every comment — they explain why each decision was made.

```python
import asyncio
import logging
import os
import httpx
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, ChatContext, ChatMessage
from livekit.plugins import openai as lk_openai, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

LLAMA_CPP_BASE_URL = os.environ["LLAMA_CPP_BASE_URL"]   # e.g. http://100.x.x.x:8080/v1
RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]          # e.g. http://rag-service:8100
KOKORO_BASE_URL = os.environ["KOKORO_BASE_URL"]          # e.g. http://kokoro:8880/v1
AGENT_NAME = os.environ.get("AGENT_NAME", "rag-assistant")

SYSTEM_PROMPT = """You are a helpful voice assistant. You answer questions based on documents 
the user has uploaded. When context is provided, use it to answer accurately. 
Be concise — your answers will be spoken aloud. Avoid bullet points, markdown formatting, 
numbered lists, or special characters. Speak naturally in complete sentences. 
If you cannot find the answer in the provided context, say so clearly and briefly."""


class RAGAssistant(Agent):
    """
    The LiveKit Agent. Wraps a standard voice pipeline with a RAG context injection hook.
    
    The on_user_turn_completed method is called after STT transcribes the user's speech
    but BEFORE the LLM generates a response. This is where we inject RAG context.
    This approach adds ~100-200ms but avoids tool-call round trips. For V0, this is correct.
    """
    
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._http_client = httpx.AsyncClient(timeout=5.0)
    
    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Hook: called after VAD+STT produce the user's transcript.
        We call the RAG service and inject retrieved context into the chat context
        before the LLM sees the message.
        """
        query = new_message.text_content
        if not query:
            return
        
        try:
            response = await self._http_client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()
            chunks = data.get("chunks", [])
            sources = data.get("sources", [])
            
            if chunks:
                context_text = "\n\n---\n\n".join(chunks)
                source_note = f"\n\nSources: {', '.join(sources)}" if sources else ""
                injection = (
                    f"The following context was retrieved from the user's documents "
                    f"and is relevant to their question:\n\n{context_text}{source_note}"
                )
                turn_ctx.add_message(role="assistant", content=injection)
                logger.info(f"Injected {len(chunks)} RAG chunks from {sources}")
            else:
                logger.info("RAG returned no chunks for query")
        
        except httpx.RequestError as e:
            logger.warning(f"RAG service unreachable: {e}. Proceeding without context.")
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}. Proceeding without context.")
        
        # Important: we return None (not raise). If RAG fails, the LLM still responds.
        # Graceful degradation is intentional.


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def session_handler(ctx: agents.JobContext):
    """
    Entry point for each new LiveKit room session.
    Agent auto-dispatches when a client joins the room (dispatch model: automatic).
    """
    logger.info(f"New session in room: {ctx.room.name}")
    
    # STT: Speaches (faster-whisper) via OpenAI-compatible API
    stt = lk_openai.STT(
        base_url="http://speaches:8000/v1",
        api_key="not-required",           # Speaches doesn't require a key
        model="large-v3-turbo",           # Must match WHISPER_MODEL env var
        language="en",
    )
    
    # LLM: llama.cpp via OpenAI-compatible API
    # Note: streaming=True is critical for TTS sentence-boundary streaming
    llm = lk_openai.LLM(
        base_url=LLAMA_CPP_BASE_URL,
        api_key="not-required",
        model="gemma",                    # llama.cpp ignores this but it must be set
    )
    
    # TTS: Kokoro via OpenAI-compatible API
    tts = lk_openai.TTS(
        base_url=KOKORO_BASE_URL,
        api_key="not-required",
        model="kokoro",                   # Kokoro's model identifier
        voice="af_sky",                   # Default Kokoro voice (adjust as desired)
    )
    
    # VAD: Silero (runs inside agent container, CPU/GPU agnostic, very lightweight)
    vad = silero.VAD.load()
    
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
    )
    
    await session.start(
        room=ctx.room,
        agent=RAGAssistant(),
    )
    
    # Send initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and briefly. Tell them you're ready to answer questions about their documents."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
```

**Dispatch model clarification:** The `@server.rtc_session(agent_name=AGENT_NAME)` decorator configures automatic dispatch — the agent starts a session as soon as a client participant joins a LiveKit room. The client frontend creates/joins a room, and the agent picks it up automatically. No explicit dispatch API call needed.

---

## 9. FRONTEND — DETAILED SPECIFICATION

The frontend is a single `index.html` file served by the RAG service FastAPI app. It must be kept simple. No build step, no frameworks, no npm.

### 9.1 `rag-service/static/index.html`

The file must implement:

1. **LiveKit WebRTC connection** using `livekit-client` from CDN (`https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js`)
2. **Mic button** (push-to-talk or always-on toggle)
3. **Document upload** (`<input type="file">` accepting `.pdf, .docx, .txt, .md`)
4. **Document list** (fetched from `GET /documents` on page load)
5. **Delete buttons** per document
6. **Indexing progress bar** via SSE (`EventSource` on `/index-progress`)
7. **Query disable** while indexing is in progress
8. **Transcript display** (optional but helpful for debugging — display what the STT heard)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Voice Assistant</title>
  <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"></script>
  <style>
    /* ── Reset + Base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f0f;
      color: #e8e8e8;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem;
      gap: 2rem;
    }
    h1 { font-size: 1.5rem; font-weight: 600; color: #fff; }
    h2 { font-size: 1rem; font-weight: 500; color: #aaa; margin-bottom: 0.75rem; }

    /* ── Panel ── */
    .panel {
      width: 100%;
      max-width: 640px;
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 12px;
      padding: 1.5rem;
    }

    /* ── Mic button ── */
    #mic-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      width: 100%;
      padding: 1rem;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 500;
      background: #2563eb;
      color: #fff;
      transition: background 0.2s;
    }
    #mic-btn:hover { background: #1d4ed8; }
    #mic-btn.active { background: #dc2626; }
    #mic-btn:disabled { background: #444; cursor: not-allowed; }

    /* ── Status ── */
    #status { font-size: 0.85rem; color: #888; margin-top: 0.5rem; text-align: center; }

    /* ── Progress ── */
    .progress-wrap { margin-top: 0.75rem; }
    progress {
      width: 100%;
      height: 6px;
      border-radius: 3px;
      appearance: none;
    }
    progress::-webkit-progress-bar { background: #333; border-radius: 3px; }
    progress::-webkit-progress-value { background: #2563eb; border-radius: 3px; transition: width 0.3s; }
    #progress-label { font-size: 0.75rem; color: #888; margin-top: 0.25rem; }

    /* ── Upload ── */
    #upload-input { display: none; }
    #upload-btn {
      padding: 0.6rem 1.2rem;
      border-radius: 6px;
      border: 1px solid #3a3a3a;
      background: #252525;
      color: #e8e8e8;
      cursor: pointer;
      font-size: 0.9rem;
    }
    #upload-btn:hover { background: #2f2f2f; }

    /* ── Document list ── */
    #doc-list { list-style: none; display: flex; flex-direction: column; gap: 0.5rem; }
    #doc-list li {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.5rem 0.75rem;
      background: #222;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    #doc-list li .del-btn {
      background: none;
      border: none;
      color: #ef4444;
      cursor: pointer;
      font-size: 0.85rem;
      padding: 0.2rem 0.4rem;
    }
    #doc-list li .del-btn:hover { text-decoration: underline; }

    /* ── Transcript ── */
    #transcript {
      background: #111;
      border-radius: 6px;
      padding: 0.75rem;
      font-size: 0.85rem;
      color: #aaa;
      min-height: 60px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>

<h1>🎙 RAG Voice Assistant</h1>

<!-- Voice Panel -->
<div class="panel">
  <h2>Voice</h2>
  <button id="mic-btn" onclick="toggleSession()" disabled>
    <span id="mic-icon">▶</span>
    <span id="mic-label">Connect</span>
  </button>
  <div id="status">Initialising…</div>
</div>

<!-- Documents Panel -->
<div class="panel">
  <h2>Documents</h2>
  <input type="file" id="upload-input" accept=".pdf,.docx,.txt,.md" onchange="uploadFile(event)">
  <button id="upload-btn" onclick="document.getElementById('upload-input').click()">
    ＋ Upload document
  </button>

  <div class="progress-wrap" id="progress-wrap" style="display:none">
    <progress id="progress-bar" value="0" max="1"></progress>
    <div id="progress-label">Indexing…</div>
  </div>

  <ul id="doc-list" style="margin-top: 1rem;"></ul>
</div>

<!-- Transcript Panel -->
<div class="panel">
  <h2>Last transcript</h2>
  <div id="transcript">—</div>
</div>

<script>
  // ── Config ─────────────────────────────────────────────────────────────────
  // IMPORTANT: Replace LIVEKIT_SERVER_URL with the actual Tailscale IP of the server.
  // The frontend runs on the Mac browser. It connects directly to the LiveKit server
  // running on Ubuntu via Tailscale.
  const LIVEKIT_SERVER_URL = "ws://100.x.x.x:7880";  // <-- SET THIS
  const RAG_BASE = "";  // Same origin (frontend served by rag-service)

  // LiveKit token endpoint — the RAG service generates tokens
  const TOKEN_URL = `${RAG_BASE}/livekit-token`;

  // ── LiveKit State ───────────────────────────────────────────────────────────
  let room = null;
  let isConnected = false;
  let isIndexing = false;

  // ── Boot ────────────────────────────────────────────────────────────────────
  window.addEventListener("DOMContentLoaded", async () => {
    await loadDocuments();
    watchIndexingProgress();
    document.getElementById("mic-btn").disabled = false;
    setStatus("Ready");
  });

  // ── Voice session ───────────────────────────────────────────────────────────
  async function toggleSession() {
    if (isConnected) {
      await disconnectSession();
    } else {
      await connectSession();
    }
  }

  async function connectSession() {
    if (isIndexing) {
      setStatus("Please wait for indexing to complete.");
      return;
    }
    try {
      setStatus("Fetching token…");
      const res = await fetch(TOKEN_URL);
      const { token } = await res.json();

      room = new LivekitClient.Room();

      // When agent speaks, play audio on remote track
      room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === LivekitClient.Track.Kind.Audio) {
          const el = track.attach();
          document.body.appendChild(el);
        }
      });

      // Show transcripts from data messages if agent sends them
      room.on(LivekitClient.RoomEvent.DataReceived, (payload) => {
        try {
          const msg = JSON.parse(new TextDecoder().decode(payload));
          if (msg.type === "transcript") {
            document.getElementById("transcript").textContent = msg.text;
          }
        } catch (_) {}
      });

      await room.connect(LIVEKIT_SERVER_URL, token);
      await room.localParticipant.setMicrophoneEnabled(true);

      isConnected = true;
      updateMicButton(true);
      setStatus("Connected — speak now");
    } catch (err) {
      console.error("Connection error:", err);
      setStatus("Connection failed: " + err.message);
    }
  }

  async function disconnectSession() {
    if (room) {
      await room.disconnect();
      room = null;
    }
    // Remove any attached audio elements
    document.querySelectorAll("audio").forEach(el => el.remove());
    isConnected = false;
    updateMicButton(false);
    setStatus("Disconnected");
  }

  function updateMicButton(connected) {
    const btn = document.getElementById("mic-btn");
    const icon = document.getElementById("mic-icon");
    const label = document.getElementById("mic-label");
    btn.classList.toggle("active", connected);
    icon.textContent = connected ? "■" : "▶";
    label.textContent = connected ? "Disconnect" : "Connect";
  }

  function setStatus(msg) {
    document.getElementById("status").textContent = msg;
  }

  // ── Document management ─────────────────────────────────────────────────────
  async function loadDocuments() {
    const res = await fetch(`${RAG_BASE}/documents`);
    const data = await res.json();
    renderDocumentList(data.documents || []);
  }

  function renderDocumentList(docs) {
    const list = document.getElementById("doc-list");
    list.innerHTML = "";
    if (docs.length === 0) {
      list.innerHTML = "<li style='color:#666'>No documents indexed yet</li>";
      return;
    }
    docs.forEach(doc => {
      const li = document.createElement("li");
      const size = (doc.size_bytes / 1024).toFixed(1) + " KB";
      li.innerHTML = `
        <span>${doc.filename} <span style="color:#666;font-size:0.8em">${size}</span></span>
        <button class="del-btn" onclick="deleteDocument('${doc.filename}')">Delete</button>
      `;
      list.appendChild(li);
    });
  }

  async function uploadFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = "";  // Reset input

    const formData = new FormData();
    formData.append("file", file);

    setIndexingUI(true, 0, "uploading");
    try {
      const res = await fetch(`${RAG_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        alert("Upload failed: " + err.detail);
        setIndexingUI(false);
      }
      // SSE watcher takes over from here
    } catch (err) {
      alert("Upload error: " + err.message);
      setIndexingUI(false);
    }
  }

  async function deleteDocument(filename) {
    if (!confirm(`Delete "${filename}" from the index?`)) return;
    const res = await fetch(`${RAG_BASE}/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    if (res.ok) {
      await loadDocuments();
    } else {
      alert("Delete failed");
    }
  }

  // ── SSE Progress ─────────────────────────────────────────────────────────────
  function watchIndexingProgress() {
    const es = new EventSource(`${RAG_BASE}/index-progress`);
    es.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const progress = parseFloat(data.progress);
      const stage = data.stage || "";

      if (progress >= 1.0) {
        setIndexingUI(false);
        loadDocuments();
        es.close();
        // Re-open the SSE stream for future uploads
        setTimeout(watchIndexingProgress, 1000);
      } else {
        setIndexingUI(true, progress, stage);
      }
    };
    es.onerror = () => {
      // Reconnect silently on error
      setTimeout(watchIndexingProgress, 3000);
    };
  }

  function setIndexingUI(active, progress = 0, stage = "") {
    isIndexing = active;
    const wrap = document.getElementById("progress-wrap");
    const bar = document.getElementById("progress-bar");
    const label = document.getElementById("progress-label");
    const micBtn = document.getElementById("mic-btn");

    wrap.style.display = active ? "block" : "none";
    bar.value = progress;
    label.textContent = active ? `${stage.replace(/_/g, " ")} (${Math.round(progress * 100)}%)` : "";

    // Disable mic while indexing (prevent querying a partial index)
    micBtn.disabled = active && !isConnected;
  }
</script>
</body>
</html>
```

**Note:** The frontend needs a `/livekit-token` endpoint on the RAG service. Add this to `main.py`:

```python
from livekit import api as livekit_api
import time

LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

@app.get("/livekit-token")
async def get_livekit_token():
    """
    Generates a LiveKit participant token for the browser client.
    In production this would require authentication. For V0 single-user demo, open is fine.
    """
    token = livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity("user-" + str(int(time.time()))) \
        .with_name("User") \
        .with_grants(livekit_api.VideoGrants(
            room_join=True,
            room="demo-room",
        ))
    return {"token": token.to_jwt()}
```

Add `livekit` to `rag-service/requirements.txt`:
```
livekit>=0.11.0
```

The agent must be configured to join `room="demo-room"` (or whatever room name you standardize). The `@server.rtc_session` decorator will auto-dispatch when a participant joins.

---

## 10. LLM INTEGRATION (llama.cpp bare metal)

llama.cpp is already running on the Ubuntu host at `100.x.x.x:8080`. It serves an OpenAI-compatible API.

### 10.1 Verify llama.cpp is working

Before building anything, verify the endpoint responds:

```bash
curl http://100.x.x.x:8080/v1/models
```

Expected: a JSON response listing available models.

### 10.2 Verify streaming works

```bash
curl http://100.x.x.x:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Say hello"}],"stream":true}' \
  --no-buffer
```

Expected: streaming SSE-format chunks. If this works, the agent's LLM integration will work.

### 10.3 Important llama.cpp flags for low-latency

Ensure llama.cpp is launched with these flags for the session (or verify they are already set):
- `--n-gpu-layers 99` (full GPU offload for Gemma 4 E4B)
- `--ctx-size 4096` (adequate context window)
- `-t 4` (4 CPU threads for token sampling, reduces sampling bottleneck)
- `--parallel 1` (single-user, no parallel slots needed)

---

## 11. BUILD & STARTUP SEQUENCE

Follow this exact order. Each step must succeed before proceeding to the next.

### Step 1: Verify Ubuntu prerequisites

```bash
# Verify CUDA
nvidia-smi

# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Verify llama.cpp is running
curl http://localhost:8080/v1/models

# Verify Tailscale
tailscale status
```

### Step 2: Clone the repo on Ubuntu

```bash
# From Ubuntu server (all development happens here via Remote SSH)
git clone <your-repo-url> /srv/rag-assistant
cd /srv/rag-assistant
```

### Step 3: Create `.env`

```bash
cp .env.example .env
# Edit .env with correct Tailscale IP (run: tailscale ip -4 to get it)
nano .env
```

Replace `100.x.x.x` everywhere in `.env` with the actual Tailscale IP output of `tailscale ip -4` on the Ubuntu server.

### Step 4: First build

```bash
docker compose build
```

This will:
- Build the `rag-service` image (downloads BAAI/bge models to Docker layer cache)
- Build the `agent` image (downloads Silero VAD files)

Expect this to take 5-10 minutes on first run due to model downloads.

### Step 5: Start services

```bash
docker compose up -d
```

### Step 6: Verify services are up

```bash
# Check all containers running
docker compose ps

# Check rag-service health
curl http://localhost:8100/health

# Check speaches
curl http://localhost:8000/v1/models

# Check kokoro
curl http://localhost:8880/v1/models

# Check livekit
curl http://localhost:7880/

# Check agent logs (look for "RAG service ready" and connection to livekit)
docker compose logs -f agent
```

### Step 7: Access from Mac browser

Open on the Mac:
```
http://100.x.x.x:8100
```
(Where `100.x.x.x` is the Ubuntu Tailscale IP.)

You should see the RAG Voice Assistant frontend.

---

## 12. DEVELOPMENT WORKFLOW

### During active development

Use **Cursor or VS Code with Remote SSH** pointed at the Ubuntu Tailscale IP. Edit files directly on Ubuntu. Rebuild and restart individual services as needed:

```bash
# Rebuild and restart only the rag-service (most common during development)
docker compose up --build rag-service -d

# Rebuild and restart only the agent
docker compose up --build agent -d

# Tail logs for both
docker compose logs -f agent rag-service

# Full restart (rarely needed)
docker compose down && docker compose up -d
```

### Testing the RAG pipeline without voice

You can test retrieval directly:

```bash
# Upload a test document
curl -X POST http://localhost:8100/upload \
  -F "file=@/path/to/test.pdf"

# Query the retrieval endpoint
curl -X POST http://localhost:8100/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total Q1 revenue?"}'
```

### Testing the LLM without voice

```bash
curl http://100.x.x.x:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

### Testing STT

```bash
# Speaches health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

---

## 13. KNOWN TRADEOFFS & CONSTRAINTS

| Tradeoff | Decision | Rationale |
|---|---|---|
| Always-on RAG | Every query hits the retrieval pipeline | Simpler for MVP; adds ~100-200ms |
| FAISS Flat vs IVF-PQ | IndexFlatIP for V0 | IVF-PQ requires corpus training; Flat is accurate and fast for small corpora |
| No auth on LiveKit token endpoint | Open token generation | Single-user demo, not a production deployment |
| Full index rebuild on delete | Re-embeds all remaining documents on deletion | FAISS doesn't support in-place deletion; fine for small document sets |
| Silero VAD | Runs CPU-mode inside agent | Silero is tiny (~30MB), GPU not needed, avoids VRAM pressure |
| No conversational memory | Session context reset on reconnect | Out of scope for V0 |
| Kokoro voice: af_sky | Default voice | Can be changed by editing the agent |
| WHISPER_MODEL env var | Controls STT model | Change to `medium.en` in .env and restart speaches if VRAM is tight |

---

## 14. LATENCY TARGETS & PROFILING

**Target TTFA (Time-To-First-Audio):** 1-2 seconds from end-of-speech to first audio from TTS.

**Breakdown:**
| Stage | Expected latency |
|---|---|
| Silero VAD detects end-of-speech | ~50-100ms |
| STT transcription (faster-whisper GPU) | ~200-400ms |
| RAG retrieval (FAISS + BM25 + reranker) | ~100-200ms |
| LLM first token (llama.cpp, Gemma 4 E4B, CUDA) | ~200-400ms |
| Kokoro TTS first audio chunk | ~100-200ms |
| WebRTC transmission | ~10-30ms |
| **Total estimated TTFA** | **~660ms - 1.3s** |

If latency is outside this range, profile in this order:
1. Check if llama.cpp is fully GPU-offloaded (`--n-gpu-layers 99`)
2. Check FAISS index size (more chunks = slower search)
3. Consider dropping `TOP_K_RETRIEVE` from 20 to 10
4. Consider dropping `faster-whisper large-v3-turbo` to `medium.en`

---

## 15. FILE CHECKLIST FOR THE CODING AGENT

Before calling the project done, verify every file listed below exists and is correct:

```
☐ docker-compose.yml
☐ .env.example
☐ .gitignore  (must exclude .env, storage/documents/, storage/index/)
☐ livekit/livekit.yaml
☐ agent/Dockerfile
☐ agent/requirements.txt
☐ agent/agent.py
☐ rag-service/Dockerfile
☐ rag-service/requirements.txt
☐ rag-service/main.py
☐ rag-service/rag/__init__.py
☐ rag-service/rag/document_processor.py
☐ rag-service/rag/chunker.py
☐ rag-service/rag/embedder.py
☐ rag-service/rag/vector_store.py
☐ rag-service/rag/bm25_store.py
☐ rag-service/rag/reranker.py
☐ rag-service/rag/retriever.py
☐ rag-service/static/index.html
☐ rag-service/storage/documents/  (empty dir, created by volume mount)
☐ rag-service/storage/index/      (empty dir, created by volume mount)
```

---

## 16. CONSTRAINTS THE CODING AGENT MUST NEVER VIOLATE

1. **llama.cpp is bare-metal.** Do not add it to docker-compose. It is already running. The agent accesses it via `LLAMA_CPP_BASE_URL`.

2. **Everything GPU-capable runs on GPU.** This means speaches, kokoro, rag-service embedding, and rag-service reranker all use the `deploy.resources.reservations.devices` GPU spec in docker-compose.

3. **No external API calls.** No OpenAI API key, no Anthropic API, no Hugging Face inference API, no cloud STT. All inference is local.

4. **No LiveKit Cloud.** The `livekit` container is self-hosted. `LIVEKIT_URL` must point to the Ubuntu Tailscale IP.

5. **Single docker-compose.yml.** All services except llama.cpp live here.

6. **Frontend is one HTML file.** No build step, no npm, no framework. Served statically by FastAPI.

7. **WHISPER_MODEL must be configurable via env var** without code changes. Same for EMBEDDING_MODEL and RERANKER_MODEL.

8. **Graceful RAG failure.** If the RAG service is unreachable during a query, the agent logs a warning and continues with the LLM's general knowledge. It does not crash or halt.

9. **SSE progress must gate queries.** The frontend must disable the mic button (or block connect) while `indexing_progress["progress"] < 1.0` and the user is not already connected.

10. **Delete triggers silent background rebuild.** No SSE progress for deletion rebuilds. The operation is silent (user sees the document disappear from the list; the index rebuilds in the background without blocking the UI).

---

*End of implementation specification. If anything is ambiguous, flag it before building. The architecture described here has been deliberately designed — do not introduce additional services, frameworks, or dependencies without explicit approval.*
