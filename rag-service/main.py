import asyncio, os, json, pickle, logging, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from livekit.api import AccessToken, VideoGrants

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

embedder: Embedder = None
reranker: Reranker = None
vector_store: VectorStore = None
bm25_store: BM25Store = None
retriever: HybridRetriever = None

indexing_progress: dict = {"progress": 1.0, "stage": "idle"}
indexing_lock = asyncio.Lock()

LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]


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

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory="/app/static"), name="static")


@app.get("/")
def root():
    return FileResponse("/app/static/index.html")


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
    
    background_tasks.add_task(rebuild_index_task, meta)
    return {"status": "deleted", "message": f"{filename} removed. Index rebuilding in background."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
    asyncio.create_task(index_document_task(file.filename, dest_path, meta))
    return {"status": "uploaded", "filename": file.filename}


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


@app.get("/livekit-token")
async def get_livekit_token():
    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity("user-" + str(int(time.time()))) \
        .with_name("User") \
        .with_grants(VideoGrants(
            room_join=True,
            room="demo-room",
        ))
    return {"token": token.to_jwt()}


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata(meta: dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2))


async def index_document_task(filename: str, path: Path, meta: dict):
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
