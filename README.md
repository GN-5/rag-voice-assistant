# RAG Voice Assistant

A privacy-first, self-hosted, real-time voice assistant powered by hybrid RAG (Retrieval-Augmented Generation). Speak questions about your documents and get spoken answers — all running on your own hardware with zero cloud dependencies.

## Architecture

```
Mac Browser (Client)  ←WebRTC→  Ubuntu Server (Backend)
                                      ├── LiveKit (WebRTC media router)
                                      ├── Speaches (faster-whisper STT, GPU)
                                      ├── Kokoro (TTS, GPU)
                                      ├── RAG Service (FAISS + BM25 + Reranker, GPU)
                                      ├── LiveKit Agent (orchestrator)
                                      └── llama.cpp (LLM, bare-metal, GPU)
```

## Key Features

- **Real-time voice** via WebRTC (browser mic → server → browser speaker)
- **Hybrid RAG**: Dense (FAISS) + Sparse (BM25) + RRF fusion + Cross-encoder reranking
- **Document management**: Upload, index, list, delete with SSE progress tracking
- **Fully self-hosted**: No external API calls, all inference stays local
- **Single-user, single-room** demo setup over Tailscale

## Prerequisites

- Ubuntu Server 24.04 LTS with NVIDIA RTX 3060 (12GB)
- Docker + nvidia-container-toolkit installed
- Tailscale running on both client and server
- llama.cpp already running bare-metal on `http://100.x.x.x:8080`

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/GN-5/rag-voice-assistant.git
cd rag-voice-assistant
cp .env.example .env
# Edit .env with your Tailscale IP

# 2. Build and start
docker compose build
docker compose up -d

# 3. Access from Mac browser
# Open http://<tailscale-ip>:8100
```

## Development

Use Remote SSH from your Mac to the Ubuntu server. Edit files directly, then:

```bash
docker compose up --build rag-service -d   # Rebuild RAG service
docker compose up --build agent -d         # Rebuild agent
docker compose logs -f agent rag-service   # Tail logs
```

## Project Structure

```
├── docker-compose.yml
├── .env.example
├── livekit/livekit.yaml
├── agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── agent.py
├── rag-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── rag/          # RAG pipeline modules
│   └── static/       # Single-page frontend
└── README.md
```

## License

Private — all rights reserved.
