# RAG Voice Assistant

A real-time voice assistant powered by hybrid RAG (Retrieval-Augmented Generation). Speak questions about your documents and get spoken answers. Speech-to-text, text-to-speech and the LLM are served by Tarka's model API; retrieval runs in-stack.

## Architecture

```
Browser  ←WebRTC→  LiveKit (media router)
                      └── LiveKit Agent (orchestrator)
                            ├── RAG Service (FAISS + BM25 + Reranker, CPU)
                            └── Tarka model API (http://tarka/v1)
                                  ├── STT: whisper-large-v3
                                  ├── LLM: qwen3.8-flash-next
                                  └── TTS: kokoro / af_heart
```

## Key Features

- **Real-time voice** via WebRTC (browser mic → agent → browser speaker)
- **Hybrid RAG**: Dense (FAISS) + Sparse (BM25) + RRF fusion + Cross-encoder reranking
- **Document management**: Upload, index, list, delete with SSE progress tracking
- **No GPU required**: all heavy inference goes through Tarka
- **Single-user, single-room** demo setup

## Deploying to Tarka Hosted Agents

1. Push to `main`. `.github/workflows/images.yml` builds and pushes
   `ghcr.io/gn-5/rag-voice-assistant-{agent,rag-service}` tagged with the short SHA and `latest`.
2. In the Tarka console (Compute → Hosted Agents), create a custom Compose host from
   `tarka/compose.yaml`, with `qwen3.8-flash-next` plus the `whisper-large-v3` and `kokoro` utilities selected.
3. Add GHCR credentials (GitHub username + classic PAT with `read:packages`) if the packages are private.
4. Set variables: `IMAGE_TAG`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (secret), and optionally
   `LIVEKIT_PUBLIC_URL` / `LIVEKIT_SERVER_URL` for an external LiveKit server.

> **Known limitation:** Tarka only exposes HTTPS routes, so the in-stack LiveKit can do signaling
> but not WebRTC media (UDP/TCP). Voice sessions need an externally reachable LiveKit server.

## Local Development

```bash
cp .env.example .env      # set TARKA_API_KEY (utilities scope)
docker compose up --build -d
# Open https://localhost (self-signed cert)
docker compose logs -f agent rag-service
```

## Project Structure

```
├── docker-compose.yml     # local dev
├── tarka/compose.yaml     # Tarka Hosted Agents stack
├── .github/workflows/     # image builds → GHCR
├── .env.example
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
├── nginx/                 # local HTTPS only
└── README.md
```

## License

Private — all rights reserved.
