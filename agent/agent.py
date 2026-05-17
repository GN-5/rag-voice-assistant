import json
import logging
import os
import time
import httpx
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, llm
from livekit.plugins import openai as lk_openai, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

LLAMA_CPP_BASE_URL = os.environ["LLAMA_CPP_BASE_URL"]
RAG_SERVICE_URL = os.environ["RAG_SERVICE_URL"]
KOKORO_BASE_URL = os.environ["KOKORO_BASE_URL"]
AGENT_NAME = os.environ.get("AGENT_NAME", "rag-assistant")

SYSTEM_PROMPT = """You are a helpful voice assistant. You answer questions based on documents 
the user has uploaded. When context is provided, use it to answer accurately. 
Be concise — your answers will be spoken aloud. Avoid bullet points, markdown formatting, 
numbered lists, or special characters. Speak naturally in complete sentences. 
If you cannot find the answer in the provided context, say so clearly and briefly."""


class RAGAssistant(Agent):
    def __init__(self, room=None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._room = room
    
    async def _broadcast(self, msg_type: str, **kwargs):
        if not self._room:
            return
        payload = {"type": msg_type, **kwargs}
        await self._room.local_participant.publish_data(
            json.dumps(payload).encode(),
        )
    
    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        query = new_message.text_content
        if not query:
            return
        
        await self._broadcast("transcript", text=query)
        await self._broadcast("state", state="thinking")
        
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
                # Workaround for livekit-agents timestamp bug:
                # Without created_at < new_message.created_at, the injected message
                # lands AFTER the user message in the final chat context sent to the LLM.
                # See: https://github.com/livekit/agents/issues/5053
                turn_ctx.add_message(
                    role="assistant",
                    content=injection,
                    created_at=new_message.created_at - 0.001,
                )
                logger.info(f"Injected {len(chunks)} RAG chunks from {sources}")
            else:
                logger.info("RAG returned no chunks for query")
        
        except httpx.RequestError as e:
            logger.warning(f"RAG service unreachable: {e}. Proceeding without context.")
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}. Proceeding without context.")


server = AgentServer()


@server.rtc_session()
async def session_handler(ctx: agents.JobContext):
    logger.info(f"New session in room: {ctx.room.name}")
    
    stt = lk_openai.STT(
        base_url="http://speaches:8000/v1",
        api_key="not-required",
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        language="en",
    )
    logger.info("STT initialized with Systran/faster-whisper-large-v3-turbo")
    
    llm_model = lk_openai.LLM(
        base_url=LLAMA_CPP_BASE_URL,
        api_key="not-required",
        model="gemma",
    )
    logger.info(f"LLM initialized with llama.cpp at {LLAMA_CPP_BASE_URL}")
    
    tts = lk_openai.TTS(
        base_url=KOKORO_BASE_URL,
        api_key="not-required",
        model="kokoro",
        voice="af_sky",
    )
    logger.info(f"TTS initialized with Kokoro at {KOKORO_BASE_URL}")
    
    vad = silero.VAD.load()
    
    session = AgentSession(
        stt=stt,
        llm=llm_model,
        tts=tts,
        vad=vad,
    )
    
    await session.start(
        room=ctx.room,
        agent=RAGAssistant(room=ctx.room),
    )
    logger.info("AgentSession started")
    
    await session.generate_reply(
        instructions="Greet the user warmly and briefly. Tell them you're ready to answer questions about their documents."
    )
    logger.info("Greeting generated")


if __name__ == "__main__":
    agents.cli.run_app(server)
