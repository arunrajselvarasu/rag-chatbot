"""FastAPI app. Routes (all under /api) match the React frontend:

GET    /api/documents            -> {"documents": ["a.pdf"]}
POST   /api/upload               multipart field "file"
POST   /api/chat                 {"question", "conversation_id"} -> {"answer", "sources", "conversation_id"}
DELETE /api/documents/{filename}  removes a document
DELETE /api/chat/{conversation_id}  clears that conversation's memory
"""
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .config import load_settings
from .graph import build_graph
from .vectorstore import IngestionError, VectorStoreManager

logger = logging.getLogger("rag")


@dataclass
class Services:
    store: VectorStoreManager
    graph: object
    memory: MemorySaver


def assemble(store, llm) -> Services:
    memory = MemorySaver()  # in-memory LangGraph checkpointer = conversation memory
    return Services(store, build_graph(store, llm, memory), memory)


def build_services() -> Services:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    s = load_settings()  # raises at startup if .env is incomplete
    embeddings = HuggingFaceEmbeddings(
        model_name=s.embedding_model, encode_kwargs={"normalize_embeddings": True})
    llm = ChatGroq(model=s.groq_model, api_key=s.groq_api_key,
                   temperature=0, timeout=30, max_retries=2)
    return assemble(VectorStoreManager(embeddings), llm)


class ChatRequest(BaseModel):
    question: str = ""
    conversation_id: Optional[str] = Field(default=None, max_length=100)


def llm_http_error(exc: Exception) -> HTTPException:
    logger.exception("RAG request failed")  # details stay in the server log only
    name, text = type(exc).__name__.lower(), str(exc).lower()
    if "authentication" in name or "invalid_api_key" in text or "401" in text:
        return HTTPException(502, "The language model service rejected the configured credentials.")
    if ("model" in text and any(k in text for k in ("not found", "decommissioned", "does not exist", "no longer supported"))) \
            or "notfound" in name:
        return HTTPException(502, "The configured GROQ_MODEL is not available. Update GROQ_MODEL in Backend/.env.")
    if "ratelimit" in name or "429" in text:
        return HTTPException(429, "The language model is rate limited. Please try again shortly.")
    if "timeout" in name or "timed out" in text:
        return HTTPException(504, "The language model timed out. Please try again.")
    return HTTPException(503, "The language model service is currently unavailable.")


def create_app(services: Optional[Services] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        yield

    app = FastAPI(title="Simple Chat RAG", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/documents")
    def documents(request: Request):
        return {"documents": request.app.state.services.store.documents()}

    @app.delete("/api/documents/{filename}")
    def delete_document(filename: str, request: Request):
        try:
            request.app.state.services.store.delete(filename)
        except IngestionError as exc:
            raise HTTPException(exc.status_code, str(exc))
        except Exception:
            logger.exception("Delete failed")
            raise HTTPException(500, "The document could not be deleted.")
        return {"deleted": filename}

    @app.post("/api/upload")
    def upload(request: Request, file: UploadFile = File(...)):
        data = file.file.read()
        try:
            chunks = request.app.state.services.store.ingest(file.filename, data)
        except IngestionError as exc:
            raise HTTPException(exc.status_code, str(exc))
        except Exception as exc:
            logger.exception("Upload failed")
            raise HTTPException(500, "The document could not be processed.")
        return {"filename": file.filename, "chunks": chunks, "message": "Document indexed."}

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request):
        svc = request.app.state.services
        cid = req.conversation_id or uuid.uuid4().hex
        try:
            result = svc.graph.invoke({"question": req.question},
                                      config={"configurable": {"thread_id": cid}})
        except Exception as exc:
            raise llm_http_error(exc)
        if not result.get("valid", False):
            raise HTTPException(400, result.get("error") or "Invalid question.")
        return {"answer": result["answer"], "sources": result.get("sources", []),
                "grounded": bool(result.get("grounded")), "conversation_id": cid}

    @app.delete("/api/chat/{conversation_id}")
    def clear(conversation_id: str, request: Request):
        request.app.state.services.memory.delete_thread(conversation_id)
        return {"cleared": conversation_id}

    return app


app = create_app()