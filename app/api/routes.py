import json
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.core.db import checkpointer
from app.rag.workflow import ask, ask_stream
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    thread_id: str = Field(default="default", max_length=100)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question, payload.thread_id)
        write_audit(payload.thread_id, payload.question, result["answer"], result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    def events():
        try:
            for event in ask_stream(payload.question, payload.thread_id):
                if event["type"] == "done":
                    write_audit(payload.thread_id, payload.question, event["answer"], event["source_used"], event["trace"])
                yield json.dumps(event) + "\n"
        except Exception:
            logger.exception("Streaming chat failed")
            yield json.dumps({"type": "error", "data": "Something went wrong. Please try again."}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.delete("/chat/{thread_id}")
def delete_chat(thread_id: str):
    checkpointer.delete_thread(thread_id)
    return {"deleted": thread_id}


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
