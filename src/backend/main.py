"""FastAPI ATS backend — sequential resume screening against local Ollama.

Run (Windows):
  uv run uvicorn src.backend.main:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)

# Allow `from src.backend...` when launched as uvicorn src.backend.main:app
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError

from src.backend.llm import get_llm_client, get_llm_model_id, screen_resume
from src.backend.parser import extract_text_from_bytes
from src.backend.recommend import get_recommendation
from src.backend.schemas import ScreeningResult

app = FastAPI(title="AI Resume Screener ATS", version="1.0.0")

_frontend = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ats-backend"}


@app.get("/api/health")
def api_health() -> dict:
    """Backend + Ollama (OpenAI-compatible) reachability."""
    llm: dict = {"reachable": False}
    try:
        from src.backend.llm import get_llm_base_url

        client = get_llm_client()
        models = client.models.list()
        ids = [m.id for m in models.data]
        llm = {
            "reachable": True,
            "base_url": get_llm_base_url(),
            "models": ids,
            "configured_model": get_llm_model_id(),
        }
    except Exception as exc:
        llm = {"reachable": False, "error": str(exc)}
    return {"status": "ok", "llm": llm}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/screen")
async def screen(
    jd_text: str | None = Form(default=None),
    jd_file: UploadFile | None = File(default=None),
    resumes: list[UploadFile] = File(...),
) -> StreamingResponse:
    """Screen resumes one-by-one (never parallel) and stream SSE progress."""
    jd_body = (jd_text or "").strip()
    if jd_file is not None and jd_file.filename:
        jd_bytes = await jd_file.read()
        if jd_bytes:
            try:
                extracted = extract_text_from_bytes(jd_bytes, jd_file.filename)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Could not parse JD file: {exc}"
                ) from exc
            if extracted:
                jd_body = extracted

    if not jd_body:
        raise HTTPException(
            status_code=400,
            detail="Provide a job description (paste text or upload PDF/TXT).",
        )

    packed: list[tuple[str, bytes]] = []
    for upload in resumes:
        if not upload.filename:
            continue
        data = await upload.read()
        if data:
            packed.append((upload.filename, data))

    if not packed:
        raise HTTPException(status_code=400, detail="Upload at least one resume PDF.")

    try:
        get_llm_model_id()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def event_stream():
        total = len(packed)
        # Sequential queue — one Ollama request at a time (6GB VRAM).
        for index, (filename, data) in enumerate(packed, start=1):
            yield _sse(
                {
                    "type": "progress",
                    "current": index,
                    "total": total,
                    "filename": filename,
                    "message": f"Scanning candidate {index} of {total}...",
                }
            )
            try:
                resume_text = extract_text_from_bytes(data, filename)
                if not resume_text:
                    raise ValueError("No text extracted from PDF")
                payload = screen_resume(jd_body, resume_text, filename)
                result = ScreeningResult(
                    filename=filename,
                    candidate_name=payload.candidate_name,
                    score=payload.score,
                    recommendation=get_recommendation(payload.score),
                    confidence_level=payload.confidence_level,
                    flag_for_human=payload.flag_for_human,
                    quick_summary=payload.quick_summary
                    or payload.strengths[:180],
                    strengths=payload.strengths,
                    weaknesses=payload.weaknesses,
                    raw_json=payload.model_dump(),
                )
                yield _sse({"type": "result", "result": result.model_dump()})
            except (APIConnectionError, APIStatusError) as exc:
                yield _sse(
                    {
                        "type": "error",
                        "filename": filename,
                        "message": f"Ollama request failed: {exc}",
                    }
                )
            except Exception as exc:
                yield _sse(
                    {
                        "type": "error",
                        "filename": filename,
                        "message": str(exc),
                    }
                )

        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
