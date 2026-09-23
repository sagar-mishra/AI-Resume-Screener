"""OpenAI-compatible client for local Ollama (Windows native)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from openai import OpenAI

from src.backend.schemas import LLMScreeningPayload

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL_ID = "resume-screener"
# Ollama's OpenAI shim does not require a real key; the SDK still validates presence.
DEFAULT_LLM_API_KEY = "ollama"

SYSTEM_PROMPT = """\
You are a resume screening engine. Compare the resume to the job description.

Return ONLY a JSON object (no markdown) with these keys:
- score: integer 0-100 (how well the candidate matches the JD)
- candidate_name: string (best-effort name from the resume; else filename-like)
- confidence_level: "High", "Medium", or "Low"
- flag_for_human: boolean (true if ambiguous or incomplete)
- strengths: string of green flags (skills/experience that match the JD)
- weaknesses: string of red flags (missing or weak vs the JD)
- quick_summary: one sentence for a recruiter table

Do NOT include a recommendation field. Score only; the server maps the tier.
"""


def get_llm_base_url() -> str:
    return (os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).strip().rstrip("/")


def get_llm_model_id() -> str:
    model_id = (os.getenv("LLM_MODEL_ID") or DEFAULT_LLM_MODEL_ID).strip()
    if not model_id:
        raise RuntimeError(
            "LLM_MODEL_ID is missing or empty. Set it in .env "
            f"(default: {DEFAULT_LLM_MODEL_ID})."
        )
    return model_id


def get_llm_client() -> OpenAI:
    """Talk to Ollama's OpenAI-compatible API (no real auth required)."""
    return OpenAI(
        base_url=get_llm_base_url(),
        api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY) or DEFAULT_LLM_API_KEY,
        timeout=180.0,
    )


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(f"Model did not return JSON: {text[:400]}") from None
        return json.loads(match.group(0))


def screen_resume(
    job_description: str,
    resume_text: str,
    filename: str,
) -> LLMScreeningPayload:
    """Call Ollama once for a single resume. Never parallelize this."""
    client = get_llm_client()
    model_id = get_llm_model_id()

    user = (
        f"## Job Description\n\n{job_description[:6000]}\n\n"
        f"## Resume ({filename})\n\n{resume_text[:8000]}\n\n"
        "Return the JSON object now."
    )

    kwargs: dict = {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some Ollama builds reject response_format; retry as plain chat.
        response = client.chat.completions.create(**kwargs)

    content = (response.choices[0].message.content or "").strip()
    raw = _parse_json(content)
    if not raw.get("candidate_name"):
        raw["candidate_name"] = Path(filename).stem.replace("_", " ").title()
    payload = LLMScreeningPayload.model_validate(raw)
    if not payload.quick_summary:
        payload.quick_summary = payload.strengths[:180]
    return payload
