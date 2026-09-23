"""Pydantic models for ATS screening (LLM output + API results)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _as_plain_text(value: Any) -> str:
    """Ollama often returns strengths/weaknesses as a JSON list; UI expects a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_as_plain_text(item) for item in value]
        return "; ".join(p for p in parts if p)
    return str(value).strip()


class LLMScreeningPayload(BaseModel):
    """Fields the local LLM is allowed to emit (no recommendation)."""

    score: int = Field(..., ge=0, le=100)
    candidate_name: str = Field(..., min_length=1)
    confidence_level: Literal["High", "Medium", "Low"] = "Medium"
    flag_for_human: bool = False
    strengths: str = Field(..., min_length=1)
    weaknesses: str = Field(..., min_length=1)
    quick_summary: str = Field(default="", description="One-line recruiter blurb.")

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("score must be an integer 0-100")
        if isinstance(value, str):
            value = value.strip()
        return int(float(value))

    @field_validator("confidence_level", mode="before")
    @classmethod
    def coerce_confidence(cls, value: Any) -> str:
        text = str(value or "Medium").strip().capitalize()
        if text not in {"High", "Medium", "Low"}:
            return "Medium"
        return text

    @field_validator("flag_for_human", mode="before")
    @classmethod
    def coerce_flag(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @field_validator("candidate_name", "strengths", "weaknesses", "quick_summary", mode="before")
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        return _as_plain_text(value)


class ScreeningResult(BaseModel):
    filename: str
    candidate_name: str
    score: int
    recommendation: str
    confidence_level: str
    flag_for_human: bool
    quick_summary: str
    strengths: str
    weaknesses: str
    raw_json: dict
