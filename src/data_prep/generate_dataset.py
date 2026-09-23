"""Generate ChatML JSONL training data by screening resumes with a teacher LLM.

Reads sample JD + raw resume PDFs, calls a configurable teacher provider
(OpenAI, Anthropic, or any OpenAI-compatible endpoint) with a structured
Pydantic schema, and appends one ChatML example per resume to data/processed/.

Configuration (project-root .env) — never hardcode keys or model IDs:
  TEACHER_PROVIDER=openai|anthropic|auto
  TEACHER_MODEL_ID=<any model id available on that provider>
  OPENAI_API_KEY / OPENAI_BASE_URL
  ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
"""

from __future__ import annotations

from dotenv import load_dotenv

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
import instructor
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project .env must win over shell vars (e.g. local LLM proxy base URLs).
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)

JD_PATH = PROJECT_ROOT / "data" / "job_descriptions" / "sample_jd.txt"
RESUME_DIR = PROJECT_ROOT / "data" / "raw_resumes"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "resume_training_data.jsonl"

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

PLACEHOLDER_KEYS = {
    "",
    "your_openai_api_key_here",
    "your_anthropic_api_key_here",
    "sk-...",
}

ProviderName = Literal["openai", "anthropic"]

ConfidenceLevel = Literal["High", "Medium", "Low"]
Recommendation = Literal[
    "Strong Shortlist",
    "Shortlist",
    "Interview Recommended",
    "Consider",
    "Borderline",
    "Hold",
    "Upskill & Reapply",
    "Alternative Role Recommended",
    "Reject",
]

# Exact 9-tier HR scale from AGENTS.md (score range inclusive).
RECOMMENDATION_TIERS: list[tuple[int, int, Recommendation]] = [
    (95, 100, "Strong Shortlist"),
    (85, 94, "Shortlist"),
    (75, 84, "Interview Recommended"),
    (65, 74, "Consider"),
    (55, 64, "Borderline"),
    (45, 54, "Hold"),
    (35, 44, "Upskill & Reapply"),
    (20, 34, "Alternative Role Recommended"),
    (0, 19, "Reject"),
]

SYSTEM_PROMPT = """\
You are an expert technical recruiter and resume screener for Senior Python/AI \
Engineer roles. Evaluate how well a candidate's resume matches the provided \
job description.

Rules:
- Be objective and specific; cite concrete skills, years, and projects.
- score must be an integer from 0 to 100.
- confidence_level must be exactly one of: High, Medium, Low.
- flag_for_human is true when the case is ambiguous, borderline, incomplete, \
or needs recruiter judgment.
- strengths and weaknesses are concise plain-language strings.
- recommendation MUST match the score using this exact 9-tier scale:
  * 95-100: Strong Shortlist
  * 85-94: Shortlist
  * 75-84: Interview Recommended
  * 65-74: Consider
  * 55-64: Borderline
  * 45-54: Hold
  * 35-44: Upskill & Reapply
  * 20-34: Alternative Role Recommended
  * 0-19: Reject
Return only fields defined by the response schema.\
"""


class ResumeScreeningResult(BaseModel):
    """Strict 6-key JSON output schema from AGENTS.md."""

    score: int = Field(..., ge=0, le=100, description="Match score from 0 to 100.")
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Model confidence: High, Medium, or Low.",
    )
    flag_for_human: bool = Field(
        ...,
        description="True if a human recruiter should review this candidate.",
    )
    strengths: str = Field(..., min_length=1, description="Key strengths vs the JD.")
    weaknesses: str = Field(..., min_length=1, description="Gaps or risks vs the JD.")
    recommendation: Recommendation = Field(
        ...,
        description="Exact HR-tier label corresponding to the score band.",
    )

    @field_validator("strengths", "weaknesses")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    @model_validator(mode="after")
    def align_recommendation_with_score(self) -> ResumeScreeningResult:
        expected = recommendation_for_score(self.score)
        if self.recommendation != expected:
            # Enforce AGENTS.md tier mapping for clean training labels.
            self.recommendation = expected
        return self


@dataclass(frozen=True)
class TeacherConfig:
    """Resolved teacher LLM settings from environment variables."""

    provider: ProviderName
    model_id: str
    api_key: str
    base_url: str


def recommendation_for_score(score: int) -> Recommendation:
    for low, high, label in RECOMMENDATION_TIERS:
        if low <= score <= high:
            return label
    raise ValueError(f"score out of range: {score}")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _is_usable_key(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip() not in PLACEHOLDER_KEYS


def resolve_teacher_config() -> TeacherConfig:
    """Load provider/model/keys from env; auto-pick a usable provider when asked."""
    raw_provider = (_env("TEACHER_PROVIDER", "auto") or "auto").lower()
    openai_key = _env("OPENAI_API_KEY")
    anthropic_key = _env("ANTHROPIC_API_KEY")
    has_openai = _is_usable_key(openai_key)
    has_anthropic = _is_usable_key(anthropic_key)

    if raw_provider == "auto":
        if has_openai:
            provider: ProviderName = "openai"
        elif has_anthropic:
            provider = "anthropic"
        else:
            raise SystemExit(
                "TEACHER_PROVIDER=auto but no usable API key found.\n"
                "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env "
                "(see .env.example)."
            )
    elif raw_provider in {"openai", "anthropic"}:
        provider = raw_provider  # type: ignore[assignment]
    else:
        raise SystemExit(
            f"Unsupported TEACHER_PROVIDER={raw_provider!r}. "
            "Use one of: openai, anthropic, auto."
        )

    if provider == "openai":
        if not has_openai:
            raise SystemExit(
                "TEACHER_PROVIDER=openai but OPENAI_API_KEY is missing or a placeholder.\n"
                "Set OPENAI_API_KEY in the project-root .env file."
            )
        model_id = _env("TEACHER_MODEL_ID") or DEFAULT_OPENAI_MODEL
        base_url = _env("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        assert openai_key is not None
        return TeacherConfig(
            provider="openai",
            model_id=model_id,
            api_key=openai_key,
            base_url=base_url.rstrip("/"),
        )

    if not has_anthropic:
        raise SystemExit(
            "TEACHER_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing or a placeholder.\n"
            "Set ANTHROPIC_API_KEY in the project-root .env file."
        )
    model_id = _env("TEACHER_MODEL_ID") or DEFAULT_ANTHROPIC_MODEL
    base_url = _env("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
    assert anthropic_key is not None
    return TeacherConfig(
        provider="anthropic",
        model_id=model_id,
        api_key=anthropic_key,
        base_url=base_url.rstrip("/"),
    )


def build_instructor_client(config: TeacherConfig) -> Any:
    """Build an instructor client for the selected provider.

    OpenAI path also works with OpenAI-compatible gateways (Groq, Together,
    Azure OpenAI, vLLM, Ollama, etc.) when OPENAI_BASE_URL points at them.
    """
    if config.provider == "openai":
        from openai import OpenAI

        raw = OpenAI(api_key=config.api_key, base_url=config.base_url)
        return instructor.from_openai(raw)

    from anthropic import Anthropic

    raw = Anthropic(api_key=config.api_key, base_url=config.base_url)
    return instructor.from_anthropic(raw)


def format_api_error(exc: BaseException, config: TeacherConfig) -> str:
    """Unwrap nested SDK/instructor errors into an actionable message."""
    parts = [f"{type(exc).__name__}: {exc}"]
    cause: BaseException | None = exc.__cause__ or exc.__context__
    depth = 0
    while cause is not None and depth < 6:
        parts.append(f"  caused by {type(cause).__name__}: {cause}")
        cause = cause.__cause__ or getattr(cause, "__context__", None)
        depth += 1

    root = " ".join(parts).lower()
    if "10061" in root or "connection refused" in root or "actively refused" in root:
        parts.append(
            "\nHint: nothing is listening on the configured base URL.\n"
            f"  provider={config.provider} base_url={config.base_url}\n"
            "  Fix the URL in .env or start your local proxy/gateway."
        )
    if "authentication" in root or ("invalid" in root and "api" in root):
        key_name = (
            "OPENAI_API_KEY" if config.provider == "openai" else "ANTHROPIC_API_KEY"
        )
        parts.append(
            f"\nHint: {key_name} was rejected by {config.provider}. "
            "Update the key in .env."
        )
    if "model" in root and (
        "not found" in root or "does not exist" in root or "invalid" in root
    ):
        parts.append(
            f"\nHint: model {config.model_id!r} is not available on "
            f"{config.provider}. Set TEACHER_MODEL_ID to a model your key can access "
            "(e.g. gpt-4o-mini, gpt-4o, o4-mini, claude-3-5-sonnet-latest)."
        )
    return "\n".join(parts)


def read_job_description(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Job description not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Job description is empty: {path}")
    return text


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a resume PDF with PyMuPDF (fitz)."""
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    # Normalize excessive blank lines for cleaner prompts.
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            cleaned.append(line)
        else:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def list_resume_pdfs(resume_dir: Path) -> list[Path]:
    if not resume_dir.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    pdfs = sorted(resume_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF resumes found in {resume_dir}")
    return pdfs


def build_user_content(job_description: str, resume_text: str, resume_name: str) -> str:
    return (
        f"## Job Description\n\n{job_description}\n\n"
        f"## Candidate Resume ({resume_name})\n\n{resume_text}\n\n"
        "Evaluate this candidate against the job description and return the "
        "structured screening result."
    )


def screen_resume(
    client: Any,
    config: TeacherConfig,
    job_description: str,
    resume_text: str,
    resume_name: str,
) -> ResumeScreeningResult:
    """Call the teacher model with structured output (provider-specific API)."""
    user_content = build_user_content(job_description, resume_text, resume_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        if config.provider == "openai":
            # Works for OpenAI and OpenAI-compatible endpoints.
            result = client.chat.completions.create(
                model=config.model_id,
                temperature=0.0,
                max_tokens=1024,
                messages=messages,
                response_model=ResumeScreeningResult,
                max_retries=2,
            )
        else:
            result = client.messages.create(
                model=config.model_id,
                max_tokens=1024,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                response_model=ResumeScreeningResult,
                max_retries=2,
            )
    except Exception as exc:
        raise SystemExit(
            "Teacher-model request failed while screening a resume.\n"
            f"{format_api_error(exc, config)}"
        ) from exc

    return result


def to_chatml_record(
    job_description: str,
    resume_text: str,
    resume_name: str,
    result: ResumeScreeningResult,
) -> dict:
    """Build one ChatML-style training example."""
    assistant_json = result.model_dump_json()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_content(job_description, resume_text, resume_name),
            },
            {"role": "assistant", "content": assistant_json},
        ]
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    config = resolve_teacher_config()
    job_description = read_job_description(JD_PATH)
    resume_paths = list_resume_pdfs(RESUME_DIR)
    client = build_instructor_client(config)

    # Fresh run: replace previous JSONL so re-runs stay idempotent.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    print(f"JD: {JD_PATH}")
    print(f"Resumes: {len(resume_paths)} PDF(s) in {RESUME_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Teacher provider: {config.provider}")
    print(f"Teacher model (TEACHER_MODEL_ID): {config.model_id}")
    print(f"API base URL: {config.base_url}")
    if "127.0.0.1" in config.base_url or "localhost" in config.base_url:
        print(
            "WARNING: base URL points at localhost. Ensure that proxy/gateway "
            "is running, or set the cloud base URL in .env."
        )
    print()

    for index, pdf_path in enumerate(resume_paths, start=1):
        print(f"[{index}/{len(resume_paths)}] Screening {pdf_path.name} ...")
        resume_text = extract_pdf_text(pdf_path)
        if not resume_text:
            print(f"  WARNING: empty text extraction for {pdf_path.name}; skipping.")
            continue

        result = screen_resume(
            client, config, job_description, resume_text, pdf_path.name
        )
        record = to_chatml_record(job_description, resume_text, pdf_path.name, result)
        append_jsonl(OUTPUT_PATH, record)
        print(
            f"  score={result.score} | {result.recommendation} | "
            f"confidence={result.confidence_level} | "
            f"flag_for_human={result.flag_for_human}"
        )

    print(f"\nWrote training data to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
