"""Local document text extraction with PyMuPDF (no cloud OCR)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    """Extract plain text from PDF bytes or UTF-8 text files."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".text"}:
        return data.decode("utf-8", errors="replace").strip()

    # Default: treat as PDF (JD/resume uploads).
    parts: list[str] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return _normalize("\n".join(parts))


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            cleaned.append(line)
        else:
            blank += 1
            if blank <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()
