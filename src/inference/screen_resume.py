"""Screen resumes with the fine-tuned LoRA adapter (Unsloth 4-bit).

Loads adapters/, reads data/job_descriptions/sample_jd.txt and all PDFs under
data/raw_resumes/, generates AGENTS.md 6-key JSON for each candidate, prints a
summary table, and optionally writes data/processed/inference_results.jsonl.

Env:
  ADAPTERS_DIR    — default: adapters/
  MAX_SEQ_LENGTH  — default: 512 (keep low on 6GB VRAM)
  MAX_NEW_TOKENS  — default: 384
  JD_PATH         — default: data/job_descriptions/sample_jd.txt
  RESUME_DIR      — default: data/raw_resumes/
  OUTPUT_JSONL    — default: data/processed/inference_results.jsonl
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fitz  # PyMuPDF
from unsloth import FastLanguageModel

DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DEFAULT_JD_PATH = PROJECT_ROOT / "data" / "job_descriptions" / "sample_jd.txt"
DEFAULT_RESUME_DIR = PROJECT_ROOT / "data" / "raw_resumes"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "inference_results.jsonl"
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 256
MIN_FREE_GB_FOR_LOAD = 4.0
# Leave room for generation inside the model context window.
GENERATION_RESERVE_TOKENS = 280

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
You are an expert technical recruiter and resume screener.
Score how well the candidate matches the job description.

Output ONLY one JSON object (no markdown, no extra text) with exactly these keys:
{
  "score": <integer 0-100>,
  "confidence_level": "High" | "Medium" | "Low",
  "flag_for_human": <true|false>,
  "strengths": "<short string>",
  "weaknesses": "<short string>",
  "recommendation": "<one of the 9 tiers below>"
}

recommendation must match score:
95-100 Strong Shortlist | 85-94 Shortlist | 75-84 Interview Recommended |
65-74 Consider | 55-64 Borderline | 45-54 Hold | 35-44 Upskill & Reapply |
20-34 Alternative Role Recommended | 0-19 Reject
"""

# Force the model to continue inside a JSON object (reduces free-form babble).
JSON_ASSISTANT_PREFIX = (
    '{\n  "score": '
)


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_max_seq_length() -> int:
    raw = os.getenv("MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"MAX_SEQ_LENGTH must be an integer, got {raw!r}") from exc
    return max(128, value)


def get_max_new_tokens() -> int:
    raw = os.getenv("MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"MAX_NEW_TOKENS must be an integer, got {raw!r}") from exc
    return max(64, value)


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def assert_cuda_ready() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Activate finetuning_env with CUDA PyTorch."
        )
    free_cuda()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} | "
        f"free={free_gb:.2f}GB / total={total_gb:.2f}GB"
    )
    if free_gb < MIN_FREE_GB_FOR_LOAD:
        print(
            f"WARNING: only {free_gb:.2f}GB free. Close other GPU apps if load fails."
        )


def recommendation_for_score(score: int) -> Recommendation:
    score = max(0, min(100, int(score)))
    for low, high, label in RECOMMENDATION_TIERS:
        if low <= score <= high:
            return label
    return "Reject"


def extract_pdf_text(pdf_path: Path) -> str:
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    text = "\n".join(parts)
    lines = [ln.rstrip() for ln in text.splitlines()]
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


def build_user_content(job_description: str, resume_text: str, resume_name: str) -> str:
    return (
        f"## Job Description\n\n{job_description}\n\n"
        f"## Candidate Resume ({resume_name})\n\n{resume_text}\n\n"
        "Return ONLY the JSON screening object now."
    )


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_user_content(
    tokenizer,
    job_description: str,
    resume_text: str,
    resume_name: str,
    max_prompt_tokens: int,
) -> tuple[str, str, str]:
    """Shrink JD/resume until the full chat prompt fits in max_prompt_tokens.

    Avoids HF right-truncation of the assistant generation header, which made
    the model continue mid-JD text instead of emitting JSON.
    """
    jd_budget = min(len(job_description), 2800)
    resume_budget = min(len(resume_text), 3500)

    for _ in range(24):
        user = build_user_content(
            job_description[:jd_budget],
            resume_text[:resume_budget],
            resume_name,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Prefill is appended after the template.
        full = prompt + JSON_ASSISTANT_PREFIX
        if _token_len(tokenizer, full) <= max_prompt_tokens:
            return user, prompt, full

        # Shrink resume first, then JD.
        if resume_budget > 600:
            resume_budget = int(resume_budget * 0.75)
        elif jd_budget > 400:
            jd_budget = int(jd_budget * 0.75)
        else:
            # Last resort: hard cut the built prompt tokens from the *user* side
            # by returning a minimal user message.
            user = (
                f"## Job Description\n{job_description[:400]}\n\n"
                f"## Resume ({resume_name})\n{resume_text[:500]}\n\n"
                "Return ONLY the JSON screening object now."
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return user, prompt, prompt + JSON_ASSISTANT_PREFIX

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(
                job_description[:400], resume_text[:500], resume_name
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return messages[1]["content"], prompt, prompt + JSON_ASSISTANT_PREFIX


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse model output into a dict; tolerate prefill, fences, trailing junk."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # If we prefilled with `{\n  "score": `, the decode may be only the tail.
    if not cleaned.lstrip().startswith("{"):
        cleaned = JSON_ASSISTANT_PREFIX + cleaned

    # Trim anything after the first complete top-level object.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt to close a truncated JSON object conservatively.
    match = re.search(r"\{[\s\S]*", cleaned)
    if match:
        candidate = match.group(0)
        # Balance braces if model stopped mid-object.
        if candidate.count("{") > candidate.count("}"):
            candidate = candidate + ("}" * (candidate.count("{") - candidate.count("}")))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Extract key fields with regex as last resort.
            score_m = re.search(r'"score"\s*:\s*(\d+)', candidate)
            if score_m:
                score = int(score_m.group(1))
                conf_m = re.search(
                    r'"confidence_level"\s*:\s*"(High|Medium|Low)"', candidate
                )
                flag_m = re.search(
                    r'"flag_for_human"\s*:\s*(true|false)', candidate, re.I
                )
                str_m = re.search(r'"strengths"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
                weak_m = re.search(r'"weaknesses"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
                return {
                    "score": score,
                    "confidence_level": conf_m.group(1) if conf_m else "Medium",
                    "flag_for_human": (
                        flag_m.group(1).lower() == "true" if flag_m else score < 65
                    ),
                    "strengths": str_m.group(1) if str_m else "See model output",
                    "weaknesses": weak_m.group(1) if weak_m else "See model output",
                    "recommendation": recommendation_for_score(score),
                }

    raise ValueError(f"No JSON object found in model output:\n{text[:500]}")


def normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce to AGENTS.md schema; align recommendation with score."""
    score = int(raw.get("score", 0))
    score = max(0, min(100, score))

    conf = str(raw.get("confidence_level", "Medium")).strip()
    if conf not in {"High", "Medium", "Low"}:
        conf = "Medium"

    flag = raw.get("flag_for_human", False)
    if isinstance(flag, str):
        flag = flag.strip().lower() in {"1", "true", "yes", "y"}
    else:
        flag = bool(flag)

    strengths = str(raw.get("strengths", "")).strip() or "Not specified"
    weaknesses = str(raw.get("weaknesses", "")).strip() or "Not specified"
    recommendation = recommendation_for_score(score)

    return {
        "score": score,
        "confidence_level": conf,
        "flag_for_human": flag,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation,
    }


def load_model(adapters_dir: Path, max_seq_length: int):
    import torch

    assert_cuda_ready()
    if not (adapters_dir / "adapter_config.json").is_file():
        raise SystemExit(
            f"No adapters found at {adapters_dir}\n"
            "Train first: python src/training/train_qlora.py"
        )

    load_kwargs: dict = {
        "model_name": str(adapters_dir),
        "max_seq_length": max_seq_length,
        "dtype": None,
        "load_in_4bit": True,
        "device_map": {"": 0},
    }
    print(f"Loading adapters: {adapters_dir}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("device_map", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except ValueError as exc:
        if "dispatched on the CPU" in str(exc) or "disk" in str(exc).lower():
            raise SystemExit(
                "Not enough free VRAM to load the 4-bit model.\n"
                "Close other GPU apps, then:\n"
                "  $env:MAX_SEQ_LENGTH='256'\n"
                "  python src/inference/screen_resume.py\n"
                f"Details: {exc}"
            ) from exc
        raise

    FastLanguageModel.for_inference(model)
    # Avoid HF warning / weird behavior when both max_length and max_new_tokens set.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = None
    free_cuda()
    return model, tokenizer


def screen_one(
    model,
    tokenizer,
    job_description: str,
    resume_path: Path,
    max_seq_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    resume_text = extract_pdf_text(resume_path)
    if not resume_text:
        return {
            "resume": resume_path.name,
            "error": "empty PDF text extraction",
            "score": None,
            "recommendation": None,
        }

    # Budget: prompt must fit with room for new tokens (no right-truncation).
    max_prompt_tokens = max(256, max_seq_length - min(max_new_tokens, GENERATION_RESERVE_TOKENS))
    _user, _prompt, full_prompt = fit_user_content(
        tokenizer,
        job_description,
        resume_text,
        resume_path.name,
        max_prompt_tokens=max_prompt_tokens,
    )

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=False,  # we already fitted the prompt
    )
    input_len = int(inputs["input_ids"].shape[-1])
    if input_len >= max_seq_length:
        # Absolute safety: keep the end (chat header + JSON prefill).
        old_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        tokenizer.truncation_side = old_side
        input_len = int(inputs["input_ids"].shape[-1])

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens, then re-attach JSON prefill for parsing.
    new_tokens = output_ids[0][input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    generated = JSON_ASSISTANT_PREFIX + completion

    try:
        parsed = extract_json_object(generated)
        result = normalize_result(parsed)
    except Exception as exc:
        return {
            "resume": resume_path.name,
            "error": f"parse failed: {exc}",
            "raw_output": generated[:800],
            "score": None,
            "recommendation": None,
        }

    result["resume"] = resume_path.name
    result["raw_output"] = generated
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print(f"{'Resume':<40} {'Score':>5}  {'Recommendation':<28} {'Conf':<6} Flag")
    print("-" * 88)
    for r in results:
        if r.get("score") is None:
            print(f"{r['resume']:<40} {'ERR':>5}  {str(r.get('error', ''))[:40]}")
            continue
        print(
            f"{r['resume']:<40} {r['score']:>5}  "
            f"{r['recommendation']:<28} {r['confidence_level']:<6} "
            f"{r['flag_for_human']}"
        )
    print("=" * 88)


def write_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in results:
            # Drop bulky raw_output from default file unless useful — keep it.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote results: {path}")


def main() -> None:
    adapters_dir = _path_from_env("ADAPTERS_DIR", DEFAULT_ADAPTERS_DIR)
    jd_path = _path_from_env("JD_PATH", DEFAULT_JD_PATH)
    resume_dir = _path_from_env("RESUME_DIR", DEFAULT_RESUME_DIR)
    output_jsonl = _path_from_env("OUTPUT_JSONL", DEFAULT_OUTPUT_JSONL)
    max_seq_length = get_max_seq_length()
    max_new_tokens = get_max_new_tokens()

    if not jd_path.is_file():
        raise SystemExit(f"Job description not found: {jd_path}")
    resumes = sorted(resume_dir.glob("*.pdf"))
    if not resumes:
        raise SystemExit(f"No PDF resumes in {resume_dir}")

    job_description = jd_path.read_text(encoding="utf-8").strip()

    print("=== AI Resume Screener — inference ===")
    print(f"Adapters:       {adapters_dir}")
    print(f"JD:             {jd_path}")
    print(f"Resumes:        {len(resumes)} PDF(s) in {resume_dir}")
    print(f"MAX_SEQ_LENGTH: {max_seq_length}")
    print(f"MAX_NEW_TOKENS: {max_new_tokens}")
    print()

    model, tokenizer = load_model(adapters_dir, max_seq_length)
    results: list[dict[str, Any]] = []

    for index, resume_path in enumerate(resumes, start=1):
        print(f"[{index}/{len(resumes)}] Screening {resume_path.name} ...")
        free_cuda()
        row = screen_one(
            model,
            tokenizer,
            job_description,
            resume_path,
            max_seq_length,
            max_new_tokens,
        )
        results.append(row)
        if row.get("score") is not None:
            print(
                f"  -> score={row['score']} | {row['recommendation']} | "
                f"confidence={row['confidence_level']}"
            )
        else:
            print(f"  -> ERROR: {row.get('error')}")

    print_summary(results)
    write_jsonl(output_jsonl, results)

    # Sanity: stronger resume should generally beat weakest sample.
    scored = [r for r in results if r.get("score") is not None]
    if len(scored) >= 2:
        best = max(scored, key=lambda r: r["score"])
        worst = min(scored, key=lambda r: r["score"])
        print(
            f"\nSanity: highest={best['resume']} ({best['score']}), "
            f"lowest={worst['resume']} ({worst['score']})"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
