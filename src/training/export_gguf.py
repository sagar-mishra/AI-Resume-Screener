"""Merge LoRA adapters and export a 4-bit GGUF for local serving.

Loads fine-tuned weights from adapters/ via Unsloth FastLanguageModel, then
writes GGUF under exported_models/ using quantization_method="q4_k_m"
(suitable for 6GB VRAM local inference).

Env (optional):
  BASE_MODEL_ID   — hint only (base is usually in adapters/adapter_config.json)
  MAX_SEQ_LENGTH  — load context (default 512; keep low on 6GB for export)
  ADAPTERS_DIR    — override adapters path (default: project adapters/)
  EXPORT_DIR      — override GGUF output dir (default: project exported_models/)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from unsloth import FastLanguageModel

DEFAULT_BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# Keep short during export load so 6GB can fit base+adapter without CPU offload.
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "exported_models"
QUANTIZATION_METHOD = "q4_k_m"
MIN_FREE_GB_FOR_LOAD = 4.5


def get_adapters_dir() -> Path:
    raw = os.getenv("ADAPTERS_DIR", "").strip()
    path = Path(raw) if raw else DEFAULT_ADAPTERS_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_export_dir() -> Path:
    raw = os.getenv("EXPORT_DIR", "").strip()
    path = Path(raw) if raw else DEFAULT_EXPORT_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_max_seq_length() -> int:
    raw = os.getenv("MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"MAX_SEQ_LENGTH must be an integer, got {raw!r}") from exc
    if value < 128:
        raise SystemExit("MAX_SEQ_LENGTH must be >= 128")
    return value


def require_adapters(adapters_dir: Path) -> None:
    adapter_config = adapters_dir / "adapter_config.json"
    weights = adapters_dir / "adapter_model.safetensors"
    bin_weights = adapters_dir / "adapter_model.bin"
    if not adapters_dir.is_dir():
        raise SystemExit(
            f"Adapters directory not found: {adapters_dir}\n"
            "Train first: python src/training/train_qlora.py"
        )
    if not adapter_config.is_file():
        raise SystemExit(
            f"Missing adapter_config.json in {adapters_dir}\n"
            "Train first: python src/training/train_qlora.py"
        )
    if not weights.is_file() and not bin_weights.is_file():
        raise SystemExit(
            f"Missing adapter weights in {adapters_dir}\n"
            "Expected adapter_model.safetensors (or .bin)."
        )


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def assert_cuda_ready_for_load() -> float:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this Python process.\n"
            "Use your CUDA-enabled finetuning_env and a fresh terminal."
        )

    free_cuda()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    print(
        f"CUDA device 0: {torch.cuda.get_device_name(0)} | "
        f"free={free_gb:.2f}GB / total={total_gb:.2f}GB"
    )
    if free_gb < MIN_FREE_GB_FOR_LOAD:
        raise SystemExit(
            f"Only {free_gb:.2f}GB free VRAM (need ~{MIN_FREE_GB_FOR_LOAD}+GB).\n"
            "bitsandbytes will try to put layers on CPU and then fail.\n\n"
            "Fix:\n"
            "  1. Close Chrome / other Python jobs\n"
            "  2. nvidia-smi   # Memory-Usage should be ~0 MiB\n"
            "  3. Get-Process python* | Stop-Process -Force\n"
            "  4. Re-run: python src/training/export_gguf.py\n"
        )
    return free_gb


def load_lora_model(adapters_dir: Path, max_seq_length: int):
    """Load adapters fully on GPU 0 (no CPU/disk 4-bit shard)."""
    import torch

    assert_cuda_ready_for_load()
    model_name = str(adapters_dir)

    load_kwargs: dict = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "dtype": None,
        "load_in_4bit": True,
        # Force entire model onto GPU — avoids:
        # "Some modules are dispatched on the CPU or the disk"
        "device_map": {"": 0},
    }

    print(f"Loading LoRA model from: {model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("device_map", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "dispatched on the CPU" in msg or "disk" in msg.lower():
            raise SystemExit(
                "4-bit load spilled layers to CPU/disk (not enough free VRAM).\n"
                f"Details: {msg}\n\n"
                "In a *new* shell:\n"
                "  nvidia-smi\n"
                "  Get-Process python* | Stop-Process -Force\n"
                "  $env:MAX_SEQ_LENGTH='256'\n"
                "  python src/training/export_gguf.py\n"
            ) from exc
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA OOM while loading adapters for export.\n"
            "Free the GPU and retry with MAX_SEQ_LENGTH=256.\n"
            f"Details: {exc}"
        ) from exc

    free_cuda()
    return model, tokenizer


def main() -> None:
    adapters_dir = get_adapters_dir()
    export_dir = get_export_dir()
    max_seq_length = get_max_seq_length()
    require_adapters(adapters_dir)

    base_hint = os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID).strip()

    print("=== AI Resume Screener — GGUF export (Unsloth) ===")
    print(f"Adapters:     {adapters_dir}")
    print(f"Export dir:   {export_dir}")
    print(f"Quantization: {QUANTIZATION_METHOD}")
    print(f"max_seq_length={max_seq_length}")
    print(f"BASE_MODEL_ID hint (env): {base_hint}")
    print()

    model, tokenizer = load_lora_model(adapters_dir, max_seq_length)

    # Merge LoRA into base + write GGUF (q4_k_m for 6GB local serving).
    export_dir.mkdir(parents=True, exist_ok=True)
    free_cuda()
    print(f"Exporting GGUF -> {export_dir} (quantization_method={QUANTIZATION_METHOD})")
    try:
        model.save_pretrained_gguf(
            str(export_dir),
            tokenizer,
            quantization_method=QUANTIZATION_METHOD,
        )
    except Exception as exc:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            raise SystemExit(
                "CUDA OOM during GGUF export/merge.\n"
                "Close other GPU apps and re-run in a fresh process.\n"
                f"Details: {exc}"
            ) from exc
        raise

    print("\nDone.")
    print(f"  GGUF output: {export_dir}")
    print("  Serve with llama.cpp / Ollama / Unsloth FastInference as needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None

