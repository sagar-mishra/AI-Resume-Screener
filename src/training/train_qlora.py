"""QLoRA fine-tune Llama-3.1 for resume screening with Unsloth + PEFT.

Optimized for RTX 3060 Laptop (6GB VRAM):
  - 4-bit base weights
  - Unsloth gradient checkpointing
  - short context (default 256; override via MAX_SEQ_LENGTH)
  - forced CE-loss chunking (UNSLOTH_CE_LOSS_TARGET_GB)
  - batch size 1 + gradient accumulation

Reads ChatML JSONL from data/processed/resume_training_data.jsonl and writes
LoRA adapter weights + tokenizer to adapters/.

Env (never hardcode model IDs at call sites):
  BASE_MODEL_ID              — default unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
  MAX_SEQ_LENGTH             — default 256 (try 128 if still OOM)
  LORA_R                     — default 8 (try 4 if still OOM)
  LORA_TARGETS               — optional comma list, e.g. q_proj,v_proj
  UNSLOTH_CE_LOSS_TARGET_GB  — default 0.05 (forces CE chunking on full GPU)
  UNSLOTH_CE_LOSS_N_CHUNKS   — default 128
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env first (before torch/unsloth) — project .env wins over shell noise.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)

# Reduce CUDA allocator fragmentation on Windows / small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# CRITICAL for 6GB: Unsloth fused CE auto-detects free VRAM. After the 8B 4-bit
# model loads, free VRAM is ~0 and it raises:
#   "No or negligible GPU memory available for fused cross entropy."
# Force a tiny per-chunk budget BEFORE importing unsloth (unsloth_zoo reads
# these env vars at import time).
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "0.05")
os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "128")

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

DEFAULT_BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "resume_training_data.jsonl"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 6GB laptop defaults — short context so activations fit next to 4-bit weights.
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_LORA_R = 8
# Rough free-VRAM floor to load Llama-3.1-8B 4-bit without CPU offload.
MIN_FREE_GB_FOR_LOAD = 4.5
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Soft char caps so JD+resume chat templates fit in short context.
MAX_SYSTEM_CHARS = 1_200
MAX_USER_CHARS = 2_500


def get_base_model_id() -> str:
    model_id = os.getenv("BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID)
    if model_id is None or not str(model_id).strip():
        raise SystemExit(
            "BASE_MODEL_ID is missing or empty.\n"
            "Set BASE_MODEL_ID in the project-root .env file "
            f"(e.g. BASE_MODEL_ID={DEFAULT_BASE_MODEL_ID})."
        )
    return str(model_id).strip()


def get_max_seq_length() -> int:
    raw = os.getenv("MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(
            f"MAX_SEQ_LENGTH must be an integer, got {raw!r}"
        ) from exc
    if value < 128:
        raise SystemExit("MAX_SEQ_LENGTH must be >= 128")
    return value


def get_lora_r() -> int:
    raw = os.getenv("LORA_R", str(DEFAULT_LORA_R))
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise SystemExit(f"LORA_R must be an integer, got {raw!r}") from exc
    if value < 4:
        raise SystemExit("LORA_R must be >= 4")
    return value


def get_target_modules() -> list[str]:
    raw = os.getenv("LORA_TARGETS", "").strip()
    if not raw:
        return list(TARGET_MODULES)
    modules = [m.strip() for m in raw.split(",") if m.strip()]
    if not modules:
        raise SystemExit("LORA_TARGETS is empty after parsing")
    return modules


def require_dataset(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(
            f"Training dataset not found: {path}\n"
            "Run the teacher pipeline first:\n"
            "  uv run python src/data_prep/generate_dataset.py"
        )
    if path.stat().st_size == 0:
        raise SystemExit(f"Training dataset is empty: {path}")
    return path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated for VRAM]"


def load_and_format_dataset(tokenizer, dataset_path: Path, max_seq_length: int):
    """Load JSONL and map messages -> Llama-3.1 chat template text."""
    dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    def formatting_prompts_func(examples: dict) -> dict:
        texts: list[str] = []
        for conversation in examples["messages"]:
            trimmed: list[dict[str, str]] = []
            for message in conversation:
                role = message["role"]
                content = message["content"]
                # Keep assistant JSON intact; shrink long system/user payloads.
                if role == "system":
                    content = _truncate(content, MAX_SYSTEM_CHARS)
                elif role == "user":
                    content = _truncate(content, MAX_USER_CHARS)
                trimmed.append({"role": role, "content": content})

            text = tokenizer.apply_chat_template(
                trimmed,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Hard cap tokens so SFT never builds 2k+ sequences on 6GB.
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )["input_ids"]
            texts.append(tokenizer.decode(token_ids, skip_special_tokens=False))
        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def assert_cuda_ready_for_load() -> float:
    """Ensure CUDA is usable and enough free VRAM exists for 8B 4-bit."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this Python process.\n"
            "Install a CUDA build of PyTorch and re-open a fresh terminal."
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
            f"Only {free_gb:.2f}GB free VRAM (need ~{MIN_FREE_GB_FOR_LOAD}+GB "
            "to load Llama-3.1-8B 4-bit without CPU offload).\n\n"
            "Fix:\n"
            "  1. Close Chrome / games / other Python training jobs\n"
            "  2. In a fresh shell:  nvidia-smi   (confirm Memory-Usage is near 0)\n"
            "  3. Restart this script (do not reuse a notebook kernel after OOM)\n"
        )
    return free_gb


def build_model_and_tokenizer(
    model_id: str,
    max_seq_length: int,
    lora_r: int,
    target_modules: list[str],
):
    """Load 4-bit Unsloth model fully on GPU and attach LoRA adapters."""
    import torch

    assert_cuda_ready_for_load()
    print(f"Loading base model: {model_id}")
    print(
        f"  max_seq_length={max_seq_length}, load_in_4bit=True, "
        f"lora_r={lora_r}, targets={target_modules}"
    )
    print(
        f"  CE chunking: UNSLOTH_CE_LOSS_TARGET_GB="
        f"{os.environ.get('UNSLOTH_CE_LOSS_TARGET_GB')} "
        f"UNSLOTH_CE_LOSS_N_CHUNKS={os.environ.get('UNSLOTH_CE_LOSS_N_CHUNKS')}"
    )

    # Force the full 4-bit model onto GPU 0.
    # Do NOT pass max_memory that under-budgets the GPU — HF then shards to
    # CPU/disk and bitsandbytes raises:
    #   "Some modules are dispatched on the CPU or the disk..."
    load_kwargs: dict = {
        "model_name": model_id,
        "max_seq_length": max_seq_length,
        "dtype": None,  # autodetect
        "load_in_4bit": True,
        "device_map": {"": 0},
    }

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except TypeError:
        # Older Unsloth may not accept device_map.
        load_kwargs.pop("device_map", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "dispatched on the CPU" in msg or "disk" in msg.lower():
            raise SystemExit(
                "4-bit load tried to place layers on CPU/disk (not enough free VRAM).\n"
                f"Details: {msg}\n\n"
                "Do this, then retry in a *new* process:\n"
                "  nvidia-smi                  # Memory-Usage should be ~0 MiB\n"
                "  # close browsers / other Python jobs using the GPU\n"
                "  $env:MAX_SEQ_LENGTH='256'\n"
                "  $env:LORA_R='8'\n"
                "  python src/training/train_qlora.py\n"
            ) from exc
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit(
            "CUDA OOM while loading the 4-bit model.\n"
            "Free the GPU (nvidia-smi), restart Python, and retry.\n"
            f"Details: {exc}"
        ) from exc

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    # Unsloth chat templates can leave placeholder EOS ("<EOS_TOKEN>") which is
    # NOT in the Llama vocab — TRL 0.24 then crashes on SFTTrainer init.
    tokenizer = fix_llama31_special_tokens(tokenizer)

    free_cuda()
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_r,  # keep alpha == r (Unsloth common default)
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # critical for 6GB VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


# TrainingArguments.to_dict() obfuscates *any* field ending in "_token":
#   eos_token -> "<EOS_TOKEN>", pad_token -> "<PAD_TOKEN>"
# Unsloth may rehydrate SFTConfig via to_dict(), which poisons real tokens.
# Never put real special-token strings into SFTConfig; fix the tokenizer only.
TOKEN_PLACEHOLDERS = {
    "<EOS_TOKEN>",
    "<PAD_TOKEN>",
    "<BOS_TOKEN>",
    "<UNK_TOKEN>",
    "<eos_token>",
    "<pad_token>",
}
LLAMA31_EOS = "<|eot_id|>"
LLAMA31_EOS_FALLBACK = "<|end_of_text|>"
LLAMA31_PAD = "<|finetune_right_pad_id|>"


def _token_id(tokenizer, token: str | None) -> int | None:
    """Resolve a token string to an id; treat placeholders / unk as missing."""
    if not token or token in TOKEN_PLACEHOLDERS:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id and token != getattr(
        tokenizer, "unk_token", None
    ):
        return None
    return int(token_id)


def fix_llama31_special_tokens(tokenizer):
    """Force real Llama-3.1 eos/pad tokens and shield against placeholders.

    Also wraps convert_tokens_to_ids so TRL/Unsloth cannot fail if a config
    still carries TrainingArguments.to_dict() placeholders like '<EOS_TOKEN>'.
    """
    # Prefer instruct end-of-turn; fall back to end_of_text.
    if _token_id(tokenizer, LLAMA31_EOS) is not None:
        tokenizer.eos_token = LLAMA31_EOS
    elif _token_id(tokenizer, LLAMA31_EOS_FALLBACK) is not None:
        tokenizer.eos_token = LLAMA31_EOS_FALLBACK
    elif _token_id(tokenizer, tokenizer.eos_token) is None:
        raise SystemExit(
            "Could not set a valid eos_token for this tokenizer. "
            f"Current eos_token={tokenizer.eos_token!r}"
        )

    if _token_id(tokenizer, LLAMA31_PAD) is not None:
        tokenizer.pad_token = LLAMA31_PAD
    elif _token_id(tokenizer, tokenizer.pad_token) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Hard-set ids (property assignment can be flaky after Unsloth patches).
    eos_id = _token_id(tokenizer, tokenizer.eos_token)
    pad_id = _token_id(tokenizer, tokenizer.pad_token)
    if eos_id is not None:
        tokenizer.eos_token_id = eos_id
    if pad_id is not None:
        tokenizer.pad_token_id = pad_id

    # Defensive: map TrainingArguments placeholders back to real ids.
    if not getattr(tokenizer, "_resume_screener_token_patch", False):
        original_convert = tokenizer.convert_tokens_to_ids

        def convert_tokens_to_ids_safe(token):  # noqa: ANN001
            if token in {"<EOS_TOKEN>", "<eos_token>"}:
                return tokenizer.eos_token_id
            if token in {"<PAD_TOKEN>", "<pad_token>"}:
                return tokenizer.pad_token_id
            if token in {"<BOS_TOKEN>", "<bos_token>"}:
                return tokenizer.bos_token_id
            return original_convert(token)

        tokenizer.convert_tokens_to_ids = convert_tokens_to_ids_safe  # type: ignore[method-assign]
        tokenizer._resume_screener_token_patch = True

    print(
        f"Tokenizer specials: eos={tokenizer.eos_token!r} "
        f"(id={tokenizer.eos_token_id}), pad={tokenizer.pad_token!r} "
        f"(id={tokenizer.pad_token_id})"
    )
    return tokenizer


def build_trainer(model, tokenizer, dataset, max_seq_length: int) -> SFTTrainer:
    """SFTTrainer tuned for tight VRAM (RTX 3060 6GB).

    Compatible with TRL >= 0.20 (processing_class + SFTConfig) and older TRL
    that still accepted tokenizer= / max_seq_length= on SFTTrainer.
    """
    tokenizer = fix_llama31_special_tokens(tokenizer)
    if getattr(model, "config", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    # CRITICAL: leave eos_token/pad_token as None on SFTConfig.
    # TrainingArguments.to_dict() turns real tokens into '<EOS_TOKEN>' /
    # '<PAD_TOKEN>'. Unsloth may rebuild the config from to_dict() and then
    # TRL raises: eos_token '<EOS_TOKEN>' not in vocabulary.
    sft_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch size = 4
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(OUTPUTS_DIR),
        report_to="none",
        save_strategy="steps",
        save_steps=60,
        save_total_limit=1,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        # Periodically free cached CUDA blocks between steps on small GPUs.
        torch_empty_cache_steps=1,
        eval_strategy="no",
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=max_seq_length,
        packing=False,
        eos_token=None,
        pad_token=None,
    )
    # Belt-and-suspenders if anything re-set placeholders on the config.
    sft_args.eos_token = None
    sft_args.pad_token = None

    try:
        # Modern TRL (0.20+ / 0.24): tokenizer -> processing_class
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_args,
        )
    except TypeError:
        # Fallback for older Unsloth/TRL stacks.
        from transformers import TrainingArguments

        legacy_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(OUTPUTS_DIR),
            report_to="none",
            save_strategy="steps",
            save_steps=60,
            save_total_limit=1,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            eval_strategy="no",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=legacy_args,
        )
    return trainer


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
            # Reset peak stats so prints are meaningful across phases.
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def print_gpu_stats(label: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"{label}: CUDA not available")
            return
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free = total - reserved
        print(
            f"{label}: {props.name} | "
            f"total={total:.2f}GB reserved={reserved:.2f}GB "
            f"allocated={allocated:.2f}GB free≈{free:.2f}GB"
        )
        if free < 0.35:
            print(
                "WARNING: <0.35GB free before train — relying on forced CE chunking.\n"
                "  If training still fails, retry with MAX_SEQ_LENGTH=128 LORA_R=4"
            )
    except Exception as exc:
        print(f"{label}: GPU stats unavailable ({exc})")


def save_adapters(model, tokenizer, adapters_dir: Path) -> None:
    adapters_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapters + tokenizer -> {adapters_dir}")
    free_cuda()
    # Cap peak memory during save on small GPUs.
    try:
        model.save_pretrained(str(adapters_dir), maximum_memory_usage=0.5)
    except TypeError:
        model.save_pretrained(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))


def maybe_export_gguf(model, tokenizer, adapters_dir: Path) -> None:
    flag = os.getenv("EXPORT_GGUF", "0").strip().lower()
    if flag not in {"1", "true", "yes", "y"}:
        print(
            "Skipping GGUF export (set EXPORT_GGUF=1 to enable). "
            "Adapters are ready for Unsloth FastLanguageModel inference."
        )
        return

    gguf_dir = adapters_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting 4-bit GGUF -> {gguf_dir}")
    free_cuda()
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF export complete: {gguf_dir}")


def main() -> None:
    model_id = get_base_model_id()
    max_seq_length = get_max_seq_length()
    lora_r = get_lora_r()
    target_modules = get_target_modules()
    dataset_path = require_dataset(DATASET_PATH)

    print("=== AI Resume Screener — QLoRA fine-tune (Unsloth) ===")
    print(f"BASE_MODEL_ID:  {model_id}")
    print(f"MAX_SEQ_LENGTH: {max_seq_length}")
    print(f"LORA_R:         {lora_r}")
    print(f"LORA_TARGETS:   {target_modules}")
    print(
        f"CE chunking:    TARGET_GB={os.environ.get('UNSLOTH_CE_LOSS_TARGET_GB')} "
        f"N_CHUNKS={os.environ.get('UNSLOTH_CE_LOSS_N_CHUNKS')}"
    )
    print(f"Dataset:        {dataset_path}")
    print(f"Adapters dir:   {ADAPTERS_DIR}")
    print(f"Outputs dir:    {OUTPUTS_DIR}")
    print()

    free_cuda()
    model, tokenizer = build_model_and_tokenizer(
        model_id, max_seq_length, lora_r, target_modules
    )
    free_cuda()
    print_gpu_stats("After model+LoRA load")

    dataset = load_and_format_dataset(tokenizer, dataset_path, max_seq_length)
    print(f"Training examples: {len(dataset)}")

    trainer = build_trainer(model, tokenizer, dataset, max_seq_length)
    free_cuda()
    print_gpu_stats("Before train")

    print("\nStarting training...")
    try:
        trainer_stats = trainer.train()
    except RuntimeError as exc:
        msg = str(exc)
        if "negligible GPU memory" in msg or "out of memory" in msg.lower():
            raise SystemExit(
                f"\n{msg}\n\n"
                "VRAM exhausted on this GPU. Try a *fresh* PowerShell process:\n"
                "  nvidia-smi   # free Memory-Usage first\n"
                "  $env:MAX_SEQ_LENGTH='128'\n"
                "  $env:LORA_R='4'\n"
                "  $env:LORA_TARGETS='q_proj,v_proj'\n"
                "  $env:UNSLOTH_CE_LOSS_TARGET_GB='0.03'\n"
                "  $env:UNSLOTH_CE_LOSS_N_CHUNKS='256'\n"
                "  python src/training/train_qlora.py\n"
            ) from exc
        raise

    print(f"Training finished: {trainer_stats.metrics}")

    free_cuda()
    save_adapters(model, tokenizer, ADAPTERS_DIR)
    maybe_export_gguf(model, tokenizer, ADAPTERS_DIR)

    print("\nDone.")
    print(f"  LoRA adapters: {ADAPTERS_DIR}")
    print(f"  Checkpoints:   {OUTPUTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
