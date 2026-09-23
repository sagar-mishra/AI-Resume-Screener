# Context: AI Resume Screener Project

## Infrastructure & Constraints
- **Hardware Limitations:** Running locally on an RTX 3060 (6GB VRAM), 16GB RAM, i7.
- **Memory Management:** Standard fine-tuning will fail. All LLM training code MUST use QLoRA (4-bit quantization) via the `unsloth` library.
- **Base Model:** Strictly use `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` to avoid Hugging Face gated repo authentication errors and benefit from better JSON formatting.
- **Data Parsing:** Use `PyMuPDF` (imported as `fitz`) for local PDF parsing. Do not use heavy cloud OCR APIs.

## Environment & Code Standards
- **Package Manager:** You MUST use `uv` for all Python dependency management. Output installation commands using `uv pip install <package>` or `uv add <package>` and assume the local virtual environment (`.venv`) is active.
- **Linting:** Use `ruff` for code formatting. Ensure all generated Python code is clean and PEP 8 compliant.
- **Inference Strategy:** Training scripts must include a step to export the final fine-tuned model to 4-bit GGUF format or utilize Unsloth's FastInference so it can be served locally on 6GB VRAM.
- **Secrets Management:** Use `python-dotenv` (`load_dotenv()`) to read environment variables from `.env`. NEVER hardcode API keys or secrets in Python scripts. Always check that `.env` is listed in `.gitignore`.
- **Config Management:** NEVER hardcode model IDs or endpoints inside Python scripts. Always load model names dynamically from environment variables using `python-dotenv` (e.g., `TEACHER_MODEL_ID` for bootstrapping and `BASE_MODEL_ID` for fine-tuning).

## Required Project Structure
All created files MUST respect this folder structure:
- `data/raw_resumes/`: Place PDF resumes here.
- `data/job_descriptions/`: Place Job Description `.txt` files here.
- `data/processed/`: Store output `.jsonl` training datasets here.
- `src/data_prep/`: Scripts for sample data generation and bootstrapping.
- `src/training/`: Scripts for Unsloth QLoRA fine-tuning.
- `src/inference/`: Scripts for model evaluation and serving.
- `adapters/`: Output folder for saved fine-tuned LoRA weights.

## Output Schema Target
All model outputs and training data must conform to strict JSON with the following keys:
1. `score` (integer 0-100)
2. `confidence_level` (string: "High", "Medium", "Low")
3. `flag_for_human` (boolean)
4. `strengths` (string)
5. `weaknesses` (string)
6. `recommendation` (string mapping to the exact 9-tier HR scale below based on the score):
   - 95-100: Strong Shortlist
   - 85-94: Shortlist
   - 75-84: Interview Recommended
   - 65-74: Consider
   - 55-64: Borderline
   - 45-54: Hold
   - 35-44: Upskill & Reapply
   - 20-34: Alternative Role Recommended
   - 0-19: Reject