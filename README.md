# AI Resume Screener — local full-stack ATS

Three tiers. Sequential inference only (RTX 3060 6GB).

| Tier | Where | URL |
|------|--------|-----|
| Ollama (GGUF) | **Windows native** | `http://localhost:11434/v1` |
| FastAPI | Windows | `http://localhost:8080` |
| Next.js dashboard | Windows | `http://localhost:3000` |

## 1. Ollama (Windows)

Serve the fine-tuned GGUF as `resume-screener`. The name must match `.env` `LLM_MODEL_ID`.

```powershell
# Create the model once (if you have a Modelfile next to the GGUF)
cd D:\AI-Projects\AI-Resume-Screener\exported_models_gguf
ollama create resume-screener -f Modelfile

# Start / pull the model into memory
ollama run resume-screener
```

Leave Ollama running. OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

Windows `.env`:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_ID=resume-screener
LLM_API_KEY=ollama
```

## 2. FastAPI backend (Windows)

```powershell
cd D:\AI-Projects\AI-Resume-Screener
uv add fastapi uvicorn python-multipart openai python-dotenv pydantic pymupdf
uv run uvicorn src.backend.main:app --host 127.0.0.1 --port 8080
```

Health: [http://localhost:8080/api/health](http://localhost:8080/api/health)

Resumes are screened **one at a time**. Recommendation tiers are computed in Python from `score` (the LLM does not emit a recommendation).

## 3. Next.js frontend

```powershell
cd D:\AI-Projects\AI-Resume-Screener\frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

1. Paste or upload a JD  
2. Select resume PDFs (list + remove)  
3. **Run AI Screening**  
4. Watch “Scanning candidate X of Y…”  
5. Leaderboard by score — click a row for green/red flags and raw JSON
