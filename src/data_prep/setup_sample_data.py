"""Generate sample job description and synthetic PDF resumes for pipeline testing.

Creates:
  - data/job_descriptions/sample_jd.txt
  - data/raw_resumes/*.pdf (5 candidates with varying fit/experience)
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# Project root: src/data_prep/ -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JD_DIR = PROJECT_ROOT / "data" / "job_descriptions"
RESUME_DIR = PROJECT_ROOT / "data" / "raw_resumes"

SAMPLE_JD = """\
Senior Python / AI Engineer
Company: Acme AI
Location: Remote (US / EU) or Hybrid — San Francisco, CA
Employment Type: Full-time
Level: Senior (IC4 / L5 equivalent)
Salary Range: $180,000 – $240,000 USD + equity

────────────────────────────────────────────────────────────
About the Role
────────────────────────────────────────────────────────────
Acme AI is building production-grade LLM applications for enterprise
knowledge work. We are hiring a Senior Python/AI Engineer to design, train,
and ship models and services that power our resume intelligence and document
understanding products.

You will own end-to-end ML features: data pipelines, fine-tuning (QLoRA /
PEFT), evaluation harnesses, and low-latency inference services. You will
partner with product and backend teams to turn research prototypes into
reliable APIs used by recruiting teams at scale.

────────────────────────────────────────────────────────────
Responsibilities
────────────────────────────────────────────────────────────
• Design and implement Python-based ML pipelines for document parsing,
  feature extraction, and model training on multi-GPU / consumer-GPU setups.
• Fine-tune and evaluate open-weight LLMs (Llama 3.x, Mistral, Qwen) using
  QLoRA, Unsloth, Hugging Face Transformers, PEFT, and TRL.
• Build robust PDF/text extraction and preprocessing (PyMuPDF, chunking,
  PII handling) for resume and job-description corpora.
• Serve models efficiently (vLLM, llama.cpp / GGUF, FastAPI) with attention
  to VRAM limits (target: 6–24 GB class GPUs) and p95 latency budgets.
• Define offline and online evaluation metrics (relevance scoring, ranking
  quality, calibration, human-in-the-loop flags).
• Write clean, tested, production Python (type hints, ruff, pytest) and
  maintain clear experiment tracking (W&B or MLflow).
• Collaborate on data labeling guidelines, synthetic data generation, and
  continuous improvement of scoring schemas (JSON-structured outputs).
• Mentor mid-level engineers and review PRs for correctness and efficiency.

────────────────────────────────────────────────────────────
Required Qualifications
────────────────────────────────────────────────────────────
• 5+ years professional software engineering experience, with at least
  3 years focused on Python ML / applied AI systems.
• Strong Python proficiency (async, packaging, typing, performance profiling).
• Hands-on experience fine-tuning or adapting LLMs (LoRA/QLoRA, SFT, DPO)
  and evaluating structured generation (JSON schemas, tool-use).
• Practical knowledge of the modern ML stack: PyTorch, Hugging Face
  Transformers / Datasets, PEFT, bitsandbytes, accelerate.
• Experience building and deploying inference APIs (FastAPI, Docker, basic
  cloud: AWS/GCP/Azure).
• Comfortable working with document data: PDFs, OCR edge cases, messy text.
• Solid software engineering fundamentals: Git, CI, code review, testing.
• Bachelor's degree in CS, ML, or equivalent practical experience.

────────────────────────────────────────────────────────────
Preferred Qualifications
────────────────────────────────────────────────────────────
• Experience with Unsloth, Axolotl, or similar efficient fine-tuning stacks.
• GGUF / llama.cpp / Ollama export and local serving on consumer GPUs.
• Ranking / retrieval (embeddings, vector DBs, hybrid search) for candidate
  matching or RAG systems.
• Prior work on HR-tech, recruiting, or resume/job matching systems.
• Contributions to open-source ML tooling or published applied ML work.
• Familiarity with Ruff, uv, and modern Python project layout.

────────────────────────────────────────────────────────────
Tech Stack (current)
────────────────────────────────────────────────────────────
Python 3.12 · PyTorch · Unsloth · Hugging Face · PEFT · bitsandbytes ·
PyMuPDF · FastAPI · Docker · uv · ruff · pytest · W&B

────────────────────────────────────────────────────────────
What Success Looks Like (first 6 months)
────────────────────────────────────────────────────────────
1. Ship a reproducible fine-tuning pipeline for structured resume scoring
   (score, confidence, strengths/weaknesses, recommendation tiers).
2. Deliver a local inference path that runs on a single 6–12 GB GPU.
3. Improve agreement with human recruiter labels by a measurable margin
   on a held-out evaluation set.
4. Document data prep, training, and serving so a teammate can re-run
   experiments end-to-end in under a day.

────────────────────────────────────────────────────────────
How to Apply
────────────────────────────────────────────────────────────
Submit a resume (PDF) highlighting relevant Python/AI projects, model
training experience, and links to GitHub or technical writing. Include
any open-source or portfolio work involving LLMs or document AI.
"""

# Five synthetic candidates spanning strong senior match → weak / junior fit.
CANDIDATES: list[dict] = [
    {
        "filename": "01_priya_nair_senior_ai.pdf",
        "name": "Priya Nair",
        "title": "Senior Machine Learning Engineer",
        "email": "priya.nair@email.example",
        "phone": "+1 (415) 555-0142",
        "location": "San Francisco, CA (Remote-friendly)",
        "summary": (
            "Senior ML engineer with 8 years of experience shipping Python-based "
            "LLM and NLP systems. Deep expertise in QLoRA fine-tuning, Hugging Face "
            "Transformers, and production FastAPI inference services on consumer "
            "and cloud GPUs. Passionate about structured generation, evaluation "
            "harnesses, and document understanding pipelines."
        ),
        "experience": [
            {
                "role": "Senior Machine Learning Engineer",
                "company": "Helix Document AI",
                "dates": "2021 – Present",
                "bullets": [
                    "Led QLoRA fine-tuning of Llama 3.1 8B for structured JSON "
                    "extraction from resumes and contracts; cut GPU cost ~4× vs full FT.",
                    "Built PDF parsing + chunking stack with PyMuPDF and custom "
                    "PII redaction; processed 2M+ pages/month.",
                    "Designed FastAPI + vLLM inference service with p95 latency "
                    "<400ms on A10G; exported GGUF builds for laptop demos.",
                    "Owned evaluation suite (precision/recall, calibration, human "
                    "agreement) and experiment tracking in Weights & Biases.",
                ],
            },
            {
                "role": "Machine Learning Engineer",
                "company": "Northstar Analytics",
                "dates": "2018 – 2021",
                "bullets": [
                    "Fine-tuned BERT/RoBERTa models for multi-label document "
                    "classification; improved F1 by 12 points over baseline.",
                    "Implemented training pipelines with PyTorch, PEFT, and "
                    "accelerate on multi-GPU nodes.",
                    "Deployed model endpoints on AWS (SageMaker + ECS) with "
                    "Docker and CI via GitHub Actions.",
                ],
            },
            {
                "role": "Software Engineer (Python)",
                "company": "CloudLedger Inc.",
                "dates": "2016 – 2018",
                "bullets": [
                    "Built backend microservices in Python (Flask/FastAPI precursors) "
                    "for financial document ingestion.",
                    "Wrote unit/integration tests (pytest) and enforced linting standards.",
                ],
            },
        ],
        "skills": (
            "Python, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, "
            "Unsloth, QLoRA, TRL, vLLM, llama.cpp/GGUF, FastAPI, PyMuPDF, Docker, "
            "AWS, W&B, pytest, ruff, Git"
        ),
        "education": [
            "M.S. Computer Science — Stanford University, 2016",
            "B.S. Computer Science — UC San Diego, 2014",
        ],
        "projects": [
            "Open-source resume-scorer demo: Llama-3.1-8B QLoRA + GGUF export "
            "(GitHub: example/priya-resume-scorer).",
            "Blog series on structured JSON generation and schema-constrained decoding.",
        ],
    },
    {
        "filename": "02_marcus_chen_mid_senior.pdf",
        "name": "Marcus Chen",
        "title": "Applied AI Engineer",
        "email": "marcus.chen@email.example",
        "phone": "+1 (206) 555-0198",
        "location": "Seattle, WA",
        "summary": (
            "Applied AI engineer with 5 years of Python experience building NLP "
            "and retrieval systems. Comfortable with Transformers fine-tuning, "
            "RAG pipelines, and FastAPI services. Seeking a senior role focused "
            "on LLM adaptation and production ML."
        ),
        "experience": [
            {
                "role": "Applied AI Engineer",
                "company": "BrightQuery",
                "dates": "2022 – Present",
                "bullets": [
                    "Fine-tuned open LLMs (Mistral-7B, Llama-3-8B) with LoRA for "
                    "customer-support answer ranking; +18% NDCG@10.",
                    "Built hybrid retrieval (embeddings + BM25) over product docs "
                    "using sentence-transformers and Qdrant.",
                    "Shipped FastAPI microservices for scoring and reranking; "
                    "containerized with Docker on GCP Cloud Run.",
                ],
            },
            {
                "role": "Data Scientist / ML Engineer",
                "company": "RetailPulse",
                "dates": "2020 – 2022",
                "bullets": [
                    "Trained classification and NER models in PyTorch for review "
                    "and ticket analytics.",
                    "Automated data prep pipelines and basic experiment tracking "
                    "with MLflow.",
                    "Collaborated with backend team to expose models via REST APIs.",
                ],
            },
            {
                "role": "Python Developer",
                "company": "StartupForge",
                "dates": "2019 – 2020",
                "bullets": [
                    "Developed ETL jobs and internal tools in Python for a B2B SaaS product.",
                    "Introduced type hints and pytest coverage for core modules.",
                ],
            },
        ],
        "skills": (
            "Python, PyTorch, Hugging Face, LoRA/PEFT, sentence-transformers, "
            "FastAPI, Docker, GCP, Qdrant, MLflow, SQL, pytest, Git"
        ),
        "education": [
            "B.S. Computer Science — University of Washington, 2019",
        ],
        "projects": [
            "RAG chatbot for internal HR policies using Llama-3 and FAISS.",
            "Personal fine-tuning notebooks for DPO preference tuning (side project).",
        ],
    },
    {
        "filename": "03_elena_vasquez_midlevel.pdf",
        "name": "Elena Vasquez",
        "title": "Software Engineer — Backend & ML Platform",
        "email": "elena.vasquez@email.example",
        "phone": "+1 (512) 555-0167",
        "location": "Austin, TX",
        "summary": (
            "Backend-focused software engineer with 4 years of Python experience "
            "and growing ML platform exposure. Strong FastAPI/Docker skills; "
            "limited hands-on LLM fine-tuning but solid data pipeline and API design "
            "background. Eager to deepen applied AI work."
        ),
        "experience": [
            {
                "role": "Software Engineer II",
                "company": "DataHarbor Systems",
                "dates": "2022 – Present",
                "bullets": [
                    "Owned FastAPI services for document upload, metadata storage, "
                    "and async job queues (Redis + Celery).",
                    "Integrated third-party OCR and embedding APIs; added retries, "
                    "circuit breakers, and observability (Prometheus).",
                    "Partnered with ML team to productionize batch scoring jobs "
                    "on Kubernetes; wrote glue code, not model training.",
                ],
            },
            {
                "role": "Junior Software Engineer",
                "company": "DataHarbor Systems",
                "dates": "2021 – 2022",
                "bullets": [
                    "Maintained Python ETL workers for CSV/JSON ingestion.",
                    "Added unit tests and refactored legacy scripts to typed modules.",
                ],
            },
            {
                "role": "Software Engineering Intern",
                "company": "CityMesh IoT",
                "dates": "Summer 2020",
                "bullets": [
                    "Built internal dashboards and REST helpers in Python/Flask.",
                ],
            },
        ],
        "skills": (
            "Python, FastAPI, Flask, Docker, Kubernetes, Redis, PostgreSQL, "
            "pytest, Git, basic PyTorch (coursework), OpenAI API, REST design"
        ),
        "education": [
            "B.S. Software Engineering — University of Texas at Austin, 2021",
            "Coursera: Deep Learning Specialization (Andrew Ng), 2023",
        ],
        "projects": [
            "Capstone: simple document classifier with scikit-learn + FastAPI wrapper.",
            "Hobby: chat wrapper around OpenAI API with prompt templates (no fine-tuning).",
        ],
    },
    {
        "filename": "04_jake_owens_junior.pdf",
        "name": "Jake Owens",
        "title": "Junior Python Developer",
        "email": "jake.owens@email.example",
        "phone": "+1 (303) 555-0133",
        "location": "Denver, CO",
        "summary": (
            "Recent graduate and junior Python developer with 1.5 years of "
            "professional experience. Strong fundamentals in web APIs and data "
            "scripts; coursework in ML but no production LLM or fine-tuning "
            "experience. Looking to break into applied AI roles."
        ),
        "experience": [
            {
                "role": "Junior Python Developer",
                "company": "MountainApps LLC",
                "dates": "2024 – Present",
                "bullets": [
                    "Maintain internal Flask/FastAPI tools for customer support workflows.",
                    "Write Python scripts to clean CSVs and generate weekly reports.",
                    "Participate in code review and learn pytest and typing practices.",
                ],
            },
            {
                "role": "Software Engineering Intern",
                "company": "PeakSoft Solutions",
                "dates": "Summer 2023",
                "bullets": [
                    "Assisted senior engineers with bug fixes in a Django monolith.",
                    "Wrote documentation for onboarding scripts.",
                ],
            },
        ],
        "skills": (
            "Python, Flask, FastAPI (beginner), Django, SQL, pandas, Git, "
            "scikit-learn (coursework), HTML/CSS"
        ),
        "education": [
            "B.S. Computer Science — Colorado State University, 2024",
            "Relevant coursework: Machine Learning, NLP Intro, Data Structures",
        ],
        "projects": [
            "University project: sentiment analysis on movie reviews with scikit-learn.",
            "Personal site and Todo API tutorial clones; no public LLM fine-tuning work.",
        ],
    },
    {
        "filename": "05_sofia_brandt_weak_match.pdf",
        "name": "Sofia Brandt",
        "title": "Senior Java Backend Engineer",
        "email": "sofia.brandt@email.example",
        "phone": "+49 30 555 0188",
        "location": "Berlin, Germany",
        "summary": (
            "Senior backend engineer with 9 years primarily in Java/Kotlin "
            "microservices and enterprise integrations. Limited Python exposure "
            "and no hands-on machine learning or LLM experience. Strong systems "
            "design and leadership skills in traditional backend domains."
        ),
        "experience": [
            {
                "role": "Senior Backend Engineer",
                "company": "EuroPay Technologies",
                "dates": "2019 – Present",
                "bullets": [
                    "Architected payment orchestration microservices in Java 17 / "
                    "Spring Boot handling 50k TPS peak.",
                    "Led team of 6 engineers; defined SLIs/SLOs and incident response.",
                    "Migrated legacy monolith modules to Kubernetes on AWS EKS.",
                ],
            },
            {
                "role": "Backend Engineer",
                "company": "LogiChain GmbH",
                "dates": "2015 – 2019",
                "bullets": [
                    "Built REST and event-driven services (Kafka) for supply-chain tracking.",
                    "Implemented integration tests with Testcontainers and CI on Jenkins.",
                ],
            },
            {
                "role": "Junior Java Developer",
                "company": "NordSoft Consulting",
                "dates": "2013 – 2015",
                "bullets": [
                    "Developed CRUD modules and SOAP connectors for enterprise clients.",
                ],
            },
        ],
        "skills": (
            "Java, Kotlin, Spring Boot, Kafka, Kubernetes, AWS, PostgreSQL, "
            "gRPC, Jenkins, limited Python scripting, no ML frameworks"
        ),
        "education": [
            "Dipl.-Inform. (M.S. equivalent) — TU Berlin, 2013",
        ],
        "projects": [
            "Internal JVM performance workshops; open-source contributions to "
            "Java testing libraries — no AI/ML portfolio.",
        ],
    },
]


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "name": ParagraphStyle(
            "ResumeName",
            parent=base["Heading1"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "headline": ParagraphStyle(
            "ResumeHeadline",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=2,
            textColor="#333333",
        ),
        "contact": ParagraphStyle(
            "ResumeContact",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor="#444444",
        ),
        "section": ParagraphStyle(
            "ResumeSection",
            parent=base["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            borderPadding=2,
        ),
        "body": ParagraphStyle(
            "ResumeBody",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "job_header": ParagraphStyle(
            "JobHeader",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            spaceBefore=4,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "ResumeBullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12.5,
            leftIndent=8,
            alignment=TA_LEFT,
        ),
        "meta": ParagraphStyle(
            "ResumeMeta",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            spaceAfter=3,
        ),
    }
    return styles


def write_job_description(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_JD, encoding="utf-8")
    print(f"Wrote job description: {path}")


def build_resume_pdf(candidate: dict, output_path: Path) -> None:
    """Render a single multi-section resume PDF with reportlab."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Resume — {candidate['name']}",
        author=candidate["name"],
    )

    story: list = [
        Paragraph(candidate["name"], styles["name"]),
        Paragraph(candidate["title"], styles["headline"]),
        Paragraph(
            f"{candidate['email']}  ·  {candidate['phone']}  ·  {candidate['location']}",
            styles["contact"],
        ),
        Paragraph("Professional Summary", styles["section"]),
        Paragraph(candidate["summary"], styles["body"]),
        Paragraph("Experience", styles["section"]),
    ]

    for job in candidate["experience"]:
        header = (
            f"<b>{job['role']}</b> — {job['company']} "
            f"<font color='#555555'>({job['dates']})</font>"
        )
        story.append(Paragraph(header, styles["job_header"]))
        items = [
            ListItem(Paragraph(b, styles["bullet"]), leftIndent=12, value="•")
            for b in job["bullets"]
        ]
        story.append(
            ListFlowable(
                items,
                bulletType="bullet",
                start="•",
                leftIndent=15,
                spaceBefore=0,
                spaceAfter=4,
            )
        )

    story.append(Paragraph("Skills", styles["section"]))
    story.append(Paragraph(candidate["skills"], styles["body"]))

    story.append(Paragraph("Education", styles["section"]))
    for edu in candidate["education"]:
        story.append(Paragraph(f"• {edu}", styles["meta"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Projects &amp; Portfolio", styles["section"]))
    for proj in candidate["projects"]:
        story.append(Paragraph(f"• {proj}", styles["meta"]))

    doc.build(story)
    print(f"Wrote resume: {output_path}")


def main() -> None:
    write_job_description(JD_DIR / "sample_jd.txt")
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        build_resume_pdf(candidate, RESUME_DIR / candidate["filename"])
    print(
        f"\nDone. Generated 1 JD and {len(CANDIDATES)} resumes under:\n"
        f"  {JD_DIR}\n  {RESUME_DIR}"
    )


if __name__ == "__main__":
    main()
