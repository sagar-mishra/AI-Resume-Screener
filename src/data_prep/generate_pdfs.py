import fitz  # PyMuPDF

# Dictionary containing the test resume texts
resumes = {
    "Alex_Chen_Strong_Shortlist.pdf": """Alex Chen
Title: Senior AI Software Engineer
Email: alex.chen@email.com

Professional Summary:
Full-stack AI engineer specializing in local LLM deployment and custom fine-tuning. Passionate about building privacy-first GenAI applications that operate completely on edge hardware without relying on OpenAI.

Experience:
Lead AI Engineer | Vertex Solutions (2023 - Present)
- Architected a local enterprise RAG system using Next.js for the frontend and a FastAPI backend to orchestrate document ingestion.
- Fine-tuned LLaMA 3 8B using Unsloth and QLoRA to generate strict JSON schemas from unstructured medical PDFs.
- Deployed quantized GGUF models on constrained local GPU hardware (RTX 3090/3060) utilizing Ollama and vLLM to strictly manage VRAM utilization and KV caching.
- Built asynchronous streaming text UIs using React and Tailwind CSS.

Backend Engineer | DataFlow Tech (2020 - 2023)
- Developed high-throughput REST APIs using Python and FastAPI.
- Implemented sequential processing queues to handle heavy PDF parsing pipelines using PyMuPDF.

Skills:
Python, TypeScript, React, Next.js, FastAPI, LLaMA 3, Unsloth, QLoRA, vLLM, Ollama, llama.cpp, Tailwind CSS, PyTorch.
""",

    "Sarah_Jenkins_Consider.pdf": """Sarah Jenkins
Title: Full-Stack Developer
Email: s.jenkins@email.com

Professional Summary:
Product-focused software engineer with 4 years of experience building modern web applications. Recently integrated AI capabilities into enterprise platforms to automate customer service workflows.

Experience:
Software Engineer | CloudScale Inc. (2022 - Present)
- Built and maintained the company’s core customer dashboard using Next.js, TypeScript, and Tailwind CSS.
- Integrated the OpenAI API (GPT-4) into a Node.js/Express backend to summarize user support tickets.
- Designed a PostgreSQL database schema to store and retrieve AI-generated insights.
- Deployed containerized applications using Docker and AWS ECS.

Junior Web Developer | StartUp Studio (2020 - 2022)
- Created responsive landing pages using React and vanilla CSS.
- Assisted in migrating a legacy monolithic PHP app to a modern API-driven architecture.

Skills:
JavaScript, TypeScript, React, Next.js, Node.js, Express, PostgreSQL, AWS, Docker, OpenAI API, Prompt Engineering.
""",

    "David_Miller_Reject.pdf": """David Miller
Title: Data Analyst & Machine Learning Researcher
Email: david.m.data@email.com

Professional Summary:
Detail-oriented data scientist with a strong mathematical background. Experienced in exploring large datasets, building predictive ML models, and creating business intelligence dashboards for executive stakeholders.

Experience:
Data Scientist | FinTech Analytics (2021 - Present)
- Cleaned and processed massive financial datasets using Python, Pandas, and NumPy.
- Built random forest and XGBoost classification models using scikit-learn to predict customer churn with 88% accuracy.
- Designed interactive data visualization dashboards using Tableau and Matplotlib.
- Presented monthly statistical findings to the VP of Marketing.

Data Analyst Intern | RetailCorp (2020 - 2021)
- Wrote complex SQL queries to extract sales data from Snowflake.
- Automated weekly reporting spreadsheets using Python scripts.

Skills:
Python, SQL, Pandas, NumPy, scikit-learn, Tableau, Matplotlib, XGBoost, Statistical Modeling, Data Cleaning, Jupyter Notebooks.
"""
}

# Generate a PDF for each resume
for filename, text_content in resumes.items():
    doc = fitz.open()  
    page = doc.new_page()  
    
    # Define a rectangle area for the text with margins (left, top, right, bottom)
    rect = fitz.Rect(50, 50, 550, 800) 
    
    # Insert the text into the rectangle
    page.insert_textbox(rect, text_content, fontsize=11, fontname="helv") 
    
    doc.save(filename)
    doc.close()
    print(f"Successfully generated: {filename}")