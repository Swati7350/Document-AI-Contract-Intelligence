# 📄 Document AI & Contract Intelligence

A production-style MVP showcasing end-to-end document AI: OCR, layout parsing, structured extraction, contract RAG, and evaluation — all in one Streamlit app.

## Features
- **OCR & Layout** — extract text, tables, stamps from PDFs/images
- **Structured Extraction** — LLM-powered JSON extraction of contract fields
- **Contract RAG** — chat over uploaded contracts with source chunks
- **Prompt Studio** — editable prompts, re-run extraction
- **Evaluation Dashboard** — precision, recall, F1 with charts

## Run locally
```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

## Deploy on Render
Push to GitHub → connect repo → Render detects `render.yaml` automatically.

## Stack
Streamlit · Plotly · Pandas · Pillow · OpenAI-compatible LLM interface
