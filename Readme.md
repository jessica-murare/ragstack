# Ragstack

This project implements a robust, enterprise-ready Retrieval-Augmented Generation (RAG) System designed to bridge the gap between simple AI prototypes and scalable, reliable production applications. Unlike basic RAG pipelines, this architecture prioritizes high-precision retrieval through Hybrid Search (BM25 + Vector), eliminates hallucinations using Strict Citation Enforcement, and ensures long-term reliability through an Automated Evaluation Pipeline (Ragas). Built with a focus on modularity and "Faithfulness" metrics, this system provides a blueprint for deploying AI that enterprise users can actually trust.

## Current Status

Implemented:
- Document ingestion from PDF files
- Document ingestion from Markdown files
- Web page loading from URLs
- Token-aware document chunking with metadata enrichment

In progress:
- Vector store integration in `src/vector_store.py`

## Project Structure

```text
ragstack/
|-- config/
|-- data/
|   `-- raw/              # local source documents, ignored by git
|-- src/
|   |-- ingestion.py      # loads PDFs, Markdown, and web pages
|   |-- chunking.py       # splits documents into token-based chunks
|   `-- vector_store.py   # reserved for vector DB integration
|-- .env
|-- .gitignore
|-- requirements.txt
`-- Readme.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add environment variables in `.env`.

Current example:

```env
USER_AGENT=groundtruth-rag/1.0
```

4. Install Ollama.
5. Pull the local models used for a typical RAG stack:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Ingestion

`src/ingestion.py` loads content from:
- `data/raw/*.pdf`
- `data/raw/*.md`
- URLs passed in at runtime

Run it directly:

```bash
python src/ingestion.py
```

What it does:
- creates `data/raw/` if it does not exist
- loads supported local files recursively
- fetches and parses web pages
- returns LangChain `Document` objects with source metadata

## Chunking

`src/chunking.py` takes loaded documents and splits them into overlapping,
token-based chunks using `RecursiveCharacterTextSplitter` plus `tiktoken`.

Run it directly:

```bash
python src/chunking.py
```

What it does:
- splits by token count instead of raw characters
- prefers natural boundaries like paragraphs and sentences
- filters low-quality chunks
- adds chunk metadata such as `chunk_index` and `token_count`

## Notes

- `data/raw/` is ignored by git, so local source documents stay out of the repo.
- `.env` is ignored by git, so keep secrets and machine-specific config there.
- `src/vector_store.py` is currently a placeholder and still needs implementation.
