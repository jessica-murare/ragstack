# RAGStack

A production-grade Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Ollama. Goes beyond demos — includes hybrid retrieval, re-ranking, citation enforcement, and an automated evaluation pipeline.

[![CI: RAG Quality Gate](https://github.com/jessica-murare/ragstack/actions/workflows/eval.yml/badge.svg)](https://github.com/jessica-murare/ragstack/actions)

---

## Live Demo

[ragstack.streamlit.app](https://ragstack.streamlit.app) — Upload a PDF or paste a URL, ask questions, get grounded answers with citations.

---

## What makes this production-grade

Most RAG tutorials stop at "it retrieves stuff and generates an answer." This system adds the layers that make it actually reliable:

- **Hybrid retrieval** — combines BM25 keyword search with vector semantic search so exact terms and conceptual queries both work well
- **Cross-encoder re-ranking** — a second-pass model scores each (query, chunk) pair together for higher precision than embedding similarity alone
- **Citation enforcement** — the system explicitly refuses to answer if retrieved chunks don't support a response, preventing hallucinations
- **Versioned config** — all prompts and model settings live in `config/settings.yaml`, behaviour changes without touching code
- **Evaluation pipeline** — a golden dataset of verified Q&A pairs with automated faithfulness scoring; CI fails the build if quality drops below threshold

---

## Architecture

```
User query
    │
    ▼
Hybrid Retrieval (BM25 + Vector)
    │  fetches top-20 candidates
    ▼
Cross-Encoder Re-ranker
    │  re-scores, keeps top-5
    ▼
Generator (Ollama / Groq)
    │  grounded answer + citations
    ▼
Response  ←  refuses if unsupported
```

---

## Tech Stack

| Component | Local (dev) | Cloud (prod) |
|---|---|---|
| LLM | Ollama (llama3.2) | Groq (llama-3.2) |
| Embeddings | Ollama (nomic-embed-text) | HuggingFace (MiniLM) |
| Vector store | ChromaDB (local) | ChromaDB (local) |
| Re-ranker | sentence-transformers | sentence-transformers |
| Keyword search | BM25 (rank-bm25) | BM25 (rank-bm25) |
| Frontend | Streamlit | Streamlit Cloud |
| Evaluation | Ragas-inspired custom | Ragas-inspired custom |
| CI | GitHub Actions | GitHub Actions |

---

## Project Structure

```
ragstack/
├── config/
│   └── settings.yaml        # versioned prompts + all model settings
├── data/
│   ├── raw/                 # drop PDFs and markdown files here
│   └── chroma/              # persistent vector store (auto-created)
├── src/
│   ├── ingestion.py         # PDF, Markdown, and web page loading
│   ├── chunking.py          # token-aware splitting with overlap
│   ├── vector_store.py      # ChromaDB wrapper + embedding switch
│   ├── retriever.py         # hybrid BM25 + vector + cross-encoder reranker
│   ├── generator.py         # grounded generation + hallucination refusal
│   ├── pipeline.py          # single entry point composing all components
│   └── config.py            # settings.yaml loader
├── eval/
│   ├── golden_dataset.json  # manually verified Q&A pairs
│   ├── evaluate.py          # faithfulness scoring + CI exit codes
│   └── eval_report.json     # generated on each eval run
├── .github/
│   └── workflows/
│       └── eval.yml         # CI quality gate — fails build below threshold
├── app.py                   # Streamlit frontend
└── requirements.txt
```

---

## Quickstart (local)

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Install

```bash
git clone https://github.com/jessica-murare/ragstack.git
cd ragstack
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Run the UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a PDF or paste a URL in the sidebar, click **Index Documents**, then ask questions in the chat.

### Run from Python

```python
from src.pipeline import RAGPipeline

rag = RAGPipeline()
rag.index(urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"])

rag.ask("How is RAG evaluated?")
# A: RAG systems are evaluated using benchmarks like BEIR and Natural Questions...
#    Sources: ['https://en.wikipedia.org/wiki/...']

rag.ask("What is the capital of France?")
# A: I cannot answer this based on the available documents.
```

---

## Configuration

All behaviour is controlled from `config/settings.yaml` — no code changes needed:

```yaml
llm:
  local_model: llama3.2
  temperature: 0.0          # 0 = deterministic, minimises hallucination risk

retrieval:
  top_k: 5                  # final chunks passed to generator
  candidates: 20            # hybrid fetch before re-ranking
  bm25_weight: 0.4
  vector_weight: 0.6

reranker:
  enabled: true             # set false to skip re-ranking

prompts:
  rag_prompt: |             # edit this to change answer behaviour
    You are a precise assistant...
```

---

## Evaluation

Run the quality evaluation against the golden dataset:

```bash
python eval/evaluate.py
```

```
=== Evaluation (5 questions) ===

  [PASS] q001: How is RAG evaluated?
  [PASS] q002: What are the limitations of RAG?
  [PASS] q003: When was the term RAG first introduced?
  [PASS] q004: What is the capital of France?
  [PASS] q005: How does RAG reduce hallucinations?

  Score     : 5/5 (100%)
  Threshold : 70%
  Result    : PASSED
  CI: BUILD PASSED
```

The script exits with code `1` if score drops below threshold — this is what GitHub Actions uses to fail the build automatically.

### Adding evaluation questions

Edit `eval/golden_dataset.json`:

```json
{
  "id": "q006",
  "question": "Your question here",
  "expected_answer": "The answer you verified manually",
  "source": "https://source-url.com"
}
```

---

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → set main file as `app.py`
3. Add secrets in the Streamlit Cloud dashboard:

```toml
GROQ_API_KEY = "your_groq_key"
HUGGINGFACEHUB_API_TOKEN = "your_hf_token"
RAG_MODE = "cloud"
```

The app automatically switches from Ollama to Groq + HuggingFace when `RAG_MODE=cloud` is set.

---

## How each component works

**Ingestion** — `PyPDFLoader` for PDFs, `UnstructuredMarkdownLoader` for `.md` files, and a custom BeautifulSoup parser for web pages that strips navigation and extracts article body only.

**Chunking** — `RecursiveCharacterTextSplitter` with `tiktoken` as the length function so splits happen in token space, not character space. Tries to split on paragraph → sentence → word boundaries before splitting mid-word.

**Hybrid retrieval** — BM25 and vector similarity scores are independently normalised to [0,1] then combined with configurable weights. Chunks appearing in both result sets get a score boost.

**Re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair jointly, unlike bi-encoder embeddings which score query and chunks independently. More accurate but slower, so it runs only on the top-20 hybrid candidates.

**Citation enforcement** — the prompt explicitly instructs the model to respond with a fixed refusal string if the context doesn't support the answer. Temperature is set to 0 to make this deterministic.

---

## Roadmap

- [ ] Add support for `.docx` and `.txt` file ingestion
- [ ] Streaming responses in the Streamlit UI
- [ ] Multi-document comparison queries
- [ ] Ragas integration for LLM-based faithfulness scoring
- [ ] Weaviate as an alternative vector store for larger corpora

---

## License

MIT