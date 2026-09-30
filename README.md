# RAG Agent V5.4


## Architecture

`ResolvedRequest → Execution Gate → Agent → Constrained Tools → Evidence → Evidence Sufficiency Gate → Answer Generation → Answer Validation → Final Response`

The semantic authority is an immutable `ResolvedRequest`. Retrieval and answer generation cannot silently replace explicit entity, variant, document, or reference identity.

## Key guarantees

- Local Ollama LLM; no cloud model fallback.
- Local Sentence-Transformers embeddings.
- Local CrossEncoder reranker; runtime downloads disabled by default.
- FAISS + BM25 hybrid retrieval.
- Request-local evidence with `KNOWN`, `UNKNOWN`, and `CONFLICTING` states.
- Dynamic entity/variant/reference resolution; no hardcoded product or document taxonomy.
- Explicit ambiguity/clarification handling.
- Bounded agent execution: 3 iterations, 3 LLM calls, 2 retrieval operations, 1 guardrail retry by default.
- Document prompt-injection content is treated as evidence, never as instructions.
- Flask API and existing PDF upload/chat UI are retained.

## Local prerequisites

Python 3.11, Ollama, the configured Ollama model, the local embedding model under `models/all-MiniLM-L6-v2`, and the local reranker under `models/ms-marco-MiniLM-L-6-v2`.

Runtime downloads are disabled. Prepare models before startup.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set `MODEL_NAME` to a model already installed in Ollama.

Start:

```powershell
python app.py
```

## Validation

Deterministic release gate:

```powershell
python scripts\v4_release_gate.py
pytest -q
```

The final release process must also extract the ZIP into a new directory and rerun the same checks there.

## Security / grounding behavior

If the corpus does not establish the requested fact, the system does not fill the gap with pretrained knowledge. If authoritative evidence conflicts, the response must acknowledge the conflict rather than silently choose a source.

## LLM backend switch

The same Agent/RAG pipeline supports either local Ollama or the NVIDIA OpenAI-compatible endpoint.

```env
USE_NVIDIA=false
MODEL_NAME=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434

NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_API_KEY=your_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

- `USE_NVIDIA=false`: local Ollama (`MODEL_NAME`).
- `USE_NVIDIA=true`: NVIDIA endpoint (`NVIDIA_MODEL`).
- No silent fallback occurs between providers.
- Embeddings, FAISS, BM25, reranking, tools, evidence validation, and conversation state remain application-side and unchanged by the provider switch.
- NVIDIA mode requires internet access and a valid API key.
