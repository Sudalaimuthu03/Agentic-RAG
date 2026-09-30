# RuntimeFix2 Approved Architecture Implementation

## Runtime authority chain

```text
User message
  ↓
SemanticInterpretation
  ↓
KnowledgeBoundary / identity verification
  ↓
Conversation + reference binding
  ↓
ResolvedRequest
  ↓
ExecutionPlan
  ↓
Bounded agent + four tools
  ↓
Hard retrieval scope
  ↓
FAISS + BM25 + reranking + parent expansion
  ↓
EvidenceBundle
  ↓
Answer validation
  ↓
Response
```

## Semantic authority

The live `/api/chat` route no longer uses the legacy `rag.resolution.resolve_request` function. That function remains only as a compatibility adapter for the original RuntimeFix2 unit tests.

The live semantic path is:

- `rag.semantic.understanding`
- `rag.semantic.verification`
- `rag.semantic.context_binding`
- `rag.semantic.planning`

## Knowledge boundary

Document factual questions are closed-world. External knowledge, web search, unrelated documents, previous answers, and retrieval similarity are not factual authorities.

## Identity

Identity is established from the authorized corpus before retrieval. Exact document identity and content-backed entity/variant matches are used; similarity retrieval cannot convert an unknown identity into a known identity.

## Conversation state

Session state tracks active targets, focused target, focused topic, active documents, presented items, references, operations, authoritative resolutions, and a monotonically increasing state version.

## Agent

The agent is bounded by the authoritative request and execution plan. Exactly four tools are retained:

1. `search_documents`
2. `list_available_documents`
3. `summarize_document`
4. `ask_clarification`

## Retrieval scope

Authorized source scope is applied immediately after hybrid candidate generation and before reranking and parent expansion. This prevents later retrieval stages from reintroducing unauthorized documents.

## Evidence

Evidence is request-local and carries target identity for multi-target requests. Answer validation rejects unsupported factual output when evidence is absent and requires explicit conflict acknowledgment when evidence conflicts.

## UI activity

Tool activity is emitted from actual backend tool invocations and surfaced through the existing NDJSON stream. `Thinking...` remains a UI status only and never exposes private reasoning.

## Regression

The full 75-question forensic matrix is stored in `tests/fixtures/runtimefix2_75q.json`. `scripts/run_75_regression.py` executes the live application against that matrix and compares observable resolved-request fields.
