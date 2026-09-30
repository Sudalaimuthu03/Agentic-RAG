# V4 Architecture

## Authority model

1. **Semantic authority** — immutable `ResolvedRequest`.
2. **Execution authority** — `ExecutionGate` plus the bounded tool-calling agent.
3. **Evidence authority** — retrieval plus `EvidenceBundle` assessment.
4. **Answer authority** — answer validation against the resolved request and evidence.

## Semantic resolution

Resolution is deterministic-first. Current-turn explicit identity outranks references, context, corpus inference, and model assistance. Ambiguous identities are not guessed.

## Agent

The LLM selects tools through LangChain tool binding. The application does not use a giant query router. Tool execution is bounded by hard per-turn budgets.

## Evidence

Evidence keeps source/page/chunk/parent boundaries. The evidence gate requires alignment with the resolved request and distinguishes known, unknown, and conflicting states.

## Offline integrity

Embeddings and reranking require local model paths. The configured Ollama model must be installed. Silent fallback and runtime Hugging Face downloads are disabled.
