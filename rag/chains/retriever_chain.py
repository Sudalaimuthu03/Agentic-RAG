"""
rag.chains.retriever_chain
-----------------------------
`RagRetriever` performs the retrieval pipeline as a single custom
BaseRetriever:

    User query
      -> Hybrid retriever (FAISS + BM25)  (fetch TOP_K child chunks)
      -> Reranker                          (keep RERANK_TOP_K)
      -> Parent Document Retrieval         (swap children for their parents)

Query rewriting from chat history now happens inside the tool-calling
agent itself (rag.agent.graph) - the LLM sees the full message history
and writes a clear, standalone search_documents query itself, rather
than a separate history-aware retriever wrapper.
"""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.retrieval.filter import apply_filter
from rag.utils.logger import get_logger, log_event

logger = get_logger("chains.retriever")


class RagRetriever(BaseRetriever):
    """Custom retriever: hybrid search -> rerank -> parent doc swap -> filter."""

    hybrid_retriever: Any
    reranker: Optional[Any] = None
    parent_store: Dict[str, Document] = {}
    rerank_top_k: int = 5
    active_filter_ref: Optional[Any] = None  # callable returning current filter dict
    scope_sources: tuple[str, ...] = ()

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        candidates = self.hybrid_retriever.invoke(query)
        # Hard semantic scope is enforced immediately after candidate generation, before reranking/parent expansion.
        if self.scope_sources:
            allowed = set(self.scope_sources)
            candidates = [d for d in candidates if d.metadata.get("source") in allowed]
        log_event(logger, "INFO", "retrieval.hybrid", status="success", candidates=len(candidates), scope=list(self.scope_sources))

        if self.reranker is not None and candidates:
            try:
                candidates = self.reranker.rerank(query, candidates, self.rerank_top_k)
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "ERROR", "retrieval.rerank", status="error", error=str(exc))
                candidates = candidates[: self.rerank_top_k]
        else:
            candidates = candidates[: self.rerank_top_k]

        # Parent document retrieval: swap each child chunk for its
        # larger parent context, de-duplicating parents.
        seen_parents = set()
        parent_docs: List[Document] = []
        for child in candidates:
            parent_id = child.metadata.get("parent_id")
            parent = self.parent_store.get(parent_id) if parent_id else None
            if parent is not None:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                # Carry the rerank score onto the parent for source display
                merged_meta = dict(parent.metadata)
                if "rerank_score" in child.metadata:
                    merged_meta["rerank_score"] = child.metadata["rerank_score"]
                merged_meta["child_preview"] = child.page_content[:250]
                parent_docs.append(Document(page_content=parent.page_content, metadata=merged_meta))
            else:
                parent_docs.append(child)

        log_event(
            logger, "INFO", "retrieval.parent_swap", status="success",
            parents_used=len(parent_docs),
            parent_ids=[d.metadata.get("parent_id") for d in parent_docs],
        )

        active_filter = self.active_filter_ref() if self.active_filter_ref else {}
        parent_docs = apply_filter(parent_docs, active_filter)

        log_event(
            logger, "INFO", "llm.context_sent", status="success",
            n_chunks=len(parent_docs), total_chars=sum(len(d.page_content) for d in parent_docs),
            chunk_ids=[d.metadata.get("parent_id") for d in parent_docs],
        )
        log_event(
            logger, "DEBUG", "llm.context_sent", status="success",
            chunks=[{"parent_id": d.metadata.get("parent_id"), "text": d.page_content} for d in parent_docs],
        )

        return parent_docs
