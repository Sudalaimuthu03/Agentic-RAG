"""
rag.retrieval.hybrid
-----------------------
Combines the FAISS retriever and BM25 retriever into a single
EnsembleRetriever with configurable weights. Falls back to
FAISS-only if HYBRID_ENABLED=false or BM25 has no documents.

Uses LangChain's built-in reciprocal-rank-fusion (RRF) instead of a
custom score-blending scheme: RRF only depends on each retriever's
rank order, not on raw FAISS/BM25 scores which live on incomparable
scales and would need per-query re-normalization (fragile - a single
outlier score skews the whole batch). Less introspectable than
logging raw per-candidate scores, but more stable ranking behaviour.
"""

from typing import Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from rag.utils.logger import get_logger, log_event

logger = get_logger("retrieval.hybrid")


def build_hybrid_retriever(
    vectorstore: FAISS,
    bm25_retriever: Optional[BM25Retriever],
    top_k: int,
    hybrid_enabled: bool,
    faiss_weight: float,
    bm25_weight: float,
):
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    if not hybrid_enabled or bm25_retriever is None:
        log_event(logger, "INFO", "retrieval.hybrid", status="faiss_only",
                  reason="hybrid_disabled_or_no_bm25")
        return faiss_retriever

    total = faiss_weight + bm25_weight
    if total == 0:
        faiss_weight, bm25_weight = 0.5, 0.5
        total = 1.0
    weights = [faiss_weight / total, bm25_weight / total]

    log_event(logger, "INFO", "retrieval.hybrid", status="ensemble_built",
              faiss_weight=round(weights[0], 2), bm25_weight=round(weights[1], 2))
    return EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=weights)
