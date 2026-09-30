"""
rag.retrieval.bm25
--------------------
Builds a langchain_community BM25Retriever over the same child
chunks used for the FAISS index, so hybrid search can combine
keyword and semantic relevance.

BM25Retriever does not have native persistence, so we persist the
underlying documents (see retrieval.vectorstore.save/load_child_docs_for_bm25)
and rebuild the in-memory BM25 index from them on load - this is
fast even for several thousand chunks.
"""

import time
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.utils.logger import get_logger, log_event

logger = get_logger("retrieval.bm25")


def build_bm25(child_docs: List[Document], k: int) -> BM25Retriever:
    t0 = time.perf_counter()
    retriever = BM25Retriever.from_documents(child_docs)
    retriever.k = k
    duration_ms = round((time.perf_counter() - t0) * 1000)
    log_event(logger, "INFO", "index.bm25.build", status="success",
              n_docs=len(child_docs), duration_ms=duration_ms)
    return retriever
