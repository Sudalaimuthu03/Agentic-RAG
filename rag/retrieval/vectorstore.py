"""
rag.retrieval.vectorstore
---------------------------
Build / save / load the FAISS index of child chunks, plus the
parent document store (pickled dict of parent_id -> Document) that
lives alongside it.
"""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.utils.logger import get_logger, log_event

logger = get_logger("retrieval.vectorstore")

FAISS_SUBDIR = "faiss_index"
PARENT_STORE_FILE = "parent_store.pkl"
BM25_STORE_FILE = "bm25_docs.pkl"


def build_faiss(child_docs: List[Document], embedder) -> FAISS:
    """Embed all child chunks, then build the FAISS index from the
    precomputed vectors. Split into two explicit steps (instead of
    FAISS.from_documents, which does both silently) so embed.run and
    index.faiss.build can be timed and logged separately.
    """
    texts = [d.page_content for d in child_docs]
    metadatas = [d.metadata for d in child_docs]

    t0 = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    embed_ms = round((time.perf_counter() - t0) * 1000)
    log_event(logger, "INFO", "embed.run", status="success",
              model=getattr(embedder, "model_name", "unknown"),
              n_chunks=len(texts), duration_ms=embed_ms)

    t1 = time.perf_counter()
    vs = FAISS.from_embeddings(list(zip(texts, vectors)), embedder, metadatas=metadatas)
    build_ms = round((time.perf_counter() - t1) * 1000)
    log_event(logger, "INFO", "index.faiss.build", status="success",
              n_vectors=len(texts), duration_ms=build_ms)

    return vs


def save_faiss(vs: FAISS, vector_db_path: str) -> None:
    path = Path(vector_db_path) / FAISS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    log_event(logger, "INFO", "index.faiss.save", status="success", path=str(path))


def load_faiss(vector_db_path: str, embedder) -> Optional[FAISS]:
    path = Path(vector_db_path) / FAISS_SUBDIR
    if not path.exists():
        return None
    try:
        return FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "ERROR", "index.faiss.load", status="error", path=str(path), error=str(exc))
        return None


def save_parent_store(parent_store: Dict[str, Document], vector_db_path: str) -> None:
    path = Path(vector_db_path) / PARENT_STORE_FILE
    with open(path, "wb") as f:
        pickle.dump(parent_store, f)
    log_event(logger, "INFO", "index.parent_store.save", status="success",
              n_parents=len(parent_store), path=str(path))


def load_parent_store(vector_db_path: str) -> Dict[str, Document]:
    path = Path(vector_db_path) / PARENT_STORE_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "ERROR", "index.parent_store.load", status="error", path=str(path), error=str(exc))
        return {}


def save_child_docs_for_bm25(child_docs: List[Document], vector_db_path: str) -> None:
    path = Path(vector_db_path) / BM25_STORE_FILE
    with open(path, "wb") as f:
        pickle.dump(child_docs, f)


def load_child_docs_for_bm25(vector_db_path: str) -> List[Document]:
    path = Path(vector_db_path) / BM25_STORE_FILE
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "ERROR", "index.bm25.load", status="error", path=str(path), error=str(exc))
        return []


def vector_store_size_mb(vector_db_path: str) -> float:
    path = Path(vector_db_path)
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)
