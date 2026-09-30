"""
rag.core.pipeline
--------------------
Orchestration layer that ties ingestion, retrieval, and chains
together. Used by both app.py (startup) and rag.cli.commands
(/reindex, /upload, /model).
"""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from config import Settings
from rag.chains.retriever_chain import RagRetriever
from rag.core.state import state
from rag.embeddings.embedder import get_embedder
from rag.ingestion.dedupe import (
    is_duplicate,
    load_metadata,
    register_file,
    save_metadata,
    sha256_of_file,
)
from rag.ingestion.loader import list_pdfs, load_pdf, check_pdf_size
from rag.ingestion.splitter import split_into_parent_child
from rag.retrieval.bm25 import build_bm25
from rag.retrieval.hybrid import build_hybrid_retriever
from rag.retrieval.reranker import get_reranker
from rag.retrieval.vectorstore import (
    build_faiss,
    load_child_docs_for_bm25,
    load_faiss,
    load_parent_store,
    save_child_docs_for_bm25,
    save_faiss,
    save_parent_store,
    vector_store_size_mb,
)
from rag.utils.context import get_active_filter
from rag.utils.logger import get_logger, log_event
from rag.core.model_provider import build_llm, configured_model_name

logger = get_logger("core.pipeline")


def build_llm(settings: Settings, model_name: Optional[str] = None):
    return __import__("rag.core.model_provider", fromlist=["build_llm"]).build_llm(settings, model_name)


def _wire_retrieval_objects(settings: Settings) -> None:
    """Given state.vectorstore / state.doc_store / bm25 docs already
    populated, (re)build hybrid retriever, reranker, RagRetriever,
    history-aware retriever, LLM, and the final chain.
    """
    bm25_docs = load_child_docs_for_bm25(settings.VECTOR_DB_PATH)
    state.bm25_retriever = build_bm25(bm25_docs, settings.TOP_K) if bm25_docs else None

    state.hybrid_retriever = build_hybrid_retriever(
        state.vectorstore,
        state.bm25_retriever,
        settings.TOP_K,
        settings.HYBRID_ENABLED,
        settings.FAISS_WEIGHT,
        settings.BM25_WEIGHT,
    )

    state.reranker = (
        get_reranker(settings.RERANKER_TYPE, settings.RERANKER_MODEL, settings.RERANKER_LOCAL_PATH, settings.RERANKER_AUTO_DOWNLOAD) if settings.RERANK_ENABLED else None
    )

    rag_retriever = RagRetriever(
        hybrid_retriever=state.hybrid_retriever,
        reranker=state.reranker,
        parent_store=state.doc_store or {},
        rerank_top_k=min(settings.RERANK_TOP_K, settings.TOP_K),
        active_filter_ref=get_active_filter,
    )

    state.current_model = state.current_model or configured_model_name(settings)
    state.llm = build_llm(settings, state.current_model)

    # The runtime route performs semantic understanding and contract resolution
    # before the bounded tool-calling agent. The agent remains responsible for
    # choosing among the four authorized execution capabilities inside that scope.
    state.agent_graph = True


def load_existing_index(settings: Settings) -> bool:
    """Try to load a previously built index from disk. Returns True
    if a usable index was found and loaded.
    """
    embedder = get_embedder(settings.EMBEDDING_MODEL)
    vs = load_faiss(settings.VECTOR_DB_PATH, embedder)
    if vs is None:
        return False

    state.vectorstore = vs
    state.doc_store = load_parent_store(settings.VECTOR_DB_PATH)

    metadata = load_metadata(settings.VECTOR_DB_PATH)
    stats = metadata.get("stats", {"num_pdfs": 0, "num_chunks": 0})
    state.num_pdfs = stats.get("num_pdfs", 0)
    state.num_chunks = stats.get("num_chunks", 0)
    state.vector_store_size_mb = vector_store_size_mb(settings.VECTOR_DB_PATH)

    _wire_retrieval_objects(settings)
    return True


def reindex_all(settings: Settings, force: bool = False, progress_cb=None) -> dict:
    """Scan UPLOAD_FOLDER, skip already-indexed (by hash) PDFs unless
    force=True, and (re)build the FAISS + BM25 + parent store.

    Returns a summary dict: {added, skipped_duplicates, skipped_errors, total_chunks}
    """
    t_start = time.perf_counter()
    metadata = load_metadata(settings.VECTOR_DB_PATH) if not force else {"files": {}, "stats": {"num_pdfs": 0, "num_chunks": 0}}
    embedder = get_embedder(settings.EMBEDDING_MODEL)

    pdf_paths = list_pdfs(settings.UPLOAD_FOLDER)
    if not pdf_paths:
        return {"added": 0, "skipped_duplicates": 0, "skipped_errors": 0, "total_chunks": 0, "no_pdfs": True}

    all_child_docs = []
    all_parent_store = {} if force else load_parent_store(settings.VECTOR_DB_PATH)
    existing_bm25_docs = [] if force else load_child_docs_for_bm25(settings.VECTOR_DB_PATH)
    all_child_docs.extend(existing_bm25_docs)

    added, dup_count, error_count = 0, 0, 0

    for i, pdf_path in enumerate(pdf_paths):
        if progress_cb:
            progress_cb(i + 1, len(pdf_paths), pdf_path.name)

        if not check_pdf_size(pdf_path, settings.MAX_PDF_SIZE_MB):
            error_count += 1
            continue

        file_hash = sha256_of_file(pdf_path)
        existing_name = is_duplicate(metadata, file_hash)
        if existing_name and not force:
            log_event(logger, "INFO", "pdf.dedupe_check", status="duplicate",
                      file=pdf_path.name, existing_as=existing_name)
            dup_count += 1
            continue
        log_event(logger, "INFO", "pdf.dedupe_check", status="new", file=pdf_path.name)

        page_docs = load_pdf(pdf_path)
        if not page_docs:
            error_count += 1
            continue

        child_docs, parent_store = split_into_parent_child(
            page_docs,
            settings.CHILD_CHUNK_SIZE,
            settings.CHILD_CHUNK_OVERLAP,
            settings.PARENT_CHUNK_SIZE,
            settings.PARENT_CHUNK_OVERLAP,
        )

        all_child_docs.extend(child_docs)
        all_parent_store.update(parent_store)
        register_file(metadata, file_hash, pdf_path.name, len(page_docs), len(child_docs))
        added += 1

    if added == 0 and not force:
        log_event(logger, "INFO", "reindex.summary", added=0, duplicates=dup_count, errors=error_count,
                  total_chunks=len(all_child_docs), duration_ms=round((time.perf_counter() - t_start) * 1000))
        return {
            "added": 0,
            "skipped_duplicates": dup_count,
            "skipped_errors": error_count,
            "total_chunks": len(all_child_docs),
            "no_pdfs": False,
        }

    if not all_child_docs:
        log_event(logger, "INFO", "reindex.summary", added=added, duplicates=dup_count, errors=error_count,
                  total_chunks=0, duration_ms=round((time.perf_counter() - t_start) * 1000))
        return {"added": added, "skipped_duplicates": dup_count, "skipped_errors": error_count, "total_chunks": 0, "no_pdfs": False}

    vs = build_faiss(all_child_docs, embedder)
    save_faiss(vs, settings.VECTOR_DB_PATH)
    save_parent_store(all_parent_store, settings.VECTOR_DB_PATH)
    save_child_docs_for_bm25(all_child_docs, settings.VECTOR_DB_PATH)
    save_metadata(settings.VECTOR_DB_PATH, metadata)

    state.vectorstore = vs
    state.doc_store = all_parent_store
    state.num_pdfs = metadata["stats"]["num_pdfs"]
    state.num_chunks = metadata["stats"]["num_chunks"]
    state.vector_store_size_mb = vector_store_size_mb(settings.VECTOR_DB_PATH)

    _wire_retrieval_objects(settings)

    log_event(logger, "INFO", "reindex.summary", added=added, duplicates=dup_count, errors=error_count,
              total_chunks=len(all_child_docs), duration_ms=round((time.perf_counter() - t_start) * 1000))

    return {
        "added": added,
        "skipped_duplicates": dup_count,
        "skipped_errors": error_count,
        "total_chunks": len(all_child_docs),
        "no_pdfs": False,
    }


def delete_index(settings: Settings) -> None:
    path = Path(settings.VECTOR_DB_PATH)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    state.vectorstore = None
    state.doc_store = None
    state.bm25_retriever = None
    state.hybrid_retriever = None
    state.agent_graph = None
    state.num_pdfs = 0
    state.num_chunks = 0
    state.vector_store_size_mb = 0.0


def switch_model(settings: Settings, model_name: str) -> None:
    installed = get_installed_ollama_models()
    if model_name not in installed:
        raise ValueError(f"Model '{model_name}' is not installed; silent fallback is disabled")
    state.current_model = model_name
    if state.vectorstore is not None:
        _wire_retrieval_objects(settings)


def get_installed_ollama_models() -> list:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return []
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return [ln.split()[0] for ln in lines[1:] if ln.split()]
    except Exception:  # noqa: BLE001
        return []
