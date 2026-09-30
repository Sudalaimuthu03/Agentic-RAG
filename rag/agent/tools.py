from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from rag.semantic_contracts import EvidenceItem, ResolvedRequest
from rag.utils.logger import get_logger
from rag.utils.context import get_active_filter

logger = get_logger("agent.tools")
_LAST_DOCUMENTS: list[Document] = []


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class SummaryInput(BaseModel):
    document: str = Field(min_length=1, max_length=300)


class ClarifyInput(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


def reset_last_documents() -> None:
    _LAST_DOCUMENTS.clear()


def get_last_documents() -> list[Document]:
    return list(_LAST_DOCUMENTS)


def _target_for_doc(doc: Document, resolved: ResolvedRequest) -> tuple[str | None, str | None, str | None]:
    source = str(doc.metadata.get("source", "unknown"))
    text = doc.page_content.lower()
    targets = list(resolved.targets or ())
    if len(targets) <= 1:
        return None, resolved.entity, resolved.variant
    for idx, target in enumerate(targets, 1):
        if target.get("document") and target.get("document") == source:
            return str(idx), target.get("entity"), target.get("variant")
    for idx, target in enumerate(targets, 1):
        entity = str(target.get("entity") or "").lower()
        variant = str(target.get("variant") or "").lower().replace("-", " ")
        if entity and entity in text and (not variant or variant in text.replace("-", " ")):
            return str(idx), target.get("entity"), target.get("variant")
    return None, resolved.entity, resolved.variant


def _to_evidence(doc: Document, resolved: ResolvedRequest) -> dict:
    source = str(doc.metadata.get("source", "unknown"))
    target_id, entity, variant = _target_for_doc(doc, resolved)
    return EvidenceItem(
        source=source,
        page=doc.metadata.get("page"),
        chunk_id=doc.metadata.get("chunk_id"),
        parent_id=doc.metadata.get("parent_id"),
        text=doc.page_content,
        entity=entity,
        variant=variant,
        topic=resolved.topic,
        retrieval_score=doc.metadata.get("score") or doc.metadata.get("retrieval_score"),
        rerank_score=doc.metadata.get("rerank_score"),
        target_id=target_id,
    ).as_dict()


def _scope_docs(docs: list[Document], resolved: ResolvedRequest) -> list[Document]:
    allowed = set(resolved.evidence_scope)
    result = [d for d in docs if not allowed or d.metadata.get("source") in allowed]
    if resolved.document:
        result = [d for d in result if d.metadata.get("source") == resolved.document]
    return result


def build_tools(rag_retriever: Any, resolved: ResolvedRequest, indexed_sources_fn):
    """Build exactly four bounded tools. The resolved request is immutable authority.

    For multi-target plans, search_documents satisfies each authorized target independently
    so retrieval ranking cannot collapse the target set into one winner.
    """
    def _retrieve(scope: tuple[str, ...], query: str) -> list[Document]:
        if not scope:
            return list(rag_retriever.invoke(query))
        from rag.chains.retriever_chain import RagRetriever
        scoped = RagRetriever(
            hybrid_retriever=rag_retriever.hybrid_retriever, reranker=rag_retriever.reranker,
            parent_store=rag_retriever.parent_store, rerank_top_k=rag_retriever.rerank_top_k,
            active_filter_ref=rag_retriever.active_filter_ref, scope_sources=scope,
        )
        return list(scoped.invoke(query))

    def search_documents(query: str) -> dict:
        if resolved.knowledge_status != "AUTHORIZED":
            return {"status": "blocked", "reason": resolved.knowledge_status, "evidence": []}
        targets = list(resolved.targets or ())
        all_docs: list[Document] = []
        if len(targets) > 1:
            for target in targets:
                source = target.get("document")
                scope = (str(source),) if source else tuple(resolved.evidence_scope)
                target_query = " ".join(str(x) for x in (query, target.get("entity"), target.get("variant"), target.get("topic"), target.get("fact")) if x)
                all_docs.extend(_retrieve(scope, target_query))
        else:
            all_docs = _retrieve(tuple(resolved.evidence_scope), query)

        # Stable source de-duplication while retaining the best retrieved occurrence.
        seen = set(); docs=[]
        for d in all_docs:
            key=(d.metadata.get("source"), d.metadata.get("parent_id"), d.metadata.get("chunk_id"))
            if key in seen: continue
            seen.add(key); docs.append(d)
        _LAST_DOCUMENTS.clear(); _LAST_DOCUMENTS.extend(docs)
        evidence = [_to_evidence(d, resolved) for d in docs]
        return {
            "status": "known" if evidence else "unknown",
            "resolved_request": resolved.as_dict(),
            "evidence": evidence,
            "sources": list(dict.fromkeys(e["source"] for e in evidence)),
            "instruction": "Use only this evidence. All targets in the authoritative plan must remain represented; do not substitute, merge, add, or remove targets.",
        }

    def summarize_document(document: str) -> dict:
        if resolved.knowledge_status != "AUTHORIZED":
            return {"status": "blocked", "reason": resolved.knowledge_status, "evidence": []}
        if document not in set(resolved.document_scope):
            return {"status": "error", "reason": "document is outside authoritative scope", "evidence": []}
        docs = _retrieve((document,), document)
        _LAST_DOCUMENTS.clear(); _LAST_DOCUMENTS.extend(docs)
        return {"status": "known" if docs else "unknown", "evidence": [_to_evidence(d, resolved) for d in docs]}

    def list_available_documents() -> dict:
        return {"status": "known", "documents": indexed_sources_fn()}

    def ask_clarification(reason: str) -> dict:
        candidates = list(resolved.candidate_documents[:10])
        return {
            "status": "clarify",
            "reason": reason,
            "candidate_entities": list(resolved.candidate_entities[:10]),
            "candidate_documents": candidates,
            "candidates": candidates,
            "instruction": "Use these discovered candidates only to formulate a natural clarification. Do not present candidate discovery as factual authorization and do not list unrelated corpus documents.",
        }

    return {
        "search_documents": StructuredTool.from_function(search_documents, name="search_documents", description="Retrieve authorized evidence for the already-resolved request.", args_schema=SearchInput),
        "summarize_document": StructuredTool.from_function(summarize_document, name="summarize_document", description="Retrieve evidence needed to summarize the already-resolved document.", args_schema=SummaryInput),
        "list_available_documents": StructuredTool.from_function(list_available_documents, name="list_available_documents", description="List the authoritative indexed document catalog."),
        "ask_clarification": StructuredTool.from_function(ask_clarification, name="ask_clarification", description="Ask for clarification when the authoritative request is incomplete or ambiguous.", args_schema=ClarifyInput),
    }


def indexed_sources() -> list[str]:
    from config import settings
    from rag.ingestion.loader import list_pdfs
    return [p.name for p in list_pdfs(settings.UPLOAD_FOLDER)]


def get_rag_retriever(resolved: ResolvedRequest | None = None):
    from rag.core.state import state
    from rag.chains.retriever_chain import RagRetriever
    return RagRetriever(
        hybrid_retriever=state.hybrid_retriever,
        reranker=state.reranker,
        parent_store=state.doc_store or {},
        rerank_top_k=5,
        active_filter_ref=get_active_filter,
        scope_sources=tuple(resolved.evidence_scope) if resolved else (),
    )
