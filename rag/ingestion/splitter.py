"""
rag.ingestion.splitter
------------------------
Implements the "better chunking" + "parent document retrieval"
requirements:

  - Parent chunks: large chunks (PARENT_CHUNK_SIZE) that are stored
    verbatim in an in-memory doc store and returned to the LLM as
    context.
  - Child chunks: small chunks (CHILD_CHUNK_SIZE) derived from each
    parent, used only for embedding / retrieval. Each child carries
    a `parent_id` in its metadata pointing back to its parent.
  - If markdown-style heading markers (#, ##, ...) are detected in
    the page text, MarkdownHeaderTextSplitter is used first so that
    section boundaries are respected; the detected heading becomes
    `section_title` metadata.
  - Simple table blocks (lines with multiple '|' pipe delimiters, a
    common PDF-extracted table shape) are detected and kept whole -
    they are never split across chunk boundaries.
"""

import re
import time
import uuid
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.utils.logger import get_logger, log_event

logger = get_logger("ingestion.splitter")

_HEADING_RE = re.compile(r"^(#{1,6})\s+.+", re.MULTILINE)
_TABLE_LINE_RE = re.compile(r"^\s*(\S+\s*\|){2,}\s*\S+\s*$", re.MULTILINE)


def _looks_like_markdown(text: str) -> bool:
    return bool(_HEADING_RE.search(text))


def _extract_table_blocks(text: str) -> Tuple[str, List[str]]:
    """Pull out contiguous table-like line blocks so they are never
    split mid-table. Returns (text_with_placeholders, [table_blocks]).
    """
    lines = text.split("\n")
    out_lines: List[str] = []
    tables: List[str] = []
    buffer: List[str] = []

    def flush_buffer():
        if buffer:
            placeholder = f"[[TABLE_BLOCK_{len(tables)}]]"
            tables.append("\n".join(buffer))
            out_lines.append(placeholder)
            buffer.clear()

    for line in lines:
        if _TABLE_LINE_RE.match(line):
            buffer.append(line)
        else:
            flush_buffer()
            out_lines.append(line)
    flush_buffer()

    return "\n".join(out_lines), tables


def _restore_table_blocks(text: str, tables: List[str]) -> str:
    for i, table in enumerate(tables):
        text = text.replace(f"[[TABLE_BLOCK_{i}]]", table)
    return text


def _split_parent_text(text: str, parent_chunk_size: int, parent_overlap: int) -> List[Tuple[str, str]]:
    """Return list of (chunk_text, section_title) for one page's text."""
    results: List[Tuple[str, str]] = []

    protected_text, tables = _extract_table_blocks(text)

    if _looks_like_markdown(protected_text):
        headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        try:
            md_docs = md_splitter.split_text(protected_text)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "WARNING", "chunk.split", status="markdown_fallback", error=str(exc))
            md_docs = [Document(page_content=protected_text, metadata={})]
    else:
        md_docs = [Document(page_content=protected_text, metadata={})]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for md_doc in md_docs:
        section_title = md_doc.metadata.get("h3") or md_doc.metadata.get("h2") or md_doc.metadata.get("h1") or ""
        restored = _restore_table_blocks(md_doc.page_content, tables)
        for parent_chunk in parent_splitter.split_text(restored):
            final_chunk = _restore_table_blocks(parent_chunk, tables) if "[[TABLE_BLOCK_" in parent_chunk else parent_chunk
            results.append((final_chunk, section_title))

    return results


def split_into_parent_child(
    page_documents: List[Document],
    child_chunk_size: int,
    child_chunk_overlap: int,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
) -> Tuple[List[Document], Dict[str, Document]]:
    """Given per-page Documents, produce:

    - child_docs: List[Document] (small, for embedding) with metadata
      {source, page, section_title, chunk_id, parent_id}
    - parent_store: Dict[parent_id -> Document] (large, for context)
    """
    t0 = time.perf_counter()
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_docs: List[Document] = []
    parent_store: Dict[str, Document] = {}

    for page_doc in page_documents:
        source = page_doc.metadata.get("source", "unknown")
        page = page_doc.metadata.get("page", 0)

        parent_pieces = _split_parent_text(
            page_doc.page_content, parent_chunk_size, parent_chunk_overlap
        )

        for parent_text, section_title in parent_pieces:
            if not parent_text.strip():
                continue
            parent_id = str(uuid.uuid4())
            parent_doc = Document(
                page_content=parent_text,
                metadata={
                    "source": source,
                    "page": page,
                    "section_title": section_title,
                    "parent_id": parent_id,
                },
            )
            parent_store[parent_id] = parent_doc

            # Tables are kept whole: if this parent chunk IS a table
            # (or dominated by table content) don't sub-split it into
            # tiny children that would fragment the table.
            is_table_dominated = parent_text.count("|") > 6

            if is_table_dominated:
                child_texts = [parent_text]
            else:
                child_texts = child_splitter.split_text(parent_text)

            for i, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                chunk_id = f"{parent_id}_{i}"
                child_docs.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "source": source,
                            "page": page,
                            "section_title": section_title,
                            "chunk_id": chunk_id,
                            "parent_id": parent_id,
                        },
                    )
                )

    duration_ms = round((time.perf_counter() - t0) * 1000)
    file_name = page_documents[0].metadata.get("source", "unknown") if page_documents else "unknown"
    log_event(
        logger, "INFO", "chunk.split", status="success", file=file_name,
        pages=len(page_documents), parent_chunks=len(parent_store),
        child_chunks=len(child_docs), duration_ms=duration_ms,
    )
    return child_docs, parent_store
