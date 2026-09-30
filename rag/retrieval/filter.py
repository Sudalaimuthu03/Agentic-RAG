"""
rag.retrieval.filter
-----------------------
Applies active metadata filters (set via the /filter CLI command)
to a list of retrieved Documents. Because BM25Retriever and the
EnsembleRetriever don't support native metadata filter kwargs the
way a pure vectorstore retriever can, filtering is done as a
post-retrieval pass here - simple, predictable, and works
identically regardless of which underlying retriever produced the
candidates.
"""

from typing import Any, Dict, List

from langchain_core.documents import Document


def parse_filter_command(arg_string: str) -> Dict[str, Any]:
    """Parse `source=manual.pdf` or `page=12` into a filter dict."""
    filters: Dict[str, Any] = {}
    if not arg_string.strip():
        return filters
    for part in arg_string.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "page":
            try:
                value = int(value)
            except ValueError:
                pass
        filters[key] = value
    return filters


def apply_filter(docs: List[Document], active_filter: Dict[str, Any]) -> List[Document]:
    if not active_filter:
        return docs

    filtered = []
    for doc in docs:
        match = True
        for key, value in active_filter.items():
            doc_value = doc.metadata.get(key)
            if key == "source":
                if not doc_value or str(value).lower() not in str(doc_value).lower():
                    match = False
                    break
            else:
                if doc_value != value:
                    match = False
                    break
        if match:
            filtered.append(doc)
    return filtered
