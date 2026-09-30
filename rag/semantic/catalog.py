from __future__ import annotations

from pathlib import Path
from typing import Any


def build_catalog(documents: dict[str, str]) -> list[dict[str, Any]]:
    catalog = []
    for source, text in documents.items():
        first_nonempty = next((x.strip() for x in text.splitlines() if x.strip()), "")
        catalog.append({
            "document_id": source,
            "filename": Path(source).name,
            "metadata": {},
            "title_hint": first_nonempty[:300],
            "catalog_membership": True,
        })
    return catalog
