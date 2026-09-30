"""
rag.ingestion.dedupe
----------------------
Duplicate PDF detection via SHA256 content hashing, backed by a
`metadata.json` file stored inside the vector store folder.

metadata.json schema:
{
  "files": {
      "<sha256>": {
          "filename": "manual.pdf",
          "num_pages": 42,
          "num_chunks": 310,
          "indexed_at": "2026-07-31T10:00:00"
      },
      ...
  },
  "stats": {
      "num_pdfs": 3,
      "num_chunks": 950
  }
}

No document title field - matching/display throughout this system
uses filenames only, by design (no title extraction anywhere).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from rag.utils.logger import get_logger, log_event

logger = get_logger("ingestion.dedupe")

METADATA_FILENAME = "metadata.json"


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _metadata_path(vector_db_path: str) -> Path:
    return Path(vector_db_path) / METADATA_FILENAME


def load_metadata(vector_db_path: str) -> Dict:
    path = _metadata_path(vector_db_path)
    if not path.exists():
        return {"files": {}, "stats": {"num_pdfs": 0, "num_chunks": 0}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log_event(logger, "ERROR", "pdf.dedupe_check", status="error", error=f"metadata.json corrupt: {exc}")
        return {"files": {}, "stats": {"num_pdfs": 0, "num_chunks": 0}}


def save_metadata(vector_db_path: str, metadata: Dict) -> None:
    path = _metadata_path(vector_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def is_duplicate(metadata: Dict, file_hash: str) -> Optional[str]:
    """Return the filename already indexed under this hash, or None."""
    entry = metadata.get("files", {}).get(file_hash)
    return entry["filename"] if entry else None


def register_file(metadata: Dict, file_hash: str, filename: str, num_pages: int, num_chunks: int) -> None:
    metadata.setdefault("files", {})[file_hash] = {
        "filename": filename,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    stats = metadata.setdefault("stats", {"num_pdfs": 0, "num_chunks": 0})
    stats["num_pdfs"] = len(metadata["files"])
    stats["num_chunks"] = sum(v["num_chunks"] for v in metadata["files"].values())
