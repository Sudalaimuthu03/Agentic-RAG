"""
rag.ingestion.loader
---------------------
Loads PDFs from disk into LangChain `Document` objects, one per
page, with basic metadata (source filename, page number) attached.
Corrupt / unreadable PDFs are skipped and logged rather than
crashing the whole ingestion run.
"""

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.utils.logger import get_logger, log_event

logger = get_logger("ingestion.loader")


def list_pdfs(folder: str) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted(p for p in folder_path.glob("*.pdf") if p.is_file())


def check_pdf_size(path: Path, max_mb: int) -> bool:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        log_event(logger, "WARNING", "pdf.load", status="skipped", file=path.name,
                   reason="too_large", size_mb=round(size_mb, 1), max_mb=max_mb)
        return False
    return True


def load_pdf(path: Path) -> List[Document]:
    """Load a single PDF into one Document per page.

    Returns an empty list (and logs the error) if the PDF cannot be
    read at all, so callers can simply skip it and continue.
    """
    t0 = time.perf_counter()
    docs: List[Document] = []
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                log_event(logger, "ERROR", "pdf.load", status="error", file=path.name,
                           error="encrypted PDF could not be decrypted")
                return []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "WARNING", "pdf.load", status="page_error", file=path.name,
                           page=page_num, error=str(exc))
                text = ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_num,
                        "full_path": str(path),
                    },
                )
            )

        duration_ms = round((time.perf_counter() - t0) * 1000)
        log_event(logger, "INFO", "pdf.load", status="success", file=path.name,
                   pages=len(docs), duration_ms=duration_ms)
    except (PdfReadError, OSError) as exc:
        log_event(logger, "ERROR", "pdf.load", status="error", file=path.name, error=str(exc))
        return []
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "ERROR", "pdf.load", status="error", file=path.name, error=str(exc))
        return []

    return docs


def load_all_pdfs(folder: str, max_size_mb: int) -> List[Document]:
    all_docs: List[Document] = []
    for pdf_path in list_pdfs(folder):
        if not check_pdf_size(pdf_path, max_size_mb):
            continue
        all_docs.extend(load_pdf(pdf_path))
    return all_docs
