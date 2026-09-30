"""
rag.utils.logger
-----------------
Single structured (JSON-lines) log file: logs/Logs.log

Every log call produces one JSON object per line, e.g.:

    {"ts":"2026-08-05 14:02:11","level":"INFO","event":"pdf.load",
     "status":"success","file":"manual.pdf","pages":18,"duration_ms":340}

Plain string log calls (logger.info("some message")) still work and
are wrapped as {"event": "log", "message": "..."} so third-party
libraries and ad-hoc debug lines don't crash the formatter.

request_id / session_id (see rag.utils.context) are attached to
every line automatically when set, so a single user question can be
traced end-to-end (query.received -> retrieval.hybrid ->
retrieval.rerank -> retrieval.parent_swap -> llm.answer -> chat.qa)
just by grepping the request_id.

Usage:
    from rag.utils.logger import get_logger, log_event
    logger = get_logger(__name__)
    log_event(logger, "INFO", "pdf.load", status="success", file="x.pdf")
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag.utils.context import get_request_id, get_session_id

_CONFIGURED = False
_LOGGER_NAME = "pdf_rag_app"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, _DATEFMT),
            "level": record.levelname,
        }

        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["event"] = "log"
            payload["message"] = record.getMessage()

        request_id = get_request_id()
        if request_id and "request_id" not in payload:
            payload["request_id"] = request_id

        session_id = get_session_id()
        if session_id and "session_id" not in payload:
            payload["session_id"] = session_id

        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({"ts": payload["ts"], "level": payload["level"], "event": "log_error",
                                "message": "failed to serialize log payload"})


def setup_logging(log_dir: str = "./logs") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_path / "Logs.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    root.propagate = False

    # Silence overly chatty third-party libraries in the log file
    for noisy in ("httpx", "urllib3", "sentence_transformers", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    # Keep every module's logger nested under the single configured
    # logger so all records reach the one JSON handler above.
    full_name = name if name == _LOGGER_NAME else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


def log_event(logger: logging.Logger, level: str, event: str, **fields) -> None:
    """Log one structured JSON event.

    log_event(logger, "INFO", "pdf.load", status="success", file="x.pdf", pages=18, duration_ms=340)
    """
    payload = {"event": event, **fields}
    logger.log(getattr(logging, level.upper(), logging.INFO), payload)
