#!/usr/bin/env python3
"""
app.py
-------
Flask entry point for pdf-rag - an offline RAG chatbot for PDFs,
served as a web app (previously a console/CLI app).

Run with:
    python app.py
"""

import sys

from flask import Flask, render_template

from config import settings
from rag.api.routes import bp as api_bp
from rag.config.validator import validate_all
from rag.core import pipeline
from rag.core.state import state
from rag.ingestion.loader import list_pdfs
from rag.utils.logger import get_logger, log_event, setup_logging

logger = get_logger("app")


def _run_startup_checks() -> bool:
    """Validate configuration & environment. Returns True if the app
    can proceed (possibly with warnings), False if it must abort.
    """
    settings.ensure_dirs()
    setup_logging(settings.LOG_DIR)

    result = validate_all(settings)

    for warning in result.warnings:
        log_event(logger, "WARNING", "startup.validation", message=warning)

    if result.errors:
        for error in result.errors:
            log_event(logger, "ERROR", "startup.validation", message=error)
        print("Startup validation failed - check logs/Logs.log for details:")
        for error in result.errors:
            print(f"  - {error}")
        return False

    if not result.model_available:
        return False
    from rag.core.model_provider import configured_model_name
    state.current_model = configured_model_name(settings)
    log_event(logger, "INFO", "startup.validation", status="passed")
    return True


def _load_or_build_index() -> None:
    loaded = pipeline.load_existing_index(settings)
    if loaded:
        return

    log_event(logger, "INFO", "startup.index", status="no_existing_index",
              upload_folder=settings.UPLOAD_FOLDER)
    if list_pdfs(settings.UPLOAD_FOLDER):
        pipeline.reindex_all(settings, force=False)
    else:
        log_event(logger, "WARNING", "startup.index", status="no_pdfs_found",
                  upload_folder=settings.UPLOAD_FOLDER)


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = settings.FLASK_SECRET_KEY
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    logger_boot = get_logger("app")
    log_event(logger_boot, "INFO", "app.starting", pid=__import__("os").getpid())

    if not _run_startup_checks():
        log_event(logger_boot, "ERROR", "app.startup_failed")
        sys.exit(1)

    print("Loading existing vector store (if any)...")
    _load_or_build_index()

    flask_app = create_app()
    log_event(logger_boot, "INFO", "app.ready", host=settings.FLASK_HOST, port=settings.FLASK_PORT,
              model=state.current_model, pdfs_indexed=state.num_pdfs, chunks_indexed=state.num_chunks)
    print(f"Serving on http://{settings.FLASK_HOST}:{settings.FLASK_PORT}")
    flask_app.run(host=settings.FLASK_HOST, port=settings.FLASK_PORT, debug=settings.FLASK_DEBUG)
