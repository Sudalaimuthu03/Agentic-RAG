import os
import time
from functools import lru_cache
from pathlib import Path
from rag.utils.logger import get_logger, log_event

logger = get_logger("embeddings.embedder")
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

@lru_cache(maxsize=4)
def get_embedder(model_name: str):
    local_path = LOCAL_MODEL_DIR / model_name.replace("/", "_")
    if not local_path.exists():
        raise FileNotFoundError(f"Local embedding model not found: {local_path}. Runtime downloads are disabled in V4.")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from langchain_community.embeddings import HuggingFaceEmbeddings
    t=time.perf_counter()
    embedder=HuggingFaceEmbeddings(model_name=str(local_path), model_kwargs={"device":"cpu","local_files_only":True}, encode_kwargs={"normalize_embeddings":True})
    log_event(logger,"INFO","embed.model_load",status="success",model=model_name,source="local",duration_ms=round((time.perf_counter()-t)*1000))
    return embedder
