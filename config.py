import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

APP_VERSION = "5.4.0"
_ENV_LOADED = load_dotenv(dotenv_path=".env", override=False)


def _s(name, default): return (os.getenv(name) or default).strip()
def _i(name, default):
    try: return int(os.getenv(name, default))
    except (TypeError, ValueError): return default
def _f(name, default):
    try: return float(os.getenv(name, default))
    except (TypeError, ValueError): return default
def _b(name, default): return str(os.getenv(name, str(default))).lower() in {"1","true","yes","on"}

@dataclass
class Settings:
    env_file_found: bool = _ENV_LOADED
    MODEL_NAME: str = field(default_factory=lambda: _s("MODEL_NAME", "llama3.2:latest"))
    USE_NVIDIA: bool = field(default_factory=lambda: _b("USE_NVIDIA", False))
    NVIDIA_MODEL: str = field(default_factory=lambda: _s("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"))
    NVIDIA_API_KEY: str = field(default_factory=lambda: _s("NVIDIA_API_KEY", ""))
    NVIDIA_BASE_URL: str = field(default_factory=lambda: _s("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    TEMPERATURE: float = field(default_factory=lambda: _f("TEMPERATURE", 0.0))
    TOP_P: float = field(default_factory=lambda: _f("TOP_P", 0.9))
    NUM_CTX: int = field(default_factory=lambda: _i("NUM_CTX", 4096))
    EMBEDDING_MODEL: str = field(default_factory=lambda: _s("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    CHILD_CHUNK_SIZE: int = field(default_factory=lambda: _i("CHILD_CHUNK_SIZE", 300))
    CHILD_CHUNK_OVERLAP: int = field(default_factory=lambda: _i("CHILD_CHUNK_OVERLAP", 50))
    PARENT_CHUNK_SIZE: int = field(default_factory=lambda: _i("PARENT_CHUNK_SIZE", 2000))
    PARENT_CHUNK_OVERLAP: int = field(default_factory=lambda: _i("PARENT_CHUNK_OVERLAP", 200))
    TOP_K: int = field(default_factory=lambda: _i("TOP_K", 10))
    RERANK_ENABLED: bool = field(default_factory=lambda: _b("RERANK_ENABLED", True))
    RERANKER_TYPE: str = field(default_factory=lambda: _s("RERANKER_TYPE", "crossencoder"))
    RERANKER_MODEL: str = field(default_factory=lambda: _s("RERANKER_MODEL", "ms-marco-MiniLM-L-6-v2"))
    RERANKER_LOCAL_PATH: str = field(default_factory=lambda: _s("RERANKER_LOCAL_PATH", "./models/ms-marco-MiniLM-L-6-v2"))
    RERANKER_AUTO_DOWNLOAD: bool = field(default_factory=lambda: _b("RERANKER_AUTO_DOWNLOAD", False))
    RERANK_TOP_K: int = field(default_factory=lambda: _i("RERANK_TOP_K", 5))
    HYBRID_ENABLED: bool = field(default_factory=lambda: _b("HYBRID_ENABLED", True))
    FAISS_WEIGHT: float = field(default_factory=lambda: _f("FAISS_WEIGHT", 0.5))
    BM25_WEIGHT: float = field(default_factory=lambda: _f("BM25_WEIGHT", 0.5))
    VECTOR_DB_PATH: str = field(default_factory=lambda: _s("VECTOR_DB_PATH", "./vectorstore"))
    UPLOAD_FOLDER: str = field(default_factory=lambda: _s("UPLOAD_FOLDER", "./uploads"))
    LOG_DIR: str = field(default_factory=lambda: _s("LOG_DIR", "./logs"))
    MAX_PDF_SIZE_MB: int = field(default_factory=lambda: _i("MAX_PDF_SIZE_MB", 100))
    MAX_QUESTION_LENGTH: int = field(default_factory=lambda: _i("MAX_QUESTION_LENGTH", 2000))
    OLLAMA_BASE_URL: str = field(default_factory=lambda: _s("OLLAMA_BASE_URL", "http://localhost:11434"))
    MAX_AGENT_ITERATIONS: int = field(default_factory=lambda: _i("MAX_AGENT_ITERATIONS", 3))
    MAX_LLM_CALLS_PER_TURN: int = field(default_factory=lambda: _i("MAX_LLM_CALLS_PER_TURN", 3))
    MAX_RETRIEVAL_OPERATIONS_PER_TURN: int = field(default_factory=lambda: _i("MAX_RETRIEVAL_OPERATIONS_PER_TURN", 2))
    MAX_GUARDRAIL_RETRIES: int = field(default_factory=lambda: _i("MAX_GUARDRAIL_RETRIES", 1))
    ALLOW_MODEL_FALLBACK: bool = field(default_factory=lambda: _b("ALLOW_MODEL_FALLBACK", False))
    FLASK_HOST: str = field(default_factory=lambda: _s("FLASK_HOST", "127.0.0.1"))
    FLASK_PORT: int = field(default_factory=lambda: _i("FLASK_PORT", 5000))
    FLASK_SECRET_KEY: str = field(default_factory=lambda: _s("FLASK_SECRET_KEY", "change-me"))
    FLASK_DEBUG: bool = field(default_factory=lambda: _b("FLASK_DEBUG", False))

    def ensure_dirs(self):
        for p in (self.VECTOR_DB_PATH, self.UPLOAD_FOLDER, self.LOG_DIR): Path(p).mkdir(parents=True, exist_ok=True)

    def as_dict(self):
        return {k:v for k,v in self.__dict__.items() if k != "FLASK_SECRET_KEY"}

settings = Settings()
