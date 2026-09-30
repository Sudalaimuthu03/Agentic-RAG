from __future__ import annotations
import shutil, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from rag.utils.logger import get_logger
logger=get_logger("config.validator")
@dataclass
class ValidationResult:
    ok: bool=True; errors: List[str]=field(default_factory=list); warnings: List[str]=field(default_factory=list); installed_ollama_models: List[str]=field(default_factory=list); ollama_running: bool=False; model_available: bool=False
    def add_error(self,msg): self.ok=False; self.errors.append(msg)
    def add_warning(self,msg): self.warnings.append(msg)

def validate_all(s):
    r=ValidationResult()
    for n,v in (("MAX_AGENT_ITERATIONS",s.MAX_AGENT_ITERATIONS),("MAX_LLM_CALLS_PER_TURN",s.MAX_LLM_CALLS_PER_TURN),("MAX_RETRIEVAL_OPERATIONS_PER_TURN",s.MAX_RETRIEVAL_OPERATIONS_PER_TURN)):
        if v<=0: r.add_error(f"{n} must be > 0")
    if s.MAX_GUARDRAIL_RETRIES<0: r.add_error("MAX_GUARDRAIL_RETRIES must be >= 0")
    if s.TOP_K<=0 or s.RERANK_TOP_K<=0: r.add_error("TOP_K and RERANK_TOP_K must be > 0")
    if not (0<=s.TEMPERATURE<=1): r.add_error("TEMPERATURE must be between 0 and 1")
    if not (0<=s.TOP_P<=1): r.add_error("TOP_P must be between 0 and 1")
    if s.FAISS_WEIGHT<0 or s.BM25_WEIGHT<0 or (s.HYBRID_ENABLED and s.FAISS_WEIGHT+s.BM25_WEIGHT==0): r.add_error("Invalid retrieval weights")
    for n,p in (("VECTOR_DB_PATH",s.VECTOR_DB_PATH),("UPLOAD_FOLDER",s.UPLOAD_FOLDER),("LOG_DIR",s.LOG_DIR)):
        try:
            Path(p).mkdir(parents=True,exist_ok=True); t=Path(p)/".v4_write_test"; t.write_text("ok"); t.unlink()
        except OSError as e: r.add_error(f"{n} is not writable: {e}")
    local=Path(__file__).resolve().parents[2]/"models"/s.EMBEDDING_MODEL.replace("/","_")
    if not local.exists(): r.add_error(f"Local embedding model not found: {local}")
    if s.RERANK_ENABLED and not Path(s.RERANKER_LOCAL_PATH).expanduser().resolve().exists(): r.add_error(f"Local reranker model not found: {Path(s.RERANKER_LOCAL_PATH).expanduser().resolve()}")

    if s.USE_NVIDIA:
        if not s.NVIDIA_API_KEY.strip():
            r.add_error("USE_NVIDIA=true but NVIDIA_API_KEY is empty")
        if not s.NVIDIA_BASE_URL.strip():
            r.add_error("NVIDIA_BASE_URL is empty")
        if not s.NVIDIA_MODEL.strip():
            r.add_error("NVIDIA_MODEL is empty")
        # The actual endpoint/model authentication is intentionally checked by
        # the configured ChatOpenAI client at runtime; no secret is logged.
        r.model_available = bool(s.NVIDIA_API_KEY.strip() and s.NVIDIA_MODEL.strip() and s.NVIDIA_BASE_URL.strip())
        r.ollama_running = False
        r.installed_ollama_models = []
        return r

    exe=shutil.which("ollama")
    if not exe: r.add_error("Ollama executable not found on PATH"); return r
    try:
        p=subprocess.run([exe,"list"],capture_output=True,text=True,timeout=10)
        if p.returncode!=0: r.add_error("Ollama is not running"); return r
        r.ollama_running=True
        lines=[x.strip() for x in p.stdout.splitlines()[1:] if x.strip()]
        r.installed_ollama_models=[x.split()[0] for x in lines if x.split()]
        r.model_available=s.MODEL_NAME in r.installed_ollama_models
        if not r.model_available: r.add_error(f"Configured model '{s.MODEL_NAME}' is not installed; silent fallback is disabled")
    except Exception as e: r.add_error(f"Ollama validation failed: {e}")
    return r
