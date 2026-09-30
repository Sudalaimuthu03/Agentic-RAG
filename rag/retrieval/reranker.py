from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from rag.utils.logger import get_logger, log_event

logger = get_logger("retrieval.reranker")

class BaseReranker:
    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]: raise NotImplementedError

class CrossEncoderReranker(BaseReranker):
    def __init__(self, local_path: str):
        p = Path(local_path).expanduser()
        if not p.is_absolute(): p = (Path.cwd() / p).resolve()
        if not p.exists(): raise FileNotFoundError(f"Local reranker model not found: {p}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(p), local_files_only=True)
        self.path = str(p)

    def rerank(self, query, docs, top_k):
        if not docs: return []
        scores = self.model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        out=[]
        for d,s in ranked:
            d.metadata = dict(d.metadata); d.metadata["rerank_score"] = float(s); out.append(d)
        return out

def get_reranker(reranker_type: str, model_name: str, local_path: str = "", auto_download: bool = False) -> Optional[BaseReranker]:
    if not local_path:
        raise ValueError("RERANKER_LOCAL_PATH is required for offline reranking")
    if not auto_download:
        return CrossEncoderReranker(local_path)
    # Explicit opt-in only; V4 default remains offline.
    return CrossEncoderReranker(local_path)
