from __future__ import annotations

from typing import Any, Optional


def build_llm(settings: Any, model_name: Optional[str] = None):
    """Build the configured chat model without changing the agent/RAG pipeline.

    USE_NVIDIA=false -> local Ollama
    USE_NVIDIA=true  -> NVIDIA OpenAI-compatible endpoint
    """
    if settings.USE_NVIDIA:
        from langchain_openai import ChatOpenAI

        api_key = (settings.NVIDIA_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("USE_NVIDIA=true but NVIDIA_API_KEY is not configured")
        return ChatOpenAI(
            model=model_name or settings.NVIDIA_MODEL,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            base_url=settings.NVIDIA_BASE_URL,
            api_key=api_key,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name or settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
        num_ctx=settings.NUM_CTX,
        base_url=settings.OLLAMA_BASE_URL,
    )


def configured_model_name(settings: Any) -> str:
    return settings.NVIDIA_MODEL if settings.USE_NVIDIA else settings.MODEL_NAME


def configured_provider(settings: Any) -> str:
    return "nvidia" if settings.USE_NVIDIA else "ollama"
