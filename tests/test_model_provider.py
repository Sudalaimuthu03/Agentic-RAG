import sys
import types


def test_ollama_provider_contract(monkeypatch):
    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_ollama", types.SimpleNamespace(ChatOllama=FakeChatOllama))
    from rag.core.model_provider import build_llm, configured_model_name, configured_provider

    class S:
        USE_NVIDIA = False
        MODEL_NAME = "llama3.2:latest"
        NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
        NVIDIA_API_KEY = ""
        NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
        TEMPERATURE = 0.0
        TOP_P = 0.9
        NUM_CTX = 4096
        OLLAMA_BASE_URL = "http://localhost:11434"

    s = S()
    llm = build_llm(s)
    assert llm.kwargs["model"] == "llama3.2:latest"
    assert llm.kwargs["base_url"] == "http://localhost:11434"
    assert configured_provider(s) == "ollama"
    assert configured_model_name(s) == "llama3.2:latest"


def test_nvidia_provider_contract(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    from rag.core.model_provider import build_llm, configured_model_name, configured_provider

    class S:
        USE_NVIDIA = True
        MODEL_NAME = "llama3.2:latest"
        NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
        NVIDIA_API_KEY = "test-key"
        NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
        TEMPERATURE = 0.0
        TOP_P = 0.9
        NUM_CTX = 4096
        OLLAMA_BASE_URL = "http://localhost:11434"

    s = S()
    llm = build_llm(s)
    assert llm.kwargs["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert llm.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert llm.kwargs["api_key"] == "test-key"
    assert configured_provider(s) == "nvidia"
    assert configured_model_name(s) == "nvidia/nemotron-3-super-120b-a12b"


def test_nvidia_requires_api_key(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    from rag.core.model_provider import build_llm

    class S:
        USE_NVIDIA = True
        MODEL_NAME = "llama3.2:latest"
        NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
        NVIDIA_API_KEY = ""
        NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
        TEMPERATURE = 0.0
        TOP_P = 0.9
        NUM_CTX = 4096
        OLLAMA_BASE_URL = "http://localhost:11434"

    try:
        build_llm(S())
    except RuntimeError as exc:
        assert "NVIDIA_API_KEY" in str(exc)
    else:
        raise AssertionError("NVIDIA mode must reject an empty API key")
