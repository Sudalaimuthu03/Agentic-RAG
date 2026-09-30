"""
rag.utils.context
--------------------
Request-scoped context (request_id, session_id) implemented with
contextvars so any module can log the current request/session
without having it threaded through every function signature.

Flask sets these once per HTTP request (see rag/api/routes.py
before_request hook); the JSON log formatter in rag.utils.logger
reads them automatically and attaches them to every log line.

Note on agent_info specifically: it is NOT stored as a plain
contextvar like the others above. LangChain's create_retrieval_chain
runs its retrieval branch through a thread pool
(RunnableParallel -> get_executor_for_config), and each worker thread
runs inside a *copy* of the caller's contextvars.Context. Reads inside
that copy see whatever was set before the copy was taken (so
request_id/session_id show up correctly in agent.* logs), but writes
made inside the copy - like AgenticRagRetriever calling
set_agent_info() - only mutate that copy and are silently lost once
the worker thread finishes; the original context back in routes.py's
generate() never sees them. Confirmed by testing: chat.qa's agent
field showed {} even though the agent.done log line for the same
request_id showed real grading data.

Fix: keep a single shared dict (_agent_info_store), not a contextvar,
keyed by request_id. Only the *variable binding* is copied when a
context is copied - the dict object itself is the same object in
memory in every thread/context, so mutating its contents (rather than
reassigning the variable) is visible everywhere, including back in
the main thread.
"""

import contextvars
import uuid

_request_id_var: contextvars.ContextVar = contextvars.ContextVar("request_id", default=None)
_session_id_var: contextvars.ContextVar = contextvars.ContextVar("session_id", default=None)
_active_filter_var: contextvars.ContextVar = contextvars.ContextVar("active_filter", default=None)

# Shared across threads/contexts on purpose - see note above. Keyed by
# request_id so concurrent requests never see each other's agent data.
_agent_info_store: dict = {}


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id():
    return _request_id_var.get()


def set_session_id(session_id: str) -> None:
    _session_id_var.set(session_id)


def get_session_id():
    return _session_id_var.get()


def set_active_filter(active_filter: dict) -> None:
    _active_filter_var.set(active_filter or {})


def get_active_filter() -> dict:
    return _active_filter_var.get() or {}


def set_agent_info(agent_info: dict) -> None:
    """Records the most recent corrective-RAG agent run (attempts,
    whether grading passed, final query used) so routes.py can attach
    it to the per-question chat.qa log line without threading it
    through every function signature.

    Keyed by the current request_id (read via the request_id
    contextvar, which - unlike agent_info - is correctly visible even
    from inside create_retrieval_chain's thread-pooled retrieval
    branch, since it was set before that branch's context was copied).
    """
    request_id = get_request_id()
    if request_id is None:
        return
    _agent_info_store[request_id] = agent_info or {}


def get_agent_info() -> dict:
    """Reads (and clears) the agent info for the current request_id,
    so entries don't accumulate in memory across requests over time.
    """
    request_id = get_request_id()
    if request_id is None:
        return {}
    return _agent_info_store.pop(request_id, {})
