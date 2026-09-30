"""
rag.memory.history
---------------------
Session-scoped chat history using LangChain's in-memory
ChatMessageHistory, wired up for use with RunnableWithMessageHistory.
"""

from langchain_community.chat_message_histories import ChatMessageHistory

from rag.core.state import state


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in state.chat_histories:
        state.chat_histories[session_id] = ChatMessageHistory()
    return state.chat_histories[session_id]


def clear_session_history(session_id: str = "default") -> None:
    if session_id in state.chat_histories:
        state.chat_histories[session_id].clear()
    state.clear_qa_history(session_id)
    # Full session-scoped reset: active entities/topic/documents,
    # recent candidates, selection context, pending clarification/
    # followup, last tool result, coreference memory - everything in
    # AgentContext. Previously only chat_histories + qa_history were
    # cleared here, so a "cleared" chat could still silently inherit
    # the prior chat's last-resolved-document/pagination/pending-
    # followup state (confirmed bug). Never touches persistent
    # knowledge (vectorstore/doc_store/etc - separate AppState fields).
    state.reset_context(session_id)
