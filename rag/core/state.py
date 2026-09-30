from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    # Authoritative current-thread state. Historical/audit fields are never used
    # to override a valid current-turn reference.
    active_targets: List[dict[str, Any]] = field(default_factory=list)
    candidate_targets: List[dict[str, Any]] = field(default_factory=list)
    comparison_set: List[dict[str, Any]] = field(default_factory=list)
    focused_target: Optional[dict[str, Any]] = None
    focused_topic: Optional[str] = None
    active_documents: List[str] = field(default_factory=list)
    presented_items: List[dict[str, Any]] = field(default_factory=list)
    recent_references: List[dict[str, Any]] = field(default_factory=list)
    recent_operations: List[dict[str, Any]] = field(default_factory=list)
    authoritative_resolutions: List[dict[str, Any]] = field(default_factory=list)
    last_successful_target: Optional[dict[str, Any]] = None
    last_successful_target_set: List[dict[str, Any]] = field(default_factory=list)
    last_operation: Optional[str] = None
    last_topic: Optional[str] = None
    last_evidence: List[dict[str, Any]] = field(default_factory=list)
    pending_clarification: Optional[dict[str, Any]] = None
    unresolved_reference: Optional[dict[str, Any]] = None
    state_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_targets": list(self.active_targets),
            "candidate_targets": list(self.candidate_targets),
            "comparison_set": list(self.comparison_set),
            "focused_target": self.focused_target,
            "focused_topic": self.focused_topic,
            "active_documents": list(self.active_documents),
            "presented_items": list(self.presented_items),
            "recent_references": list(self.recent_references),
            "recent_operations": list(self.recent_operations),
            "authoritative_resolutions": list(self.authoritative_resolutions),
            "last_successful_target": self.last_successful_target,
            "last_successful_target_set": list(self.last_successful_target_set),
            "last_operation": self.last_operation,
            "last_topic": self.last_topic,
            "last_evidence": list(self.last_evidence),
            "pending_clarification": self.pending_clarification,
            "unresolved_reference": self.unresolved_reference,
            "state_version": self.state_version,
        }


@dataclass
class AgentContext:
    resolved_request: Optional[dict] = None
    active_documents: List[str] = field(default_factory=list)
    last_resolved_document: Optional[str] = None
    selection_context: Optional[Any] = None
    pending_followup: Optional[str] = None
    last_tool_result: Optional[dict] = None
    query_target: Optional[Dict[str, Any]] = None
    conversation: ConversationState = field(default_factory=ConversationState)


@dataclass
class AppState:
    current_model: str = ""
    show_sources: bool = True
    active_filter_by_session: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vectorstore: Optional[Any] = None
    bm25_retriever: Optional[Any] = None
    hybrid_retriever: Optional[Any] = None
    reranker: Optional[Any] = None
    doc_store: Optional[Any] = None
    agent_context_by_session: Dict[str, AgentContext] = field(default_factory=dict)
    llm: Optional[Any] = None
    agent_graph: Optional[Any] = None
    chat_histories: Dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"
    num_pdfs: int = 0
    num_chunks: int = 0
    vector_store_size_mb: float = 0.0
    qa_history_by_session: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def get_filter(self,s): return self.active_filter_by_session.get(s,{})
    def set_filter(self,s,v): self.active_filter_by_session[s]=dict(v or {})
    def reset_filter(self,s): self.active_filter_by_session.pop(s,None)
    def add_qa(self,s,q,a): self.qa_history_by_session.setdefault(s,[]).append({"question":q,"answer":a})
    def get_qa_history(self,s): return self.qa_history_by_session.get(s,[])
    def clear_qa_history(self,s): self.qa_history_by_session.pop(s,None)
    def get_context(self,s): return self.agent_context_by_session.setdefault(s,AgentContext())
    def reset_context(self,s): self.agent_context_by_session[s]=AgentContext()
    def get_query_target(self,s): return self.get_context(s).query_target
    def set_query_target(self,s,v): self.get_context(s).query_target=dict(v) if v else None
    def get_last_resolved_document(self,s): return self.get_context(s).last_resolved_document
    def set_last_resolved_document(self,s,v): self.get_context(s).last_resolved_document=v or None
    def get_active_documents(self,s): return list(self.get_context(s).active_documents)
    def set_active_documents(self,s,v): self.get_context(s).active_documents=list(v or [])
    def get_pending_followup(self,s): return self.get_context(s).pending_followup
    def set_pending_followup(self,s,v): self.get_context(s).pending_followup=v or None
    def get_pagination(self,s): return self.get_context(s).selection_context
    def set_pagination(self,s,v): self.get_context(s).selection_context=v
    get_selection_context=get_pagination
    set_selection_context=set_pagination
    def get_last_tool_result(self,s): return self.get_context(s).last_tool_result
    def set_last_tool_result(self,s,v): self.get_context(s).last_tool_result=v
    def get_turn_hint(self,s): return None
    def set_turn_hint(self,s,v): return None

    def get_conversation_state(self, s: str) -> ConversationState:
        return self.get_context(s).conversation

    def update_conversation_state(self, s: str, resolved: Any, presented_items: Optional[list[dict[str, Any]]] = None, *,
                                  evidence: Optional[list[dict[str, Any]]] = None, success: bool = False) -> ConversationState:
        """Apply one resolved turn to the authoritative conversational state.

        CLARIFY preserves the existing successful target and records a pending
        clarification. CONTINUE/VERIFY never consult historical targets except the
        explicit authoritative active set. Successful execution promotes the current
        target set to the last-successful set used by deterministic references.
        """
        ctx = self.get_context(s)
        conv = ctx.conversation
        action = getattr(getattr(resolved, "context_action", None), "value", getattr(resolved, "context_action", "RESET"))
        incoming = [dict(t) for t in (getattr(resolved, "targets", ()) or ()) if isinstance(t, dict)]
        if action == "CLARIFY":
            # Never destroy a valid target merely because this turn needs clarification.
            conv.candidate_targets = incoming or list(conv.candidate_targets)
            if not conv.candidate_targets:
                docs = list(getattr(resolved, "candidate_documents", ()) or ())
                entities = list(getattr(resolved, "candidate_entities", ()) or ())
                conv.candidate_targets = [
                    {"entity": entities[i] if i < len(entities) else None, "document": docs[i] if i < len(docs) else None}
                    for i in range(max(len(entities), len(docs)))
                ]
            elif len(conv.candidate_targets) == 1 and len(getattr(resolved, "candidate_documents", ()) or ()) > 1:
                docs = list(getattr(resolved, "candidate_documents", ()) or ())
                entities = list(getattr(resolved, "candidate_entities", ()) or ())
                conv.candidate_targets = [
                    {"entity": entities[i] if i < len(entities) else None, "document": doc}
                    for i, doc in enumerate(docs)
                ]
            candidate_docs = list(getattr(resolved, "candidate_documents", ()) or ())
            candidate_entities = list(getattr(resolved, "candidate_entities", ()) or ())
            missing = list(getattr(resolved, "missing_information", ()) or ())
            kind = "TARGET_SELECTION" if len(conv.candidate_targets) > 1 else ("CONFIRMATION" if any("confirm" in str(x).lower() for x in missing) else "SCOPE_SELECTION")
            conv.pending_clarification = {
                "turn_id": getattr(resolved, "turn_id", None),
                "raw_query": getattr(resolved, "raw_query", ""),
                "original_request": getattr(resolved, "raw_query", ""),
                "clarification_question": " | ".join(missing),
                "candidate_targets": list(conv.candidate_targets),
                "candidate_documents": candidate_docs,
                "candidate_entities": candidate_entities,
                "candidate_variants": list(getattr(resolved, "candidate_variants", ()) or ()),
                "operation": getattr(resolved, "intent", None),
                "original_operation": getattr(resolved, "intent", None),
                "topic": getattr(resolved, "topic", None),
                "original_targets": [dict(t) for t in (getattr(resolved, "targets", ()) or ()) if isinstance(t, dict)],
                "missing_information": missing,
                "kind": kind,
            }
            conv.unresolved_reference = {
                "expression": getattr(resolved, "reference", None),
                "status": "PENDING" if getattr(resolved, "reference", None) else "NONE",
            }
            conv.last_operation = getattr(resolved, "intent", None)
            conv.last_topic = getattr(resolved, "topic", None) or conv.last_topic
            conv.recent_operations.append({"type": action, "targets": list(conv.active_targets), "topic": conv.focused_topic})
        else:
            if action == "RESET":
                conv.active_targets = []
                conv.comparison_set = []
                conv.focused_target = None
                conv.focused_topic = None
                conv.active_documents = []
            elif action == "SWITCH":
                previous_candidates = list(conv.candidate_targets)
                conv.active_targets = incoming
                conv.comparison_set = list(incoming) if len(incoming) > 1 else []
                # Preserve a candidate set only when the new target is one of those
                # candidates; otherwise this is a genuine topic/entity switch.
                if previous_candidates and incoming and all(
                    any(
                        (t.get("document") and t.get("document") == c.get("document"))
                        or (t.get("entity") and c.get("entity") and str(t.get("entity")).lower() == str(c.get("entity")).lower())
                        for c in previous_candidates
                    ) for t in incoming
                ):
                    conv.candidate_targets = previous_candidates
                else:
                    conv.candidate_targets = list(incoming)
            elif action in {"CONTINUE", "VERIFY"}:
                if incoming:
                    if len(incoming) == 1 and len(conv.active_targets) == 1:
                        # Continuation changes only the dimensions explicitly supplied
                        # by the current turn. The resolver's topic/fact is authoritative
                        # for this turn and replaces the stale dimension.
                        current = dict(conv.active_targets[0])
                        incoming_one = dict(incoming[0])
                        for key, value in incoming_one.items():
                            if value is not None:
                                current[key] = value
                        current_topic = getattr(resolved, "topic", None)
                        if current_topic:
                            current["topic"] = current_topic
                            current["fact"] = current_topic
                        conv.active_targets = [current]
                    else:
                        conv.active_targets = incoming
                    if len(incoming) > 1:
                        conv.comparison_set = list(incoming)
                # If the resolver deliberately returned no new target for a reference
                # turn, retain the already authoritative active set.
            elif incoming:
                conv.active_targets = incoming

            if len(conv.active_targets) == 1:
                conv.focused_target = dict(conv.active_targets[0])
                conv.focused_topic = conv.focused_target.get("topic") or getattr(resolved, "topic", None) or conv.focused_topic
            elif len(conv.active_targets) > 1:
                conv.focused_target = None
                conv.comparison_set = list(conv.active_targets)
                topics = {t.get("topic") for t in conv.active_targets if t.get("topic")}
                conv.focused_topic = next(iter(topics)) if len(topics) == 1 else (getattr(resolved, "topic", None) or conv.focused_topic)

            conv.active_documents = list(dict.fromkeys(t.get("document") for t in conv.active_targets if t.get("document")))
            conv.candidate_targets = list(conv.active_targets)
            conv.pending_clarification = None
            conv.unresolved_reference = None
            conv.last_operation = getattr(resolved, "intent", None)
            conv.last_topic = getattr(resolved, "topic", None) or conv.focused_topic

            if success and conv.active_targets:
                conv.last_successful_target_set = [dict(t) for t in conv.active_targets]
                conv.last_successful_target = dict(conv.active_targets[0]) if len(conv.active_targets) == 1 else None
                conv.last_evidence = list(evidence or [])

        if presented_items is not None:
            conv.presented_items = list(presented_items)
        if getattr(resolved, "reference", None):
            conv.recent_references.append({
                "expression": resolved.reference,
                "resolution": getattr(resolved, "reference_resolution", None),
                "state_version": conv.state_version,
            })
            conv.recent_references = conv.recent_references[-20:]
        conv.recent_operations = conv.recent_operations[-20:]
        conv.authoritative_resolutions.append(resolved.as_dict())
        conv.authoritative_resolutions = conv.authoritative_resolutions[-20:]
        conv.state_version += 1
        ctx.query_target = resolved.as_dict()
        ctx.last_resolved_document = getattr(resolved, "document", None) or ctx.last_resolved_document
        ctx.active_documents = list(conv.active_documents)
        return conv

    def mark_turn_success(self, s: str, evidence: Optional[list[dict[str, Any]]] = None) -> None:
        conv = self.get_conversation_state(s)
        if conv.active_targets:
            conv.last_successful_target_set = [dict(t) for t in conv.active_targets]
            conv.last_successful_target = dict(conv.active_targets[0]) if len(conv.active_targets) == 1 else None
            conv.last_evidence = list(evidence or [])
            conv.last_operation = conv.last_operation
            conv.last_topic = conv.focused_topic



state=AppState()
