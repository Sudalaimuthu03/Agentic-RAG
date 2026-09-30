from __future__ import annotations
from rag.execution.contracts import ExecutionPlan

TOOLS = ("search_documents", "list_available_documents", "summarize_document", "ask_clarification")

def _target_requirements(resolved):
    out=[]
    for idx,t in enumerate(resolved.targets,1):
        out.append({
            "target_id": str(idx),
            "entity": t.get("entity"), "variant": t.get("variant"),
            "topic": t.get("topic"), "document": t.get("document"),
            "required": True,
        })
    return tuple(out)

def make_plan(resolved) -> ExecutionPlan:
    targets=tuple(dict(t) for t in (resolved.targets or ()))
    reqs=_target_requirements(resolved)
    # UNKNOWN_IDENTITY is a true closed-world no-match only when there are no bounded
    # candidates. If the resolver discovered plausible indexed candidates, retrieval is
    # still authorized to determine whether the corpus actually answers the question.
    if resolved.knowledge_status == "UNSUPPORTED":
        mode = "REFUSAL"
        return ExecutionPlan(resolved.intent, targets, (), tuple(resolved.evidence_scope), tuple(resolved.evidence_scope), (), False, ("closed_world_stop",), mode, reqs)
    if resolved.knowledge_status == "UNKNOWN_IDENTITY" and not resolved.candidate_documents:
        mode = "REFUSAL"
        return ExecutionPlan(resolved.intent, targets, (), (), (), (), False, ("closed_world_stop",), mode, reqs)
    # Clarify only when the target is genuinely ambiguous/unresolved. A plausible
    # corpus candidate with an otherwise executable request should be searched before
    # refusing; this is essential for generic corpus topics and procedures.
    should_clarify = (
        resolved.context_action.value == "CLARIFY"
        or resolved.knowledge_status == "CLARIFY"
        or resolved.sufficiency == "AMBIGUOUS"
        or (resolved.sufficiency == "CONTEXT_DEPENDENT" and resolved.context_dependency not in {"RESOLVED", "OPTIONAL"})
    )
    if should_clarify:
        # Candidate documents are discovery scope only. They are deliberately not
        # promoted to evidence scope until the user resolves the ambiguity.
        candidate_scope = tuple(dict.fromkeys(resolved.candidate_documents))
        return ExecutionPlan(resolved.intent, targets, ("ask_clarification",), candidate_scope, (), (), True, ("clarification_required",), "CLARIFICATION", reqs)
    if resolved.knowledge_status == "INSUFFICIENT" and (resolved.candidate_documents or resolved.evidence_scope or targets):
        scope = tuple(dict.fromkeys(resolved.evidence_scope or resolved.candidate_documents))
        return ExecutionPlan(resolved.intent, targets, ("search_documents",), scope, scope, ("entity_alignment", "variant_alignment", "topic_alignment", "fact_alignment", "all_targets_satisfied"), False, (), "DOCUMENT_GROUNDED", reqs)
    if resolved.intent == "smalltalk":
        return ExecutionPlan("SMALLTALK", targets, (), (), (), (), False, (), "CONVERSATIONAL", reqs)
    if resolved.intent == "help":
        return ExecutionPlan("HELP", targets, (), (), (), (), False, (), "CONVERSATIONAL", reqs)
    if resolved.intent == "list_documents":
        return ExecutionPlan("LIST_DOCUMENTS", targets, ("list_available_documents",), (), (), (), False, (), "CONVERSATIONAL", reqs)
    if resolved.intent == "summarization":
        return ExecutionPlan("SUMMARY", targets, ("summarize_document",), tuple(resolved.evidence_scope), tuple(resolved.evidence_scope), ("source_scoped",), False, (), "DOCUMENT_GROUNDED", reqs)
    return ExecutionPlan(resolved.intent, targets, ("search_documents",), tuple(resolved.evidence_scope), tuple(resolved.evidence_scope), ("entity_alignment", "variant_alignment", "topic_alignment", "fact_alignment", "all_targets_satisfied"), False, (), "DOCUMENT_GROUNDED", reqs)
