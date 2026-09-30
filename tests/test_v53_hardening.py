from types import SimpleNamespace

from rag.core.state import AppState
from rag.execution.contracts import ExecutionPlan
from rag.semantic.context_binding import bind_reference
from rag.semantic.planning import make_plan
from rag.semantic.understanding import SemanticReference
from rag.semantic.verification import resolve_semantic_request
from rag.semantic_contracts import ContextAction

DOCS = {
    "sample_manual.pdf": "SkyBrew Coffee Maker Model SB-450 User Manual Warranty Period 24 months",
    "Priya_Rajesh_AI_Engineer.pdf": "Priya Rajesh AI Engineer Professional Profile Skills Python RAG",
    "candidate_resume.pdf": "Priya Nandakumar Mechanical Design Engineer Certifications CSWP Six Sigma Green Belt",
    "hospital_intake_procedures.pdf": "Patient Admission and Discharge Procedures Visiting Hours Complaint Acknowledgement Investigation Resolution",
    "policy_doc.pdf": "Employee Handbook Complaint Acknowledgement Notice Period Insurance Coverage",
}


def sem(**kw):
    base = dict(
        entities=(), variants=(), topics=(), documents=(), references=(), targets=(), target_count=0,
        sufficiency="SUFFICIENT", ambiguity=(), missing_information=(), context_dependency="NONE",
        knowledge_mode="APPLICATION", context_action="SWITCH", operation="FACTUAL_QUERY",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_check_again_binds_immediately_active_target_not_old_history():
    conversation = {
        "active_targets": [{"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "complaints"}],
        "focused_target": {"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "complaints"},
        "last_successful_target_set": [{"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "complaints"}],
        "presented_items": [],
        "authoritative_resolutions": [],
    }
    semantic = sem(
        references=(SemanticReference("VERIFICATION_REFERENCE", "check again", relative_to="PREVIOUS_REQUEST"),),
        operation="VERIFY", sufficiency="CONTEXT_DEPENDENT", context_dependency="REQUIRED", context_action="VERIFY",
    )
    resolved = resolve_semantic_request(semantic, turn_id="84", raw_query="check again", conversation=conversation, documents=DOCS)
    plan = make_plan(resolved)
    assert resolved.document == "hospital_intake_procedures.pdf"
    assert resolved.topic == "complaints"
    assert resolved.context_dependency == "RESOLVED"
    assert plan.tools == ("search_documents",)
    assert plan.execution_mode == "DOCUMENT_GROUNDED"


def test_pending_clarification_preserves_target_and_for_hospital_selects_hospital():
    conversation = {
        "active_targets": [],
        "focused_target": None,
        "presented_items": [],
        "last_successful_target_set": [],
        "pending_clarification": {
            "candidate_targets": [
                {"entity": "employee", "document": "policy_doc.pdf", "topic": "relieving procedure"},
                {"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "relieving procedure"},
            ]
        },
    }
    semantic = sem(entities=("hospital",), sufficiency="CONTEXT_DEPENDENT", context_dependency="REQUIRED", context_action="CONTINUE")
    resolved = resolve_semantic_request(semantic, turn_id="40", raw_query="for hospital", conversation=conversation, documents=DOCS)
    plan = make_plan(resolved)
    assert resolved.document == "hospital_intake_procedures.pdf"
    assert resolved.topic == "relieving procedure"
    assert resolved.context_dependency == "RESOLVED"
    assert plan.tools == ("search_documents",)


def test_both_prefers_pending_clarification_candidates():
    conversation = {
        "active_targets": [{"entity": "SkyBrew", "document": "sample_manual.pdf", "topic": "warranty"}],
        "focused_target": {"entity": "SkyBrew", "document": "sample_manual.pdf", "topic": "warranty"},
        "presented_items": [],
        "pending_clarification": {
            "candidate_targets": [
                {"entity": "Priya Nandakumar", "document": "candidate_resume.pdf"},
                {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf"},
            ]
        },
    }
    ref = SemanticReference("PRONOMINAL_REFERENCE", "both")
    result = bind_reference(SimpleNamespace(references=(ref,)), conversation)
    assert result["status"] == "RESOLVED"
    assert {x["document"] for x in result["resolved_targets"]} == {"candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"}


def test_topic_only_candidate_discovery_is_retrieval_eligible_before_refusal():
    semantic = sem(entities=("hospital",), topics=("complaints",), sufficiency="INSUFFICIENT", context_dependency="RESOLVED", context_action="CONTINUE")
    conversation = {
        "active_targets": [{"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "complaints"}],
        "focused_target": {"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "complaints"},
        "presented_items": [],
    }
    resolved = resolve_semantic_request(semantic, turn_id="81", raw_query="how are complaints acknowledged?", conversation=conversation, documents=DOCS)
    # Current active hospital target supplies the missing identity context.
    assert resolved.document == "hospital_intake_procedures.pdf"
    plan = make_plan(resolved)
    assert plan.execution_mode == "DOCUMENT_GROUNDED"
    assert plan.tools == ("search_documents",)


def test_clarification_does_not_destroy_previous_successful_state():
    state = AppState()
    first = SimpleNamespace(
        turn_id="1", raw_query="SkyBrew warranty", intent="factual_lookup", entity="SkyBrew", variant="SB-450",
        topic="warranty", document="sample_manual.pdf", evidence_scope=("sample_manual.pdf",),
        context_action=ContextAction.SWITCH, targets=({"entity":"SkyBrew","document":"sample_manual.pdf","topic":"warranty"},),
        candidate_targets=(), candidate_documents=(), candidate_entities=(), reference=None, reference_resolution=None,
        last_successful_target_set=(), as_dict=lambda: {"turn_id":"1"}
    )
    state.update_conversation_state("s", first, success=True, evidence=[{"source":"sample_manual.pdf"}])
    clarify = SimpleNamespace(
        turn_id="2", raw_query="tell me about Priya", intent="factual_lookup", entity="Priya", variant=None,
        topic=None, document=None, evidence_scope=(), context_action=ContextAction.CLARIFY, targets=({"entity":"Priya"},),
        candidate_targets=(), candidate_documents=("candidate_resume.pdf","Priya_Rajesh_AI_Engineer.pdf"),
        candidate_entities=("Priya Nandakumar","Priya Rajesh"), reference=None, reference_resolution=None,
        missing_information=("which Priya",), as_dict=lambda: {"turn_id":"2"}
    )
    conv = state.update_conversation_state("s", clarify)
    assert conv.last_successful_target_set[0]["document"] == "sample_manual.pdf"
    assert conv.pending_clarification is not None
    assert {x["document"] for x in conv.candidate_targets} == {"candidate_resume.pdf","Priya_Rajesh_AI_Engineer.pdf"}
