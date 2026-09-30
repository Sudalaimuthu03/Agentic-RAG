from types import SimpleNamespace

from rag.semantic.planning import make_plan
from rag.semantic.verification import resolve_semantic_request

DOCS = {
    "sample_manual.pdf": "SkyBrew Coffee Maker Model SB-450 User Manual Warranty Period 24 months",
    "AquaFlow_AF200_Manual.pdf": "AquaFlow AF-200 Submersible Pump User Manual Warranty coverage period 12 months",
    "AquaFlow_AF300_Manual.pdf": "AquaFlow AF-300 Countertop Beverage Brewer Product Manual 24-month limited warranty",
    "Priya_Rajesh_AI_Engineer.pdf": "Priya Rajesh AI Engineer Professional Profile certifications Python RAG",
    "candidate_resume.pdf": "Priya Nandakumar Mechanical Design Engineer Certifications CSWP Six Sigma Green Belt",
    "hospital_intake_procedures.pdf": "Patient Admission and Discharge Procedures Visiting Hours Complaint Acknowledgement",
    "policy_doc.pdf": "Employee Handbook Leave Remote Work Notice Period Insurance Coverage",
}


def semantic(**overrides):
    base = dict(
        entities=(), variants=(), topics=(), documents=(), references=(), targets=(), target_count=0,
        sufficiency="SUFFICIENT", ambiguity=(), missing_information=(), context_dependency="NONE",
        knowledge_mode="APPLICATION", context_action="SWITCH", operation="FACTUAL_QUERY",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_priya_discovery_is_bounded_to_two_candidate_documents():
    resolved = resolve_semantic_request(
        semantic(entities=("Priya",), sufficiency="AMBIGUOUS", ambiguity=("which Priya",)),
        turn_id="1", raw_query="tell me about Priya", conversation={}, documents=DOCS,
    )
    assert resolved.knowledge_status == "CLARIFY"
    assert set(resolved.candidate_documents) == {
        "candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"
    }
    assert set(resolved.candidate_entities) == {"Priya Nandakumar", "Priya Rajesh"}
    assert len(resolved.candidate_documents) < len(DOCS)


def test_priya_certification_discovery_does_not_expand_to_unrelated_corpus():
    resolved = resolve_semantic_request(
        semantic(entities=("Priya",), topics=("certifications",), sufficiency="AMBIGUOUS", ambiguity=("which Priya",)),
        turn_id="2", raw_query="priya has certifications?", conversation={}, documents=DOCS,
    )
    assert set(resolved.candidate_documents) == {
        "candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"
    }


def test_unknown_entity_does_not_inherit_topic_matches_as_candidates():
    resolved = resolve_semantic_request(
        semantic(entities=("John Smith",), topics=("certifications",)),
        turn_id="3", raw_query="John Smith certifications", conversation={}, documents=DOCS,
    )
    assert resolved.knowledge_status == "UNKNOWN_IDENTITY"
    assert resolved.candidate_documents == ()
    assert resolved.evidence_scope == ()


def test_clarification_plan_preserves_candidate_search_scope_without_evidence_authority():
    resolved = resolve_semantic_request(
        semantic(entities=("Priya",), sufficiency="AMBIGUOUS", ambiguity=("which Priya",)),
        turn_id="4", raw_query="tell me about Priya", conversation={}, documents=DOCS,
    )
    plan = make_plan(resolved)
    assert plan.execution_mode == "CLARIFICATION"
    assert plan.tools == ("ask_clarification",)
    assert set(plan.search_scope) == set(resolved.candidate_documents)
    assert plan.document_scope == ()
    assert plan.evidence_requirements == ()
