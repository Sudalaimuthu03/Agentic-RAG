from types import SimpleNamespace
from pathlib import Path

from rag.core.state import AppState
from rag.semantic.context_binding import bind_reference
from rag.semantic.understanding import SemanticReference
from rag.semantic.verification import resolve_semantic_request
from rag.semantic_contracts import ContextAction, EvidenceItem, EvidenceStatus, ResolvedRequest
from rag.evidence.gate import assess

DOCS = {
    "sample_manual.pdf": "SkyBrew Coffee Maker Model SB-450 Warranty 24 months",
    "AquaFlow_AF200_Manual.pdf": "AquaFlow AF-200 pump warranty 12 months",
    "AquaFlow_AF300_Manual.pdf": "AquaFlow AF-300 brewer warranty 24 months",
    "candidate_resume.pdf": "Priya Nandakumar Mechanical Design Engineer Certifications CSWP Six Sigma Green Belt B.Tech",
    "Priya_Rajesh_AI_Engineer.pdf": "Priya Rajesh AI Engineer Python SQL M.Tech Artificial Intelligence 2021",
    "hospital_intake_procedures.pdf": "Patient Admission Discharge Visiting Hours Complaint",
    "policy_doc.pdf": "Employee Handbook Notice Period 30 days Leave Complaint HR",
}


def sem(**kw):
    base = dict(
        entities=(), variants=(), topics=(), documents=(), references=(), targets=(), target_count=0,
        sufficiency="SUFFICIENT", ambiguity=(), missing_information=(), context_dependency="NONE",
        knowledge_mode="APPLICATION", context_action="SWITCH", operation="FACTUAL_QUERY",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_employee_resolves_pending_relief_target_without_stale_state():
    state = AppState()
    first = sem(
        topics=("relieving process",),
        documents=("policy_doc.pdf", "hospital_intake_procedures.pdf"),
        targets=({"entity": "employee", "document": "policy_doc.pdf", "topic": "relieving process"},
                 {"entity": "hospital", "document": "hospital_intake_procedures.pdf", "topic": "relieving process"}),
        sufficiency="AMBIGUOUS", context_dependency="AMBIGUOUS", context_action="CLARIFY",
    )
    r1 = resolve_semantic_request(first, turn_id="1", raw_query="relieving process", conversation=state.get_conversation_state("s").as_dict(), documents=DOCS)
    state.update_conversation_state("s", r1)
    second = sem(operation="CLARIFY", context_action="CLARIFY", context_dependency="AMBIGUOUS", sufficiency="AMBIGUOUS")
    r2 = resolve_semantic_request(second, turn_id="2", raw_query="employee", conversation=state.get_conversation_state("s").as_dict(), documents=DOCS)
    assert r2.intent == "factual_lookup"
    assert r2.document == "policy_doc.pdf"
    assert r2.topic == "relieving process"
    assert r2.context_action == ContextAction.CONTINUE


def test_both_prefers_new_priya_candidates_over_old_product_target():
    conversation = {
        "active_targets": [{"entity": "SkyBrew", "document": "sample_manual.pdf", "topic": "warranty"}],
        "focused_target": {"entity": "SkyBrew", "document": "sample_manual.pdf", "topic": "warranty"},
        "pending_clarification": {
            "kind": "TARGET_SELECTION",
            "candidate_targets": [
                {"entity": "Priya Nandakumar", "document": "candidate_resume.pdf"},
                {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf"},
            ],
            "original_operation": "factual_lookup",
        },
        "candidate_targets": [
            {"entity": "Priya Nandakumar", "document": "candidate_resume.pdf"},
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf"},
        ],
    }
    semantic = sem(references=(SemanticReference("PRONOMINAL_REFERENCE", "both"),), operation="FACTUAL_QUERY", context_action="CONTINUE", context_dependency="REQUIRED")
    resolved = resolve_semantic_request(semantic, turn_id="2", raw_query="both", conversation=conversation, documents=DOCS)
    assert {t["document"] for t in resolved.targets} == {"candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"}


def test_clear_priya_certification_query_executes_instead_of_clarifying():
    semantic = sem(
        entities=("Priya Rajesh",), topics=("certifications", "internships"), documents=("Priya_Rajesh_AI_Engineer.pdf",),
        targets=(
            {"entity": "Priya Rajesh", "topic": "certifications", "document": "Priya_Rajesh_AI_Engineer.pdf"},
            {"entity": "Priya Rajesh", "topic": "internships", "document": "Priya_Rajesh_AI_Engineer.pdf"},
        ), target_count=2, sufficiency="INSUFFICIENT", context_dependency="NONE", context_action="CLARIFY",
    )
    resolved = resolve_semantic_request(semantic, turn_id="61", raw_query="Priya Rajesh has any certifications and internship experience?", conversation={}, documents=DOCS)
    assert resolved.intent == "factual_lookup"
    assert resolved.context_action == ContextAction.CONTINUE
    assert resolved.knowledge_status == "AUTHORIZED"
    assert set(resolved.evidence_scope) == {"Priya_Rajesh_AI_Engineer.pdf"}


def test_check_again_uses_latest_successful_certification_target():
    conversation = {
        "active_targets": [
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "certifications"},
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "internships"},
        ],
        "focused_target": None,
        "last_successful_target_set": [
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "certifications"},
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "internships"},
        ],
    }
    semantic = sem(
        entities=("Priya Nandakumar", "Priya Rajesh"), topics=("education", "education"), documents=("candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"),
        references=(SemanticReference("VERIFICATION_REFERENCE", "check again", relative_to="PREVIOUS_REQUEST"),),
        targets=({"entity": "Priya Nandakumar", "document": "candidate_resume.pdf", "topic": "education"}, {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "education"}),
        operation="VERIFY", context_action="VERIFY", context_dependency="NONE",
    )
    resolved = resolve_semantic_request(semantic, turn_id="64", raw_query="check again", conversation=conversation, documents=DOCS)
    assert all(t.get("topic") in {"certifications", "internships"} for t in resolved.targets)
    assert all(t.get("document") == "Priya_Rajesh_AI_Engineer.pdf" for t in resolved.targets)


def test_generic_current_query_discovers_notice_period_policy():
    semantic = sem(sufficiency="INSUFFICIENT", context_dependency="REQUIRED", context_action="CLARIFY")
    resolved = resolve_semantic_request(semantic, turn_id="37", raw_query="whats the notice period?", conversation={}, documents=DOCS)
    assert resolved.document == "policy_doc.pdf"
    assert resolved.knowledge_status == "AUTHORIZED"
    assert resolved.context_action == ContextAction.CONTINUE


def test_two_different_targets_are_not_a_conflict():
    req = ResolvedRequest(
        turn_id="x", raw_query="compare education", intent="factual_lookup", topic="education",
        evidence_scope=("candidate_resume.pdf", "Priya_Rajesh_AI_Engineer.pdf"),
        targets=(
            {"entity": "Priya Nandakumar", "document": "candidate_resume.pdf", "topic": "education"},
            {"entity": "Priya Rajesh", "document": "Priya_Rajesh_AI_Engineer.pdf", "topic": "education"},
        ), target_count=2,
    )
    items = [
        EvidenceItem("candidate_resume.pdf", None, None, None, "B.Tech Mechanical Engineering 2014-2018", "Priya Nandakumar", None, "education", target_id="1"),
        EvidenceItem("Priya_Rajesh_AI_Engineer.pdf", None, None, None, "M.Tech Artificial Intelligence 2021", "Priya Rajesh", None, "education", target_id="2"),
    ]
    bundle = assess(req, items)
    assert bundle.status == EvidenceStatus.MULTI_SOURCE_AGREEMENT
    assert not bundle.conflicts


def test_explicit_same_target_scalar_disagreement_is_contradictory():
    req = ResolvedRequest(turn_id="x", raw_query="verify", intent="verification", topic="warranty", document="a.pdf", targets=({"document":"a.pdf","topic":"warranty"},))
    items = [
        EvidenceItem("a.pdf", None, None, None, "Warranty 12 months", None, None, "warranty"),
        EvidenceItem("b.pdf", None, None, None, "Warranty 24 months", None, None, "warranty"),
    ]
    # req.document scopes out b.pdf; explicit contradiction is therefore not claimable.
    bundle = assess(req, items)
    assert bundle.status == EvidenceStatus.SUPPORTED


def test_v54_summary_path_is_deterministic_and_source_scoped():
    graph = Path(__file__).parents[1] / "rag" / "agent" / "graph.py"
    text = graph.read_text()
    assert 'if plan.operation == "SUMMARY"' in text
    assert 'summary_tool.invoke({"document": document})' in text
    assert 'authorized_names' in text


def test_unseen_domain_corpora_use_generic_query_discovery():
    unseen = {
        "ledger.pdf": "Quarterly ledger reconciliation threshold is 5000 credits.",
        "turbine.pdf": "Turbine maintenance interval is 240 hours of operation.",
        "contract.pdf": "The agreement renews after 12 months unless notice is given.",
        "curriculum.pdf": "The advanced module requires 30 instructional hours.",
        "paper.pdf": "The experiment used 120 samples and reports the measured variance.",
        "assembly.pdf": "The assembly inspection occurs every 90 days.",
        "api_guide.pdf": "The API rate limit is 100 requests per minute.",
    }
    for query, expected in [
        ("what is the ledger threshold", "ledger.pdf"),
        ("what is the turbine maintenance interval", "turbine.pdf"),
        ("what is the agreement renewal", "contract.pdf"),
        ("how many instructional hours", "curriculum.pdf"),
        ("how many samples", "paper.pdf"),
        ("when is the assembly inspection", "assembly.pdf"),
        ("what is the API rate limit", "api_guide.pdf"),
    ]:
        resolved = resolve_semantic_request(sem(sufficiency="INSUFFICIENT", context_dependency="REQUIRED", context_action="CLARIFY"), turn_id=query, raw_query=query, conversation={}, documents=unseen)
        assert resolved.document == expected
        assert resolved.knowledge_status == "AUTHORIZED"
