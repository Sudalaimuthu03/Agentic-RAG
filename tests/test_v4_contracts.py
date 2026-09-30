from rag.execution.gate import ExecutionBudget
from rag.resolution import resolve_request
from rag.semantic_contracts import ContextAction, EvidenceStatus, EvidenceItem
from rag.evidence.gate import assess
from rag.validation.answer import validate_answer

DOCS={
'AquaFlow_AF200_Manual.pdf':'AquaFlow AF-200 Submersible Pump User Manual Warranty Information',
'AquaFlow_AF300_Manual.pdf':'AquaFlow AF-300 Countertop Beverage Brewer Product Manual Safety Warranty',
'sample_manual.pdf':'SkyBrew Coffee Maker Model SB-450 User Manual Warranty',
'Priya_Rajesh_AI_Engineer.pdf':'Priya Rajesh AI Engineer Skills Python RAG',
'candidate_resume.pdf':'Priya Nandakumar Mechanical Design Engineer Skills CAD',
'policy_doc.pdf':'Employee Handbook Notice Period Complaint Acknowledgement Leave',
'hospital_intake_procedures.pdf':'Patient Admission and Discharge Procedures Visiting Hours Complaint Acknowledgement',
}

def test_ambiguous_priya():
    r=resolve_request('Tell me about Priya','1',list(DOCS),documents=DOCS)
    assert r.context_action == ContextAction.CLARIFY
    assert set(r.ambiguity)=={'Priya_Rajesh_AI_Engineer.pdf','candidate_resume.pdf'}

def test_exact_priya_switch():
    r=resolve_request('Priya Nandakumar skills','1',list(DOCS),documents=DOCS)
    assert r.document=='candidate_resume.pdf' and r.entity=='Priya Nandakumar'

def test_variant_wins_over_parent_ambiguity():
    r=resolve_request('AquaFlow AF-300 warranty','1',list(DOCS),documents=DOCS)
    assert r.document=='AquaFlow_AF300_Manual.pdf' and r.variant=='AF-300' and r.entity=='AquaFlow'

def test_parent_only_is_ambiguous():
    r=resolve_request('AquaFlow warranty','1',list(DOCS),documents=DOCS)
    assert r.context_action == ContextAction.CLARIFY

def test_notice_period_resolves_document_without_inventing_entity():
    r=resolve_request('What is the notice period?','1',list(DOCS),documents=DOCS)
    assert r.document=='policy_doc.pdf' and r.entity is None

def test_budget_limits():
    b=ExecutionBudget(3,3,2,1)
    for _ in range(3): b.consume_iteration()
    assert not b.allow_iteration()
    for _ in range(3): b.consume_llm()
    assert not b.allow_llm()
    for _ in range(2): b.consume_retrieval()
    assert not b.allow_retrieval()

def test_supported_evidence_alignment():
    r=resolve_request('AquaFlow AF-300 warranty','1',list(DOCS),documents=DOCS)
    item=EvidenceItem('AquaFlow_AF300_Manual.pdf',1,'c','p','AquaFlow AF-300 warranty is 24 months','AquaFlow','AF-300','warranty')
    b=assess(r,[item])
    assert b.status==EvidenceStatus.KNOWN
    assert validate_answer('The warranty is 24 months.',r,b)[0]

def test_wrong_variant_is_rejected():
    r=resolve_request('AquaFlow AF-300 warranty','1',list(DOCS),documents=DOCS)
    item=EvidenceItem('AquaFlow_AF300_Manual.pdf',1,'c','p','AquaFlow AF-300 warranty is 24 months','AquaFlow','AF-300','warranty')
    b=assess(r,[item])
    ok,problems=validate_answer('The AF-200 warranty is 12 months.',r,b)
    assert not ok and 'variant_mismatch' in problems


def test_routes_imports_validate_final():
    from pathlib import Path
    text = Path("rag/api/routes.py").read_text(encoding="utf-8")
    assert "from rag.agent.graph import run_agent_turn, validate_final" in text
