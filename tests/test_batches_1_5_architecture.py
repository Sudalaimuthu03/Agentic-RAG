"""Pure architectural regression tests for Batches 1-5.
These tests intentionally avoid a live provider and validate authority/state contracts.
"""
from types import SimpleNamespace

from rag.core.state import AppState
from rag.execution.contracts import ExecutionPlan
from rag.semantic.context_binding import bind_reference
from rag.semantic.planning import make_plan
from rag.semantic.understanding import SemanticReference
from rag.semantic.verification import resolve_semantic_request
from rag.semantic_contracts import ContextAction

DOCS = {
    "sample_manual.pdf": "SkyBrew Coffee Maker Model SB-450 User Manual Warranty 24 months Price 99",
    "AquaFlow_AF200_Manual.pdf": "AquaFlow AF-200 Submersible Pump User Manual Warranty 12 months Safety precautions",
    "AquaFlow_AF300_Manual.pdf": "AquaFlow AF-300 Countertop Beverage Brewer Product Manual Warranty 24 months Safety precautions",
    "Priya_Rajesh_AI_Engineer.pdf": "Priya Rajesh AI Engineer Skills Python RAG",
    "candidate_resume.pdf": "Priya Nandakumar Mechanical Design Engineer Education B.E. Mechanical Engineering",
}


def semantic(**overrides):
    base = dict(
        entities=(), variants=(), topics=(), documents=(), references=(), targets=(), target_count=0,
        sufficiency="SUFFICIENT", ambiguity=(), missing_information=(), context_dependency="NONE",
        knowledge_mode="APPLICATION", context_action="SWITCH", operation="FACTUAL_QUERY",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_corpus_alias_resolution_is_generic():
    for entity, expected in (("SkyBrew", "sample_manual.pdf"), ("SB450", "sample_manual.pdf")):
        resolved = resolve_semantic_request(
            semantic(entities=(entity,), topics=("warranty",)), turn_id=entity, raw_query=entity,
            conversation={}, documents=DOCS,
        )
        assert resolved.knowledge_status == "AUTHORIZED"
        assert resolved.document == expected


def test_unknown_identity_stays_unknown():
    resolved = resolve_semantic_request(
        semantic(entities=("Vivo Y19",), topics=("warranty",)), turn_id="u", raw_query="Vivo Y19",
        conversation={}, documents=DOCS,
    )
    assert resolved.knowledge_status == "UNKNOWN_IDENTITY"
    assert resolved.document is None


def test_collective_reference_preserves_all_targets():
    conversation = {
        "active_targets": [
            {"entity": "AquaFlow", "variant": "AF200", "document": "AquaFlow_AF200_Manual.pdf", "topic": "warranty"},
            {"entity": "AquaFlow", "variant": "AF300", "document": "AquaFlow_AF300_Manual.pdf", "topic": "warranty"},
        ],
        "focused_target": None, "presented_items": [], "authoritative_resolutions": [],
    }
    ref = SemanticReference("PRONOMINAL_REFERENCE", "them")
    binding = bind_reference(SimpleNamespace(references=(ref,)), conversation)
    assert binding["status"] == "RESOLVED"
    assert len(binding["resolved_targets"]) == 2

    resolved = resolve_semantic_request(
        semantic(topics=("safety precautions",), references=(ref,), context_dependency="REQUIRED", context_action="CONTINUE"),
        turn_id="c", raw_query="safety precautions for them", conversation=conversation, documents=DOCS,
    )
    assert len(resolved.targets) == 2
    assert {t["document"] for t in resolved.targets} == {
        "AquaFlow_AF200_Manual.pdf", "AquaFlow_AF300_Manual.pdf"
    }


def test_continuation_updates_topic_without_creating_stale_target():
    state = AppState()
    mk = lambda action, targets, topic: SimpleNamespace(
        context_action=action, targets=targets, topic=topic, reference=None,
        reference_resolution=None, as_dict=lambda: {"targets": list(targets), "topic": topic},
    )
    state.update_conversation_state(
        "s", mk(ContextAction.SWITCH, ({"entity": "Priya Nandakumar", "document": "candidate_resume.pdf", "topic": "education"},), "education")
    )
    state.update_conversation_state(
        "s", mk(ContextAction.CONTINUE, ({"entity": "Priya Nandakumar", "document": "candidate_resume.pdf", "topic": "certifications"},), "certifications")
    )
    conversation = state.get_conversation_state("s")
    assert len(conversation.active_targets) == 1
    assert conversation.focused_target["topic"] == "certifications"


def test_execution_plan_is_authoritative():
    plan = ExecutionPlan("factual_lookup", targets=({"entity": "A"},), tools=("search_documents",))
    assert plan.authorized_tools == ("search_documents",)
    assert plan.execution_mode == "DOCUMENT_GROUNDED"
    assert "list_available_documents" not in plan.authorized_tools


def test_unknown_request_plan_has_no_document_tools():
    resolved = resolve_semantic_request(
        semantic(entities=("Unknown Product",), topics=("warranty",)), turn_id="u2", raw_query="Unknown Product warranty",
        conversation={}, documents=DOCS,
    )
    plan = make_plan(resolved)
    assert plan.tools == ()
    assert plan.execution_mode == "REFUSAL"
