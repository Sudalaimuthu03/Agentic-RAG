from dataclasses import dataclass

from rag.semantic.understanding import understand
from rag.semantic.verification import resolve_semantic_request
from rag.semantic.planning import make_plan
from rag.semantic.context_binding import bind_reference


DOCS = {
    "sample_manual.pdf": "SkyBrew Coffee Maker Model SB-450 User Manual Warranty 24 months",
    "AquaFlow_AF200_Manual.pdf": "AquaFlow AF-200 Submersible Pump User Manual Warranty 12 months",
    "AquaFlow_AF300_Manual.pdf": "AquaFlow AF-300 Countertop Beverage Brewer Product Manual Warranty 24 months",
    "Priya_Rajesh_AI_Engineer.pdf": "Priya Rajesh AI Engineer Skills Python RAG",
    "candidate_resume.pdf": "Priya Nandakumar Mechanical Design Engineer Skills CAD Experience",
}


@dataclass
class Msg:
    content: str


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
    def invoke(self, messages):
        import json
        return Msg(json.dumps(self.payload))


def payload(**kw):
    base = {
        "interaction": "DOCUMENT", "operation": "FACTUAL_QUERY", "entities": [], "variants": [],
        "topics": ["warranty"], "documents": [], "references": [], "targets": [], "target_count": 1,
        "sufficiency": "SUFFICIENT", "ambiguity": [], "missing_information": [],
        "context_dependency": "NONE", "knowledge_mode": "APPLICATION", "context_action": "SWITCH", "uncertainty": "LOW",
    }
    base.update(kw)
    return base


def test_unknown_identity_is_not_substituted():
    s = understand(FakeLLM(payload(entities=["Vivo Y19"])), "What is the warranty for Vivo Y19?", catalog=[], conversation={})
    r = resolve_semantic_request(s, turn_id="1", raw_query="What is the warranty for Vivo Y19?", conversation={}, documents=DOCS)
    assert r.knowledge_status == "UNKNOWN_IDENTITY"
    assert r.document is None


def test_exact_variant_disambiguates_aquaflow():
    s = understand(FakeLLM(payload(entities=["AquaFlow"], variants=["AF-300"])), "AquaFlow AF-300 warranty", catalog=[], conversation={})
    r = resolve_semantic_request(s, turn_id="1", raw_query="AquaFlow AF-300 warranty", conversation={}, documents=DOCS)
    assert r.knowledge_status == "AUTHORIZED"
    assert r.document == "AquaFlow_AF300_Manual.pdf"


def test_ambiguous_parent_requires_clarification():
    s = understand(FakeLLM(payload(entities=["AquaFlow"])), "AquaFlow warranty", catalog=[], conversation={})
    r = resolve_semantic_request(s, turn_id="1", raw_query="AquaFlow warranty", conversation={}, documents=DOCS)
    assert r.knowledge_status == "CLARIFY"
    assert r.context_action.value == "CLARIFY"


def test_reference_preserves_entity_and_changes_topic():
    conv = {"focused_target": {"entity": "Priya Nandakumar", "document": "candidate_resume.pdf", "topic": "skills"}, "active_targets": [], "presented_items": []}
    s = understand(FakeLLM(payload(entities=[], topics=["experience"], references=[{"reference_type":"PRONOMINAL_REFERENCE","expression":"her","target_dimension":"TARGET","requires_context":True}], sufficiency="CONTEXT_DEPENDENT", context_dependency="REQUIRED", context_action="CONTINUE")), "What about her experience?", catalog=[], conversation=conv)
    r = resolve_semantic_request(s, turn_id="2", raw_query="What about her experience?", conversation={**conv, "current_query": "What about her experience?"}, documents=DOCS)
    assert r.entity == "Priya Nandakumar"
    assert r.topic == "experience"
    assert r.document == "candidate_resume.pdf"


def test_numbered_reference_uses_structured_presented_item():
    from rag.semantic.understanding import SemanticReference
    conv = {"focused_target": {}, "active_targets": [], "presented_items": [{"position": 1, "document": "sample_manual.pdf"}, {"position": 2, "document": "AquaFlow_AF300_Manual.pdf"}]}
    ref = SemanticReference(reference_type="ORDINAL_SELECTION", expression="second one", selection_position=2)
    result = bind_reference(type("S", (), {"references": (ref,)})(), conv)
    assert result["status"] == "RESOLVED"
    assert result["resolved"]["document"] == "AquaFlow_AF300_Manual.pdf"


def test_execution_plan_does_not_allow_search_for_unknown():
    s = understand(FakeLLM(payload(entities=["Vivo Y19"])), "What is the warranty for Vivo Y19?", catalog=[], conversation={})
    r = resolve_semantic_request(s, turn_id="1", raw_query="What is the warranty for Vivo Y19?", conversation={}, documents=DOCS)
    p = make_plan(r)
    assert not p.clarification_required
    assert p.tools == ()


def test_structured_ordinal_reference_binds_without_phrase_parsing():
    from rag.semantic.understanding import SemanticReference
    conv = {"focused_target": {}, "active_targets": [], "presented_items": [
        {"position": 1, "document": "sample_manual.pdf"},
        {"position": 2, "document": "AquaFlow_AF300_Manual.pdf"},
    ]}
    ref = SemanticReference(reference_type="ORDINAL_SELECTION", expression="the second one", selection_position=2)
    result = bind_reference(type("S", (), {"references": (ref,)})(), conv)
    assert result["status"] == "RESOLVED"
    assert result["resolved"]["document"] == "AquaFlow_AF300_Manual.pdf"


def test_reference_binder_does_not_interpret_raw_phrase():
    from rag.semantic.understanding import SemanticReference
    conv = {"focused_target": {"document": "sample_manual.pdf"}, "active_targets": [], "presented_items": []}
    ref = SemanticReference(reference_type="CONTEXT_REFERENCE", expression="some arbitrary words")
    result = bind_reference(type("S", (), {"references": (ref,)})(), conv)
    assert result["status"] == "RESOLVED"
    assert result["resolved"]["document"] == "sample_manual.pdf"


def test_semantic_multiple_entities_are_preserved():
    s = understand(FakeLLM(payload(entities=["AquaFlow", "SkyBrew"], topics=["warranty"], target_count=2,
        targets=[{"entity":"AquaFlow","variant":None,"topic":"warranty","document":None,"fact":"warranty"},
                 {"entity":"SkyBrew","variant":None,"topic":"warranty","document":None,"fact":"warranty"}])),
        "warranty for AquaFlow and SkyBrew", catalog=[], conversation={})
    assert s.entities == ("AquaFlow", "SkyBrew")
    assert len(s.targets) == 2
