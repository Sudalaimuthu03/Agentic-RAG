from rag.core.state import AppState
from rag.execution.contracts import ExecutionPlan
from rag.semantic_contracts import ContextAction, ResolvedRequest
from rag.semantic.planning import make_plan


def rr(**kw):
    base = dict(turn_id="1", raw_query="q", intent="factual_lookup", entity="A", variant=None, topic="warranty", document="a.pdf", evidence_scope=("a.pdf",), context_action=ContextAction.SWITCH, targets=({"entity":"A","document":"a.pdf","topic":"warranty"},), target_count=1, sufficiency="SUFFICIENT", knowledge_status="AUTHORIZED", knowledge_mode="APPLICATION")
    base.update(kw)
    return ResolvedRequest(**base)


def test_state_switch_then_topic_continuation():
    state = AppState()
    first = rr()
    state.update_conversation_state("s", first)
    second = rr(turn_id="2", raw_query="what about price", topic="price", context_action=ContextAction.CONTINUE)
    state.update_conversation_state("s", second)
    c = state.get_conversation_state("s")
    assert c.focused_target["entity"] == "A"
    assert c.focused_target["topic"] == "price"
    assert c.active_documents == ["a.pdf"]


def test_reset_isolation_clears_old_targets():
    state = AppState()
    state.update_conversation_state("s", rr())
    new = rr(turn_id="2", raw_query="hello", intent="smalltalk", entity=None, topic=None, document=None, evidence_scope=(), targets=(), context_action=ContextAction.RESET)
    state.update_conversation_state("s", new)
    c = state.get_conversation_state("s")
    assert c.active_targets == []
    assert c.focused_target is None


def test_unknown_plan_cannot_search():
    p = make_plan(rr(knowledge_status="UNKNOWN_IDENTITY", context_action=ContextAction.CLARIFY))
    assert isinstance(p, ExecutionPlan)
    assert not p.clarification_required
    assert p.tools == ()


def test_unknown_evidence_cannot_pass_as_factual_answer():
    from rag.semantic_contracts import EvidenceBundle, EvidenceStatus
    from rag.validation.answer import validate_answer
    req = rr()
    bundle = EvidenceBundle(EvidenceStatus.UNKNOWN)
    ok, problems = validate_answer("The warranty is 24 months.", req, bundle)
    assert not ok and "unsupported_status_not_acknowledged" in problems


def test_multi_target_state_does_not_arbitrarily_focus_first_target():
    state = AppState()
    multi = rr(
        targets=(
            {"entity": "A", "document": "a.pdf", "topic": "warranty"},
            {"entity": "B", "document": "b.pdf", "topic": "warranty"},
        ),
        target_count=2,
        entity=None,
        document=None,
    )
    state.update_conversation_state("s", multi)
    c = state.get_conversation_state("s")
    assert len(c.active_targets) == 2
    assert c.focused_target is None
