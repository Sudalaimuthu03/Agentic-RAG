from types import SimpleNamespace

from rag.agent.graph import _is_authorized_tool
from rag.execution.contracts import ExecutionPlan
from rag.semantic_contracts import ContextAction, ResolvedRequest


def req(intent="factual_lookup"):
    return ResolvedRequest(
        turn_id="1", raw_query="q", intent=intent,
        entity="A", topic="warranty", document="a.pdf",
        evidence_scope=("a.pdf",), targets=(("entity", "A"),) if False else ({"entity":"A","document":"a.pdf","topic":"warranty"},),
        target_count=1, context_action=ContextAction.SWITCH,
        knowledge_status="AUTHORIZED", knowledge_mode="APPLICATION",
    )


def test_execution_plan_is_the_tool_authority():
    plan = ExecutionPlan("factual_lookup", tools=("search_documents",))
    assert _is_authorized_tool(plan, "search_documents")
    assert not _is_authorized_tool(plan, "list_available_documents")
    assert not _is_authorized_tool(plan, "ask_clarification")


def test_execution_plan_multi_target_scope_is_preserved():
    plan = ExecutionPlan(
        "factual_lookup",
        targets=(
            {"entity":"A","document":"a.pdf"},
            {"entity":"B","document":"b.pdf"},
        ),
        tools=("search_documents",),
    )
    assert len(plan.targets) == 2
    assert plan.targets[0]["entity"] == "A"
    assert plan.targets[1]["entity"] == "B"
