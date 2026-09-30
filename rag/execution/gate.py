from __future__ import annotations

from dataclasses import dataclass
from rag.semantic_contracts import ContextAction, ResolvedRequest


@dataclass
class ExecutionBudget:
    max_agent_iterations: int = 3
    max_llm_calls: int = 3
    max_retrieval_operations: int = 2
    max_guardrail_retries: int = 1
    agent_iterations: int = 0
    llm_calls: int = 0
    retrieval_operations: int = 0
    guardrail_retries: int = 0

    def allow_llm(self) -> bool:
        return self.llm_calls < self.max_llm_calls

    def consume_llm(self) -> None:
        if not self.allow_llm():
            raise RuntimeError("LLM call limit exceeded")
        self.llm_calls += 1

    def allow_retrieval(self) -> bool:
        return self.retrieval_operations < self.max_retrieval_operations

    def consume_retrieval(self) -> None:
        if not self.allow_retrieval():
            raise RuntimeError("Retrieval operation limit exceeded")
        self.retrieval_operations += 1

    def allow_iteration(self) -> bool:
        return self.agent_iterations < self.max_agent_iterations

    def consume_iteration(self) -> None:
        if not self.allow_iteration():
            raise RuntimeError("Agent iteration limit exceeded")
        self.agent_iterations += 1


class ExecutionGate:
    def authorize(self, request: ResolvedRequest) -> tuple[bool, str]:
        if request.context_action == ContextAction.CLARIFY:
            return False, "clarification_required"
        if request.intent == "smalltalk":
            return True, "smalltalk"
        if request.intent == "list_documents":
            return True, "list"
        return True, "execute"
