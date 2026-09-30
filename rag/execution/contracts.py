from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionPlan:
    operation: str
    targets: tuple[dict[str, Any], ...] = ()
    tools: tuple[str, ...] = ()
    search_scope: tuple[str, ...] = ()
    document_scope: tuple[str, ...] = ()
    evidence_requirements: tuple[str, ...] = ()
    clarification_required: bool = False
    stop_conditions: tuple[str, ...] = ()
    execution_mode: str = "DOCUMENT_GROUNDED"
    target_requirements: tuple[dict[str, Any], ...] = ()

    @property
    def authorized_tools(self) -> tuple[str, ...]:
        return self.tools

    @property
    def is_document_grounded(self) -> bool:
        return self.execution_mode == "DOCUMENT_GROUNDED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "targets": [dict(t) for t in self.targets],
            "tools": list(self.tools),
            "authorized_tools": list(self.tools),
            "search_scope": list(self.search_scope),
            "document_scope": list(self.document_scope),
            "evidence_requirements": list(self.evidence_requirements),
            "clarification_required": self.clarification_required,
            "stop_conditions": list(self.stop_conditions),
            "execution_mode": self.execution_mode,
            "target_requirements": [dict(t) for t in self.target_requirements],
        }


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    target_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    query: str = ""
    scope: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
