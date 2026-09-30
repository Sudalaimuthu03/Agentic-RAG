from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple


class ContextAction(str, Enum):
    CONTINUE = "CONTINUE"
    SWITCH = "SWITCH"
    RESET = "RESET"
    CLARIFY = "CLARIFY"
    VERIFY = "VERIFY"


class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    MULTI_SOURCE_AGREEMENT = "MULTI_SOURCE_AGREEMENT"
    INSUFFICIENT = "INSUFFICIENT"
    NO_EVIDENCE = "NO_EVIDENCE"
    CONTRADICTORY = "CONTRADICTORY"
    PARTIAL = "PARTIAL"
    # Backward-compatible aliases for existing callers/tests.
    KNOWN = "SUPPORTED"
    UNKNOWN = "NO_EVIDENCE"
    CONFLICTING = "CONTRADICTORY"


class Intent(str, Enum):
    FACTUAL = "factual_lookup"
    SUMMARY = "summarization"
    VERIFY = "verification"
    LIST = "list_documents"
    CLARIFY = "clarification"
    SMALLTALK = "smalltalk"


@dataclass(frozen=True)
class ResolvedRequest:
    turn_id: str
    raw_query: str
    intent: str
    entity: Optional[str] = None
    entity_type: Optional[str] = None
    variant: Optional[str] = None
    topic: Optional[str] = None
    document: Optional[str] = None
    reference: Optional[str] = None
    reference_resolution: Optional[str] = None
    ambiguity: Tuple[str, ...] = ()
    evidence_scope: Tuple[str, ...] = ()
    verification_target: Optional[str] = None
    confidence: float = 0.0
    context_action: ContextAction = ContextAction.RESET
    resolution_source: str = "unknown"
    candidate_entities: Tuple[str, ...] = ()
    candidate_variants: Tuple[str, ...] = ()
    candidate_documents: Tuple[str, ...] = ()
    targets: Tuple[dict[str, Any], ...] = ()
    target_count: int = 0
    sufficiency: str = "SUFFICIENT"
    knowledge_status: str = "AUTHORIZED"
    knowledge_mode: str = "APPLICATION"
    missing_information: Tuple[str, ...] = ()
    context_dependency: str = "NONE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "raw_query": self.raw_query,
            "intent": self.intent,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "variant": self.variant,
            "topic": self.topic,
            "document": self.document,
            "reference": self.reference,
            "reference_resolution": self.reference_resolution,
            "ambiguity": list(self.ambiguity),
            "evidence_scope": list(self.evidence_scope),
            "verification_target": self.verification_target,
            "confidence": self.confidence,
            "context_action": self.context_action.value,
            "resolution_source": self.resolution_source,
            "candidate_entities": list(self.candidate_entities),
            "candidate_variants": list(self.candidate_variants),
            "candidate_documents": list(self.candidate_documents),
            "targets": [dict(x) for x in self.targets],
            "target_count": self.target_count,
            "sufficiency": self.sufficiency,
            "knowledge_status": self.knowledge_status,
            "knowledge_mode": self.knowledge_mode,
            "missing_information": list(self.missing_information),
            "context_dependency": self.context_dependency,
        }


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    page: Optional[int]
    chunk_id: Optional[str]
    parent_id: Optional[str]
    text: str
    entity: Optional[str]
    variant: Optional[str]
    topic: Optional[str]
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    target_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "entity": self.entity,
            "variant": self.variant,
            "topic": self.topic,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    status: EvidenceStatus
    items: Tuple[EvidenceItem, ...] = ()
    conflicts: Tuple[str, ...] = ()
    reason: str = ""
    target_statuses: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "items": [i.as_dict() for i in self.items],
            "conflicts": list(self.conflicts),
            "reason": self.reason,
            "target_statuses": dict(self.target_statuses),
        }

@dataclass(frozen=True)
class KnowledgeBoundaryDecision:
    mode: str
    status: str
    required_authority: str
    allowed_sources: Tuple[str, ...] = ()
    prohibited_sources: Tuple[str, ...] = ()
    corpus_required: bool = True
    external_source_required: bool = False
    reason: str = ""


@dataclass(frozen=True)
class EntityResolution:
    requested_identity: Optional[str]
    resolved_identity: Optional[str]
    entity_type: Optional[str]
    variant: Optional[str]
    document_ids: Tuple[str, ...] = ()
    status: str = "UNKNOWN"
    ambiguity: Tuple[str, ...] = ()
    confidence: float = 0.0
    resolution_basis: str = "UNKNOWN"


@dataclass(frozen=True)
class ReferenceResolution:
    expression: Optional[str]
    reference_type: str
    candidate_targets: Tuple[dict[str, Any], ...] = ()
    resolved_target: Optional[dict[str, Any]] = None
    status: str = "UNRESOLVED"
    resolution_basis: str = "UNKNOWN"
    context_dependency: str = "NONE"


@dataclass(frozen=True)
class ContextOperation:
    type: str
    retained_targets: Tuple[dict[str, Any], ...] = ()
    added_targets: Tuple[dict[str, Any], ...] = ()
    removed_targets: Tuple[dict[str, Any], ...] = ()
    focused_target: Optional[dict[str, Any]] = None
    focused_topic: Optional[str] = None
