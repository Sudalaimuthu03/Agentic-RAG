from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SemanticReference:
    """Structured meaning of a user reference. Natural-language parsing belongs to Stage 1."""
    reference_type: str
    expression: str = ""
    target_dimension: str = "TARGET"
    selection_position: Optional[int] = None
    target_id: Optional[str] = None
    relative_to: Optional[str] = None
    requires_context: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_type": self.reference_type,
            "expression": self.expression,
            "target_dimension": self.target_dimension,
            "selection_position": self.selection_position,
            "target_id": self.target_id,
            "relative_to": self.relative_to,
            "requires_context": self.requires_context,
        }


@dataclass(frozen=True)
class SemanticInterpretation:
    interaction: str = "DOCUMENT"
    operation: str = "FACTUAL_QUERY"
    entities: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()
    references: tuple[SemanticReference, ...] = ()
    targets: tuple[dict[str, Any], ...] = ()
    target_count: int = 0
    sufficiency: str = "INSUFFICIENT"
    ambiguity: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    context_dependency: str = "NONE"
    knowledge_mode: str = "APPLICATION"
    context_action: str = "RESET"
    uncertainty: str = "LOW"

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction": self.interaction,
            "operation": self.operation,
            "entities": list(self.entities),
            "variants": list(self.variants),
            "topics": list(self.topics),
            "documents": list(self.documents),
            "references": [r.as_dict() for r in self.references],
            "targets": [dict(t) for t in self.targets],
            "target_count": self.target_count,
            "sufficiency": self.sufficiency,
            "ambiguity": list(self.ambiguity),
            "missing_information": list(self.missing_information),
            "context_dependency": self.context_dependency,
            "knowledge_mode": self.knowledge_mode,
            "context_action": self.context_action,
            "uncertainty": self.uncertainty,
        }


_ALLOWED = {
    "interaction": {"CONVERSATIONAL", "DOCUMENT", "UNKNOWN"},
    "operation": {"SMALLTALK", "HELP", "LIST_DOCUMENTS", "DOCUMENT_OVERVIEW", "FACTUAL_QUERY", "SUMMARY", "VERIFY", "CLARIFY"},
    "sufficiency": {"SUFFICIENT", "INSUFFICIENT", "CONTEXT_DEPENDENT", "AMBIGUOUS"},
    "context_dependency": {"NONE", "OPTIONAL", "REQUIRED", "RESOLVED", "AMBIGUOUS", "UNRESOLVED"},
    "knowledge_mode": {"CONVERSATIONAL", "APPLICATION", "EXTERNAL"},
    "context_action": {"CONTINUE", "SWITCH", "RESET", "VERIFY", "CLARIFY"},
}

_REFERENCE_TYPES = {
    "ORDINAL_SELECTION",
    "PRONOMINAL_REFERENCE",
    "VERIFICATION_REFERENCE",
    "TARGET_REFERENCE",
    "DOCUMENT_REFERENCE",
    "CONTEXT_REFERENCE",
}


def _clean_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(x).strip() for x in value if str(x).strip())


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start:end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("semantic model did not return a JSON object")


def _normalize(value: Any, key: str) -> str:
    value = str(value or "").strip().upper()
    if value not in _ALLOWED.get(key, {value}):
        return next(iter(_ALLOWED[key])) if key in _ALLOWED else value
    return value


def _parse_references(value: Any) -> tuple[SemanticReference, ...]:
    """Accept only structured semantic references; never infer meaning from reference text."""
    if not isinstance(value, list):
        return ()
    result: list[SemanticReference] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ref_type = str(item.get("reference_type") or "").strip().upper()
        if ref_type not in _REFERENCE_TYPES:
            continue
        position = item.get("selection_position")
        try:
            position = int(position) if position is not None else None
        except (TypeError, ValueError):
            position = None
        result.append(SemanticReference(
            reference_type=ref_type,
            expression=str(item.get("expression") or "").strip(),
            target_dimension=str(item.get("target_dimension") or "TARGET").strip().upper(),
            selection_position=position,
            target_id=(str(item["target_id"]).strip() if item.get("target_id") is not None else None),
            relative_to=(str(item["relative_to"]).strip().upper() if item.get("relative_to") is not None else None),
            requires_context=bool(item.get("requires_context", True)),
        ))
    return tuple(result)


def _fallback_on_parse_failure() -> SemanticInterpretation:
    return SemanticInterpretation(
        interaction="UNKNOWN",
        operation="CLARIFY",
        sufficiency="INSUFFICIENT",
        missing_information=("what you want me to do",),
        knowledge_mode="APPLICATION",
        context_dependency="UNRESOLVED",
        context_action="CLARIFY",
        uncertainty="HIGH",
    )


def understand(llm: Any, query: str, *, catalog: list[dict[str, Any]], conversation: Optional[dict[str, Any]] = None) -> SemanticInterpretation:
    catalog_text = json.dumps(catalog, ensure_ascii=False)
    state_text = json.dumps(conversation or {}, ensure_ascii=False)
    system = """You are the semantic understanding component of a closed-world document RAG application.
Understand the user's current message. Do NOT answer it. Do NOT use outside factual knowledge.
Recognition is not corpus existence. The document catalog is only an availability reference; it does not prove facts.
Interpret the current turn independently first. Use conversation state only to resolve genuine references or continuation.
Explicit current-turn identities override stale context. Keep entity, variant, topic, document, requested fact, and reference distinct.
Do not invent an entity/document merely because a similar item exists.
Do not infer that a request is a document-inventory request merely because it contains the word 'document'.
If the request is incomplete, say so. If a reference needs context, mark CONTEXT_DEPENDENT. If multiple interpretations are valid, mark AMBIGUOUS.
When an entity, topic, or document name is potentially ambiguous, use the authorized document catalog to identify the relevant candidate entities/documents rather than treating the entire catalog as the candidate set. Candidate discovery is not authorization: candidates are evidence for clarification, while factual execution still requires corpus verification. Preserve all relevant candidates when multiple distinct matches exist.
For external-world factual requests in this document-only application, set knowledge_mode=EXTERNAL.

A reference is a semantic object, not a raw phrase. If the user refers to a prior/presented object, describe the meaning structurally.
Use only these reference_type values:
ORDINAL_SELECTION, PRONOMINAL_REFERENCE, VERIFICATION_REFERENCE, TARGET_REFERENCE, DOCUMENT_REFERENCE, CONTEXT_REFERENCE.
For ordinal selection provide a 1-based selection_position. For verification use relative_to=PREVIOUS_REQUEST when appropriate.
The reference expression is descriptive metadata only; the runtime will never parse it to determine meaning.

Return ONLY valid JSON matching this shape:
{
  "interaction":"CONVERSATIONAL|DOCUMENT|UNKNOWN",
  "operation":"SMALLTALK|HELP|LIST_DOCUMENTS|DOCUMENT_OVERVIEW|FACTUAL_QUERY|SUMMARY|VERIFY|CLARIFY",
  "entities":[], "variants":[], "topics":[], "documents":[],
  "references":[{"reference_type":"ORDINAL_SELECTION|PRONOMINAL_REFERENCE|VERIFICATION_REFERENCE|TARGET_REFERENCE|DOCUMENT_REFERENCE|CONTEXT_REFERENCE","expression":"","target_dimension":"TARGET","selection_position":null,"target_id":null,"relative_to":null,"requires_context":true}],
  "targets":[{"entity":null,"variant":null,"topic":null,"document":null,"fact":null}],
  "target_count":0,
  "sufficiency":"SUFFICIENT|INSUFFICIENT|CONTEXT_DEPENDENT|AMBIGUOUS",
  "ambiguity":[], "missing_information":[],
  "context_dependency":"NONE|OPTIONAL|REQUIRED|RESOLVED|AMBIGUOUS|UNRESOLVED",
  "knowledge_mode":"CONVERSATIONAL|APPLICATION|EXTERNAL",
  "context_action":"CONTINUE|SWITCH|RESET|VERIFY|CLARIFY",
  "uncertainty":"LOW|MEDIUM|HIGH"
}
"""
    prompt = f"CURRENT USER MESSAGE:\n{query}\n\nAUTHORIZED DOCUMENT CATALOG:\n{catalog_text}\n\nCONVERSATION STATE (not factual evidence):\n{state_text}"
    try:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        except ImportError:
            messages = [system, prompt]
        response = llm.invoke(messages)
        raw = str(getattr(response, "content", "") or "")
        data = _parse_json(raw)
        targets_raw = data.get("targets") if isinstance(data.get("targets"), list) else []
        targets = []
        for item in targets_raw:
            if isinstance(item, dict):
                targets.append({
                    "entity": item.get("entity"), "variant": item.get("variant"),
                    "topic": item.get("topic"), "document": item.get("document"),
                    "fact": item.get("fact"),
                })
        entities = _clean_list(data.get("entities"))
        variants = _clean_list(data.get("variants"))
        topics = _clean_list(data.get("topics"))
        documents = _clean_list(data.get("documents"))
        refs = _parse_references(data.get("references"))
        target_count = int(data.get("target_count") or len(targets) or max(len(entities), len(documents), 0))
        return SemanticInterpretation(
            interaction=_normalize(data.get("interaction", "DOCUMENT"), "interaction"),
            operation=_normalize(data.get("operation", "FACTUAL_QUERY"), "operation"),
            entities=entities, variants=variants, topics=topics, documents=documents,
            references=refs, targets=tuple(targets), target_count=target_count,
            sufficiency=_normalize(data.get("sufficiency", "INSUFFICIENT"), "sufficiency"),
            ambiguity=_clean_list(data.get("ambiguity")),
            missing_information=_clean_list(data.get("missing_information")),
            context_dependency=_normalize(data.get("context_dependency", "NONE"), "context_dependency"),
            knowledge_mode=_normalize(data.get("knowledge_mode", "APPLICATION"), "knowledge_mode"),
            context_action=_normalize(data.get("context_action", "RESET"), "context_action"),
            uncertainty=str(data.get("uncertainty", "MEDIUM")).upper(),
        )
    except Exception:
        return _fallback_on_parse_failure()
