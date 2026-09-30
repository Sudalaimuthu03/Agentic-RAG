from __future__ import annotations

from typing import Any

from rag.semantic.understanding import SemanticReference


def _resolve_ordinal(reference: SemanticReference, presented: list[dict[str, Any]]) -> dict[str, Any] | None:
    position = reference.selection_position
    if position is None or position < 1:
        return None
    for item in presented:
        if int(item.get("position", 0) or 0) == position:
            return dict(item)
    # Position is a semantic fact; list order is authoritative when explicit positions are absent.
    if position <= len(presented):
        return dict(presented[position - 1])
    return None


def _resolve_reference(reference: SemanticReference, conversation: dict[str, Any]) -> dict[str, Any] | None:
    focused = conversation.get("focused_target") or {}
    targets = [dict(x) for x in (conversation.get("active_targets") or [])]
    presented = [dict(x) for x in (conversation.get("presented_items") or [])]

    if reference.reference_type == "ORDINAL_SELECTION":
        return _resolve_ordinal(reference, presented)

    if reference.reference_type in {"PRONOMINAL_REFERENCE", "TARGET_REFERENCE", "CONTEXT_REFERENCE"}:
        # Pending clarification candidates are authoritative for collective references
        # such as "both" while a clarification is open. This prevents an older active
        # topic from winning over the candidates the system just presented.
        pending = conversation.get("pending_clarification") or {}
        pending_targets = [dict(x) for x in (pending.get("candidate_targets") or []) if isinstance(x, dict)]
        candidate_targets = [dict(x) for x in (conversation.get("candidate_targets") or []) if isinstance(x, dict)]
        expr = (reference.expression or "").strip().lower()
        collective = expr in {"both", "all", "these", "those", "compare them", "compare both"}
        if pending_targets and collective:
            return {"_multi_target_reference": True, "targets": pending_targets}
        if candidate_targets and len(candidate_targets) > 1 and collective:
            return {"_multi_target_reference": True, "targets": candidate_targets}
        if focused:
            return dict(focused)
        if len(targets) == 1:
            return targets[0]
        # Collective references resolve to the authoritative target set, never to an arbitrary member.
        if len(targets) > 1:
            return {"_multi_target_reference": True, "targets": targets}
        return None

    if reference.reference_type == "DOCUMENT_REFERENCE":
        if reference.target_id:
            for item in presented + targets:
                if str(item.get("document") or "") == reference.target_id:
                    return item
        if focused.get("document"):
            return dict(focused)
        if len(targets) == 1 and targets[0].get("document"):
            return targets[0]
        return None

    if reference.reference_type == "VERIFICATION_REFERENCE":
        # Verification is bound to the immediately active authoritative target set.
        # Historical resolutions are audit data only and must never resurrect an older topic.
        last_success = [dict(x) for x in (conversation.get("last_successful_target_set") or [])]
        if len(targets) > 1:
            return {"_multi_target_reference": True, "targets": targets}
        if len(last_success) > 1:
            return {"_multi_target_reference": True, "targets": last_success}
        if focused:
            return dict(focused)
        if len(targets) == 1:
            return targets[0]
        if len(last_success) == 1:
            return last_success[0]
        return None

    return None


def bind_reference(semantic: Any, conversation: dict[str, Any]) -> dict[str, Any]:
    """Bind structured semantic references to authoritative conversation objects.

    This layer intentionally does not inspect or interpret natural-language phrases.
    """
    refs = list(getattr(semantic, "references", ()) or ())
    if not refs:
        return {"resolved": None, "expression": None, "status": "NONE", "reference_type": None}

    # One semantic reference is the normal contract for a context-dependent turn.
    # Multiple references are preserved and reported ambiguous rather than guessed.
    if len(refs) != 1:
        return {
            "resolved": None,
            "expression": ", ".join(r.expression for r in refs if r.expression),
            "status": "AMBIGUOUS",
            "reference_type": "MULTIPLE",
            "references": [r.as_dict() for r in refs],
        }

    reference = refs[0]
    selected = _resolve_reference(reference, conversation)
    if isinstance(selected, dict) and selected.get("_multi_target_reference"):
        return {
            "resolved": None,
            "resolved_targets": [dict(x) for x in selected.get("targets", [])],
            "expression": reference.expression or None,
            "status": "RESOLVED",
            "reference_type": reference.reference_type,
            "reference": reference.as_dict(),
        }
    result = {
        "resolved": selected,
        "resolved_targets": [selected] if selected else [],
        "expression": reference.expression or None,
        "status": "RESOLVED" if selected else ("AMBIGUOUS" if (conversation.get("active_targets") and len(conversation.get("active_targets")) > 1) else "UNRESOLVED"),
        "reference_type": reference.reference_type,
        "reference": reference.as_dict(),
    }
    return result
