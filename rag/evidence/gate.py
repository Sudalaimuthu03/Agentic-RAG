from __future__ import annotations

import re
from rag.semantic_contracts import EvidenceBundle, EvidenceItem, EvidenceStatus, ResolvedRequest


def _matches(item: EvidenceItem, req: ResolvedRequest) -> bool:
    if req.document and item.source != req.document:
        return False
    if req.variant and item.variant:
        if req.variant.lower().replace(" ", "-") not in item.variant.lower().replace(" ", "-"):
            return False
    if req.entity and item.entity:
        if req.entity.lower() not in item.entity.lower() and item.entity.lower() not in req.entity.lower():
            return False
    if req.topic and item.topic:
        if req.topic.lower() not in item.topic.lower() and item.topic.lower() not in req.topic.lower():
            return False
    return True


def _signatures(text: str) -> set[str]:
    """Extract conservative scalar signatures used only to detect explicit conflicts."""
    t = " ".join(str(text or "").lower().split())
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:months?|years?|days?|weeks?|hours?|minutes?|seconds?|%)\b",
        r"\b\d+(?:\.\d+)?\s*percent\b",
        r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b",
    ]
    out = set()
    for pattern in patterns:
        out.update(re.findall(pattern, t))
    return {" ".join(x.split()) for x in out}


def assess(req: ResolvedRequest, items: list[EvidenceItem]) -> EvidenceBundle:
    scoped = [i for i in items if _matches(i, req)]
    if not scoped:
        return EvidenceBundle(EvidenceStatus.NO_EVIDENCE, reason="No evidence aligned with the resolved request")

    target_statuses: dict[str, str] = {}
    if len(req.targets) > 1:
        for idx, target in enumerate(req.targets, 1):
            target_items = [i for i in scoped if i.target_id == str(idx)]
            if not target_items and target.get("document"):
                target_items = [i for i in scoped if i.source == target.get("document")]
            target_statuses[str(idx)] = EvidenceStatus.SUPPORTED.value if target_items else EvidenceStatus.NO_EVIDENCE.value

    missing_targets = bool(target_statuses and any(v == EvidenceStatus.NO_EVIDENCE.value for v in target_statuses.values()))

    # Multiple sources are not a conflict merely because their text differs. A conflict
    # requires the same target to have explicit scalar evidence that disagrees.
    contradiction_sources: set[str] = set()
    source_groups: dict[str, list[EvidenceItem]] = {}
    for item in scoped:
        key = item.target_id or (item.source if len(req.targets) == 1 else "")
        source_groups.setdefault(key, []).append(item)

    for group in source_groups.values():
        sources = {i.source for i in group}
        if len(sources) < 2:
            continue
        signatures_by_source = {source: set().union(*(_signatures(i.text) for i in group if i.source == source)) for source in sources}
        nonempty = [v for v in signatures_by_source.values() if v]
        if len(nonempty) >= 2:
            union = set().union(*nonempty)
            if all(v.isdisjoint(union - v) for v in nonempty):
                contradiction_sources.update(sources)

    if contradiction_sources:
        return EvidenceBundle(
            EvidenceStatus.CONTRADICTORY,
            tuple(scoped),
            tuple(sorted(contradiction_sources)),
            "Explicit scalar evidence differs across sources for the same target",
            target_statuses,
        )

    distinct_sources = {i.source for i in scoped}
    if missing_targets:
        return EvidenceBundle(EvidenceStatus.PARTIAL, tuple(scoped), reason="Evidence supports some requested targets but not all", target_statuses=target_statuses)
    if len(distinct_sources) > 1:
        reason = ("Multiple sources support the same target without an explicit contradiction"
                  if len(req.targets) <= 1 else
                  "Multiple requested targets are supported by their respective source documents")
        return EvidenceBundle(EvidenceStatus.MULTI_SOURCE_AGREEMENT, tuple(scoped), reason=reason, target_statuses=target_statuses)
    return EvidenceBundle(EvidenceStatus.SUPPORTED, tuple(scoped), reason="Evidence aligns with the resolved target and scope", target_statuses=target_statuses)
