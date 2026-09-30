from __future__ import annotations
import re
from rag.semantic_contracts import EvidenceBundle, EvidenceStatus, ResolvedRequest

_INTERNAL = re.compile(r"\b(ResolvedRequest|tool_call|agent_state|execution_gate|retrieval_operation|guardrail_retry|ExecutionPlan|EvidenceBundle)\b", re.I)
_UNSUPPORTED_MARKERS = (
    "not enough evidence", "not established", "cannot verify", "can't verify",
    "not contained", "does not contain", "not available in the indexed documents",
    "i don't have enough information", "not found in the indexed documents",
    "not stated in the indexed documents", "the indexed documents do not establish",
)
_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had", "this", "that", "it", "they", "them", "and", "or", "of", "to", "for", "with", "from", "in", "on", "by", "as", "at", "be", "been", "will", "can", "may"}


def _claim_support_problems(answer: str, evidence: EvidenceBundle) -> list[str]:
    if evidence.status not in {EvidenceStatus.SUPPORTED, EvidenceStatus.MULTI_SOURCE_AGREEMENT, EvidenceStatus.PARTIAL} or not evidence.items:
        return []
    evidence_text = " ".join(item.text.lower() for item in evidence.items)
    evidence_tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9-]{2,}", evidence_text) if t not in _STOPWORDS}
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9-]{2,}", sentence.lower()) if t not in _STOPWORDS}
        if len(tokens) < 2:
            continue
        if not (tokens & evidence_tokens):
            problems.append("claim_without_evidence_overlap")
            break
    return problems


def validate_answer(answer: str, req: ResolvedRequest, evidence: EvidenceBundle) -> tuple[bool, list[str]]:
    problems: list[str] = []
    lower = answer.lower()
    if _INTERNAL.search(answer):
        problems.append("internal_protocol_leakage")

    if evidence.status in {EvidenceStatus.NO_EVIDENCE, EvidenceStatus.INSUFFICIENT}:
        if not any(marker in lower for marker in _UNSUPPORTED_MARKERS):
            problems.append("unsupported_status_not_acknowledged")

    if evidence.status == EvidenceStatus.PARTIAL:
        missing = any(marker in lower for marker in _UNSUPPORTED_MARKERS) or any(x in lower for x in ("not available", "not found", "could not establish", "unable to verify"))
        if not missing:
            problems.append("partial_evidence_not_acknowledged")

    if evidence.status == EvidenceStatus.CONTRADICTORY and not any(x in lower for x in ("conflict", "disagree", "different sources", "sources differ", "different documents", "contradict")):
        problems.append("conflict_not_acknowledged")

    if evidence.status in {EvidenceStatus.SUPPORTED, EvidenceStatus.MULTI_SOURCE_AGREEMENT, EvidenceStatus.PARTIAL} and req.document and evidence.items and any(i.source != req.document for i in evidence.items):
        problems.append("wrong_document_evidence")
    if req.variant and evidence.status in {EvidenceStatus.SUPPORTED, EvidenceStatus.MULTI_SOURCE_AGREEMENT}:
        variants = set(re.findall(r"\b[A-Z]{1,8}-?\d{2,5}\b", answer.upper()))
        requested = req.variant.upper().replace(" ", "-")
        if variants and requested not in {v.replace(" ", "-") for v in variants}:
            problems.append("variant_mismatch")
    if len(req.targets) > 1 and evidence.target_statuses:
        missing = [k for k,v in evidence.target_statuses.items() if v == EvidenceStatus.NO_EVIDENCE.value]
        if missing and not any(marker in lower for marker in _UNSUPPORTED_MARKERS):
            problems.append("multi_target_incomplete_evidence")
    problems.extend(_claim_support_problems(answer, evidence))
    return not problems, problems
