from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from rag.semantic_contracts import ContextAction, Intent, ResolvedRequest

_VARIANT_RE = re.compile(r"\b[A-Za-z]{1,12}[- ]?\d{2,5}[A-Za-z0-9-]*\b")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _find_mentions(query: str, sources: Iterable[str]) -> list[str]:
    q = _norm(query)
    q_tokens = set(q.split())
    hits = []
    for s in sources:
        stem_tokens = [t for t in _norm(Path(s).stem).split() if len(t) >= 4]
        if stem_tokens and (" ".join(stem_tokens) in q or any(t in q_tokens for t in stem_tokens)):
            hits.append(s)
    return hits


def _variant(query: str) -> Optional[str]:
    m = _VARIANT_RE.search(query)
    return m.group(0).replace(" ", "-").upper() if m else None


def _intent(query: str) -> Intent:
    q = query.lower().strip()
    if q in {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}:
        return Intent.SMALLTALK
    if any(x in q for x in ("are you sure", "recheck", "verify", "confirm")):
        return Intent.VERIFY
    if any(x in q for x in ("summarize", "summary", "give me a summary", "summarise")):
        return Intent.SUMMARY
    if any(x in q for x in ("list documents", "what documents", "which documents", "show documents")):
        return Intent.LIST
    return Intent.FACTUAL




def _content_entity_candidates(query: str, documents: dict[str, str]) -> list[tuple[str, str]]:
    qtokens=set(_norm(query).split())
    found=[]
    for source,text in documents.items():
        first_line=next((ln.strip() for ln in text.splitlines() if ln.strip()), text[:300])
        m=re.match(r"^([A-Z][A-Za-z0-9-]{2,})(?:\s+([A-Z][A-Za-z]{2,}))?", first_line)
        names=[]
        if m:
            names.append(m.group(1))
            if m.group(2) and len(m.group(2)) >= 3: names.append(m.group(1)+" "+m.group(2))
        # Also support explicit person names appearing as a standalone heading.
        for name in names:
            nt=set(_norm(name).split())
            if (len(qtokens) >= 2 and nt <= qtokens) or (len(qtokens) == 1 and nt & qtokens):
                found.append((source,name))
    if len(qtokens) >= 2:
        multi=[(src,name) for src,name in found if len(_norm(name).split()) >= 2]
        if multi:
            found=multi
    return found

def resolve_request(
    query: str,
    turn_id: str,
    sources: list[str],
    prior: Optional[ResolvedRequest] = None,
    selected_reference: Optional[str] = None,
    documents: Optional[dict[str, str]] = None,
) -> ResolvedRequest:
    """Legacy compatibility resolver used by the original RuntimeFix2 tests only.

    The live application does not call this function. Production semantic
    authority is implemented by ``rag.semantic.understanding`` plus corpus
    verification and reference binding. This adapter remains solely to avoid
    breaking the original RuntimeFix2 unit-test contract during migration.
    """
    intent = _intent(query)
    q = query.strip()
    qn = _norm(q)
    mentions = _find_mentions(q, sources)
    variant = _variant(q)
    if documents:
        content_entities = _content_entity_candidates(q, documents)
        ce_sources = []
        for src, _name in content_entities:
            if src not in ce_sources: ce_sources.append(src)
        # Content-derived identity is stronger than filename coincidence.
        if ce_sources:
            mentions = ce_sources
        else:
            qwords={w for w in qn.split() if len(w)>=4}
            scored=[]
            for src,text in documents.items():
                words=set(_norm(text[:12000]).split())
                score=len(qwords & words)
                if score>=2: scored.append((score,src))
            if scored:
                best=max(x[0] for x in scored); best_sources=[src for score,src in scored if score==best]
                if best>=2: mentions=best_sources
    reference = None
    reference_match = re.search(r"\b(it|this|that|they|them|her|his|their|these|those)\b", q, re.I)
    if reference_match:
        reference = reference_match.group(1)

    explicit_document = mentions[0] if len(mentions) == 1 else None
    # Content-level identity resolution for documents whose filenames carry no
    # useful identity. Exact multi-word
    # names are stronger than fuzzy filename similarity.
    if not mentions and documents:
        content_hits = []
        for source, text in documents.items():
            if qn and qn in _norm(text[:5000]):
                content_hits.append(source)
        entity_hits = _content_entity_candidates(q, documents)
        entity_sources = []
        for source, _name in entity_hits:
            if source not in entity_sources:
                entity_sources.append(source)
        if len(entity_sources) == 1:
            explicit_document = entity_sources[0]
            mentions = entity_sources
        elif len(entity_sources) > 1 and any(len(set(_norm(n).split()) & set(qn.split())) >= 1 for _s,n in entity_hits):
            mentions = entity_sources
        elif len(content_hits) == 1:
            explicit_document = content_hits[0]
            mentions = content_hits
    candidates = tuple(mentions if len(mentions) != 1 else ())
    ambiguity: list[str] = []

    # Context is used only when language explicitly signals continuation.
    uses_reference = reference is not None or intent in {Intent.VERIFY, Intent.SUMMARY} and prior is not None
    if selected_reference:
        explicit_document = selected_reference
        uses_reference = True

    multi_entity = len(mentions) > 1 and bool(re.search(r"\b(and|both|each|respectively)\b", q, re.I))
    if len(mentions) > 1 and variant and documents:
        variant_hits=[src for src in mentions if variant.lower().replace("-", " ") in _norm(documents.get(src, ""))]
        if len(variant_hits) == 1:
            mentions=variant_hits
            explicit_document=variant_hits[0]
        elif not multi_entity:
            ambiguity.extend(mentions)
    elif len(mentions) > 1 and not multi_entity:
        ambiguity.extend(mentions)

    topic = None
    # Legacy-test compatibility resolver: infer a topic generically from query
    # terms that also occur in the corpus, rather than maintaining a domain taxonomy.
    stop = {"what", "what's", "tell", "about", "the", "for", "is", "are", "who",
            "which", "how", "many", "does", "do", "has", "have", "and", "or",
            "me", "you", "this", "that", "these", "those", "please", "period"}
    identity_tokens = set()
    for value in (variant, explicit_document, *mentions):
        identity_tokens.update(_norm(str(value or "")).split())
    q_tokens = [w for w in qn.split() if w not in stop and w not in identity_tokens and len(w) >= 3]
    if documents and q_tokens:
        corpus_text = " ".join(_norm(t) for t in documents.values())
        topic = next((w for w in q_tokens if w in corpus_text.split()), None)

    entity = None
    if explicit_document:
        if documents and explicit_document in documents:
            names = [name for source,name in _content_entity_candidates(q, {explicit_document: documents[explicit_document]}) if source == explicit_document]
            if names:
                qtokens=set(_norm(q).split())
                exact=[n for n in names if set(_norm(n).split()) <= qtokens]
                entity = min(exact, key=lambda n: len(_norm(n))) if exact else min(names, key=lambda n: len(_norm(n)))
                if variant and variant.lower().replace("-", " ") in _norm(entity):
                    entity = entity.split()[0]
    elif prior and uses_reference:
        entity = prior.entity
        if not variant:
            variant = prior.variant
        if prior.document:
            explicit_document = prior.document
    elif prior and variant and not mentions:
        entity = prior.entity
        if prior.topic:
            topic = prior.topic
        if prior.document and variant.lower().replace("-", "") in prior.document.lower().replace("-", ""):
            explicit_document = prior.document
    elif prior and intent == Intent.VERIFY:
        entity = prior.entity
        variant = prior.variant
        explicit_document = prior.document

    if topic is None and prior and (uses_reference or (variant is not None and not mentions)):
        topic = prior.topic

    if multi_entity:
        entity = None
        explicit_document = None
    action = ContextAction.CONTINUE if (prior and (uses_reference or variant is not None and not mentions)) else ContextAction.RESET
    if intent == Intent.VERIFY:
        action = ContextAction.VERIFY
    elif ambiguity:
        action = ContextAction.CLARIFY
    elif uses_reference and prior:
        action = ContextAction.CONTINUE
    elif explicit_document or entity:
        action = ContextAction.SWITCH if prior and prior.entity and entity != prior.entity else ContextAction.CONTINUE

    confidence = 1.0 if multi_entity or len(mentions) == 1 else 0.95 if selected_reference else 0.75 if entity else 0.35
    if ambiguity:
        confidence = 0.25

    return ResolvedRequest(
        turn_id=turn_id,
        raw_query=q,
        intent=intent.value,
        entity=entity,
        variant=variant,
        topic=topic,
        document=explicit_document,
        reference=reference,
        reference_resolution=prior.entity if uses_reference and prior else None,
        ambiguity=tuple(ambiguity),
        evidence_scope=tuple([explicit_document] if explicit_document else mentions),
        verification_target=q if intent == Intent.VERIFY else None,
        confidence=confidence,
        context_action=action,
        resolution_source="explicit_current_turn" if mentions or selected_reference else "context" if uses_reference else "deterministic",
        candidate_entities=tuple(dict.fromkeys([name for src,name in _content_entity_candidates(q, documents or {}) if src in mentions] or mentions)),
        candidate_variants=tuple([variant] if variant else []),
        candidate_documents=tuple(sources),
    )
