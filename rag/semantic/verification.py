from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def compact(value: str) -> str:
    return norm(value).replace(" ", "")


@dataclass(frozen=True)
class VerifiedIdentity:
    status: str
    document: str | None = None
    entity: str | None = None
    variant: str | None = None
    candidate_documents: tuple[str, ...] = ()
    resolution_basis: str = "UNKNOWN"


def _contains_phrase(text: str, phrase: str) -> bool:
    p = norm(phrase)
    if not p:
        return False
    return p in norm(text)


def _exact_document_match(document: str, documents: dict[str, str]) -> list[str]:
    q = norm(document)
    stem = norm(Path(document).stem)
    return [d for d in documents if norm(d) == q or norm(Path(d).stem) == stem]


def _tokenize(value: str) -> set[str]:
    return {t for t in norm(value).split() if len(t) >= 3}


def _query_terms(query: str) -> list[str]:
    """Extract generic content terms from the current user message.

    This is deliberately domain-agnostic. It is only used when the semantic model
    fails to supply a topic/identity for an otherwise searchable current query.
    """
    stop = {
        "what", "what's", "tell", "about", "the", "for", "is", "are", "who",
        "which", "how", "many", "does", "do", "has", "have", "had", "and", "or",
        "me", "you", "this", "that", "these", "those", "please", "can", "could",
        "would", "should", "i", "we", "my", "your", "any", "some", "there",
        "want", "need", "know", "information", "provide", "give", "show", "about",
        "again", "check", "recheck", "verify", "sure", "it", "its", "them",
    }
    return [w for w in norm(query).split() if w not in stop and len(w) >= 3]


def _discover_from_current_query(raw_query: str, documents: dict[str, str]) -> list[str]:
    terms = _query_terms(raw_query)
    if not terms:
        return []
    scored: list[tuple[int, str]] = []
    for source, text in documents.items():
        hay = set(norm(source).split()) | set(norm(text[:20000]).split())
        score = sum(1 for term in terms if term in hay)
        if score:
            scored.append((score, source))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [source for score, source in sorted(scored, key=lambda x: (-x[0], x[1].lower())) if score == best]


def _pending_resolution(raw_query: str, pending: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a short user response against an open clarification before LLM guesses.

    The pending clarification is authoritative because it was created by the immediately
    preceding turn. Historical conversation is never consulted here.
    """
    targets = [dict(x) for x in (pending.get("candidate_targets") or []) if isinstance(x, dict)]
    if not targets:
        return None
    q = norm(raw_query)
    if not q:
        return None
    collective = q in {"both", "all", "these", "those", "compare them", "compare both"}
    if collective and len(targets) > 1:
        return {"targets": targets, "mode": "COLLECTIVE"}
    # Exact candidate identity selection from the current clarification candidates.
    scored: list[tuple[int, dict[str, Any]]] = []
    q_tokens = set(q.split())
    for target in targets:
        vals = []
        for key in ("entity", "variant", "document", "topic", "fact"):
            value = norm(str(target.get(key) or ""))
            if value:
                vals.append(value)
        best = 0
        for value in vals:
            if value == q:
                best = max(best, 100)
            else:
                vt = set(value.split())
                overlap = len(vt & q_tokens)
                if overlap and overlap == len(vt):
                    best = max(best, 80 + overlap)
        if best:
            scored.append((best, target))
    if scored:
        scored.sort(key=lambda x: -x[0])
        top = scored[0][0]
        winners = [t for score, t in scored if score == top]
        if len(winners) == 1:
            return {"targets": winners, "mode": "SINGLE"}
    affirmative = q in {"yes", "yeah", "yep", "correct", "okay", "ok", "sure", "that one"}
    if affirmative and len(targets) == 1 and pending.get("kind") == "CONFIRMATION":
        return {"targets": targets, "mode": "CONFIRMATION"}
    return None


def _candidate_discovery_terms(target: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("entity", "variant", "topic", "fact", "document"):
        value = str(target.get(key) or "").strip()
        if value:
            terms.append(value)
    return terms


def _discover_candidate_documents(
    targets: list[dict[str, Any]],
    semantic_documents: Iterable[str],
    corpus: dict[str, str],
) -> tuple[list[str], str]:
    """Discover a bounded candidate set without treating the whole corpus as candidates.

    This is a generic corpus-discovery stage: explicit semantic document identities are
    verified first; otherwise entity/variant/topic/fact terms are matched against corpus
    content and filenames. No current-corpus names or domain taxonomy are embedded here.
    """
    explicit = []
    for item in semantic_documents:
        explicit.extend(_exact_document_match(str(item), corpus))
    explicit = list(dict.fromkeys(explicit))
    if explicit:
        return explicit, "SEMANTIC_DOCUMENT_IDENTITY"

    scored: list[tuple[float, str]] = []
    identity_constraints = [
        str(t.get("entity") or t.get("variant") or "").strip()
        for t in targets
        if str(t.get("entity") or t.get("variant") or "").strip()
    ]
    for source, text in corpus.items():
        source_norm = norm(source)
        text_norm = norm(text)
        if identity_constraints:
            # Identity-constrained discovery must first establish that the
            # candidate document actually contains the requested identity.
            if not any(_contains_phrase(text, identity) or _contains_phrase(source, identity)
                       or compact(identity) in compact(text) or compact(identity) in compact(source)
                       for identity in identity_constraints):
                continue
        source_tokens = _tokenize(source_norm)
        text_tokens = _tokenize(text_norm)
        score = 0.0
        matched = 0
        for target in targets:
            target_score = 0.0
            for term in _candidate_discovery_terms(target):
                term_norm = norm(term)
                if not term_norm:
                    continue
                term_tokens = _tokenize(term_norm)
                if term_norm in text_norm or term_norm in source_norm:
                    target_score += 4.0 if target.get("entity") else 3.0
                    matched += 1
                    continue
                corpus_tokens = text_tokens | source_tokens
                # Generic light morphology: singular/plural forms are equivalent for
                # discovery, without introducing domain-specific vocabulary.
                normalized_term_tokens = set(term_tokens)
                for token in list(term_tokens):
                    if token.endswith("s") and len(token) > 3:
                        normalized_term_tokens.add(token[:-1])
                    elif len(token) > 3:
                        normalized_term_tokens.add(token + "s")
                overlap = len(normalized_term_tokens & corpus_tokens)
                if overlap:
                    target_score += min(2.5, 0.75 * overlap)
                    matched += 1
            if target_score:
                score += target_score
        if score > 0 and matched > 0:
            scored.append((score, source))

    if not scored:
        return [], "NO_CANDIDATES"

    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    # Keep candidates that are close to the best match; this prevents unrelated
    # corpus documents from becoming ambiguity candidates while allowing multiple
    # legitimate documents to survive.
    best = scored[0][0]
    # Topic/fact-only discovery is intentionally allowed to use a lower generic
    # threshold because singular/plural and concise topic terms may score below the
    # identity-constrained threshold. Identity-constrained discovery remains strict.
    threshold = max(0.75 if not identity_constraints else 2.0, best * 0.55)
    candidates = [source for score, source in scored if score >= threshold]
    return candidates, "CORPUS_DISCOVERY"


def _candidate_documents(entity: str | None, variant: str | None, document: str | None, documents: dict[str, str], *, topic: str | None = None, fact: str | None = None) -> tuple[list[str], str]:
    if document:
        exact = _exact_document_match(document, documents)
        if exact:
            return exact, "EXACT_CANONICAL"
        return [], "UNKNOWN"

    candidates = list(documents)
    if entity:
        # Corpus content/filename is evidence for the relationship; this is not retrieval.
        e = compact(entity)
        candidates = [d for d in candidates if _contains_phrase(documents[d], entity) or _contains_phrase(d, entity) or (e and e in compact(documents[d])) or (e and e in compact(d))]
    if variant:
        # Compact normalization supports corpus-established forms such as SB450/SB-450.
        v = compact(variant)
        candidates = [d for d in candidates if v and (v in compact(documents[d]) or v in compact(d))]
    if candidates and (entity or variant):
        return candidates, "VERIFIED_METADATA"
    # If the user supplied an explicit entity/variant and the corpus cannot
    # establish that identity, topic similarity must never manufacture a
    # different document as a candidate. Topic-only discovery is valid only
    # when no identity constraint was supplied.
    if entity or variant:
        return [], "UNKNOWN"
    discovered, basis = _discover_candidate_documents(
        [{"entity": None, "variant": None, "topic": topic, "fact": fact, "document": None}],
        (),
        documents,
    )
    return discovered, basis


def verify_identity(semantic: Any, documents: dict[str, str], reference: dict[str, Any] | None = None) -> list[VerifiedIdentity]:
    """Verify identities only from the corpus identity surface.

    Recognition comes from semantic interpretation; identity comes from exact document
    names/stems or verified entity/identifier mentions in corpus content. Retrieval is
    never consulted here and similarity never upgrades UNKNOWN/AMBIGUOUS.
    """
    targets = [dict(t) for t in (getattr(semantic, "targets", ()) or ()) if isinstance(t, dict)]
    if not targets:
        entities = list(getattr(semantic, "entities", ()) or ())
        variants = list(getattr(semantic, "variants", ()) or ())
        documents_list = list(getattr(semantic, "documents", ()) or ())
        width = max(len(entities), len(variants), len(documents_list), 0)
        targets = [{
            "entity": entities[i] if i < len(entities) else None,
            "variant": variants[i] if i < len(variants) else None,
            "topic": None,
            "document": documents_list[i] if i < len(documents_list) else None,
            "fact": None,
        } for i in range(width)]

    if not targets and reference and reference.get("resolved"):
        r = dict(reference["resolved"])
        targets = [{"entity": r.get("entity"), "variant": r.get("variant"), "topic": r.get("topic"), "document": r.get("document"), "fact": r.get("fact")}]

    results: list[VerifiedIdentity] = []
    for target in targets:
        entity = target.get("entity")
        variant = target.get("variant")
        document = target.get("document")
        candidates, basis = _candidate_documents(entity, variant, document, documents, topic=target.get("topic"), fact=target.get("fact"))
        if not candidates and reference and reference.get("resolved") and not any((entity, variant, document)):
            r = dict(reference["resolved"])
            entity, variant, document = r.get("entity"), r.get("variant"), r.get("document")
            candidates, basis = _candidate_documents(entity, variant, document, documents, topic=target.get("topic"), fact=target.get("fact"))
        if len(candidates) == 1:
            results.append(VerifiedIdentity("KNOWN", candidates[0], entity, variant, tuple(candidates), basis))
        elif len(candidates) > 1:
            results.append(VerifiedIdentity("AMBIGUOUS", None, entity, variant, tuple(candidates), "AMBIGUOUS"))
        elif entity or variant or document:
            results.append(VerifiedIdentity("UNKNOWN", None, entity, variant, (), "UNKNOWN"))
        else:
            results.append(VerifiedIdentity("NOT_APPLICABLE", None, None, None, (), "UNKNOWN"))
    return results


def _derive_candidate_entities(seed_entities: Iterable[str], candidate_documents: Iterable[str], documents: dict[str, str]) -> tuple[str, ...]:
    seeds = [str(x).strip() for x in seed_entities if str(x).strip()]
    if len(seeds) != 1:
        return tuple(dict.fromkeys(seeds))
    seed = seeds[0]
    labels: list[str] = []
    seed_norm = norm(seed)
    # Candidate identity labels are derived only from the discovered documents.
    # The extraction is deliberately conservative and generic: preserve a leading
    # capitalized name/title containing the semantic seed when one exists.
    for source in candidate_documents:
        text = documents.get(source, "")
        for line in text.splitlines()[:12]:
            line = line.strip()
            if not line or seed_norm not in norm(line):
                continue
            m = re.search(r"\b" + re.escape(seed) + r"(?:\s+[A-Z][A-Za-z0-9.'-]{1,}){0,1}", line, re.I)
            if m:
                label = m.group(0).strip(" -:,;.")
                if label and label.lower() != seed.lower():
                    labels.append(label)
                    break
    return tuple(dict.fromkeys(labels or seeds))


def resolve_semantic_request(semantic, *, turn_id: str, raw_query: str, conversation: dict[str, Any], documents: dict[str, str]):
    """Resolve meaning against authoritative conversation state and corpus identity."""
    from rag.semantic_contracts import ContextAction, ResolvedRequest, EntityResolution, KnowledgeBoundaryDecision
    from rag.semantic.context_binding import bind_reference

    current = dict(conversation or {})
    current["current_query"] = raw_query
    ref = bind_reference(semantic, current)
    resolved_ref = dict(ref.get("resolved") or {})
    resolved_ref_targets = [dict(t) for t in (ref.get("resolved_targets") or []) if isinstance(t, dict)]

    # Explicit current-turn dimensions normally win. A structured verification or
    # collective reference is the exception: if the current turn contains no explicit
    # identity, the authoritative active target set wins over any stale LLM guess.
    entities = list(semantic.entities or ())
    variants = list(semantic.variants or ())
    topics = list(semantic.topics or ())
    documents_list = list(semantic.documents or ())
    reference_types = {getattr(r, "reference_type", "") for r in (semantic.references or ())}
    has_explicit_identity_reference = bool(reference_types & {"PRONOMINAL_REFERENCE", "TARGET_REFERENCE", "CONTEXT_REFERENCE", "VERIFICATION_REFERENCE", "ORDINAL_SELECTION"})
    # A reference operation is authoritative over any semantic-model guesses that are
    # not explicitly present in the current user message. This is what prevents
    # "check again" from resurrecting an older target.
    reference_only_turn = has_explicit_identity_reference and not _query_terms(raw_query)

    pending = dict(current.get("pending_clarification") or {})
    pending_targets = [dict(t) for t in (pending.get("candidate_targets") or []) if isinstance(t, dict)]
    pending_resolved = False
    pending_override = _pending_resolution(raw_query, pending)
    operation = semantic.operation
    if pending_override:
        semantic_targets_override = [dict(t) for t in pending_override["targets"]]
        pending_resolved = True
        entities = [str(t.get("entity")) for t in semantic_targets_override if t.get("entity")]
        variants = [str(t.get("variant")) for t in semantic_targets_override if t.get("variant")]
        topics = [str(t.get("topic") or t.get("fact")) for t in semantic_targets_override if t.get("topic") or t.get("fact")]
        documents_list = [str(t.get("document")) for t in semantic_targets_override if t.get("document")]
        semantic_targets = semantic_targets_override
        operation = str(pending.get("original_operation") or pending.get("operation") or "factual_lookup")
    else:
        semantic_targets = None
    entity = entities[0] if len(entities) == 1 else None
    variant = variants[0] if len(variants) == 1 else None
    topic = topics[0] if len(topics) == 1 else None
    document = documents_list[0] if len(documents_list) == 1 else None

    # Resolve a pending clarification deterministically. The user's current explicit
    # identity selects from the pending candidates instead of creating a fresh topic.
    if pending_targets and (entities or variants or documents_list):
        current_tokens = {norm(x) for x in entities + variants + documents_list if x}
        matched_pending = []
        for pt in pending_targets:
            vals = [norm(str(pt.get(k) or "")) for k in ("entity", "variant", "document", "topic")]
            if any(v and (v in current_tokens or any(tok in v.split() for tok in current_tokens)) for v in vals):
                matched_pending.append(pt)
        if matched_pending:
            pending_resolved = True
            # Preserve the pending operation/topic while applying the explicit identity.
            entities = [str(matched_pending[0].get("entity") or entities[0])] if entities or matched_pending[0].get("entity") else []
            if matched_pending[0].get("document") and not documents_list:
                documents_list = [str(matched_pending[0]["document"])]
            if matched_pending[0].get("topic") and not topics:
                topics = [str(matched_pending[0]["topic"])]
            entity = entities[0] if len(entities) == 1 else entity
            variant = variants[0] if len(variants) == 1 else variant
            topic = topics[0] if len(topics) == 1 else topic
            document = documents_list[0] if len(documents_list) == 1 else document

    # A verification/collective reference without a current explicit identity must use
    # the authoritative bound target set, not semantic guesses from older conversation.
    if semantic_targets is None:
        if reference_only_turn and resolved_ref_targets:
            semantic_targets = [dict(t) for t in resolved_ref_targets]
        else:
            semantic_targets = [dict(t) for t in (semantic.targets or ()) if isinstance(t, dict)]

    # Build target set without collapsing plural semantics.
    if semantic_targets:
        targets = []
        for t in semantic_targets:
            x = dict(t)
            x["entity"] = x.get("entity") or (resolved_ref.get("entity") if len(semantic_targets) == 1 else None)
            x["variant"] = x.get("variant") or (resolved_ref.get("variant") if len(semantic_targets) == 1 else None)
            x["topic"] = x.get("topic") or topic
            x["document"] = x.get("document") or (resolved_ref.get("document") if len(semantic_targets) == 1 else None)
            x["fact"] = x.get("fact") or x.get("topic") or topic
            targets.append(x)
    else:
        if reference_only_turn or semantic.references or semantic.context_dependency in {"CONTEXT_DEPENDENT", "REQUIRED", "RESOLVED"}:
            entity = entity or resolved_ref.get("entity")
            variant = variant or resolved_ref.get("variant")
            topic = topic or resolved_ref.get("topic")
            document = document or resolved_ref.get("document")
        targets = [{"entity": entity, "variant": variant, "topic": topic, "document": document, "fact": topic}] if any((entity, variant, topic, document)) else []

    # A multi-target pronoun/collective reference binds to the entire authoritative set.
    if len(targets) <= 1 and resolved_ref_targets and not entity and not document:
        targets = []
        for t in resolved_ref_targets:
            x = dict(t)
            x["topic"] = topic or x.get("topic")
            x["fact"] = x.get("fact") or x.get("topic") or topic
            targets.append(x)

    # Discovery is separate from identity authorization. It produces a bounded candidate
    # set for ambiguity/clarification without granting those documents factual authority.
    if not targets and not documents_list and not reference_only_turn and not pending_resolved:
        query_candidates = _discover_from_current_query(raw_query, documents)
        if len(query_candidates) == 1:
            document = query_candidates[0]
            topic = topic or " ".join(_query_terms(raw_query)) or None
            targets = [{"entity": None, "variant": None, "topic": topic, "document": document, "fact": topic}]
        elif len(query_candidates) > 1:
            candidate_documents = query_candidates
        else:
            candidate_documents = []
    else:
        candidate_documents = []

    discovery_targets = targets or [{"entity": entity, "variant": variant, "topic": topic, "fact": topic, "document": document}]
    discovered_documents, discovery_basis = _discover_candidate_documents(
        discovery_targets, documents_list, documents
    )
    if not discovered_documents and "candidate_documents" in locals() and candidate_documents:
        discovered_documents = candidate_documents
        discovery_basis = "CURRENT_QUERY_DISCOVERY"
    candidate_documents = discovered_documents
    # Explicit semantic document identities are still verified below; discovery never
    # silently promotes an arbitrary corpus document to an authorized target.

    # Verify each target independently from corpus evidence.
    class _S: pass
    s = _S(); s.targets = tuple(targets); s.entities = tuple(entities); s.variants = tuple(variants); s.documents = tuple(documents_list)
    identities = verify_identity(s, documents, ref)

    final_targets: list[dict[str, Any]] = []
    unknown = False
    ambiguous = False
    known_docs: list[str] = []
    ambiguous_docs: list[str] = []
    entity_resolutions = []
    for idx, target in enumerate(targets):
        ident = identities[idx] if idx < len(identities) else VerifiedIdentity("UNKNOWN", entity=target.get("entity"), variant=target.get("variant"))
        x = dict(target)
        if ident.status == "KNOWN":
            x["document"] = ident.document
            if ident.entity and not x.get("entity"): x["entity"] = ident.entity
            if ident.variant and not x.get("variant"): x["variant"] = ident.variant
            known_docs.append(ident.document)
        elif ident.status == "AMBIGUOUS":
            ambiguous = True; ambiguous_docs.extend(ident.candidate_documents)
        elif ident.status == "UNKNOWN":
            unknown = True
        final_targets.append(x)
        entity_resolutions.append(EntityResolution(
            requested_identity=x.get("entity") or x.get("document"), resolved_identity=ident.document,
            entity_type=None, variant=x.get("variant"), document_ids=ident.candidate_documents,
            status=ident.status, ambiguity=ident.candidate_documents if ident.status == "AMBIGUOUS" else (),
            confidence=0.95 if ident.status == "KNOWN" else 0.25, resolution_basis=ident.resolution_basis,
        ))

    if len(final_targets) == 1:
        entity = final_targets[0].get("entity") or entity
        variant = final_targets[0].get("variant") or variant
        topic = final_targets[0].get("topic") or topic
        document = final_targets[0].get("document") or document
    elif len(final_targets) > 1:
        shared_topics = {str(t.get("topic")) for t in final_targets if t.get("topic")}
        entity = variant = document = None
        topic = next(iter(shared_topics)) if len(shared_topics) == 1 else topic

    # Once a concrete target is verified, semantic-model "insufficient" often means
    # only that the answer content is not yet known. That is NOT a reason to clarify;
    # retrieval is the mechanism that establishes the evidence.
    executable_target = bool(final_targets and known_docs)
    if semantic.knowledge_mode == "EXTERNAL":
        knowledge_status = "UNSUPPORTED"
    elif pending_resolved:
        knowledge_status = "AUTHORIZED"
    elif ref.get("status") == "AMBIGUOUS":
        knowledge_status = "CLARIFY"
    elif ambiguous or semantic.sufficiency == "AMBIGUOUS":
        knowledge_status = "CLARIFY"
    elif unknown and not candidate_documents:
        knowledge_status = "UNKNOWN_IDENTITY"
    elif executable_target:
        knowledge_status = "AUTHORIZED"
    elif semantic.sufficiency == "INSUFFICIENT" and candidate_documents:
        knowledge_status = "INSUFFICIENT"
    else:
        knowledge_status = "INSUFFICIENT"

    action = semantic.context_action
    if pending_resolved:
        action = "CONTINUE"
    if operation == "VERIFY" or has_explicit_identity_reference and any(getattr(r, "reference_type", "") == "VERIFICATION_REFERENCE" for r in (semantic.references or ())):
        action = "VERIFY"
    elif knowledge_status != "CLARIFY" and (executable_target or pending_resolved):
        action = "CONTINUE"
    if knowledge_status == "CLARIFY": action = "CLARIFY"
    try: context_action = ContextAction(action)
    except ValueError: context_action = ContextAction.RESET

    boundary = KnowledgeBoundaryDecision(
        mode=semantic.knowledge_mode, status=knowledge_status,
        required_authority="A3" if semantic.knowledge_mode == "APPLICATION" else "A0",
        allowed_sources=tuple(dict.fromkeys(known_docs)),
        prohibited_sources=("external_web", "pretrained_world_knowledge") if semantic.knowledge_mode == "APPLICATION" else (),
        corpus_required=semantic.knowledge_mode == "APPLICATION", external_source_required=False, reason=knowledge_status,
    )
    _ = boundary

    return ResolvedRequest(
        turn_id=turn_id, raw_query=raw_query,
        intent={"SMALLTALK":"smalltalk", "HELP":"help", "LIST_DOCUMENTS":"list_documents", "DOCUMENT_OVERVIEW":"factual_lookup", "FACTUAL_QUERY":"factual_lookup", "SUMMARY":"summarization", "VERIFY":"verification", "CLARIFY":"clarification"}.get(operation, "factual_lookup"),
        entity=entity, variant=variant, topic=topic, document=document,
        reference=", ".join(r.expression for r in semantic.references if r.expression) or None,
        reference_resolution=resolved_ref.get("document") or resolved_ref.get("entity") if ref.get("status") == "RESOLVED" else None,
        ambiguity=tuple(dict.fromkeys(tuple(semantic.ambiguity) + tuple(ambiguous_docs))),
        evidence_scope=tuple(dict.fromkeys(known_docs or ((document,) if document else ()))),
        verification_target=raw_query if operation == "VERIFY" else None,
        confidence=0.95 if knowledge_status == "AUTHORIZED" else 0.25 if knowledge_status in {"CLARIFY", "UNKNOWN_IDENTITY"} else 0.5,
        context_action=context_action, resolution_source=f"semantic_llm+{discovery_basis.lower()}",
        candidate_entities=_derive_candidate_entities(semantic.entities, tuple(dict.fromkeys(ambiguous_docs or candidate_documents)), documents), candidate_variants=tuple(semantic.variants),
        candidate_documents=tuple(dict.fromkeys(ambiguous_docs or known_docs or candidate_documents)), 
        targets=tuple(final_targets), target_count=semantic.target_count or len(final_targets),
        sufficiency=semantic.sufficiency, knowledge_status=knowledge_status, knowledge_mode=semantic.knowledge_mode,
        missing_information=tuple(semantic.missing_information),
        context_dependency=("RESOLVED" if pending_resolved or (ref.get("status") == "RESOLVED" and (semantic.references or resolved_ref_targets)) else semantic.context_dependency),
    )
