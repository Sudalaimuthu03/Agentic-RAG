from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, session, stream_with_context

from config import APP_VERSION, settings
from rag.agent.graph import run_agent_turn, validate_final
from rag.agent.tools import build_tools, get_last_documents, indexed_sources, reset_last_documents, get_rag_retriever
from rag.core import pipeline
from rag.core.state import state
from rag.core.validation import validate_question
from rag.execution.gate import ExecutionBudget, ExecutionGate
from rag.memory.history import clear_session_history, get_session_history
from rag.semantic.catalog import build_catalog
from rag.semantic.understanding import understand
from rag.semantic.verification import resolve_semantic_request
from rag.semantic.planning import make_plan
from rag.retrieval.filter import parse_filter_command
from rag.semantic_contracts import ContextAction, EvidenceStatus, ResolvedRequest
from rag.utils.context import new_id, set_active_filter, set_request_id, set_session_id
from rag.utils.logger import get_logger, log_event
from rag.core.model_provider import configured_model_name, configured_provider

logger = get_logger("api")
bp = Blueprint("api", __name__, url_prefix="/api")


def _session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex[:12]
        session.permanent = True
    return session["session_id"]


@bp.before_app_request
def _tag_request():
    rid = new_id(); sid = _session_id()
    set_request_id(rid); set_session_id(sid); set_active_filter(state.get_filter(sid))
    request.request_id = rid; request.session_id = sid


def _documents_text() -> dict[str, str]:
    out = {}
    for source, doc in (state.doc_store or {}).items():
        name = doc.metadata.get("source", source)
        out[name] = out.get(name, "") + "\n" + doc.page_content
    return out


@bp.route("/chat", methods=["POST"])
def chat():
    sid = request.session_id; rid = request.request_id
    data = request.get_json(silent=True) or {}
    raw = data.get("question", "")
    cleaned, reason = validate_question(raw, settings.MAX_QUESTION_LENGTH)
    if reason:
        return jsonify({"error": f"Invalid question ({reason})"}), 400
    if state.llm is None:
        return jsonify({"error": "The local model is not ready."}), 503

    def generate():
        set_request_id(rid); set_session_id(sid); set_active_filter(state.get_filter(sid))
        t0 = time.perf_counter()
        history = get_session_history(sid)
        sources = indexed_sources()
        documents = _documents_text()
        conv = state.get_conversation_state(sid)
        # The semantic model receives only the authoritative current-state surface.
        # Historical resolution/audit records are intentionally excluded because they
        # are not evidence and must never seed stale entities into the current turn.
        semantic_conversation = {
            "active_targets": list(conv.active_targets),
            "candidate_targets": list(conv.candidate_targets),
            "comparison_set": list(conv.comparison_set),
            "focused_target": conv.focused_target,
            "focused_topic": conv.focused_topic,
            "active_documents": list(conv.active_documents),
            "presented_items": list(conv.presented_items),
            "last_successful_target": conv.last_successful_target,
            "last_successful_target_set": list(conv.last_successful_target_set),
            "last_operation": conv.last_operation,
            "last_topic": conv.last_topic,
            "pending_clarification": conv.pending_clarification,
            "unresolved_reference": conv.unresolved_reference,
            "state_version": conv.state_version,
            "current_query": cleaned,
        }
        budget = ExecutionBudget(
            max_agent_iterations=settings.MAX_AGENT_ITERATIONS,
            max_llm_calls=settings.MAX_LLM_CALLS_PER_TURN,
            max_retrieval_operations=settings.MAX_RETRIEVAL_OPERATIONS_PER_TURN,
            max_guardrail_retries=settings.MAX_GUARDRAIL_RETRIES,
        )

        # Stage 1: semantic interpretation. This is the single semantic entry point.
        budget.consume_llm()
        semantic = understand(state.llm, cleaned, catalog=build_catalog(documents), conversation=semantic_conversation)
        log_event(logger, "INFO", "semantic.interpretation", request_id=rid, interpretation=semantic.as_dict())
        log_event(logger, "INFO", "semantic.candidates", request_id=rid, candidate_entities=list(semantic.entities), candidate_documents=list(semantic.documents), target_count=semantic.target_count)

        # Stage 2-4: closed-world decision, corpus identity verification, and reference binding.
        resolved = resolve_semantic_request(
            semantic,
            turn_id=rid,
            raw_query=cleaned,
            conversation=semantic_conversation,
            documents=documents,
        )
        state.set_query_target(sid, resolved.as_dict())
        log_event(logger, "INFO", "query.resolved", resolved_request=resolved.as_dict())
        log_event(logger, "INFO", "scope.resolved", candidate_documents=list(resolved.candidate_documents), evidence_scope=list(resolved.evidence_scope), targets=list(resolved.targets))

        # Stage 5: explicit execution authority.
        plan = make_plan(resolved)
        log_event(logger, "INFO", "execution.plan", plan=plan.__dict__)
        answer = ""
        docs = []
        evidence_items = []
        ok = False
        bundle = None
        presented_items = None
        activity_events: list[str] = []

        # The ExecutionPlan is the sole runtime branch authority. There is no semantic
        # fallback here: every response mode is represented by the plan.
        if plan.execution_mode == "DOCUMENT_GROUNDED" and state.hybrid_retriever is None:
            answer = "The document index is not ready yet, so I cannot perform this document operation."
        elif resolved.intent == "list_documents":
            reset_last_documents()
            tools = build_tools(get_rag_retriever(resolved), resolved, indexed_sources)
            answer, _, evidence_items = run_agent_turn(state.llm, tools, resolved, plan, list(history.messages), budget, activity_events=activity_events)
            presented_items = [{"position": i + 1, "document": name} for i, name in enumerate(sources)]
        else:
            reset_last_documents()
            tools = build_tools(get_rag_retriever(resolved), resolved, indexed_sources) if state.hybrid_retriever is not None or plan.execution_mode != "DOCUMENT_GROUNDED" else build_tools(None, resolved, indexed_sources)
            answer, _, evidence_items = run_agent_turn(state.llm, tools, resolved, plan, list(history.messages), budget, activity_events=activity_events)
            docs = get_last_documents()

            if plan.execution_mode == "DOCUMENT_GROUNDED":
                ok, problems, bundle = validate_final(resolved, answer, evidence_items)
                log_event(logger, "INFO", "evidence.sufficiency", status=bundle.status.value, target_statuses=bundle.target_statuses, conflicts=list(bundle.conflicts), evidence_count=len(bundle.items), reason=bundle.reason)
                log_event(logger, "INFO", "answer.validation", valid=ok, problems=problems)
                if not ok and budget.guardrail_retries < budget.max_guardrail_retries and budget.allow_llm():
                    budget.guardrail_retries += 1
                    budget.consume_llm()
                    context = "\n\n".join(i.text for i in evidence_items)[:12000]
                    from langchain_core.messages import SystemMessage, HumanMessage
                    repair_messages = [
                        SystemMessage(content="You are the final answer-validation generator. Follow the authoritative execution plan. Use ONLY supplied evidence. Do not invent facts, targets, identities, or sources."),
                        SystemMessage(content=json.dumps({"execution_mode": plan.execution_mode, "targets": list(plan.targets), "operation": plan.operation}, ensure_ascii=False)),
                        HumanMessage(content=f"Question: {cleaned}\nEvidence:\n{context}\nPrevious answer:\n{answer}\nValidation problems: {problems}\nRewrite only the answer."),
                    ]
                    repaired = state.llm.invoke(repair_messages)
                    answer = str(getattr(repaired, "content", "") or "").strip() or answer
                    ok, problems, bundle = validate_final(resolved, answer, evidence_items)
                    log_event(logger, "INFO", "answer.validation.recheck", valid=ok, problems=problems)
                if bundle.status == EvidenceStatus.UNKNOWN and not evidence_items:
                    answer = "The indexed documents do not contain enough evidence to answer that reliably."
                    ok = True
                elif bundle.status == EvidenceStatus.CONFLICTING:
                    log_event(logger, "WARNING", "answer.conflicting", sources=list(bundle.conflicts))
                    # A conflict is answerable only when the answer explicitly acknowledges it.
                    ok, problems = validate_final(resolved, answer, evidence_items)[:2]
                if not ok:
                    # Validation failure is terminal for factual output. Do not leak the
                    # invalid LLM answer after a failed evidence/claim gate.
                    answer = "I couldn't verify that answer reliably from the indexed documents."

        # Conversation state is authoritative. Successful grounded turns promote their
        # targets to the verification/continuation anchor; clarification turns preserve
        # the last successful target and record a pending clarification instead of
        # destroying context.
        evidence_payload = [item.as_dict() for item in evidence_items]
        turn_success = bool(
            resolved.intent == "list_documents"
            or (plan.execution_mode == "DOCUMENT_GROUNDED" and evidence_items and bundle.status in {EvidenceStatus.SUPPORTED, EvidenceStatus.MULTI_SOURCE_AGREEMENT, EvidenceStatus.CONTRADICTORY, EvidenceStatus.PARTIAL} and ok)
            or (plan.execution_mode == "CONVERSATIONAL" and answer)
        ) if plan.execution_mode != "REFUSAL" else False
        state.update_conversation_state(sid, resolved, presented_items, evidence=evidence_payload, success=turn_success)
        if resolved.document:
            state.set_last_resolved_document(sid, resolved.document)
        if resolved.evidence_scope:
            state.set_active_documents(sid, list(resolved.evidence_scope))

        history.add_user_message(cleaned); history.add_ai_message(answer)
        state.add_qa(sid, cleaned, answer)
        duration = round((time.perf_counter() - t0) * 1000)
        source_payload = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "rerank_score": d.metadata.get("rerank_score")}
            for d in docs
        ] if state.show_sources else []
        log_event(logger, "INFO", "chat.qa", question=cleaned, answer=answer, sources=len(source_payload), duration_ms=duration, budget=budget.__dict__)
        for event in activity_events if "activity_events" in locals() else []:
            yield json.dumps({"tool_status": event}, ensure_ascii=False) + "\n"
        yield json.dumps({"token": answer}, ensure_ascii=False) + "\n"
        yield json.dumps({
            "done": True,
            "sources": source_payload,
            "duration_ms": duration,
            "resolved_request": resolved.as_dict(),
            "semantic_interpretation": semantic.as_dict(),
            "execution_plan": {
                "operation": plan.operation,
                "targets": list(plan.targets),
                "tools": list(plan.tools),
                "search_scope": list(plan.search_scope),
                "document_scope": list(plan.document_scope),
                "clarification_required": plan.clarification_required,
                "execution_mode": plan.execution_mode,
                "target_requirements": list(plan.target_requirements),
            },
        }, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")

@bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part."}), 400
    f = request.files["file"]
    name = Path(f.filename or "").name
    if not name.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF uploads are supported."}), 400
    dest = Path(settings.UPLOAD_FOLDER) / name
    f.save(dest)
    if dest.stat().st_size > settings.MAX_PDF_SIZE_MB * 1024 * 1024:
        dest.unlink(missing_ok=True)
        return jsonify({"error": "PDF exceeds configured size limit."}), 413
    return jsonify({"uploaded": name, "reindex": pipeline.reindex_all(settings, force=False)})


@bp.route("/reindex", methods=["POST"])
def reindex():
    force = bool((request.get_json(silent=True) or {}).get("force", True))
    return jsonify(pipeline.reindex_all(settings, force=force))


@bp.route("/stats")
def stats():
    return jsonify({"pdfs_indexed": state.num_pdfs, "chunks_indexed": state.num_chunks, "vector_store_size_mb": state.vector_store_size_mb, "active_model": state.current_model or configured_model_name(settings), "active_filter": state.get_filter(request.session_id)})


@bp.route("/history")
def history(): return jsonify({"history": state.get_qa_history(request.session_id)})


@bp.route("/clear", methods=["POST"])
def clear():
    clear_session_history(request.session_id); return jsonify({"status": "cleared"})


@bp.route("/index", methods=["DELETE"])
def delete_index():
    pipeline.delete_index(settings); return jsonify({"status": "deleted"})


@bp.route("/config")
def config(): return jsonify(settings.as_dict())


@bp.route("/version")
def version(): return jsonify({"version": APP_VERSION})


@bp.route("/health")
def health(): return jsonify({"status": "ok", "model": state.current_model, "index_ready": state.hybrid_retriever is not None})


@bp.route("/models")
def models():
    return jsonify({
        "provider": configured_provider(settings),
        "installed": pipeline.get_installed_ollama_models() if not settings.USE_NVIDIA else [],
        "active": state.current_model or configured_model_name(settings),
    })


@bp.route("/model", methods=["POST"])
def switch_model():
    name = str((request.get_json(silent=True) or {}).get("model", "")).strip()
    if not name: return jsonify({"error": "Missing model."}), 400
    pipeline.switch_model(settings, name); return jsonify({"active": state.current_model})


@bp.route("/filter", methods=["GET"])
def get_filter(): return jsonify({"active_filter": state.get_filter(request.session_id)})

@bp.route("/filter", methods=["POST"])
def set_filter():
    parsed = parse_filter_command(str((request.get_json(silent=True) or {}).get("filter", "")))
    state.set_filter(request.session_id, parsed); return jsonify({"active_filter": parsed})

@bp.route("/filter", methods=["DELETE"])
def clear_filter():
    state.reset_filter(request.session_id); return jsonify({"active_filter": {}})
