const chatScroll = document.getElementById("chat-scroll");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const reindexBtn = document.getElementById("reindex-btn");

const filterInput = document.getElementById("filter-input");
const filterApply = document.getElementById("filter-apply");
const filterClear = document.getElementById("filter-clear");
const filterStatus = document.getElementById("filter-status");

const clearChatBtn = document.getElementById("clear-chat");

function setHint(el, text, kind) {
  el.textContent = text;
  el.className = "hint" + (kind ? " " + kind : "");
}

function addMessage(role, text) {
  emptyState.style.display = "none";
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const label = document.createElement("div");
  label.className = "msg-role";
  label.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Assistant";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(label);
  wrap.appendChild(bubble);
  chatScroll.appendChild(wrap);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return bubble;
}

function addSources(afterBubble, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  sources.forEach((s) => {
    const tag = document.createElement("span");
    const page = s.page != null ? `p.${s.page}` : "";
    tag.textContent = `${s.source || "unknown"} ${page}`.trim();
    box.appendChild(tag);
  });
  afterBubble.parentElement.appendChild(box);
}

async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("stat-pdfs").textContent = data.pdfs_indexed ?? "–";
    document.getElementById("stat-chunks").textContent = data.chunks_indexed ?? "–";
    document.getElementById("stat-size").textContent =
      data.vector_store_size_mb != null ? `${data.vector_store_size_mb} MB` : "–";
    document.getElementById("stat-model").textContent = data.active_model || "–";
  } catch (e) {
    // Stats are non-critical; fail silently.
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    (data.history || []).forEach((qa) => {
      addMessage("user", qa.question);
      addMessage("ai", qa.answer);
    });
  } catch (e) {
    // ignore
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  sendBtn.disabled = true;

  const aiBubble = addMessage("ai", "");
  let fullText = "";
  let toolStatusShowing = false;

  // Friendly live status labels for each tool the agent can dynamically
  // choose to call - shown as soon as the LLM decides to call one, before
  // it actually runs, and cleared the moment real answer text starts.
  const TOOL_STATUS_LABELS = {
    search_documents: "🔍 Searching documents…",
    list_available_documents: "📄 Checking available documents…",
    ask_clarification: "❓ Thinking of a clarifying question…",
    summarize_document: "📝 Summarizing document…",
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      aiBubble.parentElement.remove();
      addMessage("error", err.error || `Request failed (${res.status})`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.tool_status) {
          toolStatusShowing = true;
          aiBubble.textContent = TOOL_STATUS_LABELS[msg.tool_status] || `⚙️ Running ${msg.tool_status}…`;
          aiBubble.style.opacity = "0.65";
          aiBubble.style.fontStyle = "italic";
          chatScroll.scrollTop = chatScroll.scrollHeight;
        } else if (msg.token) {
          if (toolStatusShowing) {
            // First real answer token - clear the status placeholder and
            // restore normal styling before appending actual content.
            toolStatusShowing = false;
            aiBubble.style.opacity = "";
            aiBubble.style.fontStyle = "";
          }
          fullText += msg.token;
          aiBubble.textContent = fullText;
          chatScroll.scrollTop = chatScroll.scrollHeight;
        } else if (msg.error) {
          aiBubble.textContent = msg.error;
          aiBubble.parentElement.classList.add("error");
        } else if (msg.done) {
          if (toolStatusShowing) {
            // Defensive: clear a lingering status placeholder even if no
            // answer tokens arrived before the stream ended.
            toolStatusShowing = false;
            aiBubble.style.opacity = "";
            aiBubble.style.fontStyle = "";
          }
          addSources(aiBubble, msg.sources);
          refreshStats();
        }
      }
    }
  } catch (e) {
    aiBubble.textContent = "Connection error. Is the server running?";
    aiBubble.parentElement.classList.add("error");
  } finally {
    sendBtn.disabled = false;
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setHint(uploadStatus, "Choose a PDF first.", "error");
    return;
  }
  setHint(uploadStatus, "Uploading & indexing…");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      setHint(uploadStatus, data.error || "Upload failed.", "error");
      return;
    }
    setHint(uploadStatus, `Indexed ${data.uploaded}.`, "success");
    fileInput.value = "";
    refreshStats();
  } catch (e) {
    setHint(uploadStatus, "Upload failed.", "error");
  }
});

reindexBtn.addEventListener("click", async () => {
  setHint(uploadStatus, "Reindexing…");
  try {
    const res = await fetch("/api/reindex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    const data = await res.json();
    setHint(uploadStatus, `Reindexed. ${data.added ?? 0} added.`, "success");
    refreshStats();
  } catch (e) {
    setHint(uploadStatus, "Reindex failed.", "error");
  }
});

filterApply.addEventListener("click", async () => {
  const value = filterInput.value.trim();
  try {
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filter: value }),
    });
    const data = await res.json();
    setHint(filterStatus, `Active: ${JSON.stringify(data.active_filter)}`, "success");
  } catch (e) {
    setHint(filterStatus, "Failed to apply filter.", "error");
  }
});

filterClear.addEventListener("click", async () => {
  try {
    await fetch("/api/filter", { method: "DELETE" });
    filterInput.value = "";
    setHint(filterStatus, "Filter cleared.", "success");
  } catch (e) {
    setHint(filterStatus, "Failed to clear filter.", "error");
  }
});

clearChatBtn.addEventListener("click", async () => {
  try {
    await fetch("/api/clear", { method: "POST" });
    chatScroll.innerHTML = "";
    chatScroll.appendChild(emptyState);
    emptyState.style.display = "block";
  } catch (e) {
    // ignore
  }
});

refreshStats();
loadHistory();
