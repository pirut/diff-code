const state = {
  history: [],
};

const chatLog = document.getElementById("chatLog");
const promptPreview = document.getElementById("promptPreview");
const loadedFiles = document.getElementById("loadedFiles");
const warnings = document.getElementById("warnings");
const modelName = document.getElementById("modelName");
const configPath = document.getElementById("configPath");
const runtimeMeta = document.getElementById("runtimeMeta");
const startupWarnings = document.getElementById("startupWarnings");
const messageInput = document.getElementById("messageInput");
const draftTemplate = document.getElementById("draftTemplate");
const codeContext = document.getElementById("codeContext");
const filePaths = document.getElementById("filePaths");
const steps = document.getElementById("steps");
const maskSpan = document.getElementById("maskSpan");
const temperature = document.getElementById("temperature");
const topK = document.getElementById("topK");
const topP = document.getElementById("topP");
const sendButton = document.getElementById("sendButton");
const clearChat = document.getElementById("clearChat");
const busyIndicator = document.getElementById("busyIndicator");

async function boot() {
  const response = await fetch("/api/state");
  const payload = await response.json();
  modelName.textContent = payload.model_name;
  configPath.textContent = payload.config_path;
  runtimeMeta.textContent = `device=${payload.device} dtype=${payload.runtime_dtype}`;
  steps.value = payload.default_steps;
  maskSpan.value = payload.default_mask_span;
  if ((payload.startup_warnings || []).length > 0) {
    startupWarnings.hidden = false;
    startupWarnings.textContent = payload.startup_warnings.join("\n\n");
  }
  renderEmptyState();
}

function renderEmptyState() {
  if (state.history.length > 0) return;
  chatLog.innerHTML = "";
  appendMessage(
    "assistant",
    "Describe the change you want. Leave Draft Template blank to auto-generate a scaffold, or provide your own draft with [MASK:n]."
  );
}

function appendMessage(role, content) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const speaker = document.createElement("div");
  speaker.className = "speaker";
  speaker.textContent = role === "user" ? "You" : "Model";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.appendChild(speaker);
  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  busyIndicator.hidden = !isBusy;
}

async function handleSend() {
  const message = messageInput.value.trim();
  if (!message) return;

  const priorHistory = [...state.history];
  appendMessage("user", message);
  state.history.push({ role: "user", content: message });
  messageInput.value = "";
  setBusy(true);

  const payload = {
    message,
    history: priorHistory,
    draft_template: draftTemplate.value,
    code_context: codeContext.value,
    file_paths_text: filePaths.value,
    steps: Number(steps.value),
    mask_span_tokens: Number(maskSpan.value),
    temperature: Number(temperature.value),
    top_k: Number(topK.value),
    top_p: Number(topP.value),
  };

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    const assistantText = result.ok
      ? (result.response && result.response.trim()
          ? result.response
          : "Model returned an empty completion. Try a smaller request, a larger mask span, or provide more code context.")
      : `Request failed: ${result.error || result.detail || `HTTP ${response.status}`}`;
    appendMessage("assistant", assistantText);
    state.history.push({ role: "assistant", content: assistantText });
    promptPreview.textContent = result.prompt || "";
    loadedFiles.textContent = (result.loaded_files || []).join("\n") || "No repo files loaded.";
    warnings.textContent = (result.warnings || []).join("\n") || (result.ok ? "No warnings." : assistantText);
  } catch (error) {
    appendMessage("assistant", `Request failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

sendButton.addEventListener("click", handleSend);
clearChat.addEventListener("click", () => {
  state.history = [];
  promptPreview.textContent = "";
  loadedFiles.textContent = "No repo files loaded.";
  warnings.textContent = "No warnings.";
  renderEmptyState();
});
messageInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    handleSend();
  }
});

boot();
