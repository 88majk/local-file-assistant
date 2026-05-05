const els = {
  folderInput: document.querySelector("#folderInput"),
  pickFolderBtn: document.querySelector("#pickFolderBtn"),
  saveFolderBtn: document.querySelector("#saveFolderBtn"),
  scanBtn: document.querySelector("#scanBtn"),
  statusMessage: document.querySelector("#statusMessage"),
  progressBar: document.querySelector("#progressBar"),
  percentLabel: document.querySelector("#percentLabel"),
  docCounter: document.querySelector("#docCounter"),
  chatMessages: document.querySelector("#chatMessages"),
  chatForm: document.querySelector("#chatForm"),
  chatInput: document.querySelector("#chatInput"),
  tabChatBtn: document.querySelector("#tabChatBtn"),
  tabPreviewBtn: document.querySelector("#tabPreviewBtn"),
  chatTabPanel: document.querySelector("#chatTabPanel"),
  previewTabPanel: document.querySelector("#previewTabPanel"),
  refreshFilesBtn: document.querySelector("#refreshFilesBtn"),
  folderFilesInfo: document.querySelector("#folderFilesInfo"),
  folderFileList: document.querySelector("#folderFileList"),
  previewCard: document.querySelector("#previewCard"),
  previewTitle: document.querySelector("#previewTitle"),
  previewMeta: document.querySelector("#previewMeta"),
  previewSummary: document.querySelector("#previewSummary"),
  previewAttrs: document.querySelector("#previewAttrs"),
  openFileBtn: document.querySelector("#openFileBtn"),
};

let selectedResult = null;

function switchTab(tabName) {
  const isChat = tabName === "chat";
  els.tabChatBtn.classList.toggle("active", isChat);
  els.tabPreviewBtn.classList.toggle("active", !isChat);
  els.chatTabPanel.classList.toggle("active", isChat);
  els.previewTabPanel.classList.toggle("active", !isChat);

  if (!isChat) {
    loadFolderFiles();
  }
}

function addMessage(kind, rawText, isMarkdown = false) {
  const node = document.createElement("div");
  node.className = `msg ${kind}`;
  if (isMarkdown && window.marked) {
    node.innerHTML = window.marked.parse(rawText || "");
  } else {
    node.textContent = rawText;
  }
  els.chatMessages.appendChild(node);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  return node;
}

function addAiResponseWithResults(answerMarkdown, results) {
  const node = document.createElement("div");
  node.className = "msg ai";

  const answer = document.createElement("div");
  answer.className = "ai-answer";
  answer.innerHTML = window.marked
    ? window.marked.parse(answerMarkdown || "Brak odpowiedzi.")
    : answerMarkdown || "Brak odpowiedzi.";
  node.appendChild(answer);

  const list = document.createElement("div");
  list.className = "inline-result-list";

  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "inline-result-item";
    empty.textContent = "Brak wynikow.";
    list.appendChild(empty);
  } else {
    results.forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "inline-result-item inline-result-action";
      row.dataset.filepath = item.filepath || "";
      row.innerHTML = `
        <span class="result-name">${item.filename || "(bez nazwy)"}</span>
        <span class="result-meta">Trafnosc: ${(item.score || 0).toFixed(3)} | Typ: ${item.document_type || "Brak"}</span>
      `;
      list.appendChild(row);
    });
  }

  node.appendChild(list);
  els.chatMessages.appendChild(node);
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function renderPreview(item) {
  els.previewCard.classList.remove("hidden");
  els.previewTitle.textContent = item.filename || "(bez nazwy)";
  els.previewMeta.textContent = `${item.document_type || "Brak typu"} | score ${(
    item.score || 0
  ).toFixed(4)}`;
  els.previewSummary.textContent = item.summary || "Brak podsumowania.";

  els.previewAttrs.innerHTML = "";
  const attrs = item.attributes || {};
  const entries = Object.entries(attrs);
  if (!entries.length) {
    const li = document.createElement("li");
    li.textContent = "Brak atrybutow";
    els.previewAttrs.appendChild(li);
    return;
  }

  entries.forEach(([k, v]) => {
    const li = document.createElement("li");
    li.textContent = `${k}: ${v}`;
    els.previewAttrs.appendChild(li);
  });
}

async function api(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function refreshFolder() {
  const payload = await api("/api/folder");
  els.folderInput.value = payload.folder_path || "";
}

function renderFolderFiles(files) {
  els.folderFileList.innerHTML = "";

  if (!files.length) {
    const li = document.createElement("li");
    li.className = "inline-result-item";
    li.textContent = "Brak plikow w podlaczonym folderze.";
    els.folderFileList.appendChild(li);
    return;
  }

  files.forEach((file) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "folder-file-item";
    button.dataset.filepath = file.filepath || "";
    button.innerHTML = `
      <div class="result-name">${file.name || "(bez nazwy)"}</div>
      <div class="folder-file-meta">${(file.size_kb ?? 0).toFixed(2)} KB</div>
    `;
    li.appendChild(button);
    els.folderFileList.appendChild(li);
  });
}

async function loadFolderFiles() {
  try {
    const payload = await api("/api/files");
    els.folderFilesInfo.textContent = `Folder: ${payload.folder_path} | Pliki: ${payload.count}`;
    renderFolderFiles(payload.files || []);
  } catch (err) {
    els.folderFilesInfo.textContent = `Blad listowania plikow: ${err.message}`;
    els.folderFileList.innerHTML = "";
  }
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    const percent = Number(status.percent || 0);

    els.statusMessage.textContent = status.message || "-";
    els.percentLabel.textContent = `${percent}%`;
    els.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    els.docCounter.textContent = `Dokumenty: ${status.documents_total || 0}`;
    els.scanBtn.disabled = Boolean(status.is_running);
  } catch (err) {
    els.statusMessage.textContent = `Blad statusu: ${err.message}`;
  }
}

async function loadPreview(filepath) {
  if (!filepath) {
    return;
  }

  try {
    const payload = await api(`/api/document?filepath=${encodeURIComponent(filepath)}`);
    selectedResult = payload;
    renderPreview(payload);
  } catch (err) {
    addMessage("ai", `Nie udalo sie pobrac podgladu: ${err.message}`);
  }
}

els.pickFolderBtn.addEventListener("click", async () => {
  try {
    const payload = await api("/api/pick-folder", { method: "POST" });
    els.folderInput.value = payload.folder_path || "";
    if (payload.cancelled) {
      addMessage("ai", "Anulowano wybor folderu.");
      return;
    }
    addMessage("ai", `Wybrano folder: ${payload.folder_path}`);
    loadFolderFiles();
  } catch (err) {
    addMessage("ai", `Nie udalo sie otworzyc pickera folderu: ${err.message}`);
  }
});

els.saveFolderBtn.addEventListener("click", async () => {
  const folderPath = els.folderInput.value.trim();
  if (!folderPath) {
    addMessage("ai", "Podaj sciezke folderu.");
    return;
  }
  try {
    await api("/api/folder", {
      method: "POST",
      body: JSON.stringify({ folder_path: folderPath }),
    });
    addMessage("ai", `Ustawiono folder: ${folderPath}`);
    loadFolderFiles();
  } catch (err) {
    addMessage("ai", `Nie udalo sie ustawic folderu: ${err.message}`);
  }
});

els.scanBtn.addEventListener("click", async () => {
  try {
    await api("/api/scan", { method: "POST" });
    addMessage("ai", "Skanowanie zostalo uruchomione.");
    await refreshStatus();
  } catch (err) {
    addMessage("ai", `Blad skanowania: ${err.message}`);
  }
});

els.chatForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const query = els.chatInput.value.trim();
  if (!query) {
    return;
  }

  addMessage("user", query);
  els.chatInput.value = "";

  const loading = addMessage("ai loading", "AI pisze...");
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: 8, use_embeddings: true }),
    });

    loading.remove();
    addAiResponseWithResults(payload.answer_markdown || "Brak odpowiedzi.", payload.results || []);
  } catch (err) {
    loading.remove();
    addMessage("ai", `Blad wyszukiwania: ${err.message}`);
  }
});

els.chatMessages.addEventListener("click", (ev) => {
  const action = ev.target.closest(".inline-result-action");
  if (!action) {
    return;
  }
  const filepath = action.dataset.filepath || "";
  loadPreview(filepath);
});

els.folderFileList.addEventListener("click", (ev) => {
  const action = ev.target.closest(".folder-file-item");
  if (!action) {
    return;
  }
  const filepath = action.dataset.filepath || "";
  loadPreview(filepath);
});

els.tabChatBtn.addEventListener("click", () => switchTab("chat"));
els.tabPreviewBtn.addEventListener("click", () => switchTab("preview"));
els.refreshFilesBtn.addEventListener("click", loadFolderFiles);

els.chatInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    els.chatForm.requestSubmit();
  }
});

els.openFileBtn.addEventListener("click", async () => {
  if (!selectedResult?.filepath) {
    return;
  }

  try {
    await api("/api/open-file", {
      method: "POST",
      body: JSON.stringify({ filepath: selectedResult.filepath }),
    });
  } catch (err) {
    addMessage("ai", `Nie mozna otworzyc pliku: ${err.message}`);
  }
});

async function boot() {
  addMessage("ai", "Interfejs uruchomiony. Mozesz rozpoczac wyszukiwanie.");
  await refreshFolder();
  await refreshStatus();
  await loadFolderFiles();
  window.setInterval(refreshStatus, 2000);
}

boot().catch((err) => {
  addMessage("ai", `Blad inicjalizacji: ${err.message}`);
});
