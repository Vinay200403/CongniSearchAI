const statusEl = document.getElementById("status");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadResult = document.getElementById("upload-result");
const docList = document.getElementById("doc-list");
const queryForm = document.getElementById("query-form");
const questionInput = document.getElementById("question-input");
const answerBox = document.getElementById("answer-box");

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusEl.textContent = `backend: ${data.embedding_backend} · ${data.generation_enabled ? "generation on" : "retrieval-only"} · ${data.documents_indexed} docs indexed`;
    statusEl.classList.add("ok");
  } catch (e) {
    statusEl.textContent = "backend unreachable";
  }
}

async function loadDocuments() {
  const res = await fetch("/api/documents");
  const data = await res.json();
  docList.innerHTML = "";

  if (!data.documents.length) {
    docList.innerHTML = `<li style="justify-content:center;color:var(--muted)">No documents indexed yet</li>`;
    return;
  }

  data.documents.forEach((doc) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${doc.filename} · ${doc.chunks} chunks</span>
      <button class="remove" data-id="${doc.doc_id}">remove</button>
    `;
    docList.appendChild(li);
  });

  docList.querySelectorAll(".remove").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/documents/${btn.dataset.id}`, { method: "DELETE" });
      await loadDocuments();
      await checkHealth();
    });
  });
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  uploadResult.textContent = "Indexing…";
  uploadResult.className = "result";

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadResult.innerHTML = `<span class="error-text">${data.error}</span>`;
      return;
    }

    uploadResult.textContent = `Indexed "${data.filename}" — ${data.chunks_indexed} chunks.`;
    fileInput.value = "";
    await loadDocuments();
    await checkHealth();
  } catch (err) {
    uploadResult.innerHTML = `<span class="error-text">Upload failed: ${err}</span>`;
  }
});

queryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  answerBox.innerHTML = `<p class="result">Searching…</p>`;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    if (!res.ok) {
      answerBox.innerHTML = `<span class="error-text">${data.error}</span>`;
      return;
    }

    const sourcesHtml = (data.sources || [])
      .map((s) => {
        const pct = Math.round(s.score * 100);
        return `
          <div class="source">
            <div class="source-meta">
              <span>${s.filename} · chunk ${s.chunk_index}</span>
              <span>${pct}% match</span>
            </div>
            <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
            <div>${s.excerpt}${s.excerpt.length >= 300 ? "…" : ""}</div>
          </div>
        `;
      })
      .join("");

    answerBox.innerHTML = `
      <div class="answer-card">
        <span class="answer-mode">${data.mode.replace("-", " ")}</span>
        <div>${data.answer.replace(/\n/g, "<br>")}</div>
        ${sourcesHtml}
      </div>
    `;
  } catch (err) {
    answerBox.innerHTML = `<span class="error-text">Query failed: ${err}</span>`;
  }
});

checkHealth();
loadDocuments();
