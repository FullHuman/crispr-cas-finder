// --- Worker setup ---
let worker;
let msgId = 0;
const pending = new Map(); // id → { resolve, reject, onProgress? }

function createWorker() {
  worker = new Worker("worker.js", { type: "module" });
  worker.onerror = (event) => {
    event.preventDefault();
    failWorker(new Error(event.message || "Analysis worker failed to load or crashed."));
  };
  worker.onmessageerror = () => failWorker(new Error("Unable to read the analysis worker response."));
  worker.onmessage = (e) => {
  const { id, result, error, progress } = e.data;
  const p = pending.get(id);
  if (!p) return;
  if (progress) {
    if (p.onProgress) p.onProgress(progress);
    return; // don't resolve yet
  }
  pending.delete(id);
  if (error) p.reject(new Error(error));
  else p.resolve(result);
  };
}

function failWorker(error) {
  worker.terminate();
  worker = null;
  for (const request of pending.values()) request.reject(error);
  pending.clear();
  setStatus(`Analysis failed: ${error.message}`, "error");
}

function callWorker(type, payload, onProgress) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    try {
      if (!worker) createWorker();
      pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ type, id, payload });
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
}

// --- DOM elements ---
const fastaInput = document.getElementById("fasta-input");
const fastaFileInput = document.getElementById("fasta-file");
const fastaZone = document.getElementById("fasta-zone");
const fastaDrop = document.getElementById("fasta-drop");
const fastaAccepted = document.getElementById("fasta-accepted");
const fastaRejected = document.getElementById("fasta-rejected");
const fastaRemove = document.getElementById("fasta-remove");
const fastaRetry = document.getElementById("fasta-retry");
const fileName = document.getElementById("file-name");
const fileDetail = document.getElementById("file-detail");
const rejectedName = document.getElementById("rejected-name");
const analyzeBtn = document.getElementById("analyze-btn");
const downloadBtn = document.getElementById("download-btn");

const minDr = document.getElementById("min-dr");
const maxDr = document.getElementById("max-dr");
const minSp = document.getElementById("min-sp");
const maxSp = document.getElementById("max-sp");
const levelMin = document.getElementById("level-min");
const minNbSpacers = document.getElementById("min-nb-spacers");
const noMism = document.getElementById("no-mism");
const enableCas = document.getElementById("enable-cas");
const metagenome = document.getElementById("metagenome");

const statusPanel = document.getElementById("status-panel");
const resultsList = document.getElementById("results-list");
const resultsEmpty = document.getElementById("results-empty");
const resultSummary = document.getElementById("result-summary");

// Option name → its field and the label shown above it.
const optionFields = {
  min_repeat_length: [minDr, "Minimum repeat length"],
  max_repeat_length: [maxDr, "Maximum repeat length"],
  min_spacer_length: [minSp, "Minimum spacer length"],
  max_spacer_length: [maxSp, "Maximum spacer length"],
  min_evidence_level: [levelMin, "Minimum evidence level"],
  min_spacer_count: [minNbSpacers, "Minimum spacer count"],
};

let lastResults = null;
let running = false;

function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = String(str);
  return el.innerHTML;
}

function count(n, noun) {
  return `${n.toLocaleString()} ${noun}${n === 1 ? "" : "s"}`;
}

// --- Status helpers ---
const ALERT_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>`;

function setStatus(message, kind = "note") {
  statusPanel.innerHTML = kind === "error"
    ? `<p class="lr-alert has-icon" role="alert">${ALERT_ICON}<span>${escapeHtml(message)}</span></p>`
    : `<p class="lr-note">${escapeHtml(message)}</p>`;
}

function setProgress({ message, current, total }) {
  const hasBar = total > 1;
  let html = `<div class="lr-progress-head">
      <p><span class="lr-dot is-running" aria-hidden="true"></span><span>${escapeHtml(message)}</span></p>
      ${hasBar ? `<p class="label-mono">${current} of ${total}</p>` : ""}
    </div>`;
  if (hasBar) {
    html += `<div class="lr-progress" role="progressbar" aria-label="${escapeHtml(message)}" aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${current}"><div style="width:${(100 * current) / total}%"></div></div>`;
  }
  statusPanel.innerHTML = html;
}

function setResult(title, text, hasResults) {
  statusPanel.innerHTML = `<div class="lr-result">
      <p>${escapeHtml(title)}</p>
      <p>${escapeHtml(text)}</p>
      ${hasResults ? `<a class="link" href="#results">View the results →</a>` : ""}
    </div>`;
}

function setIdleStatus() {
  setStatus("Runs in this tab. Keep it open until the detection finishes.");
}

function updateRunButton() {
  if (running) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Detecting…";
  } else if (!/\S/.test(fastaInput.value)) {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Add a sequence to continue";
  } else {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = enableCas.checked ? "Detect arrays and Cas systems" : "Detect CRISPR arrays";
  }
}

// --- File drop ---
function showFileState(state) {
  fastaDrop.hidden = state !== "empty";
  fastaAccepted.hidden = state !== "accepted";
  fastaRejected.hidden = state !== "rejected";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function readFastaFile(file) {
  let text;
  try {
    text = await file.text();
  } catch (error) {
    setStatus(`The file could not be read: ${error.message}`, "error");
    return;
  }
  if (!text.trimStart().startsWith(">")) {
    rejectedName.textContent = file.name;
    showFileState("rejected");
    return;
  }
  fastaInput.value = text;
  fileName.textContent = file.name;
  const sequences = (text.match(/^>/gm) || []).length;
  fileDetail.textContent = `${count(sequences, "sequence")} · ${formatBytes(file.size)} · read locally`;
  showFileState("accepted");
  setIdleStatus();
  updateRunButton();
}

function removeFastaFile() {
  fastaFileInput.value = "";
  fastaInput.value = "";
  showFileState("empty");
  updateRunButton();
}

// --- Result builders ---
function buildCrisprItems(arrays) {
  return arrays.map((array, idx) => {
    const spacerCount = array.spacers ? array.spacers.length : 0;
    const orientationMap = { Forward: "+", Reverse: "-", Unknown: "Not determined" };
    // Prefer the CRISPRDirection DB value (crispr_direction) over the AT%-based
    // orientation heuristic — the DB lookup matches the original CRISPRCasFinder.
    const dir = array.crispr_direction && array.crispr_direction !== "ND"
      ? array.crispr_direction
      : (orientationMap[array.orientation] || "Not determined");
    const drLen = array.consensus_repeat ? array.consensus_repeat.length : 0;
    return {
      type: "crispr",
      title: array.id || `${array.seq_id}_CRISPR_${idx + 1}`,
      subtitle: `${array.start.toLocaleString()} – ${array.end.toLocaleString()} · ${count(spacerCount, "spacer")}`,
      level: array.evidence_level,
      start: array.start,
      end: array.end,
      detail: {
        start: array.start,
        end: array.end,
        drConsensus: array.consensus_repeat || "—",
        drLength: drLen,
        spacerCount: spacerCount,
        direction: dir,
        evidenceLevel: array.evidence_level ?? "—",
        repeats: array.repeats || [],
        spacers: array.spacers || [],
      },
    };
  });
}

function buildCasItems(casClusters) {
  return casClusters.map((cluster) => {
    const genes = cluster.genes || [];
    const geneNames = genes.map((g) => g.name).join(", ");
    return {
      type: "cas",
      title: cluster.system,
      subtitle: `${cluster.start.toLocaleString()} – ${cluster.end.toLocaleString()} · ${count(genes.length, "gene")}`,
      start: cluster.start,
      end: cluster.end,
      detail: {
        start: cluster.start,
        end: cluster.end,
        geneCount: genes.length,
        geneNames: geneNames || "—",
        genes: genes,
      },
    };
  });
}

// --- Detail panel builders ---
function buildFacts(items) {
  let html = `<dl class="result-facts">`;
  for (const [label, value] of items) {
    html += `<div>
      <dt class="label-mono">${escapeHtml(label)}</dt>
      <dd>${escapeHtml(value)}</dd>
    </div>`;
  }
  return html + `</dl>`;
}

function buildCrisprDetail(d) {
  let html = buildFacts([
    ["Start", d.start.toLocaleString()],
    ["End", d.end.toLocaleString()],
    ["Repeat length", `${d.drLength} bp`],
    ["Spacers", d.spacerCount],
    ["Direction", d.direction],
    ["Evidence level", d.evidenceLevel],
  ]);

  if (d.drConsensus && d.drConsensus !== "—") {
    html += `<dl class="result-sequence">
      <dt class="label-mono">Consensus repeat</dt>
      <dd>${escapeHtml(d.drConsensus)}</dd>
    </dl>`;
  }

  return html;
}

function buildCasDetail(d) {
  let html = buildFacts([
    ["Start", d.start.toLocaleString()],
    ["End", d.end.toLocaleString()],
    ["Genes", d.geneCount],
  ]);

  if (d.genes.length > 0) {
    html += `<div class="lr-table"><table>
      <thead><tr>
        <th>Gene</th><th>Start</th><th>End</th><th>Strand</th>
      </tr></thead><tbody>`;
    for (const g of d.genes) {
      html += `<tr>
        <td><strong>${escapeHtml(g.name)}</strong></td>
        <td>${g.start.toLocaleString()}</td>
        <td>${g.end.toLocaleString()}</td>
        <td>${escapeHtml(g.strand === "Forward" || g.strand === "+" ? "+" : g.strand === "Reverse" || g.strand === "-" ? "-" : ".")}</td>
      </tr>`;
    }
    html += `</tbody></table></div>`;
  }

  return html;
}

// --- Render results ---
function summarize(items, separator) {
  const crisprCount = items.filter((r) => r.type === "crispr").length;
  const casCount = items.filter((r) => r.type === "cas").length;
  const parts = [];
  if (crisprCount) parts.push(count(crisprCount, "CRISPR array"));
  if (casCount) parts.push(count(casCount, "Cas system"));
  return parts.join(separator);
}

function renderResults(items) {
  resultsList.innerHTML = "";

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const rowId = `result-${i}`;
    const entry = document.createElement("li");

    // Summary row
    const row = document.createElement("button");
    row.type = "button";
    row.className = "result-row";
    row.setAttribute("aria-expanded", "false");
    row.setAttribute("aria-controls", `${rowId}-detail`);
    const level = item.level
      ? `<span class="label-mono lr-status result-level${item.level === 4 ? " is-done" : ""}">Level ${escapeHtml(item.level)}</span>`
      : "";
    row.innerHTML = `
      <span class="label-mono lr-status result-kind${item.type === "cas" ? " is-cas" : ""}">${item.type === "cas" ? "Cas" : "CRISPR"}</span>
      <span class="result-name">
        <b>${escapeHtml(item.title)}</b>
        <span class="label-mono">${escapeHtml(item.subtitle)}</span>
      </span>
      ${level}
      <svg class="result-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
    `;

    // Detail panel
    const detail = document.createElement("div");
    detail.className = "result-detail";
    detail.id = `${rowId}-detail`;
    detail.hidden = true;
    detail.innerHTML =
      item.type === "crispr"
        ? buildCrisprDetail(item.detail)
        : buildCasDetail(item.detail);

    row.addEventListener("click", () => {
      const expanded = row.getAttribute("aria-expanded") === "true";
      row.setAttribute("aria-expanded", String(!expanded));
      detail.hidden = expanded;
    });

    entry.appendChild(row);
    entry.appendChild(detail);
    resultsList.appendChild(entry);
  }

  resultSummary.textContent = summarize(items, " · ");
  resultSummary.hidden = items.length === 0;
  resultsList.hidden = items.length === 0;
  resultsEmpty.hidden = items.length > 0;
  resultsEmpty.innerHTML = `<b>Nothing found</b><small>No CRISPR array or Cas system passed the limits you set.</small>`;
}

// --- Event handlers ---
fastaDrop.addEventListener("click", () => fastaFileInput.click());
fastaRetry.addEventListener("click", () => fastaFileInput.click());
fastaRemove.addEventListener("click", removeFastaFile);

fastaFileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) readFastaFile(file);
});

fastaZone.addEventListener("dragover", () => fastaDrop.classList.add("is-over"));
fastaZone.addEventListener("dragleave", () => fastaDrop.classList.remove("is-over"));
fastaZone.addEventListener("drop", (event) => {
  fastaDrop.classList.remove("is-over");
  const file = event.dataTransfer?.files?.[0];
  if (file) readFastaFile(file);
});

// A file dropped outside the zone must not replace the tab.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

fastaInput.addEventListener("input", () => {
  if (!fastaAccepted.hidden) {
    if (/\S/.test(fastaInput.value)) fileDetail.textContent = "Edited after loading · read locally";
    else removeFastaFile();
  }
  updateRunButton();
});

enableCas.addEventListener("change", updateRunButton);

analyzeBtn.addEventListener("click", async () => {
  for (const [field] of Object.values(optionFields)) field.removeAttribute("aria-invalid");

  const fasta = fastaInput.value.trim();
  if (!fasta || !fasta.startsWith(">")) {
    setStatus("This doesn't look like FASTA. The first line must start with \">\".", "error");
    return;
  }

  const options = {
    min_repeat_length: Number(minDr.value),
    max_repeat_length: Number(maxDr.value),
    min_spacer_length: Number(minSp.value),
    max_spacer_length: Number(maxSp.value),
    min_evidence_level: Number(levelMin.value),
    min_spacer_count: Number(minNbSpacers.value),
    no_mismatch: Boolean(noMism.checked),
  };

  const reject = (message, ...names) => {
    for (const name of names) optionFields[name][0].setAttribute("aria-invalid", "true");
    setStatus(message, "error");
  };

  for (const [name, [, label]] of Object.entries(optionFields)) {
    const value = options[name];
    if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffffffff) {
      reject(`${label} must be a whole number, 1 or more.`, name);
      return;
    }
  }
  if (options.min_evidence_level > 4) {
    reject("Minimum evidence level must be between 1 and 4.", "min_evidence_level");
    return;
  }

  if (options.min_repeat_length > options.max_repeat_length) {
    reject("Minimum repeat length is above the maximum. Lower it or raise the maximum.", "min_repeat_length", "max_repeat_length");
    return;
  }
  if (options.min_spacer_length > options.max_spacer_length) {
    reject("Minimum spacer length is above the maximum. Lower it or raise the maximum.", "min_spacer_length", "max_spacer_length");
    return;
  }

  const doCas = enableCas.checked;

  running = true;
  updateRunButton();
  downloadBtn.disabled = true;

  try {
    let items;
    if (doCas) {
      setProgress({ message: "Starting the detection…", current: 0, total: 0 });

      const result = await callWorker(
        "run_full_analysis",
        {
          fasta,
          options,
          casOpts: { genetic_code: 11, metagenome: metagenome.checked },
        },
        setProgress
      );

      const crisprItems = buildCrisprItems(result.crisprs || []);
      const casItems = buildCasItems(result.cas_clusters || []);
      items = [...crisprItems, ...casItems].sort((a, b) => a.start - b.start);

      lastResults = result;
    } else {
      setProgress({ message: "Detecting CRISPR arrays…", current: 0, total: 0 });

      const hits = await callWorker("find_repeats", { fasta, options });

      if (!Array.isArray(hits)) {
        throw new Error("Unexpected output from wasm module.");
      }

      items = buildCrisprItems(hits);
      lastResults = { crisprs: hits, cas_clusters: [] };
    }

    renderResults(items);
    downloadBtn.disabled = items.length === 0;
    setResult(
      items.length ? `${summarize(items, " and ")} found` : `No CRISPR array${doCas ? " or Cas system" : ""} found`,
      `Analysis complete${doCas ? "" : " (CRISPR only)"}. Nothing was uploaded.`,
      items.length > 0
    );
  } catch (error) {
    setStatus(`Analysis failed: ${error.message}`, "error");
  } finally {
    running = false;
    updateRunButton();
  }
});

downloadBtn.addEventListener("click", () => {
  if (!lastResults) return;

  const blob = new Blob([JSON.stringify(lastResults, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "crisprcas-results.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// --- Init ---
setIdleStatus();
updateRunButton();
