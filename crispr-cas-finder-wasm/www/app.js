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
const fileName = document.getElementById("file-name");
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
const resultsPanel = document.getElementById("results-panel");
const resultsList = document.getElementById("results-list");
const resultSummary = document.getElementById("result-summary");

let lastResults = null;

// --- Status helpers ---
function setStatus(message, kind = "ok") {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "ok", "error");
  statusPanel.classList.add(kind);
}

function clearStatus() {
  statusPanel.classList.add("hidden");
  statusPanel.classList.remove("ok", "error");
}

// --- Result builders ---
function buildCrisprItems(arrays) {
  return arrays.map((array, idx) => {
    const spacerCount = array.spacers ? array.spacers.length : 0;
    const orientationMap = { Forward: "+", Reverse: "-", Unknown: "." };
    // Prefer the CRISPRDirection DB value (crispr_direction) over the AT%-based
    // orientation heuristic — the DB lookup matches the original CRISPRCasFinder.
    const dir = array.crispr_direction && array.crispr_direction !== "ND"
      ? array.crispr_direction
      : (orientationMap[array.orientation] || ".");
    const drLen = array.consensus_repeat ? array.consensus_repeat.length : 0;
    return {
      type: "crispr",
      title: `${array.seq_id}_CRISPR_${idx + 1}`,
      subtitle: `${array.start.toLocaleString()} – ${array.end.toLocaleString()}`,
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
      subtitle: `${cluster.start.toLocaleString()} – ${cluster.end.toLocaleString()} · ${genes.length} gene(s)`,
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

function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = String(str);
  return el.innerHTML;
}

// --- Detail panel builders ---
function buildCrisprDetail(d) {
  let html = `<div class="detail-grid">`;
  const items = [
    ["Start", d.start.toLocaleString(), true],
    ["End", d.end.toLocaleString(), true],
    ["DR Consensus", d.drConsensus, false],
    ["DR Length", `${d.drLength} bp`, true],
    ["Spacers", d.spacerCount, true],
    ["Direction", d.direction, true],
    ["Evidence Level", d.evidenceLevel, true],
  ];
  for (const [label, value, mono] of items) {
    html += `<div class="detail-item">
      <span class="detail-label">${escapeHtml(label)}</span>
      <span class="detail-value${mono ? " mono" : ""}">${escapeHtml(value)}</span>
    </div>`;
  }
  html += `</div>`;

  // DR consensus sequence display
  if (d.drConsensus && d.drConsensus !== "—") {
    html += `<div style="margin-top: 14px;">
      <span class="detail-label">DR Consensus Sequence</span>
      <div class="detail-value seq" style="margin-top: 4px;">${escapeHtml(d.drConsensus)}</div>
    </div>`;
  }

  return html;
}

function buildCasDetail(d) {
  let html = `<div class="detail-grid">`;
  const items = [
    ["Start", d.start.toLocaleString(), true],
    ["End", d.end.toLocaleString(), true],
    ["Gene Count", d.geneCount, true],
  ];
  for (const [label, value, mono] of items) {
    html += `<div class="detail-item">
      <span class="detail-label">${escapeHtml(label)}</span>
      <span class="detail-value${mono ? " mono" : ""}">${escapeHtml(value)}</span>
    </div>`;
  }
  html += `</div>`;

  if (d.genes.length > 0) {
    html += `<table class="gene-table">
      <thead><tr>
        <th>Gene</th><th>Start</th><th>End</th><th>Strand</th>
      </tr></thead><tbody>`;
    for (const g of d.genes) {
      html += `<tr>
        <td>${escapeHtml(g.name)}</td>
        <td>${g.start.toLocaleString()}</td>
        <td>${g.end.toLocaleString()}</td>
        <td>${escapeHtml(g.strand === "Forward" || g.strand === "+" ? "+" : g.strand === "Reverse" || g.strand === "-" ? "-" : ".")}</td>
      </tr>`;
    }
    html += `</tbody></table>`;
  }

  return html;
}

// --- Render results ---
function renderResults(items) {
  resultsList.innerHTML = "";

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const rowId = `result-${i}`;

    // Summary row
    const row = document.createElement("div");
    row.className = "result-row";
    row.setAttribute("role", "button");
    row.setAttribute("tabindex", "0");
    row.setAttribute("aria-expanded", "false");
    row.setAttribute("aria-controls", `${rowId}-detail`);
    row.innerHTML = `
      <span class="result-badge ${item.type}">${escapeHtml(item.type)}</span>
      <div>
        <div class="result-title">${escapeHtml(item.title)}</div>
        <div class="result-subtitle">${escapeHtml(item.subtitle)}</div>
      </div>
      <span class="result-chevron">›</span>
    `;

    // Detail panel
    const detail = document.createElement("div");
    detail.className = "result-detail";
    detail.id = `${rowId}-detail`;
    detail.setAttribute("aria-hidden", "true");
    detail.innerHTML =
      item.type === "crispr"
        ? buildCrisprDetail(item.detail)
        : buildCasDetail(item.detail);

    // Toggle
    const toggle = () => {
      const expanded = row.getAttribute("aria-expanded") === "true";
      row.setAttribute("aria-expanded", String(!expanded));
      detail.setAttribute("aria-hidden", String(expanded));
    };
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });

    resultsList.appendChild(row);
    resultsList.appendChild(detail);
  }

  const crisprCount = items.filter((r) => r.type === "crispr").length;
  const casCount = items.filter((r) => r.type === "cas").length;
  const parts = [];
  if (crisprCount) parts.push(`${crisprCount} CRISPR array(s)`);
  if (casCount) parts.push(`${casCount} CAS system(s)`);
  resultSummary.textContent = parts.length ? parts.join(", ") : "No elements found";
  resultsPanel.classList.remove("hidden");
}

// --- Event handlers ---
fastaFileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    fileName.textContent = "No file selected";
    return;
  }
  fileName.textContent = file.name;
  try {
    fastaInput.value = await file.text();
    clearStatus();
  } catch (error) {
    setStatus(`Failed reading file: ${error.message}`, "error");
  }
});

analyzeBtn.addEventListener("click", async () => {
  const fasta = fastaInput.value.trim();
  if (!fasta || !fasta.startsWith(">")) {
    setStatus("Please provide a valid FASTA input (must start with '>').", "error");
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

  for (const [name, value] of Object.entries(options)) {
    if (name === "no_mismatch") continue;
    if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffffffff) {
      setStatus(`Invalid option ${name}: enter a positive 32-bit integer.`, "error");
      return;
    }
  }
  if (options.min_evidence_level > 4) {
    setStatus("Evidence level must be between 1 and 4.", "error");
    return;
  }

  if (options.min_repeat_length > options.max_repeat_length) {
    setStatus("Invalid options: min DR must be <= max DR.", "error");
    return;
  }
  if (options.min_spacer_length > options.max_spacer_length) {
    setStatus("Invalid options: min spacer must be <= max spacer.", "error");
    return;
  }

  const doCas = enableCas.checked;

  analyzeBtn.disabled = true;
  downloadBtn.disabled = true;

  try {
    if (doCas) {
      setStatus("Starting CRISPR + CAS analysis...", "ok");

      const result = await callWorker(
        "run_full_analysis",
        {
          fasta,
          options,
          casOpts: { genetic_code: 11, metagenome: metagenome.checked },
        },
        (prog) => setStatus(prog.message, "ok")
      );

      const crisprItems = buildCrisprItems(result.crisprs || []);
      const casItems = buildCasItems(result.cas_clusters || []);
      const items = [...crisprItems, ...casItems].sort((a, b) => a.start - b.start);

      lastResults = result;
      renderResults(items);
      downloadBtn.disabled = items.length === 0;
      setStatus("Analysis complete.", "ok");
    } else {
      setStatus("Running CRISPR detection...", "ok");

      const hits = await callWorker("find_repeats", { fasta, options });

      if (!Array.isArray(hits)) {
        throw new Error("Unexpected output from wasm module.");
      }

      const items = buildCrisprItems(hits);
      lastResults = { crisprs: hits, cas_clusters: [] };
      renderResults(items);
      downloadBtn.disabled = items.length === 0;
      setStatus("Analysis complete (CRISPR only).", "ok");
    }
  } catch (error) {
    setStatus(`Analysis failed: ${error.message}`, "error");
  } finally {
    analyzeBtn.disabled = false;
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
setStatus("Load FASTA content, then click Analyze.", "ok");
