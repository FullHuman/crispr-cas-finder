// Web Worker: runs WASM analysis off the main thread.
// Fetches CAS model data directly to avoid 21MB postMessage overhead.
import init, {
  find_repeats,
  cas_prepare,
  cas_search_all_profiles,
  cas_finalize,
  init_panic_hook,
  initThreadPool,
} from "./pkg/crisprcas_wasm.js";

let wasmReady = false;
let casModelsData = null; // cached { models: [...], profiles: [...] }

async function ensureWasm() {
  if (wasmReady) return;
  await init();
  init_panic_hook();
  await initThreadPool(navigator.hardwareConcurrency);
  wasmReady = true;
}

function progress(id, message, current, total) {
  self.postMessage({ id, progress: { message, current, total } });
}

async function loadCasModels(id) {
  if (casModelsData) return casModelsData;
  progress(id, "Downloading CAS models...", 0, 1);
  const resp = await fetch("cas-models.json");
  if (!resp.ok) throw new Error(`Failed to fetch CAS models: HTTP ${resp.status}`);
  casModelsData = await resp.json();
  return casModelsData;
}

self.onmessage = async (e) => {
  const { type, id, payload } = e.data;

  try {
    await ensureWasm();

    if (type === "find_repeats") {
      const result = find_repeats(payload.fasta, payload.options);
      self.postMessage({ id, result });

    } else if (type === "run_full_analysis") {
      // 1. Load CAS models (fetched here, not sent through postMessage)
      const data = await loadCasModels(id);

      // 2. CRISPR detection
      progress(id, "Detecting CRISPR arrays...", 0, 0);
      const crisprs = find_repeats(payload.fasta, payload.options);

      // 3. Gene prediction + model setup
      progress(id, "Predicting genes...", 0, 0);
      const prepInfo = cas_prepare(
        payload.fasta,
        data.models,
        payload.casOpts
      );
      const totalProfiles = data.profiles.length;
      progress(id, `Predicted ${prepInfo.gene_count} genes. Searching ${totalProfiles} HMM profiles...`, 0, totalProfiles);

      // 4. Search all profiles in parallel via rayon
      progress(id, `Searching ${totalProfiles} HMM profiles (parallel)...`, 0, totalProfiles);
      const totalHits = cas_search_all_profiles(data.profiles);
      progress(id, `HMM search complete. ${totalHits} total hits.`, totalProfiles, totalProfiles);

      // 5. Finalize — clustering + system evaluation
      progress(id, "Evaluating CAS systems...", totalProfiles, totalProfiles);
      const casClusters = cas_finalize();

      self.postMessage({
        id,
        result: { crisprs, cas_clusters: casClusters },
      });

    } else {
      throw new Error(`Unknown message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};
