// Web Worker: runs WASM analysis off the main thread.
// Fetches CAS model data directly to avoid 21MB postMessage overhead.
import init, {
  find_repeats,
  cas_prepare,
  cas_search_profile,
  cas_search_all_profiles,
  cas_finalize,
  init_panic_hook,
  initThreadPool,
} from "./pkg/crispr_cas_finder_wasm.js";

let wasmReady = false;
let casModelsData = null; // cached { models: [...], profiles: [...] }
let threadPoolAttempted = false;
let threadPoolReady = false;
let threadPoolError = null;

const THREAD_POOL_TIMEOUT_MS = 8000;

function withTimeout(promise, timeoutMs, message) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(message));
    }, timeoutMs);

    promise.then(
      (value) => {
        clearTimeout(timeoutId);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeoutId);
        reject(error);
      }
    );
  });
}

async function ensureWasm() {
  if (wasmReady) return;
  await init();
  init_panic_hook();
  wasmReady = true;
}

async function ensureThreadPool(id) {
  if (threadPoolAttempted) return threadPoolReady;

  threadPoolAttempted = true;

  if (!self.crossOriginIsolated) {
    threadPoolError = "crossOriginIsolated is false";
    return false;
  }

  const requestedThreads = Math.max(1, navigator.hardwareConcurrency || 1);
  if (requestedThreads <= 1) {
    threadPoolError = "navigator.hardwareConcurrency reported a single core";
    return false;
  }

  progress(id, `Initializing WebAssembly thread pool (${requestedThreads} threads)...`, 0, 1);

  try {
    await withTimeout(
      initThreadPool(requestedThreads),
      THREAD_POOL_TIMEOUT_MS,
      `thread pool startup timed out after ${THREAD_POOL_TIMEOUT_MS / 1000}s`
    );
    threadPoolReady = true;
  } catch (error) {
    threadPoolError = error.message || String(error);
    console.warn("Falling back to single-threaded CAS search:", error);
  }

  return threadPoolReady;
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
      const threadPoolEnabled = await ensureThreadPool(id);
      if (!threadPoolEnabled && threadPoolError) {
        progress(
          id,
          `WebAssembly threads unavailable (${threadPoolError}). Falling back to single-threaded HMM search.`,
          0,
          1
        );
      }

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
      const neededProfileNames = new Set(prepInfo.needed_profiles || []);
      const profilesToSearch = data.profiles.filter(
        (profile) => neededProfileNames.size === 0 || neededProfileNames.has(profile.name)
      );
      const totalProfiles = profilesToSearch.length;
      progress(id, `Predicted ${prepInfo.gene_count} genes. Searching ${totalProfiles} HMM profiles...`, 0, totalProfiles);

      // 4. Search all needed profiles, using rayon when the browser can start the thread pool.
      let totalHits = 0;
      if (threadPoolEnabled) {
        progress(id, `Searching ${totalProfiles} HMM profiles in parallel...`, 0, totalProfiles);
        totalHits = cas_search_all_profiles(profilesToSearch);
      } else {
        for (let index = 0; index < totalProfiles; index += 1) {
          const profile = profilesToSearch[index];
          progress(id, `Searching HMM profiles sequentially (${index + 1}/${totalProfiles})...`, index, totalProfiles);
          totalHits += cas_search_profile(profile.name, profile.data);
        }
      }
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
