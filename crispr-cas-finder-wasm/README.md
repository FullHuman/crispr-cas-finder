# crispr-cas-finder-wasm Web App

This package exposes the CRISPR array and Cas system detection engines to the browser and includes a website in `www/`.

## Build WASM

```bash
python3 crispr-cas-finder-wasm/bundle_cas_models.py
./build_wasm.sh
```

This uses the repo's threaded wasm build path, including the scoped
`-Zbuild-std=panic_abort,std` flag needed for the `wasm32-unknown-unknown`
target without affecting host builds such as Criterion benchmarks.

## Run the Website

From the repository root, use the bundled server so the browser receives the
cross-origin isolation headers required by WebAssembly threads:

```bash
python3 crispr-cas-finder-wasm/www/serve.py 8080
```

Then open `http://localhost:8080`.

## Notes

- The site imports `./pkg/crispr_cas_finder_wasm.js`, so you must run
  `./build_wasm.sh` first.
- All processing runs locally in the browser.

The model bundler reads the CasFinder data included in this repository; no
sibling checkout is required. It supports `--cas-dir` and `--output` overrides.
The checked-in model bundle supports static deployment; regenerate it when
changing the source model data.

The root `vercel.json` serves `www/` as a prebuilt static site. Generate the
model bundle and `www/pkg/` before deploying; the WASM package is git-ignored.

The threaded build requires SharedArrayBuffer and cross-origin isolation even
when HMM search falls back to one thread. Use `serve.py` locally, or preserve
the COOP/COEP headers from `vercel.json` on your server. Opening `index.html`
directly or using a plain HTTP server without those headers will not work.
Use `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. Safari does not support the
`credentialless` COEP value. After changing hosting headers, redeploy (or restart
the local server) and reload the page so the document and workers receive them.
If the thread pool cannot start, the worker uses sequential profile searches.

The stepwise Cas API permits one active analysis per worker. Call `cas_finalize`
to finish it or `cas_abort` after an error before starting another analysis.
Invalid options, malformed FASTA, and missing/invalid required profiles return
errors instead of silent defaults or partial results.

## Regression checks

After building, run `node crispr-cas-finder-wasm/tests/browser-smoke.cjs` with
Playwright and Chromium installed. `PLAYWRIGHT_MODULE` can select an existing
Playwright installation; `BROWSER_EXECUTABLE` can select an installed Chrome
binary. The test runs a loopback server, exercises input/error recovery, and
compares full E. coli results from parallel and sequential HMM searches.
Run the same checks with Safari's engine using `SMOKE_BROWSER=webkit` after
installing Playwright's WebKit browser (`npx playwright install webkit`).
The test server uses the deployment headers from `www/vercel.json`.

Run bundler tests with
`python3 -m unittest discover -s crispr-cas-finder-wasm/tests -p 'test_*.py'`.

Cas detection currently accepts genetic code 11 only. See
[limitations and validation](../LIMITATIONS.md) for evidence-level behavior and
unsupported upstream model semantics. The browser distribution includes
`LICENSE` and `THIRD_PARTY_NOTICES.md`; synchronize these copies with
`python3 scripts/sync_release_metadata.py` after changing their canonical files.
