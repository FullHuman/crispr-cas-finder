# crispr-cas-finder-wasm Web App

This package exposes the CRISPR repeat detection engine to the browser and includes a website in `www/`.

## Build WASM

```bash
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
