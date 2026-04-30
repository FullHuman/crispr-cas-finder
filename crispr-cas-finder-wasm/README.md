# crisprcas-wasm Web App

This package exposes the CRISPR repeat detection engine to the browser and includes a website in `www/`.

## Build WASM

```bash
./build_wasm.sh
```

This uses the repo's threaded wasm build path, including the scoped
`-Zbuild-std=panic_abort,std` flag needed for the `wasm32-unknown-unknown`
target without affecting host builds such as Criterion benchmarks.

## Run the Website

Use any static file server from `crisprcas-wasm/www`:

```bash
cd crisprcas-wasm/www
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Notes

- The site imports `./pkg/crisprcas_wasm.js`, so you must run `wasm-pack build` first.
- All processing runs locally in the browser.
