#!/usr/bin/env bash
set -euo pipefail

# Build the WASM crate with threading support (SharedArrayBuffer + rayon).
# Requires: nightly-2025-06-01 toolchain with rust-src and wasm32-unknown-unknown target.
# Requires: wasm-bindgen-cli v0.2.114

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Building crisprcas-wasm (release, threaded)..."
cargo +nightly-2025-06-01 build \
  -p crisprcas-wasm \
  --target wasm32-unknown-unknown \
  --release

echo "==> Running wasm-bindgen..."
wasm-bindgen \
  target/wasm32-unknown-unknown/release/crisprcas_wasm.wasm \
  --out-dir crisprcas-wasm/www/pkg \
  --target web

echo "==> Done. Output in crisprcas-wasm/www/pkg/"
