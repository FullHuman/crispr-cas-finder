#!/usr/bin/env bash
set -euo pipefail

# Build the WASM crate with threading support (SharedArrayBuffer + rayon).
# Requires: nightly-2025-06-01 toolchain with rust-src and wasm32-unknown-unknown target.
# Requires: wasm-bindgen-cli v0.2.114

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the toolchain, components, and targets from rust-toolchain.toml are installed.
rustup show active-toolchain >/dev/null 2>&1 || true
rustup component add rust-src 2>/dev/null || true
rustup target add wasm32-unknown-unknown 2>/dev/null || true

echo "==> Building crisprcas-wasm (release, threaded)..."
cargo build \
  -p crisprcas-wasm \
  --target wasm32-unknown-unknown \
  --release

echo "==> Running wasm-bindgen..."
wasm-bindgen \
  target/wasm32-unknown-unknown/release/crisprcas_wasm.wasm \
  --out-dir crisprcas-wasm/www/pkg \
  --target web

echo "==> Done. Output in crisprcas-wasm/www/pkg/"
