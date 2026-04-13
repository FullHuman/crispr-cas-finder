#!/usr/bin/env bash
set -euo pipefail

# Build the WASM crate with threading support (SharedArrayBuffer + rayon).
# Requires: nightly-2026-04-01 toolchain with rust-src and wasm32-unknown-unknown target.
# Requires: wasm-bindgen-cli v0.2.114

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the toolchain, components, and targets from rust-toolchain.toml are installed.
rustup show active-toolchain >/dev/null 2>&1 || true
rustup component add rust-src 2>/dev/null || true
rustup target add wasm32-unknown-unknown 2>/dev/null || true

# Ensure wasm-bindgen-cli is installed (must match the wasm-bindgen version in Cargo.lock).
WASM_BINDGEN_VERSION="0.2.114"
if ! command -v wasm-bindgen &>/dev/null || [[ "$(wasm-bindgen --version)" != *"$WASM_BINDGEN_VERSION"* ]]; then
  echo "==> Installing wasm-bindgen-cli@${WASM_BINDGEN_VERSION}..."
  cargo install wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION"
fi

echo "==> Building crisprcas-wasm (release, threaded)..."
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
cargo build \
  -p crisprcas-wasm \
  --target wasm32-unknown-unknown \
  --release \
  -Z build-std=std,panic_abort

echo "==> Running wasm-bindgen..."
wasm-bindgen \
  target/wasm32-unknown-unknown/release/crisprcas_wasm.wasm \
  --out-dir crisprcas-wasm/www/pkg \
  --target web

echo "==> Done. Output in crisprcas-wasm/www/pkg/"
