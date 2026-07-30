#!/usr/bin/env bash
set -euo pipefail

# Build the WASM crate with threading support (SharedArrayBuffer + rayon).
# Requires: nightly-2026-04-01 toolchain with rust-src and wasm32-unknown-unknown target.
# Requires: wasm-bindgen-cli matching the wasm-bindgen version in Cargo.lock.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the toolchain, components, and targets from rust-toolchain.toml are installed.
rustup show active-toolchain >/dev/null 2>&1 || true
rustup component add rust-src 2>/dev/null || true
rustup target add wasm32-unknown-unknown 2>/dev/null || true

# Ensure wasm-bindgen-cli is installed (must match the wasm-bindgen version in Cargo.lock).
WASM_BINDGEN_VERSION="$(
  awk '
    $0 == "name = \"wasm-bindgen\"" { found = 1; next }
    found && /^version = / {
      gsub(/^version = "|"$|"/, "")
      print
      exit
    }
  ' Cargo.lock
)"
if [[ -z "$WASM_BINDGEN_VERSION" ]]; then
  echo "error: could not determine the wasm-bindgen version from Cargo.lock" >&2
  exit 1
fi
if ! command -v wasm-bindgen &>/dev/null || [[ "$(wasm-bindgen --version)" != *"$WASM_BINDGEN_VERSION"* ]]; then
  echo "==> Installing wasm-bindgen-cli@${WASM_BINDGEN_VERSION}..."
  cargo install wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION"
fi

echo "==> Building crispr-cas-finder-wasm (release, threaded)..."
# Threading flags (atomics, shared-memory, etc.) come from .cargo/config.toml.
# Keep build-std scoped to this wasm build so host benches and tests do not
# rebuild the standard library and trip duplicate lang item errors.
# Do NOT set RUSTFLAGS here — it would override the config and drop linker flags.
cargo build \
  -Zbuild-std=panic_abort,std \
  -p crispr-cas-finder-wasm \
  --target wasm32-unknown-unknown \
  --release

echo "==> Running wasm-bindgen..."
wasm-bindgen \
  target/wasm32-unknown-unknown/release/crispr_cas_finder_wasm.wasm \
  --out-dir crispr-cas-finder-wasm/www/pkg \
  --target web

echo "==> Done. Output in crispr-cas-finder-wasm/www/pkg/"
