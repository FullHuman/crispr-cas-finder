#!/usr/bin/env bash
set -euo pipefail

# Build the WASM crate with threading support (SharedArrayBuffer + rayon).
# Requires: the toolchain, rust-src component, and wasm target pinned in
# rust-toolchain.toml.
# Requires: wasm-bindgen-cli matching the wasm-bindgen version in Cargo.lock.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure the toolchain, components, and targets from rust-toolchain.toml are installed.
TOOLCHAIN="$(
  awk -F'"' '$1 ~ /^channel = / { print $2; exit }' rust-toolchain.toml
)"
if [[ -z "$TOOLCHAIN" ]]; then
  echo "error: could not determine the channel from rust-toolchain.toml" >&2
  exit 1
fi
if ! rustup toolchain list | grep -q "^${TOOLCHAIN}-"; then
  rustup toolchain install "$TOOLCHAIN" \
    --profile minimal \
    --component rust-src \
    --target wasm32-unknown-unknown
else
  rustup component add --toolchain "$TOOLCHAIN" rust-src
  rustup target add --toolchain "$TOOLCHAIN" wasm32-unknown-unknown
fi

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
  cargo install wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION" --locked
fi

echo "==> Building crispr-cas-finder-wasm (release, threaded)..."
# Threading flags (atomics, shared-memory, etc.) come from .cargo/config.toml.
# Keep build-std scoped to this wasm build so host benches and tests do not
# rebuild the standard library and trip duplicate lang item errors.
# Do NOT set RUSTFLAGS here — it would override the config and drop linker flags.
cargo build --locked \
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
