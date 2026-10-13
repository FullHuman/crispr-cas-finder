#!/usr/bin/env bash
# Prepare the original CRISPRCasFinder to run natively on Apple silicon, so both
# tools can be timed on the same machine without emulation.
#
# Vmatch has no arm64 build (bioconda, Homebrew and vmatch.de ship x86_64 only),
# so it is compiled from source. Everything is written under vendor/, which is
# not tracked.
#
# Needs:
#   VSTREE_SRC    checkout of https://github.com/genometools/vstree
#   vendor/CRISPRCasFinder          upstream checkout
#   vendor/casfinder3/CasFinder     CasFinder 3.1.0 (msf_data install -t)
#   macsyfinder 2, prodigal, hmmsearch on PATH
#   a conda env with BioPerl, EMBOSS and MUSCLE (see bench.py, CCF_CONDA_ENV)
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
vstree=${VSTREE_SRC:?set VSTREE_SRC to a checkout of genometools/vstree}
build=${VSTREE_BUILD:-$root/vendor/vstree-arm64}
native=$root/vendor/upstream-native

if [ ! -x "$build/src/Vmatch/vmatch.x" ]; then
    mkdir -p "$build"
    cp -R "$vstree/src" "$build/src"
    cd "$build/src"
    # The macOS build definition treats warnings from current clang as errors and
    # expects a sibling genometools checkout, used only for its zlib and bzip2.
    sed -e 's/ -Werror//' \
        -e 's|-I${VSTREEBASEDIR}/../../../genometools/src/external/bzip2-1.0.6||' \
        -e 's|^INCLUDEOPT=-I${VSTREEBASEDIR}/include \\|INCLUDEOPT=-I${VSTREEBASEDIR}/include|' \
        -e 's|^DEFINELDLIBS=-lm.*|DEFINELDLIBS=-lm -lz -lbz2|' \
        Makedef-osx-gcc-64 > Makedef
    sed -i '' 's|^LDLIBS = ${CURDIR}/../../../genometools/lib/libgenometools.a \\|LDLIBS = \\|' \
        Mkvtree/Makefile Vmatch/Makefile
    CONFIGGUESS=arm64-apple-darwin WORKVSTREESRC="$PWD" DIRVSTREE="$build" WORK="$build" \
        make 64bit=yes CC=cc -j"$(sysctl -n hw.ncpu)" > "$build/build.log" 2>&1
fi

mkdir -p "$native/bin" "$native/models"
cp "$build/src/Vmatch/vmatch.x" "$native/bin/vmatch"
cp "$build/src/Mkvtree/mkvtree.x" "$native/bin/mkvtree"
cp "$build/src/Mkvtree/vsubseqselect.x" "$native/bin/vsubseqselect"

# The selection function CRISPRCasFinder loads into Vmatch; upstream ships it
# prebuilt for x86_64 only.
cc -Wall -fPIC -O3 -bundle -undefined dynamic_lookup -I"$build/src/include" \
    "$build/src/Vmatch/SELECT/sel392.c" -o "$native/sel392v2.so"

ln -sf "$(command -v prodigal)" "$native/bin/prodigal"
ln -sf "$(command -v hmmsearch)" "$native/bin/hmmsearch"

# CRISPRCasFinder asks MacSyFinder for a model package named "CASFinder" in its
# default location. The package installs as "CasFinder", and macOS file names
# are case-insensitive, so the name it expects goes in a directory of its own.
ln -sfn "$root/vendor/casfinder3/CasFinder" "$native/models/CASFinder"
cat > "$native/bin/macsyfinder" <<EOF
#!/bin/bash
exec "$(command -v macsyfinder)" --models-dir "$native/models" "\$@"
EOF
chmod +x "$native/bin/macsyfinder"

"$native/bin/vmatch" -version | head -1
