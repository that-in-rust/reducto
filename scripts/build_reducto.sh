#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +"%Y%m%d%H%M%S")"
BIN_NAME="reducto-v-${TS}"

# Build the working benchmark binary
cargo build --release --bin benchmark

# Export a timestamped artifact
mkdir -p dist
cp target/release/benchmark "dist/${BIN_NAME}"

printf "Built artifact: %s\n" "dist/${BIN_NAME}"
