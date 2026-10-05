#!/usr/bin/env bash
# Remove built test binaries via test/Makefile.
# Usage: ./scripts/clean.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec make -C test clean
