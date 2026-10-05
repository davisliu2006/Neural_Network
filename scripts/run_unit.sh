#!/usr/bin/env bash
# Build and run all unit tests via test/Makefile.
# Usage: ./scripts/run_unit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec make -C test test
