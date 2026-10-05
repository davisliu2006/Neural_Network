#!/usr/bin/env bash
# Build (via test/Makefile) and run an integration test from the repo root.
# Usage: ./scripts/run_integration.sh <name>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <name>" >&2
    exit 1
fi

NAME="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC="test/integration/$NAME.cpp"
if [[ ! -f "$SRC" ]]; then
    echo "integration test not found: $SRC" >&2
    exit 1
fi

EXEEXT=
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXEEXT=.exe ;;
esac
# Native Windows (e.g. Git Bash may still report MINGW; also handle OS=Windows_NT)
if [[ "${OS:-}" == "Windows_NT" ]]; then
    EXEEXT=.exe
fi

EXE="test/bin/${NAME}${EXEEXT}"

echo "compile: $SRC"
make -C test "bin/${NAME}${EXEEXT}"

echo "run:     $EXE"
exec "$EXE"
