#!/usr/bin/env bash
# Download MNIST IDX files from Azure Open Datasets into data/mnist/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/data/mnist"
mkdir -p "$OUT_DIR"

BASE="https://azureopendatastorage.blob.core.windows.net/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for name in "${FILES[@]}"; do
    out_name="${name%.gz}"
    out_path="$OUT_DIR/$out_name"
    gz_path="$OUT_DIR/$name"

    if [[ -f "$out_path" ]]; then
        echo "exists: $out_name"
        continue
    fi

    echo "download: $name"
    curl -fsSL "$BASE/$name" -o "$gz_path"

    echo "inflate:  $out_name"
    gzip -dc "$gz_path" > "$out_path"
    rm -f "$gz_path"
done

echo "done: $OUT_DIR"
