#!/usr/bin/env bash
# render_architecture.sh
# Export Mermaid architecture diagram to SVG and PNG.
# Requirements: mermaid-cli (mmdc). Install via: npm install -g @mermaid-js/mermaid-cli
# Usage: bash deployment_package/rag_versions/rag_system/scripts/render_architecture.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MMD_FILE="${ROOT_DIR}/deployment_package/rag_versions/rag_system/docs/architecture.mmd"
OUT_SVG="${ROOT_DIR}/deployment_package/rag_versions/rag_system/docs/architecture.svg"
OUT_PNG="${ROOT_DIR}/deployment_package/rag_versions/rag_system/docs/architecture.png"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "[render_architecture] mermaid-cli (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli" >&2
  exit 1
fi

if [ ! -f "${MMD_FILE}" ]; then
  echo "[render_architecture] Source Mermaid file not found: ${MMD_FILE}" >&2
  exit 1
fi

echo "[render_architecture] Rendering SVG -> ${OUT_SVG}"
mmdc -i "${MMD_FILE}" -o "${OUT_SVG}" --scale 1 || { echo "Failed to render SVG" >&2; exit 1; }

echo "[render_architecture] Rendering PNG -> ${OUT_PNG}"
mmdc -i "${MMD_FILE}" -o "${OUT_PNG}" --scale 2 || { echo "Failed to render PNG" >&2; exit 1; }

echo "[render_architecture] Done"
