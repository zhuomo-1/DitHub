#!/bin/bash
# ============================================================
# ODinW-35 数据集下载脚本
# 包含 ODinW-13 实验所需的全部数据
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/datasets"

echo "📥 下载 ODinW-35 数据集到 ${DATA_DIR}..."
echo "   (约 10GB，首次下载可能需要较长时间)"

cd "$PROJECT_DIR"
python tools/download_odinw.py

echo ""
echo "✅ 数据下载完成！"
echo "   数据目录: ${DATA_DIR}"
ls -la "${DATA_DIR}" 2>/dev/null || echo "   (请检查数据是否正确下载)"
