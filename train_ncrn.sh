#!/bin/bash
# ============================================================
# NCRN 训练脚本 (click-to-run)
# 在 4090 服务器上运行 ODinW-13 增量学习实验
# ============================================================
set -e

# 激活环境
eval "$(conda shell.bash hook)"
conda activate dithub

# 默认参数
OUTPUT_DIR="${OUTPUT_DIR:-./output/ncrn_output}"
SEED="${SEED:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
CHECKPOINT="${CHECKPOINT:-groundingdino_swint_ogc.pth}"

echo "🚀 NCRN Training"
echo "   Output: ${OUTPUT_DIR}"
echo "   Seed:   ${SEED}"
echo "   GPUs:   ${NUM_GPUS}"
echo ""

python -u main_ncrn.py \
    --config-file test/test_odinw13 \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
    --model-checkpoint-path "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --num-gpus "${NUM_GPUS}" \
    --shuffle-tasks \
    --dithub

echo "✅ Training complete! Results in ${OUTPUT_DIR}"
