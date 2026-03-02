#!/bin/bash
# ============================================================
# NCRN 评估脚本 (click-to-run)
# ============================================================
set -e

eval "$(conda shell.bash hook)"
conda activate dithub

OUTPUT_DIR="${OUTPUT_DIR:-./output/ncrn_output}"
SEED="${SEED:-3}"
NUM_GPUS="${NUM_GPUS:-1}"
CHECKPOINT="${CHECKPOINT:-groundingdino_swint_ogc.pth}"

echo "📊 NCRN Evaluation"
echo "   Output: ${OUTPUT_DIR}"
echo ""

python -u main_ncrn.py \
    --config-file test/test_odinw13_10shot \
    --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py \
    --model-checkpoint-path "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --num-gpus "${NUM_GPUS}" \
    --shuffle-tasks \
    --dithub \
    --eval-only

echo "✅ Evaluation complete! Results in ${OUTPUT_DIR}"
