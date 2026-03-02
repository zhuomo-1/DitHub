#!/bin/bash
# 3-task fast validation: AerialMaritimeDrone(5cls) + CottontailRabbits(1cls) + Egohands(1cls)
set -e

OUTPUT_DIR="${1:-./output/ncrn_3task}"
SEED="${2:-3}"
GPU="${3:-0}"
CONFIG_DIR="test/test_odinw13_10shot"
MODEL_CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py"
MODEL_CKPT="groundingdino_swint_ogc.pth"
PYTHON="/opt/data/private/conda_envs/dithub/bin/python"

export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p "${OUTPUT_DIR}"

TASK_CONFIGS=(
    "${CONFIG_DIR}/for_train/test_AerialMaritimeDrone_tiled.py"
    "${CONFIG_DIR}/for_train/test_CottontailRabbits.py"
    "${CONFIG_DIR}/for_train/test_Egohands_generic.py"
)
TOTAL_TASKS=${#TASK_CONFIGS[@]}

echo "=============================================="
echo "NCRN 3-Task Fast Validation"
echo "  Tasks:  ${TOTAL_TASKS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

TRAIN_START=$(date +%s)

for i in $(seq 0 $((TOTAL_TASKS - 1))); do
    TASK_CONFIG="${TASK_CONFIGS[$i]}"
    TASK_NAME=$(basename "${TASK_CONFIG}" .py)
    TASK_LOG="${OUTPUT_DIR}/task_${i}_${TASK_NAME}.log"

    echo ""
    echo "====== Task $((i+1))/${TOTAL_TASKS}: ${TASK_NAME} (index=${i}) ======"
    TASK_START=$(date +%s)

    ${PYTHON} -u main_ncrn.py \
        --task-config "${TASK_CONFIG}" \
        --task-index ${i} \
        --total-tasks ${TOTAL_TASKS} \
        --model-config-file "${MODEL_CONFIG}" \
        --model-checkpoint-path "${MODEL_CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --seed ${SEED} \
        --dithub \
        > "${TASK_LOG}" 2>&1

    EXIT_CODE=$?
    TASK_END=$(date +%s)
    TASK_ELAPSED=$((TASK_END - TASK_START))

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  FAILED (exit=${EXIT_CODE}, ${TASK_ELAPSED}s)"
        tail -10 "${TASK_LOG}" | sed 's/^/    /'
        exit ${EXIT_CODE}
    fi

    LAST_LOSS=$(grep -o 'loss_total: [0-9.]*' "${TASK_LOG}" | tail -1 | awk '{print $2}')
    echo "  OK (${TASK_ELAPSED}s) | final_loss=${LAST_LOSS:-N/A}"
done

TRAIN_END=$(date +%s)
echo ""
echo "TRAINING DONE in $((TRAIN_END - TRAIN_START))s"

# Eval
echo ""
echo "====== EVALUATION ======"
EVAL_LOG="${OUTPUT_DIR}/eval.log"

${PYTHON} -u main_ncrn.py \
    --eval-only \
    --config-file "${CONFIG_DIR}" \
    --model-config-file "${MODEL_CONFIG}" \
    --model-checkpoint-path "${MODEL_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --dithub \
    > "${EVAL_LOG}" 2>&1

echo "Eval done. Results:"
grep -E "test_Aerial|test_Cottontail|test_Egohands|Average" "${EVAL_LOG}" | tail -10 | sed 's/^/  /'
echo ""
echo "Full eval log: ${EVAL_LOG}"
