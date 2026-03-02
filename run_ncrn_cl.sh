#!/bin/bash
# =============================================================
# NCRN 连续学习编排脚本
# 每个任务作为独立 Python 进程运行，避免 max_map_count 耗尽
# 通过 ncrn_shared.pth 在任务间传递增量学习状态
# =============================================================
set -e

OUTPUT_DIR="${1:-./output/ncrn_smoke}"
SEED="${2:-3}"
CONFIG_DIR="test/test_odinw13"
MODEL_CONFIG="groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py"
MODEL_CKPT="groundingdino_swint_ogc.pth"
PYTHON="/opt/data/private/conda_envs/dithub/bin/python"

export LD_LIBRARY_PATH=/opt/data/private/conda_envs/dithub/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUTPUT_DIR}"

# Collect and optionally shuffle task configs
TASK_CONFIGS=($(ls -1 ${CONFIG_DIR}/for_train/*.py | sort))
TOTAL_TASKS=${#TASK_CONFIGS[@]}

echo "=============================================="
echo "NCRN Continual Learning: ${TOTAL_TASKS} tasks"
echo "Output: ${OUTPUT_DIR}"
echo "Seed: ${SEED}"
echo "=============================================="

START_TIME=$(date +%s)

for i in $(seq 0 $((TOTAL_TASKS - 1))); do
    TASK_CONFIG="${TASK_CONFIGS[$i]}"
    TASK_NAME=$(basename "${TASK_CONFIG}" .py)

    echo ""
    echo "====== Task $((i+1))/${TOTAL_TASKS}: ${TASK_NAME} ======"
    echo "Config: ${TASK_CONFIG}"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

    ${PYTHON} -u main_ncrn.py \
        --task-config "${TASK_CONFIG}" \
        --task-index ${i} \
        --total-tasks ${TOTAL_TASKS} \
        --model-config-file "${MODEL_CONFIG}" \
        --model-checkpoint-path "${MODEL_CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --seed ${SEED} \
        --dithub \
        2>&1 | tee -a "${OUTPUT_DIR}/task_${i}_${TASK_NAME}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Task ${i} (${TASK_NAME}) failed with exit code ${EXIT_CODE}"
        echo "Check log: ${OUTPUT_DIR}/task_${i}_${TASK_NAME}.log"
        exit ${EXIT_CODE}
    fi

    echo "Task $((i+1))/${TOTAL_TASKS} done: $(date '+%Y-%m-%d %H:%M:%S')"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=============================================="
echo "All ${TOTAL_TASKS} tasks completed in ${ELAPSED}s"
echo "Shared checkpoint: ${OUTPUT_DIR}/ncrn_shared.pth"
echo "=============================================="

# Run evaluation
echo ""
echo "Starting evaluation..."
${PYTHON} -u main_ncrn.py \
    --eval-only \
    --config-file "${CONFIG_DIR}" \
    --model-config-file "${MODEL_CONFIG}" \
    --model-checkpoint-path "${MODEL_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    --dithub \
    2>&1 | tee -a "${OUTPUT_DIR}/eval.log"

echo "All done!"
