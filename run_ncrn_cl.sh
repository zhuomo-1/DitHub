#!/bin/bash
# =============================================================
# NCRN 连续学习正式训练脚本 (10-shot)
# =============================================================
# 架构：每个任务作为独立 Python 进程运行
#   - 进程退出后 OS 回收全部内存/mmap/GPU 显存
#   - 下一个任务启动时只多一步：load ncrn_shared.pth 恢复 NCRN 头
#   - 通过 ncrn_shared.pth 在任务间传递增量学习状态
# =============================================================
set -e

OUTPUT_DIR="${1:-./output/ncrn_10shot}"
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

TASK_CONFIGS=($(ls -1 ${CONFIG_DIR}/for_train/*.py | sort))
TOTAL_TASKS=${#TASK_CONFIGS[@]}

echo "=============================================="
echo "NCRN Continual Learning (10-shot)"
echo "  Tasks:  ${TOTAL_TASKS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Seed:   ${SEED}"
echo "  GPU:    ${GPU}"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

TRAIN_START=$(date +%s)
FAILED_TASKS=()

for i in $(seq 0 $((TOTAL_TASKS - 1))); do
    TASK_CONFIG="${TASK_CONFIGS[$i]}"
    TASK_NAME=$(basename "${TASK_CONFIG}" .py)
    TASK_LOG="${OUTPUT_DIR}/task_${i}_${TASK_NAME}.log"

    echo ""
    echo "====== Task $((i+1))/${TOTAL_TASKS}: ${TASK_NAME} (index=${i}) ======"
    echo "  Config: ${TASK_CONFIG}"
    echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"

    TASK_START=$(date +%s)

    # 每个任务作为完全独立的进程运行
    # 进程退出后 OS 自动回收：GPU 显存、CPU 内存、mmap 映射、文件描述符
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
        echo "  Log: ${TASK_LOG}"
        echo "  Last 10 lines:"
        tail -10 "${TASK_LOG}" | sed 's/^/    /'
        FAILED_TASKS+=("${i}:${TASK_NAME}")
        exit ${EXIT_CODE}
    fi

    # 验证 ncrn_shared.pth 已更新
    CKPT="${OUTPUT_DIR}/ncrn_shared.pth"
    if [ ! -f "${CKPT}" ]; then
        echo "  ERROR: ncrn_shared.pth not found after task ${i}!"
        exit 1
    fi

    CKPT_SIZE=$(ls -lh "${CKPT}" | awk '{print $5}')
    LAST_LOSS=$(grep -o 'loss_total: [0-9.]*' "${TASK_LOG}" | tail -1 | awk '{print $2}')

    echo "  OK (${TASK_ELAPSED}s) | final_loss=${LAST_LOSS:-N/A} | shared_ckpt=${CKPT_SIZE}"
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "  Tasks: ${TOTAL_TASKS}"
echo "  Time:  ${TRAIN_ELAPSED}s ($(echo "scale=1; ${TRAIN_ELAPSED}/60" | bc)min)"
echo "  Shared checkpoint: ${OUTPUT_DIR}/ncrn_shared.pth"
echo "=============================================="

# ===================== 评估阶段 =====================
echo ""
echo "====== EVALUATION PHASE ======"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

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

EVAL_EXIT=$?
if [ ${EVAL_EXIT} -ne 0 ]; then
    echo "  Evaluation FAILED (exit=${EVAL_EXIT})"
    echo "  Last 10 lines:"
    tail -10 "${EVAL_LOG}" | sed 's/^/    /'
else
    echo "  Evaluation OK"
    # 打印 AP 结果
    grep -E "AP|Average" "${EVAL_LOG}" | tail -20 | sed 's/^/  /'
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TRAIN_START))

echo ""
echo "=============================================="
echo "ALL DONE"
echo "  Total time: ${TOTAL_ELAPSED}s ($(echo "scale=1; ${TOTAL_ELAPSED}/60" | bc)min)"
echo "  Output: ${OUTPUT_DIR}"
echo "  Train log: ${OUTPUT_DIR}/task_*.log"
echo "  Eval log:  ${EVAL_LOG}"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
