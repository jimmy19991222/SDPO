#!/bin/bash
# =============================================================================
# TASD v8 Level-1: teacher reliability 绝对筛选 - Nebula 批量提交
#
# 设计背景：
#   v7 出现 teacher 与 student 同步熵崩塌 → 相对判据 hard gate (ent_T<ent_S)
#   失效（两者都崩仍满足条件）。Level-1 引入**绝对判据**，与 entropy_gate 正交：
#     A. teacher_abs_entropy_gate: teacher 归一化熵 < 阈值 才算可靠
#     B. teacher_prob_floor:       p_teacher(y_t) > 阈值  才算可靠
#   两者 AND 合成 teacher_reliable_mask，与现有 gate_mask 再 AND。
#
# 新增 swanlab 指标：
#   tasd/teacher_reliable_ratio           综合保留率
#   tasd/teacher_abs_ent_hit_ratio        仅 A 条件保留率
#   tasd/teacher_prob_floor_hit_ratio     仅 B 条件保留率
#
# 实验矩阵（与 v7 R2 主力配置为基线做对照）：
#   B0: abs=1.0  prob=0.00  → baseline（v7 R2 复现，控制对照）
#   B1: abs=0.5  prob=0.00  → 只开 A（温和）
#   B2: abs=0.3  prob=0.00  → 只开 A（激进）
#   B3: abs=1.0  prob=0.05  → 只开 B（标准）
#   B4: abs=0.5  prob=0.05  → A+B 合成（温和组合，推荐）
#   B5: abs=0.5  prob=0.01  → A+B 合成（更宽松，留 buffer）
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_v8_teacher_reliable_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD-v8-TeacherReliable"

# ── 数据集 ───────────────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "tooluse"
    # "sciknoweval/material"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# Level-1 teacher reliability 扫描矩阵
# 格式: "TEACHER_ABS_ENTROPY_GATE:TEACHER_PROB_FLOOR:RUN_TAG"
# =============================================================================
L1_MATRIX=(
    "1.0:0.00:absOff-probOff-baseline"
    "0.5:0.00:abs050-probOff"
    "0.3:0.00:abs030-probOff"
    "1.0:0.05:absOff-prob005"
    "0.5:0.05:abs050-prob005"
    "0.5:0.01:abs050-prob001"
)

# =============================================================================
# 固定超参：沿用 v7 R2 主力配置（ctx=group_shared, fbOn, trSuccT）
# =============================================================================
# v7 核心开关
TEACHER_CONTEXT_MODE="group_shared"
INCLUDE_ENVIRONMENT_FEEDBACK="True"
TRAIN_ON_SUCCESS="True"
MAX_ERRORS_IN_POOL="8"
ERROR_ANSWER_MAX_CHARS="1024"

# Reward & gate
REWARD_TYPE="teacher_log_prob"
ENTROPY_GATE="hard_keep_reward"
ENTROPY_GATE_RATIO="1.0"
ENTROPY_GATE_TOLERANCE="0.0"
DISTILL_TOPK="256"                    # Level-1 需要 topk 算 teacher entropy
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
REMOVE_THINKING="True"
ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION="False"
TEACHER_UPDATE_RATE="0.1"
TEACHER_REG="ema"
LR="1e-5"
SEED="42"
ENTROPY_COEFF="0.001"
TEMPERATURE="1.0"
ENTROPY_FLOOR="0.0"
ENTROPY_PENALTY_COEFF="0.0"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"

# Git 信息
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for ENTRY in "${L1_MATRIX[@]}"; do
    IFS=':' read -r TEACHER_ABS_ENTROPY_GATE TEACHER_PROB_FLOOR RUN_TAG <<< "$ENTRY"

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-${DATASET_SHORT}-${RUN_TAG}-v8L1-${MODEL_SHORT}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET"
        echo "  TEACHER_ABS_ENTROPY_GATE=$TEACHER_ABS_ENTROPY_GATE"
        echo "  TEACHER_PROB_FLOOR=$TEACHER_PROB_FLOOR"
        echo "  (v7) CTX=$TEACHER_CONTEXT_MODE FB=$INCLUDE_ENVIRONMENT_FEEDBACK TRSUCC=$TRAIN_ON_SUCCESS"
        echo "  (fixed) GATE=$ENTROPY_GATE TOPK=$DISTILL_TOPK GMM=$GROUP_MEAN_MODE NORM_STD=$NORM_ADV_BY_STD"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=ENTROPY_GATE_TOLERANCE=${ENTROPY_GATE_TOLERANCE} --env=TEACHER_ABS_ENTROPY_GATE=${TEACHER_ABS_ENTROPY_GATE} --env=TEACHER_PROB_FLOOR=${TEACHER_PROB_FLOOR} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=ENTROPY_FLOOR=${ENTROPY_FLOOR} --env=ENTROPY_PENALTY_COEFF=${ENTROPY_PENALTY_COEFF} --env=REMOVE_THINKING_FROM_DEMONSTRATION=${REMOVE_THINKING} --env=INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK} --env=ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=MAX_ERRORS_IN_POOL=${MAX_ERRORS_IN_POOL} --env=ERROR_ANSWER_MAX_CHARS=${ERROR_ANSWER_MAX_CHARS} --env=TRAIN_ON_SUCCESS=${TRAIN_ON_SUCCESS} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
            --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} \
            2>&1)
        SUBMIT_EXIT=$?
        echo "$SUBMIT_OUTPUT"
        if [ $SUBMIT_EXIT -ne 0 ]; then
            echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi

done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
