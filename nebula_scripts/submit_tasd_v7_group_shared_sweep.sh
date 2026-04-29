#!/bin/bash
# =============================================================================
# TASD v7: Group-shared teacher context + 错例池 + GT 泄漏修复 - Nebula 批量提交
#
# 核心变化（对比 v6 fbEnhanced）：
#   1. Group-shared teacher context：同一 uid 所有 rollout 共享一段 teacher prompt
#      = 问题 + 聚合错例池（去重）+ 单条 reference answer
#      → 恢复 v5 group_mean 无偏（之前 per_rollout feedback 破坏了这一性质）
#   2. Feedback 去 GT 泄漏：tooluse feedback 只保留错误类别和位置（缺哪个 key/
#      哪个 key 的 value 错），不再泄漏 GT 的具体 action 名和 action_input value
#   3. 错例池去重：按 (tag, normalized_answer) dedup，带频率聚合展示
#   4. _extract_final_answer：按 data_source 分发（tooluse/sciknoweval/gpqa/
#      mmlu_pro/math），正确剥离 <think>/<reasoning>
#   5. 四档诚实 bad case 构建（tooluse）：
#      - wrong_semantics / invalid_action_name / partial_format / no_format
#   6. train_on_success 开关：False 时 success rollout mask=0（仅作 reference）
#      → 隔离 "group shift" 与 "on-success 伪蒸馏" 两种效应
#
# v7 矩阵（固定其他超参在 v6 最优，仅 sweep teacher context 相关）：
#   R1: ctx=per_rollout   fb=off  trSucc=T  → v5 对照（纯 solution-only）
#   R2: ctx=group_shared  fb=on   trSucc=T  → 主力（v7 核心）
#   R3: ctx=group_shared  fb=on   trSucc=F  → 消融（去 on-success 伪蒸馏）
#   R4: ctx=per_rollout   fb=on   trSucc=T  → v6 对照（含 feedback GT 泄漏修复）
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_v7_group_shared_sweep.sh [--dry-run]
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
PROJECT_NAME="TASD-v7"

# ── 数据集：v7 以 tooluse 优先验证（其次可加 sciknoweval/bio）───────────
DATASETS=(
    "tooluse"
    # "sciknoweval/biology"
    # "sciknoweval/material"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# v7 矩阵：4 run 组合（ctx_mode, fb_on, train_on_success, tag）
# 用 ":" 分隔字段，避免多重笛卡尔积
# 格式: "TEACHER_CONTEXT_MODE:INCLUDE_ENVIRONMENT_FEEDBACK:TRAIN_ON_SUCCESS:RUN_TAG"
# =============================================================================
V7_MATRIX=(
    "per_rollout:False:True:ctxPer-fbOff-trSuccT"
    "group_shared:True:True:ctxGrp-fbOn-trSuccT"
    "group_shared:True:False:ctxGrp-fbOn-trSuccF"
    "per_rollout:True:True:ctxPer-fbOn-trSuccT"
)

# ── 错例池参数（仅 group_shared 生效）────────────────────────────────
MAX_ERRORS_IN_POOL="8"
ERROR_ANSWER_MAX_CHARS="1024"

# =============================================================================
# 固定超参（沿用 v6 最优配置，聚焦 teacher context ablation）
# =============================================================================
REWARD_TYPE="teacher_log_prob"
ENTROPY_GATE="hard_keep_reward"
ENTROPY_GATE_RATIO="1.0"
ENTROPY_GATE_TOLERANCE="0.0"
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
DISTILL_TOPK="256"
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"   # 始终把成功 rollout 纳入 batch；是否计 loss 由 TRAIN_ON_SUCCESS 决定
REMOVE_THINKING="True"
# v6 策略：feedback 与 solution 并存
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
for ENTRY in "${V7_MATRIX[@]}"; do
    IFS=':' read -r TEACHER_CONTEXT_MODE INCLUDE_ENVIRONMENT_FEEDBACK TRAIN_ON_SUCCESS RUN_TAG <<< "$ENTRY"

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    # v7 标签示例：-ctxGrp-fbOn-trSuccT-v7-fbFixGT
    JOB_NAME="TASD-${DATASET_SHORT}-${RUN_TAG}-v7-fbFixGT-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET"
        echo "  TEACHER_CONTEXT_MODE=$TEACHER_CONTEXT_MODE"
        echo "  INCLUDE_ENVIRONMENT_FEEDBACK=$INCLUDE_ENVIRONMENT_FEEDBACK"
        echo "  TRAIN_ON_SUCCESS=$TRAIN_ON_SUCCESS"
        echo "  MAX_ERRORS_IN_POOL=$MAX_ERRORS_IN_POOL ERROR_ANSWER_MAX_CHARS=$ERROR_ANSWER_MAX_CHARS"
        echo "  (fixed) REWARD_TYPE=$REWARD_TYPE GATE=$ENTROPY_GATE TOPK=$DISTILL_TOPK"
        echo "  (fixed) GROUP_MEAN_MODE=$GROUP_MEAN_MODE NORM_ADV_BY_STD=$NORM_ADV_BY_STD CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=ENTROPY_GATE_TOLERANCE=${ENTROPY_GATE_TOLERANCE} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=ENTROPY_FLOOR=${ENTROPY_FLOOR} --env=ENTROPY_PENALTY_COEFF=${ENTROPY_PENALTY_COEFF} --env=REMOVE_THINKING_FROM_DEMONSTRATION=${REMOVE_THINKING} --env=INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK} --env=ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=${ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=MAX_ERRORS_IN_POOL=${MAX_ERRORS_IN_POOL} --env=ERROR_ANSWER_MAX_CHARS=${ERROR_ANSWER_MAX_CHARS} --env=TRAIN_ON_SUCCESS=${TRAIN_ON_SUCCESS} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
