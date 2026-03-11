#!/bin/bash

# 设置 HF 镜像和其他环境变量
export HF_ENDPOINT=https://hf-mirror.com
export _NEBULA_USER_ID=435371
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4

export WANDB_MODE=offline

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"
DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-5)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(0.5)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)

# =============================================================================
# JOB SUBMISSION FUNCTION (Local Version)
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # 环境准备命令
    # local setup_cmds="
    #     pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0 --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/;
    #     pip install -e /home/loujieming.ljm/SDPO --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/;
    #     pip install --upgrade wandb --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/;
    #     export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH
    # "
    local setup_cmds="export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"

    # 构建完整命令
    local log_file="./logs/job_${exp_name}_$(date +%s).log"
    local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $script_args > $log_file 2>&1"
    local full_command="bash -c '$setup_cmds; $run_cmd'"

    if [ "$DRY_RUN" = true ]; then
        echo "=============================================================="
        echo "[DRY RUN] Would execute job: $exp_name"
        echo "$full_command"
    else
        echo "Starting job: $exp_name"
        # 可以改为放到后台运行加日志输出控制：
        # nohup $full_command > "/tmp/${exp_name}.log" 2>&1 &
        
        eval "$full_command"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            # 构造唯一实验名称
                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                            EXP_NAME="FINAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}"

                            # 拼接参数字符串
                            ARGS="
                                data.train_batch_size=$TRAIN_BATCH_SIZE \
                                trainer.group_name=SDPO-generalization \
                                actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
                                actor_rollout_ref.model.path=$MODEL_PATH \
                                actor_rollout_ref.actor.optim.lr=$LR \
                                actor_rollout_ref.actor.ppo_mini_batch_size=32 \
                                actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
                                algorithm.rollout_correction.rollout_is=token \
                                actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
                                actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
                                actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
                                actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
                                actor_rollout_ref.rollout.val_kwargs.n=16 \
                                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                                trainer.n_gpus_per_node=4
                            "
                            
                            # 执行作业
                            submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done
