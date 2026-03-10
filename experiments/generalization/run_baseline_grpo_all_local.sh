#!/bin/bash

# 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
export _NEBULA_USER_ID=435371

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_NAME="baseline_grpo"

DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(8 32)

LRS=(1e-5 1e-6)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)

# Job Submission Function
submit_job() {
    local exp_name="$1"
    local data_path="$2"
    shift 2
    local script_args=("$@")

    # 构建完整的命令行参数
    local args_string=""
    for arg in "${script_args[@]}"; do
        args_string+=" $arg"
    done

#     local setup_cmds="pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0 --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/; \
# pip install -e . --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/; \
# pip install --upgrade wandb --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/; \
# export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"
    local setup_cmds="export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"

    # 直接调用训练脚本
    local run_cmd="bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string"

    local full_command="bash -c '$setup_cmds; $run_cmd'"
    # local full_command="bash -c '$run_cmd'"


    if [ "$DRY_RUN" = true ]; then
        echo "--------------------------------------------------------------"
        echo "Dry-run would start process:"
        echo "$full_command"
    else
        echo "Starting job: $exp_name"
        eval "$full_command"
    fi
}

# Main Loop
for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                        
                        # 构造唯一实验名
                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                        EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}"

                        # 参数数组（每个参数作为单独的元素）
                        SCRIPT_ARGS=(
                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                            "trainer.group_name=GRPO-generalization"
                            "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                            "actor_rollout_ref.actor.optim.lr=$LR"
                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                            "actor_rollout_ref.model.path=$MODEL_PATH"
                            "algorithm.rollout_correction.rollout_is=token"
                            "actor_rollout_ref.rollout.val_kwargs.n=16"
                        )

                        # 串行执行
                        submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                    done
                done
            done
        done
    done
done
