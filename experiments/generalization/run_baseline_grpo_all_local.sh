#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false  # ← 改为false，避免警告
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

export CUDA_VISIBLE_DEVICES=0,1,2,3

DRY_RUN=false
SKIP_EXISTING=true   # ← 新增：跳过已完成的实验

CONFIG_NAME="baseline_grpo"
SDPO_HOME="/home/loujieming.ljm/SDPO"

# =============================================
# 精简后的超参数（基于论文结论）
# =============================================
DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(32)      # ← 只跑on-policy配置
LRS=(1e-5)                 # ← on-policy用1e-5
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    # "allenai/Olmo-3-7B-Instruct"  # ← 先注释掉，等Qwen跑完
)

mkdir -p ./logs
PROGRESS_FILE="./logs/progress.txt"

submit_job() {
    local exp_name="$1"
    local data_path="$2"
    shift 2
    local script_args=("$@")

    # ← 新增：检查是否已完成
    if [ "$SKIP_EXISTING" = true ] && grep -q "DONE: $exp_name" "$PROGRESS_FILE" 2>/dev/null; then
        echo "⏭️  跳过已完成: $exp_name"
        return 0
    fi

    local log_file="./logs/${exp_name}.log"

    local args_string=""
    for arg in "${script_args[@]}"; do
        args_string+=" $arg"
    done

    echo "========================================================"
    echo "▶️  开始: $exp_name"
    echo "📊 数据集: $data_path"
    echo "📝 日志: $log_file"
    echo "🕐 开始时间: $(date)"
    echo "========================================================"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] bash $SDPO_HOME/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string"
        return 0
    fi

    export PYTHONPATH=$SDPO_HOME:$PYTHONPATH

    bash "$SDPO_HOME/training/verl_training.sh" \
        "$exp_name" "$CONFIG_NAME" "$data_path" \
        $args_string \
        > "$log_file" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ 完成: $exp_name ($(date))" | tee -a "$PROGRESS_FILE"
        echo "DONE: $exp_name" >> "$PROGRESS_FILE"
    else
        echo "❌ 失败: $exp_name (exit code: $exit_code)" | tee -a "$PROGRESS_FILE"
        echo "FAILED: $exp_name" >> "$PROGRESS_FILE"
    fi

    echo "⏳ 等待GPU内存释放..."
    sleep 30
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
}

# =============================================
# 统计总job数
# =============================================
TOTAL_JOBS=0
for _ in "${DATA_PATHS[@]}"; do
  for _ in "${TRAIN_BATCH_SIZES[@]}"; do
    for _ in "${ROLLOUT_BATCH_SIZES[@]}"; do
      for _ in "${LRS[@]}"; do
        for _ in "${MODEL_PATHS[@]}"; do
          for _ in "${MINI_BATCH_SIZES[@]}"; do
            ((TOTAL_JOBS++))
          done
        done
      done
    done
  done
done

echo "📋 总计需要运行: $TOTAL_JOBS 个job"
echo "⏱️  预计总时间: ~$((TOTAL_JOBS * 6)) 小时"
echo ""

CURRENT_JOB=0

# =============================================
# 主循环
# =============================================
for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                        ((CURRENT_JOB++))
                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                        EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}"

                        echo ""
                        echo "📌 进度: [$CURRENT_JOB/$TOTAL_JOBS]"

                        SCRIPT_ARGS=(
                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                            "trainer.group_name=GRPO-generalization"
                            "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                            "actor_rollout_ref.actor.optim.lr=$LR"
                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                            "actor_rollout_ref.model.path=$MODEL_PATH"
                            "algorithm.rollout_correction.rollout_is=token"
                            "actor_rollout_ref.rollout.val_kwargs.n=4"
                            "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
                            "trainer.n_gpus_per_node=4"
                            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
                        )

                        submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                    done
                done
            done
        done
    done
done

echo ""
echo "🎉 所有job完成！"
echo "📊 结果摘要："
cat "$PROGRESS_FILE"
