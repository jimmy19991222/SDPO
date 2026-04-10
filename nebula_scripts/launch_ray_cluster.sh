#!/bin/bash
set -e
set -x

# ========== 参数解析 ==========
SCRIPT_PATH=""
WORLD_SIZE=""
JOB_NAME=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --script_path=*) SCRIPT_PATH="${1#*=}" ;;
        --world_size=*) WORLD_SIZE="${1#*=}" ;;
        --job_name=*) JOB_NAME="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export WORLD_SIZE=$WORLD_SIZE
export JOB_NAME=$JOB_NAME

echo "SCRIPT_PATH = $SCRIPT_PATH"
echo "WORLD_SIZE  = $WORLD_SIZE"
echo "JOB_NAME    = $JOB_NAME"

# ── 激活自定义 conda 环境（若存在）───────────────────────────────
# 必须在 Ray start 之前激活，让 Ray worker 继承正确的 Python 环境
# 直接 export PATH 而非 conda activate，避免非交互式 shell 下 conda hook 未初始化问题
CONDA_ENV_NAME="sdpo_env"
CONDA_ENV_BIN="/opt/conda/envs/${CONDA_ENV_NAME}/bin"
if [ -d "${CONDA_ENV_BIN}" ]; then
    export PATH="${CONDA_ENV_BIN}:${PATH}"
    echo "Activated conda env: ${CONDA_ENV_NAME} (${CONDA_ENV_BIN})"
else
    echo "[WARN] conda env '${CONDA_ENV_NAME}' not found at ${CONDA_ENV_BIN}, using system Python"
fi

# ── 在 ray start 之前设置环境变量 ──
# Ray worker 进程从 ray daemon 继承环境变量，必须在 ray start 前设置

# 1. PYTHONPATH: 让 worker 加载自定义的 verl
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 2. CUDA 环境
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
    export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
    echo "Detected $GPU_COUNT GPUs, setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 3. 禁用 deepspeed triton 功能
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# 4. vLLM / PyTorch 配置
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# 5. SwanLab 配置
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"

echo "PYTHONPATH = $PYTHONPATH"

# ========== 清理超长环境变量 ==========
clean_path() {
    echo "$1" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
}
export LD_LIBRARY_PATH=$(clean_path "$LD_LIBRARY_PATH") || true
export PATH=$(clean_path "$PATH") || true

# ========== 启动 Ray 集群 ==========
if [ "$RANK" -eq 0 ]; then
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray 2>/dev/null || true
    rm -rf /tmp/ray_session_* 2>/dev/null || true
    rm -rf ~/.ray 2>/dev/null || true
    sleep 3
    ray start --head --dashboard-host=0.0.0.0
    sleep 20
    echo "Ray head started"
else
    ray start --address="$MASTER_ADDR:6379"
fi

ray status

# ========== Rank 0 执行训练 ==========
if [ "$RANK" -eq 0 ]; then
    bash $SCRIPT_PATH
fi
