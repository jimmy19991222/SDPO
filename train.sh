source /opt/conda/bin/activate
conda activate sdpo_env
export PATH="/home/loujieming.ljm/.conda/envs/sdpo_env/bin:$PATH"

mkdir -p logs

# ============================================================================
# TASD 实验（按优先级排序）
# ============================================================================

# 1. EMA teacher + entropy bonus 扫描（最重要，当前研究重点）
#    包含：teacher_prob/log_teacher_prob × lr × entropy_coeff × ema/none × norm_std
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_tasd_ema_teacher_local.sh 2>&1 | tee logs/run_tasd_ema_teacher_local_$(date +%Y-%m-%d_%H-%M-%S).log

# 2. 新 reward 类型探索（certainty/top1/topk）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_tasd_new_rewards_local.sh 2>&1 | tee logs/run_tasd_new_rewards_local_$(date +%Y-%m-%d_%H-%M-%S).log

# 3. teacher_prob_relative（topk=1000）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_tasd_relative_rewards_local.sh 2>&1 | tee logs/run_tasd_relative_rewards_local_$(date +%Y-%m-%d_%H-%M-%S).log

# 4. Rich Feedback 数据集（lcb_v6）上的 TASD
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/rich_feedback/run_tasd_local.sh 2>&1 | tee logs/run_tasd_rich_feedback_local_$(date +%Y-%m-%d_%H-%M-%S).log

# ============================================================================
# Baseline（如果还没跑过）
# ============================================================================

# 5. GRPO baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_baseline_grpo_all_local.sh 2>&1 | tee logs/run_baseline_grpo_all_local_$(date +%Y-%m-%d_%H-%M-%S).log

# 6. SDPO baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/generalization/run_sdpo_all_local.sh 2>&1 | tee logs/run_sdpo_all_local_$(date +%Y-%m-%d_%H-%M-%S).log
