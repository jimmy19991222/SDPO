#!/bin/bash
set -e  # 遇到错误立即停止

export PYTHONPATH=.:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com

# ============================================================
# Step 1: 生成原始 json
# ============================================================
echo "=== Step 1: 生成原始 json ==="

for DOMAIN in Biology Chemistry Material Physics; do
    LOWER=$(echo $DOMAIN | tr '[:upper:]' '[:lower:]')
    echo "处理 $DOMAIN..."
    mkdir -p datasets/sciknoweval/${LOWER}
    python data/load_dataset.py \
        --dataset_name $DOMAIN \
        --output_path datasets/sciknoweval/${LOWER}/${LOWER}.json
    echo "✅ $DOMAIN 完成"
done

python data/load_dataset.py --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json

# ============================================================
# Step 3: train/test 切分
# # ============================================================
echo ""
echo "=== Step 3: train/test 切分 ==="

for DOMAIN in biology chemistry material physics; do
    echo "切分 $DOMAIN..."
    python data/split_tasks.py \
        --json_path datasets/sciknoweval/${DOMAIN}/${DOMAIN}.json \
        --output_dir datasets/sciknoweval/${DOMAIN} \
        --test_ratio 0.1 \
        --seed 42
    echo "✅ $DOMAIN 切分完成"
done

python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6

# ============================================================
# Step 4: 生成 parquet
# ============================================================
echo ""
echo "=== Step 4: 生成 parquet ==="

for DOMAIN in biology chemistry material physics; do
    echo "预处理 $DOMAIN..."
    python data/preprocess.py \
        --data_source datasets/sciknoweval/${DOMAIN}
    echo "✅ $DOMAIN parquet 完成"
done

python data/preprocess.py \
    --data_source datasets/lcb_v6

python data/preprocess.py \
    --data_source datasets/tooluse

# ============================================================
# Step 5: 验证最终 parquet
# ============================================================
echo ""
echo "=== Step 5: 验证最终 parquet ==="

for DOMAIN in biology chemistry material physics; do
    echo "验证 $DOMAIN..."
    python3 -c "
import pandas as pd
df = pd.read_parquet('datasets/sciknoweval/${DOMAIN}/train.parquet')
print('  列名:', df.columns.tolist())
print('  样本数:', len(df))
prompt = df.iloc[0]['prompt']
roles = [m['role'] for m in prompt]
print('  prompt roles:', roles)
if 'system' in roles:
    idx = roles.index('system')
    print('  system内容预览:', prompt[idx]['content'][:80])
    print('  ✅ system prompt 存在')
else:
    print('  ❌ system prompt 不存在')
gt = df.iloc[0]['reward_model']
print('  ground_truth:', gt)
print('  data_source:', df.iloc[0]['data_source'])
"
done

echo ""
echo "=== 全部完成 ==="

