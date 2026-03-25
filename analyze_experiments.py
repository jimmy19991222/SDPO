"""
实验结果分析脚本
分析 实验结果.json 中的所有指标，输出多维度对比报告
"""

import json
from collections import defaultdict

# ============ 加载数据 ============
with open("实验结果.json") as f:
    data = json.load(f)


def get_metric(exp_metrics, key):
    """获取某个实验的某个指标的最终值（step最大时）"""
    for item in exp_metrics:
        if item["key"] == key:
            try:
                return float(item["value"])
            except (ValueError, TypeError):
                return None
    return None


def get_metric_peak(exp_metrics, key, mode="max"):
    """获取某个指标的历史最优值"""
    for item in exp_metrics:
        if item["key"] == key:
            try:
                if mode == "max":
                    return float(item["max"]["data"])
                else:
                    return float(item["min"]["data"])
            except (ValueError, TypeError):
                return None
    return None


# ============ 实验名缩写映射 ============
def shorten_name(name):
    name = name.split("-20260")[0]  # 去掉时间戳
    replacements = {
        "GRPO-sciknoweval-biology-mbs32-train32-lr1e-5-Qwen3-8B": "GRPO",
        "SDPO-sciknoweval-biology-train32-alpha0.5-lr1e-5-drossTrue-Qwen3-8B": "SDPO",
        "TASD-bio-lr1e-5-rtteacher_prob-nostd-clip5.0-ent1.0-rctoken-isr0-ema0.1-Qwen3-8B": "TASD(ent1.0/isr0)",
        "TASD-bio-lr1e-5-rtteacher_prob-nostd-clip5.0-ent0.1-rctoken-isr0-ema0.1-Qwen3-8B": "TASD(ent0.1/isr0)",
        "TASD-bio-lr1e-5-rtteacher_prob-nostd-clip5.0-ent0.1-rctoken-isr1-ema0.1-Qwen3-8B": "TASD(ent0.1/isr1)",
        "TASD-bio-lr1e-5-rtteacher_prob-nostd-clip5.0-ent1.0-rctoken-isr1-ema0.1-Qwen3-8B": "TASD(ent1.0/isr1)",
    }
    for full, short in replacements.items():
        if full in name or name == full:
            return short
    return name[:40]


experiments = {shorten_name(k): v for k, v in data.items()}
exp_names = list(experiments.keys())

print("=" * 80)
print("实验列表")
print("=" * 80)
for i, name in enumerate(exp_names):
    print(f"  [{i+1}] {name}")

# ============ 关键指标列表 ============
METRICS_OF_INTEREST = {
    # 准确率类
    "acc/mean@16":      "val-core/sciknoweval/acc/mean@16",
    "acc/best@8":       "val-aux/sciknoweval/acc/best@8/mean",
    "acc/maj@8":        "val-aux/sciknoweval/acc/maj@8/mean",
    "acc/worst@16":     "val-aux/sciknoweval/acc/worst@16/mean",
    "acc/std@16":       "val-aux/sciknoweval/acc/std@16",
    # 训练稳定性
    "entropy":          "actor/entropy",
    "grad_norm":        "actor/grad_norm",
    "pg_clipfrac":      "actor/pg_clipfrac",
    "ppo_kl":           "actor/ppo_kl",
    # Advantage 分布
    "adv/mean":         "critic/advantages/mean",
    "adv/max":          "critic/advantages/max",
    "adv/min":          "critic/advantages/min",
    # Reward 分布
    "reward/mean":      "critic/rewards/mean",
    "reward/max":       "critic/rewards/max",
    "reward/min":       "critic/rewards/min",
    # Score 分布（val）
    "score/mean@16":    "val-aux/sciknoweval/score/mean@16",
    "score/best@8":     "val-aux/sciknoweval/score/best@8/mean",
    "score/worst@16":   "val-aux/sciknoweval/score/worst@16/mean",
    # 格式错误率
    "fmt_err/mean@16":  "val-aux/sciknoweval/incorrect_format/mean@16",
    # 序列长度
    "seq_len/mean":     "global_seqlen/mean",
    # rollout 相关
    "rollout_ppl":      "rollout_corr/rollout_ppl",
    "rollout_probs_diff_mean": "training/rollout_probs_diff_mean",
    # reward/acc 分布的 diversity
    "reward/std@16":    "val-aux/sciknoweval/reward/std@16",
    "acc/best@2":       "val-aux/sciknoweval/acc/best@2/mean",
}

# ============ 构建数据表 ============
table = defaultdict(dict)
for exp_name, exp_metrics in experiments.items():
    for metric_alias, metric_key in METRICS_OF_INTEREST.items():
        val = get_metric(exp_metrics, metric_key)
        table[exp_name][metric_alias] = val

# ============ 1. 准确率全面对比 ============
print("\n" + "=" * 80)
print("1. 准确率综合对比（最终step值）")
print("=" * 80)

acc_metrics = ["acc/mean@16", "acc/best@8", "acc/maj@8", "acc/worst@16", "acc/std@16"]
header = f"{'实验':28s}" + "".join(f"{m:>14s}" for m in acc_metrics)
print(header)
print("-" * (28 + 14 * len(acc_metrics)))
for name in exp_names:
    row = f"{name:28s}"
    for m in acc_metrics:
        v = table[name].get(m)
        row += f"{v:>14.4f}" if v is not None else f"{'N/A':>14s}"
    print(row)

# ============ 2. mean@16 vs best@8 gap 分析 ============
print("\n" + "=" * 80)
print("2. mean@16 vs best@8 gap（gap越小=模型越稳定）")
print("=" * 80)

header2 = f"{'实验':28s}{'mean@16':>12s}{'best@8':>12s}{'gap(b8-m16)':>14s}{'consistency%':>14s}"
print(header2)
print("-" * (28 + 12 + 12 + 14 + 14))
for name in exp_names:
    m16 = table[name].get("acc/mean@16")
    b8 = table[name].get("acc/best@8")
    if m16 is not None and b8 is not None:
        gap = b8 - m16
        consistency = (m16 / b8 * 100) if b8 > 0 else 0
        print(f"{name:28s}{m16:>12.4f}{b8:>12.4f}{gap:>14.4f}{consistency:>13.1f}%")
    else:
        print(f"{name:28s}{'N/A':>12s}{'N/A':>12s}{'N/A':>14s}{'N/A':>14s}")

# ============ 3. 训练稳定性指标 ============
print("\n" + "=" * 80)
print("3. 训练稳定性（最终step值）")
print("=" * 80)

stab_metrics = ["entropy", "grad_norm", "pg_clipfrac", "ppo_kl"]
header3 = f"{'实验':28s}" + "".join(f"{m:>14s}" for m in stab_metrics)
print(header3)
print("-" * (28 + 14 * len(stab_metrics)))
for name in exp_names:
    row = f"{name:28s}"
    for m in stab_metrics:
        v = table[name].get(m)
        row += f"{v:>14.4f}" if v is not None else f"{'N/A':>14s}"
    print(row)

# ============ 4. Entropy 历史 min（是否崩溃） ============
print("\n" + "=" * 80)
print("4. Entropy 历史最低值（越低越接近mode collapse）")
print("=" * 80)
for name, exp_metrics in experiments.items():
    ent_min = get_metric_peak(exp_metrics, "actor/entropy", mode="min")
    ent_max = get_metric_peak(exp_metrics, "actor/entropy", mode="max")
    ent_final = get_metric(exp_metrics, "actor/entropy")
    print(f"  {name:28s}  init={ent_max:.4f}  final={ent_final:.4f}  min={ent_min:.4f}  drop={ent_max-ent_min:.4f}")

# ============ 5. Advantage 分布 ============
print("\n" + "=" * 80)
print("5. Advantage 分布（最终step）")
print("=" * 80)

adv_metrics = ["adv/mean", "adv/max", "adv/min"]
header5 = f"{'实验':28s}" + "".join(f"{m:>12s}" for m in adv_metrics)
print(header5)
print("-" * (28 + 12 * len(adv_metrics)))
for name in exp_names:
    row = f"{name:28s}"
    for m in adv_metrics:
        v = table[name].get(m)
        row += f"{v:>12.4f}" if v is not None else f"{'N/A':>12s}"
    print(row)

# ============ 6. Reward 分布 ============
print("\n" + "=" * 80)
print("6. 训练 Reward 分布（最终step）")
print("=" * 80)

rew_metrics = ["reward/mean", "reward/max", "reward/min", "reward/std@16"]
header6 = f"{'实验':28s}" + "".join(f"{m:>16s}" for m in rew_metrics)
print(header6)
print("-" * (28 + 16 * len(rew_metrics)))
for name in exp_names:
    row = f"{name:28s}"
    for m in rew_metrics:
        v = table[name].get(m)
        row += f"{v:>16.4f}" if v is not None else f"{'N/A':>16s}"
    print(row)

# ============ 7. 响应长度变化 ============
print("\n" + "=" * 80)
print("7. 响应长度（最终step）")
print("=" * 80)
for name, exp_metrics in experiments.items():
    sl_min_hist = get_metric_peak(exp_metrics, "global_seqlen/mean", mode="min")
    sl_max_hist = get_metric_peak(exp_metrics, "global_seqlen/mean", mode="max")
    sl_final = get_metric(exp_metrics, "global_seqlen/mean")
    print(f"  {name:28s}  final={sl_final:.0f}  hist_min={sl_min_hist:.0f}  hist_max={sl_max_hist:.0f}")

# ============ 8. 格式错误率 ============
print("\n" + "=" * 80)
print("8. 格式错误率（最终step，mean@16）")
print("=" * 80)
for name in exp_names:
    v = table[name].get("fmt_err/mean@16")
    print(f"  {name:28s}  fmt_err_mean@16 = {v:.4f}" if v is not None else f"  {name:28s}  N/A")

# ============ 9. Rollout PPL / Probs Diff ============
print("\n" + "=" * 80)
print("9. Rollout 分布漂移（rollout_ppl 越高=漂移越大）")
print("=" * 80)
for name, exp_metrics in experiments.items():
    ppl = get_metric(exp_metrics, "rollout_corr/rollout_ppl")
    pdiff = get_metric(exp_metrics, "training/rollout_probs_diff_mean")
    ppl_max = get_metric_peak(exp_metrics, "rollout_corr/rollout_ppl", mode="max")
    pdiff_max = get_metric_peak(exp_metrics, "training/rollout_probs_diff_mean", mode="max")
    ppl_str = f"rollout_ppl={ppl:.4f}(max={ppl_max:.4f})" if ppl is not None else "rollout_ppl=N/A"
    pdiff_str = f"probs_diff={pdiff:.4f}(max={pdiff_max:.4f})" if pdiff is not None else "probs_diff=N/A"
    print(f"  {name:28s}  {ppl_str}  {pdiff_str}")

# ============ 10. 综合排名 ============
print("\n" + "=" * 80)
print("10. 综合排名（基于 mean@16 × consistency%）")
print("=" * 80)

scores = []
for name in exp_names:
    m16 = table[name].get("acc/mean@16") or 0
    b8 = table[name].get("acc/best@8") or 1
    consistency = m16 / b8 if b8 > 0 else 0
    composite = m16 * consistency
    scores.append((name, m16, b8, consistency, composite))

scores.sort(key=lambda x: x[4], reverse=True)
print(f"{'排名':<4s}{'实验':28s}{'mean@16':>10s}{'best@8':>10s}{'consistency':>13s}{'composite':>12s}")
print("-" * 80)
for i, (name, m16, b8, cons, comp) in enumerate(scores):
    print(f"  {i+1}   {name:28s}{m16:>10.4f}{b8:>10.4f}{cons:>12.1%}{comp:>12.4f}")

print("\n分析完成！")
