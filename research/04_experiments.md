# 04 — 实验状态 (DPO-TGS V2.5)

> **最近更新**: 2026-05-17 21:04 (Phase B 4 nebula tasks 提交完成)
> **当前 commit**: `299eec7` on `teacher-guided-intervention`
> **SwanLab project**: `DPO-TGS`

---

## §1 一屏现状

### 当前优先级
| 槽位 | 任务 | 状态 | 优先级 |
|---|---|---|---|
| **Notebook 4-GPU** | smoke + innov 双阶段验证 (commit `299eec7`) | ⏳ 待用户跑 | 🔴 P0 (验证 4 个 bugs 都修好) |
| **Nebula** | Phase B 4 tasks 排队中 | 🔄 RUNNING | 🟡 P0 (论文 main result + 3 ablation) |
| **下一步** | 等 nebula 第一波结果 | 待 ~1.5h 后看 trajectory | — |

### 论文 5 大风险 (持续监控)
1. ❌ **单 seed 方差 7.5%** (GRPO biology v2=0.660 vs v3=0.585) → 未来必须 ≥3 seeds 报 mean±std
2. ⚠️ **未跑 OOD 评测** (chemistry/physics/material) → 论文必须加多 subject
3. ⚠️ **DPO-TGS V2.5 真效果还没实测** → Phase B 4 个 task 跑完才能下结论
4. ⚠️ **Length collapse / explosion 三联画** → 必须监控 `dpo/length_ratio_*`, `actor/entropy`
5. ⚠️ **Entropy collapse** (Meta Figure 3) → 监控 `dpo/sigma_neg_margin_mean` (梯度幅度)

---

## §2 已提交 Nebula 任务时间线

### 第 1 批 (2026-05-17 01:32) — commit `2ea25d9`
**包含 1 个 critical bug**: parametric.sh 把 `sdpo_ref_template="Refer to this correct answer: {r}\n"` 传 cli,Hydra grammar 解析 `{r}` 失败 → **全部 fail at hydra init**。

| Task ID | 备注 |
|---|---|
| `0e53862c61fa4076b97569bd45e4a8d8` | Baseline V2 (cc, α=1.0, na=2) |
| `a49fbf624ee14616b2e44496f17ad18d` | hybrid_init_chain |
| `da3dc4dcaa544f4f97e6390cbd510fd0` | n_attempts=4 |
| `d9f0d3c5761d4345975c1e4524f826a7` | α=0.5 mix |

### 第 2 批 (2026-05-17 01:41) — commit `5d5bfdf`
修了第 1 批的 Hydra bug,但**仍含 3 bugs**:
- `gen_batch` 缺 `reward_model` → KeyError at `_rescore_reward(y_init)` (必崩,Phase 1 之后立即崩)
- `logp_actor` undefined NameError (启用 ② Teacher-Anchored 时崩)
- `_vectorized_pair_advantage` 用 `mask.dtype` (long),整数除法把 advantage 截断到 0 (silent bug)

| Task ID | 备注 |
|---|---|
| `a28b1ae2c6dd4c9c98fbcc36478a1019` | Baseline V2 |
| `e8d779d4b6154337bce9c89cc56484c3` | hybrid_init_chain |
| `841acda959cf4810b5535829383185b2` | n_attempts=4 |
| `5f8b3915c62846b0a50adea2d4a557f6` | α=0.5 mix |

→ **任由 fail**,任务起来几分钟内就崩 (KeyError 立即触发)。

### 第 3 批 (2026-05-17 21:04) — commit `299eec7` ⭐ 当前主跑

修复全部 4 个 bugs。已排队 4 jobs:

| # | Job | 关键参数 | Task ID | Logview |
|---|---|---|---|---|
| 1 | **V2 baseline** (无 innovations) | na=2, α=1.0, cc, all OFF | `c68f0bd8e0564b4a97bcafbc5ceaa029` | [view](https://nebula2.alibaba-inc.com/log/logview/xdl/view?task_id=c68f0bd8e0564b4a97bcafbc5ceaa029) |
| 2 | **V2.5 main (① ② ③)** | na=2, α=1.0, cc, **all 3 ON**, β_tok=0.2, β_cont=0.05, ΔR=linear | `d6625deddb6a4220bf0dcc4a599db1ab` | [view](https://nebula2.alibaba-inc.com/log/logview/xdl/view?task_id=d6625deddb6a4220bf0dcc4a599db1ab) |
| 3 | **hybrid pair** (Meta) | na=2, α=1.0, **hybrid_init_chain** | `da41e619621744d9b935ce981645182f` | [view](https://nebula2.alibaba-inc.com/log/logview/xdl/view?task_id=da41e619621744d9b935ce981645182f) |
| 4 | **n_attempts=4** (chain depth) | **na=4**, α=1.0, cc | `8524929500414912b441c83db69c0e60` | [view](https://nebula2.alibaba-inc.com/log/logview/xdl/view?task_id=8524929500414912b441c83db69c0e60) |

每个 250 step ~8h,4 个并行排队。

Submission log: `logs/dpo_tgs_phaseB_20260517_210253.log`

---

## §3 Phase A: Notebook Smoke (4-GPU,本地,~30 min)

```bash
cd /Users/awesome_jimmy/lazada/SDPO
git pull origin teacher-guided-intervention   # 拉 299eec7

# A1: V1 baseline plumbing (~15 min)
./run_notebook_dpo_tgs.sh smoke 2>&1 | tee logs/dpo_tgs_smoke_v1_fixed.log

# A2: V2.5 全 innovations (~15 min)
./run_notebook_dpo_tgs.sh innov 2>&1 | tee logs/dpo_tgs_innov_v1.log
```

### 通过标志

| 阶段 | 必看 metric |
|---|---|
| **A1 (smoke, V1 baseline)** | `dpo/pairs_total > 0`, `dpo/sigma_neg_margin_mean ∈ (0, 1)`, `dpo/grpo_fallback_rate < 1.0`, `val-core/dpo_implicit_reward_accuracy` 出现 |
| **A2 (innov, V2.5 全开)** | A1 全部 + `dpo/teacher_anchored_coverage > 0`, `dpo/teacher_anchored_kl_chosen_mean` 出现 (验证 logp_old bug 修了), `dpo/t_star_mean_position > 0`, `dpo/delta_r_max ≥ 0` |

### 任一异常立刻贴 log
- ❌ `KeyError`/`NameError`/`AttributeError` → 还有 bug
- ❌ training loss = 0 / 不动 → advantage 全 0 (dtype bug 没修干净)
- ❌ 训练步数卡在 0 → rollout 卡死 (chain 死循环可能)

---

## §4 Phase B: 4 Nebula 任务 (已提交,等结果)

### 实验 matrix (论文 main + 3 ablation)

| # | Job | 改的维度 vs Job 1 | 答的问题 |
|---|---|---|---|
| 1 | V2 baseline | (基准) | DPO-TGS V2 整体是否优于 GRPO/SDPO? |
| 2 | V2.5 main (① ② ③) | + 全 3 loss innovations | V2.5 三个 innovation 一起开 vs baseline,**论文 main result** |
| 3 | hybrid pair | + best-vs-worst init pair (Meta) | best-vs-worst pool 是否额外提升? |
| 4 | n_attempts=4 | + chain 加深 (Samplers ICLR-25) | chain depth = mixing 强度,边际收益? |

### 期望结果方向 (基于理论)

- **Job 1 应该 ≈ GRPO baseline** (V2 plumbing 通,但纯 v1 DPO 信号有限)
- **Job 2 应该 > Job 1**: 3 innovations 各贡献,合起来明显 (target +2~5 pts)
- **Job 3 vs Job 1**: 不确定,Meta paper 实证有效但效果边际
- **Job 4 vs Job 1**: 不确定,chain 加深 compute 翻倍 (~16h),收益要看

### 跑通 30 min 内必看 (确认 plumbing)

打开 logview 链接,Phase B 任意任务起来后:
- ✅ 训练 progress bar 真在动 (step ≥ 5)
- ✅ swanlab `DPO-TGS` project 里 4 个 experiment 都注册了
- ✅ `dpo/pairs_total > 0` (chain 真在产 pair)
- ✅ `dpo/sigma_neg_margin_mean ∈ (0, 1)` 不是 NaN
- ✅ `dpo/grpo_fallback_rate < 1.0` (DPO 真在用,不是全 fallback)

### 跑到 ~50 step / ~1.5h 看 trajectory

主要指标:
- `val-core/sciknoweval/biology/acc/mean@16` (主指标,期望 trajectory 上升)
- `val-core/dpo_implicit_reward_accuracy` (DPO 标准 val 指标,期望升到 0.6+)
- `dpo/sigma_neg_margin_mean` 是否随训练萎缩 → OFS-DPO 梯度消失警告
- `dpo/length_ratio_chosen_over_rejected` 是否爆涨 → OAIF 长度偏置警告
- `dpo/grpo_fallback_rate` 是否过高 → DPO 信号稀疏

---

## §5 Phase C: V2.5 单 innovation 解构 (待 Phase B 结果出来后决定)

如果 Job 2 (全开) 显著优于 Job 1 (baseline),Phase C 解构哪个 innovation 贡献最大。规划 4 个 task:

| Phase C # | Innovation 配置 |
|---|---|
| C1 | 仅 ① Causal-Localized (`DPO_CAUSAL_LOCALIZE=True DPO_BETA_TOKEN=0.2 DPO_BETA_CONTINUATION=0.05`) |
| C2 | 仅 ② Teacher-Anchored (`DPO_USE_TEACHER_ANCHORED_REF=True`) |
| C3 | 仅 ③ ΔR-Weighted (`DPO_DELTA_R_WEIGHT_MODE=linear`) |
| C4 | ② + ③ 不带 ① (验证 ① 是否单独有用) |

这是论文必要的 ablation table,主表证 V2.5 有用 → ablation table 解构每个 innovation 的边际贡献。

---

## §6 SwanLab 关注的 16 个 Metric

详细 metric 表见 [02_dpo_tgs_design.md §6](02_dpo_tgs_design.md)。这里给出**早期预警 watch list**:

### P0 早期预警 (跑 50 step 内必看)

| Metric | 健康范围 | 异常信号 |
|---|---|---|
| `dpo/sigma_neg_margin_mean` | 0.2 ~ 0.5 | < 0.05 → OFS-DPO 梯度消失 |
| `dpo/length_ratio_chosen_over_rejected` | ~ 1.0 (±0.2) | > 1.5 → OAIF 长度偏置 |
| `dpo/grpo_fallback_rate` | < 0.5 | > 0.8 → chain 信号太稀疏,DPO 没在学 |
| `dpo/implicit_reward_accuracy` | > 0.55 升到 0.8+ | 卡在 0.5 不动 → β 太大 / pair 质量差 |
| `val-core/sciknoweval/biology/acc/mean@16` | 单调升 | 跌头 → length collapse 警告 |

### P1 V2.5 innovation 健康度

| Metric | 含义 |
|---|---|
| `dpo/teacher_anchored_coverage` | ② 多少样本拿到 OPSD teacher logp |
| `dpo/teacher_anchored_kl_chosen_mean` | ② teacher vs ref logp gap |
| `dpo/delta_r_max`, `dpo/delta_r_std` | ③ ΔR 分布形状 (帮选 weight mode) |
| `dpo/t_star_mean_position`, `dpo/t_star_std_position` | ① divergence 位置统计 |
| `dpo/chain_attempt_success_rate@k` | k=1, k=2 ... chain attempt 边际成功率 |

---

## §7 Baseline 数据 (历史已有)

### val-core/sciknoweval/biology/acc/mean@16

| Method | seed v2 | seed v3 | best |
|---|---|---|---|
| **GRPO** | **0.660** ⚠️ (length=17 蒙对) | **0.585** (length=343 健康) | 单 seed 方差 7.5% |
| SDPO | 0.590 | 0.574 | 0.59 |
| TGDI-v3p1-aexEOS | 0.5737 | — | running (历史) |
| PS-v2b | 0.5525 | — | running (历史) |

### 数据集难度顺序 (单 seed 取 best)

material (0.79) > physics (0.78) > chemistry (0.78) > tooluse (0.71) > **biology (0.66)** ← 最难

→ DPO-TGS 主战场 biology;论文要扩到至少 chemistry + physics 才 publishable。

### Length 失败模式三联画

| 失败模式 | 案例 | length | entropy |
|---|---|---|---|
| Length collapse | PS v1 (372→18), GRPO biology v2 (330→17) | 缩到极短 | -90~99% |
| Length explosion | PS v2a (1442), TGDI-gtarg (1686) | 暴涨 | -62~97% |
| Both 异常 + val 仍可高 | GRPO biology v2 val=0.66 length=17 | 4 选 1 蒙对 | -90% |

→ DPO-TGS 必须监控 `dpo/length_ratio_*` + `actor/entropy` 防三种失败。

---

## §8 待跑实验 (Phase B 之后的规划)

### 论文必要 (~ 4-6 weeks)

| Priority | 实验 | 工作量 | 用途 |
|---|---|---|---|
| 🔴 P0 | Phase B 4 task 跑完 + 出 main result table | ~8h wallclock | 论文 headline |
| 🔴 P0 | Phase C 4 task innovation ablation | ~32h GPU (并行 ~8h) | 论文 ablation table |
| 🟡 P1 | 多 seed 重复 (≥3 seeds × 2 best configs) | ~48h GPU | 风险 #1 (单 seed 方差) |
| 🟡 P1 | 多 subject 扩展 (chemistry, physics) × 2 configs | ~32h GPU | 风险 #2 (无 OOD) |
| 🟢 P2 | β sweep (0.05 / 0.1 / 0.5) | ~24h GPU | hyperparameter section |
| 🟢 P2 | n_init / n_attempts 2D sweep | ~32h GPU | rollout shape ablation |
| ⚪ P3 | DPO val mode 在多 subject 上的 implicit_reward_accuracy 趋势 | 自动随训练 | 验证 DPO val 与 task acc 相关性 |

### 论文外探索 (Future Work,后续可能升级)

| 探索 | 依据 |
|---|---|
| EMA teacher 作 OFS-DPO Slow-Module + chase regularizer | OFS-DPO 启发 ③ |
| Listwise chain DPO (Plackett-Luce) for n_attempts ≥ 3 | Samplers-DPO 启发 ⑤ + chain depth |
| Asymmetric Push-Anchor DPO (RPO RD loss token-localized) | RPO 启发的 v2 升级路径 |
| 跨任务: DPO-TGS 在 math (Olympiad / GSM8K) 上 | 多任务泛化 |

---

## §9 工程 TODO

### 已完成 ✅
- [x] V2 adaptive rollout (n_init/n_attempts/SDPO ctx/reselect_t/lineage)
- [x] V1 线性化 DPO loss + α-mix GRPO fallback
- [x] V2.5 三个 loss innovations (① + ② + ③)
- [x] Pair collector (chain_consecutive + hybrid_init_chain)
- [x] 16 SwanLab metric (10 P0+P1 base + 6 innovation diagnostic)
- [x] DPO val mode (implicit_reward_accuracy)
- [x] ray_trainer dispatch + 4 bug fixes
- [x] Hydra config (16 knob,默认 v1 backwards-compat)
- [x] Nebula 提交脚本 (sweep + parametric)
- [x] Notebook 4-GPU local smoke (smoke/innov/pair/full)
- [x] design doc 重构 (research/ + HTML)

### 待做

| Priority | 项 | 工作量 |
|---|---|---|
| 🟡 P1 | 早停机制 (val 连续 N 个评估点低于 best × 0.9 即停) | ~50 行 |
| 🟢 P2 | swanlab-analyzer 自动 pull 脚本 | ~80 行 |
| ⚪ P3 | proper DPO loss (代替 v1 线性化,需 actor surgery + pair-aware micro-batch) | TBD |

---

## §10 Commit 链 (DPO-TGS V2.5)

```
299eec7  fix(dpo_tgs): 4 bugs blocking smoke + extend notebook script for V2.5
30ac745  docs(dpo_tgs): append V2.5 update section to design doc
2ea25d9  feat(dpo_tgs): add run_notebook_dpo_tgs.sh
ce3db1a  feat(dpo_tgs): On-Policy DPO + Teacher-Guided Sampling V2 (adaptive rollout)
7afa10f  feat(dpo_tgs): 3 teacher-guided DPO loss innovations
5d5bfdf  fix(dpo_tgs): drop CLI override for sdpo_ref_template (Hydra grammar conflict)
4f85bde  feat(tcca_lite): pivot from chain → TCCA-Lite (TCCA-Lite era,前 DPO-TGS pivot)
```

历史 commit 见 `git log teacher-guided-intervention`。

---

## §11 关键文件路径速查

详细列表见 [README.md "代码位置"](README.md)。

| 类别 | 文件 |
|---|---|
| **核心代码** | `verl/trainer/ppo/dpo_tgs/{adaptive_rollout, dpo_loss, pair_collector}.py` |
| **dispatch** | `verl/trainer/ppo/ray_trainer.py` |
| **Hydra config** | `verl/trainer/config/dpo_tgs.yaml` |
| **Nebula sweep** | `nebula_scripts/submit_dpo_tgs_sweep.sh` |
| **Nebula parametric** | `nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh` |
| **Local 4-GPU smoke** | `run_notebook_dpo_tgs.sh` |
| **设计文档 (本文档目录)** | `research/{README, 01_evolution, 02_dpo_tgs_design, 03_theory_anchor, 04_experiments}.md` |
| **理论锚点完整版** | `papers/raw/opd_papers/OPD_Deep_Analysis.html` |
