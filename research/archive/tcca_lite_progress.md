# Self-Teacher Advantage 实验进展报告

> **最近更新**：2026-05-16 22:00 (TCCA-Lite pivot)
> **当前分支**：`teacher-guided-intervention` @ `4f85bde`
> **SwanLab Projects**：`PriorShift-Tier1` / `TGDI-Tier3` / `TGDI-local` / `Baselines_v2` / `Baselines_v3`
> **凭证管理**：`.env`（已 gitignore），`set -a && source .env && set +a`

---

## 📌 一屏现状

### 论文核心 claim（**2026-05-16 21:50 pivot: TCCA → TCCA-Lite**）

> **TCCA-Lite**：在失败样本上做单步 counterfactual intervention（OPSD teacher 写 1 token + student 真续写到 EOS），用真实 ΔR 做 per-token causal credit。仅 ~25% compute overhead。

**为什么从 chain rollout pivot 到 TCCA-Lite**：
- ❌ chain rollout (8 iter 串行) 工程复杂度高，~2× compute
- ❌ V1 复用旧 tail → ΔR≡0 bug（reward 基于尾部 answer，没变）
- ✅ TCCA-Lite = 标准 GRPO n=8 + 失败样本单步 counterfactual
- ✅ student **真续写到 EOS** 修复 ΔR≡0 bug
- ✅ OPSD teacher（含 reference answer）提供更强的 corrective signal
- ✅ Additive formula (A_seq + λ·c_t) 保符号语义
- ✅ ~25% overhead → 实用级

**核心 2 个 idea (修正后)**：
- 🎯 **主推**：TCCA-Lite = 单步 OPSD counterfactual + per-token ΔR credit (causal, ~25% overhead)
- 🔬 **Ablation**：λ_div ∈ {0, 0.5, 1.0, 2.0}，OPSD vs EMA teacher

### 实验排名（截至 2026-05-16 18:00）

| Rank | 实验 | val best | dataset | seed | status |
|---|---|---|---|---|---|
| 🥇 | GRPO biology (Baselines_v2) | **0.660** | biology | seed v2 | FINISHED (但 length collapse 蒙对) |
| 🥈 | SDPO biology (Baselines_v2) | 0.590 | biology | seed v2 | FINISHED |
| 🥉 | **TGDI-v3p1-aexEOS** | **0.5737** | biology | seed 1 | **RUNNING s133/250** ⭐ |
| 🥉 | GRPO biology (Baselines_v3) | 0.5850 | biology | seed v3 | FINISHED |
| 5 | SDPO biology (Baselines_v3) | 0.5737 | biology | seed v3 | FINISHED |
| 6 | **PS-v2b** | 0.5525 | biology | seed 1 | **RUNNING s153/250** |
| 7 | PS v1 smoke | 0.5225 | biology | seed 1 | FINISHED (length collapse 18) |
| 8 | TGDI-v3p1-araw | 0.4238 | biology | seed 1 | RUNNING s70/250 |
| ❌ | fix_main / PS-v2a / TGDI-gtarg | — | — | — | KILLED (mode collapse) |

### 现在 3 个槽位的分配

| 槽位 | 任务 | 优先级 | 风险 |
|---|---|---|---|
| **Local 4-GPU** | TCCA-Lite smoke (enable_int=True) | 🔴 最高 | 低（已有 tcca_chain.py helpers） |
| **Nebula slot 1** | RLSD baseline 完整 250-step | 🟢 P0（填论文最大空缺）| 零 |
| **Nebula slot 2** | TCCA-Lite main (λ_div=1.0, base=grpo, smoke 通过后) | 🟡 待定 | Local smoke 验证决定 |

### 论文 5 大风险

1. ❌ **单 seed 方差 7.5%**（GRPO biology v2=0.660 vs v3=0.585）→ 必须 ≥3 seeds 报 mean±std
2. ❌ **RLSD baseline 数据缺失** → Nebula slot 1 解决
3. ❌ **TCCA-Lite 真 ΔR 路径还没 smoke 验证** → Local smoke 解决（已有完整实现）
4. ❌ **只跑 biology，没 OOD 评测** → Why-SD-Degrades paper 警示，必须加 chemistry/physics
5. ❌ **之前实验全部单 seed** → 长期需 3 seeds × 2-3 subjects

---

## 1. 方法概要（详见 [paper_idea.md](paper_idea.md)）

**论文核心创新**：**TCCA-Lite (Token-level Causal Credit Assignment)**——在失败样本上做单步 counterfactual intervention（OPSD teacher 写 1 token + student 真续写到 EOS），用真实 ΔR 做 per-token causal credit。这是 OPD 综述中 token 信号演化路径的下一步：
`distribution matching` → `启发式 entropy×divergence` → `统计学 log-ratio` → **真因果 ΔR**

**论文 framing**：
- 🎯 **主推**：TCCA-Lite = 单步 OPSD counterfactual + per-token ΔR credit (~25% overhead)
- 🔬 **Ablation**：λ_div ∈ {0, 0.5, 1.0, 2.0}，OPSD vs EMA teacher

**TCCA-Lite 公式 (简版)**：
```
对 failed sample y, t* = argmax divergence (排尾 8 token)
teacher 写 1 token at t* (OPSD context), student 真续写到 EOS → y'
ΔR = R(y') - R(y)
c_t[y,   t*] = -ΔR  (negative credit at student's wrong choice)
c_t[y',  t*] = +ΔR  (positive credit at teacher's choice)
response_mask[y', :t*] = 0  (Layer 2: shared prefix off)

A_t = (A_seq + λ_div · clip(c_t, ±1)) · response_mask · length_scale
      └────GRPO base──┘  └─ TCCA modulation ─┘
```

完整 pipeline (Step 0-9)、与既有方法对比、术语词典 → [paper_idea.md](paper_idea.md)
设计演化历史 (TCCA chain → TCCA-Lite pivot) → [tcca_v2_design.md](tcca_v2_design.md)
代码 → `verl/trainer/ppo/bayesian_credit/{intervention_credit.py, intervention_rollout.py, tcca_chain.py}`

---


## 4. 实验结果矩阵

### 4.1 Baseline 矩阵（Baselines_v2 + v3，5 dataset × 2 algo × 2 seed）

#### val-core/sciknoweval/acc/mean@16 best

| Subject | GRPO (seed v2 / v3) | SDPO (seed v2 / v3) | Best | **TGDI-aexEOS** |
|---|---|---|---|---|
| biology | **0.660 / 0.585** | 0.590 / 0.574 | GRPO 0.660 | 0.5737 (RUNNING s133) |
| chemistry | 0.780 / 0.733 | 0.748 / 0.741 | GRPO 0.780 | — |
| physics | **0.782 / 0.744** | 0.650 / 0.649 | GRPO 0.782 | — |
| material | 0.765 / 0.789 | 0.800 / 0.781 | SDPO 0.800 | — |
| tooluse (acc@16) | 0.707 / 0.693 | 0.635 / 0.633 | GRPO 0.707 | — |

#### 数据集难度顺序

material (0.79) > physics (0.78) > chemistry (0.78) > tooluse (0.71) > **biology (0.66)** ← 最难，我们主战场

### 4.2 TGDI 实验完整列表

| # | 名称 | base | enable_int | t* metric | status | step | val best | length | entropy chg |
|---|---|---|---|---|---|---|---|---|---|
| 5 | TGDI-v3p1-aexEOS | prior_shift | False | argmax_excl_eos | 🟢 RUNNING | 133 | **0.5737** | 149 | -47.6% |
| 6 | TGDI-v3p1-araw | prior_shift | False | argmax_raw | 🟢 RUNNING | 70 | 0.4238 | 273 | -14.3% |
| 7 | TGDI-v3p1-gtarg | prior_shift | False | g_t_argmax | ❌ KILLED s28 | 28 | 0.3438 | **1686** | **-97.4%** |
| L1 | LOCAL Phase 1 smoke | prior_shift | False | argmax_excl_eos | ✅ FINISHED | 10 | 0.340 (mean@4) | 206 | -1.4% |
| L2 | LOCAL Phase 2 smoke | grpo | True | argmax_excl_eos | ⚠️ FINISHED 但 step=0 | 0 | — | — | — |

---

## 5. 实验时间线

### 实验 1: `fix_main`（2026-05-13，run id `x4wayha2mahjwm4biri1t`）

- **配置**：`adv_mode=self_teacher`, `use_vce=true`, EMA r=0.1, entropy_coeff=0.001
- **结果**：CRASHED s89/250。val best 0.40 (s75)，entropy -88%，length 346→221
- **结论**：经典 mode collapse 三件套（entropy 崩 + advantage 全程贴边 + response 收缩）。验证了"隐式 V_CE 退化"主假说。EMA 几乎不工作（teacher ≈ student）。
- **影响**：促使我们放弃 self_teacher，转向 Bayesian Credit Assignment 框架。

### 实验 2: PS v1 smoke（2026-05-14，run id `5yioqpizdwa7oc2tk7ue2`）

- **配置**：max_ratio=10, EPS_NORM=1e-6, EMA r=0.05, n=8
- **结果**：FINISHED s250。val best **0.5225** (s130) → val final 0.455。length **372→18**（collapse）
- **诊断**：max_ratio=10 → ĝ_t clip 后归一化破坏 → 短序列 reward/token 密度高 → 正反馈循环 → length collapse
- **修复方案 (v2)**：max_ratio 10→3, length_floor=50, renormalize_after_clip=True

### 实验 3: PS-v2a（2026-05-16，run id `ou4z4zvr2n8nw309jbksi`）— KILLED

- **配置**：v2 P0 only（max_ratio=3, length_floor, **renormalize=False**）
- **KILLED 原因**：s33 时 length 反向爆炸到 **1442 (max 1838)** + entropy -62.7% → 复现 fix_main 模式
- **结论**：renormalize_after_clip 是 critical 防护。光有 max_ratio + length_floor 不够。

### 实验 4: PS-v2b（2026-05-16，run id `42zk2eqzhfcvw994v7m04`）

- **配置**：v2 P0+P1（max_ratio=3, length_floor, **renormalize=True**）
- **进展**：s35 val 0.326 (中期诊断,误判失败) → s153 val **0.5525**（持续爬升）
- **教训**：**中期 val 低不代表失败**，需更多 step
- **当前关注**：entropy -78.4% 接近 fix_main 临界，剩余 100 step 是否会 collapse

### 实验 5-7: TGDI v3p1 Phase 1 sweep（2026-05-16，3 jobs Nebula）

3 个 job 同 base (prior_shift) + enable_int=False（=纯 PS v2 等价），区别仅在 t* 选择策略：

| 实验 | t* metric | 结果 | 解读 |
|---|---|---|---|
| 5 (aexEOS) | argmax_excl_eos (排尾 8) | val 0.5737 ⭐ | 排尾 stylistic token 有效 |
| 6 (araw) | argmax 不排尾 | val 0.4238 | 选到 EOS/punct，差 0.15 acc |
| 7 (gtarg) | g_t argmax | KILLED s28 | g_t 选 narrative pivot ≠ student 错 |

**关键 takeaway**：t* 选择策略**实质性影响**最终 val。aexEOS 与 TIP paper 的 Q3 区域假说一致。

### 实验 L1-L2: Local smoke（2026-05-16 dev machine）

- **L1 (Phase 1)**：FINISHED ✅，验证 yaml/estimator/dispatch/SwanLab plumbing OK
- **L2 (Phase 2, enable_int=True)**：⚠️ FINISHED 但 step=0 无任何 metric → 疑似 init 阶段崩。**需要重跑抓 stderr**

---

## 6. 诊断与关键 insight

### 6.1 Length 失败模式三联画（统一解释）

| 失败模式 | 案例 | length 走向 | entropy 走向 | 共同根因 |
|---|---|---|---|---|
| Length **collapse** | PS v1 (372→18), GRPO biology v2 (330→17) | 缩 → 极短 | -90~99% | advantage 量级失控 + 短答 reward 密度高 |
| Length **explosion** | PS v2a (1442), TGDI-gtarg (1686) | 暴涨 | -62~97% | length_floor 单向防护，无 ceiling；wrong token reweight 鼓励生成更多 |
| **Both 异常** + val 仍可高 | GRPO biology v2 val=0.66 length=17 entropy=-90% | 极短 | -90% | 4 选 1 蒙对（**Why-SD-Degrades 现象**）|

**Why-SD-Degrades paper 的统一解释**：以上三种都是 epistemic verbalization 被压制。
- 缩短 = "Wait/Hmm" 等纠错通道被压掉
- 暴涨 = procedural step ("Therefore/Step 2:") 被过度强化
- 4 选 1 蒙对 = in-domain 短答信号噪声大，与 OOD 真实能力脱钩

### 6.2 t* 选择策略为何 aexEOS 胜出

**TIP paper 框架**：token 四象限按 (student_entropy, teacher_divergence) 分：
- Q1（高熵+高分歧）：~25-30%，矫正错误 + 巩固脆弱知识
- Q2（高熵+低分歧）：~15-22%，稳定不确定预测
- **Q3（低熵+高分歧 = "过度自信但错"）**：~3-15%，**密度最高矫正信号**
- Q4（低熵+低分歧）：~40-47%，可忽略

**aexEOS 隐式接近 Q3**：排尾 token 大概率是 stylistic（Q4），剩下的 argmax divergence 落在 reasoning 部分。Q3 区域往往集中在 reasoning content 上。

**gtarg 失败原因**：g_t 衡量 teacher 自反思惊讶，与 student entropy 完全无关 → 选到的位置可能是 Q1/Q2（teacher 觉得有信息但 student 没错），upweight 后 → 学错方向。

### 6.3 单 seed 方差 7.5% — 论文级风险

- GRPO biology v2 seed: 0.660 (length=17, "蒙对" 状态)
- GRPO biology v3 seed: 0.585 (length=343, 健康状态)
- **同 config 不同 seed 差 0.075**

意义：
- 单跑 TGDI 数字（如 0.5737）**完全可能是 noise**
- 论文必须报 mean±std over ≥3 seeds
- 报告"超过 baseline X%"需注意 baseline 也是单 seed

### 6.4 TGDI base-agnostic 设计的论文必要性

PS v2b 走到 s153 val 才 0.5525，仍未超 GRPO biology best 0.66。
**单做 prior_shift 论文站不住**——审稿人会问"为什么不用更强的 GRPO/RLSD base？"

base-agnostic 设计回应这个挑战：
- TGDI = ΔR causal layer，可叠加在任何 base 上
- 论文 main result = ΔR 在最强 RLSD baseline 上仍能涨 val

---

## 6.5 TCCA-Lite Smoke 攻略（**当前 P0 阻塞项**）

commit `4f85bde` 完成了 TCCA-Lite 完整实现：
- `intervention_rollout.py:_do_real_intervention` — 单步 counterfactual pipeline
- `tcca_chain.py` — helper 函数库（OPSD ctx, divergence, t*, teacher write, async continuation, composite build, rescore）
- `intervention_credit.py` — additive formula

### Local smoke 目标

验证 `enable_intervention=True` 时：
1. OPSD context 构造正常（需要 raw_prompt + reward_model ground_truth）
2. divergence + t* 选择正常
3. teacher 写 1 token + student async 续写正常
4. composite reward 非零 → ΔR 真正工作
5. SwanLab `intervention/*` 指标有值

### 关键风险点

| 风险 | 缓解 |
|---|---|
| `async_rollout_manager.server_manager.generate` API 没用过 | smoke 先验证 |
| raw_prompt 字段可能不在 batch 中 | try/except 退化写 0 |
| FSDP chunk divisibility (augmented batch) | 已在代码中 pad 处理 |

### 抓 Traceback 指令

```bash
cd /home/loujieming.ljm/TASD && git pull origin teacher-guided-intervention
source sdpo_env/bin/activate && set -a && source .env && set +a

# TCCA-Lite smoke (enable_int=True)
IC_ENABLE_INTERVENTION=True ./run_notebook_intervention_credit.sh smoke 2>&1 | tee /tmp/tcca_lite_smoke.log
echo "EXIT_CODE=$?"

# 找关键指标
grep -E "intervention/(delta_reward|failed_sample|applied)" /tmp/tcca_lite_smoke.log | tail -20

# 或抓 traceback
grep -B 2 -A 30 -E "Traceback|Error|Exception|raise" /tmp/tcca_lite_smoke.log | head -100
```

---

## 7. 待跑实验（TCCA-Lite 重排 priority）

### 主线（TCCA-Lite 论文 main result，必跑）

| Priority | 实验 | 配置 | 资源 | 预期 |
|---|---|---|---|---|
| 🔴 P0 | **TCCA-Lite smoke 诊断 + debug** | enable_int=True, λ_div=1.0, stderr 抓取 | local 4-GPU, 多轮迭代 | **当前最大阻塞项** |
| 🟢 P0 | **RLSD baseline 完整 250-step** | enable_int=False, base=rlsd | Nebula slot 1 ~7h | 填论文最大对照空缺 |
| 🟡 P0 | **TCCA-Lite main** (论文 main) | enable_int=True, λ_div=1.0, base=grpo, t*=aexEOS | Nebula slot 2 (smoke 通过后) | 论文 headline |
| 🟡 P1 | **多 seed 重复**（论文风险 #1） | main × 3 seeds | future | mean±std |
| 🟢 P1 | **多 subject 扩展**（论文风险 #4） | chemistry / physics 各跑 GRPO+TCCA-Lite | future | 多 dataset main figure |

### Ablation（论文配菜）

| Priority | 实验 | 用途 |
|---|---|---|
| 🟡 P1 | λ_div ∈ {0, 0.5, 1.0, 2.0} | hyperparameter 章节 (λ=0 = baseline) |
| 🟡 P1 | OPSD teacher vs EMA teacher (no ref answer) | 验证 privileged context 的价值 |
| ⚪ P2 | max_intervention_per_prompt ∈ {1, 2, 4} | 容量控制消融 |
| ⚪ P2 | Prior-Shift v2b 完整 250-step (RUNNING) | "correlational 归因不如 causal" 对照 |

---

## 8. 工程 TODO

### 已完成 ✅
- [x] Prior-Shift Tier 1 实现 (v1/v2)
- [x] Intervention-Credit base-agnostic 重构 (commit `2b9b2f0`)
- [x] dp_actor.teacher_generate_at_positions (Phase 2 V1)
- [x] **TCCA-Lite 完整实现** (commit `4f85bde`): `_do_real_intervention` 单步 counterfactual
- [x] **tcca_chain.py** helper 函数库（OPSD ctx, divergence, async continuation, composite build, rescore）
- [x] **Additive advantage formula** in `intervention_credit.py`
- [x] 13 个 intervention/* SwanLab 指标
- [x] notebook smoke / tstar / full 三模式脚本
- [x] Nebula sweep 脚本（含 dry-run）
- [x] .env 凭证统一管理（gitignore）
- [x] SwanLab 中期诊断脚本

### 待做

| Priority | 项 | 工作量 |
|---|---|---|
| 🔴 P0 | TCCA-Lite smoke 失败诊断 + 修复 | TBD |
| 🟡 P1 | 早停机制（连续 N 个 val 评估点低于 best × 0.9 即停） | ~50 行 |
| 🟢 P2 | swanlab-analyzer 自动 pull 脚本 | ~80 行 |
| ⚪ P3 | SDPO + ΔR (loss_mode=sdpo + base=grpo 组合) | TBD |

---

## 9. Related Work 5 篇核心论文

| Paper | arXiv | 与我们工作的关系 |
|---|---|---|
| **TIP** (Token Importance) | 2604.14084 | Q3 区域假说证明了 aexEOS 的合理性；指引 Q3 filtering 改进方向 |
| **Rethinking OPD** (Tsinghua) | 2604.13016 | OPD 起作用在 overlap tokens；TGDI 反向（关心 non-overlap divergence）是与 OPD 范式的根本差异 |
| **Why-SD-Degrades** | 2603.24472 | Self-distillation 系统性压制 epistemic verbalization → 统一解释我们三种 collapse 模式 + OOD 风险警示 |
| **SRPO** (Sample Routing) | 2604.02288 | failed→SDPO / correct→GRPO 路由策略；与 TGDI Mode B append 是同思想不同实现；SRPO 5-bench avg 55.5 vs GRPO 52.1 |
| **RLSD** (Self-Distilled RLVR) | 2604.03128 | Theorem 1（OPSD irreducible MI gap）+ "方向 ⊥ 大小"原理 → TGDI 用 ΔR 作 magnitude 是更直接的因果信号 |

详细分析见 [OPD_Deep_Analysis.html](file:///Users/awesome_jimmy/lazada/papers/raw/opd_papers/OPD_Deep_Analysis.html)

---

## 附录 A: Commit 链

### 当前分支：`teacher-guided-intervention`

```
4f85bde  feat(tcca_lite): pivot from chain → TCCA-Lite (single-step counterfactual + real student continuation)
f8de1e2  WIP(tcca_v2): tcca_chain.py + intervention_credit.py 重写 + ray_trainer dispatch
ccf8f17  docs(tcca_v2): Layer 3 决策固化 (C+D 双跳, λ_div=1.0 默认)
cf18d20  docs(tcca_v2): 修正 cost analysis
c62f997  docs(tcca_v2): 详细设计文档 + pseudocode
```

### 上游分支：`bayesian-credit-assignment` (PS v1/v2 历史)

```
402a8bc  fix(prior_shift): add v2 fields to hydra config
25fc2f6  fix(prior_shift): v2 - max_ratio 10→3, length floor penalty
2ffe961  feat(prior_shift): add nebulactl submit sweep script
67ff19f  feat(prior_shift): Tier 1 Bayesian Credit Assignment 主菜首发
```

---

## 附录 B: 关键文件路径

### 核心代码
- [verl/trainer/ppo/bayesian_credit/intervention_credit.py](verl/trainer/ppo/bayesian_credit/intervention_credit.py) — additive advantage estimator
- [verl/trainer/ppo/bayesian_credit/intervention_rollout.py](verl/trainer/ppo/bayesian_credit/intervention_rollout.py) — TCCA-Lite single-step counterfactual
- [verl/trainer/ppo/bayesian_credit/tcca_chain.py](verl/trainer/ppo/bayesian_credit/tcca_chain.py) — helper functions (OPSD ctx, divergence, async continuation, composite)
- [verl/trainer/ppo/bayesian_credit/prior_shift.py](verl/trainer/ppo/bayesian_credit/prior_shift.py) — Tier 1 legacy
- [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py) line 887-958 — teacher_generate_at_positions
- [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) line 1150-1175 — worker dispatch
- [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) — dispatch to intervention_rollout

### 设计文档
- [tcca_v2_design.md](tcca_v2_design.md) — TCCA V2 design + TCCA-Lite pivot
- [paper_idea.md](paper_idea.md) — 论文思路
- [design_history.md](design_history.md) — 设计演化历史

### 配置
- [verl/trainer/config/intervention_credit.yaml](verl/trainer/config/intervention_credit.yaml) — TCCA-Lite 主配置
- [verl/trainer/config/prior_shift.yaml](verl/trainer/config/prior_shift.yaml) — Tier 1

### 提交脚本
- [nebula_scripts/submit_intervention_credit_sweep.sh](nebula_scripts/submit_intervention_credit_sweep.sh) — Nebula sweep
- [nebula_scripts/intervention_credit/intervention_credit_sciknoweval_parametric.sh](nebula_scripts/intervention_credit/intervention_credit_sciknoweval_parametric.sh) — parametric
- [run_notebook_intervention_credit.sh](run_notebook_intervention_credit.sh) — local 4-GPU smoke/tstar/full

---

## 附录 C: 历史版本归档

> 旧 chain rollout 设计（8 iter 串行）已 pivot 到 TCCA-Lite（单步 counterfactual）。详见 [tcca_v2_design.md](tcca_v2_design.md) 和 commit `4f85bde`。
