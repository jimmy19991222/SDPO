# TCCA — Token-level Causal Credit Assignment

> **Building on OPD literature** (see [OPD_Deep_Analysis](file:///Users/awesome_jimmy/lazada/papers/raw/opd_papers/OPD_Deep_Analysis.html)): MiniLLM (2306) → GKD (2306) → SDPO/OPSD (2604) → RLSD (2604) → **TCCA-Lite (ours)**
> **目的**：让你 check 我们对论文思路的理解是否一致
> **最新更新**：2026-05-16 21:50 — **pivot to TCCA-Lite** (chain rollout → single-step counterfactual on failed samples)

---

## ⚡ TCCA-Lite Pivot (2026-05-16 21:00)

**从** chain rollout（8 步串行迭代，~2× compute，工程复杂度高）
**到** TCCA-Lite（标准 GRPO n=8 + 失败样本单步 counterfactual，~25% overhead）

| 维度 | 旧 TCCA (chain) | **TCCA-Lite (current)** |
|---|---|---|
| rollout | 链式 8 iter 串行 | 标准 GRPO n=8 |
| intervention 时机 | 每 iteration 都做 | **仅 failed samples** |
| 位置数 | 每 iter 1 个 (共 8 个) | **每 failed 1 个** |
| teacher 写多少 | k=1 token | **k=1 token** |
| student tail | 真续写到 EOS | **真续写到 EOS** |
| teacher context | OPSD (含 ref answer) | **OPSD (含 ref answer)** |
| 公式 | multiplicative (1+λ·c_t) | **additive (A_seq + λ·c_t)** |
| compute overhead | ~2.2× | **~1.25×** |

以下文档保留旧 TCCA 设计思路作参考，标注 🔴 处为 TCCA-Lite 中已改变的 key design。

---

## 1. 论文 one-sentence 定位

> **TCCA：把"token 的 credit 应该多少"这个问题，用 teacher 真实改写 + reward 复算的因果反事实方式来回答。**

不是 "teacher 在哪里感到惊讶"（Prior-Shift，相关性），
不是 "teacher 和 student 哪里不合"（RLSD，相关性 + log-ratio 启发式），
不是 "哪类 token 启发式上重要"（TIP，entropy×divergence 经验加权），
而是 **"如果在 t 处听 teacher 一会儿，最终 reward 真的涨了吗？"**——直接做实验，看 ΔR_t。

---

## 2. OPD 系列 → TCCA 的一步推进

| 阶段 | 代表方法 | 解决了什么 | 留下什么问题 |
|---|---|---|---|
| 1 | MiniLLM / GKD | 解决 off-policy 蒸馏的 exposure bias，引入 on-policy student rollout | token-level credit 还是均匀 KL，无差异化 |
| 2 | TIP / SCOPE / SelecTKD | 用启发式 (student entropy, teacher divergence) 选 token 加权 | **相关性启发式**，没有因果证据 |
| 3 | OPSD / SDPO | self-distillation 范式，用 privileged-context teacher 做信号源 | RLSD Theorem 1: 不可消除的 MI leakage → progressive degradation |
| 4 | RLSD | "方向 ⊥ 大小" 解耦：方向锚到 env reward，大小用 evidence ratio (P_T/P_S) | evidence ratio 仍是**统计学**信号，不是真实因果 |
| 5 | SRPO | failed→SDPO / correct→GRPO 的 sample routing | 用 distribution match 做失败样本，没改 token-level credit |
| **6 (ours)** | **TCCA** | **用真实 counterfactual ΔR_t 做 token credit；保留 RLSD 的方向⊥大小原则，但用因果代替统计学** | (待论文 finalize) |

**TCCA 的 one more step**：把 RLSD 的"用 P_T/P_S 估计 token 重要性"升级为"做 K 次真实 intervention，用真 ΔR_t 测量 token 重要性"。

---

## 3. 核心创新点（论文 contribution）

### 3.1 Token-level causal credit signal（新概念）

**定义**：对 token y_t 的 causal credit
```
c_t := ΔR_t = R(y') - R(y),  其中  y' = y_<t + teacher_replacement + y_>t
```

**性质**（论文 Theorem 1 待证）：
- **因果性**：c_t 是 token y_t 的 individual causal effect on outcome reward（满足 Rubin causal model 的 SUTVA）
- **稀疏可计算**：只需在 top-K positions 计算 K 次 intervention，不是 O(T) full sweep
- **base-agnostic**：c_t 是数据，可叠加在任何 base RL 算法上

### 3.2 TCCA-Lite advantage 公式（additive）

```
A_t (token-level) = (A_seq + λ_div · clip(c_t, ±1.0)) · response_mask · length_scale
                    ┌─────────────┘  └──────────┘
                    GRPO base        divergence-point credit (additive)
```

**TCCA-Lite 改用 additive（非 multiplicative）**：
- 旧 multiplicative: `A_t = A_seq · (1 + λ·c_t)` → c_t < 0 时仍保留正号方向，丢失负 credit
- **新 additive: `A_t = (A_seq + λ·c_t)`** → c_t < 0 真正 push down 对应 token 的梯度
- λ_div=0 → 退化为纯 GRPO（论文 ablation baseline = 选项 D）
- λ_div>0 → TCCA divergence-point modulation（论文 main = 选项 C）

### 3.3 单步 intervention 设计（TCCA-Lite 工程贡献）

**问题**：chain rollout（8 iter 串行）工程复杂度高 + ~2× compute
**TCCA-Lite 解法**：
- 保留标准 GRPO rollout n=8（不做 chain）
- 只对 **failed samples** (R < threshold) 做**单步 counterfactual**
- OPSD teacher（含 reference answer）在 argmax divergence 位置写 **1 个 token**
- student 从修正后的 prefix **真续写到 EOS**（V1 复用旧 tail 导致 ΔR≡0 bug，已修复）
- 实际成本：rollout 30s→40s (+33%), update 10s→15s (+50%), total 66s→83s (**~25% overhead**)

### 3.4 Per-token causal credit construction（新机制）

每次 intervention 产生 1 个 ΔR_k，但**写入两个地方**：

```
失败 sample y: c_t[t_k..t_k+intervention_length) = -ΔR_k   ← 这些 token 是"错"，给负权重
composite y'_k: c_t[t_k..t_k+intervention_length) = +ΔR_k   ← 这些 token 是 teacher 的"对"，给正权重
```

**这一对正负 credit 形成 contrastive pair**（同 prefix，分歧 token 反向 credit）→ 类似 DPO 的对比信号，但用真实 outcome 差驱动。

---

## 4. 完整 pipeline（TCCA-Lite）

### Step 0 — Standard GRPO rollout

```
n=8 standard rollouts per prompt (复用现有 generate_sequences)
y_i ~ π_θ(·|x), R_i = reward_fn(x, y_i)
```

### Step 1 — 失败样本检测

```
failed = {i : R_i < threshold}
```

### Step 2 — OPSD teacher forward + divergence

```
OPSD teacher context: prompt + "The correct answer is {r}.\n"
teacher_fwd_opsd → logp_T_opsd on failed samples
divergence[t] = |logp_T_opsd[t] − logp_S[t]| · response_mask
```

### Step 3 — 选 t*（单位置 argmax divergence，排尾 8 token）

```
t* = argmax divergence[t], exclude tail 8 tokens (防选 EOS)
```

### Step 4 — teacher 写 1 token at t*

```
z_t* = teacher.argmax(· | OPSD ctx, prefix y[:t*])  ← 1 token only
```

### Step 5 — student 真续写到 EOS

```
prefix = original_prompt + y[:t*] + [z_t*]
continuation = student.generate(prefix, max_new_tokens=T - t* - 1)
y' = y[:t*] ⊕ [z_t*] ⊕ continuation  ← 真续写，非复用旧 tail (修复 ΔR≡0 bug)
```

### Step 6 — re-score composite + ΔR

```
R(y') = reward_fn(x, y')
ΔR = R(y') - R(y)  ← 真实因果差
```

### Step 7 — divergence_credit 构造 + Mode B append

```
c_t[y,   t*] = -ΔR   (原 sample 的负 credit)
c_t[y',  t*] = +ΔR   (composite 的正 credit)
response_mask[y', :t*] = 0  (shared prefix 不算 loss)
augmented_batch = original ⊕ composites (同 uid)
```

### Step 8 — Advantage computation (additive)

```
A_seq = R_i - mean_group(R)   ← GRPO group-relative
A_t = (A_seq + λ_div · clip(c_t, ±1)) · response_mask · length_scale
λ_div=0 → 退化纯 GRPO（ablation baseline）
λ_div=1.0 → TCCA-Lite main
```

### Step 9 — PPO surrogate (不动)

```
L_PPO = 𝔼[min(ρ_t · A_t, clip(ρ_t, 1±ε) · A_t)]
```

---

## 5. 为什么这么做（与 OPD 文献的论证对接）

| 设计选择 | 论证依据 |
|---|---|
| 用 ΔR 作 magnitude，env reward 作 direction | **RLSD Theorem 1**：方向⊥大小，方向必须可靠+稀疏，大小可稠密+容噪 |
| 只对 failed samples intervention | **SRPO**：sample routing 实证 (correct→GRPO/failed→特殊处理) 5-bench +3.4% |
| t* = argmax |logp_T_OPSD − logp_S| 排尾 8 token | **TIP Q3 (低熵+高分歧)** 的近似；实证 v3p1-aexEOS 0.5737 > araw 0.42 |
| 单位置 + 1 token（非 K=3, ℓ=2） | 工程-理论 trade-off：单步足以产生真实 ΔR，chain 边际增益待验证 |
| 学生**真续写**到 EOS（非复用旧 tail） | V1 复用 tail → ΔR≡0 bug（reward 基于尾部 answer，未变）；真续写才产生非零 ΔR |
| OPSD teacher context（含 ref answer） | **OPSD**：teacher 看到 privileged info → 更好的 corrective signal |
| 每个 ΔR 同时给原 sample (-) 和 composite (+) | **DPO-style contrastive**：同 prefix、反向 credit → 形成对比对 |
| Additive advantage（非 multiplicative） | (A_seq + λ·c_t) 保符号语义；(A_seq·(1+λ·c_t)) 会丢失负 credit |
| ~25% overhead（非 2× chain） | 标准 GRPO n=8 + 少量 failed composite → 完全可接受 |

---

## 6. 与既有方法的 head-to-head 对比

| 方法 | direction signal | token weight 来源 | leakage | 用 outcome reward? |
|---|---|---|---|---|
| SFT (off-policy KD) | teacher tokens | uniform | N/A | ❌ |
| GRPO (RLVR baseline) | env reward (sparse) | uniform | 无 | ✅ |
| SDPO / OPSD | distribution match | distribution match | **严重** | ❌ |
| RLSD | env reward (sparse) | exp(sign(A)·(logp_T - logp_S)) (统计学) | 无 | ✅ |
| TIP | distribution match | (1-H_S) · KL_TS (启发式) | 部分 | ❌ |
| SRPO | env reward + KL | uniform on correct / KL on failed | 弱 | ✅ |
| **TCCA-Lite (ours)** | **env reward (sparse)** | **真实 ΔR (单步 counterfactual)** | **无** | **✅** |

**TCCA-Lite 的 unique selling point**：唯一同时满足
- ✅ on-policy
- ✅ env reward 锚定方向（无 leakage）
- ✅ **真实因果 token weight**（不是统计学）
- ✅ 仅 ~25% overhead（实用级）

---

## 7. 实验设计（TCCA-Lite）

### Main figure：TCCA-Lite vs baseline

| Method | biology | chemistry | physics | (multi-seed mean±std) |
|---|---|---|---|---|
| GRPO baseline | 0.66 (有 baseline) | 0.78 | 0.78 | |
| **GRPO + TCCA-Lite (λ_div=1.0)** | 0.6X (期 +1~3) | 0.X | 0.X | 📌 **论文 headline** |
| GRPO + TCCA-Lite (EMA teacher) | 0.X | — | — | ablation |
| RLSD baseline | 0.58 (待补) | ? | ? | |
| SDPO baseline | 0.59 | 0.74 | 0.65 | |

### Ablation figures

1. **λ_div 敏感性**：λ ∈ {0 (baseline), 0.5, 1.0 (main), 2.0}
2. **teacher context 消融**：OPSD (含 ref answer) vs EMA teacher (no ref)
3. **max_intervention_per_prompt**：{1, 2, 4} — 每个 prompt 最多 intervention 几个 failed sample
4. **单步 vs chain**：TCCA-Lite (single-step) vs chain rollout (future work) — 工程 trade-off 实证

---

## 8. 你需要确认的几个点

请 check 这几个 understanding 是否与你一致：

1. ✅ TCCA-Lite 创新在 **token-level credit assignment 这一层**，不在 reward / loss 层
2. ✅ ΔR 是 **单步 counterfactual 的真实因果效应**（不是 chain rollout）
3. ✅ 单位置 + 1 token 是工程-理论 trade-off，单步足以产生有效 ΔR 信号
4. ✅ **Additive formula** 保符号语义（c_t < 0 真正 push down 梯度）
5. ✅ **student 真续写到 EOS** 修复了 V1 复用旧 tail 导致的 ΔR≡0 bug
6. ✅ Mode B append 提供 **contrastive pair**（原 sample + composite 同 prefix、反向 credit）
7. ✅ **不动 PPO loss function**，只重新计算 advantage A_t
8. ✅ **~25% overhead** 是实用级的

如以上有任何不对齐，请告诉我具体哪点，我修正后再继续。

---

## 9. 与论文 OPD survey 的差异化定位

OPD survey 84 篇论文里，**没有一篇做真正的 token-level causal credit**：
- 大多数 OPD 用 KL distribution matching（相关性）
- 少数 (TIP / SCOPE) 用启发式 token 重要性（相关性 + 工程）
- RLSD / RLSD-family 用 evidence ratio（统计学近似，仍非因果）
- TCCA 用**真实 counterfactual intervention**——这是 OPD literature 的空白

**TCCA = OPD on more step**：把 OPD 的"token 级信号"从 distribution matching → 启发式权重 → 统计学比值 → **真实因果 ΔR_t** 的演化推进一步。

---

## 附录：术语索引

- **ΔR / ΔR_t**: causal counterfactual reward delta = R(y') - R(y)
- **t* / t_i**: teacher-student OPSD divergence 最大的单位置
- **c_t** (TCCA): per-token divergence_credit vector (B, T)，TCCA-Lite 的核心新字段
- **OPSD context**: teacher prompt + "The correct answer is {r}.\n" + original prompt
- **TCCA-Lite**: 标准 GRPO n=8 + 失败样本单步 counterfactual (pivot from chain rollout)
- **Additive advantage**: A_t = (A_seq + λ_div · c_t) · mask · length_scale
- **Mode B append**: composite samples 加到 batch (不替换原样本)，形成变长 group
