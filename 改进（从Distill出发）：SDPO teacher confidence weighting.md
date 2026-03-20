github: [https://github.com/jimmy19991222/SDPO](https://github.com/jimmy19991222/SDPO)

## 一、背景与动机
### 原始 SDPO 的问题
SDPO 通过让模型扮演"自教师"角色，将带反馈的教师分布蒸馏回学生，实现密集信用分配。但原始实现存在一个潜在问题：

```plain
原始 SDPO 的 loss 聚合方式：
  per_token_loss = KL(student || teacher)   # 每个位置的KL散度
  loss = token_mean(per_token_loss)         # 所有位置等权平均

问题：教师在某些位置非常不确定（高熵）
  → 该位置的蒸馏信号是噪声
  → 但仍然与其他位置等权参与训练
  → 引入无效甚至有害的梯度
```

### 核心洞察
```plain
教师熵低（确定）→ 反馈有效指导了教师 → 该位置信号可靠 → 应该高权重学习
教师熵高（不确定）→ 加了反馈教师仍困惑 → 该位置信号是噪声 → 应该低权重忽略
```

### 与已有工作的区别
| 方法 | 熵的来源 | 框架 | 效果 |
| --- | --- | --- | --- |
| Wang et al. 2025（论文baseline） | 学生的熵 | GRPO | 论文中比GRPO还差 |
| **本改进（教师熵加权）** | **教师的熵** | **SDPO** | **待验证** |


_<font style="color:rgb(139, 139, 139);background-color:rgb(246, 247, 248);">Wang et al. 2025 — "Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning"</font>_

两者方向完全不同，不存在冲突。

---

## 二、方法说明
### 加权公式
在原始 `token_mean` 聚合前，对每个 token 位置乘以基于教师熵的置信度权重：

$ w_t = \text{softmax}\left(\frac{-H(\pi_\theta(\cdot|x,f,y_{<t}))}{\tau}\right) \cdot L $

$ \mathcal{L}_{SDPO}^{ew} = \sum_t w_t \cdot \text{KL}(\pi_\theta(\cdot|x,y_{<t}) \| \pi_\theta(\cdot|x,f,y_{<t})) $

其中：

+ $ H(\cdot) $ 为教师在该位置的熵（用 top-K 近似）
+ $ \tau $ 为温度超参，控制权重尖锐程度
+ $ L $ 为序列有效长度，用于还原 token-mean 尺度

### 温度参数的直觉
```plain
τ → 0   权重极度集中在熵最低的少数位置（过激进）
τ = 0.5  比较尖锐，强调确定位置
τ = 1.0  标准 softmax（实验起点）
τ = 2.0  比较平滑，接近均匀权重
τ → ∞   完全均匀权重 = 退化为原始 SDPO
```

---

## 三、代码修改内容
### 修改文件总览
| 文件 | 修改类型 | 说明 |
| --- | --- | --- |
| `verl/workers/config/actor.py` | 新增字段 | `SelfDistillationConfig` 加两个配置项 |
| `verl/trainer/ppo/core_algos.py` | 新增函数 + 调用 | 插入 `apply_teacher_entropy_weighting` 并在 loss 计算末尾调用 |
| `verl/trainer/config/sdpo.yaml` | 新增配置项 | 默认关闭，可按需开启 |
| 训练脚本 | 新增参数 | `SCRIPT_ARGS` 中加入两个新参数 |


---

### 修改1：`verl/workers/config/actor.py`
在 `SelfDistillationConfig` 的 `distillation_topk` 字段后新增两个字段：

```python
# 修改前
distillation_topk: Optional[int] = None

# 修改后
distillation_topk: Optional[int] = None
entropy_weighting: bool = False      # 是否开启教师熵加权，默认关闭
entropy_temperature: float = 1.0     # softmax 温度，越小权重越集中
```

---

### 修改2：`verl/trainer/ppo/core_algos.py`
**新增辅助函数**（插入在 `compute_self_distillation_loss` 定义之前）：

```python
def apply_teacher_entropy_weighting(
    per_token_loss: torch.Tensor,
    teacher_topk_log_probs,
    teacher_all_log_probs,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
    use_topk: bool = True,
):
    """用教师熵对每个 token 位置的 loss 加权。"""
    with torch.no_grad():
        # 选择 logprobs 来源
        if use_topk and teacher_topk_log_probs is not None:
            teacher_logp = teacher_topk_log_probs
        elif teacher_all_log_probs is not None:
            teacher_logp = teacher_all_log_probs
        else:
            return per_token_loss, {}

        # 计算教师熵：H = -sum(p * log_p)
        teacher_probs = teacher_logp.exp()
        safe_logp = torch.clamp(teacher_logp, min=-100.0)
        teacher_entropy = -(teacher_probs * safe_logp).sum(dim=-1)  # (B, T)

        # 数值检查：若熵本身有nan/inf，跳过加权直接返回
        if torch.isnan(teacher_entropy).any() or torch.isinf(teacher_entropy).any():
            print("[EW] teacher_entropy has nan/inf, skipping weighting")
            valid_entropy = teacher_entropy[loss_mask == 1]
            ew_metrics = {
                "sdpo/teacher_entropy_mean": valid_entropy.mean().item() if valid_entropy.numel() > 0 else 0.0,
                "sdpo/teacher_entropy_std": valid_entropy.std().item() if valid_entropy.numel() > 1 else 0.0,
            }
            return per_token_loss, ew_metrics

        # padding 位置熵设为 inf，权重趋近 0
        masked_entropy = teacher_entropy.masked_fill(loss_mask == 0, 1e9)

        # 负熵 softmax：熵越低 → 权重越高
        confidence_weights = torch.softmax(-masked_entropy / temperature, dim=-1)

        # 乘有效长度，还原到 token-mean 尺度
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        confidence_weights = confidence_weights * seq_lengths

        # clip极端权重，防止梯度爆炸
        confidence_weights = torch.clamp(confidence_weights, max=10.0)

        # 最终nan检查：若权重有nan，跳过加权
        if torch.isnan(confidence_weights).any():
            print("[EW] confidence_weights has nan, skipping weighting")
            valid_entropy = teacher_entropy[loss_mask == 1]
            ew_metrics = {
                "sdpo/teacher_entropy_mean": valid_entropy.mean().item() if valid_entropy.numel() > 0 else 0.0,
                "sdpo/teacher_entropy_std": valid_entropy.std().item() if valid_entropy.numel() > 1 else 0.0,
            }
            return per_token_loss, ew_metrics

        # 监控指标（含空张量保护）
        valid_entropy = teacher_entropy[loss_mask == 1]
        if valid_entropy.numel() == 0:
            ew_metrics = {
                "sdpo/teacher_entropy_mean": 0.0,
                "sdpo/teacher_entropy_std": 0.0,
            }
        else:
            ew_metrics = {
                "sdpo/teacher_entropy_mean": valid_entropy.mean().item(),
                "sdpo/teacher_entropy_std": valid_entropy.std().item() if valid_entropy.numel() > 1 else 0.0,
            }

    return per_token_loss * confidence_weights, ew_metrics

```

**在 **`compute_self_distillation_loss`** 末尾插入调用**：

```python
# Apply rollout correction weights if provided
if rollout_is_weights is not None:
    per_token_loss = per_token_loss * rollout_is_weights

# ← 新增：教师熵加权（entropy_weighting=False 时完全跳过，行为与原版一致）
use_entropy_weighting = getattr(self_distillation_config, 'entropy_weighting', False)
if use_entropy_weighting:
    entropy_temperature = getattr(self_distillation_config, 'entropy_temperature', 1.0)
    use_topk = getattr(self_distillation_config, 'distillation_topk', None) is not None
    per_token_loss, ew_metrics = apply_teacher_entropy_weighting(
        per_token_loss=per_token_loss,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        loss_mask=loss_mask,
        temperature=entropy_temperature,
        use_topk=use_topk,
    )
    metrics.update(ew_metrics)

loss = agg_loss(
    loss_mat=per_token_loss,
    loss_mask=loss_mask,
    loss_agg_mode=loss_agg_mode,
    batch_num_tokens=loss_mask.sum().clamp(min=1.0),
)
return loss, metrics
```

---

### 修改3：`verl/trainer/config/sdpo.yaml`
```yaml
# 修改前
self_distillation:
  max_reprompt_len: 10240
  is_clip: 2.0

# 修改后
self_distillation:
  max_reprompt_len: 10240
  is_clip: 2.0
  entropy_weighting: true      # 开启教师熵加权
  entropy_temperature: 1.0     # 温度超参
```

---

### 修改4：训练脚本
两个训练脚本（科学推理 + LCBv6 编程）均做相同修改：

```bash
# 配置区新增
ENTROPY_WEIGHTINGS=(True)
ENTROPY_TEMPERATURES=(1.0)

# SCRIPT_ARGS 新增两行
"actor_rollout_ref.actor.self_distillation.entropy_weighting=$ENTROPY_WEIGHTING"
"actor_rollout_ref.actor.self_distillation.entropy_temperature=$ENTROPY_TEMPERATURE"

# EXP_NAME 加入标识
-ew${ENTROPY_WEIGHTING}-et${ENTROPY_TEMPERATURE}-
```

---

## 四、兼容性说明
```plain
entropy_weighting=False（默认）：
  代码路径与原版 SDPO 完全一致，零性能开销

entropy_weighting=True：
  仅在 agg_loss 之前多一次 with torch.no_grad() 的权重计算
  额外计算量极小（只是 softmax，无反向传播）
```

三层兼容保证：

1. `SelfDistillationConfig` 字段默认 `False`
2. `getattr(..., 'entropy_weighting', False)` 兜底
3. `sdpo.yaml` 不写该字段时走 dataclass 默认值

---

## 五、训练监控指标
开启后日志新增两个指标：

| 指标 | 含义 | 参考值 |
| --- | --- | --- |
| `sdpo/teacher_entropy_mean` | 教师平均熵 | 越低说明教师越确定，信号质量越高 |
| `sdpo/teacher_entropy_std` | 熵的标准差 | 越高说明不同位置分化明显，加权效果越显著 |


---

## 六、实验记录
### 实验配置说明
+ **基础模型**：Qwen3-8B / Olmo3-7B-Instruct
+ **基线**：原始 SDPO（`entropy_weighting=False`）
+ **实验组**：教师熵加权 SDPO（`entropy_weighting=True`）
+ **评估指标**：avg@16 准确率（科学推理）/ pass@4 准确率（LCBv6）

---

### 实验结果表格
实验setting（200epochs）

作者每个实验一般跑150steps，[https://github.com/lasgroup/SDPO/issues/17](https://github.com/lasgroup/SDPO/issues/17)

<!-- 这是一张图片，ocr 内容为：JONHUE ON FEB 7 MEMBER I THINK IT WAS AROUND 150 STEPS. -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132756355/1773641931665-76dfc3ad-9277-449c-b444-edb11edd9ffc.png)

bio对比实验: 

+ val avg@16: [https://swanlab.cn/@awesome_jimmy/SDPO/runs#ejltOXJq-OWJwOWZYLUg=](https://swanlab.cn/@awesome_jimmy/SDPO/runs#ejltOXJq-OWJwOWZYLUg=)

<!-- 这是一张图片，ocr 内容为：样式 图表 数据 VAL-CORE/SCLKNOWEVAL/ACC/MEAN@16 FINAL-SDPO-SCIKNOWEVAL-BLOLOGY-TRAIN32-ALPHAO.5-ROLLOUT8-IRLE-5-DROSSTRJE-EVI TRUB-ETL.0-QWEN-QWEN3-8B-2026-03-16.08-07-59 VAL-CORE/SCIKNOWEVA//ACC/M OANG16 FINAL-SDPO-SCIKNOWEVEVAL-BIOLOGY-TRAIN32-ALOHA0.5-RC 1.5PX FINAL-SDPO-SCIKNOWOVAL-BIOLOGY-TRAIN32-ALPHA0.5-ROLLOUTB-IRLE-5-DRASSTRUE-QW ON-QWON3-BB-2026-03-13_19-45-57 VAL-CORO/SCKNOWOVAL/ACC/MOANE16 一1.5PX FINAL-SDPO-SCIKNOWEVAL-BIOLOCTY-TRAIN32-ALNHAO.5-RC FINAL-GRPO-SCIKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-ROLLOUT8-IRLE-5-MODELQWEN-Q WEN3-8B-2026-03-13.10-39-25'VAL-CORE/SCLKNOWEVAL/ACC/MEAN@16 FINAL-GRPO-SCIKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-RO SLEP 取消 应用 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132756355/1773641166672-5d9a8589-2807-4523-91fd-8d5094dba69e.png)

+ val major@16: [https://swanlab.cn/@awesome_jimmy/SDPO/runs#c2Jjb3ph-bGdtUGRlWEk=](https://swanlab.cn/@awesome_jimmy/SDPO/runs#c2Jjb3ph-bGdtUGRlWEk=)

<!-- 这是一张图片，ocr 内容为：图表 样式 数据 VAL-CORE/SCIKNOWEVAL/ACC/MAJ@16/MEAN -- AND-SCFO-SCKIONIAL-S-OUTR-TRAN3-SRRAIS-S-S-S-DO-S-DO-S-DO-S-OUTRO-ONES-ONES-S-S-3C-3C-3-M,S-S-O7-5 FINAL-SDPO-SCKNOWEVAL-DIOLOGY-TRAIN32-ALPHAO.5-ROLLOUT8-IRLE-5-DROSSTRUO-QY FINAL-SDPO-SCIKNOWOVAL-BIOLOGY-TRAIN32-ALPHA0.5-RC 1.5PX FIN-GWON3-88-2628-03-13-13-19-47-AL-AL-SCIKROW FINAL-SDPO-SGLKNOWEVAL-BIOLOGY-TRAIN32-AIPHAO.5-RC 1.5PX FINAL-GRPO-SCIKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-ROLLOUTB-IRTE-5-MODELGWEN-Q WEN3-8B-2026-03-13_10-39-25 VAL-CORE/SCIKNOWEVAL/ACC/MAJE16/MEAN FINAL-GRPO-SCLKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-RO 1.GPX 应用 取消 120 200 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132756355/1773641204591-c90d0bd1-2832-4de9-986f-3ea81f72844e.png)

<!-- 这是一张图片，ocr 内容为：本中介 VAL-CORE/SCIKNOWEVAL/ACC/MAJ@16/MEAN VAL-CORE/SCIKNOWEVAL/ACC/MEAN@16 FINAL-SDPO-SCIKNOWEVAL-BIOLOQY-TRAIN32-ALPHAO.5-ROLLOUT8-IR1E-5-DROS FINAL-SDPO-SCIKNOWEVAL-BIOLOGY-TRAIN32-ALPHA0.5-ROLLOUT8-IR1E-5-DROS FINAL-SDPO-SCIKNOWEVAL-BIOLOGY-TRAIN32-ALPHAO.5-ROLLOUT8-IR1E-5-DROS FINAL-SDPO-SCIKNOWEVAL-BIOLOGY-TRAIN32-ALPHA0.5-ROLLOUT8-IR1E-5-DROS FINAL-SDPO-SCIKNOWEVAL-BIOLOGY-TRAIN32-ALPHAO.5-ROLLOUT8-IR1E-5-DROS FINAL-SDPO-SCIKNOWEVAL-BIOLOGY-TRAIN32-ALPHA0.5-ROLLOUT8-IR1E-5-DROS FINAL-GRPO-SCIKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-ROLLOUT8-IR1E-5-MODE FINAL-GRPO-SCIKNOWEVAL-BIOLOGY-MBS-32-TRAIN32-ROLLOUT8-IR1E-5-MODE 0 0.3 0.5 0.3 0.7 0.1 0.5 0.6 0.6 0.4 0.1 0.2 0.7 0.2 0.4 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/132756355/1773752262017-d4375fbe-1a59-484f-94b1-26ed19bb5fb1.png)

#### 科学推理任务（Qwen3-8B）
| 方法 | entropy_temperature | biology<br/>(maj@16/avg@16) | Chemistry <br/> | Physics | Materials | Tool Use |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO<br/>(baseline) | - | 0.55/0.52<br/>论文（49.8%） |  |  |  |  |
| SDPO (baseline) | - | 0.56/0.56<br/>论文（56.8%） |  |  |  |  |
| SDPO + EW | 0.5 |  |  |  |  |  |
| SDPO + EW | 1.0 | 0.66/0.63 |  |  |  |  |
| SDPO + EW | 2.0 |  |  |  |  |  |


#### 科学推理任务（Olmo3-7B-Instruct）
| 方法 | entropy_temperature | biology<br/>(maj@16/avg@16) | Chemistry <br/> | Physics | Materials | Tool Use |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO<br/>(baseline) | - |  |  |  |  |  |
| SDPO (baseline) | - |  |  |  |  |  |
| SDPO + EW | 0.5 |  |  |  |  |  |
| SDPO + EW | 1.0 |  |  |  |  |  |
| SDPO + EW | 2.0 |  |  |  |  |  |


#### LCBv6 编程任务（Qwen3-8B）
| 方法 | entropy_temperature | 最终准确率 | 达到 GRPO 最终准确率所需 generations | teacher_entropy_mean | 备注 |
| --- | --- | :---: | :---: | :---: | --- |
| GRPO (论文) | - | 41.2% | - | - | 论文结果 |
| SDPO (baseline) | - | 48.8% | 4× 更少 | - | 论文结果 |
| SDPO + EW | 0.5 |  |  |  |  |
| SDPO + EW | 1.0 |  |  |  |  |
| SDPO + EW | 2.0 |  |  |  |  |


#### Response Length 对比
| 方法 | entropy_temperature | Qwen3-8B 平均长度 | Olmo3-7B 平均长度 | 相对 SDPO baseline 变化 |
| --- | --- | :---: | :---: | :---: |
| SDPO (baseline) | - | 255.8 | 343.9 | - |
| SDPO + EW | 0.5 |  |  |  |
| SDPO + EW | 1.0 |  |  |  |
| SDPO + EW | 2.0 |  |  |  |


---

## 思考
```plain
论证链条：
  teacher熵低 → teacher确定 → 信号可靠 → 高权重

但这个链条有漏洞：

teacher熵低的原因可能是：
  a. feedback让teacher真正理解了这个位置 ← 想要的
  b. 这个token本来概率就集中（如标点、格式token）← 无意义

teacher熵高的原因可能是：
  a. feedback无法消除这个位置的不确定性 ← 噪声
  b. 这个位置本身是关键推理分叉点 ← 恰恰是最需要学习的地方

所以"熵低=信号可靠"这个假设不一定成立
```

```plain
要真正验证"teacher熵=信号质量"这个假设
需要：

消融1：可视化高权重token vs 低权重token
  高权重token是否真的是关键推理步骤？
  低权重token是否真的是噪声？

消融2：对比student熵加权
  如果用student熵加权效果类似
  说明EW的作用不是"过滤teacher噪声"
  而只是一种正则化

消融3：对比直接降低learning rate
  如果效果类似
  说明EW只是隐式降低了有效学习率
```

```plain
teacher熵低 + teacher在student选的token上概率低
→ teacher很确定，但确定的是别的token
→ student明显选错了
→ 需要重点学 → 高权重

teacher熵高
→ teacher自己也不确定
→ 这个位置本来就ambiguous
→ 不值得强迫student改变 → 低权重

学习的目标（被优化的量）：
  KL_t = KL(student_t || teacher_t)
  → 这是loss本身，需要被最小化
  → 不应该同时又作为weight

决定学习权重的量（独立的判断）：
  H_teacher_t  → teacher可信吗？
  P_teacher(ŷ_t) → teacher认为student选对了吗？

方案A：直接相乘

  w_t = (1 - P_teacher(ŷ_t)) * (1 - H_teacher_t / H_max)
        ↑ student选错的程度      ↑ teacher的确定程度

  归一化后作为weight：
  w_t = softmax(w_t / τ) * L

方案B：只用P（最简单）

  w_t = softmax(-P_teacher(ŷ_t) / τ) * L
  
  P低 → -P大 → softmax后权重高
  直觉：teacher越不认同student的选择，越需要学

方案C：P和H联合（你最初的直觉）

  confidence_t = -P_teacher(ŷ_t) * (1 - H_teacher_t)
  w_t = softmax(confidence_t / τ) * L

  两个条件都要满足：
    P低（student选错）✅
    H低（teacher确定）✅
  才给高权重

  w_t = f(P_teacher(ŷ_t), H_teacher_t)   ← 独立于KL
  loss = Σ_t w_t * KL_t                  ← KL只作为loss
```

```python
H低 + wrong高（P低）
  teacher确定 且 student选错
  → 比如teacher确定答案是"2"，student选了"3"
  → 最需要纠正 → 权重最高 ✅

H低 + wrong低（P高）
  teacher确定 且 student选对
  → teacher和student都认为是"2"
  → 没有分歧，KL本来就小 → 权重低 ✅

H高 + wrong高（P低）
  teacher不确定 且 student选错
  → teacher对这个位置有多种可能的选择
  → 信号不可靠，虽然student可能错了但不确定 → 权重中 ⚠️

H高 + wrong低（P高）
  teacher不确定 且 student碰巧选对
  → teacher虽然不确定，但student选的token概率还可以
  → 权重最低 ✅
```

## V2 公式
```python
# Step 1: teacher确定程度
teacher_certainty = 1.0 - H_normalized      # H_normalized = H / H_max

# Step 2: student选错程度  
student_wrong = 1.0 - teacher_prob_on_student   # = 1 - P_teacher(ŷ_t)

# Step 3: 联合score
joint_score = teacher_certainty * student_wrong

# Step 4: softmax加权
confidence_weights = softmax(joint_score / τ)  * L
```

---

## 展开写
$ w_t = \text{softmax}\left(\frac{(1 - \hat{H}_t) \cdot (1 - P_{\theta}(\hat{y}_t \mid x, f, y_{<t}))}{\tau}\right) \cdot L $

其中：

```plain
Ĥ_t = H_teacher_t / H_max          ← 归一化到[0,1]的teacher熵

P_θ(ŷ_t | x, f, y<t)              ← teacher在student token上的概率

τ                                   ← 温度

L                                   ← 有效序列长度（还原token-mean尺度）
```

---

## 和旧版EW的对比
```plain
旧版（只用熵）：
  joint_score = -H_teacher_t
  w_t = softmax(-H_t / τ) * L

新版（熵 + P联合）：
  joint_score = (1 - H_normalized) * (1 - P_teacher(ŷ_t))
  w_t = softmax(joint_score / τ) * L -> v2

或者试试 w_t = softmax(sqrt((1-H)(1-P)) / τ) * L？-> v3

区别：
  旧版：只考虑teacher是否确定
  新版：同时考虑teacher是否确定 + student是否选错
```



**核心问题**：V1/V2/V3都无法区分以下两种情形：

```plain
情形A：标点符号 "."
  student熵 ≈ 0.01（本来就确定）
  teacher熵 ≈ 0.01（feedback没有改变任何事）
  → ΔH ≈ 0 → 低权重 ✅

情形B：推理关键词 "increase"
  student熵 ≈ 1.5（不确定是increase还是decrease）
  teacher熵 ≈ 0.3（看了feedback后teacher变确定了）
  → ΔH = 1.5 - 0.3 = 1.2 → 高权重 ✅

情形C：噪声位置
  student熵 ≈ 0.8
  teacher熵 ≈ 1.2（看了feedback反而更困惑）
  → ΔH = 0.8 - 1.2 = -0.4 → clamp到0 → 零权重 ✅
```

**信息增益的本质**：feedback在这个位置带来了多少信息量？

$ \Delta H_t = H(\pi_\theta(\cdot|x,y_{<t})) - H(\pi_\theta(\cdot|x,f,y_{<t})) $

$ w_t = \text{softmax}\left(\frac{\Delta H_t^+}{\tau}\right) \cdot L, \quad \Delta H_t^+ = \max(\Delta H_t, 0) $

## 四个版本的完整对比
```plain
位置类型          H_student  H_teacher  ΔH    V1权重  V2权重  V4权重
─────────────────────────────────────────────────────────────────
标点 "."          0.01       0.01       0.00   高      ?      零   ✅
格式词 "Answer"   0.05       0.04       0.01   高      ?      极低 ✅
推理分叉点        1.50       0.30       1.20   低      ?      最高 ✅
噪声位置          0.80       1.20      -0.40   低      ?      零   ✅
```



---

## 最终loss
$ \mathcal{L} = \sum_t w_t \cdot \text{KL}(\pi_\theta(\cdot|x,y_{<t}) \| \pi_\theta(\cdot|x,f,y_{<t})) $

# 附录


## 细节
SDPO的KL散度是**logit粒度**聚和。

## 从代码看
```python
# compute_self_distillation_loss 中
kl_loss = F.kl_div(
    student_distill_log_probs,   # (B, T, K)  K=top-K词表
    teacher_distill_log_probs,   # (B, T, K)
    reduction="none",
    log_target=True
)
per_token_loss = kl_loss.sum(-1)  # (B, T)  ← 在词表维度求和，得到每个token位置的KL
```

每个位置 $ t $ 的loss是：

$ \text{KL}_t = \sum_{v \in \text{TopK}} p_{\text{student}}(v|x,y_{<t}) \log \frac{p_{\text{student}}(v|x,y_{<t})}{p_{\text{teacher}}(v|x,f,y_{<t})} $

---

## 三个粒度的对比
| 粒度 | 含义 | 维度 |
| --- | --- | --- |
| **序列级** | 每条序列一个标量 | (B,) |
| **token级** | 每个位置一个标量 | (B, T) |
| **logit级** | 每个位置×每个词 | (B, T, V) |


```plain
GRPO：序列级  → 一条回答一个reward，所有token共享
SDPO：logit级 → 先在V维度求和 → 变成token级的KL
              → 再在T维度聚合 → 变成最终loss标量
```

熵加权就是在 `per_token_loss`（B, T）上操作：

```python
per_token_loss = kl_loss.sum(-1)        # (B, T) ← token级KL
per_token_loss = per_token_loss * w_t   # (B, T) ← 乘token级权重
loss = token_mean(per_token_loss)       # 标量
```

所以 $ w_t $ 和 $ \text{KL}_t $ **粒度完全匹配**，乘法是逐token的，设计上没有问题。

