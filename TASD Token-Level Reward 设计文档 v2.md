## 一、核心问题回顾
经过多次实验，发现所有基于teacher/student**分布差异**的reward都有根本缺陷：

| reward_type | 问题 |
| --- | --- |
| `log_ratio` | 失败reward≈0，成功/失败信号不对称 |
| `prob_diff` | 同上，且训练中熵爆炸 |
| `jsd` | teacher分布过于尖锐导致方向反转 |
| `vocab_log_ratio` | student topk排序偏差导致方向反转 |


---

## 二、根本原因
这类reward隐含的问题：

```plain
teacher做forward时，输入是student已生成的token序列
→ teacher被迫给student的每个token分配概率
→ 对失败token，teacher和student概率差距不大
→ log_ratio ≈ 0，失败惩罚信号消失
```

更深层的原因在于这类reward回答的问题本身就错了：

```plain
❌ 错误的问题："teacher比student更认可这个token吗？"
   student选了y_t → student对y_t概率天然不低
   → 分母天然偏大，信号被稀释

✅ 正确的问题："teacher认可这个token吗？"
```

---

## 三、新的设计直觉
**直接看teacher的认可度，不看相对差异。**

```plain
teacher context包含正确答案
→ teacher知道哪些token在正确推理路径上
→ 正确路径上的token → teacher给高概率
→ 错误路径上的token → teacher给低概率

直接用π_teacher(y_t)作为reward：
  成功token → teacher认可 → 高概率 → reward高  ✅
  失败token → teacher不认可 → 低概率 → reward低 ✅
```

---

## 四、Reward设计
### 1. teacher_prob（基础版）
```python
reward_t = π_teacher(y_t) = teacher_log_probs.exp()  # (B, T) ∈ (0, 1)
```

**含义：** teacher对student实际选择的token有多认可。

**优点：**

+ 天然有界 `(0, 1)`，不需要transform
+ 方向天然正确，无反转风险
+ 零额外计算（`teacher_log_probs`已有）

---

### 2. teacher_prob_certainty（增强版）
```python
# teacher的近似熵（基于topk）
teacher_topk_probs  = teacher_topk_log_probs.exp()           # (B, T, K)
teacher_entropy_t   = -(teacher_topk_probs *
                         teacher_topk_log_probs).sum(dim=-1) # (B, T)
H_max               = log(K)
teacher_certainty_t = 1.0 - teacher_entropy_t / H_max        # (B, T) ∈ (0, 1)

reward_t = π_teacher(y_t) * teacher_certainty_t              # (B, T) ∈ (0, 1)
```

**含义：** teacher的认可度 × teacher在该位置的确定性。

**Teacher Certainty的直觉：**

```plain
熵低（certainty高）：teacher对该位置有明确偏好
  → 关键决策token，如答案选项、关键推理步骤
  → 信号可信，放大reward

熵高（certainty低）：teacher对该位置不确定
  → 过渡token、连接词等可替换的词
  → 信号噪声大，压缩reward
```

天然过滤噪声，强化关键位置的信号。

---

### 3. top1_match（离散版）
```python
teacher_top1   = teacher_topk_indices[:, :, 0]   # (B, T) teacher概率最高的token
student_chosen = student_topk_indices[:, :, 0]   # (B, T) student实际选的token

reward_t = (teacher_top1 == student_chosen).float()  # (B, T) ∈ {0, 1}
```

**含义：** student的选择是否是teacher的首选。

**缺点：** binary信号粒度粗，teacher第2名和第100名没有区别。

---

### 4. topk_match（离散连续版）
```python
student_chosen = student_topk_indices[:, :, 0]    # (B, T)

# student选的token在teacher topk里的排名
match_matrix = (teacher_topk_indices == student_chosen.unsqueeze(-1))  # (B, T, K)
rank          = match_matrix.float().argmax(dim=-1)   # (B, T) ∈ [0, K-1]
matched       = match_matrix.any(dim=-1)              # (B, T) bool

# 排名越靠前reward越高，不在topk内reward=0
reward_t = torch.where(
    matched,
    1.0 - rank.float() / K,          # rank=0 → 1.0，rank=K-1 → 1/K
    torch.zeros_like(rank.float()),   # 不在topk内 → 0.0
)  # (B, T) ∈ [0, 1]
```

**含义：** student的选择在teacher topk中的排名，排名越靠前reward越高。

```plain
student选了teacher rank=0的token  → reward = 1.0
student选了teacher rank=1的token  → reward = 0.99
student选了teacher rank=99的token → reward = 0.01
student选了topk外的token          → reward = 0.0
```

比`top1_match`更平滑，比`teacher_prob`更直观。

---

## 五、Reward类型对比
|  | teacher_prob | teacher_prob_certainty | top1_match | topk_match |
| --- | --- | --- | --- | --- |
| 范围 | (0,1) | (0,1) | {0,1} | [0,1] |
| 方向 | ✅ | ✅ | ✅ | ✅ |
| 信号粒度 | 连续 | 连续 | 离散binary | 离散连续 |
| 噪声过滤 | 无 | ✅ certainty加权 | 无 | 部分（排名） |
| 同义词容忍 | ✅ | ✅ | ❌ | ✅ |
| 额外计算 | 无 | topk entropy | 无 | argmax on K |
| 推荐场景 | 快速验证 | 正式训练 | - | 对比实验 |


---

## 六、Advantage计算
```python
# group内flat均值作为baseline
all_rewards = torch.cat([
    token_rewards[i][mask[i].bool()]
    for i in valid_indices
])
group_mean   = all_rewards.mean()
advantage_t  = reward_t - group_mean
```

**为什么flat均值足够，不需要seq_mean：**

```plain
teacher_prob ∈ (0, 1)，每个token独立
不随长度累积（不像sum会随长度增大）
→ 长response和短response的token reward量级相同
→ flat均值不受长度偏差影响
```

---

## 七、预期的reward分布
```plain
成功response：student选了正确推理路径上的token
  → teacher对这些token概率高
  → token_reward_mean_success 高

失败response：student选了错误路径上的token
  → teacher对这些token概率低
  → token_reward_mean_fail 低

group_mean介于两者之间：
  成功token advantage > 0  ✅
  失败token advantage < 0  ✅
```

---

## 八、和旧设计的对比
|  | 旧设计（分布差异） | 新设计（teacher认可度） |
| --- | --- | --- |
| 核心问题 | teacher比student更认可吗 | teacher认可吗 |
| 失败reward | ≈0（信号消失） | 低，有明确负advantage |
| 方向稳定性 | 容易反转 | 天然稳定 |
| 是否需要verifier | 是（补充失败信号） | 否（信号本身已区分） |
| 范围 | 无界或不对称 | 天然(0,1) |


---

## 九、实现
```python
elif reward_type == "teacher_prob":
    reward = teacher_log_probs.exp()  # (B, T)

elif reward_type == "teacher_prob_certainty":
    teacher_topk_probs  = teacher_topk_log_probs.exp()
    teacher_prob_t      = teacher_log_probs.exp()
    teacher_entropy_t   = -(
        teacher_topk_probs * teacher_topk_log_probs
    ).sum(dim=-1)
    H_max = torch.log(torch.tensor(
        teacher_topk_log_probs.shape[-1],
        dtype=teacher_topk_log_probs.dtype,
        device=teacher_topk_log_probs.device,
    ))
    teacher_certainty_t = 1.0 - teacher_entropy_t / H_max
    reward = teacher_prob_t * teacher_certainty_t  # (B, T)

elif reward_type == "top1_match":
    teacher_top1   = teacher_topk_indices[:, :, 0]
    student_chosen = student_topk_indices[:, :, 0]
    reward = (teacher_top1 == student_chosen).float()  # (B, T)

elif reward_type == "topk_match":
    student_chosen = student_topk_indices[:, :, 0]
    match_matrix   = (teacher_topk_indices == student_chosen.unsqueeze(-1))
    rank           = match_matrix.float().argmax(dim=-1)
    matched        = match_matrix.any(dim=-1)
    K              = teacher_topk_indices.shape[-1]
    reward = torch.where(
        matched,
        1.0 - rank.float() / K,
        torch.zeros_like(rank.float()),
    )  # (B, T)
```

---

## 十、配置
```yaml
# 快速验证
tasd:
  reward_type: teacher_prob
  reward_transform: none
  reward_scale: 1.0
  topk: 100

# 正式训练
tasd:
  reward_type: teacher_prob_certainty
  reward_transform: none
  reward_scale: 1.0
  topk: 100

# 对比实验
tasd:
  reward_type: topk_match   # 或 top1_match
  reward_transform: none
  reward_scale: 1.0
  topk: 100
```

---

## 十一、待验证
```plain
1. token_reward_mean_success > token_reward_mean_fail  ← 方向验证
2. success_rate 稳定不下降                             ← 训练稳定性
3. response_length 不爆炸                              ← 长度控制
4. teacher_prob vs teacher_prob_certainty 信号质量对比
```

