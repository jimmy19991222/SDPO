```plain
GRPO告诉模型：
  "y_2比y_1好，多生成y_2这样的"
  但不告诉模型：哪些token让y_2更好

SDPO告诉模型：
  "看了feedback后，这个位置应该输出这个分布"
  但不直接告诉模型：y_2整体比y_1好

联合：
  SDPO提供token级reward（每步怎么走）
  GRPO提供group级标准化advantage
```



# Token-Level Advantage via Self-Distillation (TASD)
## 核心思想
```plain
GRPO的思路：用reward做归一化，得到advantage，再做policy gradient
SDPO的思路：用teacher得到token级的dense信号

本方法：
  用SDPO的思路得到token级reward
  用GRPO的思路做归一化得到advantage
  两者结合，实现token级的精细credit assignment
```

---

## 方法框架
### Step 1：采样Rollout
对每个prompt x，采样n条response：

$ \{y_1, y_2, \ldots, y_n\} \sim \pi_\theta(\cdot \mid x) $

获取环境feedback $ f_i $（runtime error、失败测试等）。

---

### Step 2：构建Self-Teacher
对每条response，构建teacher的context：

$ \text{teacher context} = [x, \ y_\text{best}, \ f_i, \ y_i] $

```plain
y_best：组内最优response（成功的rollout，或reward最高的）
f_i：环境feedback
y_i：student的原始response

成功response：y_i本身就是y_best，teacher天然认可
失败response：teacher看到y_best后，对y_i重新评价
```

Teacher前向，得到每个token位置的分布：

$ \pi_\theta(\cdot \mid x, f_i, y_{i,<t}) \quad \text{（self-teacher）} $

---

### Step 3：计算Token级Reward
用teacher和student的分布差异作为token级reward，具体形式待实验验证，候选包括：

**候选A：KL divergence（分布级）**

$ r_{i,t} = -\text{KL}\left(\pi_\text{teacher}(\cdot \mid x, f_i, y_{i,<t}) \,\|\, \pi_\text{student}(\cdot \mid x, y_{i,<t})\right) $

```plain
优点：包含完整分布信息（top-K token）
     和SDPO原始loss一脉相承
缺点：永远非正，依赖full/top-K logits计算
```

**候选B：Per-token log ratio（生成token级）**

$ r_{i,t} = \log \pi_\text{teacher}(y_{i,t} \mid x, f_i, y_{i,<t}) - \log \pi_\text{student}(y_{i,t} \mid x, y_{i,<t}) $

```plain
优点：有正有负，天然有baseline
     只需要生成token的log prob，计算高效
缺点：只看生成的单个token，丢失分布信息
```

两者本质关系：

$ \text{KL}_t = \mathbb{E}_{\hat{y} \sim \pi_\text{teacher}}\left[\log \frac{\pi_\text{teacher}(\hat{y})}{\pi_\text{student}(\hat{y})}\right] $

$ \log \frac{\pi_\text{teacher}(y_t)}{\pi_\text{student}(y_t)} = \text{KL的单点估计（}\hat{y}=y_t\text{）} $

```plain
KL是log ratio在teacher分布下的期望
log ratio是KL的单点（Monte Carlo）估计
KL信息量更丰富，log ratio计算更高效
```

---

### Step 4：统一Token级归一化
将group内所有response的所有token的reward放在同一个池子里归一化：

$ A_{i,t} = \frac{r_{i,t} - \mu}{\sigma + \epsilon} $

其中$ \mu $和$ \sigma $在batch内所有有效token上计算：

$ \mu = \frac{\sum_{i,t} r_{i,t} \cdot m_{i,t}}{\sum_{i,t} m_{i,t}}, \quad \sigma = \text{std}_{i,t}(r_{i,t} \cdot m_{i,t}) $

$ m_{i,t} $为有效token的mask。

**归一化的含义：**

```plain
baseline = 当前batch所有token的平均reward水平

A_{i,t} > 0：这个token的teacher-student差距
             优于batch平均水平 → 强化
A_{i,t} < 0：这个token的teacher-student差距
             劣于batch平均水平 → 抑制
A_{i,t} ≈ 0：和batch平均水平相当 → 无需处理
```

**为什么token间可以统一归一化：**

```plain
r_{i,t}对所有(i,t)：
  单位一致：teacher和student的log概率差
  含义一致：teacher相对student对该位置的认可程度
  → 在同一尺度上，可以统一比较 ✅
```

---

### Step 5：Policy Gradient
$ \mathcal{L} = -\frac{\sum_{i,t} A_{i,t} \cdot \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}) \cdot m_{i,t}}{\sum_{i,t} m_{i,t}} $

---

## 数值示例
```plain
prompt x，n=4条rollout，T=4：

token级reward r_{i,t}（以候选A为例）：

         t=0    t=1    t=2    t=3
y_1:  [-0.01, -0.01, -0.02, -0.02]   ← 成功，KL整体小
y_2:  [-0.01, -0.01, -0.82, -0.75]   ← 失败，t=2,3错误
y_3:  [-0.01, -0.01, -0.79, -0.71]   ← 失败
y_4:  [-0.01, -0.01, -0.02, -0.02]   ← 成功

μ = -0.20,  σ = 0.35

归一化后A_{i,t}：

         t=0    t=1    t=2    t=3
y_1:  [+0.54, +0.54, +0.51, +0.51]   ← 全部正advantage
y_2:  [+0.54, +0.54, -1.77, -1.57]   ← 错误位置强负
y_3:  [+0.54, +0.54, -1.69, -1.46]   ← 错误位置强负
y_4:  [+0.54, +0.54, +0.51, +0.51]   ← 全部正advantage
```

---

## 为什么有效
### 自动捕获两个维度
```plain
组间对比（哪条response整体更好）：
  成功response：r_{i,t}整体高（KL小）
  → 归一化后advantage整体正 ✅
  
  失败response：r_{i,t}在错误位置极低（KL大）
  → 归一化后advantage在错误位置强负 ✅

Token级对比（哪个位置更重要）：
  关键错误token：r_{i,t}极低
  → 强负advantage，获得最强梯度 ✅
  
  无信息token：r_{i,t}接近0
  → advantage接近均值，梯度小 ✅

两个维度在一次归一化中自然实现。
```

### 解决Advantage Collapse
```plain
GRPO的问题：
  全部成功 → r_i都=1 → std=0 → collapse
  全部失败 → r_i都=0 → std=0 → collapse

本方法：
  reward是token级的连续值
  即使全部成功，各token的KL各不相同
  即使全部失败，各token的错误程度各不相同
  → std永远≠0 → 不会collapse ✅
```

### 不需要环境Reward
```plain
GRPO：必须有环境的binary reward
本方法：teacher提供所有信号
       不依赖任何环境reward
       在reward sparse甚至无reward的情况下仍然有效
```

---

## 实现
```python
def compute_advantage(
    teacher_logp,    # (n, T, K) top-K teacher log probs（候选A）
    student_logp,    # (n, T, K) top-K student log probs（候选A）
    teacher_logp_generated,  # (n, T) 生成token的teacher logp（候选B）
    student_logp_generated,  # (n, T) 生成token的student logp（候选B）
    loss_mask,       # (n, T)
    reward_type='kl',
    eps=1e-6
):
    if reward_type == 'kl':
        # 候选A：用KL作为reward
        # top-K近似KL
        kl = (teacher_logp.exp() * 
              (teacher_logp - student_logp)).sum(dim=-1)  # (n, T)
        reward = -kl

    elif reward_type == 'log_ratio':
        # 候选B：用log ratio作为reward
        reward = (teacher_logp_generated 
                  - student_logp_generated)               # (n, T)

    # 统一token级归一化
    valid = reward[loss_mask.bool()]
    mu = valid.mean()
    sigma = valid.std()

    A = (reward - mu) / (sigma + eps)
    A = A * loss_mask

    return A


def compute_loss(A, student_logp_generated, loss_mask):
    loss = -(A * student_logp_generated * loss_mask).sum()
    loss = loss / loss_mask.sum()
    return loss
```

---

## 与现有方法对比
|  | GRPO | SDPO | **TASD（本方法）** |
| --- | --- | --- | --- |
| **Reward来源** | 环境binary reward | Teacher KL（隐式） | Teacher KL/log ratio |
| **Reward粒度** | 序列级 | Token级 | Token级 |
| **归一化** | 组间（序列级） | 无（直接minimize） | 组内所有token统一 |
| **组间对比** | ✅ 显式 | ❌ | ✅ 隐式包含 |
| **Token级区分** | ❌ 广播 | ✅ 等权 | ✅ 归一化加权 |
| **成功response参与** | ✅ | ❌ | ✅ |
| **Advantage collapse** | 易发生 | 不发生 | 不发生 |
| **需要环境reward** | ✅ | ❌ | ❌ |


---

## 与SDPO的关系
```plain
SDPO：
  直接minimize KL
  L_SDPO = mean_{i,t}(KL_t)
  每个token等权，无组间对比

TASD：
  用KL（或log ratio）作为token级reward
  做统一归一化得到advantage
  再做policy gradient

本质升级：
  SDPO："让student无条件向teacher靠拢"
  TASD："让student重点学习
         相对于batch平均水平更需要改进的token"

两者可以联合使用：
  L_total = L_TASD + α * L_SDPO
  L_TASD提供相对信号（哪里更重要）
  L_SDPO提供绝对信号（整体向teacher靠拢）
```

---

## 待验证的实验问题
```plain
1. Reward形式：
   候选A（KL）vs 候选B（log ratio）
   哪种reward作为token级信号更有效？
```

---

## 一句话总结
```plain
TASD = SDPO的token级reward + GRPO的归一化思路

用teacher和student的分布差异（KL或log ratio）
作为token级reward

对batch内所有token统一 归一化得到advantage

自然地同时实现：
  token级精细credit assignment
  无需环境reward
  不会advantage collapse
```



实验崩溃->TASD-sciknoweval-biology-mbs32-train32-rollout8-lr1e-5-rtlog_ratio-tfnone-topk100-usatTrue-isrTrue-Qwen-Qwen3-8B-2026-03-19_19-08-41

## 崩溃的直接证据
### 1. 验证集准确率急剧下降
```plain
val-core/sciknoweval/acc/mean@16:
  step 10: 0.3737  ← 峰值
  step 20: 0.1600  ← 崩溃，跌回基线以下
  
val-core/sciknoweval/acc/maj@16/mean:
  step 10: 0.4114
  step 20: 0.2612  ← 大幅下跌
```

### 2. success_rate崩溃
```plain
tasd/success_rate:
  step 5:  0.4648  ← 峰值
  step 22: 0.0195  ← 接近0，几乎全部失败
```

### 3. response长度爆炸
```plain
response_length/mean:
  step 13: 257     ← 正常
  step 20: 5839    ← 爆炸，22倍
  step 22: 5737    ← 持续

response_length/clip_ratio:
  step 22: 0.6757  ← 67%的response被截断到8192
```

### 4. PPL爆炸
```plain
rollout_corr/rollout_ppl:
  step 1:  1.65   ← 正常
  step 22: 6214   ← 完全崩溃
```

---

## 根本原因分析
### 核心问题：advantage_pos_rate趋近于0
```plain
tasd/advantage_pos_rate:
  step 2:  0.363   ← 正常，36%的token有正advantage
  step 22: 0.004   ← 只有0.4%的token有正advantage
```

**几乎所有token的advantage都是负的，模型被疯狂惩罚。**

### 为什么会这样？
看token_reward的分布：

```plain
tasd/token_reward_mean_success（成功response）:
  step 2:  0.397
  step 16: 6.550   ← 成功response的reward越来越大

tasd/token_reward_mean_fail（失败response）:
  step 2:  -0.166
  step 22: -0.0009 ← 失败response的reward趋近于0
```

**这里有个恶性循环：**

```plain
success_rate下降（0.46 → 0.02）
  ↓
几乎没有成功response → 没有y_best可以放入teacher context
  ↓
self_distillation/reprompt_sample_fraction下降（0.97 → 0.16）
  ↓
大多数response的teacher context = 只有自己（或为空）
  ↓
teacher ≈ student → log_ratio ≈ 0
  ↓
所有token的reward ≈ 0
  ↓
归一化后：少数有teacher context的token reward偏大
          大多数token reward ≈ 0，归一化后advantage为负
  ↓
模型被强烈惩罚 → 输出混乱 → response变长 → success_rate更低
```

---

## 验证这个假设
```plain
self_distillation/reprompt_sample_fraction:
  step 12: 0.9687  ← 97%有teacher context
  step 22: 0.1562  ← 只有16%有teacher context

self_distillation/success_sample_fraction:
  step 12: 0.9687
  step 22: 0.1562  ← 完全一致，说明teacher context完全依赖成功response
```

你的配置`use_self_as_teacher_on_success=True`，但没有环境feedback，所以：

```plain
有成功response → reprompt_fraction高 → teacher有信号
没有成功response → reprompt_fraction低 → teacher无信号
```

**一旦success_rate下降，整个系统就失去信号来源，进入死亡螺旋。**

---

## 具体问题定位
### 问题1：advantage归一化在success_rate低时失效
```python
# 当只有16%的response有teacher context时：
# 84%的response: token_reward ≈ 0（teacher=student）
# 16%的response: token_reward有正有负

group_mean ≈ 接近0（被84%的0拉低）
group_std  ≈ 很小

# 结果：
# 有teacher context的response：advantage被放大
# 没有teacher context的response：advantage = (0 - mean) / std ≈ 负值
```

**84%的response无辜地获得了负advantage。**

### 问题2：response长度爆炸的直接原因
```plain
actor/entropy:
  step 1:  4.05   ← 初始高熵
  step 13: 约0.5  ← 正常收敛
  step 22: 0.548  ← 但response却很长？
```

entropy没有爆炸，但response很长，说明模型在重复生成相同token（低熵但长序列）。

这是负advantage导致的：模型被惩罚所有"正常"输出，开始退化。

---

## 解决方案
### 方案1（最直接）：当没有成功response时，跳过TASD更新
```python
# 在compute_tasd_advantage中
if self_distillation_mask.sum() == 0:
    # 本batch没有任何有效teacher context
    # 返回全0 advantage，不更新
    return torch.zeros_like(token_level_rewards), torch.zeros_like(token_level_rewards)
```

### 方案2：只对有teacher context的response计算advantage
```python
# 当前：没有teacher context的response advantage = 负值
# 修改：没有teacher context的response advantage = 0

for i in indices:
    if effective_mask[i].sum() == 0:
        # 没有teacher context，不更新这条response
        advantages[i] = 0
        continue
    adv_i = (token_level_rewards[i] - group_mean) / group_std
    advantages[i] = adv_i * effective_mask[i]
```

### 方案3：混合verifier reward作为fallback
```python
# 当没有teacher context时，fallback到GRPO-style advantage
if self_distillation_mask[i] == 0:
    # 用verifier reward
    advantages[i] = verifier_advantage[i]
else:
    # 用TASD advantage
    advantages[i] = tasd_advantage[i]
```

### 方案4（治本）：加入环境feedback
你的配置显示：

```plain
self_distillation/feedback_available_fraction: 0  ← 没有环境feedback
```

Biology任务有没有办法提供feedback？即使是简单的"回答错误"也比没有好，可以让失败response也有teacher context。

---

## 最快的fix
**方案2是最小改动且最合理的：**

```python
@register_adv_est(AdvantageEstimator.TASD)
def compute_tasd_advantage(...):
    with torch.no_grad():
        effective_mask = response_mask
        if self_distillation_mask is not None:
            effective_mask = response_mask * self_distillation_mask.unsqueeze(1).float()

        advantages = torch.zeros_like(token_level_rewards)

        for uid, indices in uid_to_indices.items():
            # 只收集有teacher context的response的token
            group_token_rewards = []
            valid_indices = []
            for i in indices:
                if effective_mask[i].sum() == 0:
                    continue  # 跳过没有teacher context的response
                valid_tokens = token_level_rewards[i][effective_mask[i].bool()]
                group_token_rewards.append(valid_tokens)
                valid_indices.append(i)

            if len(valid_indices) <= 1:
                continue  # group内有效response太少，跳过

            all_rewards = torch.cat(group_token_rewards)
            group_mean = all_rewards.mean()
            group_std = all_rewards.std(unbiased=False).clamp(min=epsilon)

            # 只更新有teacher context的response
            for i in valid_indices:
                adv_i = (token_level_rewards[i] - group_mean) / group_std
                advantages[i] = adv_i * effective_mask[i]
            # 没有teacher context的response: advantages[i] = 0（不更新）

    return advantages, advantages
```

这样的逻辑是：

+ 有teacher context → 参与归一化，获得正或负advantage
+ 没有teacher context → advantage=0，梯度为0，不更新



另一个思路：

能不能在prompt里选一个成功，一个失败的案例呢？



# 附录