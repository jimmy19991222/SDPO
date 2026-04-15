# TASD (Test-time Self-Distillation) 创新点总结

## 1. 熵门控机制 (Entropy Gate)

### 问题背景
在知识蒸馏中，并非所有 token 位置的 teacher 信号都同样可靠。当 teacher 自身不确定（熵高）时，其指导信号可能引入噪声。

### 创新方案
引入熵门控机制，基于 teacher-student 熵差异筛选有效训练信号：

- **Hard Gate**：`gate_mask = (H_teacher < H_student)`，二值筛选
- **Soft Gate**：`gate_weight = (H_student - H_teacher)_+`，连续权重

### 关键技术细节

#### 1.1 TopK 熵归一化校正
TopK 计算的熵最大值为 `log(K)`，不同 K 值的熵不可直接比较。

**校正公式**：
```python
H_normalized = H_topk / log(K)
```

校正后熵 ∈ [0, 1]，使不同 `DISTILL_TOPK` 设置下的熵门控阈值稳定。

#### 1.2 Gate 后 Token 梯度修复
被 entropy gate 过滤的 token（`gate_mask=0`）：
- `reward` 置 0
- **关键修复**：从 `effective_mask` 中排除，不参与 `group_mean` 计算
- `advantage` 直接置 0，无梯度传播

**意义**：避免被过滤 token 产生错误的负梯度。

---

## 2. Advantage 归一化保护机制 (adv_std_floor)

### 问题背景
Group 内所有 token 的 reward 接近时，`group_std` 可能趋近于 0，导致归一化后的 advantage 爆炸。

### 创新方案：三态配置

| 配置值 | 计算方式 | 适用场景 |
|--------|----------|----------|
| `"none"` / `"0.0"` | `floor = 0` | 不使用保护 |
| `"auto"` | `floor = 1/√N` (N=group_size) | 自动适配 |
| `float` (如 `"0.1"`) | `floor = float` | 固定阈值 |

**"auto" 模式原理**：当 group 内只有 1 个正确样本时，binary reward 的 std ≈ `1/√N`。

---

## 3. 简化版 TASD 架构设计

### 核心原则
- 移除复杂特性（Teacher-GAE、Future-KL Modulation 等），聚焦核心算法
- 配置化设计，所有超参可通过命令行覆盖

### 关键参数

```yaml
algorithm:
  tasd:
    reward_type: "teacher_prob"    # teacher_prob | teacher_log_prob
    entropy_gate: "none"          # none | hard | soft
    distill_topk: 100             # 熵门控需要的 topk 规模
    
    norm_adv_by_std: false        # 是否 std 归一化
    adv_std_floor: "auto"         # std 下界保护
    clip_adv: true                # advantage clipping
    clip_adv_value: 2.0           # clipping 阈值
```

### 代码路径简化

| 模块 | 文件 | 职责 |
|------|------|------|
| Token Reward 计算 | `core_algos.py::compute_tasd_token_rewards` | reward + entropy gate |
| Advantage 计算 | `core_algos.py::compute_tasd_advantage` | group 归一化 + clipping |
| Teacher Forward | `dp_actor.py::compute_teacher_log_probs` | teacher log_prob 获取 |

---

## 4. 实验配置体系

### 4.1 按数据集拆分的 Sweep 脚本
- `submit_tasd_simple_sweep.sh`：TASD 简化版
- `submit_baseline_sciknoweval_sweep.sh`：sciknoweval 基线
- `submit_baseline_lcb_sweep.sh`：LCB 基线

### 4.2 参数化训练脚本
所有超参通过环境变量注入，支持 Hydra 命令行覆盖：

```bash
algorithm.tasd.norm_adv_by_std=${NORM_ADV_BY_STD}
algorithm.tasd.adv_std_floor=${ADV_STD_FLOOR}
algorithm.tasd.clip_adv=${CLIP_ADV}
```

---

## 5. 与原始 TASD 仓库的对齐

### 已修复的差异

| 问题 | 原始行为 | 修复后 |
|------|----------|--------|
| solution 选择 | `solution_idxs[0]` 固定 | `random.choice` 随机 |
| group_std 计算 | `std()` (unbiased=True) | `std(unbiased=False)` |
| entropy gate 梯度 | gate 后 token 参与 group_mean | 从 effective_mask 排除 |

### 保持一致的实现
- `teacher_prob` reward 计算：`clamp(min=-20).exp()`
- `compute_tasd_advantage` group 归一化逻辑
- `use_self_as_teacher_on_success` 成功 rollout 处理

---

## 附录：实验命名规范

```
TASD-simple-{dataset}-rt_{reward_type}{entropy_tag}-clip{value}{topk_tag}{rep_tag}{std_tag}-{model}-{timestamp}
```

示例：
- `TASD-simple-sciknoweval-biology-rt_teacher_prob-noGate-clip2.0-rep1.05-std_auto-Qwen3-8B-20260410_120000`
- `TASD-simple-sciknoweval-biology-rt_teacher_prob-gate_hard-clip2.0-topk100-rep1.05-normStd-Qwen3-8B-20260410_120000`
