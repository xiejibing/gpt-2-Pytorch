# Linear Transformer vs Gated DeltaNet 论文对比

## 概览

| | Linear Transformer (2020) | Gated DeltaNet (2024) |
|---|---|---|
| 论文 | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | Gated Delta Networks: Improving Mamba 2 with Delta Rule |
| 作者 | Katharopoulos et al. (Idiap/EPFL) | Yang et al. (MIT CSAIL / NVIDIA) |
| 发表 | ICML 2020 | arXiv 2024.12 |
| 核心思想 | kernel trick 近似 softmax attention | delta 规则 + 门控，在线线性回归 |
| 状态更新 | `S_t = S_{t-1} + φ(k_t)·v_t^T` | `S_t = α_t·S_{t-1}·(I - β_t·k_t·k_t^T) + β_t·v_t·k_t^T` |
| 复杂度 | O(N) 时间，O(N) 或 O(1) 内存 | O(N) 时间，O(1) 内存（推理） |

---

## 一、共同的起源

两篇论文都起源于同一个问题：**如何把 O(N²) 的 softmax attention 降到 O(N)**。

```
                     Softmax Attention O(N²)
                            │
                    如何降到 O(N)？
                            │
            ┌───────────────┴───────────────┐
            │                               │
    Linear Transformer (2020)         Gated DeltaNet (2024)
    kernel 逼近路线                    在线学习 + 门控路线
            │                               │
    用 φ(q)·φ(k) ≈ exp(q·k)          放弃逼近 softmax
    保留 attention 的语义             转而做在线回归
            │                               │
    状态: S = Σ φ(k_i)·v_i^T          状态: S = SGD 优化权重
            │                               │
    问题：记忆碰撞、无遗忘              解决：delta 精准更新 + 门控衰减
```

---

## 二、状态更新公式的演化

### Linear Transformer — 起点

```
S_t = S_{t-1} + φ(k_t) · v_t^T

o_t = φ(q_t)^T · S_t  /  (φ(q_t)^T · z_t)

其中：
  φ(x) = elu(x) + 1          ← kernel 特征映射，逼近 exp
  z_t  = z_{t-1} + φ(k_t)    ← 归一化项（额外状态）
```

- 每条记忆等价累加，没有选择性
- k 之间越相似，查出来的值越混在一起（记忆碰撞）
- 需要维护额外的归一化状态 z

### Gated DeltaNet — 终点

```
S_t = S_{t-1} · [α_t · (I - β_t · k_t · k_t^T)] + β_t · v_t · k_t^T

o_t = S_t · q_t    （不需要归一化分母）
```

- 同 key 的旧信息先被擦除再写入（delta 规则）
- 不同 key 的信息通过 α 门控衰减
- 不需要 kernel 函数，不需要归一化分母

---

## 三、核心理念的根本分歧

| | Linear Transformer | Gated DeltaNet |
|---|---|---|
| **数学本质** | softmax 的核近似 | 在线线性回归的 SGD |
| **理论根基** | Mercer 定理 / 核方法 | 在线凸优化 / delta rule (Widrow et al., 1960) |
| **状态 S 是什么** | key-value 关联的累加和 | 线性模型的权重矩阵 |
| **更新方式** | Hebbean 累加 `v·φ(k)^T` | Delta 修正（先擦旧值再写新值） |
| **对 key 冲突的处理** | 叠加（碰撞） | 替换（精准更新） |
| **遗忘机制** | 无 | α 门控（全局衰减）+ β 控制更新强度 |
| **特征映射** | 必须（ELU+1 逼近 softmax） | 不需要（仅 L2 归一化稳定数值） |
| **归一化分母** | 需要（z，额外状态） | 不需要 |

---

## 四、数学推导对比

### Linear Transformer 的逻辑链

```
softmax(Q·K^T) 无法分解为 Q 和 K 各自函数的乘积
         ↓
找 kernel 函数 φ，使得 φ(q)^T·φ(k) ≈ exp(q·k/√d)
         ↓
attention 写成:
  o_i = φ(q_i)^T · Σ_{j≤i} φ(k_j)·v_j^T  /  φ(q_i)^T · Σ_{j≤i} φ(k_j)
                  └──── S_i ────┘              └── z_i ──┘
         ↓
每步只需 O(d²) 更新 S 和 z，整体 O(N·d²)，线性于序列长度
```

### Gated DeltaNet 的逻辑链

```
不想逼近 softmax 了，换个问题：
  给定流式数据 (k_1,v_1), (k_2,v_2), ...
  在线维护一个线性模型 S，使得 S·k_t ≈ v_t
         ↓
损失函数: L(S) = ½ · ||S·k_t - v_t||²
         ↓
每步做一步 SGD:
  S_t = S_{t-1} - β_t·∇L(S_{t-1})
      = S_{t-1}·(I - β_t·k_t·k_t^T) + β_t·v_t·k_t^T
         ↓
再加门控 α_t（类似神经网络训练中的 weight decay）:
  S_t = α_t·S_{t-1}·(I - β_t·k_t·k_t^T) + β_t·v_t·k_t^T
```

### 统一到在线学习框架

| 方法 | 正则化项 | 损失函数 |
|------|---------|---------|
| Linear Transformer | `||S_t - S_{t-1}||²` | `<S_t·φ(k_t), v_t>` |
| Mamba2 | `||S_t - α_t·S_{t-1}||²` | `<S_t·k_t, v_t>` |
| DeltaNet | `||S_t - S_{t-1}||²` | `<S_t·k_t, β_t·(v_t - S_{t-1}·k_t)>` |
| **Gated DeltaNet** | `||S_t - α_t·S_{t-1}||²` | `<S_t·k_t, β_t·(v_t - α_t·S_{t-1}·k_t)>` |

---

## 五、代码对比（核心循环）

### Linear Transformer

```python
s = zeros(D, M)    # 状态矩阵
z = zeros(D, 1)    # 归一化项

for i in range(N):
    # 特征映射
    phi_k = elu(k_i) + 1
    phi_q = elu(q_i) + 1

    # 无条件累加（Hebbian 规则）
    s = s + phi_k * v_i^T
    z = z + phi_k

    # 需要除以归一化
    o_i = (phi_q^T @ s) / (phi_q^T @ z)
```

### Gated DeltaNet

```python
S = zeros(D, V)    # 状态矩阵，无归一化项

for i in range(N):
    # 步骤 1: 全局衰减
    S = S * exp(g_i)           # g_i = ln(α_i)

    # 步骤 2: delta 修正
    v_old = S @ k_i             # 查当前预测值
    v_new = beta_i * (v_i - v_old)  # 误差 × 学习率
    S = S + k_i * v_new^T       # 写入修正

    # 步骤 3: 输出（不需要分母）
    o_i = S @ q_i
```

---

## 六、为什么 Gated DeltaNet 不需要 kernel 函数

| 防护层 | Linear Transformer | Gated DeltaNet |
|-------|-------------------|----------------|
| 数值稳定 | kernel φ (ELU+1) + 分母归一化 | L2 norm (`||k||=1`) |
| 记忆爆炸 | 归一化分母 z | α 门控指数衰减 |
| 同 key 冲突 | 无（只能叠加） | delta 规则先擦后写 |
| softmax 近似 | 依赖 kernel 逼近质量 | 不需要（不是 softmax 的近似） |

Gated DeltaNet 的数值稳定性来自三个方面：
1. **L2 norm** — 防止 `v·k^T` 数值爆炸
2. **门控 α** — 历史信息自动指数衰减
3. **Delta 规则** — 修正而非累加，防止同 key 信息堆叠

---

## 七、性能与局限

### Linear Transformer 的优势

- 第一个把 Transformer 写成 RNN — **开创性**
- 推理极快：MNIST 317× 加速，CIFAR-10 4462× 加速
- 训练可并行（chunkwise 的前提）
- 常数内存推理（只需存 S 和 z）

### Linear Transformer 的局限

- **无遗忘机制** → 长序列记忆碰撞，检索退化
- **无选择性** → 所有 key-value 同等对待
- **kernel 是近似** → 极限性能不如原始 softmax attention
- **需要归一化分母 z** → 额外的状态开销和浮点除法

### Gated DeltaNet 如何解决这些问题

| 问题 | Linear Transformer | Gated DeltaNet |
|------|-------------------|----------------|
| 记忆碰撞 | 会（累加机制，S-NIAH-2 仅 14%） | 不会（delta 替换，S-NIAH-2 达 92%） |
| 遗忘 | 无 | α 门控，α→0 快速清空 |
| 选择性更新 | 无（所有 kv 等价） | 有（β 控制每步更新强度） |
| 特征映射 | 必须（φ=ELU+1） | 不需要（L2 norm 即可） |
| 1.3B 语言建模 | 未报告 | PPL 16.42，平均准确率 55.32 |
| 训练速度 | — | 与 Mamba2 接近（仅慢 2-3K tps） |

---

## 八、演化脉络

```
Linear Transformer (2020) ─── 开山之作，证明 attention 可写成 RNN
   │
   │ 问题: 无遗忘、无选择、记忆碰撞
   ▼
RetNet / GLA (2023-24) ─── 加入数据依赖的门控衰减
   │
   ▼
Mamba / Mamba2 (2023-24) ─── α_t·S + v·k^T，统一衰减
   │
   │ 问题: 衰减无差别，无法精准更新单条记忆
   ▼
DeltaNet (2024) ─── S·(I-β·k·k^T) + β·v·k^T，精准更新
   │
   │ 问题: 无批量遗忘，真实文本中记忆饱和
   ▼
Gated DeltaNet (2024) ─── α·S·(I-β·k·k^T) + β·v·k^T
                          门控 + delta 统一，当前最优
```

---

## 九、一句话总结

> **Linear Transformer (2020)** 证明了 softmax attention 可以写成 RNN，通过 kernel trick 把复杂度从 O(N²) 降到 O(N)。它是一条"逼近路线"——用 φ(q)·φ(k) 近似 exp(q·k)，本质还是在模仿 softmax attention。

> **Gated DeltaNet (2024)** 则完全放弃了逼近 softmax，转而把每个 token 当作一次在线学习的训练样本，用 SGD + weight decay 维护一个不断自我修正的记忆矩阵。它是一条"学习路线"——S 是一个被在线训练的线性模型权重，delta 规则负责精准更新，门控负责全局遗忘。

> 两者之间隔了四年的研究进展，从"如何更高效地算 attention"变成了"如何更好地管理记忆"。

---

## 十、状态容量的瓶颈与应对

### 瓶颈的本质

Gated DeltaNet 的状态 S 是 `dk × dv` 的矩阵，秩最多为 `min(dk, dv)`。这意味着它**最多只能存储约 dk 个相互正交的 key-value 对**，超出后必然碰撞：

```
假设 dk = 128, context = 100K tokens
→ 最多存 128 个"完全独立"的 key-value 关联
→ 第 129 个全新方向的 token 必然覆盖或混入旧信息
```

这是所有线性 RNN 逃不掉的固有瓶颈。

### 应对策略

#### 策略一：Delta 规则 — "覆盖"而非"新增"

Linear Transformer 是累加，相似 key 的 v 会叠加碰撞；Gated DeltaNet 的 delta 规则是**先擦再写**：

```
k1 ≈ k2（两个语义相似的 token）

Linear Attention:
  S = v1*k1^T + v2*k2^T
  查 k1: 得到 v1 + v2 的混合 → 碰撞！

Gated DeltaNet:
  S = S_old - β*(S_old*k2)*k2^T + β*v2*k2^T
  查 k2: 得到 v2（精准）→ 旧值被替换
```

**同一个"语义槽位"可以被无限次复用。** 反复讲同一个话题不消耗额外容量，只有讲全新话题（key 与之前所有 key 都正交）才需要新存储空间。实际语言中语义高度聚集——几万个 token 可能只涉及几十个核心概念。

#### 策略二：门控 α — "主动遗忘"

```
α_t → 1:  保留全部记忆（当前话题重要）
α_t → 0:  清空记忆（话题切换，立即腾出空间）
0 < α_t < 1: 渐变过渡
```

举例：从聊篮球切换到聊编程时，α→0 让篮球记忆整体衰减，空间回收给编程话题。

#### 策略三：混合架构 — "分工协作"

```
GatedDeltaNet-H1:  [GatedDeltaNet] → [SlidingWindow Attention] → ...
GatedDeltaNet-H2:  [Mamba2] → [GatedDeltaNet] → [SWA] → ...
```

分工逻辑：
- **滑动窗口 Attention (SWA)**：负责局部模式、精确匹配，不需要记忆，每次重算。分担了 GatedDeltaNet 的局部检索压力
- **GatedDeltaNet**：专注全局长期记忆，状态 S 只存跨远距离的关键信息

LongBench 上 GatedDeltaNet-H2 的 few-shot 得分 40.5 vs 纯 GatedDeltaNet 30.0 验证了这一点。

#### 策略四：多头 — "分工存储"

```python
num_heads = 9       # 1.3B 模型
dk_per_head = 128
dv_per_head = 128
# 总状态参数: 9 × 128 × 128 ≈ 147K
# 实际总容量 = 单头容量 × 头数
```

不同头可以分工：头 1-3 管语法，头 4-6 管实体，头 7-9 管语义。互相补充。

#### 策略五：多层叠加

```python
n_layer = 24    # 每层有自己独立的 S
# 总状态: 24 × 9 × 128 × 128 ≈ 3.5M 参数
# 高层可以存储更抽象的信息
```

### 为什么实际够用

```
理论最坏: 信息均匀分散在 100K 个正交方向 → 128 维不够

实际情况:
  1. 自然语言高度冗余，语义维度远小于词汇量
  2. 大部分 token 服务于少数主题
  3. delta 规则 → 同主题 token 复用 slots
  4. 门控 α → 旧主题被遗忘，slots 被回收
  5. 混合架构 → 局部匹配不消耗状态
  6. 多头 + 多层 → 总容量远大于单头 dk
```

论文实验验证：纯 GatedDeltaNet 在 LongBench 14 任务平均 16.6 vs Transformer++ 11.0，证明固定状态在实际长文本任务中足够。

### 未来的改进方向

| 方向 | 思路 | 代表工作 |
|------|------|---------|
| 更大状态 | 增加 dk/dv 或头数 | 各模型默认做法 |
| 混合架构 | 线性 RNN + attention | GatedDeltaNet-H1/H2, Samba, Griffin |
| 多层叠加 | 每层独立状态，层层抽象 | 标准 Transformer 做法 |
| 状态扩展 | 动态扩展状态维度 | HGRN2 (state expansion) |
| **向量门控** | dk 维门控替代标量 α，维度级选择性遗忘 | GLA, 论文提到的 future work |

最关键的是最后一行：当前 α 是标量（每个 head 一个值），如果换成每维度一个值的**向量门控**：

```
当前:  α ∈ (0,1)        → 全局统一衰减，整块白板一起淡化
未来:  α ∈ (0,1)^dk     → 每个 key 维度独立衰减
                         可以只忘掉某个"方向"的信息
                         其他方向的信息保留不动
```

这将让状态管理更精细，从根本上缓解固定容量的限制。论文在 Related Work 结尾（Section 5）明确提到了这一点作为 future work。
