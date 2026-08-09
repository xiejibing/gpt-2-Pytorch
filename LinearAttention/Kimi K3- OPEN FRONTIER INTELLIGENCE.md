# Kimi K3 — Model Architecture 详解

> 基于论文《Kimi K3: Open Frontier Intelligence》Technical Report 的架构分析笔记。

---

## 总览

Kimi K3 是一个 **2.78T 参数的 Mixture-of-Experts 模型**，激活参数 **104.2B**，原生支持视觉，上下文窗口 **1M tokens**。相比 Kimi K2，总体 scaling efficiency 提升约 **2.5×**。

架构设计的核心思想是沿着**三个维度**扩展信息流：

| 维度 | 机制 | 解决的问题 |
|------|------|-----------|
| **序列长度** (sequence) | **Hybrid Attention** = KDA (线性注意力) + Gated MLA (全局注意力) | 长上下文 token mixing 的效率与容量平衡 |
| **网络深度** (depth) | **Attention Residuals (AttnRes)** | 残差连接将深层信息压缩为单一向量 → 深度维度的信息瓶颈 |
| **模型宽度** (width) | **Stable LatentMoE** | 大量专家的稀疏激活，降低 channel mixing 的通信与计算开销 |

```
Input → MoonViT-V2 (视觉) / Token Embedding (文本)
  ↓
[Block × 23] 每个 Block 内含:
  ├── KDA Layer ×3         ← 线性注意力，O(N) 复杂度
  ├── Gated MLA Layer ×1   ← 全局 softmax 注意力
  ├── Stable LatentMoE     ← 2 shared experts + 16/896 routed experts
  └── Attention Residuals  ← 跨 block 的深度维度检索
  ↓
Gated MLA (最终额外 1 层)  ← 确保最后输出做全局 attention
  ↓
Output
```

总计：**69 层 KDA + 24 层 MLA = 93 层**

---

## 关键参数：Kimi K2 vs Kimi K3

| 参数 | Kimi K2 | Kimi K3 | 变化 |
|------|---------|---------|------|
| 架构 | MoE | MoE | – |
| 总层数 | 61 | **93** | ↑52% |
| 总参数量 | 1.04T | **2.78T** | ↑167% |
| 激活参数量 | 32.6B | **104.2B** | ↑220% |
| Hidden Dim | 7,168 | 7,168 | = |
| Latent MoE Dim | – | 3,584 (0.5×) | 新增 |
| MoE Hidden Dim per Expert | 2,048 | **3,072** | ↑50% |
| 路由专家数 | 384 | **896** | ↑133% |
| 每 Token 激活专家 | 8 | **16** | ↑100% |
| 共享专家 | 1 | **2** | ↑100% |
| 注意力头数 | 64 | **96** | ↑50% |
| Dense 层数 | 1 | 1 | = |
| 词表大小 | 160K | 160K | = |
| 训练上下文长度 | 128K | **1M** | 8× |
| 注意力机制 | MLA | **Hybrid KDA–MLA** | – |
| 注意力层组成 | 61 MLA | 69 KDA + 24 MLA | – |
| 激活函数 | SwiGLU | **SiTU-GLU** | – |
| MTP 层 | 1 | 1 | = |
| ViT 总参数 | – | 401M | 新增 |
| ViT 层数 | – | 27 | 新增 |
| ViT Patch Size | – | 14 | 新增 |
| ViT 注意力头数 | – | 12 | 新增 |

---

## 1. Hybrid Attention（混合注意力）

### 1.1 整体设计

每个 block 内按 **3:1 比例**混合两种注意力：

```
┌──────────────────────────────────────────┐
│  KDA Layer 1   ← 线性注意力，O(N)，位置感知  │
│  KDA Layer 2   ← 线性注意力，recency 偏向    │
│  KDA Layer 3   ← 线性注意力，高效长程 mixing  │
│  Gated MLA     ← 全局 softmax，高容量交互     │  ← 每 4 层出现 1 次
└──────────────────────────────────────────┘
```

- **Backbone 末尾额外 1 层 Gated MLA**：确保最终输出前做一次完整的全局交互
- **3:1 比例是工程权衡**：太少 MLA → 全局交互不足；太多 MLA → 长序列推理变慢

### 1.2 Kimi Delta Attention (KDA) — 线性注意力部分

KDA 扩展了 delta-rule recurrence，加入了 **通道级遗忘门 (channel-wise forget gate)**。

#### 单头 recurrent 形式

```
S_t = (I - β_t · k_t · k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T
õ_t = S_t^T · q_t

其中:
  α_t ∈ (0,1)^{dk}   ← 通道级逐步保留因子 (channel-wise retention factor)
  β_t ∈ (0,1)        ← delta-rule 写入强度
  S_t ∈ R^{dk×dv}    ← 循环状态矩阵
```

#### Per-head 参数化

每个 head 的 q, k, v, β, α 都通过输入依赖的方式计算：

```
q_t^h, k_t^h = L2Norm(Swish(ShortConv(W_{q/k}^h · x_t)))     ∈ R^{dk}
v_t^h        = Swish(ShortConv(W_v^h · x_t))                  ∈ R^{dv}
β_t^h        = Sigmoid(W_β^h · x_t)                           ∈ (0,1)
z_t^h        = W_α^↑ · W_α^↓ · x_t + b_α^h                   ∈ R^{dk}    (logit of decay)
```

- **ShortConv + Swish**：在 q/k/v 投影前加轻量时序卷积 + 非线性，捕获局部上下文
- **L2Norm on q, k**：防止点积数值过大
- **低秩投影** `W_α^↑·W_α^↓` 生成精细的逐通道 decay logit

#### Chunkwise 并行形式

KDA 在实际计算中采用 **chunk 间循环、chunk 内并行** 的混合策略：

- 将序列分成大小为 C 的 chunk
- chunk 之间：用矩阵 `S_{[t]}` 传递循环状态
- chunk 内部：所有位置并行计算

```
定义:
  γ_{i→j}^{[t]} = Π_{r=i}^{j} α_r^{[t]}          ← 通道级累积衰减

A^{[t]} = Tril((Q^{[t]} ⊙ Γ_{1→C}^{[t]}) · (K^{[t]} / Γ_{1→C}^{[t]})^T)   ← intra-chunk attention

O^{[t]} = (Γ_{1→C}^{[t]} ⊙ Q^{[t]}) · S_{[t]}    ← inter-chunk (历史状态)
        + A^{[t]} · ṽ^{[t]}                         ← intra-chunk (chunk 内交互)
```

其中 `Tril` 保留矩阵的下三角（含对角线），保证因果性——对角线保留是因为每个输出读取的是**当前 token 更新之后**的状态。

#### Kimi K3 对 KDA 的两项改进

**改进 1 — Lower-bounded decay**

Kimi Linear 使用负 Softplus 映射 log-decay：
```
g_t^h = -e^{A^h} · Softplus(z_t^h) ∈ (-∞, 0)^{dk}
```
这个无下界的映射导致对角 tile 的 reciprocal rescaling 可能溢出 BF16 范围，必须用显式的 position-pair 逐对计算，成为 chunk 内计算的主要瓶颈。

Kimi K3 改用 scaled sigmoid，给 log-decay 加了下界：
```
g_t^h = g_min · Sigmoid(e^{A^h} · z_t^h) ∈ (g_min, 0)^{dk}
α_t^h = exp(g_t^h) ∈ (e^{g_min}, 1)^{dk}

其中 g_min = -5（固定），A^h 为可学习的 per-head log-scale（初始化为 0）
```

当 `g_min = -5` 时，每个保留因子 `α > e^{-5} ≈ 6.7×10^{-3}`，一个 16-token tile 的累积 log-decay 在 `(-80, 0)` 范围内，对应的 reciprocal rescaling 因子小于 `e^{80}` → **BF16 可表示**。

**效果**：对角 tile 也可以使用密集 Tensor Core 矩阵乘法，消除了 Kimi Linear 中最慢的 position-pair 逐对计算路径。

```
Kimi Linear:
  对角 tile → 显式 position-pair 计算 (慢)
  非对角 tile → Tensor Core 矩阵乘法 (快)

Kimi K3:
  所有 tile → Tensor Core 矩阵乘法 (全快)
```

**改进 2 — Full-rank output gate**

Kimi Linear 使用低秩参数化的输出门控，Kimi K3 改为输入依赖的全秩投影：

```
y_t = W_o [Sigmoid(W_g · x_t) ⊙ RMSNorm(õ_t)]

其中 W_g: R^d → R^d   (full-rank)
```

这与 Gated MLA 的门控设计保持一致。

### 1.3 Gated MLA — 全局注意力部分

详见 [第 2 节](#2-gated-mla-详解)。

---

## 2. Gated MLA 详解

### 2.1 MLA 基础：KV Cache 压缩

标准 Multi-Head Attention (MHA) 的推理瓶颈在于 KV Cache：

```
KV Cache per token = 2 × n_heads × d_head × n_layers
```

对于 Kimi K3（96 heads, 93 layers, 1M context），MHA 的 KV Cache 将达 TB 级别，不可行。

MLA (Multi-head Latent Attention) 的核心思想：**将 Key 和 Value 压缩到一个低维潜在向量中缓存，推理时通过 learned up-projection 实时重建**。

```
标准 MHA:
  缓存完整的 K, V → 每个 head 独立存储 → 内存巨大

MLA:
  输入 x_t → 下投影 W_c → c_t (低维 latent) → 只缓存 c_t
           推理时: k_t = W_UK · c_t,  v_t = W_UV · c_t  ← 实时重建
```

KV Cache 从 `2 × n_heads × d_head` 降到 `d_c`（通常 `d_c ≈ d/4 ~ d/8`），压缩比 **4~16 倍**。

### 2.2 Kimi K3 对 MLA 的三项改进

#### 改进 1 — NoPE（去掉位置编码）

| 模型 | 位置编码 |
|------|---------|
| DeepSeek-V2 | RoPE（解耦旋转位置编码） |
| Kimi K2 | RoPE |
| **Kimi K3** | **NoPE（无显式位置编码）** |

**为什么可以去掉？** 这是 Hybrid Attention 设计的精髓：

- KDA 层通过 recurrent decay 机制隐式编码了位置和时序信息
- MLA 层只需要专注**无约束的全局内容交互**
- 避免扩展 context length 时需要调整 RoPE base frequency 或做 YaRN 插值

> 本质是一种**分工**：KDA 负责"时序/位置敏感"的序列混合，MLA 负责"位置无关"的全局内容聚合。

#### 改进 2 — Full-Rank Output Gate（Gated 的来源）

```
标准 MLA:
    õ_t = Attention(q_t, k_t, v_t)      ← 原始 attention 输出
    y_t = W_o · õ_t                      ← 直接线性输出投影

Gated MLA:
    õ_t = Attention(q_t, k_t, v_t)      ← 原始 attention 输出
    y_t = W_o · [Sigmoid(W_g · x_t) ⊙ õ_t]  ← 先门控，再输出投影
                                                ↑
                                          full-rank data-dependent gate
```

- `W_g` 是 `d × d` 的**全秩矩阵**（非低秩分解），每个 channel 有独立的门控参数
- `Sigmoid` 将门控值压缩到 `(0, 1)`，让模型学会"对当前 token，哪些通道的全局 attention 信息更有用"
- 这与 KDA 的 full-rank output gate（Eq.6）设计一致，两种注意力共享相同的门控参数化

#### 改进 3 — FP32 训练精度

FlashAttention 在低精度（BF16）下存在**有偏舍入误差**（biased rounding error），MLA 因涉及 latent 空间压缩/重建，对此误差更敏感。

- attention 输出 tile 在训练时保留 **FP32** 精度
- 代价是片上内存占用翻倍 → 重新设计 training kernel：
  - 将输出 tile 与 KV staging buffer 重叠（而非与 query tile 重叠）
  - 释放共享内存给更深的 KV pipeline
  - 保持高训练吞吐

### 2.3 Gated MLA 在 Hybrid Attention 中的角色

```
                         │  复杂度   │  位置信息  │  交互范围  │
KDA (69 层)              │  O(N)   │  隐式编码  │  长程偏向  │   ← 主力，处理长序列
Gated MLA (24 层)        │  O(N²)  │  NoPE     │  全局     │   ← 周期性"校准"，高容量交互
```

KDA 覆盖多数层保证效率，MLA 周期性插入保证容量，末尾强制 MLA 保证最终输出的全局性。

---

## 3. Attention Residuals（注意力残差）

### 3.1 核心动机：深度维度的信息瓶颈

Attention Residuals 的出发点是一个精妙的类比：

> RNN 时代：信息沿**时间步**被压缩进单一的 hidden state → **时间维度的瓶颈**
>
> Transformer 的解决方案：用 Attention 替代循环，让每个位置**选择性关注**所有前序位置 → 打破了时间瓶颈
>
> 但 Transformer 的**深度方向**：残差连接仍然把所有前序层压缩进一个向量 → **深度维度的瓶颈**依然存在！

**Attention Residuals 的洞见**：

> 在序列维度上，我们用 Attention 替代了 recurrence；
> 那为什么不**同样用 Attention 来替代深度方向的 uniform accumulation**？

```
标准残差连接:
  h_L = x + f_1 + f_2 + ... + f_L      ← 所有层权重 = 1，完全均匀累加
                                          深层的信息被浅层的信息"淹没"

Attention Residuals:
  h_L = α_0·x + α_1·f_1 + α_2·f_2 + ... + α_{L-1}·f_{L-1}
                                        ← 每层学习自己的 retrieval weights
                                        ← 不同层关注不同前序层
```

### 3.2 Full Attention Residuals

#### 机制

把深度方向当作一个"序列"，每一层是一个"token"：

```
对每一层 l，定义:

  Pseudo-Query:  q_l = w_l ∈ R^d          ← 每层一个可学习的检索偏好向量
  
  Keys & Values: k_i = v_i = { h_1              i = 0 (token embedding)
                              { f_i(h_i)        1 ≤ i ≤ l-1 (各层输出)
  
  Attention weights (softmax kernel):
    score_i = q_l^T · RMSNorm(k_i)        ← 点积 + RMSNorm on keys
    α_{i→l} = exp(score_i) / Σ_{j=0}^{l-1} exp(score_j)

  该层的最终输入:
    h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

#### 关键设计细节

**① 可学习的 Pseudo-Query `w_l`**

- 每层一个独立的可学习向量 `w_l ∈ R^d`
- **不依赖输入 token**，纯粹是一个层特定的检索偏好
- 在训练中自动学会"第 l 层应该关注哪些前序层"
- 同一层内所有 token 共享相同的深度注意力权重（token-agnostic）
- 不同层可以形成不同的检索模式：浅层可能关注 embedding，中层关注前面的 block，深层均匀分布等

**② RMSNorm on Keys**

```
α ∝ exp(q_l^T · RMSNorm(k_i))
```

- 防止某些前序层输出 magnitude 过大而在 softmax 中指数级膨胀
- 将所有层的表示拉到同一尺度 → 保证 softmax 竞争公平

**③ Embedding 作为第 0 个 source**

```
k_0 = v_0 = h_1   (token embedding)
```

- 始终保留 token embedding 作为可检索的 source
- 让深层也能直接访问最原始的 token 信息，不被中间层变换所稀释

#### 复杂度

```
计算量: O(L² · d)     L=93, d=7168 → ~62M 操作，相对每层 self-attention 的 O(L·d²) 约 1.3% 额外开销
内存:   O(L · d)      需要保持所有 L 层输出，以及流水线并行下的跨 stage 通信
```

- 计算量完全可接受（网络深度不大，L < 100）
- 真正的瓶颈是内存/通信：需要保持所有 L 层输出

### 3.3 Block Attention Residuals（实用版本）

为了解决 `O(Ld)` 的内存/通信开销，将层划分为 block，在 block 级别做 attention。

#### 结构

```
完整 93 层 → 8 个 block（每 block ~12 层）+ 1 个部分 block
加上 embedding → 共 9 个 source

  Block 1 (Layers 1-12)
  │  ├── 层内: 标准残差 + full-rank 层计算
  │  ├── 层间: attention over [b_0, running partial sum]
  │  └── 压缩: b_1 = Σ_{j∈B₁} f_j(h_j)          ← 12 层的输出压为 1 个向量
  │
  Block 2 (Layers 13-24)
  │  ├── 层内: 标准残差 + full-rank 层计算
  │  ├── 层间: attention over [b_0, b_1, running partial sum]
  │  └── 压缩: b_2 = Σ_{j∈B₂} f_j(h_j)
  │
  ...
  │
  Block 8 + 部分 Block 9
     └── 最终层: attention over 所有 9 个 block 表示
```

#### 两种粒度的表示

```
粒度 1 — Block 完整压缩表示（跨 block 用）:
  b_n = Σ_{j∈Block_n} f_j(h_j)         ← block 内所有层输出的求和压缩

粒度 2 — Block 内部分和（block 内用）:
  b_i^n = Σ_{j=1}^{i} f_j(h_j)         ← block n 内前 i 层输出的 running sum

特殊项:
  b_0 = h_1                            ← token embedding，始终可访问
```

**关键区别**：`b_n` 要等整个 block 跑完才能得到；`b_i^n` 是 running sum，block 内逐层累加。

#### V 矩阵构造规则

对于 Block n 的第 i 层：

```
if i == 1 (block 首层):
  V = [b_0, b_1, b_2, ..., b_{n-1}]          ← embedding + 之前所有已完成的 block 压缩

if i >= 2 (block 内后续层):
  V = [b_0, b_1, ..., b_{n-1}, b_{i-1}^n]   ← 同上 + 本 block 前 i-1 层的部分和
```

**注意首层的特殊性**：Block 首层只能看到之前已完成的 block（本 block 还没有产出），所以 V 中不包含本 block 的部分和。

#### Attention 计算方式（所有 source 共享）

```
对 V 中的每个 source j:

  q_l = w_l                                ← 层 l 的可学习 pseudo-query
  score_j = w_l^T · RMSNorm(v_j)           ← 对每个 source 独立打分
  α_j = softmax(scores)_j                  ← 归一化
  h_l = Σ_{j∈V} α_j · v_j                 ← 加权求和 → 作为该层的输入
```

然后正常处理：`f_l(h_l)` = self-attention + MoE-FFN 的完整层计算。

#### 具体例子：完整追踪 Block 间 Attention

假设简化模型：3 个 block，每 block 3 层，共 9 层。

##### Block 1 — Layer 1（block 首层）

```
此时已完成: 无（只有 embedding）
V = [b_0]                                  ← 只有 1 个 source

h_1 = α_{0→1} · b_0 = b_0                 ← 唯一 source，权重实际为 1
f_1(h_1) → b_1^1 = f_1(h_1)               ← running sum 初始化
```

##### Block 1 — Layer 2（block 内后续层）

```
已完成: embedding + layer 1
V = [b_0, b_1^1]                           ← 2 个 source

score_0 = w_2^T · RMSNorm(b_0)            ← 对 embedding 打分
score_1 = w_2^T · RMSNorm(b_1^1)          ← 对 layer 1 输出打分
α = softmax([score_0, score_1])

h_2 = α_0 · b_0 + α_1 · f_1(h_1)          ← 可以"拉取"原始 embedding 或 layer 1 的输出
f_2(h_2) → b_2^1 = b_1^1 + f_2(h_2)
```

##### Block 1 — Layer 3（block 最后层）

```
V = [b_0, b_2^1]                           ← embedding + layer1~2 的部分和

score_0 = w_3^T · RMSNorm(b_0)
score_1 = w_3^T · RMSNorm(f_1 + f_2)      ← 对前两层的聚合信息打分
α = softmax([score_0, score_1])

h_3 = α_0 · h_1 + α_1 · (f_1(h_1) + f_2(h_2))

f_3(h_3) → b_3^1 = b_2^1 + f_3(h_3) = f_1 + f_2 + f_3

Block 1 完成 → 压缩为: b_1 = f_1 + f_2 + f_3
```

##### Block 2 — Layer 4（跨 block 首层）⭐关键

```
此时已完成: embedding + Block 1（完整压缩）
V = [b_0, b_1]                             ← 只有 2 个 source

b_1 = f_1 + f_2 + f_3                     ← 三层输出被求和压缩为一个向量

score_0 = w_4^T · RMSNorm(b_0)
score_1 = w_4^T · RMSNorm(b_1)            ← 对整个 Block 1 整体打分
α = softmax([score_0, score_1])

h_4 = α_0 · b_0 + α_1 · (f_1 + f_2 + f_3)  ← 不能单独选择 Block 1 中的某一层
                                              只能对整个 block 的整体信息做选择
f_4(h_4) → b_1^2 = f_4(h_4)
```

##### Block 2 — Layer 5（block 内后续层）

```
V = [b_0, b_1, b_1^2]                      ← 3 个 source

b_1^2 = f_4(h_4)                           ← Block 2 第 1 层的输出

score_0 = w_5^T · RMSNorm(b_0)            ← embedding
score_1 = w_5^T · RMSNorm(b_1)            ← Block 1 整体
score_2 = w_5^T · RMSNorm(f_4)            ← Block 2 内部前面的层
α = softmax([score_0, score_1, score_2])

h_5 = α_0·b_0 + α_1·b_1 + α_2·f_4(h_4)

f_5(h_5) → b_2^2 = b_1^2 + f_5(h_5) = f_4 + f_5
```

##### Block 2 — Layer 6

```
V = [b_0, b_1, b_2^2]                      ← partial sum 更新为 f_4 + f_5

Block 2 完成 → b_2 = f_4 + f_5 + f_6
```

##### Block 3 — Layer 7（跨两个 block）

```
V = [b_0, b_1, b_2]                        ← embedding + Block1 + Block2
                                              4 个 source: embedding + 2 个 block + 可能的部分和

h_7 = α_0·b_0 + α_1·b_1 + α_2·b_2

... 逐层推进到 Layer 9
Block 3 完成 → b_3 = f_7 + f_8 + f_9
```

##### 最终输出层

```
V = [b_0, b_1, b_2, b_3]                  ← 全部 3 个 block + embedding = 4 个 source

h_final = Σ_{n=0}^{3} α_n · b_n           ← 全局加权聚合
                                          重新审视所有 block 的贡献
```

在 Kimi K3 中，V = 9 个 source（8 个 block + embedding），最终层在此之上做深度维度的全局信息整合。

#### 复杂度改进

```
Full AttnRes:    内存 O(Ld)  → 93 个层输出需要保持
Block AttnRes:   内存 O(Nd)  → 只需 ~8 个 block 压缩表示

Kimi K3: L=93, N=8, 共 9 个 source
内存减少: 93 → 9，约 10×
```

实验表明 N≈8 就能恢复 Full AttnRes 的大部分收益。

#### Block 内 partial sum 的设计动机

为什么 block 内用 partial sum 而非每层独立？

```
方案 A（每层独立）:
  Layer 6 的 V = [b_0, b_1, b_2, f_4, f_5]     ← 需要存 block 内每层独立输出
  → memory 增加

方案 B（partial sum，实际采用）:
  Layer 6 的 V = [b_0, b_1, b_2, f_4+f_5]       ← 只需存 running sum → 一个向量
  → memory 最小，且保持了因果性
```

#### 推理优化

Block 结构还带来了推理时的优化机会：

- **跨 block attention** 可并行计算（block 表示是提前算好的）
- **block 内部分和** 是顺序的（随层推进逐步累加）
- 通过 **online softmax** 将两者流式合并 → 显著降低推理延迟

> "this block structure also bounds the inference-time state, enabling the parallel inter-block results to be better merged with the sequential intra-block partial sums via online softmax"

### 3.4 与标准残差的对比

| 方面 | 标准残差 | Attention Residuals |
|------|---------|-------------------|
| 信息聚合方式 | 等权重累加（所有层权重=1） | 可学习的 selective attention |
| 梯度流动 | 等权重反向传播 | 高权重路径梯度更强 |
| 表示幅度 | 随深度无界增长 | 加权平均，幅度受控 |
| 层间关系 | 仅相邻层有直接路径 | 任意两层可交互 |
| source 数量 | 1（前一层输出） | 多个（embedding + 所有 block + 部分和） |
| 可解释性 | 无 | 每层的 attention pattern 可视化 |

### 3.5 为什么这很重要

1. **打破深度信息瓶颈**：深层不再只能通过前一层的残差压缩来间接获取浅层信息，可以直接"伸手"去 embedding 或任何中间层

2. **改善训练动态**：
   - 输出幅度在深度上保持有界（不会随层数增长发散）
   - 梯度范数在各层间分布更均匀（浅层不再梯度消失）

3. **推理效率**：Block 设计将运行时状态从 O(Ld) 降到 O(Nd)

4. **实证收益**：在原论文实验中，推理密集型任务（GPQA +7.5, Math +3.6, HumanEval +3.1）收益最大——这正是深度信息瓶颈最制约的能力类型

---

## 4. Stable LatentMoE

### 4.0 问题设定：为什么要 "Stable"？

Kimi K3 做了一件非常激进的事：把路由专家扩展到 **896 个**，每 token 激活 **16 个**，稀疏比高达 **56:1**（896/16）。

传统 MoE 的问题是每个选中的 expert 接收完整的 `d=7168` 维 token 表示 → 通信量和 expert 参数流量随 routing multiplicity `k` 线性增长。**LatentMoE** [32] 的核心思路是把"通用变换"和"专业化变换"分离：

- **共享专家（Shared Experts）**：保留全维度路径 `d=7168`，处理所有 token 都需要的通用变换，始终激活
- **路由专家（Routed Experts）**：在压缩的潜在空间 `ℓ = d/2 = 3584` 中操作，处理专业化变换，稀疏激活

这大幅降低了通信和参数开销，使得极端规模的 expert 扩展成为可能。但在 **2.78T 参数、896 experts** 的极端配置下，会暴露**两个失败模式**：

| 失败模式 | 原因 | 后果 |
|----------|------|------|
| **激活爆炸** | 路由路径由 `W↓ → experts → W↑` 将近四层连续的矩阵乘法链式组成，条件数差，在 2.78T 规模下产生爆炸性内部激活 | 低精度（BF16）下溢出，训练崩溃 |
| **负载失衡** | 896 个 expert 远超现有 auxiliary-loss-free 方法的调节能力，固定步长更新 `b += γ·sign(ℓ̄−ℓ_j)` 在收敛速度 vs 负载震荡间无法兼得 | Expert 并行效率低，部分 expert 训练不足（濒死 expert） |

**Stable LatentMoE 用三个组件分别解决这两个问题**：

| 组件 | 解决哪个失败模式 | 机制 |
|------|-----------------|------|
| Normalized LatentMoE | 激活爆炸 + 训练不稳定 | RMSNorm 稳定路由分支 scale |
| SiTU-GLU | 激活爆炸 | 有界激活函数，\|f\| ≤ 100 |
| Quantile Balancing | 负载失衡 | 分位数一步到位，无超参 |

---

### 4.1 架构概览

```
输入 x ∈ R^d  (d = 7168)
│
├── 共享路径（始终激活，N_s = 2）:
│   ├── E_1^shared(x)  ──→  y_1 ∈ R^d      ← 全维度 expert，处理通用变换
│   └── E_2^shared(x)  ──→  y_2 ∈ R^d      ← 全维度 expert
│
├── 路由路径（稀疏激活，16/896）:
│   │
│   │  Step 1 — 下投影到潜在空间:
│   │  z = W_↓ · x                          ∈ R^ℓ   (ℓ = 3584 = d/2)
│   │
│   │  Step 2 — Router 计算:
│   │  s = Sigmoid(W_r · x)                 ∈ R^n   (n = 896, per-expert scores)
│   │
│   │  Step 3 — Biased Top-k 选择:
│   │  T = argtopk(s + b)                   ∈ [n]^k (k = 16, b 只用于选择)
│   │  p_i = s_i / Σ_{r∈T} s_r                      (权重用原始 score，b 不参与)
│   │
│   │  Step 4 — Expert 计算（在潜在空间）:
│   │  u = Σ_{i∈T} p_i · E_i^routed(z)      ∈ R^ℓ   (加权聚合)
│   │
│   │  Step 5 — 归一化 + 上投影（Kimi K3 新增）:
│   │  u_norm = RMSNorm(u)                  ∈ R^ℓ   ← 创新 1: Normalized
│   │  y_routed = W_↑ · u_norm              ∈ R^d
│
└── 合并:
    y = Σ_{j=1}^{2} E_j^shared(x)  +  W_↑ · RMSNorm( Σ_{i∈T} p_i · E_i^routed(W_↓·x) )
        └──── 共享路径 ────┘         └────────── 路由路径 ──────────────────┘
```

**关键设计细节**：

- `W_↓ : R^d → R^ℓ` 和 `W_↑ : R^ℓ → R^d` 是**所有 expert 共享**的投影矩阵，不随 expert 变化
- Router bias `b` 只用于 Top-k 的**选择**（dispatch），不出现在聚合权重 `p_i` 中 → 调节负载但不影响梯度和 mixture weights
- 每个 expert `E_i^routed` 在潜在空间 `R^ℓ` 内做 FFN 变换（输入和输出都是 `ℓ` 维）
- 论文中使用的具体符号：`N_s = 2`（每层 2 个共享 expert），`ℓ = d/2 = 3584`

---

### 4.2 创新 1 — Normalized LatentMoE

#### 动机

原版 LatentMoE 直接将 `W_↑` 作用于聚合后的路由表示 `u`，但 `u` 的尺度在不同 token 之间波动很大——取决于选中了哪些 expert、各自的 routing weight 多大。这种尺度方差直接传递给 `W_↑` 的输出，再与全维度的共享分支输出相加时产生**数值不匹配**。

更本质地说，路由路径是一个四层链式结构 `W_↓ → Expert_i → Aggregate → W_↑`，在 2.78T 参数规模下这个链的条件数很差，小的输入扰动会放大为大的输出波动。

#### 改动

在 expert 聚合和 up-projection 之间插入 **RMSNorm**：

```
原版 LatentMoE:
    u = Σ p_i · E_i(z)    →    W_↑ · u        ← u 尺度随 expert 选择剧烈变化

Kimi K3 (Normalized):
    u = Σ p_i · E_i(z)    →    RMSNorm(u)    →    W_↑ · RMSNorm(u)
                               ↑
                         归一化到单位方差，消除 scale 波动
```

在第 4.1 节的公式中，这体现为：

```
y = Σ E_j^shared(x) + W_↑ · RMSNorm(u)      (Eq. 11)
```

而非 `y = Σ E_j^shared(x) + W_↑ · u`。

#### 效果

- 降低路由分支对 scale 变化的敏感度，使 `W_↑` 的输入始终在良好范围内
- 减少全维度共享分支和压缩路由分支合并时的数值不匹配 → 训练更稳定
- 不仅稳定训练，还**持续改善** validation loss 和下游 benchmark（说明这不是纯数值技巧，而是让优化更容易找到更好的解）

---

### 4.3 创新 2 — SiTU-GLU（Sigmoid Tanh Unit GLU）

#### 4.3.1 动机：SwiGLU 的无界性与稀疏放大的溢出风险

SwiGLU [107] 是当前 LLM 最主流的 FFN 激活函数：

```
SwiGLU(x) = (x ⊙ Sigmoid(x))  ⊙  W_u·x
             └──── gate ────┘     └─ up ─┘
```

**问题**：gate 分支的 `x·σ(x)`（Swish）无上界，up 分支的 `W_u·x` 也无上界。两个无界的因子相乘 → 偶尔出现极大的 activation outlier → BF16 下溢出风险。

在常规 dense 模型或稀疏度较低的 MoE 中，这个问题可通过梯度缩放等技巧缓解。但在 Kimi K3 的 **56:1 极端稀疏**下，问题被放大：
- 大值坐标恰好落到被选中的少数 expert 的概率更高
- 每个 expert 处理的 token 更多 → 统计上更容易遇到 outlier 组合
- 一旦溢出，稀疏路由使得误差难以被其他 expert 补偿

原始的 GLU [26] 用 `Sigmoid` 做 gate ∈ (0,1)（有界），但丢失了 Swish 在正半轴近似线性的特性——这被广泛认为是 SwiGLU 经验效果好的关键。

#### 4.3.2 SiTU-GLU 的数学定义

SiTU-GLU 的核心思想是：**用 smooth cap（`β·tanh(x/β)`）独立约束 gate 和 up 两个分支**，同时保留 SwiGLU 的局部响应形状：

```
SiTU-GLU(x) = [ β₁·tanh(W_g·x / β₁)  ⊙  Sigmoid(W_g·x) ]    ← gate 分支
             ⊙ [ β₂·tanh(W_u·x / β₂) ]                        ← up 分支

其中 β₁ = 4 (gate cap),  β₂ = 25 (up cap)
```

**标量形式**对比（设输入标量为 `z`，忽略权重矩阵以理解形状）：

| 函数 | Gate 分支 `g(z)` | Up 分支 `u(z)` | 全局有界？ |
|------|-----------------|---------------|-----------|
| GLU | `σ(z)` ∈ (0,1) | `z` | gate 有界，up 无界 |
| SwiGLU | `z·σ(z)` ∈ (0,∞) | `z` | **两者无界** |
| **SiTU-GLU** | `β₁·tanh(z/β₁)·σ(z)` | `β₂·tanh(z/β₂)` | **\|f\| ≤ β₁β₂ = 100** |

#### 4.3.3 三个关键性质

**性质 1 — 局部一阶近似 SwiGLU**

`β·tanh(z/β)` 在原点附近做 Taylor 展开：

```
β·tanh(z/β) = z + O(z³/β²)          (Eq. 18)
```

因此 SiTU-GLU 在原点附近 **一阶精确匹配 SwiGLU**：
```
SiTU-GLU(z) ≈ z·σ(z) · z = SwiGLU(z)    (当 z → 0)
```

这保留了 SwiGLU 在原点附近的良好局部响应特性（近似线性，这被认为是 Swish/SwiGLU 成功的关键）。

**性质 2 — 严格有界输出**

```
|SiTU-GLU(x)| ≤ β₁ · β₂ = 4 × 25 = 100      (Eq. 19)
```

任何输出坐标的绝对值不超过 100。在 BF16 下完全安全（BF16 max ≈ 3.39×10³⁸），从根本上消除了 activation outlier 导致溢出的可能。

作为对比，`β₁, β₂ → ∞` 时 SiTU-GLU 逐点恢复为 SwiGLU，因此 SwiGLU 可以看作是 SiTU-GLU 的无界极限。

**性质 3 — 非零梯度（vs Hard Clamping）**

与硬截断（`clamp(x, -c, c)`，饱和边界梯度 = 0）不同，`tanh` 的 smooth cap 处处保持非零梯度：

```
d/dz [β·tanh(z/β)] = sech²(z/β) > 0    (处处为正，虽渐近于 0)
```

硬截断在边界处梯度完全消失 → 被截断的坐标无法通过梯度恢复。Smooth cap 避免了这个问题，论文明确指出训练行为更好。

#### 4.3.4 β₁=4, β₂=25 的设计考量

两个分支的 cap 值不同，反映了对两者角色和量级的不同理解：

- **Gate 分支** (`β₁=4`)：已经是 `tanh · sigmoid` 的复合，sigmoid 本身有饱和效应 `→(0,1)`，所以实际 gate 值天然在 `(-4, 4)` 附近，cap 设得紧一些
- **Up 分支** (`β₂=25`)：只有 `tanh` 一个约束，需要更大的动态范围来保留表达力，cap 设得松一些
- 总 bound `4×25=100` 在 BF16 范围内绝对安全

```
局部行为示意（x ∈ [-10, 100]）:

SwiGLU:      原点附近 ~x²，远处 → 无限增长 ↗↗
SiTU-GLU:    原点附近 ~x²（匹配 SwiGLU），远处 → 渐近于水平线 y=100
              ↓
         保留了 SwiGLU 的好特性，但永远不会爆炸
```

#### 4.3.5 为什么两个分支都要 cap？

一个自然的问题是：只 cap 一个分支（比如只 cap gate 回到 GLU 的 `σ(x)`）是否就够了？

答案是**不够**。在 56:1 的极端稀疏下，up 分支的 `W_u·x` 本身就可以非常大（不受 gate 约束），即使 gate ∈ (0,1)，product 仍可能溢出。独立 cap 两个分支确保 **product 的每一个因子都有界**，不存在单点故障。

---

### 4.4 创新 3 — Quantile Balancing (QB)

这是三个创新中数学最深的一个。

#### 4.4.1 背景：Auxiliary-Loss-Free Routing 的困境

Kimi K3 采用 **auxiliary-loss-free routing** [30]：不添加辅助负载均衡损失（避免干扰主语言建模目标），纯粹通过 expert-specific bias `b_j` 调节路由决策。

```
路由规则:
    T_i = argtopk(s_i + b)            ← biased scores 决定 dispatch
    p_{i,j} = s_{i,j} / Σ_{r∈T_i} s_{i,r}   ← 原始 score 归一化为权重（b 不参与）
```

**关键设计**：bias `b` 只影响 dispatch（哪些 expert 被选中），不影响 mixture weights `p_i`。这保证了负载调节不会扭曲梯度信号。

**原方法的固定步长更新** [30]：

```
b_j^{(t+1)} = b_j^{(t)} + γ · sign(ℓ̄ - ℓ_j^{(t)})

其中 ℓ̄ = 目标负载,  ℓ_j = 实际负载
```

这个更新有一个根本性的两难：

```
γ 太小 → 收敛慢，很多步才能追上负载变化
γ 太大 → 过冲 + 震荡，负载在目标值附近来回跳跃
```

在 896 个 expert、16 激活/token 的规模下，这个 tradeoff 变得更加严重：
- Expert 池越大，每个 expert 的目标负载越小 → 相对波动更大
- 极端稀疏下，小的 bias 变化会导致大的负载变化 → 固定 γ 不够灵活

#### 4.4.2 QB 的核心思想

**直接从 router score 的分位数一步算出每个 expert 应该有的 bias**，不需要学习率，不需要逐步调节。

QB 把负载均衡形式化为一个**优化问题**：

> 给定 m 个 token、n 个 expert、每个 token 选 k 个 expert，如何分配使得：
> - 每个 expert 恰好服务 `q = mk/n` 个 token（完美均衡）
> - 总 score `Σ x_{i,j} · s_{i,j}` 最大化（选择最高质量的 expert）

```
优化问题（Eq. 20）:

max_{x ∈ {0,1}^{m×n}}   Σ_{i,j}  x_{i,j} · s_{i,j}

s.t.   Σ_j x_{i,j} = k         ∀i    (每个 token 选 k 个)
       Σ_i x_{i,j} = mk/n      ∀j    (每个 expert 服务 q 个)
```

这是一个**二分图最优 b-matching 问题**。通过线性规划松弛和对偶理论，可以推导出解析的更新规则。

#### 4.4.3 对偶推导（附录 C 精要）

**Step 1 — 线性松弛**：将 `x ∈ {0,1}` 放松为 `x ∈ [0,1]`。由于 b-matching 多面体的全幺模性（total unimodularity），松弛后的最优解仍是整数解 → 松弛是精确的。

**Step 2 — 拉格朗日对偶**：引入 token 侧乘子 `α_i` 和 expert 侧乘子 `β_j`：

```
min_{α,β}  max_{x∈[0,1]}  Σ_{i,j} x_{i,j}·(s_{i,j} - α_i - β_j)  +  k·Σ_i α_i  +  (mk/n)·Σ_j β_j
```

内层 max 的解为：
```
x*_{i,j} = 1   iff   s_{i,j} - α_i - β_j > 0
x*_{i,j} = 0   iff   s_{i,j} - α_i - β_j < 0
（等号情况概率测度为零，可忽略）
```

代入 `x*` 得到凸对偶目标（Eq. 23）：

```
L(α, β) = Σ_{i,j} max(0, s_{i,j} - α_i - β_j) + k·Σ_i α_i + (mk/n)·Σ_j β_j
```

**Step 3 — 交替坐标下降**：固定 `β` 优化 `α`，再固定 `α` 优化 `β`，每个子问题有闭式解。

**Token 侧**（固定 `β`）：
```
min_α  k·α + Σ_j max(0, s_{i,j} - β_j - α)
```
这个分段线性函数的最小值恰好在 `s_i - β` 的第 `k` 和第 `(k+1)` 大值之间取得。取第 `(k+1)` 大值：
```
α*_i = quantile_{1-k/n}( s_i - β )        (Eq. 25)
```

**Expert 侧**（固定 `α`）：
```
min_β  (mk/n)·β + Σ_i max(0, s_{i,j} - α_i - β)
```
对称地，解为：
```
β*_j = quantile_{1-k/n}( s_{:,j} - α )    (Eq. 26)
```

两个更新都是 `(1-k/n)`-分位数（沿 token 轴和 expert 轴），这正是"Quantile Balancing"名字的由来。

**从 assignment 回到 routing**：在最优解处，选中的 expert 满足 `s_{i,j} - α*_i - β*_j > 0`，结合 token 约束 `Σ_j x_{i,j}=k`，等价于 `argtopk(s_i - β*)`。路由只需要 expert 阈值 `β`（即 bias `b = -β`）——token 阈值 `α` 是随 batch 变化的中间变量，训练后丢弃。这保证了**训练-推理一致性**：部署时只需固定的 Top-k + 固定 bias，无需任何分位数计算。

#### 4.4.4 QB 更新规则（实用形式）

回到原始符号（`b = -β`），QB 的实用更新为：

```
对每个 expert j:

1.  用 Top-(k+1) 在 s_i + b 上做 routing:
      → 第 1~k 个 = 实际路由的 expert
      → 第 k+1 个 = 该 token 的 cutoff α_i（"门槛值"）

2.  计算 margins:  m_{i,j} = s_{i,j} - α_i
      (这是 s_{i,j} 减去该 token 的 cutoff)

3.  新 bias = margins 的 (1-k/n)-分位数取负，再去均值:
      b̂_j = -quantile_{1-k/n}( m_{:,j} )
      b   = b̂ - mean(b̂)

4.  在下一步生效: t+1 步的 routing 用 t 步算出的 bias（因果性）
```

#### 4.4.5 为什么是 (1-k/n)-分位数？

```
需要: expert j 服务恰好 q = mk/n 个 token

即: 恰好 q 个 token 满足  s_{i,j} + b̂_j > α_i
即: 恰好 q 个 margin 满足  m_{i,j} > -b̂_j
即: -b̂_j 是第 (q+1) 大的 margin（有 q 个比它大）

分位数:  (q+1)/m ≈ q/m = k/n  →  -b̂_j 是 (1-k/n)-分位数
```

#### 4.4.6 与 Sign-based Update 的关系

固定步长更新 `b += γ·sign(ℓ̄-ℓ_j)` 可以理解为对偶目标上的 SignSGD 步：

```
∂L/∂β_j = mk/n - Σ_i χ(s_{i,j} - α_i - β_j > 0)
        = 目标负载 - 实际负载                    (Eq. 27)
```

SignSGD 只保留了梯度的**方向**信息（多了还是少了），丢失了**大小**和**结构**信息。QB 则直接跳到该坐标的**精确最小化点**——不需要 `γ`，一步到位。

```
Sign update (γ 超参):   "负载多了？往反方向走一小步"
QB (无超参):            "最优 bias 精确等于 margins 的 (1-k/n)-分位数"
```

这也解释了 QB 的效率：即使对近 10³ 个 expert，也只需几次交替更新即可收敛。

#### 4.4.7 图示说明（m=8, n=4, k=1）

```
参数: 8 个 token, 4 个 expert, 每个 token 选 1 个 → 目标 q=2 token/expert

┌─────────────────────────────────────────────────────────┐
│ (a) 更新前 — Imbalanced routing                         │
│                                                          │
│     t1 ●──────────→ E1                                   │
│     t2 ●──────────→ E1  ← 4 tokens, 过热                 │
│     t3 ●──────────→ E2  ← 3 tokens                       │
│     t4 ●──────────→ E2                                   │
│     t5 ●──────────→ E3  ← 1 token, 濒死                  │
│     t6 ●──────────→ E1                                   │
│     t7 ●──────────→ E2                                   │
│     t8 ●──────────→ E1                                   │
│                      E4  ← 0 token, 完全死亡!             │
│                                                          │
│ (b) QB 更新过程                                          │
│     每个灰色条 = margin s_{i,j} - α_i                     │
│     虚线红线 = bias 调整量 -b̂_j                           │
│     放在第 (q+1)=3 大的 margin 处                        │
│     → 刚好 q=2 个 margin 在红线之上                       │
│                                                          │
│ (c) 更新后 — Balanced routing (2,2,2,2)                  │
│     红色边 = QB 改变了的 assignment                      │
│     每个 expert 恰好服务 2 个 token                       │
└─────────────────────────────────────────────────────────┘
```

#### 4.4.8 直方图估计（附录 D）— 工程实现

**为什么需要估计？**

每步的 margins 数量 = `m × n`，在百万 token × 896 expert ≈ **十亿级**，分布在多个 data-parallel rank 和 gradient-accumulation micro-batch 上。Gather 所有 margins 做精确分位数 → 通信成本不可接受。

**关键洞察**：QB 更新不需要 margins 本身，只需要它们的 **per-expert 分布**。一个直方图足以总结这个分布，且通信成本固定（不随 m 增长）。

**完整流程**：

```
Step 1 — 定义 required bias:
    r_{i,j} = α_i - s_{i,j}
    (= 刚好让 expert j 进入 token i 的 Top-k 所需的 bias)
    → 对 margins 取负反转顺序 → QB 目标 b̂_j = quantile_{k/n}(r_{:,j})

Step 2 — Binning 范围（自适应）:
    s_{i,j} ∈ (0,1)  (sigmoid 输出)
    α_i = s_{i,j'} + b_{j'}  (cutoff 本身是某个 expert 的 biased score)
    → α_i ∈ (b_min, 1 + b_max)
    → r_{i,j} ∈ [b_min - 1, b_max + 1]

    均匀划分 B 个 bin（实践中 B ≈ 1000）
    bin width w = (b_max - b_min + 2) / B
    → 范围随 bias 自适应调整

Step 3 — 累积（Forward 阶段）:
    每个 rank 把本地的 r_{i,j} scatter-add 到 per-expert 计数的矩阵 H ∈ N^{n×B}
    → 跨所有 micro-batch 累积，无需通信

Step 4 — 通信 + 恢复（Step 结束时）:
    一次 all-reduce 汇总所有 rank 的 local histograms
    → 得到全局直方图：每个 expert j 的 B 个 bin count
    → 每个 rank 独立恢复分位数:
        找到累积 count ≥ ⌈q⌉ 的第一个 bin β_j
        在 bin 内线性插值:
            b̂_j = b_min - 1 + [β_j + clip((q - c_j)/h_j, 0, 1)] · w
    → 去均值: b = b̂ - mean(b̂)
```

**三个实用性质**：

| 性质 | 说明 |
|------|------|
| **精确度** | 真实分位数和估计值在同一个 bin 内，误差 ≤ bin width `w`；B=1000 时误差 ~10⁻³ 量级，观测不到剩余负载失衡 |
| **通信成本** | 仅需一次 all-reduce 传输 `n×B` 个整数（896 × 1000 ≈ 0.9M ints ≈ 3.6MB），不到传输原始 margins 的 **1%** |
| **全局一致性** | counts 可加 → 汇总的直方图精确等于全局 batch 的直方图（而非 per-rank 分位数的平均，后者一般不同） |

还可以对估计值做**指数移动平均**以降低 batch-to-batch 采样噪声，进一步提升负载均衡质量。

#### 4.4.9 QB 与 BIP 的关系

QB 与 BIP [116] 解决同一个 assignment 问题，但 BIP 使用不等式约束 `Σ_j x_{i,j} ≤ k` 和 `Σ_i x_{i,j} ≤ mk/n`。不等式约束引入非负性条件 `α≥0, β≥0`，给两个更新都加上了 `max(0,·)` 裁剪。这个裁剪**只能抑制过热 expert，不能提升欠载 expert**，在 896 expert 的规模下收敛明显更慢。QB 使用等式约束 → 无裁剪 → 更快均衡。

---

### 4.5 总结：三个组件如何协同

```
Stable LatentMoE — 896 experts, 16 active/token, 56:1 sparsity
│
├── Normalized (RMSNorm)
│   ├── 插入位置: expert 聚合后, W_↑ 之前
│   ├── 解决: 路由分支 scale 波动
│   └── 效果: 共享/路由分支平稳合并, val loss + downstream 改善
│
├── SiTU-GLU
│   ├── Gate: β₁·tanh(W_g·x/β₁) ⊙ Sigmoid(W_g·x),  β₁=4
│   ├── Up:   β₂·tanh(W_u·x/β₂),                    β₂=25
│   ├── 解决: 激活爆炸, |f| ≤ 100
│   ├── 保留: SwiGLU 原点附近局部响应
│   └── 优势: 非零梯度（vs hard clamp）
│
└── Quantile Balancing
    ├── 核心: bias = -quantile_{1-k/n}(margins)
    ├── 解决: 896 expert 负载均衡, 无需 γ 超参
    ├── 实现: 直方图估计 (B≈1000 bins)
    │   ├── 通信: 一次 all-reduce, <1% 原始成本
    │   └── 误差: ≤ bin width ~10⁻³
    └── 理论: 对偶坐标下降精确解, 2-3 步收敛
```

这三个改进共同使得在 **2.78T 总参数、896 路由专家、16 激活/token、56:1 极端稀疏** 的配置下，训练依然稳定且高效。

---

## 5. Native Vision（原生视觉）

### 5.1 设计理念

Kimi K3 是**原生多模态**的：文本、图像、视频由同一个 backbone 在统一的上下文窗口中处理，不需要后置的 modality-alignment 阶段。

```
文本:  tokenizer → embedding
图像:  MoonViT-V2 → MLP projector → shared embedding space
视频:  MoonViT-V2 (时空分解 attention) → temporal pooling → MLP projector → shared space

所有模态 → 共享 backbone → next-token prediction
```

这为 long-horizon vision-in-the-loop 行为提供了架构基础：模型可以写代码、截取渲染结果的截图或视频帧、检查产物、迭代修正——整个过程在同一 token stream 中完成，没有跨模型交接。

### 5.2 MoonViT-V2

**关键变化**：完全从零训练（from scratch），不用 SigLIP 等对比学习预训练权重初始化。

**为什么不用预训练？** 主要是训练稳定性：

- SigLIP 初始化的 MoonViT-3D：持续较高的梯度范数，频繁出现 spike
- MoonViT-V2 from scratch：梯度范数更低，训练全程稳定

论文还指出：在足够大的规模下，MoonViT-V2 在各视觉评估上匹配了 SigLIP 初始化的基线，说明**对比预训练作为多模态 LLM 的初始化并非必要**。

**架构规格**：

| 参数 | 值 |
|------|-----|
| 层数 | 27 |
| 参数量 | ~0.4B |
| 归一化 | RMSNorm |
| Bias | 全部移除（linear + attention） |
| Patch Size | 14 |
| 注意力头数 | 12 |

**处理流程**：

- 图像/视频共享参数（与 MoonViT-3D 一致）
- 注意力分解：intra-frame spatial + inter-frame temporal
- Temporal pooling 压缩时间维度
- Pixel-shuffle 2×2 下采样 → 视觉 token 数减少 4×
- 支持最高 **3584×3584** 像素输入

---

## 6. Per-Head Muon 优化器

Kimi K3 使用 Muon 优化器处理矩阵参数。对于 attention 的 Q/K/V 投影矩阵，进一步细化为 **per-head** 变体：

```
标准 Muon:
  对 Q 投影矩阵 W_Q ∈ R^{d × (n_heads·d_head)} 整体做 Newton-Schulz 正交化

Per-Head Muon:
  将 W_Q 沿 head 维度分区 → 对每个 head 的 W_Q^h ∈ R^{d × d_head} 独立做正交化
```

**动机**：

- 全矩阵正交化将所有 head 耦合在一起
- head 间梯度/动量尺度不均衡 → 大尺度的 head 主导更新方向 → 小尺度的 head 更新不足
- Per-head 正交化均衡各 head 的更新尺度

**附带收益**：Newton-Schulz 迭代在细长的 per-head 矩阵上比在全矩阵上更便宜。

---

## 7. 总结：架构创新全景

```
Kimi K3 的架构创新围绕一个统一的设计哲学：

  "让信息在序列、深度、宽度三个维度上高效流动"

  ┌─────────────────────────────────────────────────────┐
  │                                                       │
  │  序列维度 (Sequence)                                  │
  │  ├── KDA: 线性注意力, O(N), 下界衰减 + 全秩输出门     │
  │  └── Gated MLA: 全局注意力, NoPE, KV压缩, 全秩输出门   │
  │     → 3:1 混合, 平衡效率与容量                         │
  │                                                       │
  │  深度维度 (Depth)                                     │
  │  └── Attention Residuals: 把 Attention 用于深度方向    │
  │     ├── Full: O(Ld) memory, 每层独立检索所有前序层     │
  │     └── Block: O(Nd) memory, 8 个 block, 10× 节省     │
  │     → 打破残差连接的"深度 RNN 瓶颈"                    │
  │                                                       │
  │  宽度维度 (Width)                                     │
  │  └── Stable LatentMoE: 896 路由专家, 16 激活/Token     │
  │     ├── Normalized: RMSNorm 稳定路由分支               │
  │     ├── SiTU-GLU: 有界激活函数, 防溢出                  │
  │     └── Quantile Balancing: 分位数均衡负载              │
  │     → 极端稀疏下的稳定训练                             │
  │                                                       │
  │  视觉: MoonViT-V2 from scratch, 原生多模态             │
  │  优化器: Per-Head Muon, 均衡 head 间更新               │
  │                                                       │
  │  → 综合效果: ~2.5× scaling efficiency over K2          │
  └─────────────────────────────────────────────────────┘
```

---

> 参考文献:
> - Kimi K3 Technical Report
> - [63] Kimi Delta Attention / Kimi Linear
> - [57] Attention Residuals (arXiv:2603.15031)
> - [28] DeepSeek-V2 MLA (arXiv:2405.04434)
> - [32] LatentMoE
> - [53] Muon optimizer
