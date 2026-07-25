# DeepSeek Sparse Attention (DSA) 实现解读

> 基于 DeepSeek-V3.2-Exp 推理代码的架构分析
> 文档日期: 2026-06-29

## 目录

- [1. 概述](#1-概述)
- [2. 配置参数](#2-配置参数)
- [3. 整体架构](#3-整体架构)
- [4. Indexer 模块（核心）](#4-indexer-模块核心)
  - [4.1 初始化](#41-初始化)
  - [4.2 输入与输出](#42-输入与输出)
  - [4.3 前向传播完整流程](#43-前向传播完整流程)
- [5. 阶段详解](#5-阶段详解)
  - [5.1 Query 构造：复用 MLA 的 QR](#51-query-构造复用-mla-的-qr)
  - [5.2 Key 构造：独立投影、单头](#52-key-构造独立投影单头)
  - [5.3 RoPE：非交错式的特殊约定](#53-rope非交错式的特殊约定)
  - [5.4 Hadamard 正交旋转](#54-hadamard-正交旋转)
  - [5.5 FP8 块量化](#55-fp8-块量化)
  - [5.6 Per-Head 权重计算](#56-per-head-权重计算)
  - [5.7 fp8_index Kernel](#57-fp8_index-kernel)
  - [5.8 Mask + TopK 选择](#58-mask--topk-选择)
- [6. 关键设计问答](#6-关键设计问答)
  - [6.1 为什么 Indexer 需要独立的 wq_b？](#61-为什么-indexer-需要独立的-wq_b)
  - [6.2 wq_b(MLA) 的作用和低秩分解的好处？](#62-wq_bmla-的作用和低秩分解的好处)
  - [6.3 为什么 Q 复用 qr 而 K 直接用 x？](#63-为什么-q-复用-qr-而-k-直接用-x)
  - [6.4 为什么 Key 不做多头？（面试官与简历）](#64-为什么-key-不做多头面试官与简历)
  - [6.5 为什么用 x 算 weights 不用 qr？](#65-为什么用-x-算-weights-不用-qr)
  - [6.6 x 不是多头，怎么按头赋予权重？](#66-x-不是多头怎么按头赋予权重)
  - [6.7 64 个 weights 如何与 q 的 64 头对应？](#67-64-个-weights-如何与-q-的-64-头对应)
  - [6.8 为什么需要做 FP8 量化？（工程与算法两面）](#68-为什么需要做-fp8-量化工程与算法两面)
- [7. DSA 在 MLA 中的集成](#7-dsa-在-mla-中的集成)
  - [7.1 Prefill 模式（MHA）](#71-prefill-模式mha)
  - [7.2 Decode 模式（MQA）](#72-decode-模式mqa)
- [8. 稀疏注意力 vs 全注意力对比](#8-稀疏注意力-vs-全注意力对比)
  - [8.1 Indexer 与稀疏 MLA 的计算量对比](#81-indexer-与稀疏-mla-的计算量对比)
  - [8.2 KV Cache 大小分析](#82-kv-cache-大小分析)
- [9. 总结](#9-总结)

---

## 1. 概述

DSA (DeepSeek Sparse Attention) 是 DeepSeek-V3.2-Exp 的核心创新，一种**细粒度稀疏注意力机制**。

**核心思路**：用一个轻量的 **Indexer 模块**预先计算 query-key 相关性分数，选出 top-k 个最相关的 key 位置，然后 MLA 只在这 k 个位置上做精细注意力。

**收益**：长序列场景下将 O(n^2) 的注意力复杂度降至 O(n x topk)，大幅减少计算量和内存带宽，同时保持与全注意力几乎一致的模型质量。

---

## 2. 配置参数

以 671B 模型实际配置为例：

| 参数 | 值 | 说明 |
|------|-----|------|
| `q_lora_rank` | 1536 | MLA Q 低秩瓶颈维度 |
| `kv_lora_rank` | 512 | MLA KV 低秩瓶颈维度 |
| `n_heads` | 128 | MLA 注意力头数（若 TP=8，每 rank 16 头） |
| `qk_nope_head_dim` | 128 | QK 内容部分维度 |
| `qk_rope_head_dim` | 64 | QK 位置部分维度 |
| `v_head_dim` | 128 | Value 维度 |
| `index_n_heads` | 64 | Indexer 头数 |
| `index_head_dim` | 128 | Indexer 每头维度（rope=64 + nope=64） |
| `index_topk` | 2048 | 稀疏注意力选 top-k |
| `dtype` | fp8 | 权重和缓存精度 |
| `scale_fmt` | ue8m0 | 量化 scale 格式 |

Indexer 头数(64)远多于 MLA 的 head per rank(16)，64 个头是为了更精细的相关性检索——每个头可以学会一种不同的匹配模式。

---

## 3. 整体架构

```
输入 x (b,s,7168)
     |
     +------------------+------------------+
     |                  |                  |
     v                  v                  v
  wq_a->norm        wkv_a               wk
     |                  |                  |
   qr(1536)        kv k_pe            k(128)
     |                  |                  |
  +--+--+             |                  |
  |     |             |                  |
  v     v             v                  v
wq_b  wq_b(Idx)   kv_norm           k_norm
(MLA) (Indexer)      |         RoPE->Hadamard
     |     |    wkv_b|         ->act_quant->Cache
     |     |         |              |
     |  Index Q -----+------ fp8_index(q,k)
     |     |         |              |
     v     v         v          topk_indices
  MLA Attention <---- DSA Mask ----+
     |                     |
     v                     v
    wo->output         (2048 个位置参与 softmax)
```

---

## 4. Indexer 模块（核心）

Indexer 是 DSA 的核心，负责快速计算每个 query 与全量 key 的相似度，选出最相关的 topk 位置。

### 4.1 初始化

```python
class Indexer(nn.Module):
    def __init__(self, args):
        # Query 投影：复用 MLA 的 qr（低秩表示）
        self.wq_b = Linear(q_lora_rank=1536, n_heads x head_dim=64 x 128)

        # Key 投影：独立，单头 128 维
        self.wk = Linear(dim=7168, head_dim=128)

        # Key 归一化
        self.k_norm = LayerNorm(128)

        # Per-head 权重（可学习）：从 x 学习每头的重要性
        self.weights_proj = Linear(7168, 64, dtype=float32)

        # FP8 格式的 K Cache
        self.k_cache = zeros(batch, max_seq_len, 128, dtype=fp8_e4m3)
        self.k_scale_cache = zeros(batch, max_seq_len, 1, dtype=fp32)
```

### 4.2 输入与输出

| 方向 | 参数 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `x` | `(b, s, dim=7168)` | 当前层的隐藏状态，用于生成 key 和 per-head weights |
| 输入 | `qr` | `(b, s, q_lora_rank=1536)` | MLA 的低秩 Q 表示（复用，零额外投影开销） |
| 输入 | `start_pos` | scalar int | 当前序列在 KV cache 中的起始位置 |
| 输入 | `freqs_cis` | `(s, rope_dim/2)` | 预计算 RoPE 频率 |
| 输入 | `mask` | `(s, s)` 或 None | causal mask |
| 输出 | `topk_indices` | `(b, s, 2048)` | 每个 query 选出的 top-k key 位置索引 |

### 4.3 前向传播完整流程

```
forward(x, qr, start_pos, freqs_cis, mask):

  输入:
    x:    (b, s, 7168)    -- 当前层隐藏状态
    qr:   (b, s, 1536)    -- MLA 的低秩 Q 表示（复用）
    start_pos: integer    -- 当前序列在 cache 中的起始位置

  +-----------------------------------------------------------+
  | 阶段一：Query 构造                                         |
  |   q = wq_b(qr)                               1536->64x128 |
  |   q_pe, q_nope = split(q)                    位置/内容分离 |
  |   q_pe = apply_rotary_emb(q_pe, non-interleaved)  非交错式 |
  |   q = cat([q_pe, q_nope])                     (b,s,64,128) |
  +-----------------------------------------------------------+
  | 阶段二：Key 构造                                           |
  |   k = wk(x)                                   7168->128   |
  |   k = k_norm(k)                               单头归一化   |
  |   k_pe, k_nope = split(k)                                  |
  |   k_pe = apply_rotary_emb(k_pe, non-interleaved)           |
  |   k = cat([k_pe, k_nope])                      (b,s,128)  |
  +-----------------------------------------------------------+
  | 阶段三：量化                                              |
  |   q = rotate_activation(q)    -- Hadamard                 |
  |   k = rotate_activation(k)    -- Hadamard                 |
  |   q_fp8, q_scale = act_quant(q, block_size=128)           |
  |   k_fp8, k_scale = act_quant(k, block_size=128)           |
  |   k_cache[ : , start_pos:end_pos] = k_fp8                 |
  +-----------------------------------------------------------+
  | 阶段四：Per-Head 权重                                      |
  |   weights = weights_proj(x) x 1/sqrt(64)                  |
  |   weights = weights.unsqueeze(-1) x q_scale x 1/sqrt(128) |
  +-----------------------------------------------------------+
  | 阶段五：稀疏检索                                           |
  |   index_score = fp8_index(q_fp8, weights, k_cache,        |
  |                           k_scale_cache)                   |
  |   if mask: index_score += mask    -- causal mask           |
  |   topk_indices = index_score.topk(2048, dim=-1).indices   |
  +-----------------------------------------------------------+
  | 输出: topk_indices (b, s, 2048)                            |
  +-----------------------------------------------------------+
```

---

## 5. 阶段详解

### 5.1 Query 构造：复用 MLA 的 QR

```python
# MLA 已经计算好的（mla.forward 第 560 行）
qr = q_norm(wq_a(x))          # (b,s,7168) -> (b,s,1536)

# Indexer 只加一个上投影
q  = wq_b(qr)                 # (b,s,1536) -> (b,s,8192)
q  = q.view(b,s,64,128)       # reshape 为 64 头，每头 128 维
```

**值得注意**：`qr` 是三维的 `(b, s, q_lora_rank)`，不是二维 `(b, s)`。第三维 1536 是低秩瓶颈维度，编码了每个 token 被压缩后的 query 表示。

**优点**：
- 如果从 x 直接投影需 `7168 x 8192 = 59M` 参数，复用 qr 只需 `1536 x 8192 = 12.6M`
- `wq_a` 是 MLA 本来就有的，Indexer 骑在这个共享瓶颈上，零额外代价拿到一个已经训练好的 query 压缩表示

### 5.2 Key 构造：独立投影、单头

```python
k = wk(x)          # (b,s,7168) -> (b,s,128)，单头，128 维
k = k_norm(k)      # LayerNorm
```

Key 的设计与 Query 形成鲜明对比：

| 设计 | 说明 |
|------|------|
| **单头** | 仅 128 维，不分多头，被 64 个 query head 共享计算内积 |
| **独立投影** | 不复用 MLA 的 kv_latent。kv_latent 联合编码了 K 和 V，语义上不匹配 |
| **参数量小** | `7168 x 128 = 0.9M`，完全可以接受 |

### 5.3 RoPE：非交错式的特殊约定

Indexer 和 MLA 使用不同的 RoPE 布局。README 特别指出这是已修复的重要 bug。

**MLA（交错式 interleaved）**：

```python
# 默认 interleaved=True
apply_rotary_emb(q_pe, freqs_cis)  # MLA 的 default

# 布局: [real0, imag0, real1, imag1, real2, imag2, ...]
```

**Indexer（非交错式 non-interleaved）**：

```python
# 显式 interleaved=False
apply_rotary_emb(q_pe, freqs_cis, interleaved=False)  # Indexer
apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False)

# 布局: [real0, real1, real2, ..., imag0, imag1, imag2, ...]
```

两种布局不能混用，否则位置编码的语义完全错误。这是推理代码正确性的关键细节。

### 5.4 Hadamard 正交旋转

```python
def rotate_activation(x):
    return hadamard_transform(x, scale=hidden_size ** -0.5)
```

**作用**：乘以正交 Hadamard 矩阵，将向量信息均匀分布到所有维度上。

**为什么需要**：某些维度可能存在离群值（如 500），此时 FP8 量化的 scale 被该维度主导，其余 127 维的信息几乎丢失。Hadamard 将能量摊平：

```
原始: [0.01, 0.02, 500, 0.01, ...]   -- 第 2 维主导，量化后其余 127 维信息丢失
旋转: [~7.8, ~7.2, ~8.1, ...]         -- 能量均匀，每维损失可控
```

量化后每维损失一丁点，但 128 维综合得出的内积仍然稳健。

**来源**：QuaRot / SpinQuant 技术。

### 5.5 FP8 块量化

```python
y_fp8 = clamp(x / scale, -448, 448)    # scale = amax(x) / 448
```

| 属性 | 值 |
|------|------|
| 分块大小 | 128 维一组（正好覆盖 1 个 head） |
| 量化格式 | `fp8_e4m3`（3 指数位 + 4 尾数位），范围 [-448, 448] |
| Scale 格式 | `ue8m0`（纯 8 位指数，无尾数），2 的幂 |
| Scale 计算 | `fast_round_scale` 将 scale 取整到 2 的幂 |

Key 侧的 scale 同时服务于两个目的：让 kernel 做反量化恢复数值，以及修正 index_score 的量级（`kernel.py:246-247`）。

### 5.6 Per-Head 权重计算

```python
weights = weights_proj(x) x 1/sqrt(64)              # (b,s,64)
weights = weights.unsqueeze(-1) x q_scale x 1/sqrt(128)  # (b,s,64,1)
```

四个成分各司其职：

| 成分 | 含义 |
|------|------|
| `weights_proj(x)` | **可学习的路由**：从 x 预测每个头对这个 token 的重要性。`Linear(7168->64)` 的 64 行并行从 x 读取不同方向的信号 |
| `x 1/sqrt(64)` | **头数归一化**：防止 64 头跨头求和后分数膨胀 |
| `x q_scale` | **FP8 量化补偿**：恢复 q/k 内积的真实量级。不同 head 的量化 scale 不同，不加补偿则分数不可比 |
| `x 1/sqrt(128)` | **标准 softmax scale**：防止内积随 head_dim 增大 |

### 5.7 fp8_index Kernel

Indexer 的计算热点，用 TileLang 编写，在 GPU 上高效执行。

#### 调用参数

| 实参 | 形状 | 类型 | 含义 |
|------|------|------|------|
| `q_fp8` | `(b, m, 64, 128)` | fp8 | m 个 query token，64 头，每头 128 维 |
| `weights` | `(b, m, 64, 1)` | fp32 | 融合后的每头权重 |
| `k_cache` | `(b, n, 128)` | fp8 | 全历史 key，单头、无头维 |
| `k_scale_cache` | `(b, n, 1)` | fp32 | 每个 key 位置的量化 scale |
| 输出 `index_score` | `(b, m, n)` | fp32 | 每个 query 对每个 key 的相关性分数 |

#### Kernel 内部逻辑

```
对每个 (batch, query_pos):

  1. 加载 Q
     q_smem[64 x 128]    = q_fp8[b][m]       -- 当前 token 的 64 头，共享内存
     q_weight[64]        = weights[b][m]      -- 64 个 per-head 权重，寄存器

  2. 分块扫描全量 key
     blk_n1=512, blk_n2=128（外层并行，内层流水线）

     for i1 in 0..ceil(n/512):       -- 并行在不同 SM
       for i2 in 0..4:               -- 流水线

         // 加载 K 块
         k_smem[128 x 128]   = k_cache[b][i1*512+i2*128 ..]
         k_scale[128]        = k_scale_cache[b][...]

         // 3. FP8 GEMM: k_smem x q_smem^T
         logits[128 x 64] = k_smem @ q_smem^T
         // logits[j][h] = sum_d k[j][d] x q[h][d]

         // 4. ReLU + per-head weight
         logits[j][h] = max(logits[j][h], 0) x q_weight[h]
         // ReLU 诱导稀疏（负相关清零）

         // 5. 跨头求和
         logits_sum[128] = sum_h logits[128 x 64]

         // 6. Key scale 修正
         logits_sum[128] x= k_scale[128]

         // 7. 写出结果块
         index_score[b][m][offset:offset+128] = logits_sum[128]
```

**核心数学公式**（对 query m, key j）：

$$score(m, j) = k_scale_j \cdot \sum_{h=0}^{63} ReLU\left( \sum_{d=0}^{127} k_{j,d} \cdot q_{m,h,d} \right) \cdot w_{m,h}$$

### 5.8 Mask + TopK 选择

```python
if mask is not None:
    index_score += mask    # prefill: 叠加 causal mask（上三角 -inf）

topk_indices = index_score.topk(min(2048, end_pos), dim=-1).indices
# (b, s, 2048): 每个 query 选出的 key 位置索引
```

---

## 6. 关键设计问答

### 6.1 为什么 Indexer 需要独立的 wq_b？

Indexer 和 MLA 的 query 配置不同：

| | MLA | Indexer |
|---|---|---|
| 头数 | 16 head per rank | 64 heads |
| 每头维度 | qk_nope=128 + qk_rope=64 = 192 | rope=64 + nope=64 = 128 |
| 用途 | 精确注意力计算（加权求和 value） | 相关性检索（选 topk 位置） |

`wq_b(Idx)` 负责将共享的低秩 `qr` 投影到 Indexer 自己的多头空间。不能复用 MLA 的 wq_b，因为维度（192 vs 128）、头数（16 vs 64）和任务目标都不同。

### 6.2 wq_b(MLA) 的作用和低秩分解的好处？

MLA 的核心创新是**低秩分解**，Q 和 KV 都走"压缩->展开"的瓶颈结构：

```
Q:  dim=7168 ---wq_a--> q_lora_rank=1536 ---wq_b(MLA)--> 16 x 192
KV: dim=7168 --wkv_a--> kv_lora_rank=512 ---wkv_b-----> 16 x (128+128)
```

**最大收益是 KV Cache 压缩**。传统 MHA 需缓存每头的完整 K 和 V，而 MLA 只缓存低秩潜变量 `kv_lora_rank=512`：

```
传统 MHA KV cache:  n_heads x (k_dim + v_dim) x seqlen = 16 x 256 x seqlen
MLA KV cache:       kv_lora_rank x seqlen                = 512 x seqlen
```

压缩比约 **8 倍**。`wkv_b` 在需要时把低秩表示展开回完整的 K 和 V，不需要持久存储。

此外，低秩瓶颈 `qr` 被 Indexer 的 `wq_b(Idx)` 复用，零额外成本拿到浓缩后的 query 表示。

### 6.3 为什么 Q 复用 qr 而 K 直接用 x？

**Q 侧**：MLA 已有 `qr`，天然适合做 Indexer 的 query 源。

| 方案 | Indexer Q 参数 |
|------|---------------|
| 直接从 x 投影: `Linear(7168->8192)` | ~59M |
| 复用 qr + `Linear(1536->8192)` | ~12.6M |

**K 侧**：MLA 的 KV 路径产出的是 `kv_latent`（512 维），这是一个**联合编码了 K 和 V 的潜变量**：

```
kv_latent: "我是压缩的 KV 联合体，同时包含 key 和 value 的信息"
index key: "我是干净的 key，只需要和 query 算内积"
```

两者语义不匹配。且 key 只需 128 维，参数仅 0.9M，直接用 x 投影也毫无压力。

| | Q | K |
|---|---|---|
| 是否有可复用的表示 | 已解决，qr (MLA 已算) | 否，kv_latent 语义不匹配 |
| 新增参数 | 12.6M (wq_b) | 0.9M (wk) |
| 直接从 x 投影的方案 | 59M（多花 46M） | 0.9M（本来就少） |

### 6.4 为什么 Key 不做多头？（面试官与简历）

Key 是**被匹配方**，多样性只需要在匹配方（query 侧）实现。

一个类比：

```
Query (64 个头)  =  64 个面试官，各有各的评判标准
  head_0: "远距离因果关联"
  head_1: "语法相似性"
  head_5: "同义词语义匹配"
  ...

Key (单头)      =  一份简历，包含候选人所有必要信息
                    不需要为每个面试官重写一份简历
```

kernel 中同一个 key 被 64 个 query head 共享计算内积。如果 key 也做 64 头：

|维度|单头 Key（实际）|64 头 Key（假设）|浪费|
|---|---|---|---|
|wk 参数|0.9M|58M|x64|
|K Cache / 每 token|128B|8KB|x64|
|GEMM 内存带宽|扫 n x 128B|扫 n x 8KB|x64|

而最终跨头求和 `sum_h ReLU(q_h . k) x w_h` 中，query 侧的 64 种视角 + `weights_proj` 的动态加权已提供足够的检索多样性。让 key 也多头相当于"把同一份简历复印 64 份"。

### 6.5 为什么用 x 算 weights 不用 qr？

`weights_proj` 要回答的问题，和 query-key 匹配是**不同维度**的决策：

```
由 query-key 匹配（qr 够用）："这个 token 的语义是什么，应该和怎样的 key 对齐"
由 weights 决定（需要 x）：   "这个 token 难还是简单？处在什么语法位置？
                              需要多少上下文？应该用哪些检索模式？"
```

`qr`（1536 维）经过 `wq_a` 的压缩瓶颈，保留的信息优先服务于 query-key 匹配。而语法角色、语言模式、上下文复杂度这些元信息可能已被丢弃。

类比：MoE 的 Gate 同样用完整的 x 做路由，而非某个压缩表示。

```python
# model.py:687
scores = linear(x.float(), self.weight.float())  # MoE Gate 也用 x
```

### 6.6 x 不是多头，怎么按头赋予权重？

```python
self.weights_proj = Linear(7168, 64, dtype=float32)
# 权重矩阵 W 属于 R^(64 x 7168)
```

64 行，每行是一套独立的读出头。各行从同一个 x（7168 维）中读取不同方向的信号，产出 64 个标量：

```
w_0  = W[0,:] . x    -- "这个 token 需要近距离语法匹配"
w_1  = W[1,:] . x    -- "这个 token 需要远距离语义关联"
...
w_63 = W[63,:] . x   -- "这个 token 在代码上下文中"
```

和 MLA 里 `wq_b: Linear(1536, 64x128)` 本质上一样——权重矩阵没有显式的"头"结构，是靠 reshape 或输出位置来"解读"为多头：

```python
# MLA wq_b: 输出 8192 维，view 成 64 x 128 就是 64 头
q = self.wq_b(qr).view(b, s, 64, 128)

# weights_proj: 输出 64 维，自然就是 64 个头各一个权重，无需 reshape
weights = self.weights_proj(x)  # (b,s,64)
```

### 6.7 64 个 weights 如何与 q 的 64 头对应？

kernel 内的索引对齐：

```python
for i_h in 0..63:               # i_h 遍历 64 个 head
    for i3_n in 0..127:         # i3_n 遍历 128 个 key 位置（当前块内）

        logits[i3_n, i_h] = ReLU(...) x q_s_frag[i_h]
```

`i_h` 是同一个循环变量，贯穿 kernel 的 GEMM、ReLU、加权、归约全过程。**下标 `h` 天然对齐**：

| 对齐方式 | 说明 |
|----------|------|
| Kernel 内 | `q_smem[h, d]` 和 `q_s_frag[h]` 共用索引 `h` |
| 权重矩阵 | `wq_b` 的第 h 个 128 维块 对应 `weights_proj` 的第 h 行 |
| 训练梯度 | 梯度耦合：`dL/dq_h` 和 `dL/dw_h` 通过求和操作绑定，head 和 weight 自动形成语义配对 |

### 6.8 为什么需要做 FP8 量化？（工程与算法两面）

**工程层面**：不量化 Indexer 跑不动长序列。

| 精度 | K Cache / 每 token | 128K 序列总量 | 每次 decode 读 cache |
|------|--------------------|--------------|-------------------|
| BF16 | 128 x 2B = 256B | 32MB | 16KB |
| FP8 | 128 x 1B = 128B | 16MB | 8KB |

FP8 使 K Cache 减半、内存带宽减半。FP8 GEMM 的硬件吞吐也是 BF16 的数倍。

**算法层面**：Indexer 的输出是 **排序结果（topk 集合）**，不是精确的注意力权重，天然容忍量化噪声。

```
MLA     = 高考阅卷（0.5 分差影响录取）
Indexer = 海选筛简历（挑出前 5% 即可，内部排序不重要）
```

Hadamard 变换 + FP8 量化组合拳：**Hadamard 让 FP8 量化伤不到排序质量，FP8 让 Indexer 扫全序列时不受内存墙限制**。两者缺一，要么算不准，要么跑不动。

---

## 7. DSA 在 MLA 中的集成

DSA 以**加性稀疏掩码**的方式集成到 MLA 中：非选中位置设为 `-inf`，softmax 后权重自然归零。

```
原始分数: [0.2, 0.1, 0.3, 0.0, 0.1, 0.0, 0.2, 0.1, ..., 0.0]
DSA 注入: [-inf, 0.1, -inf, -inf, 0.1, -inf, 0.2, -inf, ..., -inf]
softmax:  [0, 0.2, 0, 0, 0.2, 0, 0.4, 0, ..., 0.2]
                  ^                                ^
                  |  仅选中的位置参与 softmax 竞争    |
```

### 7.1 Prefill 模式（MHA）

```python
# 1. 标准 MLA：全注意力
q = cat([q_nope, q_pe])
k = cat([k_nope, k_pe.expand(-1,-1,n_local_heads,-1)])
scores = einsum("bshd,bthd->bsht", q, k) x softmax_scale

# 2. Indexer 选 topk
topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)

# 3. 构造稀疏掩码
index_mask = full(-inf)[b,s,s].scatter(-1, topk_indices, 0)
index_mask += mask     # 叠加 causal mask
scores += index_mask.unsqueeze(2)   # 广播到各 head

# 4. 仅选中位置 softmax + 加权求和
scores = scores.softmax(dim=-1)
x = einsum("bsht,bthd->bshd", scores, v)
```

### 7.2 Decode 模式（MQA）

利用 MLA 的低秩结构，不显式展开 key，直接操作 cache：

```python
# 1. 吸收 wkv_b 到 q_nope（避免显式构建完整 K）
q_nope = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])

# 2. 分别算内容分数和位置分数
scores = (einsum("bshc,btc->bsht", q_nope, kv_cache[:b,:n]) +
          einsum("bshr,btr->bsht", q_pe,   pe_cache[:b,:n])) x scale

# 3. DSA 掩码
index_mask = full(-inf)[b,1,n].scatter(-1, topk_indices, 0)
scores += index_mask.unsqueeze(2)

# 4. softmax + 加权求和
scores = scores.softmax(dim=-1)
x = einsum("bsht,btc->bshc", scores, kv_cache[:b,:n])
x = einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])
```

Decode 模式下，选中位置内的注意力分数由原始 q/k 内积决定，不是随便给的；被 DSA 屏蔽的位置权重严格为 0。

---

## 8. 稀疏注意力 vs 全注意力对比

| | 全注意力 MLA | DSA |
|---|---|---|
| 每个 query 看的 key | 全部 n 个 | topk=2048 个 |
| softmax 分母 | sum over 全部 n | sum over 选中的 2048 个 |
| 非选中位置权重 | 非零，有值 | 严格为 0 |
| key/value 读取 | 读全部 n 个 | Indexer 便宜扫全部 n，注意力只 gather 2048 个 |
| 计算复杂度 | O(n x d x h) | O(n x d_cheap) + O(2048 x d x h) |
| n=128K 时估算 | ~3.1B FLOPs | Indexer ~16M + Attention ~50M ~= 66M |

**Demo 代码与生产 Kernel 的差异**：

- **Demo 代码**（可读性优先）：先全量算分，再 mask 掉非选中位置。数学语义清晰，但计算没省。相当于"先翻完 128000 本书做好笔记，再把 125952 本撕掉只留 2048 本"。

- **生产 Kernel**（性能优先）：Indexer 选完 topk 后，Kernel 只 gather 选中的 key/value 位置，从**头到尾不加载、不计算**非选中位置。省的不是"算完再扔"，而是**从一开始就不读不算**。相当于"fmt 直接看索引卡挑 2048 本，只看这些"。

DSA 节省的并非 mask 操作本身，而是**避免了加载和计算那 98% 非选中位置的内存流量和算力**。



### 8.1 Indexer 与稀疏 MLA 的计算量对比

DSA 的 Indexer 自身也要消耗算力（扫全序列打分），必须和它节省的 attention 计算量放在一起算总账。

> **FLOPs 计算约定**：矩阵乘法 `A(M,K) @ B(K,N)` 的 FLOPs = `2 × M × K × N`。每个输出元素需要 K 次 **乘加（multiply-add）**，每次乘加 = 1 次乘法 + 1 次加法 = 2 FLOPs。这个 `×2` 是工业界（NVIDIA、Meta、Google）的通用计数惯例。

#### 关键维度

| 符号 | 值 | 含义 |
|------|-----|------|
| `d` | 7168 | hidden dim |
| `q_lora_rank` | 1536 | qr 的 rank |
| `h_idx` | 64 | Indexer 头数 |
| `d_idx` | 128 | Indexer 每头维度 |
| `h_mla` | 128 | MLA 头数 |
| `qk_head_dim` | 192 (128+64) | MLA 每头 QK 维度 |
| `v_head_dim` | 128 | MLA value 维度 |
| `topk` | 2048 | 每 query 选的位置数 |

#### Indexer 各步骤 FLOPs（prefill s=4096，单 batch，单层）

| 步骤 | 公式 | FLOPs |
|------|------|-------|
| ① `wk`: x(7168)→k(128) | `2 × s × d × d_idx` | `2×4096×7168×128 = 7.5×10⁹` |
| ② `wq_b`: qr(1536)→q(8192) | `2 × s × q_lora_rank × h_idx×d_idx` | `2×4096×1536×8192 = 1.03×10¹¹` |
| ③ `weights_proj`: x(7168)→w(64) | `2 × s × d × h_idx` | `2×4096×7168×64 = 3.76×10⁹` |
| ④ `fp8_index` GEMM | `2 × s² × h_idx × d_idx` | `2×4096²×64×128 = 2.75×10¹¹` |
| ⑤ 其他（RoPE/Hadamard/量化） | 元素级操作 | 可忽略 |
| **Indexer 合计** | | **≈ 3.9×10¹¹** |

Indexer 的绝大部分开销来自第④步 `fp8_index` 的 GEMM（全量 `q @ kᵀ`），占总量的 **~70%**。

#### MLA 稀疏注意力 FLOPs（topk=2048）

MLA 的 attention 部分分两步（Q 和 KV 的投影 `wq_a`/`wkv_a`/`wkv_b` 是固定开销，不论是否使用 DSA 都要算，不计入比较）：

| 步骤 | 公式 | FLOPs |
|------|------|-------|
| A. 稀疏打分 `Q @ Kᵀ` | `2 × s × h_mla × qk_head_dim × topk` | `2×4096×128×192×2048 = 4.12×10¹¹` |
| B. 稀疏加权 `Score @ V` | `2 × s × h_mla × v_head_dim × topk` | `2×4096×128×128×2048 = 2.75×10¹¹` |
| **MLA 稀疏合计** | | **≈ 6.87×10¹¹** |

#### 对比：Indexer 占稀疏 MLA 的多少？

```
Indexer / 稀疏MLA = 3.9×10¹¹ / 6.87×10¹¹ ≈ 57%
```

每花 1 块钱做稀疏 attention，Indexer 要花 **~5 毛 7** 来筛选位置。

#### DSA 净收益：扣除 Indexer 后还剩多少？

```
无 DSA 的全量 attention = 2 × s² × h_mla × qk_head_dim    (打分)
                        + 2 × s² × h_mla × v_head_dim    (加权)
                      ≈ 1.375×10¹²  (s=4096)

DSA 总计算 = Indexer + 稀疏MLA = 3.9×10¹¹ + 6.87×10¹¹ = 1.077×10¹²

净节省 = 1.375×10¹² - 1.077×10¹² = 2.98×10¹¹  → 约 22%
```

#### 序列越长收益越大

`topk=2048` 是**固定值**，而 Indexer 的 `q@kᵀ` 随 `O(s²)` 增长，MLA 全量 attention 也随 `O(s²)` 增长。s 越长，DSA 的相对收益越可观：

| s | 全量 attention | DSA 总计算（Indexer + 稀疏MLA） | 净节省比例 |
|---|---|---|---|
| 2048 | 3.44×10¹¹ | 3.85×10¹¹ | **≈ 0%（topk=s，无节省）** |
| 4096 | 1.38×10¹² | 1.08×10¹² | **22%** |
| 8192 | 5.50×10¹² | 3.35×10¹² | **39%** |
| 16384 | 2.20×10¹³ | 1.15×10¹³ | **48%** |
| 32768 | 8.80×10¹³ | 4.28×10¹³ | **51%** |

s 从 4K 增长到 32K，净节省比例从 22% 升至 ~51%。长序列下，Indexer 的 `O(s²)` 虽也在涨，但 MLA attention 省掉的 `O(s²)` 更多——因为 MLA 有 128 头 × 192 维，Indexer 只有 64 头 × 128 维（`24576 vs 8192` 因子）。**Indexer 的"窄通道"设计（少头 + 低维 + FP8）是它能以较小代价完成全量搜索的关键。**

#### FP8 硬件加速进一步压低 Indexer 开销

Indexer 的核心 GEMM 使用 **FP8** 精度，而 MLA attention 是 BF16。在 H100/H90 上，FP8 Tensor Core 吞吐是 BF16 的 **2 倍**。换算成墙上时间：

```
Indexer 有效耗时 = 3.9×10¹¹ / 2(FP8加速) ≈ 1.95×10¹¹ "等效BF16 FLOPs"
Indexer / 稀疏MLA (墙上时间) = 1.95×10¹¹ / 6.87×10¹¹ ≈ 28%
```

**账面上** Indexer 是稀疏 MLA 的 57%，但 **硬件上**受 FP8 加速后仅 **~28%**。

#### 总结

- Indexer 不是免费的——它自己做了一次全量 `q@kᵀ` 搜索（64 头 × 128 维 × FP8）
- 但它的"窄通道 + FP8"组合拳使得检索成本（FLOPs）远低于 MLA 全量 attention（128 头 × 192 维 × BF16），**因子比约为 `8192FP8 : 24576BF16 ≈ 1:6`**
- s=4096 时，Indexer 开销 ≈ 稀疏 attention 的 57%，但净赚回 **22%** 的总 attention FLOPs
- s ≥ 16K 时，净节省接近 **50%**，越长越划算



### 8.2 KV Cache 大小分析

#### MLA 的压缩式 KV Cache

MLA 的核心创新之一是**低秩 KV 压缩**，不缓存每头的完整 K 和 V，而是缓存低秩潜变量。从配置和代码算出实际大小：

```python
# model.py:541-542
self.kv_cache = zeros(max_batch_size, max_seq_len, kv_lora_rank=512)    # 压缩潜变量
self.pe_cache = zeros(max_batch_size, max_seq_len, qk_rope_head_dim=64) # 位置编码
```

| 缓存 | 形状 | 数据类型 | 每 token 大小 |
|------|------|---------|--------------|
| `kv_cache`（压缩后的 KV 潜变量） | `(s, 512)` | BF16（2B） | **1024 B** |
| `pe_cache`（位置编码，RoPE 精度敏感） | `(s, 64)` | BF16（2B） | **128 B** |
| **MLA 单层合计** | | | **1152 B ≈ 1.125 KB** |

#### 对比传统 MHA（假设同等规模）

传统 MHA 需要为每头分别缓存完整的 K 和 V：

```
传统 K: n_heads=128 × qk_head_dim=192 × 2B = 48 KB / token
传统 V: n_heads=128 ×  v_head_dim=128 × 2B = 32 KB / token
传统 MHA 单层合计: 80 KB / token
```

**MLA 压缩比（单层）：** `1.125 KB / 80 KB ≈ 1:71`

#### 61 层总账

| 方案 | 单层 per token | **61 层总计 per token** |
|------|:-:|:-:|
| 传统 MHA（128 KV heads） | 80 KB | **4.88 MB** |
| **MLA（BF16 推理）** | 1.125 KB | **68.6 KB** |
| **MLA（生产 FP8 部署）** | ~656 B | **~40 KB** |

生产部署下 `kv_cache` 存 FP8 + ue8m0 scale：

```
kv_cache FP8:   512 × 1B              = 512 B
kv_scale:       512/128=4 个 × 4B(fp32) =  16 B
pe_cache BF16:   64 × 2B              = 128 B（RoPE 仍用 BF16）
生产单层合计:                             656 B
```

#### 128K 序列长度下显存占用量

| 方案 | 单层 | **61 层总计** |
|------|:-:|:-:|
| 传统 MHA | 10.2 GB | **~610 GB**（多 GPU 都塞不下） |
| MLA BF16 | 144 MB | **~8.4 GB**（单 GPU 可行） |
| MLA FP8 | 84 MB | **~4.9 GB**（单 GPU 无压力） |

传统 MHA 128K 需要 610 GB KV cache，远超单 GPU 显存（H100 80GB）。**MLA 的 ~70× 压缩才是长上下文可行的根本原因。**

#### Indexer 的 K Cache（额外开销）

Indexer 在 MLA 的 KV cache 之外，还有自己独立的 K cache（用于稀疏检索）：

```python
# model.py:453-454
self.k_cache = zeros(b, s, 128, dtype=fp8_e4m3)        # FP8 单头 key
self.k_scale_cache = zeros(b, s, 1, dtype=fp32)         # 量化 scale
```

| 缓存 | 单层 per token |
|------|:-:|
| `k_cache` | 128 × 1B = **128 B** |
| `k_scale_cache` | 1 × 4B = **4 B** |
| **Indexer 单层合计** | **132 B** |
| **61 层总计** | **8 KB** |

相比省掉的传统 KV cache（4.88 MB/token），Indexer 额外 132 B/token 是非常值得的投入。

#### 总结

| | 传统 MHA | MLA | 压缩比 |
|---|---|---|---|
| 单层 per token | 80 KB | 1.125 KB | **71×** |
| 61 层 per token | 4.88 MB | 68.6 KB | **73×** |
| 128K 总显存 | ~610 GB | ~8.4 GB (BF16) / ~4.9 GB (FP8) | **73-125×** |
| 额外 Indexer cache | — | +132 B/token (61层: +8 KB) | 零头 |

> **关键理解**：MLA 的 KV cache 压缩（71×）和 DSA 的稀疏注意力（topk=2048）是**两个独立但互补的维度**——MLA 解决了"KV cache 放不下"的内存问题，DSA 解决了"全量 attention 算不完"的计算问题。两者共同支撑了 128K+ 长上下文的可行性。

---

## 9. 总结

DSA 的精妙设计：

1. **轻量 Indexer**：复用 MLA 的 qr（省 46M 参数），Key 单头 128 维（省 64x cache），外加 64 头 + 动态权重提供检索多样性
2. **全链路 FP8**：Hadamard 抗量化噪声 -> FP8 GEMM 高吞吐 -> ReLU 天然稀疏 -> 归约效率高
3. **生产 kernel 融合**：Indexer 扫全序列（轻量 FP8），注意力只 gather topk 位置（精细 BF16），两阶段各取所需
4. **软掩码集成**：以加性掩码进入 MLA，不破坏选中位置间的原始分数竞争关系

DSA = **一个便宜的"快筛器"**：先轻量化打分从全序列筛出 2048 个候选人，再让 MLA 只在这 2048 个位置上做精细注意力。

---

> **本文档基于 DeepSeek-V3.2-Exp 推理代码逐行分析整理而成。所有代码引用均来自 `inference/model.py` 和 `inference/kernel.py`。**
