# DeepSeek V4 架构深度解析

> 基于 DeepSeek V4 论文与开源代码（`model.py`, `kernel.py`, `config.json`）的逐层分析，以问答形式组织。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Hyper-Connections 详解](#2-hyper-connections-详解)
3. [Hyper-Connections 的计算开销](#3-hyper-connections-的计算开销)
4. [CSA 与 HCA 概述](#4-csa-与-hca-概述)
5. [CSA 逐步骤详解（完整 Shape 追踪）](#5-csa-逐步骤详解完整-shape-追踪)
   - 3.1 [滑动窗口索引](#step-31-滑动窗口索引)
   - 3.2 [Indexer 选择压缩 KV 的 top-k](#step-32-indexer-选择压缩-kv-的-top-k)
   - 3.3 [合并索引](#step-33-合并索引)
   - 3.4 [滑动窗口 + 压缩 KV 的分工设计](#34-滑动窗口--压缩-kv-的分工设计)
   - 3.5 [Prefill vs Decode 完整对比](#35-prefill-vs-decode-完整对比)
   - 3.6 [Compressor 完整计算流程](#36-compressor-完整计算流程)
   - 3.7 [Indexer 与主 Compressor 的关系](#37-indexer-与主-compressor-的关系)
   - 3.8 [KV Cache 存储分析](#38-kv-cache-存储分析)
6. [wq_b 为什么输出 65536 而不是 7168](#6-wq_b-为什么输出-65536-而不是-7168)
7. [Overlap 机制与 Score 详解](#7-overlap-机制与-score-详解)
8. [Indexer 的 KV Cache 计算与加权池化](#8-indexer-的-kv-cache-计算与加权池化)
9. [架构设计问答（为什么）](#9-架构设计问答为什么)
   - 9.1 [为什么要有 Indexer？](#91-为什么要有-indexer)
   - 9.2 [为什么要有 Compressor？](#92-为什么要有-compressor)
   - 9.3 [Compressor 如何压缩 token 为 block](#93-compressor-如何压缩-token-为-block)
   - 9.4 [Indexer 的 top-k 怎么在主 Compressor 中用上](#94-indexer-的-top-k-怎么在主-compressor-中用上)
   - 9.5 [为什么还要保留 raw KV cache](#95-为什么还要保留-raw-kv-cache)
   - 9.6 [既保存 raw 又保存压缩，还能省 KV cache 吗](#96-既保存-raw-又保存压缩还能省-kv-cache-吗)
   - 9.7 [压缩块会被淘汰吗？](#97-压缩块会被淘汰吗indexer-选出的块没被缓存怎么办)
   - 9.8 [分工总结：谁省了什么](#98-分工总结谁省了什么)

---

## 1. 整体架构概览

### 1.1 核心参数（DeepSeek-V4-Pro, `config.json`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `dim` | 7168 | 隐藏维度 |
| `n_layers` | 61 | Transformer 层数 |
| `n_heads` | 128 | 注意力头数 |
| `head_dim` | 512 | 每头维度 |
| `rope_head_dim` | 64 | RoPE 作用的维度（512 维中只有最后 64 维） |
| `n_routed_experts` | 384 | 路由专家数 |
| `n_activated_experts` | 6 | 每 token 激活专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `q_lora_rank` | 1536 | Q 投影低秩瓶颈维度 |
| `o_lora_rank` | 1024 | O 投影低秩维度 |
| `o_groups` | 16 | O 投影分组数 |
| `hc_mult` | 4 | Hyper-Connection 副本数 |
| `window_size` | 128 | 滑动窗口大小 |
| `dtype` | fp8 | 默认精度 |
| `expert_dtype` | fp4 | 专家权重精度 |
| `n_hash_layers` | 3 | 前 3 层使用哈希路由 |
| `index_n_heads` | 64 | Indexer 的 head 数 |
| `index_head_dim` | 128 | Indexer 的 head 维度 |
| `index_topk` | 1024 | Indexer 选出的 top-k 压缩块数 |

### 1.2 整体数据流（`Transformer.forward`, `model.py:801`）

```
Input IDs → ParallelEmbedding → [expand HC copies] → Block×61 → HC Head → Logits
                                               ↓ (可选)
                                          MTP Block → Logits₂
```

```python
def forward(self, input_ids, start_pos=0):
    h = self.embed(input_ids)
    # 扩展为 hc_mult=4 份副本用于 Hyper-Connections
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)   # [b, s, 4, 7168]
    for layer in self.layers:
        h = layer(h, start_pos, input_ids)
    logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
    return logits
```

### 1.3 架构全景图

```
                    Input Tokens
                         │
                    ParallelEmbedding
                         │
                    [b, s, 4, 7168]  ← HC 扩展
                         │
          ┌──────────────┴──────────────┐
          │     Block × 61              │
          │  ┌──────────────────────┐   │
          │  │ hc_pre (4→1 混合)    │   │
          │  │   + Sinkhorn 分解     │   │
          │  │ RMSNorm              │   │
          │  │ MLA Attention        │   │
          │  │  ├─ 低秩 Q (1536)    │   │
          │  │  ├─ 单头 KV (512)    │   │
          │  │  ├─ Sliding Window   │   │
          │  │  ├─ Compressed KV    │   │
          │  │  ├─ Indexer (topk)   │   │ ← CSA 层
          │  │  └─ Grouped O Proj   │   │
          │  │ hc_post (1→4 扩展)   │   │
          │  │                       │   │
          │  │ hc_pre (4→1 混合)    │   │
          │  │ RMSNorm              │   │
          │  │ MoE (384专家, top-6) │   │
          │  │  ├─ Hash路由 (前3层) │   │
          │  │  ├─ 分数路由 (后58层)│   │
          │  │  ├─ SwiGLU + FP4     │   │
          │  │  └─ Shared Expert    │   │
          │  │ hc_post (1→4 扩展)   │   │
          │  └──────────────────────┘   │
          └──────────────┬──────────────┘
                         │
                    HC Head (4→1)
                         │
                      Logits
```

### 1.4 五大核心创新

| 创新 | 论文名称 | 代码对应 |
|------|---------|---------|
| 低秩 KV 联合压缩注意力 | **MLA** (Multi-head Latent Attention) | `Attention` 类 |
| 混合稀疏/密集注意力 | **CSA + HCA** (Compressed / Heavily Compressed Attention) | `Compressor` + `Indexer` + `sparse_attn` |
| 多通道可学习残差路由 | **mHC** (Manifold-Constrained Hyper-Connections) | `hc_pre` + `hc_post` + `hc_split_sinkhorn` |
| 384 专家 + 1 共享专家 | **DeepSeekMoE** | `MoE` + `Gate` + `Expert` |
| FP8/FP4 混合精度 | Block-wise Quantization | `act_quant` + `fp8_gemm` + `fp4_gemm` |

---

## 2. Hyper-Connections 详解

### 2.1 动机

标准残差连接 `y = x + f(Norm(x))` 在 61 层深 + 384 专家的超大规模模型下面临挑战：
- 深层梯度沿残差链路衰减
- MoE 层输出方差大，直接加回残差流可能破坏已有信息
- 单一信息通道限制了层间交互的灵活性

**Hyper-Connections 维护 `hc_mult=4` 条并行的隐藏状态副本，每层通过学习决定如何混合它们。**

### 2.2 整体数据流

```python
# Transformer.forward
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [b,s,1,d] → [b,s,4,d]

# Block.forward
def forward(self, x, start_pos, input_ids):
    # x: [b, s, 4, 7168]

    # --- Attention 子层 ---
    residual = x                           # 保存 4 份残差
    x, post, comb = self.hc_pre(x, ...)   # 4→1 混合 + 生成 post/comb
    x = self.attn_norm(x)                  # 对 1 份做 RMSNorm
    x = self.attn(x, start_pos)           # Attention（输入输出都是 1 份）
    x = self.hc_post(x, residual, post, comb)  # 1→4 + 与残差混合

    # --- FFN/MoE 子层 ---
    residual = x
    x, post, comb = self.hc_pre(x, ...)   # 4→1 混合
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)  # 1→4

    return x  # [b, s, 4, 7168]
```

### 2.3 hc_pre：4→1 混合

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # x: [b, s, 4, 7168]
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()                                    # [b, s, 28672]

    # RMS 归一化 → 标准化 mixes 尺度
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)

    # 线性投影 → 24 个标量
    mixes = F.linear(x, hc_fn) * rsqrt   # hc_fn: [24, 28672], mixes: [b, s, 24]

    # Sinkhorn 分解 → pre, post, comb
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                                         sinkhorn_iters=20, eps=1e-6)

    # pre 加权求和：4 份副本 → 1 份
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)     # [b, s, 7168]
    return y, post, comb
```

#### mixes 向量的结构（24 维）

```
mixes[0:4]   → pre  (4 个值): sigmoid → (eps, 1+eps)
mixes[4:8]   → post (4 个值): 2*sigmoid → (0, 2)
mixes[8:24]  → comb (16 个值): softmax + Sinkhorn 迭代 → 4×4 doubly stochastic 矩阵
```

### 2.4 hc_post：1→4 扩展 + 残差混合

```python
def hc_post(self, x, residual, post, comb):
    # x:       [b, s, 7168]        ← 子层输出
    # residual:[b, s, 4, 7168]    ← 子层输入时的 4 份副本
    # post:    [b, s, 4]           ← 输出扩展权重
    # comb:    [b, s, 4, 4]        ← 残差混合矩阵

    # 第 1 项: 子层输出 × post 权重 → 扩展回 4 份
    y = post.unsqueeze(-1) * x.unsqueeze(-2)                    # [b,s,4,7168]

    # 第 2 项: 残差的 4 份副本通过 comb 矩阵混合
    y += torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    return y
```

展开到每个副本 i：

```
y_i = post_i · x                          ← 子层输出贡献
    + comb_{i,0} · residual_0             ← 来自残差副本 0
    + comb_{i,1} · residual_1
    + comb_{i,2} · residual_2
    + comb_{i,3} · residual_3
```

### 2.5 Sinkhorn 迭代

确保 `comb` 矩阵是 **doubly stochastic**（行列和均为 1），信息总量守恒。

实际 kernel (`hc_split_sinkhorn_kernel`, `kernel.py:371-427`) 的逻辑：

```python
# comb 原始值: hc_fn 线性投影得到的 [hc, hc] = [4, 4] 矩阵

# 初始化: comb = softmax(dim=-1) + eps
# 使用数值稳定的 softmax (max-subtract + exp + row-sum-normalize)
row_max = comb.max(dim=1)              # 每行最大值
comb = exp(comb - row_max)             # 数值稳定 exp
comb = comb / comb.sum(dim=1) + eps    # 行归一化 + eps (保证 >0)

# 列归一化
comb = comb / (comb.sum(dim=0) + eps)  # 列和 → 1

# Sinkhorn 迭代 (共 sinkhorn_iters-1 = 19 轮, 交替行列归一化)
for _ in range(self.hc_sinkhorn_iters - 1):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)   # 行和 → 1
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)   # 列和 → 1
```

**Sinkhorn 定理**保证交替行列归一化收敛到 doubly stochastic 矩阵：
- 每行和 = 1：每个新副本接受的总残差量守恒
- 每列和 = 1：每个旧副本被完整"分发"
- `+eps` 保证 `comb > 0`（严格正矩阵），满足 Sinkhorn 收敛条件

### 2.6 最终 Head 的 HC

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    # x: [b, s, 4, 7168] → 需要坍缩回 [b, s, 7168]
    x = x.flatten(2).float()                             # [b, s, 28672]
    rsqrt = torch.rsqrt(x.square().mean(-1) + eps)
    mixes = F.linear(x, hc_fn) * rsqrt                   # hc_fn: [4, 28672]
    pre = sigmoid(mixes * hc_scale + hc_base) + eps      # [b, s, 4]
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y  # [b, s, 7168]
```

Head 的 HC 只需要 `pre`（不需要 post 和 comb，不需要 Sinkhorn），只需要决定如何合并 4 份副本。

### 2.7 与传统残差对比

```
标准残差:
  x_new = x_old + f(Norm(x_old))     ← 一条线性的信息通路

Hyper-Connections (hc=4):
  阶段1 (hc_pre):   x_mixed = Σ_i pre_i · x_i         ← 4→1 (内容选择)
  阶段2:            x_sub = f(Norm(x_mixed))           ← 子层计算
  阶段3 (hc_post):  x_new_i = post_i · x_sub           ← 1→4 (输出分配)
                           + Σ_j comb_{i,j} · x_old_j  ← 4→4 (残差重分配)
```

---

## 3. Hyper-Connections 的计算开销

### 3.1 每层 HC 总开销

| 操作 | FLOPs (per token) |
|------|-------------------|
| hc_pre: flatten + RMS 归一化 | ~57K |
| hc_pre: 线性投影 `[28672]→[24]` | ~1.38M |
| hc_pre: Sinkhorn 迭代 (20轮, 4×4) | ~640 |
| hc_pre: pre 加权求和 | ~57K |
| hc_post: post 输出扩展 | ~29K |
| hc_post: comb 矩阵混合残差 | ~230K |
| **每子层 HC 合计** | **~1.7M** |
| **每层（2 子层）合计** | **~3.4M** |

### 3.2 与主干计算量对比

| 组件 | FLOPs (per token) |
|------|-------------------|
| MLA Q 投影 | ~220M |
| MLA KV 投影 | ~7M |
| MLA O 投影 | ~100M |
| Sparse Attention | ~50M |
| MoE (6 专家 + 1 共享) | ~930M |
| **主干合计** | **≈ 1.3G** |
| **HC 开销** | **≈ 3.4M (0.26%)** |

### 3.3 为什么开销这么低？

HC 把混合决策压缩到极低维空间：

```
隐藏状态: [b, s, 4, 7168] → flatten → [b, s, 28672]
                ↓
          hc_fn [24, 28672]        ← 瓶颈：28672→24（输出维度仅 24）
                ↓
          mixes: [b, s, 24]        ← 仅 24 个标量
                ↓
    pre [4] + post [4] + comb [4×4]  ← 控制信号
                ↓
    用这些标量去 scale/sum 高维张量  ← 计算量 O(d)，不是 O(d²)
```

24 个标量决定 28672 维向量的命运——输出维度远小于输入维度，FLOPs 天然极低。

### 3.4 参数开销

| 参数 | 形状 | 用途 |
|------|------|------|
| `hc_attn_fn` | `[24, 28672]` | Attn 混合投影 |
| `hc_attn_base` | `[24]` | Attn bias |
| `hc_attn_scale` | `[3]` | Attn pre/post/comb 缩放 |
| `hc_ffn_fn` | `[24, 28672]` | FFN 混合投影 |
| `hc_ffn_base` | `[24]` | FFN bias |
| `hc_ffn_scale` | `[3]` | FFN pre/post/comb 缩放 |

每层 HC 参数 ≈ 1.38M。61 层总计 ≈ 84M——相对于总模型规模（数百 B）可忽略。

---

## 4. CSA 与 HCA 概述

### 4.1 问题背景

传统因果注意力 O(N²)，百万 token 不可接受。DeepSeek V4 使用**混合注意力**：不同层用不同类型的注意力。

### 4.2 三种注意力类型（由 `compress_ratios` 决定）

`config.json` 中 61 层的压缩率：
```
[128, 128, 4, 128, 4, 128, 4, ..., 128, 4, 0]
```

| 类型 | compress_ratio | Indexer | 注意力方式 | 论文名称 |
|------|---------------|---------|-----------|---------|
| ratio=4 + 有 Indexer | 4 | ✅ | 稀疏 (top-1024) | **CSA** |
| ratio=128 + 无 Indexer | 128 | ❌ | 密集 | **HCA** |
| ratio=0 | 0 | ❌ | 纯滑动窗口 (128) | SWA |

```python
# 代码中的分发逻辑 —— __init__ (model.py:466-471)
# 每层根据 compress_ratio 决定是否创建 Compressor 和 Indexer
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:        # ← CSA: ratio=4 → 有 Indexer
        self.indexer = Indexer(args, self.compress_ratio)
    else:                                # ← HCA: ratio=128 → 无 Indexer
        self.indexer = None

# 代码中的分发逻辑 —— forward (model.py:508-514)
# 每层的 attention 计算中，根据 indexer 是否存在，选择不同的压缩索引策略
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win  # prefill 时 offset=seqlen
    if self.indexer is not None:                     # CSA: Indexer 选 top-k
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                                            # HCA: 返回所有压缩块的索引
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

### 4.3 核心区别

| 维度 | CSA | HCA |
|------|-----|-----|
| 压缩率 | m = 4 | m' = 128 |
| Overlap 压缩 | ✅ | ❌ |
| Indexer | ✅ 选 top-k | ❌ 无 |
| 注意力 | **稀疏** (top-1024) | **密集** (attend 全部压缩 KV) |
| 每 query attend 位置数 | 128(swa) + 1024(compressed) ≈ 1152 | 128(swa) + N/128(compressed) |

### 4.4 共享基础：Token-Level Compressor

CSA 和 HCA 都先将连续的 m 个 token 的 KV 压缩为 1 个：

```
C = x · W_KV          # 原始 KV entries
Z = x · W_Z           # 压缩权重
S = Softmax_row(Z + B) # B 是可学习位置偏置
C_Comp = Σ S ⊙ C      # 加权池化
```

### 4.5 混合架构的交错策略

```
层 0-1:  HCA (m'=128) — 粗粒度扫全局
层 2:    CSA (m=4)    — 细粒度 + 稀疏选择
层 3:    HCA (m'=128)
层 4:    CSA (m=4)
...
层 60:   SWA (m=0)    — 纯滑动窗口，保局部精度
```

### 4.6 效率收益

| | KV cache 相对 baseline | 说明 |
|---|---|---|
| CSA | ~0.08× | 4:1 压缩 × FP8 量化 |
| HCA | ~0.03× | 128:1 压缩 × FP8 量化 |
| 整体 | ↓90%+ | 混合使用 |

---

## 5. CSA 逐步骤详解（完整 Shape 追踪）

以 DeepSeek-V4-Pro、prefill 阶段（`start_pos=0`）、单 TP rank 为例。

**输入**: `x = [1, 4096, 7168]`

### 第一阶段：Q 的生成

#### Step 1.1: Q 降维投影

```python
qr = self.q_norm(self.wq_a(x))
#  wq_a:   [1, 4096, 7168] @ [7168, 1536]^T  =  [1, 4096, 1536]
#  q_norm: [1, 4096, 1536]
# → qr = [1, 4096, 1536]          ← 压缩查询向量，后续复用
```

#### Step 1.2: Q 升维 + 拆头 + QK 归一化

```python
q = self.wq_b(q).unflatten(-1, (128, 512))
#  wq_b:      [1, 4096, 1536] @ [1536, 65536]^T  =  [1, 4096, 65536]
#  unflatten: [1, 4096, 128, 512]

q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
#  每 head 内独立 RMS 归一化
# → q = [1, 4096, 128, 512]
```

#### Step 1.3: 部分 RoPE（仅最后 64 维）

```python
apply_rotary_emb(q[..., -64:], freqs_cis)
#  q_nope = [1, 4096, 128, 448]    ← 不变
#  q_rope = [1, 4096, 128, 64]     ← 施加 RoPE
# → q = [1, 4096, 128, 512]
```

### 第二阶段：KV 的生成

#### Step 2.1: KV 投影 + 归一化

```python
kv = self.wkv(x)
#  wkv: [1, 4096, 7168] @ [7168, 512]^T  =  [1, 4096, 512]
#  不区分 K 和 V → Shared KV MQA

kv = self.kv_norm(kv)            # KV 归一化
# → kv = [1, 4096, 512]
```

#### Step 2.2: 部分 RoPE + Nope 量化

```python
apply_rotary_emb(kv[..., -64:], freqs_cis)            # rope 部分旋转
act_quant(kv[..., :-64], 64, "ue8m0", fp8, inplace=True)  # nope 部分 FP8 模拟量化
#  kv_nope: [1, 4096, 448]  ← quant+dequant, 每 64 维共享 2^x scale
#  kv_rope: [1, 4096, 64]   ← 保持 BF16
# → kv = [1, 4096, 512]
```

### 第三阶段：构建注意力索引（topk_idxs）

#### Step 3.1: 滑动窗口索引

这里的"topk"一词有误导性——**滑动窗口内部不做稀疏选择，是密集的（全量关注）**。
函数名 `get_window_topk_idxs` 是因为其返回格式与 Indexer 返回的 `compress_topk_idxs` 同为 `[b, seqlen, k]` 的索引矩阵，
最终通过 `torch.cat` 合并给 `sparse_attn` kernel 统一处理。

##### 窗口大小

窗口大小来自 `config.json` 的 `window_size = 128`。每个 query token **始终密集关注最近 ≤128 个历史 token 的原始 KV**（未经压缩）。

```
对于 query token t:
  t=0:    关注 [0]                             (1 个 token，自身)
  t=5:    关注 [0,1,2,3,4,5]                   (6 个 token)
  t=100:  关注 [0,1,2,...,100]                  (101 个 token，不足 128)
  t=500:  关注 [373,374,...,500]                (刚好 128 个不同 token)
  t=4095: 关注 [3968,3969,...,4095]             (128 个 token)
```

##### Prefill 阶段的索引生成

```python
# model.py:254-265, prefill (start_pos=0)
base = torch.arange(seqlen).unsqueeze(1)                      # [4096, 1]
matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
# [max(0, t-127), max(0, t-127)+1, ..., t]  ← 因果滑动窗口

# 前几个 token 不足 128 位置时，开头位置会重复:
# t=0:  [0, 0, 0, ..., 0]      ← 128 个全部是 0 (只有一个有效)
# t=5:  [0, 0, ..., 0, 1, 2, 3, 4, 5]  ← 前 123 个是 0 (重复)

matrix = torch.where(matrix > base, -1, matrix)  # 把超过当前位置的重复值标为 -1
# t=5:  [0, -1, -1, ..., -1, 1, 2, 3, 4, 5]   ← 只保留第一个 0

# → topk_idxs = [1, 4096, 128]
#   有效值: [0, t] 的整数, 无效值: -1 (sparse_attn kernel 会跳过)
```

##### Decode 阶段的索引生成

```python
# decode (start_pos > 0): 只处理 1 个 query
# 利用循环缓冲区语义 — start_pos 映射到 kv_cache 中 [0, win) 的某个偏移
if start_pos >= window_size - 1:
    start_pos %= window_size
    # 例 start_pos=500, 500%128=116
    # → [117,118,...,127, 0,1,...,116]   — 循环顺序
    matrix = torch.cat([torch.arange(start_pos+1, window_size),
                        torch.arange(0, start_pos+1)], dim=0)
else:
    matrix = F.pad(torch.arange(start_pos+1), (0, window_size-start_pos-1), value=-1)

# → topk_idxs = [1, 1, 128]   ← 单 query 在 kv_cache 的循环缓冲位置
```

##### 为什么需要滑动窗口？

CSA 的压缩注意力有一个根本性的信息盲区：**每个 query 只能 attend 到"已经完成的压缩块"，不能 attend 到同一压缩块内的其他 token**。

```
压缩块 1: tokens [4,5,6,7]
Query token 7: 属于压缩块 1，但块 1 要到 token 7 处理完后才被写入 kv_cache
  → query 7 通过压缩 KV 只能看到块 0 (tokens [0,1,2,3])
  → query 7 看不到 tokens [4,5,6] — 它们和 token 7 在同一压缩块内
```

滑动窗口填补了这片空白：token 7 通过滑动窗口可以密集关注 [0,1,2,3,4,5,6,7] 全部最近 128 个 token，
包括自己的 block 内 token [4,5,6]。

```
query 7 的注意力来源:
  ┌─ 滑动窗口:  [0,1,2,3,4,5,6,7]  ← 原始 KV，全量密集，填补同块盲区
  └─ 压缩 KV:   块 0               ← 4:1 pooling，粗粒度远距离信息
      (块 1 不可见—它包含 query 7 自身)
```

**对比**：HCA（`compress_ratio=128`）的同块盲区更大——query token 可能看不到前 127 个 token，
所以同样需要滑动窗口来保证近距离的精细依赖不被压缩丢失。

```python
topk_idxs = get_window_topk_idxs(win=128, bsz=1, seqlen=4096, start_pos=0)
```

#### Step 3.2: Indexer 选择压缩 KV 的 top-k

##### 3.2a: Indexer Q 生成

```python
q = self.wq_b(qr)                          # [1, 4096, 1536] → [1, 4096, 8192]
q = q.unflatten(-1, (64, 128))             # → [1, 4096, 64, 128]
apply_rotary_emb(q[..., -64:], freqs_cis)   # 部分 RoPE
q = rotate_activation(q)                   # Hadamard 旋转
fp4_act_quant(q, 32, inplace=True)         # FP4 量化
# → q = [1, 4096, 64, 128]
```

##### 3.2b: Indexer 压缩 KV cache 构建

Indexer 拥有独立的 `Compressor`（`head_dim=128`, `compress_ratio=4`, `rotate=True`），其 `prefill` 阶段（`start_pos=0`）的完整逻辑：

```python
# Compressor.__init__ (model.py:283-305)
# self.wkv   = Linear(dim=7168, coff*head_dim=256)   # coff = 1+overlap = 2
# self.wgate = Linear(dim=7168, coff*head_dim=256)
# self.ape   = Parameter([compress_ratio, coff*head_dim]) = [4, 256]
# self.norm  = RMSNorm(head_dim=128)

# --- Compressor.forward (model.py:316-377), start_pos=0 ---
kv    = self.wkv(x)         # [1, 4096, 7168] @ [7168, 256]^T → [1, 4096, 256]
score = self.wgate(x)       # [1, 4096, 7168] @ [7168, 256]^T → [1, 4096, 256]

# 计算是否需要压缩（至少需要 ratio 个 token）
should_compress = seqlen >= ratio  # 4096 >= 4 → True
remainder = seqlen % ratio         # 4096 % 4 = 0
cutoff = seqlen - remainder        # 4096 - 0 = 4096

# 当 overlap=True 且 cutoff >= ratio 时，保存最后 ratio 个 token 的 KV/score
# 到 state buffer，供后续 decode 阶段的 overlap 使用
if overlap and cutoff >= ratio:  # True
    # kv[:, cutoff-ratio : cutoff] = kv[:, 4092:4096]  → 最后 4 个 token
    self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]            # [1, 4, 256]
    self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape  # [1, 4, 256]

# 由于 remainder=0，跳过 remainder 分支
# (当 seqlen%ratio≠0 时，尾部不足 ratio 个 token 会被暂存到 state，不参与本次压缩)

# 正常压缩部分 (cutoff 个 token = 4096 个)
kv    = kv[:, :cutoff]       # [1, 4096, 256]（remainder=0 时相同）
score = score[:, :cutoff]    # [1, 4096, 256]

kv    = kv.unflatten(1, (-1, ratio))           # → [1, 1024, 4, 256]
score = score.unflatten(1, (-1, ratio)) + self.ape  # [1, 1024, 4, 256] + [4, 256]

# overlap_transform: [1, 1024, 4, 256] → [1, 1024, 8, 128]
# 前 4 个 position (128d) ← 来自前一个 block 的 overlap 半通道
# 后 4 个 position (128d) ← 来自当前 block 的 normal 半通道
kv    = self.overlap_transform(kv, fill_value=0)
score = self.overlap_transform(score, fill_value=float("-inf"))

# 逐维度 softmax 加权池化
kv = (kv * score.softmax(dim=2)).sum(dim=2)    # [1, 1024, 8, 128] → [1, 1024, 128]

# 后处理（仅当 should_compress=True 时执行）
kv = self.norm(kv)                             # RMSNorm → [1, 1024, 128]
apply_rotary_emb(kv[..., -rd:], freqs_cis)      # rd = rope_head_dim = 64
kv = rotate_activation(kv)                      # Hadamard 旋转
fp4_act_quant(kv, fp4_block_size=32, inplace=True)  # FP4 模拟量化

# 最后写入 Indexer 的 kv_cache（在 Indexer.forward 中完成，此处 Compressor 返回 kv）
# → kv_cache_indexer[:bsz, :1024] = kv    = [1, 1024, 128]
```

> **注意**：Compressor.forward 在 `should_compress=False` 时返回 `None`（不压缩）；`should_compress=True` 时返回压缩后的 kv 并同时写入 `self.kv_cache`。Indexer.forward 将 `self.kv_cache` 指向 Indexer 自己的 buffer，所以 Compressor 写入的是 Indexer 专用的 cache。

##### 3.2c: 计算 Index Score

```python
weights = self.weights_proj(x) * (1/√128 * 1/√64)
#  weights_proj: [1, 4096, 7168] @ [7168, 64]^T = [1, 4096, 64]
# → weights = [1, 4096, 64]

index_score = einsum("bshd,btd->bsht", q, kv_cache_indexer)
#  q:         [1, 4096, 64, 128]
#  kv_cache:  [1, 1024, 128]
# → [1, 4096, 64, 1024]

# ReLU + 加权聚合
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
# → [1, 4096, 1024]
```

##### 3.2d: 因果掩码 + Top-k 选择 + 二次过滤

```python
# idxs_to_compress_blocks 的最后一维是 end_pos // ratio 个压缩块
# prefill (start_pos=0, seqlen=4096):   end_pos // ratio = 4096 // 4 = 1024
# decode (start_pos=4096, seqlen=1):    end_pos // ratio = 4097 // 4 = 1024
# 长上下文 (start_pos=0, seqlen=16384): end_pos // ratio = 16384 // 4 = 4096

# 第一步: 因果掩码 —— query t 只能看到 ≤ floor(t/4) 的压缩块
if start_pos == 0:
    mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= \
           torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    # mask[t, i] = (i >= (t+1)//4) ? True : False    shape: [seqlen, seqlen//ratio]
    index_score += torch.where(mask, float("-inf"), 0)

# 第二步: Top-k —— 从所有可见候选块中选分数最高的 k 个
# k = min(self.index_topk, end_pos // ratio)
#     prefill seqlen=4096:  min(1024, 4096//4) = 1024  → 全选（候选=可选=1024）
#     prefill seqlen=16384: min(1024, 16384//4) = 1024  → 从 4096 个中选 1024
#     百万token decode:     min(1024, 250000) = 1024    → 从 25 万个中选 1024
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
# → [seqlen, k]    k 随上下文长度动态变化

# 第三步: 二次过滤 + 索引平移
# topk 面对全 -inf 时行为不确定，可能选出不应该可见的块
# 二次 mask 确保所有越界的块被标记为 -1
if start_pos == 0:
    mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    # mask[t, k] = True 当压缩块 topk_idxs[t,k] 对 query t 不可见
    topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    # offset = kv.size(1) = seqlen (滑动窗口 KV 的数量)
    # -1:  无效位置（因果性屏蔽），sparse_attn kernel 内部会跳过
    # +offset: 把压缩块索引平移到 kv_cache 中 [window_size:] 之后的实际位置
else:
    topk_idxs += offset
    # decode 阶段不需要 mask（所有历史块均已可见）

# → compress_topk_idxs = [1, seqlen, k]
#   prefill seqlen=4096: [1, 4096, 1024]
#   有效值范围: [offset, offset + end_pos//ratio)，-1 表示无效
```

#### Step 3.3: 合并索引

```python
# 合并滑动窗口索引 + 压缩 KV 索引
# topk_idxs 在 step 3.1 中已赋值为 SWA 的 [1, 4096, 128]
compress_full = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
#  swa_topk:           [1, 4096, 128]      ← token 级别索引，值 [0, seqlen)
#  compress_topk:      [1, 4096, k]        ← 压缩块索引（经过 offset 平移），值 [offset, ...) 或 -1
# → compress_full = [1, 4096, 128 + k]     ← prefill seqlen=4096 时 = [1, 4096, 1152]

topk_idxs = compress_full   # 后续 sparse_attn 使用此合并后的索引
# 对于 HCA (ratio=128): get_compress_topk_idxs 返回所有可见压缩块的密集索引
#   → [1, 4096, 128 + seqlen//128]
```

#### 3.4 滑动窗口 + 压缩 KV 的分工设计

CSA 和 HCA 的注意力索引由**两部分拼接**而成：

```
每个 query 最终的注意力来源:
  ┌─ 滑动窗口 KV (128 个)  — 原始 token 级 KV，密集关注，不压缩
  └─ 压缩 KV (k 个)        — 压缩块级 KV，可选稀疏(top-k)或密集(全量)
```

##### 为什么需要两套 KV？

压缩 KV 有一个**因果性盲区**：

```
在 prefill 阶段，每个压缩块要等它包含的最后一个 token（即 block 内的 token 4n+3）被处理完后，
才会被 Compressor 压缩并写入 kv_cache。在此之前，同一压缩块内的其他 query token 
无法访问这个压缩块——因为它们"在块里面"。

例如，ratio=4, query=6:
  - query 6 属于 block 1 (covers tokens [4,5,6,7])
  - block 1 还没写完（要等 token 7 才完成压缩）
  - query 6 通过压缩 KV 只能看到 block 0 (tokens [0,1,2,3])
  - query 6 看不到 tokens [4,5] — 它们在因果性上已经过去但还没形成压缩块!

滑动窗口填补了这片空白:
  - query 6 通过滑动窗口密集关注 tokens [0,1,2,3,4,5,6]
  - tokens [4,5] 通过原始 KV（未压缩）可见
```

对于 HCA（`ratio=128`），这个盲区更大——token 可能看不到前 127 个 token 的压缩块。
滑动窗口确保近距离依赖始终无损。

##### 两部分索引的合并

```python
# Attention.forward (model.py:507-514):
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)  # SWA
if self.compress_ratio:
    if self.indexer is not None:    # CSA: Indexer 选 top-k
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:                           # HCA: 返回所有可见压缩块
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

两部分索引的语义区分：

| 索引来源 | 语义 | 索引值范围 | 位置含义 |
|---------|------|-----------|---------|
| `get_window_topk_idxs` | 原始 token 位置 | `[0, seqlen)` | kv_cache 的 `[0, win)` 区（循环缓冲） |
| `compress_topk_idxs` (CSA) | Indexer 选出的压缩块 | `[offset, offset+end//ratio)` 或 `-1` | kv_cache 的 `[win, win+compressed]` 区 |
| `compress_topk_idxs` (HCA) | 所有可见压缩块（密集） | `[offset, offset+end//ratio)` 或 `-1` | 同上 |

CSA 和 HCA 的区别仅在于压缩部分——滑动窗口部分始终是 128 个密集位置。

##### sparse_attn kernel 如何使用合并索引

```python
# model.py:528 (prefill) / model.py:533 (decode)
o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
```

`sparse_attn` kernel 对传入的所有位置**一视同仁**——不管来自 SWA 还是压缩 KV，
它根据 `topk_idxs` 从 `kv_cache` 中 gather 对应位置计算标准注意力。
"sparse" 指的是 gather-based 取址（跳过 `-1` 位置），而非 dense matmul。
**SWA 部分在取址上是密集的**（128 个连续位置全部取出），只有压缩部分在 CSA 时是稀疏的（1024 out of thousands）。

##### 量化精度差异

SWA 的 KV 和压缩 KV 在量化精度上也不同：

| KV 类型 | rope 部分 (64d) | nope 部分 (448d) | 压缩方式 |
|---------|----------------|------------------|---------|
| SWA KV | BF16 | FP8 (block_size=64, ue8m0) | 不压缩，只保留 128 个最近 token |
| 主 Compressor 压缩 KV | BF16 | FP8 (block_size=64, ue8m0) | 4:1 or 128:1 pooling |
| Indexer 压缩 KV | FP4 (Hadamard+quant) | FP4 (Hadamard+quant) | 4:1 pooling |

#### 3.5 Prefill vs Decode 完整对比

Attention.forward 在 prefill 和 decode 阶段走的是**完全不同的路径**。理解这两条路径的区别是理解 CSA 的关键。

##### Prefill 路径（start_pos == 0）

```python
# model.py:518-528
# --- 步骤 A: SWA KV 写入 kv_cache（只保留最近 128 个 token 给后续 decode）---
if seqlen <= win:
    self.kv_cache[:bsz, :seqlen] = kv                           # seqlen<=128 时全量写入
else:
    cutoff = seqlen % win                                        # 循环缓冲对齐
    self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
        kv[:, -win:].split([win-cutoff, cutoff], dim=1)         # 只写最后 128 个 token

# --- 步骤 B: 主 Compressor 压缩（同时写入 kv_cache 压缩区 + 返回）---
if self.compress_ratio:
    if (kv_compress := self.compressor(x, start_pos)) is not None:
        kv = torch.cat([kv, kv_compress], dim=1)
        # kv: [1, seqlen, 512] + [1, seqlen//ratio, 512]
        # → [1, seqlen + seqlen//ratio, 512]  ← 本地变量，包含全部 raw KV！
        # kv 没有被截断！因为每个 query 的窗口范围不同，需要不同的 raw KV 段:
        #   query=100  → 需要 raw KV[0:101]
        #   query=500  → 需要 raw KV[373:501]
        #   query=8000 → 需要 raw KV[7873:8001]
        # 如果 kv_cache 只保留 128 个，query=100 需要的 raw KV 早已被覆写

# --- 步骤 C: sparse_attn 用本地变量 kv ---
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
#                    ↑
#               [1, seqlen+seqlen//ratio, 512]  ← 全部 raw + 全部压缩
#               用完即弃，不持久化
```

**关键区分**：

| | `self.kv_cache` | 本地变量 `kv` |
|---|---|---|
| 何时使用 | 后续 decode 步骤 | 仅本次 prefill |
| raw KV 大小 | 固定 128 条 | 全部 seqlen 条 |
| 生命周期 | 跨步持久 | 单次 forward 用完释放 |
| 写入者 | Prefill 步骤 A（SWA）+ Compressor（压缩区） | Compressor 拼接 |

##### Decode 路径（start_pos > 0）

```python
# model.py:529-533
# --- 步骤 A: SWA KV 写入 kv_cache（循环覆写）---
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)   # 一次写 1 个新 token
# 例 start_pos=500, 500%128=116 → kv_cache[0,116] = 新 token
# 覆写了最旧的 token（start_pos-128 位置的 token）

# --- 步骤 B: 主 Compressor（增量压缩，可能返回 None）---
if self.compress_ratio:
    self.compressor(x, start_pos)     # 不接返回值！
    # 每 ratio 个 token 才产生 1 个新压缩块 → 写入 kv_cache 压缩区
    # 未凑够时返回 None（token 暂存 kv_state）

# --- 步骤 C: sparse_attn 从 kv_cache 读取 ---
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
#                    ↑
#               [1, 128 + end_pos//ratio, 512]  ← kv_cache，包含所有历史
#               SWA: 128 个最近 token（循环覆写）
#               压缩: end_pos//ratio 个块（持续追加）
```

##### Prefill vs Decode 对比表

| | Prefill (start_pos=0) | Decode (start_pos>0) |
|---|---|---|
| **输入 x** | `[b, seqlen, 7168]`，seqlen 任意 | `[b, 1, 7168]`，每次 1 token |
| **SWA KV 写入** | 一次写 128 个（最近 token） | 一次写 1 个（循环覆写 `kv_cache[start_pos%128]`） |
| **SWA 索引** | `[max(0,t-127), ..., t]`，不足用 -1 补齐 | 128 个循环缓冲位置，始终全覆盖 |
| **Compressor** | 一次性压缩所有 → 批量写入压缩区 | 每 ratio 步压缩 1 次 → 单条追加 |
| **Compressor 返回值** | 拼接后喂给 sparse_attn | **丢弃**——副作用已写入 kv_cache |
| **sparse_attn kv 来源** | 本地变量 `kv`（全部 raw + 全部压缩） | `kv_cache`（128 SWA + 持续增长的压缩区） |
| **offset（索引平移）** | `offset = seqlen`（raw KV 的数量） | `offset = win = 128`（kv_cache SWA 区大小） |
| **Indexer 候选块** | `end_pos // ratio ≈ seqlen // 4` | 随 decode 增长 → 百万 token 时 = 250,000 |
| **Indexer top-k** | `min(1024, seqlen//ratio)` | `min(1024, end_pos//ratio)` → 百万 token 时 = 1024 |
| **因果 mask** | 需要（prefill 时 query 可能看到未来块） | 不需要（所有历史块均已可见） |

#### 3.6 Compressor 完整计算流程

##### 3.6.1 参数与 Buffer 初始化

以**主 Compressor**（`head_dim=512`, `compress_ratio=4`, `overlap=True`, `rotate=False`）为例。
Indexer Compressor 和 HCA 的差异在括号中标注。

```python
# __init__ (model.py:283-305)
self.dim = 7168
self.head_dim = 512                                            # Indexer: 128
self.rope_head_dim = 64                                        # 从 args 继承
self.nope_head_dim = 512 - 64 = 448                            # Indexer: 128-64=64
self.compress_ratio = 4                                        # HCA: 128
self.overlap = True                                            # HCA: False
self.rotate = False                                            # Indexer: True
coff = 1 + self.overlap                                        # = 2  (HCA: 1)

# 可学习参数
self.ape   = Parameter([ratio, coff*head_dim])                 # [4, 1024]  (HCA: [128, 512])
#   ape[j, :] = position j (0~3) 在每个维度上的固定偏置

self.wkv   = Linear(7168, coff*head_dim, dtype=fp32)           # [7168, 1024] (HCA: [7168, 512])
#   前 512 行 = overlap 半通道权重  (HCA: 无此区分)
#   后 512 行 = normal 半通道权重

self.wgate = Linear(7168, coff*head_dim, dtype=fp32)           # 同上

self.norm  = RMSNorm(head_dim)                                 # RMSNorm(512)  (Indexer: RMSNorm(128))

# State buffer (decode 增量压缩用)
self.kv_state    = zeros([max_batch, coff*ratio, coff*head_dim])
#                   zeros([4, 8, 1024])                         (HCA: zeros([4, 128, 512]))
#   [:, :4, :] = overlap 区 (前一个 block 的 overlap 视图)
#   [:, 4:, :] = 普通区   (当前 block token 的 normal 视图)
self.score_state = full([4, 8, 1024], -inf)
```

##### 3.6.2 Prefill 路径 (start_pos=0)，Overlap=True (CSA)

以 seqlen=4096, ratio=4, cutoff=4096, remainder=0 为例。

**Step 0: 投影**

```python
x = x.float()                              # [1, 4096, 7168] → fp32

kv    = self.wkv(x)                        # [1, 4096, 7168] @ [7168, 1024]^T
                                           # → [1, 4096, 1024]
                                           # 前 512d = overlap 半通道
                                           # 后 512d = normal 半通道

score = self.wgate(x)                      # [1, 4096, 1024]
                                           # 同样: [overlap_512 | normal_512]
```

**Step 1: State buffer 保存（供后续 decode 增量压缩）**

```python
should_compress = seqlen >= ratio           # 4096 >= 4 → True
remainder = seqlen % ratio                  # 0
cutoff    = seqlen - remainder              # 4096
offset    = ratio                           # 4 (overlap 时，普通区的起始偏移)

# 保存最后 ratio 个 token 到 overlap 区（供下一个 decode block 的 overlap 使用）
if overlap and cutoff >= ratio:             # True
    self.kv_state[:bsz, :ratio]    = kv[:, cutoff-ratio:cutoff]
    # kv[:, 4092:4096] = [1, 4, 1024] → kv_state[:, :4, :]
    self.score_state[:bsz, :ratio] = score[:, cutoff-ratio:cutoff] + self.ape
    # [1, 4, 1024] → score_state[:, :4, :]

# remainder=0 → 跳过尾部暂存
```

**Step 2: 分块**

```python
kv    = kv[:, :cutoff]                      # [1, 4096, 1024] (remainder=0 不变)
score = score[:, :cutoff]                   # [1, 4096, 1024]

kv    = kv.unflatten(1, (-1, ratio))        # → [1, 1024, 4, 1024]
#  1024 个 block × 4 token/block × 1024d

score = score.unflatten(1, (-1, ratio)) + self.ape
# [1, 1024, 4, 1024] + [4, 1024] → [1, 1024, 4, 1024]
```

**Step 3: overlap_transform**

```python
# overlap_transform(tensor, fill_value)  (model.py:307-314)
# tensor: [b, blocks, ratio, coff*head_dim] = [1, 1024, 4, 1024]
# fill_value=0 for kv, fill_value=-inf for score

b, num_blocks, _, _ = tensor.shape
new_tensor = tensor.new_full((b, num_blocks, 2*ratio, head_dim), fill_value)
# → [1, 1024, 8, 512]

# 规则 A: normal 半通道 (后 512d) → 自己 block 的 pos [4:8]
new_tensor[:, :, ratio:, :] = tensor[:, :, :, head_dim:]
# 例: block_i.pos_4 = token_{4i+0}.normal_512d

# 规则 B: overlap 半通道 (前 512d) → 下一个 block 的 pos [0:4]
new_tensor[:, 1:, :ratio, :] = tensor[:, :-1, :, :head_dim]
# 例: block_i.pos_0 = token_{4i-4}.overlap_512d
# block_0.pos_0~3 = fill_value (无前驱 block)
```

变换后每个 block 的可视化：

```
block 0 (tokens [0,1,2,3]):
  pos0-3: fill_value           ← kv=0, score=-inf → 压缩权重=0
  pos4:   token0.normal_512
  pos5:   token1.normal_512
  pos6:   token2.normal_512
  pos7:   token3.normal_512

block 1 (tokens [4,5,6,7]):
  pos0:   token0.overlap_512   ← 来自 block 0 的 overlap 视图!
  pos1:   token1.overlap_512
  pos2:   token2.overlap_512
  pos3:   token3.overlap_512
  pos4:   token4.normal_512    ← 自己 block 的 normal
  pos5:   token5.normal_512
  pos6:   token6.normal_512
  pos7:   token7.normal_512
```

**Step 4: 逐维度加权池化**

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
# [1, 1024, 8, 512] * softmax(dim=2) → sum → [1, 1024, 512]
# 对于每个 (block, dimension): 8 个 position 各自贡献 exp(score)/Σexp(score)
```

**Step 5: 后处理**

```python
kv = self.norm(kv.to(dtype))              # RMSNorm → [1, 1024, 512]
freqs_cis = self.freqs_cis[:cutoff:ratio]  # 取每 4 个位置的 RoPE freq → [1024, 32]
apply_rotary_emb(kv[..., -64:], freqs_cis) # 最后 64d 加 RoPE

# 主 Compressor (rotate=False):
act_quant(kv[..., :-64], 64, scale_fmt, scale_dtype, True)  # 前 448d FP8 量化

# Indexer Compressor (rotate=True):
# kv = rotate_activation(kv)               # Hadamard 旋转
# fp4_act_quant(kv, 32, True)              # 全部 128d FP4 量化
```

**Step 6: 写入 + 返回**

```python
self.kv_cache[:bsz, :seqlen // ratio] = kv  # 批量写入压缩区
return kv                                     # 返回 [1, 1024, 512]
```

##### 3.6.3 Prefill 路径，Overlap=False (HCA, ratio=128)

与 CSA 的核心差异——无 overlap/normal 区分，无 overlap_transform：

```python
# Step 0
kv    = self.wkv(x)                        # [7168, 512] → [1, 4096, 512]  ← 只有 512d
score = self.wgate(x)                      # [1, 4096, 512]

# Step 1: 直接分块（coff=1, 无 overlap）
kv    = kv.unflatten(1, (-1, 128))         # [1, 32, 128, 512]
score = score.unflatten(1, (-1, 128)) + self.ape  # [1, 32, 128, 512] + [128, 512]

# Step 2: 无 overlap_transform → 跳过

# Step 3: 128 个 token 逐维度加权池化
kv = (kv * score.softmax(dim=2)).sum(dim=2) # [1, 32, 128, 512] → [1, 32, 512]

# Step 4-5: 同上 → norm → RoPE → FP8 → 写入 → 返回
```

##### 3.6.4 Decode 路径 (start_pos > 0)，Overlap=True (CSA)

**A. 未凑够一个完整 block（连续 ratio-1 步）**

```python
# 例: start_pos=4096, ratio=4, overlap=True
seqlen = 1
kv    = self.wkv(x)                        # [1, 1, 1024]
score = self.wgate(x)                      # [1, 1, 1024]

should_compress = (4096+1) % 4 == 0        # 1 != 0 → False

score += self.ape[start_pos % ratio]       # ape[0]: position 0 的偏置

# 暂存到普通区
self.kv_state[:bsz, ratio + start_pos % ratio]    = kv.squeeze(1)
# kv_state[:, 4+0] → [1024]
self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)

# 不压缩 → 返回 None！
```

连续 3 次暂存后 state 状态：

```
kv_state[:, 4, :] = token_4096 (1024d)
kv_state[:, 5, :] = token_4097 (1024d)
kv_state[:, 6, :] = token_4098 (1024d)
kv_state[:, 7, :] = token_4099 (1024d)  ← 凑够!
```

**B. 凑够一个完整 block（每 ratio 步触发一次）**

```python
# start_pos=4099, ratio=4
should_compress = (4099+1) % 4 == 0         # True

# 最后一步的 token 先写入普通区
score += self.ape[4099 % 4]                  # ape[3]
self.kv_state[:bsz, 4+3, :]    = kv.squeeze(1)
self.score_state[:bsz, 4+3, :] = score.squeeze(1)

# 合并 overlap 区 + 普通区
kv_state = torch.cat([
    self.kv_state[:bsz, :4, :d],             # overlap 区: 前 4pos, 取前 512d
    self.kv_state[:bsz, 4:, d:]              # 普通区: 后 4pos, 取后 512d
], dim=1)
# → [1, 8, 512]
#   pos0-3: 上一个 block 的 overlap 视图
#   pos4-7: tokens [4096,4097,4098,4099] 的 normal 视图

score_state = torch.cat([
    self.score_state[:bsz, :4, :d],
    self.score_state[:bsz, 4:, d:]
], dim=1)
# → [1, 8, 512]

# 加权池化
kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
# → [1, 1, 512]    ← 1 个新压缩块

# 滚动 buffer: 当前普通区 → 下一个 block 的 overlap 区
self.kv_state[:bsz, :4]    = self.kv_state[:bsz, 4:]
self.score_state[:bsz, :4] = self.score_state[:bsz, 4:]
# 清空普通区，为下一个 block 的 4 个 token 做准备
```

**C. 后处理（与 prefill 相同）**

```python
kv = self.norm(kv.to(dtype))                # [1, 1, 512]
freqs_cis = self.freqs_cis[start_pos+1-ratio].unsqueeze(0)  # 单个位置的 RoPE freq
apply_rotary_emb(kv[..., -64:], freqs_cis)
act_quant(kv[..., :-64], 64, ...)

self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)    # 单条追加
return kv  # [1, 1, 512]
```

##### 3.6.5 Compressor 返回值语义

| 场景 | `should_compress` | 返回值 | 副作用（kv_cache 写入） |
|------|-------------------|--------|------------------------|
| Prefill, seqlen ≥ ratio | True | `[b, seqlen//ratio, head_dim]` | 批量写入压缩区 |
| Decode, 凑够完整 block | True | `[b, 1, head_dim]` | 单条定位写入 |
| Decode, 未凑够 | False | **`None`** | 无（token 暂存 kv_state） |

调用方处理：

```python
# Prefill: 用返回值拼接
if (kv_compress := self.compressor(x, start_pos)) is not None:
    kv = torch.cat([kv, kv_compress], dim=1)    # 拼接后喂给 sparse_attn

# Decode: 丢弃返回值（kv_cache 已被副作用更新）
self.compressor(x, start_pos)                   # 不接返回值
```

#### 3.7 Indexer 与主 Compressor 的关系

##### 3.7.1 两个独立的 Compressor 实例

CSA 层同时持有两个 Compressor，它们**处理同一批 token、用同样的 ratio=4 分块**，但**参数独立、head_dim 不同、量化精度不同**：

```
Attention.compressor (主):
  head_dim=512, coff=2, wkv=[7168,1024], rotate=False
  压缩后: [N/4, 512] 块, BF16(rope)+FP8(nope)
  写入: self.kv_cache[128:, :]    ← 主 kv_cache 的压缩区

Indexer.compressor:
  head_dim=128, coff=2, wkv=[7168,256], rotate=True
  压缩后: [N/4, 128] 块, 全部 FP4 (Hadamard旋转后)
  写入: Indexer 自己的 kv_cache   ← Indexer 独立的 buffer
```

##### 3.7.2 Block 编号对齐保证

两个 Compressor 用同一个输入 `x`、同一个 `ratio=4`、同一个 `unflatten(1, (-1, ratio))`，
分块边界由 token 位置决定，**与 head_dim 无关**：

```
tokens [0,1,2,3]    → block 0  ← 两个 Compressor 覆盖同一组 token
tokens [4,5,6,7]    → block 1
tokens [8192-4, ...] → block 2047
```

Indexer 输出的 raw index (0~2047) 经过 `+offset` 平移后，正好指向主 kv_cache 中同号 block：

```python
# Indexer 输出: raw_idx = 17  → 指向自己的 kv_cache[17] (128d)
# +offset (=seqlen or win)   → 指向主 kv_cache[offset+17] (512d)
# 两者对应同一个 block，覆盖同一组 token
```

##### 3.7.3 两阶段检索设计

Indexer 的 128d 压缩 KV 只用于**近似打分**（判断哪些块相关），不参与核心注意力。
主 Compressor 的 512d 压缩 KV 用于**精确注意力计算**。Indexer 输出整数索引，主注意力按索引 gather：

```
阶段 1 (Indexer):  Q(64heads×128d) · kv_idx(128d) → top-1024 整数索引
阶段 2 (主 Attn):  Q(128heads×512d) · kv_main(512d)[topk_idxs] → 精确注意力
```

**为什么 Indexer 不能用主 Compressor 的 512d KV？**
Indexer Q 只有 128d，和 512d KV 维度不匹配。与其投影 512d→128d，不如直接从 hidden state 学到最优的 128d 压缩——这就是 Indexer 独立 Compressor 存在的理由。

两套权重各自优化各自的目标：Indexer 学习"如何为快速检索编码"，主 Compressor 学习"如何为精确注意力编码"。

##### 3.7.4 对比总表

| | Indexer Compressor | 主 Compressor (CSA) | 主 Compressor (HCA) |
|---|---|---|---|
| **head_dim** | 128 | 512 | 512 |
| **coff** | 2 | 2 | 1 |
| **wkv 输出** | 256d | 1024d | 512d |
| **ape 形状** | `[4, 256]` | `[4, 1024]` | `[128, 512]` |
| **overlap** | ✅ | ✅ | ❌ |
| **rotate** | ✅ | ❌ | ❌ |
| **量化** | 全部 FP4 | nope FP8, rope BF16 | nope FP8, rope BF16 |
| **压缩后块维度** | 128d | 512d | 512d |
| **所属 kv_cache** | Indexer 独立 buffer | 主 `kv_cache[128:]` | 主 `kv_cache[128:]` |
| **对主注意力的贡献** | **输出索引**（选哪些块） | **输出内容**（块的 512d 表示） | **输出内容**（密集关注） |
| **训练信号** | 近似匹配 + 蒸馏 | 语言模型 loss | 语言模型 loss |

#### 3.8 KV Cache 存储分析

##### 3.8.1 SWA 是常数开销

```
SWA 区:  128 tokens × 512d = 65,536 个值  ← 固定，不随 N 增长

N=4096:   SWA 占 128/4096 = 3.1%
N=64K:    SWA 占 128/65536 = 0.2%
N=1M:     SWA 占 128/1M = 0.01%          ← 占比趋近于零
```

##### 3.8.2 压缩节省的计算量

对比三种方案（per layer，主 KV 512d per token/block，不含 Indexer）：

| 方案 | 每层 KV cache | N=4K | N=64K | N=1M |
|------|-------------|------|-------|------|
| 无压缩（MLA raw） | N × 512 | 2M | 32M | 512M |
| **CSA** (ratio=4) | 128×512 + N/4×512 | 0.59M | 8.3M | 128M |
| **HCA** (ratio=128) | 128×512 + N/128×512 | 0.08M | 0.3M | 4.1M |

##### 3.8.3 混合架构 61 层总存储（N=1M）

```
30 CSA layers × (128 + N/4)      = 30 × 250,128   ≈  7.5M 值
29 HCA layers × (128 + N/128)    = 29 × 7,940     ≈  0.23M 值
2  SWA layers × N                 = 2 × 1,000,000  ≈  2.0M 值
                                              Total ≈  9.7M 值

对比无压缩: 61 × 1,000,000 = 61M
减少了 84%
```

SWA 贡献 `61 × 128 = 7,808` 条 KV，在 61M 总量中占比仅 0.01%。

##### 3.8.4 时间分层设计

```
kv_cache 布局:
┌──────────────────────┬──────────────────────────────────┐
│  SWA 区 (128 slots)   │  压缩区 (N/ratio slots)           │
│  ─────────────────    │  ─────────────────────────        │
│  最近 128 token       │  所有历史 token 的压缩表示          │
│  原始 KV, 512d/tok    │  每 ratio 个 token → 1 个 512d     │
│  BF16(rope)+FP8(nope) │  BF16(rope)+FP8(nope)            │
│  循环覆写, 固定大小    │  持续追加, 随上下文增长             │
│  密集关注 (全部128)    │  CSA: sparse top-1024             │
│                        │  HCA: 密集关注 (全部)              │
└──────────────────────┴──────────────────────────────────┘
         ↑                          ↑
    "刚才说了什么"              "很久以前说了什么"
     token级精度                  块级摘要
    O(1) 存储                    O(N/ratio) 存储
```

### 第四阶段：KV 缓存写入与压缩

```python
# Attention.forward (model.py:518-533) —— prefill 分支

# --- 步骤 A: 滑动窗口 KV 写入 (循环缓冲区) ---
if seqlen <= win:   # seqlen=4096 > 128, 走 else 分支
    self.kv_cache[:bsz, :seqlen] = kv
else:
    cutoff = seqlen % win   # cutoff = 4096 % 128 = 0
    # 只保留最近 128 个 token 的 KV，写入 kv_cache 的 SWA 区
    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = \
        kv[:, -win:].split([win - cutoff, cutoff], dim=1)
    # 当 cutoff=0: kv_cache[:128] = kv[:, -128:]  (最近128个token)

# --- 步骤 B: 主 Compressor 压缩 KV ---
if self.compress_ratio:  # CSA: ratio=4
    # Compressor (head_dim=512, ratio=4, coff=2, rotate=False)
    # prefill 阶段: should_compress = (4096 >= 4) = True
    # 压缩过程同 Indexer Compressor, 关键区别:
    #   - head_dim=512 (而非 128), coff*head_dim=1024 (而非 256)
    #   - rotate=False → 不执行 Hadamard 旋转
    #   - 量化: act_quant(FP8) 而不是 fp4_act_quant
    if (kv_compress := self.compressor(x, start_pos)) is not None:
        # kv_compress = [1, 1024, 512]
        # 写入 kv_cache 的压缩区: kv_cache[:bsz, win : win+1024] = kv_compress
        kv = torch.cat([kv, kv_compress], dim=1)
        # kv: [1, 4096, 512] + [1, 1024, 512] → [1, 5120, 512]
        # 前 4096 为原始 SWA KV（其中只有最近 128 个被缓存），
        # 后 1024 为压缩 KV

# prefill 结束时的 kv_cache 布局:
# kv_cache[0,   0:128]  ← 最近 128 个 token 的原始 KV
# kv_cache[0, 128:1152] ← 1024 个压缩 KV 块
# kv_cache.shape = [1, 1152, 512]
```

> **主 Compressor 与 Indexer Compressor 的核心差异**：主 Compressor (`rotate=False`) 量化 nope 部分为 FP8（`act_quant(kv[..., :-64], 64, scale_fmt, scale_dtype, True)`）；Indexer Compressor (`rotate=True`) 对整个向量先做 Hadamard 旋转再量化全部维度为 FP4。Hadamard 旋转使各维度方差均匀，减少 FP4 量化误差。

```python
# 主 Compressor 的压缩后处理 (rotate=False 路径):
kv = self.norm(kv.to(dtype))                            # RMSNorm
apply_rotary_emb(kv[..., -rd:], freqs_cis)               # RoPE 最后 64 维
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)  # nope 部分 FP8

# Indexer Compressor 的压缩后处理 (rotate=True 路径):
kv = self.norm(kv.to(dtype))                            # RMSNorm
apply_rotary_emb(kv[..., -rd:], freqs_cis)               # RoPE 最后 64 维
kv = rotate_activation(kv)                               # Hadamard 旋转
fp4_act_quant(kv, fp4_block_size, True)                  # 全部维度 FP4
```

### 第五阶段：稀疏注意力

```python
o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
#  q:          [1, 4096, 128, 512]
#  kv_cache:   [1, 1152, 512]
#  topk_idxs:  [1, 4096, 1152]
#  attn_sink:  [128]  ← 每 head 可学习的 sink logits
# → o = [1, 4096, 128, 512]
```

Kernel 内部执行在线 softmax（FlashAttention 风格）+ attention sink 偏置。

### 第六阶段：去位置化（反 RoPE）

```python
apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)
#  因为 KV 充当 value，attention output 继承了位置信息
#  反旋转将绝对位置转为相对位置
# → o = [1, 4096, 128, 512]
```

### 第七阶段：分组输出投影

```python
# 分组: 128 heads → 16 groups, 每组 8 heads
o = o.view(1, 4096, 16, 4096)                            # [1, 4096, 16, 4096]

# 组内低秩投影
wo_a = self.wo_a.weight.view(16, 1024, 4096)
o = einsum("bsgd,grd->bsgr", o, wo_a)                    # [1, 4096, 16, 1024]

# 最终投影回 hidden_dim
x = self.wo_b(o.flatten(2))                               # [1, 4096, 7168]
```

### 完整 Shape 流转图（prefill, start_pos=0, seqlen=4096）

```
输入 x: [1, 4096, 7168]
│
├─ Q (主): wq_a→[1,4096,1536]→q_norm
│   qr: [1,4096,1536]  ← 复用
│   ├─→ wq_b→[1,4096,65536]→unflatten→[1,4096,128,512]→QKnorm→RoPE(-64:)
│   │                                                                       
│   └─→ Indexer  ──────────────────────────────────────────────────────────┐
│       wq_b(qr)→[1,4096,8192]→unflatten→[1,4096,64,128]                   │
│       →RoPE(-64:)→Hadamard→FP4(32)                                       │
│       Indexer Compressor (head_dim=128, rotate=True):                     │
│         wkv→[1,4096,256]→unflatten→[1,1024,4,256]                        │
│         wgate→[1,4096,256]→unflatten+ape→[1,1024,4,256]                  │
│         overlap_transform→[1,1024,8,128]                                  │
│         softmax(dim=2)+pool→[1,1024,128]                                  │
│         →norm→RoPE→Hadamard→FP4(32)→ kv_cache[:1024] = [1,1024,128]      │
│       weights_proj(x)→[1,4096,64]                                         │
│       einsum(q·kv^T)→[1,4096,64,1024]→relu+weighted_sum→[1,4096,1024]    │
│       mask(-inf)+topk(min(1024,end//ratio))+2nd_mask(-1)+offset(4096)     │
│         → compress_topk_idxs: [1,4096,k]    (k=1024 when seqlen=4096)     │
│                                                                            │
├─ KV (主): wkv→[1,4096,512]→k_norm→RoPE(-64:)→FP8(nope,64,ue8m0)          │
│                                                                           │
├─ SWA topk: get_window_topk_idxs(win=128)→[1,4096,128]                     │
│                                                                           │
├─ topk_idxs = cat(swa, compress_topk_idxs)→[1,4096,128+k] ≈ [1,4096,1152] │
│                                                                           │
├─ 主 Compressor (head_dim=512, rotate=False):                              │
│     wkv→[1,4096,1024]→...→overlap→pool→norm→RoPE→FP8(nope)               │
│       → kv_cache[128:, :1024] = [1,1024,512]                              │
│   kv = cat(swa_kv, compressed_kv)→[1,5120,512]  (prefill only)            │
│                                                                           │
├─ sparse_attn(q=[1,4096,128,512], kv=[1,5120,512],                          │
│              topk_idxs=[1,4096,1152], attn_sink=[128])                     │
│   → o: [1,4096,128,512]                                                   │
│                                                                           │
├─ de-RoPE(o[...,-64:], inverse=True)→[1,4096,128,512]                      │
│                                                                           │
└─ 输出投影: view(16groups)→einsum·wo_a→wo_b→[1,4096,7168]                  │
```

---

## 6. wq_b 为什么输出 65536 而不是 7168

### 6.1 直接答案

65536 = 128 heads × 512 head_dim。`wq_b` 不是产生"一个向量"，而是同时产生 128 个独立的 query head。

```python
q = self.wq_b(qr)                       # [b,s,1536] → [b,s,65536]
q = q.unflatten(-1, (128, 512))         # → [b,s,128,512]
```

### 6.2 三种投影 Shape 对比

| 投影 | 输入→输出 | 含义 |
|------|----------|------|
| `wq_a` | 7168→1536 | 降维到 Q 潜在空间 |
| `wq_b` | 1536→**65536** | 从潜在空间升维到 128 个 query head（各 512d） |
| `wo_a` | 128×512//16=4096→16×1024=16384 | 分组低秩 O 投影 |
| `wo_b` | 16384→7168 | 最终投影回 hidden_dim |

### 6.3 为什么 Hidden dim 和 Q 输出维度无关

```
隐藏状态 (7168d)  →  残差流，信息在层间传递
Q 向量 (65536d)   →  128 个独立侦查员，仅用于 attention 内部计算
O 投影            →  桥梁：把 attention 输出合并回 7168d 残差流
```

```
hidden:[b,s,7168] ──→ wq_a→wq_b→[b,s,65536]→128 heads→sparse_attn
    │                                                          ↓
    │                                              [b,s,128,512]→de-RoPE
    │                                                          ↓
    └──→ wkv→[b,s,512]→KV cache────→sparse_attn────→分组wo_a→wo_b→[b,s,7168]
```

Q 走 attention 内部路径，残差走 O 投影。两条路径在 O 投影处合流。

### 6.4 MLA 的不对称哲学

```
Q:  7168 → 1536 → 65536  (128 heads, 512d each)   ← 128 个"视角"
KV: 7168 → 512           (1 head, 512d)            ← 1 个共享"快照"
```

- **KV 极度压缩**：只需存 512d per token → KV cache 极小
- **Q 非常丰富**：128 个头从不同角度查询同一个 KV 快照 → 表达能力不降
- **Q 不需要缓存** → 增大 Q 成本仅在 FLOPs，不在显存

### 6.5 为什么是 1536

```
q_lora_rank 越大 → 表达能力越强，但参数越多，瓶颈效果越弱
q_lora_rank 越小 → 压缩率越高，但可能丢失信息

压缩比 = 7168/1536 ≈ 4.67×
Q 参数 = 7168×1536 + 1536×65536 ≈ 111M
直接做 (7168→65536) = 7168×65536 ≈ 470M → 节省 4×
```

---

## 7. Overlap 机制与 Score 详解

### 7.1 为什么需要 Overlap？

**无 overlap 时**：压缩块的边界一刀切，相邻两个 token（如 token 3 和 4）明明只差一个位置，却被分到不同的压缩块，边界处信息断裂。

**有 overlap 时**：每个压缩块不仅包含自己的 4 个 token，还包含前一个 block 的"overlap 视图"，边界被模糊化。

```
原始: [0,1,2,3] [4,5,6,7] [8,9,10,11]
无overlap: C₀=[0,1,2,3]  C₁=[4,5,6,7]  C₂=[8,9,10,11]  ← 边界生硬
有overlap: C₀=[0,1,2,3]  C₁=[0,1,2,3的overlap + 4,5,6,7]  C₂=[4,5,6,7的overlap + 8,9,10,11]
                                              ↑ 平滑过渡
```

### 7.2 Score 的含义

`score = self.wgate(x)` 输出一个与 KV 同维的向量，每个维度编码该维度在该 position 的 **"重要性"**。

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

**逐维度 softmax**：对于 512 个语义维度中的每一个，模型独立学习"在 8 个 position 中，哪个最重要"。

```
C_compressed[dim=d] = Σ_pos  kv[pos, d] × exp(score[pos,d]) / Σ_j exp(score[j,d])
```

### 7.3 为什么 wkv 和 wgate 要分开？

内容和重要性是正交的两个维度：

```
wkv("the"):       [0.1, 0.2, ...]   ← 语义内容平淡
wgate("the"):     [0.8, 0.9, ...]   ← 结构重要性高（冠词标记名词短语开头）

wkv("elephant"):   [0.9, 0.8, ...]  ← 语义丰富
wgate("elephant"): [0.2, 0.1, ...]  ← 结构不重要（只是列举项）
```

两套独立权重让模型能解耦"有什么"和"重要吗"。

### 7.4 ape（位置偏置）的作用

```python
self.ape = Parameter([4, 256])  # ratio=4, coff*head_dim=256
score = score.unflatten(1, (-1, 4)) + self.ape
```

`ape[j,:]` 是跟 position j 绑定的可学习偏置。由于压缩会丢失 token 级别的位置信息，`ape` 在压缩时注入位置偏置，让模型在池化时能顾及位置——例如倾向于给 block 内最后一个 token 更高权重。

### 7.5 overlap_transform 的机械原理

#### 输入结构

`wkv` 输出 1024 维 = 512 (overlap) + 512 (normal)，两个半通道使用**不同的权重**：

```
token_j 的 overlap 表示:  x_j · W_KV[:512, :]     ← 为"跨 block 传递信息"优化
token_j 的 normal 表示:   x_j · W_KV[512:, :]     ← 为"在自己 block 内表示"优化
```

#### 变换过程

以主 Compressor 为例（`head_dim=512`, `coff=2`, `ratio=4`），`wkv` 输出 `coff×512 = 1024d`。

```python
# overlap_transform(tensor, fill_value)  (model.py:307-314)
# tensor: [b, num_blocks, ratio, coff*head_dim] = [1, 1024, 4, 1024]
#  → 1024d = 前512d (overlap半通道) + 后512d (normal半通道)

b, num_blocks, _, _ = tensor.shape
new_tensor = tensor.new_full((b, num_blocks, 2*ratio, head_dim), fill_value)
# new_tensor: [1, 1024, 8, 512]

# 规则 A: normal 半通道 → 自己 block 的 position [ratio:2*ratio] = [4:8]
new_tensor[:, :, ratio:, :] = tensor[:, :, :, head_dim:]
# block_i 的 pos 4-7 ← block_i 的 tokens 的 normal 半通道(后512d)

# 规则 B: overlap 半通道 → 下一个 block 的 position [0:ratio] = [0:4]
new_tensor[:, 1:, :ratio, :] = tensor[:, :-1, :, :head_dim]
# block_i 的 pos 0-3 ← block_{i-1} 的 tokens 的 overlap 半通道(前512d)
# block_0 的 pos 0-3 ← 保持 fill_value (没有前驱 block)
```

可视化（block 0 和 block 1 的变换）：

```
变换前 (unflatten后):
  block 0: [tok0 | tok1 | tok2 | tok3], 每个 1024d = [o(512) | n(512)]
  block 1: [tok4 | tok5 | tok6 | tok7], 每个 1024d = [o(512) | n(512)]

变换后 (每个 block 有 8 个 position, 各 512d):
  block 0 pos 0-3: fill_value (无前驱 block)
  block 0 pos 4:   tok0[n:512]        ← block0 自己的 normal
  block 0 pos 5:   tok1[n:512]
  block 0 pos 6:   tok2[n:512]
  block 0 pos 7:   tok3[n:512]
  
  block 1 pos 0:   tok0[o:512]        ← block0 的 overlap 半通道
  block 1 pos 1:   tok1[o:512]
  block 1 pos 2:   tok2[o:512]
  block 1 pos 3:   tok3[o:512]
  block 1 pos 4:   tok4[n:512]        ← block1 自己的 normal
  block 1 pos 5:   tok5[n:512]
  block 1 pos 6:   tok6[n:512]
  block 1 pos 7:   tok7[n:512]
```

### 7.6 完整示例：Token 3 和 Token 7 的命运（主 Compressor）

以 12 个 token（3 个 block）的主 Compressor 为例（`head_dim=512`, `coff=2`）：

```
Token 3 (block 0 最后一个 token):
  Step 1: wkv(token3) → [o₃(512d) | n₃(512d)]
  Step 2: unflatten → tensor[block0, pos3] = [o₃ | n₃]
  Step 3: overlap_transform
    o₃ → new_tensor[block1, pos3]     ← overlap 滑到 block 1 的 pos 3
    n₃ → new_tensor[block0, pos7]     ← normal 留在 block 0 的 pos 7
  Step 4: 压缩
    Block 0 的 pos 7 贡献: C₀[d] += n₃[d] × softmax_weight[7, d]
    Block 1 的 pos 3 贡献: C₁[d] += o₃[d] × softmax_weight[3, d]
  → Token 3 的完整 1024d 信息被拆开:
    overlap 视图 (512d) → C₁, normal 视图 (512d) → C₀

Token 7 (block 1 最后一个 token):
  Step 1: wkv(token7) → [o₇(512d) | n₇(512d)]
  Step 2: unflatten → tensor[block1, pos3] = [o₇ | n₇]
  Step 3: overlap_transform
    o₇ → new_tensor[block2, pos3]     ← overlap 滑到 block 2
    n₇ → new_tensor[block1, pos7]     ← normal 留在 block 1 的 pos 7
  Step 4: 
    Block 1 的 pos 3 (来自 token3 的 overlap) 和 pos 7 (来自 token7 的 normal)
      共同贡献到 C₁
    Block 2 的 pos 3 (来自 token7 的 overlap) 贡献到 C₂
```



---

## 8. Indexer 的 KV Cache 计算与加权池化

### 8.1 Indexer 与主 Compressor 的对比

| | 主 Compressor | Indexer Compressor |
|---|---|---|
| `head_dim` | 512 | 128 |
| `coff` | 2 | 2 |
| `wkv` 输出 | 1024d | 256d |
| 压缩后操作 | `act_quant(FP8)` | `rotate(Hadamard)` + `fp4_act_quant` |

### 8.2 wkv 和 wgate：两套独立透镜

```
kv    = x · W_KV     ← "内容是什么"  (256d)
score = x · W_GATE   ← "有多重要"    (256d)
```

对每个 token，两套独立权重产生两个独立的 256d 向量。

### 8.3 为什么需要 Score 加权池化

#### 不要 score 的问题

```python
# 如果简单平均：
kv = kv.mean(dim=2)   # 8 个 position 同等权重
# 问题：标点符号和关键动词贡献相等，模型没有发言权
```

#### 有了 score 的能力

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
# 8 position × 128 dims = 1024 个独立权重
```

关键：**逐维度门控 vs 逐 token 门控**

**逐 token 门控（粗糙）**：
```
token 整体得分 → 一个标量权重 → 所有维度一律放大/缩小
```

**逐维度门控（精细，实际做法）**：
```
对于 dim=d: 从 8 个 position 中独立选择最相关的
  → 从 token A 取"语义"维度, 从 token B 取"语法"维度, 忽略 token C
```

### 8.4 完整数据流

```
x: [1, 4096, 7168]
│
├─ wkv(x):   [1, 4096, 256]     ← "内容是什么"
│   ├── overlap(前128d):  适合跨 block 参考的概括表示
│   └── normal(后128d):   适合本 block 的精确表示
│
├─ wgate(x): [1, 4096, 256]     ← "有多重要"
│   ├── overlap(前128d):  对相邻 block 的重要性
│   └── normal(后128d):   对本 block 的重要性
│
│  unflatten:      [1, 1024, 4, 256]
│  + ape:          位置偏置注入
│  overlap_transform: [1, 1024, 8, 128]
│    每个 block 有 8 个 position, 前 4 来自 overlap, 后 4 来自 normal
│
│  score.softmax(dim=2):  逐维度在 8 个 position 上归一化
│  kv * score_softmax:    逐维度加权
│  sum(dim=2):            [1, 1024, 8, 128] → [1, 1024, 128]
│
└─ kv_compressed: [1, 1024, 128]   ← Indexer 的压缩 KV cache
```

### 8.5 类型汇总表

| Key | 维度 | 存储类型 | 用途 |
|-----|------|---------|------|
| Q (主，低秩瓶颈前) | 1536 | BF16 | 压缩查询表示，复用给 indexer |
| Q (主，拆头后) | 128 × 512 = 65536 | BF16 | 128 个头各自查询 |
| Q (Indexer) | 64 × 128 = 8192 | FP4 | 快速近似打分 |
| KV (主) | 512 | BF16(rope 64) + FP8(nope 448) | KV cache：`[swa(128) + compressed(1024), 512]` |
| KV (Indexer 压缩后) | 128 | FP4 | Indexer KV cache：`[1024, 128]` |
| Attn Sink | 128 | FP32 | 每 head 的可学习 logits |
| Compress Weights (主) | 1024 = 512×2 | BF16→pool→FP8 | 主压缩器：产生 `[1024blocks, 512]` |
| Compress Weights (Indexer) | 256 = 128×2 | BF16→pool→FP4 | Indexer 压缩器：产生 `[1024blocks, 128]` |

---

## 9. 架构设计问答（为什么）

> 核心组件存在的原因、组件间的分工、以及常见误解的澄清。

### 9.1 为什么要有 Indexer？

**Indexer 是 CSA 存在的理由。没有它，CSA 要么退化成 HCA，要么贵到不可行。**

##### 从计算量看

CSA 的候选池在长上下文时非常大——1M tokens / 4 = 250,000 个压缩块。如果对每个块都做精确注意力：

```
主 Q·K^T: 128 heads × 512d × 250,000 = 16G FLOPs/query  → 不可接受
```

Indexer 用一个更小更便宜的 Q 做近似打分，只选出 1024 个候选：

```
Indexer Q·K^T: 64 heads × 128d × 250,000 = 2G FLOPs/query (FP4 下更省)
主 Q·K^T:      128 heads × 512d × 1,024    = 67M FLOPs/query  → 可控
总计:          ~2.1G FLOPs/query vs 16G → 节省 ~87%
```

##### 三种替代方案的失败

| 替代方案 | 问题 |
|---------|------|
| **A: 不做选择，全做密集注意力** | 16G FLOPs/query → 不可接受 |
| **B: 用固定规则采样（如均匀间隔）** | 不看内容，质量崩——"Paris"和"的"选出相同的块 |
| **C: 不用 Indexer，增大压缩率** | 就是 HCA！分辨率太低（128:1），细粒度依赖丢失 |

##### Indexer 的成本

```
Indexer Compressor 参数:  [7168, 256] × 2(wkv+wgate) ≈ 3.7M
Indexer Q 投影参数:       [1536, 64×128] ≈ 12.6M
Indexer weights 参数:     [7168, 64] ≈ 0.5M
Indexer KV cache:         N/4 × 128d × FP4 ≈ 16MB per CSA layer (N=1M)

总计: ~17M 参数 (~0.003% of model) + 16MB cache per CSA layer
节省: ~14G FLOPs/query/layer
```

##### 一句话

Indexer 是一个极便宜的近似过滤器——用小 Q(64heads×128d×FP4)快速扫描 25 万个候选块，只选出最有希望的 1024 个交给主注意力做精确计算。

---

### 9.2 为什么要有 Compressor？

##### 三种角色

**角色 1：降低候选基数（为 Indexer 服务）**

```
无压缩:   Indexer scan 1,000,000 个候选 → 选 1024 个（覆盖率 0.1%）
4:1 压缩:  Indexer scan 250,000 个候选  → 选 1024 个（每个块含 4 token 信息）
128:1 压缩: 无需 Indexer                → 密集 attend ~8,000 个
```

候选基数降了，Indexer 的计算量等比下降。

**角色 2：聚合上下文（为注意力质量服务）**

单个 token 是孤立的。"Paris" 这一个 token 能提供多少上下文？但 "Paris" 和周围的 "the capital of" 一起压缩，语义就完整了。逐维度 softmax 加权池化让模型学习性地保留重要的、丢弃冗余的。

**角色 3：减少 KV cache 存储（为内存服务）**

```
无压缩:  N × 512d × BF16
N=1M → 1GB per layer → 61GB → 放不进 GPU

CSA: 128 + N/4 × 512d → ~128MB per layer
HCA: 128 + N/128 × 512d → ~4MB per layer
```

##### HCA 的 Compressor 不需要 Indexer

HCA 压缩 128:1 后候选降到 ~8K，密集注意力可直接承受。所以 HCA 只有 Compressor 没有 Indexer——Compressor 单独完成了"降存储 + 降计算"两项任务。

---

### 9.3 Compressor 如何压缩 token 为 block

核心操作只有一行：

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

以 HCA（ratio=128）为例最清晰：

```
kv:     [1, 32 blocks, 128 tokens, 512 dims]
score:  [1, 32 blocks, 128 tokens, 512 dims]

→ softmax dim=2:  每个 (block, dim) 在 128 个 token 上独立归一化
→ kv * softmax:   每个 token 的每个维度被自己的权重缩放
→ sum dim=2:      128 个权重化后的 token 求和 → 1 个 512d 向量
```

##### 为什么逐维度 softmax 而非逐 token

**逐 token（粗糙）**：token 整体一个标量权重 → "要么全要，要么全不要"。
**逐维度（精细，实际做法）**：512 个维度各自独立决定 128 个 token 中谁最重要。

```
token "Paris" 维度 10 (location):       score 高 → 权重 0.4 → 贡献最大
token "Paris" 维度 50 (color, 无关):    score 低 → 权重 0.01 → 几乎被忽略
token "the" 维度 30 (structure marker): score 高 → 权重 0.3 → 保留结构信息
```

##### score 从哪里来

```python
score = self.wgate(x)    # 独立的线性投影，与 wkv 参数不同
```

同一个 token 经过两条不同路径：

```
x [7168] ──→ wkv   → kv    [512]    ← "说了什么"（内容编码）
x [7168] ──→ wgate → score [512]    ← "重不重要"（门控打分）
```

---

### 9.4 Indexer 的 top-k 怎么在主 Compressor 中用上

**Indexer 不传给主 Compressor。它直接传给 sparse_attn kernel。**

```
                        主 Compressor
                             │
                             │ 把所有 token 压成 1024 个 512d 块
                             │ 写入 kv_cache[128:] 
                             ▼
                      kv_cache [1, 1152, 512]
                      ┌──────┬─────────────────┐
                      │ SWA  │  压缩 KV 块       │
                      │ 128  │  1024 个 × 512d  │
                      └──────┴─────────────────┘
                             ▲
                             │ sparse_attn 按索引 gather
                             │
    Indexer ──→ topk_idxs ────┘
    输出:      [1, 4096, 1024] 整数
    含义:      "取 kv_cache 的第 4123, 4100, 4096, ... 号位置"
```

Indexer 和主 Compressor 在代码中没有直接调用关系。它们的唯一接口是 `sparse_attn` 的 `topk_idxs` 参数。

##### 具体的 gather 过程

```python
# sparse_attn kernel 内部:
idx = topk_idxs[by, bx, j]        # 取出第 j 个索引
if idx == -1:                      # -1 → Indexer 标为无效
    score = -inf
else:
    kv_vec = kv[by, idx, :]        # 按索引从主 kv_cache 取值
    # idx=4096 → block 0 的主 Compressor 512d 压缩 KV
    # idx=100  → token 100 的原始 SWA KV
    score = q_vec @ kv_vec
```

Indexer 只是开了一张"取货单"，主注意力 kernel 照单从主 kv_cache 取数据。

---

### 9.5 为什么还要保留 raw KV cache

压缩 KV 有两个不可弥补的缺陷：

##### 缺陷 1：因果性盲区

```
ratio=4, query token=4102:

block 1025: tokens [4100,4101,4102,4103]  ← 没完成！query=4102 在这块里，不可见
block 1024: tokens [4096,4097,4098,4099]  ← 可见 ✓

query=4102 通过压缩 KV 看到的最新 token 是 4099
tokens [4100,4101] 在因果上已过去但还没形成压缩块 → 不可见！
```

越近的 token 越重要，但也越可能落在未完成的压缩块里。SWA 绕过了这个问题——最近 128 个 token 的原始 KV 永远直接可见。

##### 缺陷 2：压缩不可逆

4 个 token → 512d 摘要。你能读出"这 4 个 token 的大意"，但无法还原"token 3 具体是 Paris"。程序、数学、多跳推理需要 token 级精度，摘要不够。

##### SWA 的存储代价

```
SWA: 128 × 512d = 65K floats per layer，常数
N=1M: SWA 占总 KV cache 的 0.01%
```

---

### 9.6 既保存 raw 又保存压缩，还能省 KV cache 吗

**能。因为 raw KV 只存 128 条就丢弃，不是存全部。**

```
token 诞生 → 进入 SWA 区（原始 KV）→ 128 步后被覆写 → 丢弃
                                  ↓
                    但在这 128 步期间，每 4 个 token 被压缩一次
                    → 摘要进入压缩区 → 永久保留
```

```
N=1M:
  无压缩:   1,000,000 × 512  = 512M  floats per layer
  CSA:          128 × 512  +  250,000 × 512  = 128M  floats → 节省 75%
  HCA:          128 × 512  +  7,812 × 512    = 4M    floats → 节省 99%
  
SWA 贡献: 128 条，占比 0.01%
```

**压缩区长度随 N 线性增长但增速只有原来的 1/ratio。SWA 是常数。总存储远小于无压缩。**

---

### 9.7 压缩块会被淘汰吗？Indexer 选出的块没被缓存怎么办

**不会。压缩块永不淘汰。**

```python
# SWA 区 (kv_cache[0:128]):    循环覆写 ← 会淘汰
# 压缩区 (kv_cache[128:]):     顺序追加 ← 永不覆写

# Compressor.forward, decode:
self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
# start_pos 单调递增 → 写入位置单调递增 → 新块追加，旧块不动
```

Indexer 每次打分是对**所有已存在的块**扫描：

```python
index_score = einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
#                                                           ↑
#                                            扫描全部 end_pos//ratio 个块！
```

每次的 top-1024 可能不同，但候选池始终是全量。上次没选中的块下次仍然可能被选中。

---

### 9.8 分工总结：谁省了什么

```
                   Compressor                    Indexer
                 ┌──────────┐              ┌──────────────┐
省 KV cache:       ✅ 主力                   ❌ 不参与
                  (N→N/ratio)               (Indexer cache
                                            是额外开销)

省计算 (HCA):      ✅ 粗粒度压缩完成           ❌ 不需要
                  (候选降到 N/128，
                   密集注意力可控)

省计算 (CSA):      ⚠️ 辅助                   ✅ 主力
                  (候选降到 N/4；             (从 N/4 中选 1024)
                   但候选仍 250K 太大)

省计算 (decode):  ✅ 增量追加                  ✅ 从 250K 选 1024
                  (每 ratio 步才写 1 block)    (每步都 scan+select)
```

**一句话**：Compressor 负责降存储和降候选基数；Indexer 负责在候选基数仍太大时，进一步筛选到可控的 1024。HCA 的 Compressor 粗粒度到不需要 Indexer；CSA 的 Compressor 细粒度到必须 Indexer 兜底。

---

## 附录：关键代码路径索引

| 组件 | 文件 | 行号 |
|------|------|------|
| Transformer | `model.py` | 769-809 |
| Block | `model.py` | 647-700 |
| Attention | `model.py` | 436-543 |
| Compressor | `model.py` | 279-377 |
| Indexer | `model.py` | 380-433 |
| Gate (MoE) | `model.py` | 546-584 |
| Expert (SwiGLU) | `model.py` | 587-606 |
| MoE | `model.py` | 609-644 |
| ParallelHead (hc_head) | `model.py` | 703-735 |
| MTPBlock | `model.py` | 738-766 |
| hc_split_sinkhorn kernel | `kernel.py` | 371-438 |
| sparse_attn kernel | `kernel.py` | 277-368 |
| fp8_gemm kernel | `kernel.py` | 203-273 |
| fp4_gemm kernel | `kernel.py` | 441-536 |
| act_quant | `kernel.py` | 105-125 |
| fp4_act_quant | `kernel.py` | 186-200 |
| 模型配置 | `config.json` | 1-35 |
