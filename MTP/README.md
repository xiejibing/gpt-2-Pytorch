# MTP（Multi-Token Prediction）推测解码加速原理

## 概述

MTP 是 vLLM 中实现的一种**推测解码（Speculative Decoding）**方案。它的核心思想是：用一个极轻量的 draft 模型"猜"多个后续 token，然后目标模型一次性验证所有草案 token——用廉价的 draft 计算换取昂贵的自回归串行等待。

vLLM 中目前支持 **18 种** MTP 模型：
DeepSeek V2/V3/V4、Gemma4、MiMo/V2、GLM4 MoE/Lite/OCR、Ernie 4.5、Nemotron H、Exaone MoE/4.5、Qwen3 Next/3.5、LongCat Flash、Pangu Ultra、Step3.5、HY V3。

---

## 1. 为什么 MTP 能加速推理

### 1.1 推理是 memory-bandwidth bound

LLM 解码阶段（小 batch）的主要瓶颈不是计算，而是**从显存加载权重**：

```
一次目标模型 forward 的耗时分解（以 A100 + Llama-70B 为例）:
  - 加载权重: ~80-90% 的时间
  - 实际计算: ~10-20% 的时间
```

### 1.2 推测解码的核心优势：合并权重加载

```
无推测（产生 4 个 token）:
  加载60层权重→算→加载60层权重→算→加载60层权重→算→加载60层权重→算
  → 权重加载了 4 次，40ms

有推测（产生 ~3 个有效 token）:
  draft×3（0.1ms×3）→ 加载60层权重→一次性算4个token
  → 权重只加载了 1 次，~16ms

加速比 ≈ 40/16 ≈ 2.5×
```

**核心洞察**：处理 4 个 token 的边际计算开销远小于省下来的 3 次权重加载时间。推测解码不是靠"减少总计算量"来加速，而是靠**减少权重加载次数**。

> 注意：在 batch size 很大或 prefill 阶段，模型变成 compute-bound，推测解码收益会缩水甚至为负。

---

## 2. MTP 的架构设计

### 2.1 Draft 模型有多轻量？

以 DeepSeek MTP 为例，一个 draft 预测层只有：

```
1× RMSNorm (enorm)              — 归一化输入的 token embedding
1× RMSNorm (hnorm)              — 归一化目标模型的 hidden states
1× Linear(2h → h)               — 融合 embedding 和 hidden states
1× DecoderLayer (mtp_block)     — 1 层 transformer decoder
1× RMSNorm + lm_head            — 产出 logits
```

对比目标模型 60+ 层 decoder，draft 每步只跑**1 层**，计算量可忽略不计。

当 `num_speculative_tokens > num_mtp_layers` 时，同一层会被循环复用：

```python
# deepseek_mtp.py
current_step_idx = spec_step_idx % self.num_mtp_layers
```

### 2.2 权重共享节省显存

MTP 模型的 `embed_tokens` 和 `lm_head` 与目标模型共享，不需要额外存储：

```python
# llm_base_proposer.py:1271-1277
else:
    # MTP model
    share_embeddings = True  # embedding 共享
    share_lm_head = True     # lm_head 共享
```

这意味着 MTP 模型的额外显存开销极小——只需要那几层 MTP decoder layer 的权重。

---

## 3. 完整流水线

### 3.1 两步流水线总览

```
Step N-1:                               Step N:
┌──────────────────────────────┐        ┌─────────────────────────────────┐
│ 1. 目标模型 forward           │        │ 1. 目标模型 forward              │
│    输入: [..., A, B, C]       │        │    输入: [..., C, T, d0, d1, d2]│
│    输出: logit_C → 采样得 T    │        │    输出: logits at T,d0,d1,d2   │
│                              │        │                                 │
│ 2. 拒绝采样验证上一轮草案     │        │ 2. 拒绝采样验证 [d0, d1, d2]     │
│    （如果有的话）              │        │    → 对比目标 logit vs draft    │
│                              │        │    → 输出被接受的 tokens          │
│ 3. Draft 模型推测下一轮:      │        │                                 │
│    → d0, d1, d2              │───────→│ 3. Draft 模型推测下一轮:          │
│    存入 draft_tokens          │        │    → d0', d1', d2'              │
│                              │        │    存入 draft_tokens             │
└──────────────────────────────┘        └─────────────────────────────────┘
```

### 3.2 验证就发生在下一轮推理中

**验证不是独立的一步**——上轮的 draft tokens 被拼接到本轮的输入中，目标模型在 draft 位置上产生的 logits 就是验证依据：

```
目标模型输入: [..., C, T, d0, d1, d2]

causal attention 下，每个位置只看到它之前的 token：
  位置 T:   看到 [..., C, T]             → logit_T  → 采样得 bonus token（新输出）
  位置 d0:  看到 [..., C, T, d0]         → logit_d0 → argmax == d0? → 接受/拒绝
  位置 d1:  看到 [..., C, T, d0, d1]     → logit_d1 → argmax == d1? → 接受/拒绝
  位置 d2:  看到 [..., C, T, d0, d1, d2] → logit_d2 → argmax == d2? → 接受/拒绝
```

验证和新 token 采样是**同一次目标前向传播的两个输出**，零额外开销。

### 3.3 Draft tokens 如何喂入目标模型？

通过 `combine_sampled_and_draft_tokens_kernel` 将所有 draft tokens 写入目标模型的 `input_ids`：

```python
# input_batch.py:319-333
# 写入上一轮验证后采纳的 token
tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

# 把 draft tokens 紧接着写入
for each draft_token:
    tl.store(input_ids_ptr + query_end - num_draft_tokens + block, draft_tokens)
```

### 3.4 拒绝采样（验证逻辑）

```python
# rejection_sampler.py 核心逻辑
# Greedy 模式：
for pos in range(num_draft_tokens):
    if not rejected:
        if draft_token_ids[pos] == target_argmax[pos]:
            output[pos] = draft_token_ids[pos]   # 接受
        else:
            rejected = True
            output[pos] = target_argmax[pos]     # 拒绝，用目标模型的预测
if not rejected:
    output[num_draft_tokens] = bonus_token_id    # 全部接受 → 奖励一个 bonus token

# Random 模式：
for pos in range(num_draft_tokens):
    if not rejected:
        accept_prob = min(1.0, target_prob[pos] / draft_prob[pos])
        if random() < accept_prob:
            output[pos] = draft_token_ids[pos]   # 接受
        else:
            rejected = True
            output[pos] = recovered_token_ids[pos]  # 拒绝，从修正分布采样
```

### 3.5 Draft 模型的自回归 proposal 循环

```python
# llm_base_proposer.py:525-591
draft_token_ids_list = [draft_0]  # 第一步用目标的 hidden states 猜
for token_index in range(num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()  # 上一个草案 token

    ret_hidden_states = self.model(
        input_ids=input_ids,              # 当前 token ID
        hidden_states=hidden_states,      # 上一步的 hidden state
    )
    draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
    draft_token_ids_list.append(draft_token_ids)

# 返回 [batch_size, num_speculative_tokens]
return torch.stack(draft_token_ids_list, dim=1)
```

---

## 4. 关键设计问题

### 4.1 为什么需要归一化 embedding（enorm/hnorm）？

MTP 层拼接了两个**来源完全不同、数值尺度差异巨大**的张量：

```
enorm:  inputs_embeds（来自 embedding 查表，尺度取决于初始化和训练，无约束）
hnorm:  previous_hidden_states（经过 60+ 层 RMSNorm，尺度接近单位方差）

拼接前各自归一化 → eh_proj 能公平融合两者信息
```

如果不做归一化，尺度大的那一方会主导整个线性投影，另一方信息被淹没。

### 4.2 为什么目标模型只需要 input_ids，MTP 却需要额外传 hidden_states？

**目标模型**有完整的 KV Cache，attention 可以直接 attend 到每个历史位置的 K/V，所以只需 `input_ids`。

**MTP Draft 模型**只有 1 层 decoder，没有目标模型 60+ 层的全量 KV Cache（复制一份会翻倍显存开销）。替代方案是：**用目标模型最后一层的 hidden state 作为整个前缀的压缩表示**——这是一个看过完整上文的向量，用它来代替显式的 KV Cache。

| | 目标模型 | MTP Draft 模型 |
|---|---|---|
| 上下文来源 | KV Cache（显式存储每个历史 token） | Hidden States（一个向量压缩上文） |
| 层数 | 60+ | 1 |
| 每步输入 | `input_ids` | `input_ids` + `hidden_states` |

### 4.3 为什么需要同时传 input_ids 和 hidden_states 给 MTP？

`hidden_states` 告诉 MTP "整个上文说了什么"（语义上下文），`inputs_embeds` 告诉 MTP "上一步具体生成了哪个词"（token 身份）。两者缺一不可——去掉 embedding 就像让一个人根据上下文猜下一个词但不告诉他前一个词是什么。

### 4.4 Draft tokens 的 KV Cache 如何处理？

目标模型会为所有输入 token（包括 draft tokens）计算 KV Cache：

```
目标模型 forward 为 [..., T, d0, d1, d2] 全部计算 KV：
  T 的 KV:   ✓ 始终保留
  d0 的 KV:  ✓ 接受 → 保留；✗ 拒绝 → 作废
  d1 的 KV:  前面被拒 → 作废
  d2 的 KV:  前面被拒 → 作废
```

被拒绝位置的 KV Cache 直接丢弃。这是推测解码必须付出的代价——但只要接受率足够高，省下来的目标 forward 次数完全覆盖这点浪费。

---

## 5. 加速比分析

### 5.1 理论加速比

假设目标模型 60 层，MTP 1 层，`num_speculative_tokens=3`，平均接受 2 个草案（即每次有效输出 3 个 token）：

| | 无推测 | 有推测 |
|---|---|---|
| 目标模型 forward 次数 | 3 次 | 1 次 |
| 目标模型处理 token 数 | 每次 1 个 | 一次 4 个 |
| 权重加载次数 | 3 次 | 1 次 |
| **理论加速比** | — | **~2.5×** |

### 5.2 什么时候有效？

- **Decode 阶段 + 小 batch** → 有效（memory-bound）
- **Prefill 阶段 + 大 batch** → 无效甚至为负（compute-bound）
- **接受率是关键**：接受率越高，加速比越大。MTP 的 draft 层是专门训练来预测目标模型下一个 token 的，接受率通常 70-90%

---

## 6. 关键代码文件索引

| 文件 | 作用 |
|---|---|
| `vllm/v1/spec_decode/llm_base_proposer.py` | Draft proposal 核心逻辑（`SpecDecodeBaseProposer`） |
| `vllm/v1/sample/rejection_sampler.py` | 拒绝采样验证（`rejection_sample()`） |
| `vllm/v1/worker/gpu/input_batch.py` | `combine_sampled_and_draft_tokens_kernel` — 将草案写入目标模型输入 |
| `vllm/v1/worker/gpu/model_runner.py` | 完整流水线编排 |
| `vllm/model_executor/models/deepseek_mtp.py` | DeepSeek MTP 模型实现 |
| `vllm/config/speculative.py` | MTP 配置（`MTPModelTypes`、`SpeculativeConfig`） |

---

# DFlash：并行投机解码

## 概述

DFlash 是 vLLM 中另一种投机解码方案，与 MTP 的最大区别在于：**draft model 使用 cross-attention 一次性并行产出所有 draft token，而不是自回归逐 token 预测。**

目前仅支持 **Qwen3.5 DFlash** 模型。

---

## 1. 与 MTP 的核心区别

| | MTP（自回归式） | DFlash（并行式） |
|---|---|---|
| Draft 方式 | 自回归逐 token 预测 | 一次 forward 预测所有 draft token |
| Attention 类型 | Causal self-attention | Cross-attention（non-causal） |
| Draft model 输入 | 上一个 draft token + target hidden states | target hidden states 作为 context（K/V） |
| Forward 次数 | N 次（或 N 合 1 的 parallel_drafting） | **1 次** |
| Context K/V 来源 | Draft model 自己的 KV cache | Target model hidden states 投影 |

---

## 2. DFlash 的并行机制

### 2.1 核心设计：Cross-attention 替代 Causal Self-attention

MTP 中，draft model 用 causal self-attention，每个 MASK token 只能看到它之前的 token：

```
MTP draft input: [bonus, MASK₁, MASK₂, MASK₃]
causal mask:
  [1, 0, 0, 0]
  [1, 1, 0, 0]   ← MASK₁ 能看到 bonus，看不到 MASK₂/MASK₃
  [1, 1, 1, 0]   ← MASK₂ 能看到 bonus + MASK₁，看不到 MASK₃
  [1, 1, 1, 1]
```

DFlash 中，draft model 用 cross-attention，**完全不加 causal mask**：

```
DFlash: 所有 MASK token 的 Q 同时 attend 到同一份 context K/V

K/V_cache = target_model.hidden_states → 投影 → [context_len, d_k]
                                                 ↑
                                MASK₁ 的 Q ────→│
                                MASK₂ 的 Q ────→│  同一个 K 矩阵！
                                MASK₃ 的 Q ────→│

scores = Q @ K_cacheᵀ  →  一次矩阵乘法，所有 MASK 位置并行计算
```

### 2.2 流水线分两个阶段

**阶段 1：`precompute_and_store_context_kv`**

Target model 产出 hidden states 后，DFlash draft model 将其一次性投影为所有层的 K/V 并写入 KV cache：

```python
# dflash.py:270 → qwen3_dflash.py:344
all_kv_flat = F.linear(
    normed_context_states,  # target hidden states: [num_ctx, d]
    self._fused_kv_weight,  # 所有层的 KV 权重拼接: [L * 2 * kv_size, d]
)
# → [L * 2 * kv_size, num_ctx] → reshape → [L, num_ctx, kv]
# 然后逐层插入 KV cache
```

这里做了关键优化：**一次 GEMM 投影所有层的 K/V**，避免逐层调用。

**阶段 2：Draft model forward（仅 query 位置）**

```python
# dflash.py:278
input_ids = [bonus_token, MASK, MASK, MASK]   # shape: [1 + num_spec_tokens]

# Embedding → [N, d]
# 过所有 attention 层：
#   Q = W_q(hidden_states)    → [N, d_k]
#   K/V 从 cache 读取（已在阶段 1 插入）
#   attn = softmax(Q @ Kᵀ) @ V  → [N, d]
# 过 FFN → [N, d]
# lm_head → [N, vocab]

# 采样（跳过 bonus 位置）：
is_sample = is_query & (query_off > 0)  # 只对 MASK 位置采样
draft_tokens = argmax/sample(logits[1:])
```

### 2.3 为什么一次 forward 就能产出所有位置的 logits

Transformer 中，每个位置的 hidden state 计算在矩阵层面是独立的：

```
输入: [bonus, MASK₁, MASK₂, MASK₃]   → Embedding → [4, d]

Attention:
  Q = W_q @ [4, d]                        ← 一次矩阵乘法，4 个 query 同时产出
  scores = [4, d_k] @ [d_k, ctx_len]      ← 一次 GEMM，4 行同时计算
  output = softmax(scores) @ V            ← 一次 GEMM，4 行并行

FFN:
  hidden = W2 @ ReLU(W1 @ [4, d])         ← position-wise，4 行独立并行

lm_head:
  logits = [4, d] @ [d, vocab]            ← 一次 GEMM → [4, vocab]
                                             每行是每个位置的词表分布
```

**关键**：DFlash 的 K/V 来自 target hidden states（固定不变），不是来自 draft tokens 自己（像自回归那样需要等前面 token 先跑完）。因此所有 MASK token 互不依赖，可以一起算。

### 2.3.1 具体例子：从矩阵运算看并行

假设 context 为「我今天吃了」（4 个 token），`num_speculative_tokens=3`，`d=8`。

**阶段 1：预计算 context K/V**

```
H_target = [4, 8]   ← "我今天吃了" 四个 token 的 target hidden states

K_context = W_k @ H_targetᵀ    →  [4, 8]
V_context = W_v @ H_targetᵀ    →  [4, 8]

KV_cache = [K_context, V_context]  ← 写入 GPU 显存，固定不变
```

**阶段 2：Draft model forward**

输入构造：

```
input_ids = [bonus_token, MASK, MASK, MASK]   →  shape [4]

Embedding → [4, 8]
         col0 col1 col2 col3 col4 col5 col6 col7
bonus:  [0.1  0.3 -0.2  0.5 -0.1  0.2  0.4 -0.3]   ← 第 0 行
MASK₁:  [0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]   ← 第 1 行
MASK₂:  [0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]   ← 第 2 行
MASK₃:  [0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]   ← 第 3 行
```

（实际 MASK token 的 embedding 不是全零，这里简化为 0 方便理解）

Attention 层：

```
qkv = W_qkv @ [4, 8]               → [4, 24]
q, k, v = split(qkv)               → 各 [4, 8]

# q 的 4 行分别对应 bonus、MASK₁、MASK₂、MASK₃ 四个位置的 query
```

Attention score 计算 —— **一次矩阵乘法，4 行并行**：

```
scores = q @ K_contextᵀ            → [4, 8] @ [8, 4] = [4, 4]

         ctx₀   ctx₁   ctx₂   ctx₃   (context position: 我 今天 吃 了)
bonus:  [0.5,   0.2,  -0.1,   0.3]  ← 第 0 行
MASK₁:  [0.3,  -0.1,   0.4,   0.2]  ← 第 1 行
MASK₂:  [-0.2,   0.5,   0.1,   0.4]  ← 第 2 行
MASK₃:  [0.4,   0.1,  -0.2,   0.5]  ← 第 3 行
```

每一行是该位置 query 对 context 4 个 token 的 attention score。**四行同时计算，互不依赖**——因为 `K_context` 是固定的。

Softmax + 加权求和 —— 同样是一次数值运算覆盖 4 行：

```
attn_weights = softmax(scores, dim=-1)   → [4, 4]
output = attn_weights @ V_context         → [4, 4] @ [4, 8] = [4, 8]
```

FFN（position-wise）：

```
hidden = W₂ @ ReLU(W₁ @ [4, 8])          → [4, 8]
```

lm_head：

```
logits = hidden @ W_lm_headᵀ             → [4, 8] @ [8, vocab] = [4, vocab]

         "我"  "今天"  "吃了"  "饭"  "面"  "苹果"  ...
bonus:  [0.1,  0.05,   0.02,  0.3,  0.2,  0.1,  ...]  ← bonus 位置，不使用
MASK₁:  [0.05, 0.02,   0.01,  0.4,  0.3, 0.15,  ...]  ← 采样 → "饭"
MASK₂:  [0.02, 0.01,   0.05,  0.2,  0.5,  0.1,  ...]  ← 采样 → "面"
MASK₃:  [0.01, 0.01,   0.02,  0.1, 0.15,  0.6,  ...]  ← 采样 → "苹果"
```

采样跳过第 0 行 bonus 位置（`is_sample = is_query & (query_off > 0)`），对 MASK₁~MASK₃ 分别 argmax/sample，得到 3 个 draft token：`["饭", "面", "苹果"]`。

整个流程中，`[4, d]` 矩阵从头流到尾，没有 token 需要等前面的 token 先算完。**并行的本质就是 batch 维度的矩阵运算。**


### 2.4 前后 token 的关联性从哪来

Draft token 之间的顺序依赖**全部压缩在 target hidden states 里**。

Target model 的 hidden state 不是孤立的——以 position 3（"了"）为例，它在 causal target 中已经编码了 "我今天吃了" 的完整依赖链。MASK₂ 虽然不知道 MASK₁ 是什么，但可以通过 cross-attention 关注 "了" 的 K/V，间接感受到前文的语义约束。

同时，**位置编码**在这里起关键作用：position 1 和 position 2 的 embedding 不同，产生不同的 Q，从而关注 context 中不同的部分。

DFlash 训练时的假设是：给定相同的 target context，多个未来 token 的相互影响可以在 target hidden states 中被充分捕获，使得并行预测足够准确。

---

## 3. DFlash vs MTP 总结

| 维度 | MTP | DFlash |
|---|---|---|
| Draft 并行度 | 自回归（串行） | Cross-attention（并行） |
| Draft forward 次数 | num_spec_tokens 次 | 1 次 |
| 额外显存 | 极小（几层 MTP 权重） | 极小（几层 DFlash 权重） |
| Context 信息来源 | target hidden states（一个压缩向量） | target hidden states（投影为 K/V cache） |
| 需要的特殊 token | 无 | MASK token（需在 vocab 中定义） |
| 支持模型 | DeepSeek V2/V3/V4、Gemma4 等 18 种 | Qwen3.5 DFlash |
| Bonus token | 有 | 有 |

---

## 4. 关键代码文件索引

| 文件 | 作用 |
|---|---|
| `vllm/v1/spec_decode/dflash.py` | DFlashProposer — 并行 draft proposal 核心逻辑 |
| `vllm/model_executor/models/qwen3_dflash.py` | DFlashQwen3ForCausalLM — DFlash draft model 实现 |
| `vllm/v1/spec_decode/utils.py` | `copy_and_expand_dflash_inputs_kernel` — Triton kernel 构建输入 |
| `vllm/v1/spec_decode/llm_base_proposer.py` | `SpecDecodeBaseProposer` — 公共基类，parallel_drafting 逻辑 |

---

# DSpark：并行 Backbone + 序列依赖 Markov Head

## 概述

DSpark 是 DeepSpec 项目中的第三种推测解码方案。它在架构上介于 **MTP（全自回归）** 和 **DFlash（全并行）** 之间：

| | MTP | DSpark | DFlash |
|---|---|---|---|
| Draft 产出方式 | 逐 token 自回归 | 并行 backbone + **序列修正** | 一次并行 forward |
| Token 间依赖 | causal self-attention | cross-attention + Markov Head | cross-attention 仅靠 target context |
| 序列建模 | 显式（causal mask） | **Markov Head 补丁式修正** | 隐式（target hidden states） |

---

## 1. 核心设计：并行骨架 + 序列补丁

DSpark 的 draft 生成分两层：

```
┌──────────────────────────────────────────────────┐
│  第一层：并行 Backbone（cross-attention）         │
│  → 所有 MASK token 同时 attend target context     │
│  → 产出 base_logits: [batch, block_size, vocab]   │
│  → 完全并行，无 token 间依赖                      │
├──────────────────────────────────────────────────┤
│  第二层：序列 Markov Head（逐位置修正）            │
│  → 从第 0 位开始，逐位置用上一个 token 修正 logits │
│  → 修正后采样 → 作为下一位置的输入                │
│  → 引入 token 间顺序依赖                          │
└──────────────────────────────────────────────────┘
```

**第一层（并行）**：和 DFlash 类似，所有 draft 位置的 Q 同时 attend target hidden states 的 K/V，一次 forward 得到所有位置的 `base_logits`——这一步零序列依赖。

**第二层（序列）**：Markov Head 逐位置修正 logits。每生成一个 token，就用它计算下一个位置的 bias。这一步是**自回归**的，引入 token 间的顺序约束。

类比：Backbone 负责"看懂上文"（语义理解），Markov Head 负责"前后 token 搭配合理"（局部连贯性）。两者分工，互补。

---

## 2. Markov Head：三种变体

### 2.1 VanillaMarkov：纯统计 bigram bias

结构最简，仅两个矩阵：

```
输入: 上一个 token x_{k-1}
         │
    ┌────▼────┐
    │  W₁      │  Embedding(vocab, rank)  — 将 token 压缩为 rank 维稠密向量
    │  [V, r]  │
    └────┬────┘
         │ e = W₁[x_{k-1}]   shape: [rank]
    ┌────▼────┐
    │  W₂      │  Linear(rank, vocab, bias=False)
    │  [r, V]  │
    └────┬────┘
         │
         ▼
    bias = W₂(W₁[x_{k-1}])    shape: [vocab]
         │
         ▼
    final_logits = base_logits + bias
```

**完全不看 backbone hidden states**——只根据上一个 token ID 做纯统计 bigram bias。参数量仅 `2 × V × rank`（例如 V=150k, rank=256 时约 76M，相比 backbone 的数十亿参数极小）。

### 2.2 GatedMarkovHead：上下文感知的门控

VanillaMarkov 的问题：同样的前一个 token（如 "cat"），在不同上下文中的 bias 应该不同（"the cat sat" vs "the black cat from"）。

GatedMarkovHead 引入一个由 backbone hidden state 控制的**门（gate）**：

```
输入: x_{k-1} + h_k（backbone hidden state）
         │
    ┌────┴────┬──────────────────┐
    ▼         │                  │
W₁[x_{k-1}]   │    gate_proj([h_k; W₁[x_{k-1}]])
 [r]          │                  │
    │         │                  ▼
    │         │    gate = sigmoid(Linear([h_k; e]))
    │         │    gate ∈ (0, 1)^r  — 每个维度一个门控值
    │         │                  │
    └────┬────┘                  │
         │                       │
         ▼                       ▼
    gated_emb = gate ⊙ W₁[x_{k-1}]    （逐元素乘法）
         │
         ▼
    bias = W₂(gated_emb)
```

**Gate 的"关掉"机制**——关键在 `gate ⊙ embedding` 的逐元素乘法：

```
W₁["cat"]  = [0.5,  -0.3]     ← token "cat" 的 rank-2 embedding
gate       = [0.9,  0.05]     ← backbone 根据上下文 h_k 决定各维度通过比例

gated_emb  = [0.9×0.5, 0.05×(-0.3)]
           = [0.45,  -0.015]
```

| 维度 | embedding 原值 | gate | gated 结果 | 效果 |
|------|:---:|:---:|:---:|------|
| dim 0 | 0.5 | 0.9 | 0.45 | gate≈1 → **放行**，原信息几乎完整保留 |
| dim 1 | -0.3 | 0.05 | -0.015 | gate≈0 → **关掉**，值被压缩到接近 0 |

`W₁` 的不同维度可能编码 token 的不同属性：

- **dim 0（gate=0.9，放行）**：编码"cat 后面常跟动词"（sat, ran, sleeps...）——在大多数上下文都相关
- **dim 1（gate=0.05，关掉）**：编码"cat 后面常跟介词"（on, in, under...）——但 backbone 根据当前上下文判断介词语义不相关，将其抑制

类比：`W₁[token]` 是一个**全频段均衡器**（所有可能的后续模式都编码在内），gate 是一个**上下文滤波器**（backbone 根据语义选择哪些频率通过），`W₂` 再把过滤后的信号翻译成词表空间的具体 bias。

### 2.3 RNNHead：累积前缀历史的循环状态

Vanilla 和 Gated 都只看上一个 token `x_{k-1}`。但有时需要更长历史——比如 "New York" 后面跟什么，取决于整个 "New York" 而不仅仅是 "York"。

RNNHead 维护一个类似 GRU 的**循环状态**，在 block 内逐位置传播：

```
初始: state₀ = 0

对于 k = 0, 1, 2, ...:
  z = [state_{k-1}; W₁[x_{k-1}]; h_k]    ← 拼接三个信息源
  proj = joint_proj(z)                    ← [*, 3×rank]
  [gate_raw, candidate_raw, output_raw] = proj.chunk(3)

  gate      = sigmoid(gate_raw)                        ← GRU 风格的更新门
  candidate = tanh(candidate_raw)                       ← 候选新信息
  state_k   = gate ⊙ state_{k-1} + (1-gate) ⊙ candidate ← 状态更新

  bias = W₂(tanh(output_raw))             ← 输出当前步的 bias
```

**具体例子**：生成 "New York is"：

```
Step 0（生成 "York"）:
  state = [0, 0]
  prev_token = "New", h₀ = backbone hidden
  → state₀ = [0.2, -0.1]          ← 吸收 "New" 的信息
  → bias 修正 logits → 采样 "York"

Step 1（生成 "is"）:
  state = [0.2, -0.1]              ← 携带了 "New" 的信息！
  prev_token = "York", h₁ = backbone hidden
  gate = 0.8, candidate = tanh(...)
  → new_state = 0.8×[0.2,-0.1] + 0.2×candidate = [0.15, -0.05]
  → state 融合了 "New" + "York" 的联合历史
  → bias 修正 logits → 采样 "is"（而非 "city" 或 "times"）
```

关键区别：位置 1 的 bias 不仅依赖 "York"，还通过 `state` 间接依赖 "New"。这样即使 backbone 的并行 forward 不知道 draft token 间的顺序关系，RNNHead 也能补上这个缺口。

---

## 3. 训练 vs 推理的差异

### 3.1 训练时：Teacher Forcing（全并行）

训练时使用**真实的 target token**（而非采样结果）作为 Markov Head 的输入：

```python
# qwen3/modeling.py:489-494
if self.markov_head is not None:
    draft_logits = self.markov_head.apply_block_logits(
        draft_logits,
        token_ids=prev_token_ids,        # ground-truth tokens，非采样结果
        hidden_states=output_hidden_4d,   # backbone hidden states
    )
```

以 RNNHead 为例，`apply_block_logits` 的实现：

```python
# markov_head.py:191-225
def apply_block_logits(self, base_logits, *, token_ids, hidden_states):
    state = zeros(...)
    for k in range(block_size):
        prev_emb = self.get_prev_embeddings(token_ids[..., k])  # 真实 token k
        h_k = hidden_states[..., k, :]
        state, bias = self._rnn_step(state, prev_emb, h_k)
        output_logits.append(base_logits[..., k, :] + bias)
    return stack(output_logits)
```

虽然代码里是 for 循环逐位置计算，但 `token_ids` 用的是 ground-truth（不是上一步的采样结果），所以**梯度计算无依赖链**——所有位置可以并行做矩阵运算。循环只是逻辑表达，不构成计算瓶颈。

### 3.2 推理时：自回归采样（串行）

推理时没有 ground-truth，必须用上一步的**采样结果**：

```python
# markov_head.py:55-90
def sample_block_tokens(self, base_logits, *, first_prev_token_ids, ...):
    prev_token_ids = first_prev_token_ids
    for step_idx in range(proposal_len):
        step_logits = base_logits[:, step_idx, :] + bias(prev_token_ids)
        next_token_ids = sample_tokens(step_logits, temperature)
        prev_token_ids = next_token_ids   # ← 采样结果喂给下一步，形成依赖链
```

这引入了串行依赖——每一步必须等上一步采样完才能算下一个 bias。但因为 Markov Head 本身极轻量（仅 `2×V×r` 参数、两次矩阵运算），串行开销远小于 backbone forward。

---

## 4. 与 MTP / DFlash 的架构对比

```
MTP（全自回归）:
  draft₀ → draft₁ → draft₂ → draft₃
  每步跑一次 draft model forward（1层transformer）
  串行开销 = N × (1层transformer时间)

DFlash（全并行）:
  [draft₀, draft₁, draft₂, draft₃] = parallel_forward()
  一次 forward，所有位置并行产出
  串行开销 = 0（但缺少 token 间依赖建模）

DSpark = DFlash 并行骨架 + Markov Head 序列修正:
  base_logits = parallel_backbone()     ← 并行（同 DFlash）
  draft₀ = sample(base_logits[0] + bias(bonus))
  draft₁ = sample(base_logits[1] + bias(draft₀))     ← 轻量串行修正
  draft₂ = sample(base_logits[2] + bias(draft₁))
  draft₃ = sample(base_logits[3] + bias(draft₂))
  串行开销 = N × (2次矩阵乘法时间)，远小于 MTP 的 N × transformer
```

**本质**：DSpark 把序列依赖建模从 transformer attention 中剥离出来，交给极轻量的 Markov Head。Backbone 负责"理解上文"（高成本、并行），Markov Head 负责"前后连贯"（低成本、串行）。

---

## 5. 关键代码文件索引

| 文件 | 作用 |
|---|---|
| `deepspec/modeling/dspark/markov_head.py` | Markov Head 三种实现（Vanilla / Gated / RNN） |
| `deepspec/modeling/dspark/qwen3/modeling.py` | Qwen3 DSpark 模型 — backbone + Markov Head 集成 |
| `deepspec/eval/dspark/draft_ops.py` | 推理时 draft proposal 流程（`build_dspark_proposal`） |
| `deepspec/modeling/dspark/loss.py` | 训练 loss 计算（CE + L1 + confidence） |
| `deepspec/eval/dspark/confidence_head.py` | Confidence Head 评估（预测每个 draft 位置的接受概率） |
| `config/dspark/dspark_qwen3_8b.py` | 训练配置（`markov_rank=256`, `markov_head_type='vanilla'`） |

---

# Confidence Head：草稿模型的"自知之明"

## 概述

Confidence Head 是 DSpark 框架中的一个轻量级组件，它在草稿模型生成每个 token 的同时，**预测该 token 被目标模型接受的概率**。这赋予了草稿模型"自省能力"——知道自己的预测有多可靠。

```python
# deepspec/modeling/dspark/common.py:43-49
class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)  # 极简：单层线性投影 → 标量 logit

    def forward(self, features):
        return self.proj(features).squeeze(-1)  # [B, N, S] → [B, N, S]
```

---

## 1. 它解决什么问题

推测解码的核心公式：$L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}$

DSpark 的并行 Backbone + Markov Head 解决了分子中的 $T_{\text{draft}}$ 问题，但还有一个关键瓶颈：**盲目地把整个 block 送去验证会浪费 $T_{\text{verify}}$**。

两个维度的浪费来源：

| 维度 | 问题 |
|------|------|
| **数据侧** | 不同任务接受率差异大：代码 ~77%，闲聊 ~46%，后缀 token 大概率被拒 |
| **系统侧** | 低负载时多验证几乎免费，高负载时每个浪费的验证位置都挤占其他请求的 batch capacity |

Confidence Head 正是为了回答：**"在这个位置，token 被接受的概率是多少？"**，从而支持按需截断验证。

---

## 2. 架构设计

### 2.1 两种输入模式

输入特征的拼接由 `confidence_head_with_markov` 标志控制：

```
                    ┌─────────────────────────────┐
                    │    AcceptRatePredictor        │
                    │    Linear(input_dim, 1)       │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
  with_markov=False                   with_markov=True
              │                                 │
  input = hidden_states              input = [hidden_states; markov_emb]
  (维度: hidden_size)                (维度: hidden_size + markov_rank)
```

对应代码（`modeling.py:293-308`）：

```python
def predict_confidence_step(self, hidden_states, prev_token_ids=None):
    if self.confidence_head_with_markov:
        # 拼接 backbone 隐藏状态 + 前一个 token 的马氏嵌入
        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)
        features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(features).float()
    # 只用 backbone 隐藏状态
    return self.confidence_head(hidden_states).float()
```

经过 sigmoid 后得到条件接受概率：

$$c_k = \sigma\left(w^\top [h_k; W_1[x_{k-1}]]\right) \in (0, 1)$$

### 2.2 累积前缀存活概率

根据链式法则，前 $j$ 个 token 全部被接受的**联合概率**为：

$$a_{r,j} = \prod_{i \leq j} c_{r,i}$$

这个累积乘积是后续 Hardware-Aware Prefix Scheduler 的**核心输入**。

---

## 3. 训练：监督信号从哪里来

### 3.1 标签构造：目标分布与草稿分布的 L1 距离

```python
# deepspec/modeling/dspark/loss.py:60-70
def _compute_accept_rate_3d(outputs, aligned_target_logits):
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)     # 草稿分布
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)   # 目标分布
    accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
    #                   ↑ 总变分距离 / 2
    # accept_rate ∈ [0, 1]，值越大表示分布越接近 → 被接受概率越高
    return accept_rate_3d.clamp_(0.0, 1.0)
```

直观解释：如果两个分布在 token 上完全一致，接受率≈1.0；如果截然不同，接受率≈0.0。

### 3.2 BCE Loss

```python
# loss.py:157-163
confidence_targets = accept_rate_3d.detach()   # 目标接受率作为标签
confidence_errors = F.binary_cross_entropy_with_logits(
    outputs.confidence_pred.float(),   # Confidence Head 原始输出（logits）
    confidence_targets,                # 标签（0~1 之间）
    reduction="none",
) * loss_weight_mask
```

### 3.3 训练时追踪的关键指标

```python
# loss.py:164-182
confidence_probs = outputs.confidence_pred.float().sigmoid()
confidence_error = confidence_probs - accept_rate_3d

# 单点指标
confidence_abs_error    # |σ(pred) - accept_rate|  绝对误差
confidence_bias         # σ(pred) - accept_rate    正偏=高估，负偏=低估

# 累积指标 — 核心！
confidence_prefix_probs  = (confidence_probs * valid_mask).cumprod(dim=-1)
confidence_prefix_targets = (accept_rate_3d * valid_mask).cumprod(dim=-1)
confidence_cumprod_bias   # 累积前缀概率偏差 ← 直接影响调度器决策质量
```

### 3.4 τ（期望接受长度）的计算

```python
# loss.py:40-57
def _compute_local_probabilistic_stats(outputs, accept_rate_3d, valid_block_weights):
    valid_accept_rate = accept_rate_3d * outputs.eval_mask
    expected_draft_accepted = valid_accept_rate.cumprod(dim=-1).sum(dim=-1)
    tau_prob_per_block = expected_draft_accepted + 1.0  # +1 是 bonus token
    tau_prob_sum = (tau_prob_per_block * valid_block_weights).sum()
    return tau_prob_sum, pos_accept_sums
```

`tau_probabilistic` 是调度器的优化目标——最大化期望接受长度同时最小化不必要的验证开销。

**总 loss**：
```
total_loss = ce_loss_alpha × CE_Loss 
           + l1_loss_alpha × L1_Loss 
           + confidence_head_alpha × Confidence_Loss
```

---

## 4. 具体数值例子

假设 `block_size=8`，对于某个输入：

**Step 1: Confidence Head 预测**

```
位置 k:        0      1      2      3      4      5      6      7
token:        the    cat    sat    on     the    mat    and    slept
logit:        2.5    1.8    0.3   -0.5   -1.2   -2.0   -0.1   -3.0
σ(logit):    0.924  0.858  0.574  0.378  0.231  0.119  0.475  0.047
```

**Step 2: 累积前缀存活概率**

```
位置 0: a₀ = 0.924           → P(token₀被接受)
位置 1: a₁ = 0.924×0.858=0.793 → P(token₀且token₁都被接受)
位置 2: a₂ = 0.793×0.574=0.455 → P(token₀且token₁且token₂都被接受)
位置 3: a₃ = 0.455×0.378=0.172
...
```

**Step 3: 阈值截断（threshold=0.5 时）**

逐个检查 `σ(logit) < 0.5`：
- 位置 0: 0.924 ✓ → 位置 1: 0.858 ✓ → 位置 2: 0.574 ✓
- 位置 3: **0.378 ✗** → **在此截断！**

最终只提交 `[the, cat, sat]` 共 3 个 token 给目标模型验证，节省 5/8 = 62.5% 的验证计算。

---

## 5. 后置校准：Sequential Temperature Scaling (STS)

**为什么需要 STS？** 神经网络的置信度估计普遍存在**过度自信**问题。论文实验显示原始 Confidence Head 的 ECE（Expected Calibration Error）高达 3-8%，直接使用会导致对吞吐量的估计失真。

STS 流程——从左到右逐位置校准累积概率：

```
原始: c₁, c₂, ..., c_γ

Step 1: 固定 c₁ → 验证集上 grid search T₁ 最小化 ECE(σ(c₁/T₁))
         → 校准后 ĉ₁ = σ(c₁/T₁)

Step 2: 固定 ĉ₁, c₂ → grid search T₂ 最小化 ECE(ĉ₁ × σ(c₂/T₂))
         → 校准后 ĉ₂ = σ(c₂/T₂)

Step k: 固定 ĉ₁...ĉ_{k-1}, c_k → grid search T_k 
        最小化 ECE(∏_{i≤k-1} ĉᵢ × σ(c_k/T_k))
```

**关键特性**：温度缩放是保序变换，不会破坏 token 之间的相对排序。

---

## 6. 评估指标

评估时（`confidence_threshold=0`，不截断），Confidence Head 收集所有位置的预测与真实验证结果对比：

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **ECE** | Expected Calibration Error | 预测概率 vs 实际接受率的校准偏差（越低越好） |
| **AUROC** | 区分接受/拒绝位置的排序能力 | 理想值 > 0.8 |
| **Brier Score** | 概率预测的均方误差 | $(p - y)^2$ 的均值 |

同时生成 **Reliability Diagram**（可靠性图）：
- X 轴：预测的前缀接受概率
- Y 轴：实际观察到的接受率
- 完美校准 = 落在 y=x 对角线上

评估代码在 `deepspec/eval/dspark/confidence_head.py` 中的 `ConfidenceHeadRecorder` 类。

---

## 7. 关键代码文件索引

| 文件 | 作用 |
|------|------|
| `deepspec/modeling/dspark/common.py:43-49` | `AcceptRatePredictor` — Confidence Head 模型定义 |
| `deepspec/modeling/dspark/qwen3/modeling.py:255-268` | Confidence Head 初始化逻辑 |
| `deepspec/modeling/dspark/qwen3/modeling.py:293-308` | `predict_confidence_step` — 推理时预测置信度 |
| `deepspec/modeling/dspark/qwen3/modeling.py:505-517` | 训练时 confidence 特征构建 |
| `deepspec/modeling/dspark/loss.py:60-70` | `_compute_accept_rate_3d` — 训练标签计算 |
| `deepspec/modeling/dspark/loss.py:157-182` | Confidence BCE Loss + 累积偏差追踪 |
| `deepspec/modeling/dspark/loss.py:40-57` | `tau_probabilistic` — 期望接受长度 τ 计算 |
| `deepspec/eval/dspark/confidence_head.py` | `ConfidenceHeadRecorder` — 评估：ECE/AUROC/Brier + Reliability Diagram |
| `deepspec/eval/dspark/draft_ops.py:57-79` | `_predict_confidence_logits` — 推理时调用 Confidence Head |
| `deepspec/eval/dspark/draft_ops.py:82-93` | `_confident_prefix_length` — 基于置信度的前缀截断 |

---

# Hardware-Aware Prefix Scheduler：硬件感知前缀调度器

## 概述

Hardware-Aware Prefix Scheduler 是 DSpark 框架的**推理端调度组件**。它接收所有活跃请求的置信度序列和硬件容量曲线，**动态决定每个请求应该验证多长的草稿前缀**，以最大化全局系统吞吐量。

```
  ┌──────────────────────────────────────────────────────────┐
  │                DSpark 推测解码循环                         │
  │                                                          │
  │  1. Parallel Backbone  → hidden_states [h₁, ..., h_γ]   │
  │  2. Sequential Head    → sampled_tokens [x₁, ..., x_γ]  │
  │  3. Confidence Head    → confidence scores [c₁, ..., c_γ]│
  │                                                          │
  │  4. Hardware-Aware Prefix Scheduler  ◄── 核心             │
  │     ├→ 输入: 所有请求的 cᵣ,ⱼ + SPS(B) 曲线               │
  │     └→ 输出: 每个请求的最优验证长度 ℓ*ᵣ                   │
  │                                                          │
  │  5. Target Model Verification（只验证 ℓ*ᵣ 个前缀）       │
  └──────────────────────────────────────────────────────────┘
```

**与静态阈值的本质区别**：静态阈值对每个请求一视同仁，而硬件感知调度器根据**数据特征**（该请求的置信度）和**系统状态**（当前负载/SPS 曲线）联合决策。

---

## 1. SPS 曲线：调度器的"硬件地图"

### 1.1 SPS 是什么

**SPS = Steps Per Second**（每秒执行步数），即目标模型引擎每秒能完成的验证轮数。

论文 Section 3.2.2 的精确原文：

> "*Let SPS(B) denote the engine throughput, measured in **steps per second**, for a given forward-pass batch size B.*"

### 1.2 横纵坐标

| 轴 | 符号 | 含义 | 单位 |
|----|------|------|------|
| **横坐标 (X)** | $B$ | 一次验证中目标模型处理的总 token 数 | token count |
| **纵坐标 (Y)** | $\text{SPS}(B)$ | 批次大小为 B 时每秒执行步数 | steps/second |

批次大小的构成：$B = \sum_{r=1}^{R} (1 + \ell_r)$ — 每个请求 1 个 bonus token + $\ell_r$ 个验证前缀 token。

### 1.3 曲线特征

论文 Section 5.2 的原文：

> "*the true hardware capacity SPS(B) is inherently discrete, exhibiting a **jagged, step-wise degradation**.*"

SPS 曲线不是平滑函数，而是**锯齿状阶跃衰减**的非平滑阶梯曲线——因为 GPU 有离散的并行粒度边界（warp、tile、SM 分配等）。

用论文附录 A 中的具体数值示意：

```
SPS(B)
  ↑
1.0 ┤●
    │
0.5 ┤    ●
    │
0.45┤        ●
    │
    └───┼───┼───┼──→ B
        1   2   3
```

### 1.4 Profile 方法

论文原文（第8页）：

> "*Crucially, this capacity curve is **profiled once during engine initialization** and stored as a lightweight lookup table.*"

论文没有给出具体脚本，但方法推断如下：

```
┌─────────────────────────────────────────────────┐
│  引擎初始化阶段（离线，一次性）                     │
│                                                   │
│  for B in [1, 2, 4, 8, ..., B_max]:              │
│      构造一个 batch_size = B 的 dummy forward     │
│      预热（warmup）N_warmup 次                    │
│      测量 N_measure 次 forward 总耗时 T           │
│      SPS[B] = N_measure / T                      │
│                                                   │
│  → 存为轻量级查找表，推理时 O(1) 查询              │
└─────────────────────────────────────────────────┘
```

---

## 2. 调度算法（论文 Algorithm 1）

### 2.1 问题形式化

服务系统中有 $R$ 个活跃请求，每个请求 $r$ 有 $\gamma$ 个候选 token 及其校准后的置信度 $\hat{c}_{r,1}, \ldots, \hat{c}_{r,\gamma}$。

**目标**：选择截断长度 $\ell_r^* \in \{0, \ldots, \gamma\}$，最大化全局吞吐量：

$$\Theta = \tau^* \cdot \text{SPS}(B) = \left(\sum_{r=1}^R \left[1 + \sum_{j=1}^{\ell_r} a_{r,j}\right]\right) \cdot \text{SPS}\left(\sum_{r=1}^R (1 + \ell_r)\right)$$

其中 $a_{r,j} = \prod_{i \leq j} c_{r,i}$ 是前缀累积存活概率。

### 2.2 算法步骤

```
Algorithm 1: Hardware-Aware Prefix Scheduler

输入: 活跃请求 r∈{1,...,R}
      校准后置信度 c_{r,1},...,c_{r,γ}
      Profiled SPS(B) 查找表

1. 对每个请求 r，计算前缀存活概率:
   a_{r,j} ← ∏_{i≤j} c_{r,i}   (j=1,...,γ)

2. 构建候选空间 E ← {(r, j) | a_{r,j} > 0}
   按 a_{r,j} 降序排列

3. 初始化:
   ℓ_r ← 0 (所有请求)
   B ← R     (起始 batch = 每请求的 bonus token)
   τ* ← R    (起始期望接受 = 每请求 1 个 bonus)
   Θ_best ← R · SPS(R)

4. 按 a_{r,j} 从高到低贪心遍历:
   对每个 (r, j):
     ℓ_r ← j
     B ← B + 1
     τ* ← τ* + a_{r,j}
     Θ_current ← τ* · SPS(B)     ← O(1) 查表

     if Θ_current > Θ_best:
         Θ_best ← Θ_current
         ℓ*_r ← ℓ_r
     else:
         break   ← early-stop: 后续候选置信度更低，不可能改善

5. 返回 {ℓ*_r}
```

**核心直觉**：
- 按置信度从高到低贪心选择 token（高置信度的优先获得验证资格）
- 每增加一个验证 token → $B \uparrow$ → $\text{SPS}(B) \downarrow$（每步变慢）
- 权衡：$\tau^*$ 的提升 vs SPS 的下降
- 当吞吐不再改善时停止——低置信度的后缀不值得拖慢整个 batch

---

## 3. 具体数值例子

假设 4 个并发请求，$\gamma = 5$：

### Step 1: Confidence Head 输出（校准后）

```
请求 1 (代码生成):   c = [0.95, 0.90, 0.85, 0.70, 0.50]
请求 2 (代码生成):   c = [0.92, 0.88, 0.80, 0.65, 0.45]
请求 3 (闲聊):       c = [0.80, 0.60, 0.40, 0.25, 0.15]
请求 4 (闲聊):       c = [0.75, 0.55, 0.35, 0.20, 0.10]
```

### Step 2: 累积存活概率 $a_{r,j}$

```
位置 j:            1      2      3      4      5
请求 1:           0.950  0.855  0.727  0.509  0.254
请求 2:           0.920  0.810  0.648  0.421  0.189
请求 3:           0.800  0.480  0.192  0.048  0.007
请求 4:           0.750  0.413  0.144  0.029  0.003
```

### Step 3: 构建候选空间，按 $a_{r,j}$ 降序

```
排名  (r,j)   a_{r,j}
 #1   (1,1)   0.950
 #2   (2,1)   0.920
 #3   (1,2)   0.855
 #4   (2,2)   0.810
 #5   (3,1)   0.800
 #6   (4,1)   0.750
 #7   (1,3)   0.727
 #8   (2,3)   0.648
 #9   (1,4)   0.509
#10   (3,2)   0.480
#11   (2,4)   0.421
#12   (4,2)   0.413
#13   (1,5)   0.254
#14   (3,3)   0.192
#15   (2,5)   0.189
#16   (4,3)   0.144
#17   (3,4)   0.048
#18   (4,4)   0.029
#19   (3,5)   0.007
#20   (4,5)   0.003
```

### Step 4: 贪心调度（结合 profiled SPS 曲线）

假设 SPS(B) 查找表：

```
B:         4     5     6     7     8     9    10    11    12
SPS(B):  100    98    95    91    86    80    73    65    56
```

贪心遍历：

```
初始: B=4, τ*=4, Θ_best=4×100=400, ℓ*=[0,0,0,0]

 #1 a(1,1)=0.950: B=5, τ*=4.95, Θ=4.95×98=485.1 > 400 → ℓ₁=1
 #2 a(2,1)=0.920: B=6, τ*=5.87, Θ=5.87×95=557.7 > 485  → ℓ₂=1
 #3 a(1,2)=0.855: B=7, τ*=6.73, Θ=6.73×91=612.0 > 557  → ℓ₁=2
 #4 a(2,2)=0.810: B=8, τ*=7.54, Θ=7.54×86=648.0 > 612  → ℓ₂=2
 #5 a(3,1)=0.800: B=9, τ*=8.34, Θ=8.34×80=667.2 > 648  → ℓ₃=1
 #6 a(4,1)=0.750: B=10,τ*=9.09, Θ=9.09×73=663.6 < 667  → break!
```

最终结果：

```
请求 1 (代码):  ℓ*₁ = 2  [████████░░░]  验证 2/5 → 节省 60%
请求 2 (代码):  ℓ*₂ = 2  [████████░░░]  验证 2/5 → 节省 60%
请求 3 (闲聊):  ℓ*₃ = 1  [████░░░░░░░]  验证 1/5 → 节省 80%
请求 4 (闲聊):  ℓ*₄ = 0  [░░░░░░░░░░░]  验证 0/5 → 节省 100%

总验证 token = 5（含 bonus: 4×1 + 2+2+1+0 = 9）
固定验证:     4 × (1+5) = 24 token
节省:         62.5% 的验证计算
```

**关键观察**：
- 代码类请求（高置信度）获得更多验证机会
- 闲聊类请求（低置信度）只验证最前面的 token 甚至不验证
- `a(4,1)=0.750` 虽然绝对值不低，但排在 #6，此时 SPS(9)→SPS(10) 的跌幅太大，边际收益为负

---

## 4. 工程挑战与生产适配

### 4.1 锯齿 SPS 的局部最优问题

平滑 SPS 假设下，Algorithm 1 的 early-stopping 能找到全局最优（$\Theta(B)$ 单峰）。但真实 SPS 是锯齿状的——$\Theta(B)$ 可能先降再升，导致 early-stopping 陷入局部最优。

论文解决方案（Section 5.2）：**生产部署中移除 early-stopping break**，做**无约束全局搜索**。但直接做全局搜索会破坏因果性（non-anticipating property）。

### 4.2 异步调度：解决因果性问题

无约束全局搜索意味着决策时可能"看到"未来的 token 值——这是推测解码必须避免的。

生产部署方案——**异步调度**（论文 Section 5.2）：

```
时间线:
  Step t-2          Step t-1          Step t (当前)
  ─────────         ─────────         ─────────
  生成 c_{r,j}      用 t-2 的置信度   严格用当前置信度
  存入历史缓存      估算容量 K         排序候选token
                    决定截断长度       Top-K 准入
                    
  历史预测用于       ◄────────────────► 置信度排序用
  确定容量上限 K     异步解耦（2步延迟）   最新的值保证保序
```

**两层设计**：
- **截断长度 K**（容量上限）：用两步前的历史置信度估算 → 避免因果泄露
- **候选排序**：严格用最新的校准置信度 → 保证高置信度 token 优先验证

```
核心原则:
  "the most confident draft tokens are always prioritized for verification"
  最高置信度的草稿 token 始终被优先验证
```

### 4.3 ZOS（Zero-Overhead Scheduling）兼容

ZOS 要求当前 step 的 batch size 在下个 step 开始前就确定，而同步调度会卡住 GPU 流水线。异步设计**完全隐藏调度延迟**，与 ZOS 无缝集成。

---

## 5. 静态阈值 vs 硬件感知调度

论文实验（Figure 5）展示了静态阈值在不同领域的表现差异：

```
置信度阈值 sweep（Qwen3-4B）:

        阈值 0       →      阈值 0.6
Math:  接受率 76.9%   →   92.5%（接受 tokens 更多被保留）
Code:  接受率 67.6%   →   92.0%
Chat:  接受率 45.7%   →   95.7%（pruning 最显著，从浪费到高效）
```

静态阈值的问题：对所有请求一刀切，不考虑系统负载。阈值 0.6 在低负载时过于严格（浪费了本可以免费验证的 token），在高负载时可能又不够严格（仍浪费 batch capacity）。

硬件感知调度：负载低时（SPS 比率接近 1），几乎所有高置信 token 都会获得验证资格；负载高时（SPS 比率低），只验证最高置信的前缀——**负载感知的动态决策**。

---

## 6. 论文对调度的总结

> "*By preventing severe throughput degradation under strict interactivity constraints, DSpark enables performance tiers that were previously unattainable, shifting the Pareto frontier of our serving system.*"

在 DeepSeek-V4 生产环境中，DSpark 相比 MTP-1 baseline：
- **V4-Flash**：per-user 生成速度提升 **60%–85%**
- **V4-Pro**：per-user 生成速度提升 **57%–78%**
- 在严格 SLA 约束下（Flash 120 TPS、Pro 50 TPS）突破 baseline 的性能悬崖

---

## 7. 关键代码文件索引

| 文件 | 作用 |
|------|------|
| `deepspec/eval/dspark/draft_ops.py:82-93` | `_confident_prefix_length` — 静态阈值前缀截断（离线评估用） |
| `deepspec/eval/dspark/draft_ops.py:96-153` | `build_dspark_proposal` — 推理 proposal 完整流程 |
| `deepspec/eval/dspark/confidence_head.py:30-171` | `PerPositionConfidenceMetrics` — ECE/AUROC/Brier 计算 |
| `deepspec/eval/dspark/confidence_head.py:313-606` | `ConfidenceHeadRecorder` — 可靠性图 + TensorBoard |
| `deepspec/eval/dspark/evaluator.py:36-66` | Evaluator 中 Confidence Head 集成 |
| `deepspec/modeling/dspark/loss.py:40-57` | `tau_probabilistic` — 训练时 τ 计算 |
| DSpark 论文 Section 3.2.2 | Algorithm 1 — 硬件感知前缀调度器 |
| DSpark 论文 Section 5.2 | 异步调度 + ZOS 适配 + 锯齿 SPS 全局搜索 |
