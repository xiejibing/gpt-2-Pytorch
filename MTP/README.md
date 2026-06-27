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
