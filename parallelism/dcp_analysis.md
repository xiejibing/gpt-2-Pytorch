# vLLM Decode Context Parallelism (DCP) 深度分析

> 本文档从源码层面拆解 vLLM 中 DCP 的实现机理：KV cache 如何跨卡切分、一次 decode 的通信节奏、
> LSE 为什么是跨卡拼 softmax 的关键、以及各种变体与约束。包含数学推导、具体数字例子与关键代码引用（`file:line`）。
>
> 分析基于本仓库当前分支（含 PCP / MRv2 等新特性），部分实现细节与上游 main 可能有差异。
>
> 设计文档见 `docs/serving/context_parallel_deployment.md`。

---

## 目录

- [第一部分：DCP 解决什么问题](#第一部分dcp-解决什么问题)
- [第二部分：KV cache 怎么切](#第二部分kv-cache-怎么切)
- [第三部分：一次 decode 的通信节奏](#第三部分一次-decode-的通信节奏)
- [第四部分：为什么必须 AllGather Q](#第四部分为什么必须-allgather-q)
- [第五部分：LSE —— 跨卡拼 softmax](#第五部分lse--跨卡拼-softmax)
- [第六部分：为什么不直接搬 score](#第六部分为什么不直接搬-score)
- [第七部分：reduce_scatter 的角色](#第七部分reduce_scatter-的角色)
- [第八部分：变体与优化](#第八部分变体与优化)
- [第九部分：约束与边界情况](#第九部分约束与边界情况)
- [附录：关键代码索引](#附录关键代码索引)

---

# 第一部分：DCP 解决什么问题

## 1.1 核心心智模型：KV cache 的两个可切维度

一个请求的 KV cache 大小是 `num_kv_heads × T`（`T` = 上下文长度）。可切分的维度只有两个：

```
                        KV cache
        ┌──────────────────┬──────────────────┐
        │  沿 head 维切      │  沿 token 维切     │
        │  = 普通 TP        │  = DCP            │
        ├──────────────────┼──────────────────┤
        │ 零通信（天然对齐）  │ 需要集合通信        │
        │ head 数有限 → 复制  │ 无上限，随 T 增长   │
        └──────────────────┴──────────────────┘
```

TP 沿 head 维切，但 **head 数由模型结构决定，是有限的**。当 `tp_size > num_kv_heads` 时，多出来的
TP rank 只能**复制**同一份 KV：

- DeepSeek-R1（MLA，1 个 kv head）`-tp 8` → KV **8× 复制**
- Qwen3-235B（4 个 kv head）`-tp 8` → KV **2× 复制**

DCP 沿 token 维再切一刀，把这份复制消掉。

**关键点：`dcp_size` 不扩大 world size。** 它复用已经存在的 TP rank（`config/parallel.py:353`）：

> Number of ranks that shard the decode KV cache. DCP does not expand the process
> world size. Without PCP, DCP reuses TP ranks.

所以 `-tp 8 -dcp 8` 仍然是 8 张卡，只是每张卡的 KV 从 `1×T` 变成 `1/8×T`，省出的显存全部变成 batch size。

## 1.2 收益的本质：容量，而非计算

DCP **不减少单卡的 attention 计算量**——因为 head 维会 gather 回来（见第四部分）：
原来每卡算 `H_local × T`，现在算 `H_local×N × T/N`，总 FLOPs 相同。它换来的是 **KV 容量**，
从而能开更大的 batch。

唯一的例外是 MLA：KV 只有一个共享 latent，每卡要读的 KV 从 `T` 降到 `T/N`，
这是实打实的访存收益（decode 是 memory-bound 的）。

## 1.3 约束：dcp ≤ tp / num_kv_heads

想消掉复制，DCP 组必须是「那些恰好互为 replica 的 rank」。非 MLA 模型：

```python
# vllm/config/model.py:1445
decode_context_parallel_size = parallel_config.decode_context_parallel_size
if decode_context_parallel_size > 1 and not self.use_mla:
    total_num_kv_heads = self.get_total_num_kv_heads()
    if tensor_parallel_size <= total_num_kv_heads:
        raise ValueError(...)          # tp 必须 > kv head 数，否则没有复制可消
    max_dcp_size = tensor_parallel_size // total_num_kv_heads
    if decode_context_parallel_size > max_dcp_size:
        raise ValueError(...)
    num_q_per_kv = total_num_attention_heads // total_num_kv_heads
    if num_q_per_kv % decode_context_parallel_size != 0:
        raise ValueError(...)          # 见第四部分的 head 归属
```

MLA 模型不受这条限制（1 个 kv head 时 `dcp` 可以等于 `tp`）。

TP 侧的校验（`config/parallel.py:558`）：

```python
if pcp == 1:
    if tp % dcp != 0:
        raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")
elif dcp not in (1, pcp, tp * pcp):
    raise ValueError("When PCP is enabled, DCP must be disabled, span the PCP "
                     "axis, or span the full TP x PCP axis. ...")
```

---

# 第二部分：KV cache 怎么切

## 2.1 交错分片（interleaved sharding）

DCP 不用「连续切块」，而是**交错**——这是能支持 decode 持续追加 KV 的关键。

```python
# vllm/v1/attention/backends/utils.py:1094
def get_dcp_local_seq_lens(seq_lens, dcp_size=1, dcp_rank=None,
                           cp_kv_cache_interleave_size=1):
    base = seq_lens_tiled // cp_kv_cache_interleave_size // dcp_size \
           * cp_kv_cache_interleave_size
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(remainder - rank_offsets * cp_kv_cache_interleave_size,
                           0, cp_kv_cache_interleave_size)
    return base + remainder
```

记 `I = cp_kv_cache_interleave_size`，`N = dcp_size`：

> **全局 token `t` 归属 rank `(t // I) % N`。**

`I=1` 就是最朴素的 `token i → rank i % N`；`I=block_size` 是块级对齐。
`I` 的约束：`I ≤ block_size` 且 `block_size % I == 0`（`config/vllm.py:3126`）。

**为什么必须交错**：decode 每步新生成一个 token，交错布局让这个 token 天然落到一个确定的
owner rank 上，不需要任何数据重排；调度器也不需要理解「哪个 token 在哪张卡」。

## 2.2 虚拟块 = 物理块 × N

这是 DCP 能无侵入接入 vLLM 调度器的核心技巧：

```python
# vllm/v1/core/kv_cache_utils.py:668
def resolve_dcp_kv_block_size(spec, dcp_world_size):
    """Return the token span of a cache block under DCP."""
    if all(isinstance(s, AttentionSpec) for s in iter_layer_specs(spec)):
        return spec.block_size * dcp_world_size   # attention 才缩放
    return spec.block_size
```

```
   调度器眼里的"虚拟块"（512 token, N=8, block_size=64）
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │ r0 │ r1 │ r2 │ r3 │ r4 │ r5 │ r6 │ r7 │   每个 rank 各存 64 个 token
   └────┴────┴────┴────┴────┴────┴────┴────┘
     ▲
     └── 一个虚拟块 = N 张卡各一块物理 page
```

于是：

| 量 | 非 DCP | 有 DCP |
|---|---|---|
| `max_num_blocks_per_req` | `cdiv(max_len, block_size)` | `cdiv(max_len, block_size × N)`（`kv_cache_interface.py:532`） |
| 显存核算的分母 | `block_size` | `block_size × N`（`kv_cache_interface.py:564`） |
| `KVCacheManager.block_size` | `block_size` | `block_size × N`（`single_type_kv_cache_manager.py:106-110`） |

**注意纠正一个常见误解**：block table 的**内容在 DCP 组内各 rank 上逐个相同**（同一批 logical block id，
因为调度器只有一个），但**表的宽度缩了 N 倍**。每个 rank 的物理块数和总槽位都只有原来的 `1/N`——
这才是 KV 显存省下 `1/N` 的出处。

只有全注意力 spec（含 MLA）参与缩放；Mamba / sliding window 等保持每卡完整状态：

```python
# vllm/v1/core/kv_cache_utils.py:695
def dcp_world_size_for_kv_cache_spec(spec, dcp_world_size):
    """Full-attention KV (including MLA) is sharded across DCP ranks, so prefix
    hashing and manager block_size use the process DCP size. Other specs keep
    replicated per-rank state (Mamba, sliding window, chunked-local) and must
    keep dcp_world_size=1 even when the process runs with DCP > 1."""
```

调度器侧的 block size 由 `resolve_kv_cache_block_sizes` 统一（`kv_cache_utils.py:717`）：
单 group 时 `scheduler_block_size = cache_config.block_size × dcp`；多 group 时取各 group
有效块大小的 LCM。

## 2.3 写路径：slot mapping 就地反解

写 KV 时，非本 rank 的 token 直接写 `PAD_SLOT_ID`，等于丢弃：

```python
# vllm/v1/worker/block_table.py:447 (V1 runner)
virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
virtual_block_indices = pos // virtual_block_size
virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) \
           % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
local_block_offsets = (virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)) \
                      * CP_KV_CACHE_INTERLEAVE_SIZE \
                      + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)
block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
slot_ids = tl.where(is_local, slot_ids, PAD_ID)          # :476
```

V2 runner 有一份等价实现（`vllm/v1/worker/gpu/block_table.py:333`）。读侧在 attention kernel
里做同一套反解（`mla/sparse_utils.py:153-168`），非本 rank 的 token 置 `-1` 变无效：

```python
owning_rank = (tok // DCP_INTERLEAVE) % DCP_SIZE
is_remote = owning_rank != DCP_RANK
local_idx = (tok // (DCP_SIZE * DCP_INTERLEAVE)) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
```

**位置编号（`positions`）仍然是全局的**——这是整个设计的枢纽：kernel 和块表都只知道全局位置，
交错布局对它们是透明的。

---

# 第三部分：一次 decode 的通信节奏

默认后端（`dcp_comm_backend="ag_rs"`）每一层都是固定的三拍：

```
AllGather Q ──► local attention over 1/N KV ──► AllGather LSE ──┐
                                                               ├─► 本地加权修正 ──► ReduceScatter out
                                                               ┘
```

## 3.1 第一拍：AllGather Q

```python
# vllm/v1/attention/backends/flash_attn.py:1522
query_across_dcp = get_dcp_group().all_gather(query, dim=1)      # dim=1 = head
```

MLA 路径走 `dcp_manager.query_gather`（`dcp.py:1582`），或 `dcp_q_replicate` 时直接跳过
（`mla_attention.py:1103`）。

规模：`[T_q, H_local × N, D]`。**decode 时 `T_q` = batch 中的请求数（每请求 1 个 token），
与上下文长度无关**——这是整套节奏 decode 友好的根本原因。

## 3.2 第二拍：Compute

只对本地的 `1/N` context 做 attention，`causal=False`（query 整块都在 context 之后）：

```python
# vllm/v1/attention/backends/flash_attn.py:1587-1609
context_attn_out, context_lse = flash_attn_varlen_func(
    q=query_across_dcp, k=key_cache, v=value_cache,
    seqused_k=attn_metadata.dcp_context_kv_lens,
    max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
    causal=False,
    return_softmax_lse=True,
)
```

产出**局部 softmax 的分子和分母**：`out_r`（已局部归一化）+ `lse_r`。
所以后端必须支持返回 LSE：

```python
# vllm/v1/worker/cp_utils.py:48
assert layer_impl.need_to_return_lse_for_decode, \
    "Decode Context Parallelism (DCP) requires attention implementations to " \
    "return the softmax LSE during decode, ..."
```

## 3.3 第三拍：AllGather LSE + ReduceScatter out

```python
# vllm/v1/attention/ops/dcp.py:439
lses = cp_group.all_gather(cp_attn_lse, dim=0)          # [N, T, H]，小张量
out, lse = correct_attn_out(out, lses, cp_group.rank_in_group, ctx)
out = cp_group.reduce_scatter(out, dim=1)               # :475，大头
```

`correct_attn_out` 做两件事（kernel 见 `dcp.py:159-200`）：算出全局 LSE，再把自己那份输出
乘上 `factor = exp(lse_local - lse_global)`。于是剩下的跨卡求和恰好就是一次 reduce——
**用 RS 顺带把 head 维切回 TP 归属**。

数学细节见第五部分，reduce_scatter 的角色见第七部分。

## 3.4 收尾：与本地 query 段合并

第三拍只合并了 **context 部分**；当前 token 之间的 causal attention 是另算的一段，
两段用 online softmax 再合并一次：

```python
# flash_attn.py:1642
merge_attn_states(output, context_attn_out_cor, context_lse_cor,
                  query_attn_out, query_lse)
```

MLA 侧对应 `mla_attention.py:1131` 的 `self.dcp_manager.combine(...)`。

## 3.5 与其它并行策略的节奏对比

| | 每层通信 | 搬什么 | 依赖结构 |
|---|---|---|---|
| DCP (`ag_rs`) | 3 次固定集合通信 | Q（小）、LSE（小）、out | 无流水、无 ring、每层独立 |
| DCP (`a2a`) | 2 次 | 同上（LSE 打包进 out dtype） | 同上 |
| ring attention | 点对点 N-1 轮 | **KV**（大） | 有依赖链，必须流水 |

DCP 之所以"simple"，是因为 **KV 已经被交错布局预置在各卡显存里，一个字节都不用上网络**；
ring attention 要搬的恰是 KV，所以才需要流水与调度。

---

# 第四部分：为什么必须 AllGather Q

这是最容易漏掉的一环。答案在于 **Q 和 KV 用了两套不同的切法**。

## 4.1 head 归属：Q 按 tp_rank，KV 按 replica

```python
# vllm/model_executor/layers/linear.py:1049
self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

# vllm/model_executor/layers/linear.py:1295-1298
if loaded_shard_id == "q":
    shard_rank = self.tp_rank                              # Q: 按 tp_rank 连续切
else:
    shard_rank = self.tp_rank // self.num_kv_head_replicas  # KV: 连续 replicas 共享同一 kv head
```

即 **KV head 按 `tp_rank // replicas` 分配，相邻的 replica 共享同一个 kv head**。
而 DCP 组是从 TP 轴上连续切的（`parallel_state.py:2133` 的 `reshape(-1, dcp_size)`），
所以 **DCP 组恰好就是同一个 kv head 的 replica 集合**。

具体例子：`tp=8, num_q_heads=32, num_kv_heads=4, dcp=2`（`replicas = 8/4 = 2`）：

| rank | q heads | kv head |
|---|---|---|
| 0 | `q[0:4]` | 0 |
| 1 | `q[4:8]` | 0 |
| 2 | `q[8:12]` | 1 |
| … | | |

`q[0:8]` 这 8 个 q head 都归属 kv head 0。DCP 组 `{0,1}` 切的是**同一份 KV 的 token 维**，
但它们各自只持有 `q[0:4]` 和 `q[4:8]`。

## 4.2 只用本卡的 Q 会坏在哪

```
        tokens:  0    1    2    3    4    5  ...
rank 0:          q[0:4] × {t0, t2, t4}    → (out₀, lse₀)
rank 1:          q[4:8] × {t1, t3, t5}    → (out₁, lse₁)
```

问题不是"数据不全所以精度差"，而是**这两份 partial 描述的是不同的 head**。
跨卡归约 `Σ_r w_r · out_r` 要成立，前提是所有 rank 算的是**同一组 head**、只有 token 分片不同；
否则 `out_r` 根本不在同一个 head 空间里，求和没有意义。

而 `q[0:4]` 在 `t1,t3,t5...` 上的那块结果，算它需要那些 token 的 KV，那些 KV 只在 rank 1 手里。
所以补全只能靠搬东西。

## 4.3 两条路，DCP 选了搬 Q

**路 A：搬 Q（当前设计）。** AG 之后每张卡都有 `q[0:8]`，各自对自己的 token 分片算完整 softmax 的局部：

```
rank 0:  q[0:8] × {t0, t2, t4, ...}  → (out₀, lse₀)
rank 1:  q[0:8] × {t1, t3, t5, ...}  → (out₁, lse₁)
```

现在两份 partial 的 head 集合相同、token 分片互补，LSE 加权合并才有定义。

**注意这不是"为了冗余计算才 gather"**：与不用 DCP 的 TP 相比，单卡 FLOPs 一点没多——
原来是 `4 heads × T tokens`，现在是 `8 heads × T/2 tokens`。AG Q 做的是把计算
沿 **head 维 → token 维**重新分配，换来 KV 显存 `÷N`。

**路 B：搬 KV。** 每张卡把别人的 token 分片也拿到，各算各的 4 个 head——结果是精确的，
连 LSE 合并都省了。但每层要搬 ~全量的 context KV，正是 DCP 想避免的东西。
ring attention 走的就是这条路（用流水避免完整物化，代价是依赖链）。

选 A 的理由是量级差：decode 时 Q 是 `[T_q, H, D]`，`T_q` = batch 请求数，**与上下文长度无关**；
KV 是 `T_ctx × d_kv`，随上下文线性增长。

**旁证**：`dcp_q_replicate`（`config/parallel.py:376`）的做法是在组内冗余重算小的 q 投影，
让每张卡本地就拥有完整 head 集合，从而跳过这次 AG。它的存在恰好说明
**AG Q 的语义是"补齐 head 集合"，而不是"搬运某种不可复制的数据"**。

---

# 第五部分：LSE —— 跨卡拼 softmax

## 5.1 局部归一化丢掉了什么

一个具体反例。score `s = [0,1,2,3]`（4 个 context token），`v = [1,0,0,0]`。
正确答案：

```
softmax([0,1,2,3]) = [0.032, 0.087, 0.237, 0.644]
out = 0.032 × 1 = 0.032
```

按 DCP 切成两个 rank：

```
rank 0 (t0,t1):  本地 softmax = [0.269, 0.731]   out₀ = 0.269×1 + 0.731×0 = 0.269
rank 1 (t2,t3):  本地 softmax = [0.269, 0.731]   out₁ = 0.269×0 + 0.731×0 = 0.000
```

两卡平均：`(0.269 + 0)/2 = 0.134`，正确答案是 **0.032，差了 4 倍**。

荒谬之处在于：**rank 1 的输出是 0，但它握着 88% 的注意力权重**（`0.237 + 0.644 = 0.881`）。

> **部分输出本身，对「这段 context 整体占多大分量」完全不携带信息。**

两个局部输出长得再正常，你也无法判断该 1:1 还是 1:7.4 加权。任何只基于部分输出
的加权方案都不可能正确。

## 5.2 缺的信息就是"分母"

一次 softmax 只能归一化一次。分片计算时每片被迫做了"除以自己的局部总和"：

$$out_r = \sum_{j \in r} \frac{e^{s_j}}{e^{lse_r}} v_j, \qquad lse_r = \log \sum_{j \in r} e^{s_j}$$

`lse` = **LogSumExp** = softmax 分母的对数。一个 rank 返回 `(out_r, lse_r)`,
本质上等价于返回 **(「我这段的分子」, 「我这段的分母」)**，只不过用 log 缩放的形式存着以免溢出。

**为什么用 log 存而不是直接存 `Σ e^s`**：数值范围。`e^{100}` 在 fp32 下就是 `inf`，
log 域可以把任意大的分数压到一个小范围，也让合并公式对称稳定。

## 5.3 合并推导 + 数字例子

关键是**乘上 `e^{lse_r}` 再除掉 `e^{lse_r}`**，把"除以本地分母"换成"除以全局分母"：

$$
\begin{aligned}
out_{global} &= \sum_j e^{\,s_j - lse_{global}}\, v_j \\
&= \sum_r \sum_{j\in r} e^{\,s_j - lse_{global}}\, v_j \\
&= \sum_r \underbrace{e^{\,lse_r - lse_{global}}}_{\text{factor}_r} \cdot
   \underbrace{\sum_{j\in r} e^{\,s_j - lse_r}\, v_j}_{\text{这正是 } out_r} \\
&= \sum_r \text{factor}_r \cdot out_r
\end{aligned}
$$

整个过程是**代数上严格等价**地重写了一遍全局 softmax，没有近似。

数字例子（`s = [0,1,2,3]`，`v = [1,2,4,8]`）：

| | rank 0 (t0,t1) | rank 1 (t2,t3) |
|---|---|---|
| `e^s` | `[1, 2.71828]` | `[7.38906, 20.08554]` |
| 本地分母 `e^{lse_r}` | **3.71828** | **27.47459** |
| `lse_r` | 1.31326 | 3.31326 |
| 本地权重 | `[0.26894, 0.73106]` | `[0.26894, 0.73106]` |
| **`out_r`** | `0.269×1+0.731×2 = 1.73105` | `0.269×4+0.731×8 = 6.92424` |

合并（对应 `dcp.py:159-200`）：

```
lse_max    = 3.31306
Σexp(lse−max) = e^{−2} + e^0 = 1.13534
lse_global = 3.31306 + log(1.13534) = 3.44019     ← 等于 log(e⁰+e¹+e²+e³) ✓

factor_0 = exp(1.31326 − 3.44019) = 0.119203
factor_1 = exp(3.31306 − 3.44019) = 0.880797
                         相加 →  1.000000     ← 恰好为 1

out = 0.119203 × 1.73105 + 0.880797 × 6.92424 = 0.206359 + 6.098837 = 6.305196
```

与直接做全局 softmax 对照：权重 `[0.03206, 0.08716, 0.23686, 0.64392]`，

```
0.03206×1 + 0.08716×2 + 0.23686×4 + 0.64392×8 = 6.305188   ✓
```

## 5.4 为什么 factor 之和必然是 1

因为 `Σ_r e^{lse_r} = e^{lse_{global}}`（全局分母的定义），所以：

$$\sum_r \text{factor}_r = \sum_r e^{\,lse_r - lse_{global}} = 1$$

代码因此不需要在最后再除一次。这也说明 **factor 只由 LSE 决定，与 `out` 的数值大小无关**——
5.1 那个反例里 rank 1 的 out 是 0，权重照样是 0.88。

## 5.5 数值细节

kernel 实现（`dcp.py:157-200`）：

```python
lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)  # NaN/inf → -inf
lse_max = tl.max(lse, axis=0)
lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)   # 全 -inf 时避免 -inf-(-inf)=NaN
lse -= lse_max
lse_exp = tl.exp(lse)                # IS_BASE_E 为真
lse_acc = tl.sum(lse_exp, axis=0)
lse = tl.log(lse_acc) + lse_max

lse_offset = lse_idx * lses_stride_N + ...              # 本 rank 自己那一份
factor = tl.exp(lse_tmp - lse_global)
output = output * factor
output = tl.where(factor == 0.0, 0.0, output)            # 0 × 任意值 的兜底
```

三个要点：

1. **底数可以是 2**。GPU 上 `exp2`/`log2` 比 `exp`/`log` 便宜，FlashInfer 返回 base-2 的 LSE
   （`flashinfer.py:292` 传 `is_lse_base_on_e=False`），kernel 用 `IS_BASE_E` 切换
   `exp` 与 `exp2`（`dcp.py:162-169`）。
2. **空分片必须 mask 成 −inf**。某 rank 本地 context 为 0 时其 LSE 是未初始化值，
   参与 `max` 或 `Σexp` 会把结果污染成 NaN：

   ```python
   # vllm/v1/attention/ops/dcp.py:438
   mask_dcp_empty_shards_(cp_attn_lse, seq_lens, query_start_loc)
   ```
3. **`-inf` 是语义正确的结果**：没有 token 的上下文其 logsumexp 就是 −inf，
   对应权重 `exp(-inf - lse) = 0`，自动退出求和。

CPU 参考实现见 `dcp.py:518` 的 `_lse_weighted_combine`，可直接用于数值对照测试。

## 5.6 LSE 在别处的同类用法

同一套数学在 vLLM 里反复出现，理解一个就理解全部：

| 场景 | 分片维度 | 代码 |
|---|---|---|
| DCP | 跨 rank 的 token 分片 | `dcp.py:418-449` |
| split-KV | 同一序列的 KV 切给多个 CTA | FA 的 `num_splits` |
| 本地段合并 | context 段 vs 当前 query 段 | `flash_attn.py:1642` `merge_attn_states` |
| MLA chunked prefill | 跨 chunk | `mla_attention.py` |

---

# 第六部分：为什么不直接搬 score

"先把 score 收齐再算全局 softmax"数学上完全可行，问题在**数据量**，差三个数量级。

## 6.1 充分统计量视角

分布式求平均的类比：不会把原始数据传到一台机器再算，而是每台返回 `(Σx, n)`，汇总 `ΣΣx / Σn`。
因为**平均是可结合归约**，原始数据里除了这两个统计量以外的信息对结果都没用。

- `n` ↔ `lse`（取了对数）
- `Σx` ↔ `e^{lse_r} · out_r`

注意力输出只需要这两个**可加**的量。score 矩阵是它们的信息超集——每个 token 各自的权重
本地乘完 V 就没用了，从来不需要跨卡。

## 6.2 量级对比

per (token, head) 的元素数：

| 张量 | 元素数 |
|---|---|
| score 行 | `T_ctx` 个 |
| LSE | **1 个** |
| out | `D` 个（128） |

**LSE 是 score 行的 `T_ctx → 1` 压缩。**

以 DeepSeek-R1 量级配置估算（`tp=8, dcp=8`，128 个 q head，MLA latent 576，
`T_ctx = 128K`，batch `T_q = 128`，bf16，**一层**）：

| | 每卡数据量 |
|---|---|
| 本卡要读的 KV（MLA latent） | `16384 × 576` ≈ **19 MB** |
| 本卡要算的 score | `128 × 128 × 16384` ≈ **537 MB** |
| AG Q 流入 | `128 × 128 × 128` ≈ 4 MB |
| AG LSE 流入 | `128 × 128` ≈ **32 KB** |
| RS out 流入 | ≈ 0.5 MB |
| 全组 score 矩阵 | `128 × 128 × 131072` ≈ **4.3 GB** |

`ag_rs` 每层每卡流几 MB 就完事（**与 `T_ctx` 无关**）；搬 score 每卡要收 ~3.8 GB。**差 ~1000 倍。**

注意：搬 score 理论上确实能省掉那次 AG Q——省 4 MB，换来 3.8 GB，等于没有。

## 6.3 二次伤害：物化 score

搬 score 意味着**先把 score 矩阵物化到显存**，而这正是 FlashAttention 存在的全部意义所在
（逐 tile 在寄存器里算完就扔）。强行物化：

- 每卡写 537 MB + 读 537 MB 的 HBM 流量，是它读 KV（19 MB）的 **~50 倍**
- **还没上网络，光在显存里倒一遍就已经输了**

## 6.4 与 ring attention 搬 KV 的对比

$$\frac{\text{score}}{\text{KV}} = \frac{T_q \times H}{d_{kv}}$$

decode（`T_q` = batch 大小）下这个比值 ≫ 1，所以正确的成本排序是：

> **score ＞ KV ＞ (out, LSE)**

搬 KV 都比搬 score 划算——而 DCP 连 KV 都不搬，因为 KV 已经被交错布局预置在各卡显存里了。
只有当 `T_q` 很大（prefill）时 score 交换才可能有利，但那时 DCP 换了策略：
MLA chunked prefill 直接 gather KV 再 reorganize（`mla_attention.py:3074`），而非交换 score。

---

# 第七部分：reduce_scatter 的角色

## 7.1 语义

`reduce_scatter` = **「先跨卡求和，再把和切成 N 份，每张卡只留自己那一份」**。
数学上等价于 `all_reduce` 之后本地切片，但通信量少一半（省掉 all-gather 那一半）。

在 DCP 里：

```
input:  [T, H_group, D]     ← 每张卡的 factor_r × out_r，head 维完整
output: [T, H_local, D]     ← rank r 只留第 r 个 head 切片
```

最小例子（1 token，2 head，D=1）：

```
rank 0 partial: [h0=1, h1=3]
rank 1 partial: [h0=2, h1=4]
                ─────────────────
       逐元素相加: [h0=3, h1=7]
                ─────────────────
   reduce_scatter: rank 0 拿 head 0 → 3
                   rank 1 拿 head 1 → 7
```

## 7.2 为什么这里能这么切

两个不变式（都由前面的机制建立）：

1. **所有 rank 的 partial 覆盖同一套 head**，逐元素相加才有意义。→ 由 **AG Q** 保证。
2. **切片顺序 = group rank 顺序**，所以"第 r 段"正好是 rank r 自己的 head。vLLM 的实现：

   ```python
   # vllm/distributed/device_communicators/base_device_communicator.py:295
   input_tensor = input_.movedim(0, dim).contiguous()   # head 维挪到第 0 维
   ```

   AG Q 时按 rank 顺序拼接（`[rank0 的 head | rank1 的 head | ...]`），
   所以 RS 切回来的第 r 段恰好是 rank r 原本 TP 拥有的 head。

## 7.3 为什么不用 all_reduce

因为下游只要自己那片（o_proj 是 TP 切的）：

| | 每卡收到的输出 | 上例实际大小 |
|---|---|---|
| `all_reduce` | `[T, H_group, D]`（全 head，重复） | 4 MB |
| `reduce_scatter` | `[T, H_local, D]` | 0.5 MB |

通信量 `(N-1)/N × S` vs `2(N-1)/N × S`——**RS 恰好是 AR 的一半**，且少物化一份全 head 输出。

## 7.4 RS 之后 LSE 也要切片

输出只剩 `H_local` 个 head，LSE 得跟着切，否则 `merge_attn_states` shape 对不上：

```python
# vllm/v1/attention/ops/dcp.py:477-481
cp_num_heads = lse.shape[1] // cp_group.world_size
cp_rank = cp_group.rank_in_group
lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
```

---

# 第八部分：变体与优化

## 8.1 a2a：3 次通信 → 2 次

`dcp_comm_backend="a2a"`（`dcp.py:922` `dcp_a2a_lse_reduce`）把 AG(LSE) + RS(out) 合并成
一次 all-to-all，partial output 与 LSE **打包在同一个张量里**传输——fp16 时一个元素塞两个 LSE：

```python
# vllm/v1/attention/ops/dcp.py:587
def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
```

配置注释（`config/parallel.py:365`）：

> - "ag_rs": AllGather + ReduceScatter (existing behavior)
> - "a2a": All-to-All exchange of partial outputs + LSE, then combine with Triton
>   kernel. Reduces NCCL calls from 3 to 2 per layer for MLA models.

内核选择在 `MLADCPManager._init_combine`（`dcp.py:1489-1527`）。

## 8.2 对称内存直连（symm-mem direct）

再往上一级是绕过 NCCL：把 workspace 分配到对称内存（NVLink peer-accessible），
用 Triton kernel 直接读写对端显存 + multicast：

- `DirectDCPA2AWorkspace`（`dcp.py:1036`）
- `DirectDCPQGatherWorkspace`（`dcp.py:1166`）
- `DirectDCPKVGatherWorkspace`（`dcp.py:1322`）
- 公共部分 `DirectCPWorkspace`（`ops/cp_common.py:90`），能力探测 `_symm_mem_spans_group`

## 8.3 dcp_q_replicate：省掉第一拍

```python
# vllm/config/parallel.py:376
dcp_q_replicate: bool | None = None
"""Replicate the MLA query projection within each DCP group so decode can skip the
query all-gather.

With DCP the KV cache is sharded across the group, so the standard MLA decode path
all-gathers the query every step. Replicating the (small) query projection at load
time lets each rank materialize the full group-local head set and skip that
collective, at the cost of computing the projection redundantly on every rank
in the group."""
```

用（便宜的）冗余计算换掉（相对昂贵的）集合通信延迟。默认值由模型的
`verify_and_update_config` 里调用 `set_dcp_defaults` 决定（`config/parallel.py:575`），
本仓库中 `GlmMoeDsaForCausalLM` 传了 `comm_backend="a2a", q_replicate=True`。

## 8.4 PCP + DCP 共存

- DCP 组跨越 PCP 轴或整个 TP×PCP 块（`parallel_state.py:2133` 的 transpose 技巧）
- 合并换用 **all_reduce** 而非 RS（`dcp.py:1516-1522` 选 `cp_lse_ag_out_ar`）：
  组内按 **query 位置**切，每张卡都需要完整 head 维，RS 的"每卡恰好只留一片 head"前提不成立
- Q 的 gather 改走 TP 组，或在 `dcp == pcp` 时干脆省掉（`mla_attention.py:1094-1098`）
- MRv2 下 PCP+DCP 有额外限制：仅支持 sparse MLA 模型、要求 `dcp_comm_backend="ag_rs"`
  （`vllm/v1/worker/gpu/pcp_manager.py:159,175`）

---

# 第九部分：约束与边界情况

## 9.1 配置约束速查

| 项 | 约束 | 位置 |
|---|---|---|
| `tp % dcp == 0` | PCP 关闭时必需 | `config/parallel.py:558` |
| PCP 开启时 `dcp ∈ {1, pcp, tp×pcp}` | | `config/parallel.py:558` |
| 非 MLA：`dcp ≤ tp / num_kv_heads` | 且整除 GQA ratio | `config/model.py:1445` |
| 后端必须支持 | `supports_dcp` ClassVar | `vllm/v1/attention/backend.py:341,805` |
| 后端必须返回 LSE | `need_to_return_lse_for_decode` | `vllm/v1/worker/cp_utils.py:48` |
| `I ≤ block_size` 且 `block_size % I == 0` | | `config/vllm.py:3126` |
| sliding window / chunked local attn | 不支持 | `kv_cache_interface.py:845`、`single_type_kv_cache_manager.py:1346` |
| DBO（双 batch overlap） | 与 CP 互斥 | `config/vllm.py:3026` |
| HiSparse / return-routed-experts | 不支持 | `config/vllm.py:1650,1288` |
| PD 分离 + NIXL | `dcp>1` 仅 MLA | `config/vllm.py:1332` |
| MLA + MTP 且 `I > 1` | 需 impl 声明 `supports_mtp_with_cp_non_trivial_interleave_size` | `attention/backend.py:806` |
| MLA + `I > 1` | 不支持 varlen，强制 `reorder_batch_threshold = 1` | `attention/backend.py:645`、`mla/flashattn_mla.py:126` |

支持 DCP 的实现（`supports_dcp = True`）：FlashAttention（`flash_attn.py:1053`）、
FlashInfer（`flashinfer.py:1797`），以及 MLA 家族
（`flashinfer_mla.py:213`、`flashattn_mla.py:258`、`flashmla.py:210`、
`flashmla_sparse.py:636`、`flashinfer_mla_sparse.py:300`、`triton_mla.py:184`、
`cutlass_mla.py:106`、`tokenspeed_mla.py:157`、`rocm_aiter_mla.py:1426`）。

## 9.2 运行时边界

**空 shard**：短序列时某些 rank 本地 token 数为 0，其 LSE 是垃圾值 → `mask_dcp_empty_shards_`
强制置 `-inf`（`dcp.py:71`）。

**skip 分支必须 rank 无关**：`should_skip_dcp_context_attention` 只能看全局 context 长度，
不能看本 rank 的 local 长度——某卡本地为 0 而别卡仍有 context 是正常情况，
各 rank 必须走同一分支：

```python
# vllm/v1/worker/cp_utils.py:117-127
# Must be computed from rank-invariant inputs only (the global context lengths,
# NOT this rank's local share ...): every DCP rank must take the same branch.
```

这也是为什么 metadata 里同时存在 `dcp_tot_seq_lens`（全局）和
`dcp_local_seq_lens`（本卡）两个字段。

**prefix caching 对齐**：命中前缀必须是虚拟块整数倍。`scheduler_block_size = block_size × dcp`，
多 group 时取 LCM，hash 粒度再按 GCD 收敛（`kv_cache_utils.py:717-809`）。

**CUDA graph**：用 `dcp_dummy_context_len = dcp_size × I` 造假的 context 保证形状一致
（`cp_utils.py:58-72`），fake batch 里还要伪造 block table 行（`prepare_dcp_dummy_context_metadata`）。

**MLA prefill 的 KV all-gather**：本地 KV 是残缺的，需 gather 到放大 `1/N` 的 workspace
再 reorganize 成 TP 布局：

```python
# mla_attention.py:3074 附近；workspace 形状断言见 dcp.py:1610
workspace.shape[0] == max_gathered_tokens + max_gathered_tokens // world_size
```

`reorg_kvcache`（`mla_attention.py:2665`）负责把 `[T0_0, T0_1, ..., T0_4, pad, T1_0, ...]`
重排成 `[T0_0, T0_1, ..., T0_5, T1_0, T1_1, ...]`。

## 9.3 调参经验（来自设计文档）

> 对 decode context parallel，先把 `-tp` 加到性能满意，再加 `-dcp` 减少 KV 复制。

| 模型 | `tp` | KV 复制 | 建议 `dcp` |
|---|---|---|---|
| DeepSeek-R1（MLA, 1 kv head） | 8 | 8× | 8（完全消除） |
| Kimi-K2（MLA） | 16 | 16× | 16（完全消除）或 8（复制 2×，通信更省） |
| Qwen3-235B（4 kv head） | 8 | 2× | 2（完全消除） |

DCP 越大，KV 复制越少，但通信开销越大；且 DCP 通信只在 `dcp` 同一节点内时更便宜。

---

# 附录：关键代码索引

## 配置与校验

| 主题 | 位置 |
|---|---|
| `decode_context_parallel_size` 定义 | `vllm/config/parallel.py:353` |
| `dcp_comm_backend` / `dcp_q_replicate` / `cp_kv_cache_interleave_size` | `vllm/config/parallel.py:365,376,387` |
| TP/PCP/DCP 关系校验 | `vllm/config/parallel.py:558` |
| 模型侧约束（非 MLA） | `vllm/config/model.py:1445` |
| DBO / HiSparse / PD 互斥 | `vllm/config/vllm.py:3026,1650,1332` |
| interleave vs block_size | `vllm/config/vllm.py:3126` |

## 进程组

| 主题 | 位置 |
|---|---|
| `get_dcp_group()` | `vllm/distributed/parallel_state.py:1579-1584` |
| 组构造（transpose 技巧） | `vllm/distributed/parallel_state.py:2133` |
| world size 断言 | `vllm/distributed/parallel_state.py:2326` |

## KV cache 布局

| 主题 | 位置 |
|---|---|
| `get_dcp_local_seq_lens`（交错公式） | `vllm/v1/attention/backends/utils.py:1094` |
| `resolve_dcp_kv_block_size`（虚拟块） | `vllm/v1/core/kv_cache_utils.py:668` |
| 哪些 spec 参与切分 | `vllm/v1/core/kv_cache_utils.py:695` |
| `resolve_kv_cache_block_sizes` | `vllm/v1/core/kv_cache_utils.py:717` |
| 块表宽度 / 显存核算 | `vllm/v1/kv_cache_interface.py:532,564` |
| Manager block_size 缩放 | `vllm/v1/core/single_type_kv_cache_manager.py:106-110` |
| 写侧 slot mapping（V1 / V2） | `vllm/v1/worker/block_table.py:414-477`、`vllm/v1/worker/gpu/block_table.py:333` |
| 读侧反解（sparse MLA） | `vllm/v1/attention/backends/mla/sparse_utils.py:153` |
| MLA metadata 的 `dcp_virtual_block_size` | `vllm/model_executor/layers/attention/mla_attention.py:2403-2410` |
| MLA decode metadata 替换 seq_lens | `vllm/model_executor/layers/attention/mla_attention.py:2620-2633` |
| MLA chunked prefill 行预算 | `vllm/model_executor/layers/attention/mla_attention.py:2049` |

## 通信与计算

| 主题 | 位置 |
|---|---|
| Q all-gather（GQA） | `vllm/v1/attention/backends/flash_attn.py:1522` |
| Q all-gather（MLA） | `vllm/model_executor/layers/attention/mla_attention.py:1092-1105` |
| DCP context attention 调用 | `vllm/v1/attention/backends/flash_attn.py:1587-1616` |
| 两段合并 | `vllm/v1/attention/backends/flash_attn.py:1642` |
| combine 选择 | `vllm/v1/attention/backends/flash_attn.py:1151-1156`、`dcp.py:1516-1522` |
| LSE combine kernel | `vllm/v1/attention/ops/dcp.py:95-343`（kernel body 159-200） |
| `_cp_lse_common` / `cp_lse_ag_out_rs` / `cp_lse_ag_out_ar` | `vllm/v1/attention/ops/dcp.py:418,452,485` |
| LSE 参考实现（CPU） | `vllm/v1/attention/ops/dcp.py:518` |
| a2a 打包 / 归约 | `vllm/v1/attention/ops/dcp.py:587,922` |
| `MLADCPManager` | `vllm/v1/attention/ops/dcp.py:1438` |
| reduce_scatter 实现 | `vllm/distributed/device_communicators/base_device_communicator.py:280-311` |
| 空 shard mask | `vllm/v1/attention/ops/dcp.py:71` |

## head 归属（第四部分的核心前提）

| 主题 | 位置 |
|---|---|
| `num_kv_head_replicas` | `vllm/model_executor/layers/linear.py:1049` |
| Q 用 `tp_rank` / KV 用 `tp_rank // replicas` | `vllm/model_executor/layers/linear.py:1295-1298` |

## 运行器侧

| 主题 | 位置 |
|---|---|
| dcp_size / dcp_rank | `vllm/v1/worker/gpu_model_runner.py:548-549` |
| `dcp_local_seq_lens` 填充 | `vllm/v1/worker/gpu_model_runner.py:2477-2490` |
| CUDA graph dummy context | `vllm/v1/worker/cp_utils.py:58-114` |
| skip 分支的 rank 无关性 | `vllm/v1/worker/cp_utils.py:117-127` |
| LSE 必需性断言 | `vllm/v1/worker/cp_utils.py:43-48` |

## 设计文档

- `docs/serving/context_parallel_deployment.md` — DCP/PCP 的设计动机与调参经验
- `docs/design/attention_backends.md` — attention 后端能力矩阵
