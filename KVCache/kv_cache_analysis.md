# vLLM KV Cache 深度分析：block table、物理内存布局与 prefix cache

> 本文档从源码层面拆解 vLLM v1 的 KV cache：block table 的数据结构与查询方式、物理内存布局、跨层语义，以及 prefix cache 的 hash 映射链。含关键代码引用（`file:line`）与具体数字例子。
>
> 关键源码：`vllm/v1/worker/block_table.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/v1/core/block_pool.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/kv_cache_interface.py`、`vllm/v1/kv_cache_layout.py`、`vllm/v1/worker/utils.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/attention/backends/flash_attn.py`。

---

## 目录

- [第一部分：核心概念与总览](#第一部分核心概念与总览)
- [第二部分：Block Table（逻辑块 → 物理块）](#第二部分block-table逻辑块--物理块)
- [第三部分：KV Cache 物理内存布局](#第三部分kv-cache-物理内存布局)
- [第四部分：跨层语义（逻辑 vs 物理）](#第四部分跨层语义逻辑-vs-物理)
- [第五部分：Prefix Cache](#第五部分prefix-cache)
- [第六部分：Sliding Window Attention 的 Prefix Cache](#第六部分sliding-window-attention-的-prefix-cache)
- [第七部分：KV Cache Group 划分（SWA vs FullAttention）](#第七部分kv-cache-group-划分swa-vs-fullattention)
- [附录：关键代码索引](#附录关键代码索引)

---

# 第一部分：核心概念与总览

## 1.1 一张图串起所有概念

```
                 token 序列
                    │  (pos)
                    ▼
        ┌──────────────────────────────┐
        │  Block Table（索引表，per-group）│   block_table[req][logical_blk] = 物理块号 B
        └──────────────────────────────┘
                    │  查出 B
                    ▼
        ┌──────────────────────────────┐
        │  KV Cache 本体（全局大 buffer）  │   [L, B, H, N, C]
        │  kv_caches[layer][B]           │   每层一个 4D view
        └──────────────────────────────┘
```

三个必须分清的对象：

| 对象 | 是什么 | 形状/结构 |
|------|--------|-----------|
| **Block Table** | 逻辑块 → 物理块的**索引表**，per-group 一张 | `[max_num_reqs, max_num_blocks_per_req]` int32 |
| **KV Cache 本体** | 存 K/V 数据的**大 int8 buffer**，所有层共享底层内存 | 逻辑 `[L, B, H, N, C]` |
| **一个 block（页）** | 固定 L、B 后的一页数据 | `[H, N, C]`（或 `[N, H, C]`） |

## 1.2 关键结论速览

1. **Block table 是 per-group 一张，不是 per-layer 一张**；同 group 内所有层共享同一张 block table、同一套物理块号。
2. **Block table 不是 hash map**，是稠密二维数组：行 = req_idx，列 = 逻辑块序号，值 = 物理块号 B。
3. **物理 KV cache 是每层一份张量**（`kv_caches[layer]`），但从**同一个大 int8 buffer** 用 `as_strided` 切出 view。
4. **一个 block 内 K/V 不分离**，每个 `(head, token)` 单元把 K、V 头尾拼接（`head_size + head_size_v` 个元素）。
5. **调度上 block 跨层**（一个 B 值 → 所有层各占一页）；**默认物理布局下不连续**（每层一份，间隔 `layer_stride`）。
6. **prefix cache 的 hash 是 token-based、跨层的**：hash → group_id → 物理块号 B 只查一次表，B 自动落到所有层。

---

# 第二部分：Block Table（逻辑块 → 物理块）

## 2.1 数据结构定义

`BlockTable`（`block_table.py:57`）的核心是一个**稠密二维 int32 数组**，不是 hash map，没有真正的 key-value：

```python
self.block_table = self._make_buffer(
    self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
)  # [max_num_reqs, max_num_blocks_per_req]
```

| 维度 | 含义 | 对应关系 |
|------|------|----------|
| 行索引（第 0 维） | `req_idx`，请求/序列在 batch 的编号 | 相当于 **key** |
| 列索引（第 1 维） | 该序列的**逻辑块序号** | key 的二级下标 |
| 单元格值 | **物理块号** `block_id`（`0..num_gpu_blocks-1`） | 相当于 **value** |

即 `block_table[req_idx][logical_block_idx] = physical_block_id`。

配套字段（`block_table.py:114-117`）：

- `num_blocks_per_row`：`int32[n]`，每行实际用了几个块（不同序列长度不同，行不满）。
- 底层是 `CpuGpuBuffer`（`block_table.py:281`），同时维护 `.np`（CPU numpy，逻辑上增删改）和 `.gpu`（GPU tensor，给 kernel 用）。`commit_block_table()` 只做 `block_table.copy_to_gpu(num_reqs)` 把 CPU 修改同步到 GPU（`block_table.py:231`）。

物理块本身是 `KVCacheBlock`（`kv_cache_utils.py:176`），`block_id` 是它的编号（`0..num_gpu_blocks-1`），由 KV cache manager 分配；block table 里存的就是这个 `block_id`。

## 2.2 per-group，不 per-layer

`MultiGroupBlockTable`（`block_table.py:289`）的 docstring 直接写明："The BlockTables for **each KV cache group**."，`__getitem__` 按 group 索引（`block_table.py:393-395`）。

group 与 layer 的关系在 `kv_cache_utils.py:1078`：

```python
KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
```

多个 layer 合成一个 group，共享同一 spec。所以：

- **同构模型**（标准 Llama，所有 attention 层 KV 规格相同）→ 所有层并入**一个 group** → 只有**一张 block table**。
- **异构模型**（MLA + Mamba、不同 page size 等）→ 拆多个 group，每个 group 一张。

物理 KV cache 张量 `kv_caches` 是 dict，key 为 `layer_name`（`gpu_model_runner.py` 中 `kv_caches[layer_name] = ...`），**每层一份**；但 block table 是 per-group 共享的。

## 2.3 查询路径

分两条路径，都从 block table 出发：

### 写路径（prefill 写 K/V）

`compute_slot_mapping()`（`block_table.py:201`）用 Triton kernel（`block_table.py:414`）把每个 token 转成 `slot_mapping`，核心公式（忽略 CP 交错）：

```
logical_block   = pos // block_size
offset          = pos %  block_size
physical_block  = block_table[req_idx][logical_block]      # 查表
slot_id         = physical_block * block_size + offset     # 展平成线性槽位
```

（对应 `block_table.py:454-476`）`slot_mapping` 是 `int64[num_tokens]`，随后交给 `reshape_and_cache_flash(...)`（`flash_attn.py:1536`）把每个 token 的 K/V 散写进对应物理块。

### 读路径（attention 取历史 K/V）

attention kernel 直接拿 block table 做 **paged attention**：对每个 query token，按 `block_table[req_idx][:]` 拿到物理块列表，逐块 gather K/V（`flash_attn.py:526` 的 `block_table` 字段；`gpu_model_runner.py:2307` 的 `blk_table = self.input_batch.block_table[kv_cache_gid]`）。

**一句话**：block table 是"逻辑块→物理块"翻译表；slot_mapping 是它按 token 粒度展开的"写地址表"；读的时候 kernel 直接用 block table。

---

# 第三部分：KV Cache 物理内存布局

## 3.1 逻辑 5D 形状 `[L, B, H, N, C]`

`kv_cache_layout.py:11-12` 定义 5 个逻辑维：

```
[L, B, H, N, C]
 L = layer（层）
 B = block（物理块）
 H = head（KV 头）
 N = state（块内 token/状态位置，即 block_size）
 C = content（K/V 内容，按字节）
```

对应一个 block（固定 L、B 后）剩下 `[H, N, C]`，维度值来自 `compute_layer_kv_cache_shape_bytes()`（`kv_cache_interface.py:295`）：

```python
return (
    num_blocks * blocks_per_page,     # B
    spec.num_heads,                   # H = num_kv_heads
    spec.get_num_kernel_states(bs),   # N = block_size（token 数/块）
    spec.state_content_size_bytes,    # C = (head_size + head_size_v) * dtype_size
)
```

## 3.2 C 维：K/V 拼接，不分离

`state_content_size_bytes`（`kv_cache_interface.py:521-525`）：

```python
return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)
```

所以 K 和 V **不是两块独立内存**，而是在每个 `(head, token)` 单元里首尾相接：

```
一个 (head, token) 单元 = [ K: head_size 个元素 | V: head_size_v 个元素 ]
```

`flash_attn.py:1526-1527` 印证：

```python
# (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
```

最后一维是 `head_size + head_size_v`，`split(head_size)` 一刀切成 `[K | V]`。默认 `head_size_v == head_size`（`kv_cache_interface.py:511-512`）。

## 3.3 KVCacheLayout：物理 stride 重排

逻辑 `[L, B, H, N, C]` 在内存里按 layout 的 `stride_order` 重排（`kv_cache_layout.py:23-28`）：

| Layout | 物理顺序（外层→内层） | 别名 |
|--------|----------------------|------|
| `LBHNC` | L, B, **H, N**, C | HND |
| `LBNHC` | L, B, **N, H**, C | NHD（**默认**） |
| `LHBNC` | L, H, B, N, C | |
| `BLHNC` | B, L, H, N, C | |
| `BLNHC` | B, L, N, H, C | |
| `BHLNC` | B, H, L, N, C | |

默认是 `LBNHC`（NHD），见 `_DEFAULT_LAYOUT_PREFERENCE`（`utils.py:193-202`）。一个 block 内部物理顺序是 `[N → H → C]`，C 恒为最内层。`H` 和 `N` 谁在外决定数据是"token 优先"还是"head 优先"。

## 3.4 一个 block 的内存图（默认 LBNHC、bf16）

```
块内字节数 = num_kv_heads × block_size × (head_size + head_size_v) × dtype_size

物理排布（LBNHC）:
  token0: head0[K·V]  head1[K·V]  ...  head7[K·V]
  token1: head0[K·V]  head1[K·V]  ...  head7[K·V]
  ...
```

**数字例子**：Llama-8B（8 KV 头、head_size=128、block_size=16、bf16）：

```
一个块 = 8 × 16 × (128+128) × 2 = 65536 字节 = 64 KB
```

## 3.5 分配：一个大 buffer + as_strided

`allocate_kv_cache`（`utils.py:395`）只分配**一个大连续 int8 buffer**（`utils.py:427`）：

```python
buf = torch.zeros(buf_size, dtype=torch.int8, device=device)
```

地址公式（docstring，`utils.py:403`）：

```
layer l 的 block b 起始偏移 = offset + l × layer_stride + b × block_stride
```

每层用 `create_kv_cache_views()`（`kv_cache_interface.py:353`）里的 `torch.as_strided`（`kv_cache_interface.py:404`）切出 4D view，`compute_layout_strides()`（`kv_cache_interface.py:314`）按 layout 的 `stride_order` 算各维字节 stride。

## 3.6 怎么理解 B

B 就是**物理块号（physical block index）**，即 block table 里存的 `block_id`。

- 范围 `0 ~ num_blocks-1`；`num_blocks` 由**启动时 GPU 剩余显存 profiling** 得出，同 group 所有层相同。
- `kv_cache[B]` 切出一整个 block（一页）。
- **B 就是 block table 的 value**：`block_table[req][logical] = B`，kernel 拿到 B 后去 `kv_cache[B]` 取数据。
- B 是"页号"，让 KV cache 变成**分页（paged）**的：不同请求 KV 不必连续、可动态分配/回收、可 copy-on-write 前缀共享——这正是 vLLM PagedAttention 的核心。

**关键澄清**：`[L, B, H, N, C]` 描述的是**整个 KV cache 大缓冲区的排布**（全局所有层、所有块），不是某个 req 的排布。某个 req 的 KV **没有连续内存**，是散落在多个 B 的块的集合，靠 block table 拼起来。

---

# 第四部分：跨层语义（逻辑 vs 物理）

## 4.1 调度/逻辑层面：block 跨层

分配物理块号 B 时，**所有层同时占用各自的 B 号块**。证据：

- block table 每个 group 只有一张，一个 B 值被组内所有层共用；
- `slot_mapping` 每个 group 只算一次，再复制给每层（`gpu_model_runner.py:4096-4097`）；
- 写入时同一个 B 写到 `kv_caches[layer0][B]`、`kv_caches[layer1][B]`……每一层。

所以调度视角下，"一个 block"是跨层的原子单位：**block table 里存一个 B，就等于在所有层各占了一页**。这也正是"一张 block table 服务所有层"的原因。

## 4.2 物理层面：默认不连续

默认 layout 是 **layer-compact（L 最外层）**。地址公式里 `l × layer_stride` 在 `b × block_stride` **外面**，意味着先按层堆叠、层内才是块序列。所以固定 B=b，不同层的 block b 之间隔着 `layer_stride`（≈ 一整层所有块的总大小），是**分散的 L 份**：

```
物理内存（默认 LBNHC）:
  layer0: [b0][b1][b2]...      ← 一整层所有块连在一起
  layer1: [b0][b1][b2]...
  ...
  b0 分散在每层同一位置，层与层隔一个 layer_stride
```

要定位"block b"必须给两个下标：`kv_caches[layer][B]`。

## 4.3 block-outermost 布局：block 真的含所有层

B 在最外层的布局（`BLHNC`/`BLNHC`/`BHLNC`）下，物理内存是：

```
block0: [layer0][layer1][layer2]...   ← 一个 block 物理上把各层收在一起
block1: [layer0][layer1][layer2]...
```

此时"一个 block"物理上确实包含所有层（L 维在 B 维内部）。

## 4.4 总结

| 层面 | 一个 block 是否含所有层 |
|------|------------------------|
| 调度 / block table（逻辑） | **是**，一个 B 值跨所有层 |
| 物理内存 · 默认 `LBNHC` | **否**，每层一份，用 `kv_caches[layer][B]` 定位 |
| 物理内存 · block-outermost（`B*` 布局） | **是**，各层连续 |

---

# 第五部分：Prefix Cache

## 5.1 hash 是 token-based、跨层的

`hash_block_tokens`（`kv_cache_utils.py:684`）：

```python
return BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
```

hash 的输入是 `(父块 hash, 当前块 token ids, extra_keys)`，**只跟 token 序列有关**，跟每层 KV 内容无关。所以：

- 一段 token 前缀（一个 block 那么多 token）**只算一个 hash**；
- 这个 hash 代表"这段 token 前缀在所有层的 KV"；
- 不存在"layer 0 的 block hash"这种区分。

hash key 挂的是 **group_id** 不是 layer_id（`kv_cache_utils.py:74-83`）：

```python
BlockHashWithGroupId = block_hash + group_id.to_bytes(4, "big", signed=False)
```

`KVCacheBlock` 上挂的 `_block_hash` 也是这一个（`kv_cache_utils.py:189`）。加 group_id 是区分不同 KV cache group（异构多 group 时同一 token 前缀在每个 group 各产生一块），组内所有层仍共用同一 hash。

命中后复用**同一个物理块 B**（copy-on-write 加 ref_cnt），`KVCacheBlock` 一个对象、一个 `block_id`、一个 `ref_cnt`、一个 `_block_hash`，天然跨层，无 per-layer 命中判断。

## 5.2 逻辑 hash → 跨层物理块的映射链

```
① token 序列
   ↓ hash_block_tokens(parent_hash, curr_tokens, extra_keys)   kv_cache_utils.py:684
② 逻辑 block hash（一个，token-based，链式）
   ↓ make_block_hash_with_group_id(hash, group_id)             kv_cache_utils.py:74
③ BlockHashWithGroupId = block_hash + group_id(4字节)
   ↓ block_pool.get_cached_block(hash, group_ids)              block_pool.py:198
④ cached_block_hash_to_block 查表                             block_pool.py:35
   ↓ 返回 KVCacheBlock（一个对象）
⑤ block.block_id = 物理块号 B
   ↓ 把 B 写进 block table
⑥ kv_caches[layer0][B], kv_caches[layer1][B], ...              ← 所有层共用同一个 B
```

**中间没有任何"按层"的步骤**：从 hash 到物理块只查一次，得到**一个** B，B 自动落到所有层。

### 各步说明

- **①→②**：请求创建/追加 token 时算好 `Request.block_hashes`（`kv_cache_utils.py:853` 的 `get_request_block_hasher`），每 `hash_block_size` 个 token 一个 hash，链式（父 hash + 当前块 tokens）保证前缀唯一。
- **②→③**：hash 挂 group_id（`kv_cache_utils.py:83`）。
- **③→④→⑤**：`get_cached_block`（`block_pool.py:198`）拿 `block_hash_with_group_id` 查 `cached_block_hash_to_block`（`block_pool.py:35`），value 是 `KVCacheBlock`，`block_id` 即 B。
- **⑤→⑥**：B 追加进请求的 block table（逻辑块 → B），kernel 用 B 去 `kv_caches[layer][B]` 索引每一层。物理层之间只差地址偏移（`l × layer_stride`），块号 B 是同一个。

### 写入缓存（反向）

miss 时分配新块，块写满后 `cache_full_blocks`（`block_pool.py:226`）把 `block_hash_with_group_id → blk` 插回 `cached_block_hash_to_block`（`_insert_block_hash`，`block_pool.py:630`），供后续请求命中。

## 5.3 匹配粒度与匹配方式

**默认按 block 维度匹配，但精确粒度是 `hash_block_size`，可细于物理块。**

`hash_block_size` 的取值（`resolve_kv_cache_block_sizes`，`kv_cache_utils.py:739`）：

- **单 group**：`hash_block_size == scheduler_block_size == block_size × dcp`（`kv_cache_utils.py:763-782`），即**一个 hash 正好对应一个物理块**。
- **多 group**：默认取各 group block size 的 GCD，或被 `prefix_match_unit` 覆盖（`kv_cache_utils.py:814-815`）。
- `prefix_match_unit`（`config/cache.py:89`）可设成**比物理 block size 更细**（只要整除），让命中落在物理块内部边界；它只控制匹配粒度，不改变存储频率。

匹配过程（`get_computed_blocks`，`single_type_kv_cache_manager.py:753`）：

**Phase 1 —— 整块连续匹配**（`single_type_kv_cache_manager.py:800-808`）：

```python
# A missing block implies every later block misses too (chained hashes).
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break                       # 一个 miss，后面全 miss，直接停
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)     # 命中，收集物理块
```

**Phase 2 —— 块内细粒度探测**（仅 fine-grained 模式，`single_type_kv_cache_manager.py:812-831`）：当 `hash_block_size < block_size` 时，第一个没整块命中的块内部从高到低探测 hash 边界。

三个关键语义：

1. **链式 hash 决定"只能前缀匹配"**：hash 是 `(父块 hash, 当前块 tokens)` 算出来的，第 k 个 hash 蕴含前 k 块内容。第 k 块 miss → 第 k+1 块 hash 必然也查不到，所以匹配天然是"从 prompt 头开始的最长前缀"，不是任意区间。
2. **按 group 匹配，不按层**：`get_cached_block(block_hash, kv_cache_group_ids)`（`block_pool.py:198`）对每个 group_id 各查一次，任一 group miss 整体 miss。
3. **返回值是物理块**：`computed_blocks`（每个 group 一个 `list[KVCacheBlock]`）+ `hit_length`（命中 token 数）。

## 5.4 驱逐策略（suffix eviction）

**结论：vLLM 的 GPU prefix cache 偏向"先驱逐 suffix（尾部块）"，但这是释放顺序造成的相对优先级，不是"专门挑后缀"。**

### GPU tier 机制

请求释放时逆序归还（`single_type_kv_cache_manager.py:592`）：

```python
# Free blocks in reverse order so that the tail blocks are freed first.
self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

`free_blocks`（`block_pool.py:777`）把有 hash 的 cached 块按传入顺序 `append` 到 free queue 尾部（`block_pool.py:799-806`），分配时 `popleft` 从队头取（`block_pool.py:684`）：

```
一个请求的块（分配顺序）:  [head ... ... tail]
reversed 后传入 free_blocks: [tail ... ... head]
append 到 free queue 尾部:    queue = [..., tail, ..., head]
popleft 驱逐方向:  ────────────────────→
```

所以**同一请求内，tail 排在 free queue 更靠前，比 head 更早被驱逐**。意图是保住公共前缀（head）以最大化命中率。

### 关键澄清：这里的 "LRU" 指什么

"tail 更易驱逐"看起来和"最近使用"矛盾（decode 阶段 tail 才是最近被写的），但 vLLM 的 LRU 追踪的**不是写入时间**：

| 直觉上的 LRU | vLLM 实际追踪的 |
|--------------|----------------|
| 写入/访问时间：decode 每步都写 tail | **释放回池子的时间 + 被前缀命中的时间** |

- 请求存续期间，所有块 `ref_cnt > 0`，**不在 free queue 里**（不可驱逐），所以 decode 写 tail 不影响驱逐顺序。
- 请求结束那一刻，所有块**同时释放**，跨请求 LRU 下它们是**平局**。
- `reversed` 是在平局上**故意反转的 tie-break**，让 tail 先走、head 后走——因为 prefix cache 的复用是**位置性**（前缀复用）而非时间性。

`touch()`（`block_pool.py:755`）在块被前缀命中时把它**从 free queue remove**（而非挪到队尾），所以 head 因更常被命中而额外存活。

### CPU tier：同策略、独立实现

CPU offloading 用 `CPUOffloadingManager`（`vllm/v1/kv_offload/cpu/manager.py:46`），默认 LRU policy（可换 ARC）。suffix-first 同样靠 `reversed` + rank 递增实现：

```python
# lru.py:92-104
def touch(self, keys, req_context):
    self._update_recency(reversed(list(keys)))      # tail→head

def _assign_new_rank(self, key):
    self._next_rank += 1
    self._ranks[key] = self._next_rank              # 越晚处理 rank 越大（越新）
```

tail 先处理 → rank 小 → 在 min-heap 里更旧 → `heappop`（`lru.py:125`）先驱逐。

**差异**：GPU tier 固定 free-queue LRU；CPU tier 的 policy 可替换（换 ARC 后 suffix-first 不保证成立），且 CPU tier 有 `store_threshold` 入场过滤（`manager.py:265-274`）与 pinning/recency 解耦。

---

# 第六部分：Sliding Window Attention 的 Prefix Cache

SWA 的 prefix cache 处理跟 full attention **方向完全相反**：匹配的是"尾巴窗口"而非"公共头前缀"。

## 6.1 匹配方向反转：从右往左

`SlidingWindowManager.find_longest_cache_hit`（`single_type_kv_cache_manager.py:977`）：

```python
# 命中需要的连续块数 = 窗口大小对应的块数
sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
    kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
)   # = cdiv(sliding_window - 1, block_size)

# 从右往左扫（tail → head），凑够连续窗口块即命中
for i in range(max_num_blocks - 1, -1, -1):
    if cached_block := block_pool.get_cached_block(block_hashes[i], ...):
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= sliding_window_contiguous_blocks:
            break
    else:
        num_contiguous_blocks = 0
```

对比 full attention 的 `for block_hash in ...`（head→tail、miss 即 break），SWA 是 tail→head，且**允许中间断**——只要最后窗口那几块连续命中即可，head 全 miss 也无所谓。

## 6.2 窗口外的块：null + 释放 + 不缓存

- `computed_blocks` 初始化**全填 null block**（`single_type_kv_cache_manager.py:1016-1019`），只有窗口内命中块被替换成真实块。
- `get_num_common_prefix_blocks` 恒返回 0（`single_type_kv_cache_manager.py:1193-1199`）："The prefix blocks are null blocks for sliding window layers."——SWA 层**没有公共前缀块概念**。
- 窗口滑过的老块被主动回收：`get_num_skipped_tokens`（`single_type_kv_cache_manager.py:1156`）= `max(0, num_computed_tokens - sliding_window + 1 - extra_retained_tokens)`，配合 `remove_skipped_blocks`（`single_type_kv_cache_manager.py:690`）。

代码自带的例子（`single_type_kv_cache_manager.py:1162-1173`）：

```
sliding_window=4, num_computed_tokens=7
Tokens:  [0 1 2 3 4 5 6 7]
                               ^ 下一个 token
                      |--------| 窗口 = 4~7
         |--skipped--|
tokens 0~3 被 skip；get_num_skipped_tokens(7) == 4
```

## 6.3 缓存写入侧：block_mask 只缓存窗口尾部块

`cache_blocks` 调 `reachable_block_mask`（`single_type_kv_cache_manager.py:516-534`）生成 `block_mask`，只有每个对齐边界处窗口内的尾部块标记 True。`cache_full_blocks` 里被 mask 掉的块跳过、不进 prefix cache（`block_pool.py:277`）。原因：SWA 命中只需窗口尾部块，中间块缓存了也永不会成为命中目标。

`reachable_block_mask` 还支持 `retention_interval` 做**稀疏 checkpoint**（`single_type_kv_cache_manager.py:1114-1152`），每 segment 只留一个窗口大小的尾部 checkpoint，减少内存占用（"Only SWA acts on it"）。

## 6.4 对比总结

| 维度 | Full Attention | Sliding Window |
|------|---------------|----------------|
| 匹配方向 | 左→右（head→tail） | **右→左（tail→head）** |
| 命中条件 | 从 head 连续，head miss 全 miss | 窗口内 `cdiv(W-1, block)` 连续块，head 可 miss |
| head 块 | 真实块 = 公共前缀 | **null block** |
| 窗口外块 | 不存在 | skip + 主动释放 |
| 缓存写入 | 所有满块都缓存 | **block_mask 只缓存窗口尾部块** |
| 公共前缀 | `get_num_common_prefix_blocks` 有值 | **恒返回 0** |
| 可稀疏化 | 否 | 是（`retention_interval`） |

---

# 第七部分：KV Cache Group 划分（SWA vs FullAttention）

**SWA 和 FullAttention 是两个不同的 KV cache group**，根因是 spec 类型不同。

## 7.1 层 → spec：类型天然不同

每层通过 `get_kv_cache_spec`（`vllm/model_executor/layers/attention/attention.py:606`）决定 spec：

```python
if self.sliding_window is not None:
    return SlidingWindowSpec(...)     # attention.py:659
else:
    return FullAttentionSpec(...)     # attention.py:670
```

## 7.2 分组：按 spec 分桶

`get_kv_cache_groups`（`kv_cache_utils.py:1535`）先按 spec 分组，注释直接给了例子：

```python
# E.g., 2 full attention layers and 3 sliding window attention layers,
# -> (full.0, full.1), (sw.0, sw.1, sw.2).
same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)
```

`SlidingWindowSpec` 与 `FullAttentionSpec` 类型不同，`type(specs[0]).merge(...)`（`kv_cache_utils.py:1553`）无法把它们 merge 到一起，所以必然落在两个 bucket。

## 7.3 各配各的 manager，block size 也独立

| group | spec | manager |
|-------|------|---------|
| full attention | `FullAttentionSpec` | `FullAttentionManager` |
| sliding window | `SlidingWindowSpec` | `SlidingWindowManager` |

SWA 层**自己选 block_size**，脱离 `--block-size`（`attention.py:623-624`），所以两个 group 的 block size、block table、page 大小都可能不同——这也是多 group 时 `hash_block_size` 用 GCD/LCM 的原因（`kv_cache_utils.py:815`）。

## 7.4 历史口径

`FullAttentionSpec` 仍保留 `sliding_window` 字段（`kv_cache_interface.py:563`），docstring 描述旧设计：hybrid allocator disabled 时 SWA 被当成 `FullAttentionSpec(sliding_window=W)` 塞进 full 的同一 group（块按全 token 分配、不回收窗口外块，只在 model runner 计算时按 SWA 走）。当前代码路径（`attention.py:619-668`）已直接返回 `SlidingWindowSpec`，默认分两个 group。

---

# 附录：关键代码索引

| 主题 | 位置 |
|------|------|
| `BlockTable` 类 | `vllm/v1/worker/block_table.py:57` |
| block table 张量 `[req, block]` | `block_table.py:114-117` |
| CPU→GPU 同步 | `block_table.py:231` |
| `MultiGroupBlockTable`（per-group） | `block_table.py:289` |
| slot mapping kernel + 公式 | `block_table.py:414-478` |
| `KVCacheBlock`（block_id/ref_cnt/hash） | `vllm/v1/core/kv_cache_utils.py:176-220` |
| `hash_block_tokens`（token-based） | `kv_cache_utils.py:684` |
| `make_block_hash_with_group_id` | `kv_cache_utils.py:74-83` |
| `KVCacheGroupSpec(layer_names, spec)` | `kv_cache_utils.py:1078` |
| `compute_layer_kv_cache_shape_bytes` | `vllm/v1/kv_cache_interface.py:295` |
| `AttentionSpec`（num_kv_heads/head_size_v） | `kv_cache_interface.py:486-525` |
| `state_content_size_bytes`（K/V 拼接） | `kv_cache_interface.py:521-525` |
| `create_kv_cache_views`（as_strided） | `kv_cache_interface.py:353` |
| `KVCacheLayout`（stride 重排） | `vllm/v1/kv_cache_layout.py:15-28` |
| 默认 layout `LBNHC` | `vllm/v1/attention/backends/utils.py:193-202` |
| `allocate_kv_cache`（大 buffer） | `vllm/v1/worker/utils.py:395` |
| 地址公式 `l*layer_stride + b*block_stride` | `utils.py:403` |
| K/V split（`transpose(1,2).split`） | `vllm/v1/attention/backends/flash_attn.py:1526-1527` |
| slot_mapping 按 group 复制到层 | `gpu_model_runner.py:4096-4097` |
| 层间 KV 共享 `kv_caches[layer]` | `gpu_model_runner.py:7285` |
| `BlockHashToBlockMap`（hash→block） | `vllm/v1/core/block_pool.py:35` |
| `get_cached_block` | `block_pool.py:198` |
| `cache_full_blocks` / `_insert_block_hash` | `block_pool.py:226` / `block_pool.py:630` |
| `free_blocks`（逆序释放/suffix eviction） | `block_pool.py:777` |
| `touch`（命中移出驱逐池） | `block_pool.py:755` |
| `resolve_kv_cache_block_sizes`（hash_block_size） | `kv_cache_utils.py:739` |
| `get_kv_cache_groups`（按 spec 分组） | `kv_cache_utils.py:1535` |
| `prefix_match_unit`（匹配粒度配置） | `vllm/config/cache.py:89` |
| `get_computed_blocks`（Phase1/Phase2 匹配） | `vllm/v1/core/single_type_kv_cache_manager.py:753` |
| `free`（逆序释放入口） | `single_type_kv_cache_manager.py:592` |
| `SlidingWindowManager.find_longest_cache_hit` | `single_type_kv_cache_manager.py:977` |
| `get_num_skipped_tokens`（窗口外 skip） | `single_type_kv_cache_manager.py:1156` |
| `reachable_block_mask`（窗口尾部 block_mask） | `single_type_kv_cache_manager.py:1075` |
| `SlidingWindowSpec` | `vllm/v1/kv_cache_interface.py:831` |
| `get_kv_cache_spec`（SWA/full 分流） | `vllm/model_executor/layers/attention/attention.py:606-677` |
| `LRUCachePolicy`（CPU tier） | `vllm/v1/kv_offload/cpu/policies/lru.py:16` |
