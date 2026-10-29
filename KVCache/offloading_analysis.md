# vLLM KV Cache Offloading 深度分析：CPU/SSD 性能、Chunk 粒度与加载 vs 重算

> 本文档回答四个问题：CPU 内存和本地 NVMe 哪个加载 KV cache 更快、为什么；一次 32 GB 的加载要多久；chunk 默认多大、有什么影响；以及「加载 vs 重算」这个决策在 vLLM / LMCache / Mooncake 里分别是怎么做的。含关键代码引用（`file:line`）与数量级估算。
>
> 关键源码：`vllm/v1/kv_offload/tiering/manager.py`、`vllm/v1/kv_offload/tiering/spec.py`、`vllm/v1/kv_offload/cpu/gpu_worker.py`、`vllm/v1/kv_offload/cpu/shared_offload_region.py`、`vllm/v1/kv_offload/tiering/fs/io.py`、`vllm/v1/kv_offload/tiering/fs/manager.py`、`vllm/v1/kv_offload/tiering/backpressure.py`、`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`、`docs/features/kv_offloading_usage.md`。

---

## 目录

- [第一部分：结论速览](#第一部分结论速览)
- [第二部分：带宽与延迟——两个不同的指标](#第二部分带宽与延迟两个不同的指标)
- [第三部分：CPU DRAM vs Local NVMe](#第三部分cpu-dram-vs-local-nvme)
- [第四部分：32 GB KV Cache 加载时间估算](#第四部分32-gb-kv-cache-加载时间估算)
- [第五部分：Chunk 粒度](#第五部分chunk-粒度)
- [第六部分：加载 vs 重算（vLLM 现状）](#第六部分加载-vs-重算vllm-现状)
- [第七部分：LMCache 的做法](#第七部分lmcache-的做法)
- [第八部分：Mooncake 的做法](#第八部分mooncake-的做法)
- [第九部分：三方对比](#第九部分三方对比)
- [第十部分：Prefill 时间粗估模型](#第十部分prefill-时间粗估模型)
- [第十一部分：给 vLLM 的优化建议](#第十一部分给-vllm-的优化建议)
- [附录：关键代码索引](#附录关键代码索引)

---

# 第一部分：结论速览

1. **从 CPU 内存加载到 GPU 明显快于从本地 SSD 读**：本机（CPU→GPU Gen5 x16 ~50 GB/s vs SSD Gen4 x4 ~7 GB/s）带宽高约 7 倍，延迟低 2–3 个数量级。
2. **这不是调参结论，是架构约束**：vLLM 的 tiering 设计里，SSD 是被设计成 CPU 之后更冷的一层，数据必须经 `SSD → CPU → GPU` 两跳，GPU 从不直接读写 SSD（`tiering/manager.py:9-19`）。
3. **32 GB 加载量级**：CPU→GPU 约 0.65–1.3 s；NVMe 理想顺序读约 2.5–4.6 s；NVMe 默认小 chunk 下约 6–8 s（多线程）甚至 ~40 s（串行）。
4. **chunk 默认 = 1 个 GPU block**（CUDA 上 16 token，Llama-8B 约 64 KB）。对 SSD 层偏小，建议调大。
5. **vLLM 没有「加载 vs 重算」的代价模型**：命中即加载。只有静态的 `max_load_tokens`、`kv_load_tiers` 等外部开关。
6. **LMCache 和 Mooncake 同样没有代价模型**，分别提供静态阈值 `min_retrieve_tokens` / 可插拔但信息不足的 `KVLoadPolicy`，以及 TENT 的截止时间降级（钩子为空）和 SGLang HiCache 的超时预取（实际可用）。
7. **真正的「部分加载 + 部分重算」**目前只有 SGLang HiCache 的 timeout 预取在生产路径上跑通了；它的关键是利用「已到的是连续前缀、未到的是连续后缀」这个性质。

---

# 第二部分：带宽与延迟——两个不同的指标

## 2.1 定义

| 指标 | 含义 | 单位 | 决定什么 |
|------|------|------|---------|
| **带宽（Bandwidth）** | 单位时间最多搬多少数据 | GB/s | **大批量**数据传输的总耗时 |
| **延迟（Latency）** | 发出一次请求到数据开始返回的固定等待 | ns / µs | **大量小请求**的总耗时 |

一次 I/O 的耗时可以近似为：

```
单次耗时 ≈ 延迟 + 数据量 / 带宽
总耗时   ≈ I/O 次数 × 单次耗时 / 并发度
```

- 搬一个大文件 → 看**带宽**。
- 搬几十万个小 block → 延迟会显著累积（`fs/io.py` 是一个 block 一个文件）。

## 2.2 一个容易混淆的点：NVMe 也走 PCIe

NVMe SSD 本身也是插在 PCIe 上的。对比的不是「PCIe vs 非 PCIe」，而是**同一条 PCIe 总线上不同设备的差异**。注意每个设备的 PCIe 代数与通道数**由设备自身规格和所处插槽共同决定，并不统一**：同一台机器上 GPU、SSD、网卡可以分别跑在 Gen5、Gen4、Gen3 上。

以下按本机（云上）基准配置：**GPU Gen5 x16 / SSD PCIe Gen4 x4 / 网卡 Gen3 x16**。

| 路径 | 链路 | 理论带宽 | 实际可达 | 延迟 |
|------|------|---------|---------|------|
| CPU DRAM → GPU | PCIe Gen5 **x16**（本机） | 64 GB/s | ~50 GB/s | ~1–2 µs（DMA 启动），DRAM 本体 ~100 ns |
| CPU DRAM → GPU | PCIe Gen4 **x16**（旧平台） | 32 GB/s | ~25 GB/s | 同上 |
| NVMe SSD 读 | PCIe Gen4 **x4**（本机） | 8 GB/s | ~7 GB/s（顺序） | ~50–100 µs（NAND 读） |
| NVMe SSD 读 | PCIe Gen5 **x4**（参考） | 16 GB/s | ~12–14 GB/s（顺序） | ~50–100 µs |
| 网卡收发 | PCIe Gen3 **x16**（本机） | 16 GB/s | ~11–12 GB/s | ~10–20 µs（RDMA 单程） |
| 网卡收发 | PCIe Gen4 **x16**（参考） | 32 GB/s | ~25 GB/s | 同上 |

差距来源有三处：

1. **PCIe 代数 × 通道数**：本机 GPU 吃满 Gen5 x16（64 GB/s），单块 SSD 只有 Gen4 x4（8 GB/s），差 **8 倍**；网卡 Gen3 x16（16 GB/s）介于两者之间。旧平台上 GPU 若是 Gen4 x16，则只比 SSD 快 4 倍。
2. **网卡还有线速上限**：Gen3 x16 约 126 Gb/s，刚好够 100GbE 跑满（~12 GB/s）；200GbE / 400GbE 需要 Gen4 / Gen5 x16 才不成为瓶颈，否则 PCIe 先饱和。
3. **介质**：DRAM 读一次 ~100 ns，NAND 闪存读一页要几十 µs，差约 500 倍。

因此本文后续估算都按「CPU→GPU 走 Gen5 x16（~50 GB/s）、本地 SSD 走 Gen4 x4（~7 GB/s）」计算；远端 tier 则由网卡（Gen3 x16 ≈ 100GbE，~12 GB/s）封顶。

---

# 第三部分：CPU DRAM vs Local NVMe

## 3.1 vLLM 的 tiering 架构把结论写死在了代码里

`vllm/v1/kv_offload/tiering/manager.py:9-19` 的 docstring 直接列出设计原则：

```
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Chunks in secondary tiers must be promoted to the
   primary tier before GPU can access them
```

`lookup()`（`tiering/manager.py:382-465`）的顺序是：

```
先查 CPU primary tier
   ├── HIT        → 直接返回（一跳）
   ├── HIT_PENDING→ 直接返回
   └── MISS       → 遍历 secondary tiers
                     ├── HIT   → _initiate_promotion()（SSD→CPU）
                     └── RETRY / MISS
```

一条 SSD 命中的数据要走：

```
SSD ──(thread pool + os.readv / O_DIRECT)──▶ CPU primary tier ──(cudaMemcpyAsync DMA)──▶ GPU
              第二跳（冷、慢）                                第一跳（热、快）
```

而 CPU 命中只需一跳。

`docs/features/kv_offloading_usage.md:15` 也写死了：

> Only the CPU primary tier has direct GPU access. Secondary tiers cannot read from or write to GPU memory; all GPU↔secondary transfers are staged through the CPU primary tier.

**所以「SSD 命中」从结构上就不可能比「CPU 命中」快**，哪怕 SSD 带宽再高，也要多一次 CPU 侧落盘再起跳。

## 3.2 两种传输的代码机制完全不同

### CPU ↔ GPU：DMA + pinned memory + 异步流

`vllm/v1/kv_offload/cpu/gpu_worker.py`：

- `_select_swap_blocks_fn`（`gpu_worker.py:39-65`）注释：「GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton」，走 `ops.swap_blocks_batch`（底层是 `cudaMemcpyAsync` 的批量 DMA）。
- 主机端必须是 **pinned memory**：`pin_mmap_region`（`gpu_worker.py:201-270`）用 `cudaHostRegister` 把 mmap 区域 pin 住，否则 DMA 无法直接访问 / Triton 无法解引用 CPU 指针。
- 每个方向一个 `SingleDirectionOffloadingHandler`，独立 CUDA stream + event 链，多个 transfer 串行但**与模型计算流重叠**（`transfer_async`，`gpu_worker.py:565-735`）。文档也写明「Transfers … run asynchronously alongside model computation」。

### SSD：文件 I/O + 线程池 + O_DIRECT

`vllm/v1/kv_offload/tiering/fs/io.py` 与 `tiering/fs/manager.py`：

- `_load_block`（`fs/io.py:133-163`）是 `os.open` + `os.readv`，一个 block 一个文件。
- 使用 **`O_DIRECT`**（`fs/io.py:26`，探测与回退见 `fs/manager.py:191-201`）绕过 OS page cache —— 意味着它是真读盘，而不是命中 DRAM 里的 page cache。
- 靠 `DualQueueThreadPool`（`fs/manager.py:203-207`）的 read/write 线程池提高并发，但本质仍是「系统调用 + 块设备」路径。

两者机制上的差距就是答案本身：**DMA + pinned 内存 + 异步流** vs **文件 syscall + 块设备**。

## 3.3 那 SSD 层的意义是什么？

不是速度，而是**容量、持久化、共享**：

1. **容量**：DRAM 几百 GB～2 TB 顶天，SSD 可以上多 TB，能缓存更多前缀、提高命中率。`cpu_bytes_to_use` 决定 CPU 层大小；文档 Tuning Tips 明确指出 CPU 层越大，去 SSD 的 trip 越少。
2. **持久化 / 跨进程共享**：`fs` tier 用 block hash 命名文件，多个 vLLM 实例共享同一 `root_dir`（如共享 PVC）即可复用彼此算好的 KV，跨重启也不丢。CPU 层是进程内 volatile 的。
3. **级联写入**：`complete_store`（`tiering/manager.py:709-755`）把每个 chunk **同时写到所有 tier**（cascade），所以 SSD 是「顺手冷备份」，不是替代 CPU 的热路径。

---

# 第四部分：32 GB KV Cache 加载时间估算

## 4.1 情况 A：从 CPU 内存加载到 GPU

| GPU 链路 | 实际带宽 | 32 GB 耗时 |
|----------|---------|-----------|
| PCIe Gen5 x16（本机） | ~50 GB/s | **~0.65 s** |
| PCIe Gen4 x16 | ~25 GB/s | **~1.3 s** |

延迟在批量 DMA 下可以忽略：vLLM 用 `swap_blocks_batch` 一次提交大量 descriptor，固定开销被摊薄。耗时基本等于 `数据量 / 带宽`。若 GPU 只插到 Gen4 插槽或降到 x8，带宽按比例减半，耗时翻倍。

## 4.2 情况 B：从 local NVMe 加载（理想顺序读）

数据两跳，但两段可以流水线重叠，总耗时由慢的一跳（SSD）决定：

| SSD | 顺序读带宽 | 32 GB 耗时 |
|-----|-----------|-----------|
| Gen4 x4 单盘（本机） | ~7 GB/s | **~4.6 s** |
| Gen5 x4 单盘（参考） | ~13 GB/s | **~2.5 s** |
| Gen4 x4 × 4 盘 RAID0 | ~28 GB/s | ~1.2 s（接近 PCIe Gen5 x16） |

单盘 SSD 只占 x4 通道，插槽剩下的通道是闲置的。要突破 ~7 GB/s，靠加盘 RAID0 比换 Gen5 盘更直接，但需要 CPU 侧有足够的 PCIe 通道（4 × x4 = 16 lanes）和 RAID 卡。

## 4.3 情况 C：NVMe 的现实——大量小 block

按 Llama-8B、block_size=16 计算，一个 block 约 64 KB（见第五部分）：

```
32 GB / 64 KB = 524,288 次 I/O
```

vLLM 的 fs tier 是**一个 chunk 一个文件**，每次都要 `open` + `readv` + `close`。此时延迟成为关键：

| 并发度 | 估算 | 耗时 |
|-------|------|------|
| QD=1（串行） | 每次 ~80 µs → 64 KB / 80 µs ≈ 0.8 GB/s | **~40 s** |
| 32 线程（`n_read_threads=32`） | 打到 SSD 随机读上限 ~4–6 GB/s | **~6–8 s** |
| 加大 `blocks_per_chunk`（如 16 → 1 MB/文件） | I/O 次数降到 ~3.2 万，接近顺序读 | **~5 s** |

这就是 vLLM 提供 `n_read_threads`、`blocks_per_chunk`、`O_DIRECT` 的原因：用并发掩盖延迟，用大 I/O 减少延迟的次数。

## 4.4 汇总

| 来源 | 32 GB 耗时 | 瓶颈 |
|------|-----------|------|
| CPU DRAM → GPU（Gen5 x16，本机） | ~0.65 s | GPU 插槽 PCIe 带宽 |
| CPU DRAM → GPU（Gen4 x16） | ~1.3 s | GPU 插槽 PCIe 带宽 |
| NVMe 理想顺序读（Gen4 x4，本机） | ~4.6 s | SSD x4 带宽 |
| 远端加载（Gen3 x16 网卡 ≈ 100GbE） | ~2.7 s | 网卡线速 |
| NVMe 小 block + 多线程 | ~6–8 s | SSD 随机读 IOPS / 延迟 |
| NVMe 小 block 串行 | ~40 s | 纯延迟 |

## 4.5 实际部署要打折扣的地方

1. **TP 多卡并行**：TP=8 时每卡只加载 `32/8 = 4 GB`，各走自己的 PCIe 插槽，CPU→GPU 理论上 Gen5 x16 ~0.08 s、Gen4 x16 ~0.16 s。但多卡经常共享 PCIe switch / root complex（尤其 x16 插槽有限时会被降为 x8），实际要打折扣。SSD 层则是所有卡共用几块盘，并行化后劣势更明显。
2. **NUMA**：pinned 内存若分配在另一个 socket，要跨 UPI，带宽掉约 20–40%。
3. **不需要一次全部加载**：vLLM 是按请求、按命中的前缀 block 增量 promote 的，并与计算重叠。真正影响 TTFT 的是**单个请求命中的那部分前缀**——例如一个 32K token 的请求在 Llama-8B 上只要 4 GB，不是 32 GB。
4. **对比重算**：最终决策应该是「加载 vs 重新 prefill」。即使 SSD 要几秒，对长上下文仍可能比重算划算（见第十部分）。

> 以上均为按本机基准配置（GPU Gen5 x16、SSD PCIe Gen4 x4、网卡 Gen3 x16）推算，而非实测。精确数字需要 `fio` 测 SSD 的 64 KB 随机读 / O_DIRECT 吞吐，以及用 pinned 内存的 `cudaMemcpy` 测 PCIe 实际带宽。

---

# 第五部分：Chunk 粒度

## 5.1 默认值

**默认 1 个 chunk = 1 个 GPU block。**

来自 `docs/features/kv_offloading_usage.md` 的配置表：

| Key | 默认 | 含义 |
|-----|------|------|
| `blocks_per_chunk` | `1` | 一个 offload chunk 包含几个 GPU block |
| `block_size` | GPU block size | 一个 chunk 覆盖多少 token，须为 GPU block size 的整数倍；与 `blocks_per_chunk` 二选一 |

文档原话：*"By default, a chunk maps to a single accelerator block."*（`docs/features/kv_offloading_usage.md:56`）

GPU block size 在 CUDA 上默认 16 token。按 Llama-8B（8 KV 头、head_size 128、bf16）：

```
1 chunk = 1 block = 16 token = 8 × 16 × (128+128) × 2 B = 64 KB
```

## 5.2 对 SSD 层的影响

fs tier 是一个 chunk 一个文件（`FileMapper` 使用 `blocks_per_file=offloading_spec.blocks_per_chunk`）。默认 64 KB 正好落在小 I/O、延迟主导的区域。

调大 chunk：

```json
"block_size": 256
// 等价于 blocks_per_chunk = 16，每文件约 1 MB
```

32 GB 对应文件数从 ~52 万降到 ~3.2 万，SSD 读取更接近顺序读带宽。

**代价**：lookup 粒度变粗。前缀命中要按 256 token 对齐，尾部不足一个 chunk 的部分无法命中。

## 5.3 注意

64 KB 取决于模型的 KV 头数、head_size、dtype。换模型需重算：

```
chunk 字节数 = blocks_per_chunk × num_kv_heads × block_size(16) × (head_size + head_size_v) × dtype_size
```

### 与 LMCache 的对比

LMCache 默认 `chunk_size = 256`（`config.py:90`），比 vLLM 大 16 倍，天然规避了上述小 I/O 问题。

---

# 第六部分：加载 vs 重算（vLLM 现状）

## 6.1 结论

**vLLM 没有「加载 vs 重算」的代价模型。** 只要 lookup 命中就加载，不会比较「从 SSD 读要多久」和「在 GPU 上重算要多久」。

## 6.2 vLLM 现有的相关机制

| 机制 | 位置 | 作用 | 是否 cost-based |
|------|------|------|----------------|
| `max_load_tokens` | `offloading/scheduler.py:764-768` | 每请求最多加载多少 token，超出部分重算；0 = 完全不加载 | 否，外部按请求传入 |
| `kv_load_tiers` | `tiering/manager.py:437` | 每请求可查询的 secondary tier 子集 | 否，静态过滤 |
| 不足一个 chunk 就跳过 | `offloading/scheduler.py:816-818` | 可加载部分 < 一个 chunk 时直接返回 0 | 粒度限制 |
| backpressure | `tiering/backpressure.py` | EMA 度量 store 延迟，慢了就丢写 | 只管**写**，不管读 |
| `LOOKUP_ASYNC_DELAY` / `READ_TIME` / `READ_BYTES` 等指标 | `tiering/spec.py:148-175, 125-147` | 记录各层读取耗时 | 有数据，但不用于决策 |

也就是说，**接口和数据都有了，缺的是中间那段决策逻辑**。

## 6.3 为什么这个空白重要

`TieringOffloadingManager.lookup()` 在 secondary tier 命中、准备 `_initiate_promotion` 的位置，**同时知道命中长度（key 集合大小）和所在层（tier_idx）**。这是做代价决策最自然的位置。

理想的判断应随 token 位置变化，而不是简单的二选一：

```
重算第 i 个 chunk 的代价 ∝ chunk 长度 × 已有上下文长度 i（attention 是 O(n²)，越靠后越贵）
                         + 线性部分（MLP/投影，和位置无关）
加载第 i 个 chunk 的代价 ≈ chunk 字节数 / 实测带宽（可取自 READ_TIME / READ_BYTES 的 EMA）
```

由此推出几个结论：

1. **前缀越长越值得加载**：重算代价超线性增长，加载代价是常数。真实决策往往是「前 N 个 chunk 重算、后面加载」或整体加载。
2. **短命中直接重算**：几百 token 时 GPU prefill 几毫秒就算完，没必要付 SSD 的 promotion 往返（还要等一个调度 step 的 RETRY）。
3. **GPU 空闲时重算更划算**：低负载时算力空着，重算几乎无成本；高负载时算力是瓶颈，应优先加载。阈值最好随 batch 负载自适应。
4. **理想情况是并行**：一部分重算、一部分加载同时进行，耗时取 max。但 vLLM 当前的 prefix hit 要求连续前缀，且请求要等 KV 全部就绪才被调度，这一条较难落地。

## 6.4 两条可行路径

### 路径 1：外部决策（不改代码）

由 router / EPP 或客户端在发请求前估算，通过 `kv_transfer_params` 下发 `max_load_tokens` 或 `kv_load_tiers`：

```python
# 离线标定的常数
t_load    = hit_tokens * bytes_per_token / ssd_bw   # 两跳流水线后取较大者
t_compute = prefill_time(hit_tokens, context_len)    # 第十部分的模型

if t_load > t_compute:
    params = {"max_load_tokens": 0}       # 直接重算
# 或者：只允许从 CPU 加载，不查 SSD
params = {"kv_load_tiers": []}
```

**缺点**：外部看不到某个 block 实际在哪一层（CPU or SSD），只能按最坏情况估。

### 路径 2：vLLM 内部加 cost-aware 决策（需改代码）

位置：`TieringOffloadingManager.lookup()` 的 secondary 命中分支，`_initiate_promotion` 之前。

建议分两步落地：

1. 先加静态配置，如 `min_promotion_tokens` / 按 tier 的 `min_load_tokens`，命中长度低于阈值就不 promote。
2. 再用已有 `READ_TIME / READ_BYTES` 指标做带宽 EMA，结合离线 prefill 吞吐曲线做自适应阈值。

---

# 第七部分：LMCache 的做法

> 以下内容来自对 `/Users/xiegoudan/Documents/LMCache` 的代码调研，部分细节未逐行复核，引用前建议核对。

## 7.1 结论

同样**没有代价模型**，默认命中即加载。区别是多了静态阈值和一个可插拔策略接口。

## 7.2 能让「命中」变成「重算」的机制

| 机制 | 位置 | 行为 |
|------|------|------|
| **`KVLoadPolicy`**（仅多进程 server 模式） | `lmcache/v1/multiprocess/kv_load_policy.py`；调用点 `modules/lookup.py:291-309` | 每请求判断，`should_load()` 返回 False 时报 0 命中。可通过 `--kv-load-policy pkg.mod:Class` 替换 |
| `min_retrieve_tokens` | `config.py:462`（默认 0 = 关闭）；生效于 `vllm_v1_adapter.py:1446-1527` | 命中 token 数低于阈值则重算 |
| `lmcache.max_tokens_per_load` | `vllm_v1_adapter.py:580-586` | 限制最多加载多少 token，超出重算。目的是防止 GPU block 池被占满，不是为了延迟 |
| `lmcache.skip_load` | 按请求配置 | 等价于 vLLM 的 `max_load_tokens: 0` |
| 降级模式 | `manager.py:109, 237` | LMCache 不健康时整体退回重算（容错） |

默认的 `KVLoadPolicy` 只看请求是否带 `skip_load`：

```python
return not (ctx.request_configs or {}).get("lmcache.skip_load", False)
```

**关键缺陷**：策略在 prefetch **之前**调用，`KVLoadContext` 只有 `num_lookup_tokens`（命中数上界）、`chunk_size`、模型名和请求配置，**看不到实际命中了多少 token，也看不到数据在哪一层**。即使自己写代价模型，也只能按最坏情况估算。

## 7.3 间接缓解机制（比 vLLM 多的部分）

1. **按层流水线（`use_layerwise`，默认关闭）**：`cache_engine.py:1001` 的 `retrieve_layer` 一层一层加载，第 i 层计算时第 i+1 层在加载，加载时间可被计算掩盖大部分。vLLM 原生 offloading 没有这个——vLLM 要等整个请求 KV 全部加载完才开始调度。
2. **CacheBlend 部分重算**（`enable_blending`、`blend_recompute_ratios`）：加载 KV 后只重算一小部分 token 来修正。主要面向 RAG 等非前缀复用场景，但思路上就是「部分加载 + 部分计算」的混合。
3. **异步 lookup**（`enable_async_loading`）：lookup 可先返回「未就绪」，下一 step 再查，类似 vLLM tiering 的 RETRY。

## 7.4 默认值

- `chunk_size = 256`（`config.py:90`）。
- 存储层级：本地 CPU（默认开启，5 GB）→ 本地磁盘（默认关闭）→ 远端（默认关闭）。查询按后端插入顺序，**没有带宽感知排序**。
- `save_unfull_chunk = False`：尾部不满一个 chunk 的部分不保存，命中按 256 token 对齐。
- `lookup_timeout_ms` 默认 3000。

---

# 第八部分：Mooncake 的做法

> 以下内容来自对 `/Users/xiegoudan/Documents/Mooncake` 的代码调研，部分细节未逐行复核。

## 8.1 结论

Mooncake 也**没有完整的代价模型**。仓库中搜不到 cost model、TTFT 估算或 cache-aware prefill 调度的实现。唯一真正接近的机制在传输引擎 TENT 里：**预测截止时间是否会被错过，赶不上就取消传输，让调用方改为本地重算**。

## 8.2 TENT 的截止时间降级（代码有，但未接通）

设计文档：`docs/source/design/tent/deadline-scheduling.md`（RFC #2519 / #2792）。

预测方式：

```
预测负载比 MLU = (已在发送的字节 + 本次请求字节) / 链路带宽 / 截止时间窗口
若 MLU ≥ θ（mlu_local_threshold）→ 标记 CANCELED，并触发 on_local_decode_suggested
```

文档中的例子：一个 256 MiB 请求、6 ms 截止窗口，预测传输 8.7 ms，MLU = 1.45 ≥ 1.2，于是取消，「改为本地重算，而不是等它」。

代码：`mooncake-transfer-engine/tent/include/tent/runtime/admission_queue.h:59-61`（阈值）、`:103`（钩子）。

**问题在后半段**，文档自己的 Known Limits 承认：
- 引擎装的是**空的** `DegradationHooks`，**没有公开 API 注册**，调用方只看到一个 `CANCELED`。
- vLLM 和 SGLang 都没有代码收到 `CANCELED` 后去重新 prefill。
- 判断的是「传输会不会超时」，**不是**「传输是否比重算慢」；截止时间由调用方给，重算代价不在模型里。

## 8.3 SGLang HiCache 的超时预取（Mooncake 文档描述，实现在 SGLang）

`docs/source/design/hicache-design.md:45-62` 描述三种预取策略：

| 策略 | 行为 |
|------|------|
| `best_effort` | GPU 一空出来能做 prefill 就停止预取，未取到的部分重算 |
| `wait_complete` | 等全部取完 |
| `timeout` | 超时 = `prefetch_timeout_base` + `prefetch_timeout_per_ki_token` × token 数（千 token），到时间就用已取到的部分 |

关键性质：**已到的是连续前缀，剩下要重算的是连续后缀**，与 prefix cache 语义无缝结合。

这是三家里**唯一在生产路径上真正跑通了「部分加载 + 部分重算」**的机制。它靠时间预算而非代价模型，但效果接近：慢存储取不完的部分自动交给重算。

## 8.4 Conductor：只做了索引，没有决策

`mooncake-conductor/src/prefixindex/prefix_indexer.cpp:505-665` 的 `PrefixCacheTable::Query` 返回**每个实例、每一层**（`device` / `cpu_local` / `cpu_share` / `disk`）各命中多少 token。

这正好是做代价决策所需的输入。但「怎么选实例」的打分策略留给外部路由器，代码里没有把传输时间和 TTFT 比较。FAST'25 论文中的「按前缀命中长度、排队时间、传输时间估算 TTFT 再调度」**只存在于论文 PDF（`FAST25-release/Mooncake-FAST25.pdf`）里，仓库中没有实现**。

## 8.5 存储层级

- Store 池化管理内存与存储。
- Conductor 把副本类型折叠为两个逻辑层：`cpu`（内存副本）与 `disk`（本地磁盘、NVMe-oF SSD、DFS）。
- HiCache 使用 GPU(L1) / host memory(L2) / Mooncake(L3)。
- 传输侧有 smart NIC 选择（EWMA + NUMA 分层）、最早截止优先（EDF）准入队列、按 NIC 带宽仲裁、QoS 优先级。

---

# 第九部分：三方对比

| | vLLM 原生 | LMCache | Mooncake |
|---|---|---|---|
| 加载 vs 重算的代价模型 | 无 | 无 | 无（只在论文里） |
| 静态最小命中阈值 | 无 | `min_retrieve_tokens` | 无 |
| 按请求禁止加载 | `max_load_tokens: 0` | `skip_load` | — |
| 可插拔决策接口 | 无 | `KVLoadPolicy`（看不到实际命中和所在层） | `DegradationHooks`（为空，无法注册） |
| 基于时间的降级 | 无 | 无 | TENT 截止时间预测（未接通）+ SGLang 超时预取（可用） |
| 部分加载 + 部分重算 | 只有 `max_load_tokens` 静态截断 | CacheBlend（非前缀场景）、`max_tokens_per_load` | HiCache timeout：已到前缀用、剩余后缀重算 |
| 加载与计算重叠 | 无（等 KV 全部加载完才调度） | 按层流水线 `use_layerwise` | 异步预取 / 写回 |
| 「命中长度 + 所在层」信息 | 有，在 `TieringOffloadingManager.lookup()` 内部 | 决策点拿不到 | Conductor 能查到，但没人用来决策 |
| 默认 chunk | 16 token（1 个 block） | 256 token | — |

**共同空白**：三家都没有「加载 vs 重算」的代价模型。能用的都是近似方法——静态阈值、按请求开关、超时预算。最有价值的一步是在**知道实际命中长度和所在层之后**再做决策。

---

# 第十部分：Prefill 时间粗估模型

## 10.1 主公式：FLOPs / 有效算力

prefill 是计算密集的，用 FLOPs 除以实测算力即可，粗估误差通常在 2 倍以内。

处理 `n` 个新 token、前面已有 `c` 个上下文 token：

```
线性层 FLOPs   ≈ 2 × P × n                        # P = 参数量（MoE 用激活参数量）
Attention FLOPs ≈ 4 × L × d × n × (c + n/2)        # L = 层数，d = num_heads × head_dim
                                                   # QKᵀ 与 AV 各 2 次乘加；causal 只算一半

时间 ≈ 总 FLOPs / (峰值算力 × MFU)
```

- **线性部分**与 token 数成正比，每 token 代价固定。
- **Attention 部分**随上下文超线性增长，这就是「越靠后的 token 重算越贵」的来源。
- **MFU**：prefill 一般 **40–60%**，粗估取 50%。

### 推导：线性层为什么是 `2Pn`

每个权重对每个 token 做一次乘加（MAC），按惯例一个 MAC 记 2 FLOPs：

```
y = Wx，W 有 P 个参数
一个输出元素 = d_in 次乘 + (d_in-1) 次加 ≈ 2·d_in FLOPs
全部输出元素 = 2P FLOPs / token
n 个 token 合批（GEMM）= 2Pn
```

口径：

- **P 包含所有带权重的层**：Q/K/V/O 投影、MLP（gate/up/down）、各层 norm。attention 打分本身无权重，不在这里。
- **MoE 用激活参数量**：一个 token 只经过路由到的专家 + shared expert，不是总参数。
- **embedding 查表不算 FLOPs**；lm_head 若只取最后 token 的 logits，也不按 n 个 token 全算（粗估通常忽略）。
- norm、激活、残差是逐元素操作，约比线性层低三个数量级，忽略。

### 推导：Attention 为什么是 `4Ldn(c + n/2)`

这一项只算 **QKᵀ 和 AV**，不含 Q/K/V 投影（已计入 `2Pn`）。

单个 (query, key) 对：点积维度为 `d = num_heads × head_dim`，QKᵀ 是 `2d` FLOPs，AV 再 `2d`，共 `4d`。

对数：只有 n 个新 token 需要算输出（c 个 context 的 K/V 已缓存，输出不重算）。第 i 个新 token 能看到 `c + i` 个 key，求和：

```
Σ(c + i) = n·c + n(n+1)/2 ≈ n·(c + n/2)
```

`+ n/2` 是 causal 下三角的贡献——新 token 之间平均能看到一半。合并：

```
每层: 4d × n × (c + n/2)
L 层: 4 × L × d × n × (c + n/2)
```

若不 mask，m 个 token 两两打分是 `2dm²`；causal 砍掉上三角后 `≈ dm²`，再乘上「只算 n 行 query」即得上式。

### 适用范围与偏差

| 情形 | 修正 |
|---|---|
| 短 prompt（几百 token） | 转为**访存瓶颈**，下限是读一遍权重：`16 GB / 3.35 TB/s ≈ 5 ms`，取 `max(权重读取, FLOPs/算力)`（10.4） |
| c ≫ n（长上下文追加） | attention 项 ≈ `4Ld·c·n`，每 token 线性增长 |
| chunked prefill | n 拆段后交叉项 `c·s` 求和不变、平方项变小；分块后总 FLOPs 略降 |
| 多请求合批 | 线性项按总 token 数相加；attention 项逐序列计算，c 不能合并 |
| GQA/MQA | `d` 取 query heads × head_dim；KV 头少主要省 KV cache 和 K/V 投影 |
| MLA（DeepSeek） | 公式不直接适用：KV 压缩、投影可吸收，等效维度与 KV cache 需另算 |
| softmax | 每个分数 3–5 个非 matmul 操作，d=128 时可忽略；极短上下文占比上升 |
| FP8 | 峰值算力约翻倍，FLOPs 不变，时间约减半（MFU 可能变化） |

### 与 10.3 的关系

线性项按每 token 写为 `2P`，attention 项对 n 求导得每 token 增量 `4Ld(c + n/2)`。加载每 token KV 的时间不超过重算时间，即：

```
KV_bytes / BW ≤ [2P + 4Ld(c + n/2)] / 有效算力
```

解出 BW 就是 10.3 的盈亏平衡带宽——这两条公式同时是耗时估算和「加载 vs 重算」决策的输入。

## 10.2 例子：Llama-3-8B，H100，bf16

参数：P = 8e9，L = 32，d = 32 × 128 = 4096。H100 bf16 dense 峰值约 989 TFLOPS，50% MFU → 有效算力 ≈ **500 TFLOPS**。

从零 prefill 32K token（c = 0，n = 32768）：

```
线性：      2 × 8e9 × 32768                ≈ 5.2e14
Attention： 4 × 32 × 4096 × 32768 × 16384  ≈ 2.8e14
合计 ≈ 8.0e14 FLOPs  →  8.0e14 / 5e14 ≈ 1.6 s
```

32K 时 attention 已占约 35%；到 128K 时 attention 将占大头。

## 10.3 盈亏平衡带宽：直接与加载对比

把两边换算成每 token 代价：

```
每 token 重算时间 ≈ 2P / 有效算力                    （只算线性部分；c 越大实际越贵）
每 token 加载时间 ≈ 每 token KV 字节 / 存储带宽

盈亏平衡带宽 = 每 token KV 字节 × 有效算力 / 2P
```

存储带宽高于该值，即使 c = 0，加载也比重算快。

| 模型 | 每 token KV 字节 | 2P | 盈亏平衡带宽（500 TFLOPS） |
|------|------|------|------|
| Llama-3-8B（GQA，8 KV 头） | 32 层 × 8 × 256 × 2 B ≈ 128 KB | 1.6e10 | **≈ 4 GB/s** |
| Llama-3-70B（GQA，8 KV 头） | 80 层 × 8 × 256 × 2 B ≈ 320 KB | 1.4e11 | **≈ 1.2 GB/s** |

回到 32K token 的例子（KV 约 4 GB）：

| 方式 | 耗时 |
|------|------|
| 重算 | ~1.6 s |
| 从 CPU 加载（Gen5 x16，50 GB/s） | ~0.08 s |
| 从 CPU 加载（Gen4 x16，25 GB/s） | ~0.16 s |
| 从 SSD 加载（Gen4 x4，7 GB/s） | ~0.57 s |
| 从远端加载（Gen3 x16 ≈ 100GbE，12 GB/s） | ~0.33 s |

几个结论：

- **只看带宽，长前缀从 SSD 加载基本都比重算划算**；模型越大越明显。
- GQA / MLA 模型每 token KV 很小，更偏向加载。
- 可能让重算胜出的是下一节的**固定开销**。

## 10.4 短 prompt：公式会低估，需要下限

token 数少时，prefill 不是计算瓶颈，而是受限于**读权重的显存带宽**——每次前向都要把全部权重从 HBM 读一遍：

```
单次前向下限 ≈ 权重字节 / HBM 带宽
8B bf16：16 GB / 3.35 TB/s ≈ 5 ms
```

实际时间可近似为：

```
T(n, c) ≈ max(权重读取下限, FLOPs / 有效算力) + 固定开销（调度、kernel 启动，~1 ms）
```

加载侧也有固定开销：SSD 单次 I/O 延迟几十 µs，vLLM tiering 的 promotion 还要多等一个调度 step（RETRY）。

因此**几百 token 的短命中，重算往往更快**——这正是 LMCache 设 `min_retrieve_tokens` 的道理。

另外，若请求与其他请求合批，权重读取成本被分摊，重算的边际代价会更低。

## 10.5 更准的做法：实测标定

粗估之后建议实测拟合，因为 MFU 会随长度、kernel、批大小变化：

```
T(n, c) = a + b·n + k·n·(c + n/2)
```

用 `vllm bench` 或离线脚本测几组 `(n, c)`（例如 n ∈ {512, 2K, 8K, 32K}，c 取 0 或 32K，用 prefix cache 命中构造 c），再最小二乘拟合：

- `a`：固定开销
- `b`：每 token 线性代价
- `k`：attention 系数

三个参数就够用，且可直接用于加载/重算决策。

## 10.6 需要调整的因素

- **FP8**：峰值算力约翻倍，重算变便宜，盈亏平衡带宽跟着翻倍。
- **TP**：TP=8 时算力与每卡加载量都按 8 分摊，比值大致不变，但要加上 all-reduce 通信开销。
- **MoE**：P 用激活参数量；总权重仍很大，所以短 prompt 的权重读取下限更高。
- **MLA（DeepSeek）**：每 token KV 非常小，几乎总是加载更划算。
- **chunked prefill**：长 prefill 会按 `max_num_batched_tokens` 拆到多个 step，并与 decode 混批，实际 TTFT 还要加排队时间。

---

# 第十一部分：给 vLLM 的优化建议

## 11.1 可借鉴的机制

1. **SGLang 的超时 + 连续后缀重算**：最实用。它不需要准确的代价模型，只要一个时间预算。对应到 vLLM：secondary tier 的 promotion 超过预算仍未完成时，把 `max_load_tokens` 截到已经到达 CPU 的前缀，剩下的重算。**当前 vLLM 的 tiering 遇到慢存储只会一直返回 RETRY 等下去，没有放弃机制。**
2. **TENT 的预测方式**：用排队字节 + 链路带宽预测完成时间，比事后超时更早决策。vLLM 已有 `READ_TIME / READ_BYTES` 指标，可以做带宽 EMA。
3. **决策位置**：`TieringOffloadingManager.lookup()` 在 secondary 命中时**同时知道命中长度和所在层**，比 LMCache 的 `KVLoadPolicy`（prefetch 前、信息不足）和 Mooncake 的空钩子都更适合放决策逻辑。
4. **代价模型本身**：三家都没有现成的，需要按模型标定 prefill 曲线（第十部分）再与预测加载时间比较。

## 11.2 建议的落地顺序

1. **静态阈值**：加 `min_promotion_tokens` / 按 tier 的 `min_load_tokens`，短命中不 promote。
2. **超时放弃**：promotion 超过时间预算后降级为「已到前缀加载 + 剩余重算」。
3. **自适应**：用 `READ_TIME / READ_BYTES` 的 EMA 作为实时带宽估计，结合第十部分的 `a, b, k` 拟合曲线做 per-chunk 决策。

## 11.3 前置工作

- **测交叉点**：用你的模型实测 prefill 时间随长度变化的曲线，用 `fio` 测 SSD 的 64 KB 随机读 / O_DIRECT 吞吐，找出「多长的前缀开始值得加载」。
- **查重**：按 AGENTS.md 要求，提 PR 前先查 vLLM 是否已有相关 issue / PR，可搜 `load recompute`、`promotion timeout`、`cost aware offload`。

## 11.4 相关研究（未在本仓库验证，需自行核实）

- **CachedAttention / AttentionStore**（ATC'24）：分层 KV 存储，讨论加载与计算重叠。
- **Cake**（*Compute or Load KV Cache? Why not both*, 2024）：直接研究该问题——从两端同时进行，一边从前往后计算、一边从后往前加载，中间汇合。
- **LMCache**：vLLM 生态的外部 KV 缓存层，可作为 vLLM connector 使用。
- **Mooncake**：FAST'25 论文中的 cache-aware prefill scheduling（TTFT 估算）设计。

---

# 附录：关键代码索引

| 主题 | 位置 |
|------|------|
| Tiering 设计原则（primary 为网关） | `vllm/v1/kv_offload/tiering/manager.py:9-19` |
| `lookup()`：CPU 优先，miss 才查 secondary | `tiering/manager.py:382-465` |
| `_initiate_promotion`（secondary→CPU） | `tiering/manager.py:467-515` |
| `complete_store`（cascade 到所有 tier） | `tiering/manager.py:709-755` |
| Tiering 配置解析 / metrics 定义 | `vllm/v1/kv_offload/tiering/spec.py` |
| `_select_swap_blocks_fn`（DMA vs Triton） | `vllm/v1/kv_offload/cpu/gpu_worker.py:39-65` |
| `pin_mmap_region`（cudaHostRegister） | `cpu/gpu_worker.py:201-270` |
| `transfer_async`（独立 stream + event） | `cpu/gpu_worker.py:565-735` |
| `_load_block`（os.open + readv） | `tiering/fs/io.py:133-163` |
| `O_DIRECT` 常量与探测 | `tiering/fs/io.py:26`、`tiering/fs/manager.py:191-201` |
| `DualQueueThreadPool`（读写线程池） | `tiering/fs/manager.py:203-207` |
| backpressure 策略（只管写） | `vllm/v1/kv_offload/tiering/backpressure.py` |
| `max_load_tokens` 上限 | `offloading/scheduler.py:764-768` |
| 不足一个 chunk 跳过 | `offloading/scheduler.py:816-818` |
| `max_offload_tokens` | `offloading/scheduler.py:977-980` |
| Offloading 使用文档 / 配置表 | `docs/features/kv_offloading_usage.md` |
| chunk 定义与默认值 | `docs/features/kv_offloading_usage.md:54-56` |
| 配置项默认值表 | `docs/features/kv_offloading_usage.md:100-113` |

### 外部系统（来自子调研，未逐行复核）

| 主题 | 位置 |
|------|------|
| LMCache `KVLoadPolicy` | `lmcache/v1/multiprocess/kv_load_policy.py` |
| LMCache lookup 调用点 | `lmcache/v1/multiprocess/modules/lookup.py:291-309` |
| LMCache `min_retrieve_tokens` | `lmcache/v1/config.py:462` |
| LMCache vLLM 适配器阈值逻辑 | `lmcache/integration/vllm/vllm_v1_adapter.py:1446-1527` |
| LMCache 按层流水线 | `lmcache/v1/cache_engine.py:1001` |
| Mooncake TENT 截止时间降级 | `docs/source/design/tent/deadline-scheduling.md` |
| Mooncake 准入队列阈值/钩子 | `mooncake-transfer-engine/tent/include/tent/runtime/admission_queue.h:59-103` |
| Mooncake HiCache 预取策略 | `docs/source/design/hicache-design.md:45-62` |
| Mooncake Conductor 前缀索引 | `mooncake-conductor/src/prefixindex/prefix_indexer.cpp:505-665` |
