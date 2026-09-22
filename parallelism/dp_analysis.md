# vLLM 数据并行 (DP) 深度分析

> 本文档从源码层面拆解 vLLM 中 Data Parallelism (DP) 的实现原理，包含数学推导、具体数字例子与关键代码引用（`file:line`）。
>
> 适用读者：已了解 TP / EP / EPLB（参见 `tp_ep_eplb_analysis.md`），希望从"请求切分 + 专家切分"两个角度理解 vLLM DP 的读者。

---

## 目录

- [第一部分：DP 的心智模型（双面性）](#第一部分dp-的心智模型双面性)
- [第二部分：配置层](#第二部分配置层)
- [第三部分：进程组拓扑](#第三部分进程组拓扑)
- [第四部分：稠密模型——独立副本](#第四部分稠密模型独立副本)
- [第五部分：MoE 模型——DP 折叠进 EP](#第五部分moe-模型dp-折叠进-ep)
- [第六部分：Lockstep 与 wave 协调](#第六部分lockstep-与-wave-协调)
- [第七部分：Batch 协调（dp_utils）](#第七部分batch-协调dp_utils)
- [第八部分：DP attention / sequence parallel](#第八部分dp-attention--sequence-parallel)
- [第九部分：请求层负载均衡](#第九部分请求层负载均衡)
- [第十部分：DPCoordinator 与统计发布](#第十部分dpcoordinator-与统计发布)
- [第十一部分：DP 适用场景与选型](#第十一部分dp-适用场景与选型)
- [附录：关键代码索引](#附录关键代码索引)

---

# 第一部分：DP 的心智模型（双面性）

## 1.1 和训练 DP 的本质区别

训练里的 DP：每个 rank 一份完整模型，各吃不同 batch，**梯度 all-reduce** 把参数更新同步。

vLLM 推理里的 DP：没有梯度，没有参数同步。`data_parallel_size > 1` 意味着**启动多个 engine core，每个 engine core 是一份完整模型副本（有自己的 TP/PP 组），请求在这些副本之间做负载均衡**。

但 vLLM 的 DP 有个关键陷阱——**它有两种完全不同的形态**，分界点是"模型是不是 MoE"。这一刀切在 engine core 的构造处（`vllm/v1/engine/core.py:1303-1312`）：

```python
if data_parallel and vllm_config.model_config.is_moe:
    # MoE：DP rank 属于模型的一部分，专家在 DP×TP×PCP 上分片
    parallel_config.data_parallel_rank = dp_rank
    engine_core = DPEngineCoreProc(*args, **kwargs)      # lockstep（耦合）
else:
    # 稠密：DP rank 完全独立，等价于 DP=1 的多实例
    parallel_config.reconfigure_for_independent_dp_rank()
    engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
```

| | 稠密模型（dense） | MoE 模型 |
|---|---|---|
| DP 是什么 | **独立副本**，等价于多实例 | **EP 的额外维度**，专家分片维 |
| DP rank 之间 | 不通信，各跑各的请求 | 通信（all-to-all 路由 token） |
| 请求 batch | 每个 core 独立调度 | **lockstep**：必须步调一致 |
| engine core 类型 | `EngineCoreProc` | `DPEngineCoreProc` |

这一章先记住结论，后面 4、5 两章分别展开。

## 1.2 为什么 MoE 的 DP 会耦合

一句话：**EP 的通信半径是 all-to-all，它要求参与 all-to-all 的 rank 步调一致**。

`FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp`（`vllm/model_executor/layers/fused_moe/config.py:1116`）把 DP 折叠进 EP：

```python
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

EP 组横跨 `DP × PCP × TP` 个 rank（`parallel_state.py:1918-1927`）。这意味着一批 token 的专家计算被摊到多个 DP rank 上，这些 rank 必须**同时**进入 all-to-all，否则一方的 `dispatch` 在等另一方永远不来的 token，直接死锁。这就是"lockstep"的根源。

## 1.3 为什么 MoE 的 DP 不复制专家

稠密模型的 DP 是"复制"，MoE 模型的 DP 是"切分"，根因是两者的**参数大头性质相反**：

| | 稠密模型 | MoE 模型 |
|---|---|---|
| 参数大头 | 每层 1 个 FFN，`d × intermediate` | 每层 N 个专家 FFN，`N × d × intermediate` |
| 每个 token 激活多少权重 | **全部**（每个权重每个 token 都用） | **只有 top_k 个专家**（如 8/256） |
| 复制一倍的代价 | 参数翻倍（70B→140B，可接受） | 专家翻倍（650B→1.3T，不可接受） |
| 切分一倍的代价 | 每个 token 都跨 rank 归约（all-reduce） | 每个 token 只去 top_k 个 rank（all-to-all） |

- **稠密：复制划算**。每个 token 都要碰每一个权重，没有稀疏性可切。TP（切）会让每个 token 的算劈到 p 卡再 all-reduce；DP（复制）零通信但内存翻倍。稠密模型（如 70B）单卡/几卡能装下，所以复制是最划算的吞吐扩展。
- **MoE：切分划算**。MoE 存在的意义就是"参数多、每个 token 只激活一小部分"。复制会让 248/256 个专家在每份副本里几乎不被激活，纯浪费内存；切分（EP）把专家摊到 DP×TP 卡上，每个 token 只去 top_k 卡——内存 = 一份专家，通信只搬 top_k 份 token。

所以同一个 `data_parallel_size`，在稠密里是"复制"，在 MoE 里是"切专家"。

> MoE 想"复制几份提吞吐"怎么办？那属于 **ExternalDP**（`all_ranks` 布局里的 `-1` 维，verl 集成用，见第三部分），或多起几个独立 vLLM 实例，而不是 `data_parallel_size`。

---

# 第二部分：配置层

DP 的配置集中在 `ParallelConfig`（`vllm/config/parallel.py:119`）：

| 字段 | 行 | 含义 |
|---|---|---|
| `data_parallel_size` | `:129` | DP 组数（engine core 数） |
| `data_parallel_size_local` | `:132` | 本节点内的 DP 数，0 表示由 engine-args 层外部指定 |
| `data_parallel_rank` | `:136` | 当前 rank 的 DP 序号 |
| `data_parallel_rank_local` | `:139` | 本节点内的 DP 序号，仅 SPMD 模式 |
| `data_parallel_backend` | `:147` | `"mp"` 或 `"ray"` |
| `data_parallel_external_lb` | `:149` | 外部 LB 模式（K8s "one-pod-per-rank" wide-EP） |
| `data_parallel_hybrid_lb` | `:156` | 混合 LB：vLLM 只在本节点 DP rank 间均衡，节点间靠外部 LB |

## 2.1 world_size 与 DP 的关系

`__post_init__`（`parallel.py:831-837`）：

```python
self.world_size = (
    self.pipeline_parallel_size
    * self.tensor_parallel_size
    * self.prefill_context_parallel_size
)
```

**DP 不进入 `world_size`**。每个 engine core 自己的 world size 仍是 `PP × TP × PCP`，DP 是通过"复制多个 engine core 进程"实现的，不是把一个进程扩成多个 rank。

`world_size_across_dp`（`parallel.py:548-551`）：

```python
@property
def world_size_across_dp(self) -> int:
    return self.world_size * self.data_parallel_size
```

## 2.2 几个决定 DP 行为的派生属性

这三个 property 是理解 MoE DP 的关键开关（`parallel.py:672-707`）：

```python
@property
def use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend in (...)
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
        and self.data_parallel_size > 1          # ← DP 参与才开启
    )

@property
def use_all2all(self) -> bool:
    return (
        self.data_parallel_size > 1              # ← DP>1 直接用 all2all
        or self.use_sequence_parallel_moe
        or (self.enable_expert_parallel and self.prefill_context_parallel_size > 1)
    )

@property
def use_batched_dp_moe(self) -> bool:
    return (
        self.all2all_backend in ("deepep_low_latency", "nixl_ep")
        and self.enable_expert_parallel
        and self.data_parallel_size > 1
    )
```

- `use_all2all`：只要 `DP > 1` 就强制走 all-to-all（因为专家分片了，token 必须路由）。
- `use_sequence_parallel_moe`：见第八部分，DP 与 TP 同时 >1 时才启用。

---

# 第三部分：进程组拓扑

`initialize_model_parallel`（`vllm/distributed/parallel_state.py:1742`）一次性构建 6 个进程组。核心是 `all_ranks` 的五维 reshape（`parallel_state.py:1808-1823`）：

```python
# the layout order is: ExternalDP x DP x PP x PCP x TP
# ExternalDP is the data parallel group that is not part of the model
# DP is the data parallel group that is part of the model
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size,
)
```

然后每个维度通过"转置到最后一维 → reshape 二维 → unbind"来抽取该维度的分组：

| 组 | 代码 | 抽法 | 语义 |
|---|---|---|---|
| `_TP` | `:1828` | `view(-1, TP).unbind(0)` | 相同 DP/PP/PCP、不同 TP |
| `_DCP` | `:1852` | `reshape(-1, DCP).unbind(0)` | decode context parallel |
| `_PCP` | `:1863` | `transpose(3,4)` | 相同 DP/PP/TP、不同 PCP |
| `_PP` | `:1883` | `transpose(2,4)` | 相同 DP/TP/PCP、不同 PP |
| `_DP` | `:1899` | `transpose(1,4).reshape(-1, DP)` | 相同 TP/PP/PCP、不同 DP |
| `_EP` | `:1918` | `transpose(1,2).reshape(-1, DP×PCP×TP)` | 专家在 DP×PCP×TP 上分片 |

## 3.1 具体数字例子：DP=2, TP=2, PP=1, PCP=1

`world_size = 2×1×1×2 = 4`，rank 编号 0~3。`all_ranks = arange(4).reshape(1, 2, 1, 1, 2)`：

```
位置 [ExternalDP=0, DP, PP=0, PCP=0, TP]:
  DP=0: [0, 1]      ← rank 0 是 TP0，rank 1 是 TP1
  DP=1: [2, 3]      ← rank 2 是 TP0，rank 3 是 TP1
```

- **`_TP` 组**：`view(-1,2)` → `[[0,1],[2,3]]` → `{0,1}`、`{2,3}`。同 DP、不同 TP。✓
- **`_DP` 组**：`transpose(1,4)` 把 DP 换到最后一维，再 `reshape(-1,2)` → `[[0,2],[1,3]]` → `{0,2}`、`{1,3}`。

```
_DP 组 = 相同 TP/PP/PCP 位置、不同 DP 的 rank
       {0, 2}：都是"DP 副本里的 TP0"
       {1, 3}：都是"DP 副本里的 TP1"
```

- **`_EP` 组**：`transpose(1,2).reshape(-1, 2×1×2=4)` → `[[0,1,2,3]]` → `{0,1,2,3}`，即**全部 4 个 rank**。专家在全部 rank 上分片。✓

## 3.2 `_DP` 组 vs `_EP` 组的角色分工

```
        DP rank 0            DP rank 1
        ┌──────┬──────┐     ┌──────┬──────┐
        │ TP0  │ TP1  │     │ TP0  │ TP1  │
        │ rank0│ rank1│     │ rank2│ rank3│
        └──────┴──────┘     └──────┴──────┘
          │      │             │      │
          └─ _DP 组 {0,2} ─────┘      └─ _DP 组 {1,3}
          └────────── _EP 组 {0,1,2,3} ──────────┘
```

- **`_EP` 组**：做 MoE 的 all-to-all（token 路由），跨 DP 搬运。
- **`_DP` 组**：做 lockstep 协调（见第六、七部分）——比如"全体是否还有未完成请求""batch 大小对齐到多少"。它是 CPU gloo 组，因为协调发生在 engine 进程（可能没 CUDA 设备）。

`get_dp_group()`（`parallel_state.py:1413`）返回这个组，`dp_utils.py` 和 `forward_context.py` 都从它拿 `cpu_group` / `device_group`。

---

# 第四部分：稠密模型——独立副本

稠密模型的 DP 极其简单：每个 DP rank 就是一个独立的 engine core，和"起 N 个独立 vLLM 实例"等价，只是共享一个 API server 做 LB。

关键在 `reconfigure_for_independent_dp_rank`（`parallel.py:1038-1047`）：

```python
def reconfigure_for_independent_dp_rank(self) -> None:
    """Reconfigure for a single independent non-MoE DP rank."""
    nnodes = self.nnodes_within_dp
    node_rank = self.node_rank_within_dp
    self.data_parallel_size = 1          # 每个副本眼里 DP=1
    self.data_parallel_size_local = 1
    self.data_parallel_rank = 0
    self.nnodes = nnodes
    self.node_rank = node_rank
```

把 DP 字段全部"折叠回 1"，只保留 `data_parallel_index`（`parallel.py:375`）标记"我原来是第几个副本"，用于 LB 路由和日志。这样稠密模型**完全感知不到 DP 的存在**，模型代码、进程组、调度器都按 DP=1 跑。

**为什么稠密模型不用 lockstep**：因为稠密模型里每个 engine core 的 batch 完全独立，没有 all-to-all，请求 A 在 core 0 算和 core 1 算没有任何跨 core 依赖，自然不需要协调。这也是为什么离线 SPMD 模式下稠密 DP 直接被拒绝（`parallel.py:899-903`）：

```python
if self.data_parallel_size > 1 and self.is_moe_model is False:
    raise ValueError("Offline data parallel mode is not supported/useful for dense models.")
```

（离线场景没有 API server 做 LB，多个独立副本无法被路由到，所以"没用"。）

---

# 第五部分：MoE 模型——DP 折叠进 EP

## 5.1 专家在 DP 维度的分片

MoE 模型的 DP 不是"独立副本"，而是**专家切分的第三个维度**（前两个是 TP 和 PCP）。

`_EP` 组的构建（`parallel_state.py:1914-1946`）：

```python
global _EP
assert _EP is None, "expert parallel group is already initialized"
if config.model_config is None or config.model_config.is_moe:
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_model_parallel_size
            * tensor_model_parallel_size,
        )
        .unbind(0)
    )
    ...
    _EP = init_model_parallel_group(group_ranks, ..., group_name="ep", use_all2all=use_all2all)
```

EP 组大小 = `DP × PCP × TP`。结合 TP/EP 文档里的 `flatten_tp_across_dp_and_pcp`，完整的折叠链是：

```
TP=2, DP=2, PCP=1, enable_ep=True
  → flatten_tp_size = 2 × 1 × 2 = 4
  → ep_size = 4, ep_rank = dp_rank × pcp × tp + ... ∈ {0,1,2,3}
  → 专家被切成 4 份，分散在 rank 0,1,2,3
```

即：**原本"DP 副本"的语义被 EP 完全吸收，DP rank 变成了专家分片的物理载体**。

## 5.2 一个具体数字例子

8 专家、TP=2、DP=2、EP 开启：

```
              无 DP（EP=2）             有 DP（EP=4，折叠 DP）
rank 0 (DP0/TP0): 专家 0,1,2,3      rank 0 (DP0/TP0): 专家 0,1
rank 1 (DP0/TP1): 专家 0,1,2,3      rank 1 (DP0/TP1): 专家 2,3
                                     rank 2 (DP1/TP0): 专家 4,5
                                     rank 3 (DP1/TP1): 专家 6,7
        ↑ 专家在 TP 上重复             ↑ 专家在 DP×TP 上唯一分片
```

没有 DP 时，TP 两卡各持一份完整专家集（TP 切的是专家内部 intermediate 维，不是专家集）；有 DP 后，专家集被唯一地摊到 DP×TP=4 卡上，**每卡专家数减半，单卡内存减半**。这是 MoE DP 的第二个价值：**不仅多副本提吞吐，还能靠 EP 降低单卡专家内存**。

## 5.3 TP=2, DP=2 的两种切法：EP 开 vs 关

一个容易误解的点：**不管开不开 `--enable-expert-parallel`，专家都会切到 4 张卡上，只是切法不同**。`FusedMoEParallelConfig.make` 的判断（`config.py:1208-1211`）：

```python
use_ep = (dp_size * pcp_size * tp_size > 1) and enable_expert_parallel
#        = (2 × 1 × 2 > 1)         and enable_expert_parallel
```

### 情况 A：默认（`enable_expert_parallel=False`）→ TP=4

`use_ep=False`，MoE 层走 `flatten_tp_across_dp_and_pcp`，DP 折叠进 TP。docstring 原文（`config.py:1173-1181`）：

```
When TP = 2, DP(PCP) = 2 and EP = False:
- device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
- device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
- device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
- device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
- Comment: tensors are sharded across 4 devices.
```

**切法**：`TP={4, rank}` —— 每个专家内部沿 intermediate 维切 4 份，每卡持全部 N 个专家、但每个专家只有 1/4 intermediate。通信 all-reduce。

### 情况 B：`--enable-expert-parallel` → EP=4

`use_ep=True`，`ep_size=4`、`tp_size=1`。docstring 原文（`config.py:1198-1206`）：

```
When TP = 2, DP(PCP) = 2 and EP = True:
- device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
- device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
- device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
- device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
- Comment: experts are split between the 4 devices.
```

**切法**：`EP={4, rank}` —— 专家集合切 4 份，每卡持 N/4 个完整专家。通信 all-to-all。

### 对比

| | 情况 A：TP=4 | 情况 B：EP=4 |
|---|---|---|
| 每卡持有什么 | 全部 N 专家，各切 1/4 intermediate | N/4 个完整专家 |
| 每卡专家内存 | 总专家 ÷ 4 | 总专家 ÷ 4 |
| 通信 | all-reduce（跨 4 卡） | all-to-all（只去 top_k 卡） |
| token 去哪 | 每个 token 碰全部 4 卡 | 每个 token 只碰 top_k 卡 |

**每卡专家内存两者一样（都 ÷4），区别只在通信**。`enable_expert_parallel` 决定"用哪种切法"，不决定"切不切"。

**隐含结论**：因为 `core.py:1303-1312` 里 MoE + DP 永远走 `DPEngineCoreProc`（lockstep），不管开不开 EP，MoE 的 DP 都参与专家分片、都 lockstep。区别只是 EP 关时 DP 折叠进 TP（all-reduce）、EP 开时 DP 折叠进 EP（all-to-all）。

---

# 第六部分：Lockstep 与 wave 协调

MoE DP 的 engine core 是 `DPEngineCoreProc`（`core.py:1306`），它比普通 engine 多了一套"全局 running/paused 状态机"。

## 6.1 为什么需要 wave

因为 DP rank 的 all-to-all 是 lockstep 的：如果一个 rank 停步而另一个还在跑，停的那个会在下一次 all-reduce/all-to-all 里永远等一个不来的消息。所以**要么全体跑，要么全体停**。vLLM 用"wave（请求波）"来命名一次"全体从 running → paused"的周期：

- 一个 engine 收到请求 → 通知 coordinator 唤醒所有 engine（`START_DP_WAVE`）。
- 所有 engine 一起跑，直到全体都没有未完成请求。
- 全体一起进入 paused，wave 数 +1。

## 6.2 主循环：每 32 步 all-reduce 一次

`DPEngineCoreProc` 的 stepping 循环末尾（`core.py:2138-2161`）：

```python
# 3) All-reduce operation to determine global unfinished reqs.
self.engines_running = self._has_global_unfinished_reqs(local_unfinished_reqs)

if not self.engines_running:
    if self.dp_rank == 0 or not self.has_coordinator:
        self.output_queue.put_nowait(
            (client_index, EngineCoreOutputs(wave_complete=self.current_wave)))
    self.current_wave += 1
    self.step_counter = 0
```

`_has_global_unfinished_reqs`（`core.py:2165-2179`）：

```python
def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
    self.step_counter += 1
    if self.step_counter % 32 != 0:
        return True                       # 优化：每 32 步才真正同步一次

    has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
        self.dp_group,
        has_unfinished=local_unfinished,
        pending_pause=self.pending_pause,
    )
    if pause_consensus:
        self.ignore_start_dp_wave = True
        self.pending_pause = False
```

**优化点**：不是每步都 all-reduce（那样 CPU 同步会拖慢吞吐），而是每 32 步同步一次。中间 31 步乐观地假设"还有活干"（返回 True）。

## 6.3 `sync_dp_state`：一次 SUM 干两件事

`sync_dp_state`（`parallel.py:738-762`）：

```python
@staticmethod
def sync_dp_state(dp_group, has_unfinished: bool, pending_pause: bool):
    tensor = torch.tensor([int(has_unfinished), int(pending_pause)],
                          dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=dp_group)
    dp_size = dp_group.size()
    pause_count = tensor[1].item()
    has_unfinished_global = tensor[0].item() > 0 or pause_count % dp_size != 0
    return has_unfinished_global, pause_count == dp_size
```

用一个 2 元素 tensor 的 SUM，同时算两件事：

- **元素 [0]**：`SUM > 0` ≡ 逻辑 OR。任何 rank 有未完成请求 → 全局继续跑。
- **元素 [1]**：`SUM == dp_size` ≡ 逻辑 AND。所有 rank 都举手要暂停 → pause 达成共识。

这是把两个逻辑量编码进一次 all-reduce 的经典 trick（省一次通信）。

## 6.4 两阶段暂停（防竞态）

`_pause_complete`（`core.py:1984-2000`）实现两阶段暂停协议：

- **Phase 1**：本地置 `pending_pause=True`，并强制 `engines_running=True`，让所有 rank 都进入 stepping 循环、到达 all-reduce 检查点。
- **Phase 2**：在 `_has_global_unfinished_reqs` 里，一旦 all-reduce 确认**所有** rank 都 `pending_pause`，集体停步并置 `ignore_start_dp_wave=True`，防止陈旧的 `START_DP_WAVE` 消息重新唤醒引擎。

这个协议要防的竞态：rank A 先暂停、rank B 后暂停，如果 A 停步后 B 才收到新请求，B 会发 `START_DP_WAVE` 想把 A 叫醒，但 A 已经"决定休息"了。`ignore_start_dp_wave` 就是让这种 stale 唤醒失效。

---

# 第七部分：Batch 协调（dp_utils）

这是 MoE DP 最精细的地方：**即使请求是独立调度的，每个 DP rank 每步要处理的 batch 形状也必须对齐**，否则 cudagraph 重放和 all-to-all 都对不上。

## 7.1 `coordinate_batch_across_dp`

入口 `coordinate_batch_across_dp`（`vllm/v1/worker/dp_utils.py:164`）：

```python
if parallel_config.data_parallel_size == 1:
    return False, None, cudagraph_mode     # 早退

should_attempt_ubatching = check_ubatch_thresholds(...)
(should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
    _synchronize_dp_ranks(num_tokens_unpadded, num_tokens_padded,
                          should_attempt_ubatching, cudagraph_mode,
                          parallel_config))
```

每个 rank 贡献 4 个量，通过一次 all-reduce 广播（`_run_ar`，`dp_utils.py:36-54`）：

```python
tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)
tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch   # 原始 token 数
tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch # 填充后 token 数
tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0    # 是否要 ubatch
tensor_cpu[3][dp_rank] = cudagraph_mode               # cudagraph 模式
```

## 7.2 三个协调规则

all-reduce 之后，用三个后处理函数做决策：

**① cudagraph 模式取 min**（`_post_process_cudagraph_mode`，`dp_utils.py:92`）：

```python
return int(tensor[3, :].min().item())
```

任何 rank 想跑 eager（NONE=0），全体跑 eager。因为 cudagraph 需要固定 batch 形状，一个 rank 不满足条件会破坏锁步。

**② 是否 ubatch 取 AND**（`_post_process_ubatch`，`dp_utils.py:57-74`）：

```python
should_ubatch = bool(torch.all(tensor[2] == 1).item())  # 全体=1 才 ubatch
if not should_ubatch:
    return False
# 再检查：如果 ubatch 后第二个 micro-batch 是空的，放弃 ubatch
if is_last_ubatch_empty(orig_min, padded_max, num_ubatches):
    should_ubatch = False
```

**③ DP padding 取 max**（`_post_process_dp_padding`，`dp_utils.py:77-89`）：

```python
if should_dp_pad:
    max_num_tokens = int(num_tokens_across_dp.max().item())
    return torch.tensor([max_num_tokens] * len(num_tokens_across_dp), ...)
```

当 cudagraph 或 ubatch 激活时，所有 rank 的 token 数 pad 到**跨 rank 最大值**。

## 7.3 为什么必须 pad 到一致

```
        rank 0                rank 1
请求:   50 tokens             80 tokens          ← 各自独立调度
        │                     │
all-reduce 协调 ↓
        pad 到 80             pad 到 80          ← cudagraph 需要统一形状
        │                     │
all-to-all（MoE dispatch）: 两个 rank 的 token 数对齐，才能正确收发
```

- **cudagraph 视角**：graph 是按 token 数捕获的，各 rank 形状不同就无法重放同一个 graph。
- **all-to-all 视角**：`dispatch` 是"我发你多少 token，你发我多少 token"的协议，两侧计数必须一致。

调用点在 `gpu_model_runner.py:4097-4119`：

```python
if self.vllm_config.parallel_config.data_parallel_size > 1:
    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        coordinate_batch_across_dp(...))
    if num_tokens_across_dp is not None:
        dp_rank = self.parallel_config.data_parallel_rank
        num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
        # 用 DP padding 后的 token 数重新 dispatch cudagraph
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(...)
```

cudagraph 维度的协调还有另一个实现 `sync_cudagraph_and_dp_padding`（`vllm/v1/worker/gpu/dp_utils.py:16`），逻辑同构：把 `num_tokens`、`cg_mode`、`uniform_token_count` 塞进一个 3×dp_size 的 tensor，all-reduce 后取 min/max。

## 7.4 dummy run：空 batch 也不能跳过

因为 lockstep，一个 rank 没活干也不能直接 return，否则别的 rank 会在 all-to-all 里等它。所以有空 batch 的 rank 要跑 **dummy run**（用随机输入喂一次模型，`gpu_model_runner.py:5839` 的 `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` 甚至随机化 dummy 输入，避免 dummy 的专家选择把某几个专家刷成热点）。

---

# 第八部分：DP attention / sequence parallel

## 8.1 问题：TP 的 o_proj 会"复制" token

回顾 TP 文档：attention 的 `o_proj` 是 row-parallel，出口做 all-reduce，结果每个 TP rank 都拿到**复制态**的完整 hidden state。

如果直接把这个复制态的 hidden state 送进 MoE（EP 已经切了专家），那么**同一个 token 会在多个 TP rank 上被重复计算、重复 all-to-all**。TP 越大，重复越严重。

## 8.2 解法：sequence parallel MoE

`use_sequence_parallel_moe`（`parallel.py:673`）在 `TP > 1` 且 `DP > 1` 时开启。核心思想：MoE 层的输入**不复制**，而是按 token 维（sequence）切分到 TP 和 DP rank 上，每个 token 只被一个 rank 算一次。

`DPMetadata`（`forward_context.py:72-99`）记录了这个切分：

```python
@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor   # 每个 DP rank 各有多少 token

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        # 跨 sequence parallel rank 的累计 token 偏移
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)
```

`cu_tokens_across_sp` 算出"每个 SP rank 该处理的 token 区间"，MoE 层据此把 token 分发到 `DP × TP` 的 rank 网格上，实现真正的 sequence parallel——和 TP/EP 文档里的 `use_all2all` / dispatch / combine 是同一套机制。

## 8.3 一个具体数字例子

TP=2, DP=2，一批 8 个 token：

```
没有 SP（复制）：            有 SP（sequence parallel）：
rank 0 (DP0/TP0): 8 tokens   rank 0: token 0-1
rank 1 (DP0/TP1): 8 tokens   rank 1: token 2-3
rank 2 (DP1/TP0): 8 tokens   rank 2: token 4-5
rank 3 (DP1/TP1): 8 tokens   rank 3: token 6-7
   ↑ 32 token 份计算，4 倍浪费      ↑ 8 token 份计算，零浪费
```

SP 把每个 token 唯一地放到一个 rank 上，MoE 的 all-to-all 只搬"真正属于该 rank 专家"的 token，省掉了复制态的重复计算。

---

# 第九部分：请求层负载均衡

请求进哪个 engine core，由 `DPLBAsyncMPClient.get_core_engine_for_request` 决定（`vllm/v1/engine/core_client.py:1471-1522`）。

## 9.1 路由规则

```python
def get_core_engine_for_request(self, request) -> EngineIdentity:
    if (eng_index := request.data_parallel_rank) is None and (
        eng_index := get_late_interaction_engine_index(...)) is None:
        # 动态 LB：扫所有 engine 取分数最小者
        ...
        for i in range(num_engines):
            idx = (self.eng_start_index + i) % num_engines
            waiting, running, kv_cache_usage = current_counts[idx]
            inflight = self.engine_inflight[self.core_engines[idx]]
            score = max(self.client_count * inflight, waiting + running)
            if waiting:
                # waiting 请求按 KV cache 压力加权（>50% 用量才开始罚）
                score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
            if score < min_score:
                min_score = score
                eng_index = idx
    ...
    self.reqs_in_flight[request.request_id] = chosen_engine   # 用于 abort 路由
    self.engine_inflight[chosen_engine] += 1
```

## 9.2 打分公式

```
score = max(client_count × inflight,  waiting + running)
      + waiting × 6.0 × max(0, kv_cache_usage - 0.5)
```

拆解：

- **`inflight` 是精确下界**：本 client 自己发出的、还没完成的请求数，不能被 coordinator 的统计快照回滚抹掉。乘以 `client_count` 归一化到全局量纲。
- **`waiting + running` 是 coordinator 快照**：反映其他 client / 陈旧请求带来的负载。
- **KV cache 压力罚**：engine 有排队（waiting>0）且 KV cache 使用率 > 50% 时，罚分从 0 线性涨到 `3×waiting`。含义：KV 打满的 engine 队列走得慢，新请求应强烈避开；KV 充足时队列只是瞬时（突发中），不罚以保持精确 round-robin。

## 9.3 两个细节

- **起始偏移轮转**：`self.eng_start_index = (self.eng_start_index + 1) % num_engines`（`:1516`），让平局（分数相同）不总是偏向同一个 engine。
- **本地 waiting 计数**：路由后 `current_counts[eng_index][0] += client_count`（`:1510`），在 coordinator 两次统计发布之间（100ms 间隔）本地先垫高该 engine 的 waiting，让突发能分散到多 engine。

---

# 第十部分：DPCoordinator 与统计发布

`DPCoordinator`（`vllm/v1/engine/coordinator.py:23`）是一个独立进程，夹在多个 engine rank 和 API server 之间，职责有三：

## 10.1 收集并发布负载统计

从每个 engine 的 `scheduler_stats` 里拿 `num_waiting_reqs`、`num_running_reqs`、`kv_cache_usage`（`coordinator.py:414-416`），打包后 publish 给所有 API server（`:281-283`），供 LB 打分用。统计变化时 100ms 发布一次，否则 5s 兜底（`:261`）。

## 10.2 维护 wave 状态机

```python
current_wave = 0        # 当前请求波
engines_running = False # 全局 running/paused
```

- 收到 engine 的 `wave_complete` → `current_wave += 1`，`engines_running = False`（`:422-435`）。
- 收到 API server 的新请求（paused 时）→ 广播 `START_DP_WAVE` 唤醒（`:349-365`，`_send_start_wave` `:457`）。

## 10.3 弹性 EP 的 scale up/down

coordinator 收到 `SCALE_ELASTIC_EP` 消息后动态增删 `EngineState` 列表（`:311-345`），配合 `core_client.py:1617-1873` 的弹性 EP 逻辑完成 DP rank 的动态扩缩容（`prepare_elastic_ep` / `_commit_scale_up_elastic_ep` / `_commit_scale_down_elastic_ep`）。

---

# 第十一部分：DP 适用场景与选型

DP 是**吞吐维度**，不是**时延/容量维度**：它提升"单位时间服务的请求数"，不降低单请求时延，也不帮你把单个大模型塞进卡里（那是 TP/PP 的活）。

## 11.1 四个适用场景

**① 稠密模型——吞吐翻倍（副本 scale-out）**
模型能装进单卡（或小 TP 组），瓶颈是并发请求数而非单请求速度（典型：decode-heavy 在线服务）。`--data-parallel-size N` 起 N 个独立副本，API server 内部 LB。等价于起 N 个独立 vLLM 实例，只是共享一个 API server 和统一 LB。

**② MoE 模型——wide-EP 降专家内存**
大 MoE（DeepSeek/Kimi/MiniMax）专家是内存大头，单卡装不下所有专家。`--data-parallel-size N --enable-expert-parallel` 把专家在 DP×TP×PCP 上分片，每卡专家内存 ÷ N。MoE 里 `data_parallel_size` 语义自动从"复制"切换成"切专家"。

**③ K8s 宽 EP（one-pod-per-rank）**
部署在 Kubernetes，想用编排器做弹性扩缩和健康管理。`--data-parallel-rank -dpn` 每个 DP rank = 一个 pod（仅 MoE，非 MoE 请直接起独立实例，见 `arg_utils.py:1076-1079`）；`--data-parallel-multi-port-external-lb -dpm` 节点内每个 rank 一个 API server + supervisor 聚合健康。

**④ 混合 LB（跨节点）**
多节点，节点内 rank 少、节点间要自己均衡。`--data-parallel-hybrid-lb -dph`：节点内 vLLM 做 DP LB，节点间靠外部 LB（与 external-lb 互斥，见 `arg_utils.py:2048-2052`）。

## 11.2 什么时候不该用 DP

| 你的痛点 | 该用 | 不该用 DP 的原因 |
|---|---|---|
| 单请求太慢 | TP | DP 不降时延，甚至因 LB 竞争微增 |
| 模型权重装不进卡 | TP/PP | DP 复制只会更装不下 |
| 模型层数太多 | PP | DP 是整份模型，帮不了切层 |
| MoE 专家负载不均 | EP + EPLB | DP 是 EP 的一个维度，不是替代 |

## 11.3 一条判断规则

```
模型装得进单卡/小 TP 组？
├─ 是，且要吞吐        → DP（稠密：副本）
└─ 否
   ├─ 稠密权重太大      → 先 TP/PP 切，再叠加 DP 提吞吐
   └─ 专家太大          → EP（MoE），DP 作为 EP 的额外分片维
```

**DP 与 TP 是叠加关系，不是二选一**：TP 负责"切"（单请求能跑/跑得快），DP 负责"复制"（并发够多）。例如 TP=2 塞 2 卡跑一个请求、DP=4 复制 4 份 → 8 卡、并发 ×4。这是最常见的生产配置。

一句话收尾：**要单请求快用 TP，要并发多用 DP，要省专家内存用 EP（MoE 的 DP 自动并入 EP）。**

---

# 附录：关键代码索引

| 概念 | 位置 |
|---|---|
| `ParallelConfig` 的 DP 字段 | `vllm/config/parallel.py:129-162` |
| `world_size_across_dp` | `vllm/config/parallel.py:548` |
| `use_sequence_parallel_moe` | `vllm/config/parallel.py:673` |
| `use_all2all` | `vllm/config/parallel.py:690` |
| `use_batched_dp_moe` | `vllm/config/parallel.py:698` |
| `has_unfinished_dp` / `sync_dp_state` / `sync_kv_cache_memory_size` | `vllm/config/parallel.py:727` / `:738` / `:765` |
| `reconfigure_for_independent_dp_rank` | `vllm/config/parallel.py:1038` |
| dense vs MoE 的分叉（构造 engine core） | `vllm/v1/engine/core.py:1303-1312` |
| `DPEngineCoreProc` 主循环 / wave | `vllm/v1/engine/core.py:2138-2161` |
| `_has_global_unfinished_reqs`（每 32 步） | `vllm/v1/engine/core.py:2165` |
| 两阶段暂停 `_pause_complete` | `vllm/v1/engine/core.py:1984` |
| `all_ranks` 五维 reshape | `vllm/distributed/parallel_state.py:1817` |
| `_TP` / `_PCP` / `_PP` / `_DP` / `_EP` 组构建 | `parallel_state.py:1828` / `:1863` / `:1883` / `:1899` / `:1918` |
| `get_dp_group` | `vllm/distributed/parallel_state.py:1413` |
| `coordinate_batch_across_dp` | `vllm/v1/worker/dp_utils.py:164` |
| `_run_ar` / 三个后处理 | `vllm/v1/worker/dp_utils.py:36` / `:57` / `:77` / `:92` |
| cudagraph + DP padding 协调 | `vllm/v1/worker/gpu/dp_utils.py:16` |
| `DPMetadata` / `cu_tokens_across_sp` | `vllm/forward_context.py:72` / `:123` |
| `coordinate_batch_across_dp` 调用点 | `vllm/v1/worker/gpu_model_runner.py:4097` |
| dummy run 随机化 | `vllm/v1/worker/gpu_model_runner.py:5839` |
| `DPLBAsyncMPClient.get_core_engine_for_request`（LB 打分） | `vllm/v1/engine/core_client.py:1471` |
| `DPCoordinator` / `DPCoordinatorProc` | `vllm/v1/engine/coordinator.py:23` / `:146` |
| `_send_start_wave` | `vllm/v1/engine/coordinator.py:457` |
| 弹性 EP 扩缩容 | `vllm/v1/engine/core_client.py:1617-1873` |
| `FusedMoEParallelConfig.make`（use_ep 判断） | `vllm/model_executor/layers/fused_moe/config.py:1128` |
| `flatten_tp_across_dp_and_pcp`（DP 折叠进 TP/EP） | `vllm/model_executor/layers/fused_moe/config.py:1116` |
| DP CLI 参数 | `vllm/engine/arg_utils.py:1069-1131` |
