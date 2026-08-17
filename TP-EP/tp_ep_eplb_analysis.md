# vLLM 张量并行 (TP)、专家并行 (EP) 与 EPLB 深度分析

> 本文档记录了从源码层面拆解 vLLM 中 TP / EP / EPLB 的实现原理，包含数学推导、具体数字例子与关键代码引用（`file:line`）。
>
> 适用读者：已了解 Transformer / MoE 基本结构，希望从"并行切分"的角度理解 vLLM 源码的读者。

---

## 目录

- [第一部分：TP（Tensor Parallelism）](#第一部分tp-tensor-parallelism)
- [第二部分：EP（Expert Parallelism）](#第二部分ep-expert-parallelism)
- [第三部分：EPLB（Expert Parallel Load Balancing）](#第三部分eplb-expert-parallel-load-balancing)
- [附录：关键代码索引](#附录关键代码索引)

---

# 第一部分：TP（Tensor Parallelism）

## 1.1 核心心智模型：数据的两种"态"

理解 TP 的关键，是认识到数据在流水线里只有两种状态：

```
        复制态 (replicated)               分片态 (sharded)
        ┌──────────────────┐            ┌──────────────────┐
        │ 每个 rank 都一样  │            │ 每个 rank 不同   │
        │ (完整 x)         │            │ (按 head/列切)    │
        └──────────────────┘            └──────────────────┘
              ▲    │                          ▲    │
              │    │ column 层                 │    │ row 层
              │    │ (输入复制→输出分片)          │    │ (输入分片→输出复制)
              │    ▼                          │    ▼
             qkv/gate_up/proj              o_proj/down_proj
```

- **column 层**：复制态 → 分片态，**零通信**（切分是免费的，因为下一层就要分片）。
- **row 层**：分片态 → 复制态，**必须 all-reduce**（把"部分和"合并成完整复制态）。

all-reduce 的唯一作用，就是**把分片态变回复制态**，好让下一个 column 层能接收。整条流水线在两种态之间交替，all-reduce 只出现在 row 层的出口。

## 1.2 Column-parallel（列并行）

### 数学定义

`Y = XA + b`，其中 `A` 按【列】切成 `A = [A_1, ..., A_p]`（`linear.py:404-405`）。切的是**权重矩阵的列**，即输出维度。

```
               A (切列)               Y (切列)
        ┌────────┬────────┐     ┌────────┬────────┐
   X    │   A_0  │   A_1  │  =  │   Y_0  │   Y_1  │
 ┌────┐  │        │        │     │        │        │
 │完整│ ×│ (rank0)│ (rank1)│     │ (rank0)│ (rank1)│
 └────┘  └────────┴────────┘     └────────┴────────┘
 (复制)   输入×本地列               本地输出列
```

### 具体数字例子

`input_size=4, output_size=6, tp_size=2`：

```
X = [batch, 4],  W = [4, 6]
切列后：
  rank 0: W_0 = W[:, 0:3] → Y_0 = X @ W_0 = [batch, 3]  (前 3 列)
  rank 1: W_1 = W[:, 3:6] → Y_1 = X @ W_1 = [batch, 3]  (后 3 列)

完整结果 Y = [Y_0 | Y_1] = [batch, 6]
```

因为 X 在两个 rank 上完全相同，`Y_0` 和 `Y_1` 拼起来正好是完整 Y，**零通信**。

### 三个要点

1. **权重按列切，输出按列切**：`output_size_per_partition = output_size // tp_size`（`linear.py:461`）。
2. **输入 X 是复制的，不切**：`forward` 里 `input_` 直接拿来乘，没有 split（对比 row 层有 `split_tensor_along_last_dim`）。
3. **输入必须复制的数学前提**：`Y` 的第 j 列 = `Σ_i X[:, i] * W[i, j]`，这个求和需要全量 X。如果 X 分片，`Σ_i` 就少加一半项，结果就错。

### weight_loader：加载时切权重

切分发生在**加载权重**时，不是 forward 时（`linear.py:542-559`）：

```python
def weight_loader(self, param, loaded_weight):
    output_dim = getattr(param, "output_dim", None)   # = 1（数学上的列）
    ...
    shard_size = param_data.shape[output_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

把完整 checkpoint 权重 `W [4,6]` 用 `narrow` 切出本 rank 的列段：rank 0 取 `[:, 0:3]`，rank 1 取 `[:, 3:6]`。

### bias 也按列切

`self.bias = Parameter(torch.empty(self.output_size_per_partition))`，切法和权重一样（`linear.py:500-509`）。

### gather_output 选项

默认 `gather_output=False`，输出保持分片。当下一层不想要分片输入时（如某些 MLA / flash-attn 实现要求全量 head），设 `True` 做 all-gather（`linear.py:578-582`）。

## 1.3 Row-parallel（行并行）

### 数学定义

`A` 按【行】切成 `[A_1; A_2; ...]`，`X` 按列切成 `[X_1, X_2, ...]`（`linear.py:1509-1515`）：

```
        [ Wo_0 ]                          [ attn_out_0 ]
  Wo =  [ Wo_1 ]        attn_out =        [ attn_out_1 ]      (按 head 分片)
        [ Wo_2 ]
```

### forward（`linear.py:1635-1661`）

```python
if self.input_is_parallel:
    input_parallel = input_           # 输入已分片，直接用
else:
    split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
    input_parallel = split_input[self.tp_rank].contiguous()

output_parallel = self.quant_method.apply(self, input_parallel, bias_)
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)   # ← 合并部分和
```

### 为什么 row 层输出必须 all-reduce

`o = attn_out @ Wo = attn_out_0 @ Wo_0 + attn_out_1 @ Wo_1`，每个 rank 只算出一个**部分和**。完整 o = 两部分之和，所以必须 all-reduce。

**为什么不能直接进 FFN**：FFN 的第一个投影 `gate_up_proj` 是 column-parallel，要求输入在每个 rank 上完全相同（复制态）。而 o_proj 未 reduce 的输出是分片的（rank 0 只有 head 0-3 的贡献），直接进 FFN 会"每一列少算一半 x"，结果错。

## 1.4 Attention 的 TP 切分

`llama.py:163-178`：

```python
self.qkv_proj = QKVParallelLinear(...)   # 继承 ColumnParallelLinear，切 head
self.o_proj = RowParallelLinear(...)     # 切 head，all-reduce
```

完整链：

```
x (复制)
  ├─ qkv_proj: ColumnParallel → 输出按 head 天然分片【零通信】
  ├─ attention: 每个 head 独立，本地算【零通信】
  └─ o_proj:   RowParallel    → 部分和 →【这里才 all-reduce】
```

**为什么中间零通信**：attention 的 softmax、对 k/v 的加权和本质是**按 head 独立**的。head 0-3 和 head 4-7 之间没有数据依赖，各自本地算完。

用 8 head、TP=2 走一遍：

```
                    rank 0                          rank 1
x               [完整 x]                        [完整 x]         ← 复制，不切
qkv_proj    Wq[:, 0:4d] → q 的 head 0-3     Wq[:, 4d:8d] → q 的 head 4-7
                 │                                 │
attention   head 0-3 本地算 softmax            head 4-7 本地算
                 │                                 │
            attn_out 的 head 0-3              attn_out 的 head 4-7
o_proj      Wo[0:4d, :] → 部分和 o_0          Wo[4d:8d, :] → 部分和 o_1
                 └──────────── all-reduce(o_0 + o_1) ────────────┘
                                最终 o (每卡都有完整 o)
```

核心洞察：**column 层的输出分片，恰好就是 row 层需要的输入分片**，所以中间零通信，通信被"推"到最后一个投影之后。

## 1.5 多 head 是怎么来的

**head 不是"投影算出来"的，是 `view` 免费拆出来的**。

1. 投影输出宽度 `q_size = num_heads * head_dim`（`llama.py:158`），head 信息"藏"在最后一维里。
2. `qkv.split([q_size, kv_size, kv_size], dim=-1)` 分出 q/k/v，仍是扁平 `[b, s, num_heads*head_dim]`（`llama.py:228`）。
3. `view` 把最后一维拆成两维（`attention.py:521-525`）：

```python
query = query.view(-1, self.num_heads, self.head_size)
output = output.view(-1, self.num_heads, self.head_size_v)
```

`view` 零拷贝，只是给同一块连续内存换 shape 解释。

4. 之后 `transpose` 成 `[b, num_heads, s, head_dim]`，让每个 head 独立做 `q @ k^T`。

**为什么 view 就能拆对**：`Wq` 的第 `[0:head_dim]` 列算 head 0，第 `[head_dim:2*head_dim]` 列算 head 1……列的分组就是 head 的分组，投影一结束 head 顺序就确定了。

**Wq 的形状**：

| | 数学形式 | vLLM 存储形式 |
|---|---|---|
| 单独 Wq | `[d, num_heads·head_dim]` | `[num_heads·head_dim, d]`（转置） |
| 打包 qkv_proj | `[d, (num_heads + 2·num_kv_heads)·head_dim]` | 同上转置 |

存储转置的原因：PyTorch `F.linear(x, W)` 内部算 `x @ Wᵀ`，weight 存成 `[out, in]` 省一次转置（`linear.py:184-193`，`output_dim=0`、`input_dim=1`）。

## 1.6 FFN 的 TP 切分

Llama 的 SwiGLU MLP（`llama.py:93-120`）：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,                 # d
    output_sizes=[intermediate_size] * 2,   # gate 和 up 各一份
)
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,                # d
)
self.act_fn = SiluAndMul()                  # SiLU(gate) ⊙ up

def forward(self, x):
    x, _ = self.gate_up_proj(x)   # [b, s, 2·intermediate]
    x = self.act_fn(x)            # 拆 gate/up → SiLU(gate)·up → [b, s, intermediate]
    x, _ = self.down_proj(x)      # [b, s, d]
    return x
```

SwiGLU 数学：`down(SiLU(x@W_gate) ⊙ (x@W_up))`。

### 具体数字例子：d=512, intermediate=2048, tp=2

**① gate_up_proj（column，切 intermediate 维）**

`W_gate_up = [512, 4096] = [W_gate(2048列) | W_up(2048列)]`，各按列切 tp=2 份：

```
rank 0: [gate 前1024列 | up 前1024列] = [512, 2048]
rank 1: [gate 后1024列 | up 后1024列] = [512, 2048]
```

每个 rank 的 intermediate 从 2048 缩到 1024。输入 x 复制，输出分片。

**② act_fn（逐元素，零通信）**

`[gate(1024) | up(1024)] → SiLU(gate) ⊙ up → [b, s, 1024]`，intermediate 维每个位置独立，本地算。

**③ down_proj（row，切 intermediate 维）**

`W_down = [2048, 512]` 按行切：rank 0 持 `[0:1024, :]`，rank 1 持 `[1024:2048, :]`。输入（act 输出）正好分片，输出部分和，all-reduce。

### FFN 与 Attention 切分同构

```
Attention:  qkv_proj (column, 切 head)   → attn (本地)   → o_proj (row)
FFN:        gate_up  (column, 切 intermediate) → act (本地) → down (row)

            ↑切一个"中间维度"        ↑逐元素/逐head独立    ↑跨中间维度求和
```

两者唯一差异是"切的那个中间维度是什么"：attention 切 head，FFN 切 intermediate。**本质都是：column 层沿某个内部独立维度切分，中间操作该维度无依赖（零通信），row 层再沿该维度求和（all-reduce）。**

---

# 第二部分：EP（Expert Parallelism）

## 2.1 核心：切"专家集合"，而非"专家内部"

MoE 的 FFN 比稠密 FFN 多了一个"专家"维度（N 个并排的 FFN + router）。切分有两个选择：

| | 沿 intermediate 维切 | 沿 expert 维切 |
|---|---|---|
| 就是 | **TP** | **EP** |
| 每个专家内部 | 切：每 rank 拿 `intermediate/tp` 片 | 不切：专家完整保留 |
| N 个专家 | 不切：每 rank 都算全部 N 个的一截 | 切：每 rank 拿 `N/ep` 个完整专家 |
| 通信 | all-reduce（down_proj 出口） | all-to-all（token 路由） |

用 8 专家、EP=2 画出来：

```
TP (切 intermediate)：                  EP (切 expert)：
rank 0: 专家0~7 各拿 [.., interm/2]    rank 0: 专家 0,1,2,3 完整
rank 1: 专家0~7 各拿 [.., interm/2]    rank 1: 专家 4,5,6,7 完整
        ↑ 每个专家被劈成两半                  ↑ 每个专家完整，只分数量
```

EP 语义即 `parallel.py:165` 那句 docstring：**"instead of tensor parallelism"**。

## 2.2 ep_size 怎么来：折叠 tp_size

`FusedMoEParallelConfig.make`（`config.py:1208-1255`）：

```python
use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1
          and vllm_parallel_config.enable_expert_parallel)

# flatten_tp_across_dp_and_pcp：把 dp×pcp×tp 摊平成一个大 tp
tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(...)

if not use_ep:
    return FusedMoEParallelConfig(tp_size=tp_size, ..., ep_size=1, ...)

# EP 开启：ep_size = 摊平后的 tp，而 tp_size 归 1
ep_size = tp_size
ep_rank = tp_rank
return FusedMoEParallelConfig(tp_size=1, ..., ep_size=ep_size, ...)
```

关键动作：**`ep_size = tp_size`，同时 `tp_size = 1`**。原本用于 TP 的 rank 改组为 EP 组。文档例子（`config.py:1198-1206`）：`TP=2, DP=2, EP=True` → 最终 `EP={4, rank}`、`TP={1, 0}`。

`flatten_tp_across_dp_and_pcp`（`config.py:1116-1125`）：

```python
flatten_tp_size = dp_size * pcp_size * tp_size
flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
```

## 2.3 专家映射：expert_map

`determine_expert_map`（`expert_map_manager.py:22-113`），两种摆放策略：

```python
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)

if expert_placement_strategy == "linear":
    start_idx = ep_rank * base_experts + min(ep_rank, remainder)
    expert_map[start_idx : start_idx + local_num_experts] = torch.arange(...)
elif expert_placement_strategy == "round_robin":
    local_log_experts = torch.arange(ep_rank, global_num_experts, ep_size, ...)
    expert_map[local_log_experts] = torch.arange(...)
```

产出 `expert_map`：`expert_map[global_id] = local_id`，**-1 表示不在本 rank**。

- **linear**：rank 0 → 专家 `[0..k)`，rank 1 → `[k..2k)`（连续块）
- **round_robin**：rank r → 专家 `r, r+ep, r+2·ep...`（交错，对 grouped-expert 更均衡）

## 2.4 token 路由：dispatch / combine

### 朴素实现（默认 allgather_reducescatter）

`AgRsAll2AllManager`（`all2all.py:44-150`）：

```python
def dispatch(self, hidden_states, topk_weights, topk_ids, ...):
    gathered_tensors = dist_group.all_gatherv(...)      # 全量收集 token
    ...

def combine(self, hidden_states, ...):
    hidden_states = dist_group.reduce_scatterv(...)     # 按 token 归属散回
    return hidden_states
```

朴素 "dispatch" 不是真 all-to-all，而是 **all-gather**（全量收集，本地只算自己的专家）；"combine" 用 **reduce-scatter** 归约散回。专用 kernel（DeepEP/MoRI/NIXL）做真 all-to-all。

### 完整 forward 流程

```
                      rank 0 (专家 0-3)          rank 1 (专家 4-7)
1. router 选专家:        token A 选中专家 2 和 5
2. dispatch (all-to-all):  A 发到 rank0(算专家2)  A 发到 rank1(算专家5)
3. 本地 GEMM:             expert 2 算 A            expert 5 算 A
4. combine (all-to-all):  结果发回 A 的"家"
5. 加权求和              weighted sum(专家2, 专家5)
```

代码（`naive_dp_ep.py:112-209`）：

```python
def prepare(...):   # = dispatch
    res = get_ep_group().dispatch(a1q, topk_weights, topk_ids, ...)
def finalize(...):  # = combine
    output.copy_(get_ep_group().combine(out, ...))
```

最终 `_maybe_reduce_final_output`（`moe_runner.py:459-493`）在 `ep_size > 1` 且输出未 reduce 时 all-reduce——因为 top_k 专家跨 rank 时，各 rank 算的是部分和。

## 2.5 为什么 EP 比 TP 省通信

### 先纠正：省的是通信，不是参数

TP 和 EP 单卡参数其实一样（都是 `N×d×intermediate/p`）。EP 真正省的是**通信**。

### 根本原因：token 是"广播"还是"路由"

```
TP 切专家【内部】：每个 token 的计算劈到 p 卡 → 归约拉上全部 p 卡 → all-reduce
EP 切专家【集合】：每个 token 只去 top_k 卡 → 只搬 top_k 次 → all-to-all
```

### 定量对比

- TP 通信 = 1 次 all-reduce，`∝ T × d × (p-1)/p`（每个 token 跨 p 卡归约）
- EP 通信 = 2 次 all-to-all，`∝ top_k × T × d`（每个 token 只搬 top_k 份）

比值：

```
TP 通信 / EP 通信 ≈ (p-1) / top_k
```

- p=8, top_k=1（Mixtral）：TP 是 EP 的 **7 倍**
- p=64, top_k=8（DeepSeek-V3）：TP 是 EP 的 **~8 倍**
- p=64, top_k=1：TP 是 EP 的 **63 倍**

### 一个具体例子：8 专家、8 卡、top_k=1、token 选中专家 3

**TP**：专家 3 切成 8 片散在 8 卡，token 复制到 8 卡各算 1/8，8 卡 all-reduce 拼回。通信牵扯全部 8 卡。

**EP**：专家 3 完整在卡 3，token 只发到卡 3（1 次 dispatch），卡 3 算完发回（1 次 combine）。通信只牵扯 2 卡。

**结论：TP 让 p 张卡为一个 token 忙活并通信，EP 只让 top_k 张卡忙活。归约半径从 p 缩小到 top_k。**

### 边界：什么时候 EP 不再占优

1. **top_k 大**：top_k 接近 p 时 `(p-1)/top_k → 1`，优势消失；top_k=N 时退化成比 all-reduce 更差。
2. **负载不均**：热门专家挤在一张卡会成热点，省下的通信以负载失衡形式还回来（需要 EPLB）。

---

# 第三部分：EPLB（Expert Parallel Load Balancing）

EPLB 是四步闭环：**统计负载 → 决策重排 → 搬移权重 → 提交新映射**。核心思想：**给热门专家加"副本"，把副本分散到空闲卡**。

## 3.1 三个专家概念（`eplb_state.py:6-27`）

- **Logical Expert（逻辑专家）**：模型里真正的专家，有一套权重。DeepSeek-R1 有 256 个。
- **Redundant Expert（冗余专家）**：为热门逻辑专家额外创建的副本。加 32 个 → 288 个。
- **Physical Expert（物理专家）**：实例化在某张卡上的专家，是逻辑专家的一个副本，可跨卡重排。

> DeepSeek-R1：256 逻辑 + 32 冗余 = 288 物理专家；32 EP rank，每卡 288/32 = 9 个本地物理专家。

**关键：一个逻辑专家可以有多个物理副本，router 把 token 路由到任意副本，实现负载分摊。**

## 3.2 负载统计：router 里顺手数 token

每次 forward，router 用 Triton kernel 把"每个物理专家被选中多少 token"累加进 `expert_load_view`（`base_router.py:85-93`）：

```python
tl.atomic_add(out_ptr + safe_physical_id, 1, mask=valid)   # 每命中一次 +1
```

`expert_load_pass`：shape `(num_moe_layers, num_physical_experts)`。累积进滑动窗口 `expert_load_window`，每 `step_interval` 步触发重排。

**优化**：`_should_record_current_step`（`eplb_state.py:661`）只在临近重排的最后 `window_size` 步记录，更早的记录会被窗口覆盖、白记。

## 3.3 决策：rebalance 算法（adapted from DeepSeek EPLB）

`rebalance_experts`（`default.py:275`）用两个贪心子算法 + 三层 hierarchical 结构。

### 子算法一：balanced_packing（装箱均衡，`default.py:23-73`）

把 n 个带权对象装进 m 个箱子，每个装 n/m 个、总权重尽量均衡。贪心：**每次把最重的对象放进当前最轻的箱子**。

### 子算法二：replicate_experts（副本分配，`default.py:76-101`）

把 num_log 专家复制成 num_phy 副本，最小化最大副本负载。贪心：**每次把冗余副本分配给 `weight/replica_count` 最大的专家**（最热且副本最少的优先加副本）。

### 三层 hierarchical（`default.py:104-189`）

节点内 NVLink 快、节点间慢，重排优先把流量留在节点内：

```
Step 1: 把 expert group 打包到 node    ← 跨节点流量最小化
Step 2: 在 node 内复制热门专家         ← 冗余副本只在节点内分配
Step 3: 把物理专家打包到 GPU          ← 每 GPU 负载均衡
```

输出 `phy2log`：`[layers, num_physical_experts]`，即新的"每个物理槽位装哪个逻辑专家"。

## 3.4 一个具体例子（正确版）

> 注意：重排**不增减专家**，物理槽位总数固定（`num_logical + num_redundant`），只重写"哪个槽装哪个逻辑专家"。

4 逻辑专家 + 2 冗余 = 6 物理槽位，3 卡、每卡 2 槽。

初始映射由 `build_initial_global_physical_to_logical_map`（`eplb_state.py:310-314`）：

```python
# [0,1,2,3] + [0,1] = [0,1,2,3,0,1]
physical_to_logical_map = [0,1,2,3,0,1]
```

```
物理槽位:  0    1    2    3    4    5
装的专家:  E0   E1   E2   E3   E0   E1
卡0: 槽0(E0), 槽1(E1)   → E0=1, E1=1
卡1: 槽2(E2), 槽3(E3)   → E2=1, E3=1
卡2: 槽4(E0), 槽5(E1)   → E0=2, E1=2   (冗余副本默认给前 2 个逻辑专家)
```

负载统计：E0 超热（60%），E1/E2/E3 冷。重排把冗余副本从冷的 E1 挪给热的 E0。

重排后 `physical_to_logical_map = [0, 2, 0, 3, 0, 1]`：

```
物理槽位:  0    1    2    3    4    5
装的专家:  E0   E2   E0   E3   E0   E1
卡0: 槽0(E0), 槽1(E2)   → E0 副本①
卡1: 槽2(E0), 槽3(E3)   → E0 副本②
卡2: 槽4(E0), 槽5(E1)   → E0 副本③
```

| | E0 副本数 | E0 每副本负载 | E0 分布 |
|---|---|---|---|
| 重排前 | 2 | 60% ÷ 2 = 30% | 卡0、卡2 |
| 重排后 | 3 | 60% ÷ 3 = 20% | 卡0、卡1、卡2 |

**"给 E0 加副本"= "把原本装冷专家 E2 的槽 2 改成装 E0 的第 3 份权重"**。槽位总数不变，副本是重新分配出来的。

## 3.5 权重搬移：P2P 搬专家

`rearrange_expert_weights_inplace`（`rebalance_execute.py:511`）→ `transfer_layer`（`427`）→ P2P send/recv，通过 `EplbCommunicator`（`eplb_communicator.py`）搬权重。后端可选 nixl（RDMA 零拷贝）/ torch_gloo（CPU 中转）/ pynccl。

**关键优化**：`preserve_intragpu_slots`（`default.py:191-272`）——重排后若某专家仍留在同一张卡，就让它保持原物理槽位，避免"原地不动"的专家被白搬一遍（权重搬移是 EPLB 最贵开销）。

## 3.6 提交：新映射生效

`_commit_eplb_maps`（`eplb_state.py:1282`）把新的三个 map 写回 model_state：

- `physical_to_logical_map`：物理槽位 → 逻辑专家
- `logical_to_physical_map`：逻辑专家 → 它的所有物理副本（`-1` 表示无）
- `logical_replica_count`：每个逻辑专家有几个副本

## 3.7 副本选择：哈希，不是 round robin

一个逻辑专家有多个物理副本时，router 怎么选？**用 token 位置做 Knuth 乘法哈希取模**（`base_router.py:43-62`）：

```python
# 1. 该逻辑专家有几个副本
replica_count = tl.load(logical_replica_count_ptr + safe_expert_id, ...)
replica_count = tl.maximum(replica_count, 1)

# 2. token 在 batch 里的位置
token_idx = (offs // num_active_experts).to(tl.int64)

# 3. Knuth 乘法哈希 → 取模选副本
KNUTH_MULTIPLIER = 2654435769          # floor(2^32 / φ)
hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
replica_idx = hashed % replica_count   # ← 哈希取模，不是 round robin

# 4. 查表得 physical id
map_index = safe_expert_id * map_slots + replica_idx
physical_id = tl.load(logical_to_physical_ptr + map_index, ...)
```

**为什么用哈希不用 round robin**：

| 需求 | round robin | 哈希 |
|---|---|---|
| 选副本需要状态吗 | 需要递增计数器（全局状态） | **无状态** |
| 并发/多流安全 | 计数器加锁/原子竞争 | 天然并行 |
| 跨 rank 一致性 | 要同步计数器 | 相同 token_idx 算相同结果 |
| CUDA graph 重放 | 计数器要重置 | 纯函数，结果一致 |
| 均匀性 | 严格轮流但难处理副本数不同 | Knuth 哈希天然均匀（雪崩效应） |

**重排后副本数变了，同一个哈希 kernel 自动把 token 散到新副本集合，路由代码一行都不用改**——因为 `logical_replica_count` 和 `logical_to_physical_map` 每次 forward 都重新读。

## 3.8 完整周期时间线

```
forward × N 步:
   router 选专家 → atomic_add 记录每个物理专家的 token 数
        │
        ▼ 每 step_interval 步触发一次
rearrange():
   1. 各 rank 负载 all-reduce 合并（拿到全局专家热度）
   2. rebalance_experts：贪心装箱 + 复制 + 分层，算出新 phy2log
   3. rearrange_expert_weights_inplace：P2P 搬权重
   4. _commit_eplb_maps：新映射生效
        │
        ▼ 回到 forward，用新映射路由
```

---

# 附录：关键代码索引

| 概念 | 位置 |
|---|---|
| `enable_expert_parallel` / `all2all_backend` | `vllm/config/parallel.py:165` / `:188` |
| `EPLBConfig` | `vllm/config/parallel.py:58` |
| `ColumnParallelLinear` | `vllm/model_executor/layers/linear.py:401` |
| `MergedColumnParallelLinear` | `vllm/model_executor/layers/linear.py:639` |
| `QKVParallelLinear` | `vllm/model_executor/layers/linear.py:965` |
| `RowParallelLinear` | `vllm/model_executor/layers/linear.py:1504` |
| `LlamaMLP` | `vllm/model_executor/models/llama.py:80` |
| `LlamaAttention`（q_size/forward） | `vllm/model_executor/models/llama.py:123` / `:222` |
| q/k/v 拆 head 的 view | `vllm/model_executor/layers/attention/attention.py:521` |
| `FusedMoEParallelConfig.make`（折叠 tp→ep） | `vllm/model_executor/layers/fused_moe/config.py:1128` |
| `determine_expert_map` | `vllm/model_executor/layers/fused_moe/expert_map_manager.py:22` |
| `AgRsAll2AllManager`（dispatch/combine） | `vllm/distributed/device_communicators/all2all.py:44` |
| `_maybe_reduce_final_output` | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:459` |
| EPLB 三个专家概念 | `vllm/distributed/eplb/eplb_state.py:6` |
| 负载记录 atomic_add | `vllm/model_executor/layers/fused_moe/router/base_router.py:93` |
| 哈希选副本 | `vllm/model_executor/layers/fused_moe/router/base_router.py:43` |
| balanced_packing | `vllm/distributed/eplb/policy/default.py:23` |
| replicate_experts | `vllm/distributed/eplb/policy/default.py:76` |
| hierarchical 重排 | `vllm/distributed/eplb/policy/default.py:104` |
| preserve_intragpu_slots | `vllm/distributed/eplb/policy/default.py:191` |
| 权重搬移 | `vllm/distributed/eplb/rebalance_execute.py:511` |
