# DSA Indexer、Sparse KV Offload 与 Staged SFA Graph 总体设计

## 1. 文档范围

本文是 Ascend 上 DSA（DeepSeek Sparse Attention）相关能力的统一设计入口，覆盖：

- Lightning Indexer 的计算和 cache 生命周期；
- MLA latent KV 的 LMCache offload 与按 Top-K 稀疏加载；
- resident sparse scratch cache 和 compact remap；
- vLLM Ascend、vLLM、LMCache、LMCache Ascend 的接口边界；
- Staged SFA PIECEWISE graph 的切分、资源、并行和生命周期设计；
- 正确性、性能、故障处理和发布验收要求。

状态：默认关闭、按能力矩阵显式启用。只有通过本文验收矩阵的配置才可以视为受支持；
其余配置必须在模型执行和 cache 修改前选择安全原生路径、重新计算或明确拒绝。

## 2. 背景与目标

DSA 将一次 attention 分成两步：

1. Lightning Indexer 使用较小的 index key 对全部历史 token 进行打分，选择 Top-K。
2. Sparse Flash Attention 只读取 Top-K token 对应的 MLA latent KV。

它天然提供计算稀疏性，但原始实现仍可能让完整 latent KV 常驻 NPU。随着上下文长度和
并发数增长，latent KV 会成为显存主要开销。本设计在不改变 Indexer 和 SFA 数学语义的
前提下增加内存稀疏性：完整历史 latent 可以存入 LMCache，decode 时只把当前 Top-K
需要的数据加载到固定 scratch。

主要目标：

- Indexer 对完整历史可见，Top-K 结果与原生路径一致；
- 只物化本次 SFA 需要的 latent KV，降低 NPU 常驻显存；
- cache、scratch、graph workspace 和 connector 内存全部进入统一显存预算；
- LMCache 是唯一外部 KV 生命周期边界，保存完成前不释放唯一有效数据；
- eager、标准 PIECEWISE 和 staged SFA graph 有明确且互不混淆的支持范围；
- TP/DP、MTP、取消、抢占和异常不会产生跨请求数据污染或 stale graph 地址。

非目标：

- 不通过关闭正确性检查换取 graph replay；
- 不把 Indexer key 当成可按 Top-K 稀疏加载的数据；
- 不允许在 graph replay 中执行隐式 host 同步、动态分配或 Python 状态推进；
- 当前不把 layerwise prefill P 节点的双 bank callback 捕获进 staged SFA graph；
- 未经单独验收的 LoRA、CP、PP/多 virtual engine、C8、MLAPO 等组合不自动获得支持。

## 3. DSA Cache 数据模型

### 3.1 两类 Cache

| Cache | 典型内容 | 生命周期 | 设计要求 |
|---|---|---|---|
| Indexer cache | `kv_cache[2]`，C8 时含 `kv_cache[3]` scale | 覆盖完整上下文并常驻 NPU | Indexer 每步需要对全部历史打分，不能仅加载 Top-K |
| MLA latent KV | `kv_cache[0] = k_nope`、`kv_cache[1] = k_pe` | prefill 生成，decode 按 Top-K 使用 | 可持久化到 LMCache，并按需加载到 sparse scratch |

Indexer cache 和 latent cache 必须是不同的 cache group、block table 和 slot mapping。
启用 two-group/unbundle 时：

- latent group 使用自己的 `block_table` / `slot_mapping`；
- indexer group 使用 `indexer_block_table` / `indexer_slot_mapping`；
- SFA forward 在调用算子前按契约重组 cache tuple；
- 两组的 token 范围必须一致，但 block 数、每 token 大小和 resident 策略可以不同。

### 3.2 Cache 不变量

- Indexer key 覆盖 `[0, sequence_length)`，不得因 latent offload 缩短可见历史。
- `topk_indices` 使用绝对序列位置，compact scratch 使用局部位置，两者必须显式 remap。
- `-1` padding、inactive row 和超出有效长度的位置不能访问真实 cache。
- 当前 token 的 latent/index key 写入必须先于本层选择和 attention 消费。
- LMCache 报告完整持久化之前，不能释放 NPU 上唯一有效的 latent 数据。
- request、KV group、layer、generation 和 cache epoch 必须共同确定数据身份。

## 4. Lightning Indexer

### 4.1 作用

Lightning Indexer 是 DSA 的轻量候选选择器。它为每个 query token 对完整历史 key
打分，输出最多 `index_topk` 个绝对 token 位置；GLM-5.1 当前通常为 2048。随后 SFA
只对这些位置执行较重的 attention。

Indexer 不是 LMCache 查询器。它只负责：

- 生成当前 token 的 index key；
- 把 key 写入常驻 indexer cache；
- 生成 query 和权重；
- 调用 lightning indexer kernel 得到 `topk_indices`。

### 4.2 前处理：生成 Key

入口为 `AscendSFAImpl.indexer_select_pre_process`，逻辑为：

```text
x
 -> wk
 -> RMSNorm
 -> RoPE
 -> 可选 Hadamard + dynamic quant(C8)
 -> k_li / k_li_scale
```

生成结果通过 indexer slot mapping 写入 resident paged cache。普通模式使用
`slot_mapping`，two-group 模式使用 `indexer_slot_mapping`。

### 4.3 后处理：选择 Top-K

入口为 `AscendSFAImpl.indexer_select_post_process`，逻辑为：

```text
x -> weights_proj ------------------------- weights
q_c -> wq_b -> RoPE -> 可选 C8 quant ----- q_li

(q_li, resident index key, weights,
 actual_seq_lengths_query/key, indexer block table)
 -> lightning indexer
 -> topk_indices
```

`topk_indices` 的主要契约：

- dtype 为整数；
- 形状以实际算子契约为准，常见为 `[query_tokens, 1, index_topk]`；
- 每一行是绝对 sequence position；
- 有效位置前置，剩余位置使用 `-1`；
- padded query row 必须有独立有效性掩码，不能读取其他请求数据。

### 4.4 Kernel 选择

- GLM-5.1 默认使用 `torch_npu.npu_lightning_indexer`；
- sparse C8 使用量化 kernel，并额外读取 key scale；
- 其他支持模型可以使用 `_C_ascend` 对应算子；
- kernel 类型必须进入 graph structural key 和启动能力检查，不能在 replay 期间切换。

## 5. Sparse Latent KV 与 Resident Scratch

### 5.1 Scratch 模型

decode 使用固定容量、连续、预分配的 sparse scratch，而不是临时 mini pager。scratch
按最大并发请求数、`index_topk`、latent 维度、dtype 和 MTP 宽度核算，地址在 graph
capture/replay 生命周期内保持稳定。

scratch 逻辑上包含：

```text
[ LMCache 选中前缀 | NPU resident/current decode token | padding ]
```

实际布局可以复用同一连续空间，但必须通过 counts、target slots 和 row masks 明确每段
范围，不能依靠未初始化内容或 Python list 长度推断。

### 5.2 Gather 与 Remap

每层 decode 的基本步骤：

1. Indexer 输出绝对 `topk_indices`。
2. 根据 committed/remap frontier 将位置分为 LMCache source 和 resident source。
3. 对多 query/MTP 行执行 union、去重和排序，形成最小加载集合。
4. LMCache 把选中的 latent 写入 scratch 指定 slot。
5. resident 部分复制到其目标 slot。
6. 构造 compact `sparse_indices`、scratch block table、每行 count 和 mask。
7. SFA 读取 scratch 执行 attention。

remap 必须满足：

- 绝对位置到 compact slot 一一对应；
- union 不改变每个 query 自身的选择顺序语义；
- intersection、空洞复用和 resident hit 不产生重复加载；
- `-1`、inactive row 和不足 Top-K 的尾部保持 padding；
- block 边界 127/128/129 及 `index_topk±1` 正确；
- 同一 request 的多个 MTP row 可以共享 source，但 target row 必须保持隔离。

### 5.3 Resident Sparse Cache

resident sparse cache 可以缓存近期已经物化的 latent 和 planning 结果，减少重复加载和
remap 开销，但必须遵守：

- cache key 包含 request/generation、layer、绝对 token、dtype/layout 和 cache epoch；
- request 完成、抢占、重新计算、cache recreation 后失效；
- miss 或部分覆盖不能伪装成完整 hit；
- capacity 有上限并计入显存预算；
- 复用不能跳过 producer/load/consumer event 依赖。

## 6. Prefill 与 Decode 数据流

### 6.1 Prefill

```text
hidden states
 -> exec_kv 生成 k_nope/k_pe
 -> 生成并写入 indexer key
 -> prefill attention
 -> latent/indexer 按 connector 协议持久化
```

普通 D 节点保持原有 full-resident/cache 路径。PD 分离的 P 节点可以使用独立的
layerwise prefill 双 bank 方案，但当前不得与 staged SFA graph 同时启用：P 节点需要
每层 runtime load/save callback，staged cross-layer graph 的 model-boundary save 无法保证
双 bank 数据尚未被覆盖。

P 节点详细需求见 vLLM 仓库：
`docs/requirements/dsa_kv_cache_indexer_bundle_pool_requirements.md`。

### 6.2 Decode

```text
1. 写当前 token latent 与 index key
2. Lightning Indexer 对完整历史选择 Top-K
3. 读取 LMCache committed/remap frontier
4. 构造 union/remap/target slots
5. selective load latent 到 scratch
6. 合并 resident/current token latent
7. Sparse Flash Attention 读取 compact scratch
8. 完成当前层尾部计算
9. 在请求/窗口边界按协议提交 decode save
```

decode offload window、异步持久化和 `committed_end` 推进由 LMCache 定义；SFA 只能使用
scheduler 已发布的 frontier，不得把“任务已发出”或“D2H 已完成”当成可释放/可读取。

## 7. LMCache 接口边界

vLLM Ascend 负责：

- 生成 request/layer/group 对应的稳定 metadata；
- 提供 Top-K selected tokens、scratch slot mapping 和 payload event；
- 在消费 scratch 前等待 load completion；
- 仅在数据已稳定后发起 save；
- 把 native operator 所需的 tensor contract 固定下来。

LMCache/LMCache Ascend 负责：

- `start_load_kv` 和逐层 `wait_for_layer_load` 生命周期；
- 按 selected tokens 将 latent 写入指定 scratch slot；
- layerwise latent/indexer store；
- pinned host MemoryObj、NPU stream/event 和后端 persistence 生命周期；
- cache hit/miss、partial coverage、完成、重试和 abort；
- 只有全部必需 group/owner 完成后才发布 committed frontier。

接口必须保持：

- selected token 顺序、slot mapping 和实际写入位置一致；
- load 的 producer event 在读取 Graph A 输出前生效；
- source/destination 地址在 native kernel 完成前保持有效；
- connector cursor 每层恰好推进一次；
- 异常后不允许继续进入依赖不完整数据的下一 graph island。

## 8. 标准 PIECEWISE 与 Staged SFA Graph

### 8.1 两条路径

标准 PIECEWISE 路径把 `torch.ops.vllm.mla_forward` 作为 opaque split，Python SFA
body 和 connector callback 在 runtime 执行。这是 layerwise P 节点所依赖的路径。

Staged SFA 路径把 SFA 拆成 graph-safe 的 Graph A/Graph B，并让
`vllm::sfa_lmcache_retrieve` 成为唯一 eager split。外层 vLLM PIECEWISE compiler 捕获
相邻 split 之间的跨层 graph island，不在 `sfa_v1.py` 内再嵌套 per-layer graph。

### 8.2 目标调度

```text
静态预检和全 rank admission
  -> bootstrap：准备 layer 0 index group
  -> Island 0：Graph A(0)
  -> Split 0：selective latent load(0) + index load(1)
  -> Island 1：Graph B(0) + layer tail(0) + Graph A(1)
  -> Split 1
  -> ...
  -> Island N：Graph B(N-1) + layer tail(N-1)
  -> connector finalize 与 deferred save
```

对 N 个本地 SFA layer，retrieve-only split 理论上形成 N+1 个 outer graph island。
profiler 必须证明不存在旧的 Graph-A/Graph-B nested wrapper。

### 8.3 Graph A 契约

Graph A 包含：

- Q/K preprocessing；
- 当前 token latent/index 写入；
- Lightning Indexer 和 Top-K；
- inactive row mask；
- sparse union、count、target slot 和 remap planning；
- 记录供 LMCache load stream 等待的 producer event。

Graph A 的输出必须全部是显式 tensor，不得依赖 Python 对象身份、动态列表或隐式
ForwardContext 状态。输出地址由 runner-owned buffer arena 管理。

### 8.4 Retrieve Split 契约

retrieve split 是唯一 eager host 边界，负责：

- 读取 Graph A 输出；
- 只向 LMCache 发送真实请求行，不发送 padded/inactive row；
- 加载当前层 latent group；
- 准备下一层 index group；
- 对成功、miss、短传输和异常给出确定结果；
- 在允许 Graph B 消费前闭合 producer/load/consumer event 链。

在 Graph A 已产生 Top-K 后发生的 load 失败属于 post-mutation 失败，不能静默 fallback
到可能缺少 latent 的路径；必须使用已证明安全的 recompute 方案或 fail-stop。

### 8.5 Graph B 契约

Graph B 包含：

- scratch/resident source 合并；
- compact sparse indices；
- Sparse Flash Attention；
- attention output projection 前后的 graph-safe 算子。

Graph B 只消费对应 split 已确认完成的 scratch slot。上一 invocation 未完成时，arena
slot、connector resource 和动态 metadata 不得复用。

## 9. Structural Key、Registry 与稳定内存

graph structural key 至少包含：

- model/operator/CANN/torch-npu fingerprint；
- dtype、quant/indexer kernel 模式；
- TP/DP/PP/virtual-engine 身份；
- local SFA layer 集合；
- exact/padded request capacity；
- query token capacity和 MTP candidate width；
- `index_topk`、scratch layout 和 two-group 模式；
- cache epoch、weights epoch 和 graph ABI version。

runner 负责：

- `StagedSFAExecutionPlan`：记录 key、island、layer 和 callback 顺序；
- `StagedSFABufferArena`：持有 bridge、mask、counts、top-k、scratch 和 event；
- capture registry：只在启动/显式 recapture 阶段建立；
- graph admission：模型执行前选择 staged/native/recompute/fatal；
- lifecycle invalidation：cache/weight/engine 变化时拒绝或重建。

release replay 不得重新扫描 tensor 签名、发现 connector capability 或创建新 stream/event。

## 10. 支持与拒绝策略

### 10.1 当前基础支持范围

- 目标 DSA 模型；
- fp16/bf16；
- two-group/unbundle LMCache；
- 标准 PIECEWISE；
- staged exact Q=1 的已配置 request count；
- TP 的已验收规模；
- 一个 DP replica、一个 virtual engine；
- mixed prefill/decode 整步走安全 native 路径。

### 10.2 分阶段扩展

1. Exact Q=1：每个配置 key 完成 capture/replay 与 trace 验收。
2. Padded Q=1：固定 bucket，inactive row 不接触真实 cache。
3. Fixed-width MTP：按 request 合并 Top-K，保留每个 token row 的目标隔离。
4. DP/PP/virtual engine：加入 namespace、empty rank 和 in-flight slot。
5. LoRA、C8、MLAPO、CP、prefetch 等模式逐项验收。

### 10.3 必须拒绝或禁用 replay 的情况

- layerwise prefill P 节点与 staged SFA graph 同时启用；
- FULL graph 中仍存在 host-driven LMCache callback；
- 未配置/未封存的 structural key；
- scratch/bridge 地址或 cache epoch 已变化；
- mixed phase 尚未通过对应 staged 设计；
- PP/多 virtual engine 尚无隔离契约；
- connector 缺少 required group、payload event 或稳定 slot binding；
- 任一 rank 无法对 staged route 达成相同静态判断。

## 11. 并行与故障语义

### 11.1 Tensor Parallel

- route decision 只依赖广播 scheduler 输入和启动静态状态；
- 每个 key、layer、rank 的 graph entry 必须在 readiness 前封存；
- connector owner 数和 completion 聚合必须与 TP 配置一致；
- 一 rank post-mutation 失败时采用有限超时和 fail-stop，不允许其他 rank 继续 collective；
- 不在每层热路径额外增加无证据的 HCCL verdict。

### 11.2 Data Parallel

- 所有 DP rank 对 bucket/capacity 路由达成一致；
- empty rank 使用 connector-free dummy execution；
- padding row 永远不能发送 LMCache 请求或修改真实 slot；
- 外部 launcher DP 在 rank/cache ownership 未定义前拒绝。

### 11.3 PP 与 Virtual Engine

- graph registry、cache epoch、arena 和 connector cursor 必须按 stage/engine 隔离；
- 未实现隔离前显式拒绝；
- 支持并发 invocation 时必须提供多个 in-flight slot，否则强制串行并测试。

### 11.4 取消、抢占和重新计算

- connector callback、cursor 和 store/load future 恰好清理一次；
- pending NPU load/save 完成前 source/destination 不释放；
- request generation 变化后旧 completion 和 graph metadata 失效；
- preemption/recompute route 在任何 cache mutation 前确定；
- 已进入 Graph A 后不能回退到缺少相同 latent/indexer 数据的 native 路径。

## 12. 资源模型

显存预算必须在 KV block sizing 之前包含：

- indexer resident cache；
- latent full/compact resident 部分；
- sparse scratch 和 MTP union scratch；
- bridge、mask、counts、slot mapping 和 block table；
- graph workspace、ACL executable 和 capture high-water；
- connector staging buffer；
- stream/event 和安全余量。

启动时采用临时 cache/graph profile 获取真实 high-water，完成后清理临时 graph 和 cache，
再进行最终 KV sizing。真实 capture 失败使用 vLLM 现有 allocator/capture 错误路径，不以
乐观估算继续启动。

资源必须满足：

- graph key 和 entry 数有上限；
- stream/event 数与本地 layer/topology 相符；
- arena slot 在 consumer 完成前不复用；
- capture 后不允许无界 recapture；
- `max_num_seqs`、MTP width、Top-K 和 dtype 变化均反映到预算。

## 13. 可观测性

启动日志至少包含：

- staged 功能开关和 admission 结果；
- structural key、capture size、local SFA layer；
- graph/island 数、arena/workspace 预留；
- connector capability、two-group 和 scratch layout；
- 被拒绝配置的明确原因。

运行时按需记录：

- 实际选择的 key、staged/native/recompute/fatal route；
- Graph A、retrieve split、Graph B 次数；
- request row 到 LMCache row 的映射；
- load/save/frontier、cache hit/miss 和异常层；
- graph quarantine、cache epoch 和 replay count。

profile 必须能够证明：

- N+1 cross-layer island；
- 每层只有一个 eager retrieve split；
- Graph A producer event 先于 load；
- load completion 先于 Graph B；
- 热路径无 `.item()`、全局 synchronize、tensor/stream/event 动态创建；
- 没有静默 recapture 或旧 nested wrapper。

## 14. 验证方案

### 14.1 单元与 Mock 集成

- indexer preprocess/postprocess 的 shape、dtype、C8 与 kernel 路由；
- single/two-group slot mapping 和 block table；
- union、去重、空洞、padding、resident hit 和 compact remap；
- exact/padded/MTP structural key 隔离；
- arena 地址稳定和 cache epoch 失效；
- connector callback 次数、cursor、异常和取消；
- store-before-free、短传输、partial group 和 stale completion；
- mixed phase 只查询 decode row frontier；
- graph entry 数、resource formula 和 unsupported-mode rejection。

### 14.2 NPU 正确性矩阵

staged 路径同时对比 resident eager reference 和有效的 native compact-scratch LMCache
路径，检查：

- top-k indices；
- 当前 token latent/index 写入；
- scratch 内容和 compact indices；
- 每层 attention output、logits 和 greedy token；
- request reorder、arrival/finish/cancel/preempt/recompute；
- block 边界 127/128/129；
- decode window 255/256/257；
- `index_topk-1/index_topk/index_topk+1`；
- all-warm、cold load、partial frontier、true miss 和 timeout；
- TP1/2/8，再扩展 DP/PP/VE；
- MTP width 和部分接受模式。

### 14.3 性能与可靠性

- 报告 TTFT、TPOT、吞吐的 p50/p90/p99，而不是单次峰值；
- 分离 indexer、SFA、projection、HCOM、LMCache load、event fence 和 host gap；
- 比较不同 batch/key、layer 数、TP 规模和上下文长度；
- decode-window 边界单独统计，确认 save-before-free；
- 长生成和高 churn soak 中无 parity、cursor、stale pointer 或跨请求污染；
- steady HBM、stream/event 和 graph entry 保持有界；
- kill switch 能在后续 step 禁用问题 key，不能继续执行已发生数据破坏的 in-flight step。

## 15. 发布门槛

一个 structural key 只有同时满足以下条件才可以进入生产 allowlist：

1. 模型执行前完成确定的 route admission，所有 rank 结果一致。
2. graph 输入输出、scratch、event 和 connector binding 地址稳定且有明确 owner。
3. Indexer、Top-K、latent load、SFA 和 save 生命周期与 eager reference 数值一致。
4. cache miss、短传输、失败、取消和重算不存在 unsafe fallback。
5. capture/replay 在全部参与 rank 和边界 case 上通过。
6. padded/MTP row 对 cache、scratch 和 LMCache payload 完全隔离。
7. graph、workspace、arena、scratch 和 connector 资源在 KV sizing 前有界。
8. 生命周期变化可以安全失效/重建，或在变化前明确拒绝。
9. profiler 证明目标 island/event 结构且不存在热路径全局同步和静默 recapture。
10. 签署工作负载达到性能目标，指标、quarantine 和 kill switch 可观测。

未满足全部门槛时，功能必须保持 experimental/default-off，并输出精确的 admission、
fallback、recompute 或 fatal 原因。
