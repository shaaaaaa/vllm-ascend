# Layerwise prefill cache 定向迁移

目标：四仓 `lmy_merge_prefill_layerwise_cache`。
来源：`glm52-model-port..glm52-model-port-prefill-offload` 中与
layerwise prefill cache 有关的改动，按代码块迁移，未整分支 merge。

## 迁入范围

| 仓库 | 来源提交 | 迁移内容 |
| --- | --- | --- |
| vllm | 7d41ea73e（部分） | P-node 全局共享 slab、双 bank 子分配器、显式分配类型、调度和 worker 的 bank 元数据、分配失败回滚 |
| vllm-ascend | a258c998（部分）、d92bce9a（P 侧部分） | 独立 latent/indexer bank、MTP bank 元数据、attention 前等待与 HCOM 前后分离的传输回调、forward 内元数据复用 |
| LMCache | 7641b858（部分）、d3302973、1b8bfda5 | 双 bank 请求映射、分组游标、延迟加载/保存协议、源对象生命周期、整 chunk 冷命中 release frontier 修复 |
| LMCache-Ascend | 16c941f、7c30bd3 | NPU 传输事件和 bank 复用保护、D2H 完成后发布、deferred store 显式传递 KV group |

目标分支已有 MTP unpadding 保留 indexer 地址的修复，保留该实现并补充回归，
不重复覆盖。也保留目标分支已有的 RemoteFill、checkpoint/preemption、
prepared transfer 快路径、性能日志开关，以及 latent/indexer 两组加载完整性验证。

适配时额外修正 P-node 并发容量统计：子池容量是 parent bundles × latent
物理层数，而每请求需要两份 bank，不能仍按单层 parent 容量统计。

## 没有迁入的内容

- 源分支的纯 D-node 角色开关及其专用参数校验。
- D-node sparse 驻留预算、admission/look-up 超时及失败请求传播改动；
  BalanceScheduler 的 compact-external-load 改动。
- 通用 RoPE cache 扩容及模型加载后的 RoPE 长度校验。
- `glm52_prefill_profile.py`、对应 profile 指南和测试。
- baseline launcher 的提交：目标分支已有脚本，本次未改动。
- decode-full-graph 分支的其他优化；本次没有从该分支取代码。

## 使用边界

该功能仅在 `VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true` 的 P 节点使用。
需要 DSA unbundle/two-groups/shared-pool、两组都能持久化的 layerwise
producer-capable LMCache Ascend connector。P 使用 `SHRINK_LATENT=0`；
D 保持其原有配置，不需要新增上述纯 D-node 开关。

P 节点可以关闭 MTP；启用时 `num_speculative_tokens` 必须为 1，启动时会拒绝
大于 1 的配置，避免重复执行 MTP 层推进同一套逐层加载/保存游标。
关闭 P-node 开关时不受此限制。

生产 P/D 都可使用 `--speculative-config
'{"num_speculative_tokens":1,"method":"deepseek_mtp"}'`，不依赖测试工具开关。
保存范围包含已注册的 draft KV 层，latent/indexer 使用各自的物理层序号；
target forward 结束后保留 connector 元数据，MTP forward 完成后才结束保存。

## 热路径开销收敛（2026-09-17）

| 项目 | 处理方式及剩余开销 |
| --- | --- |
| 固定配置 | 初始化时解析 P-node/indexer 开关；逐层不再读取环境变量 |
| 层名查找 | KV cache 注册/刷新时建两组序号表；逐层字典查找，不扫描列表 |
| 传输映射 | 每个 forward、每请求、每组的两个 bank 各切片/校验一次；连续区间用 view，不调用 cat；非连续区间各拼接一次 |
| 异步持久化 | 完成/取消请求只等自身以及复用前缀的未完成 put；不排空其他请求队列 |
| 新增依赖记录 | 按已有 chunk key 记录尚未完成的 futures，不重新 hash prompt、不复制 KV；完成后清理 |
| 必要等待 | 保留本地 bank 复用事件、队列满时限流、最终远端持久化及 RemoteFill 完成；退出时等待全部任务 |

未新增逐层日志、TP CPU all-reduce、KV 内容扫描或设备同步。映射复用仅作用于
deferred layerwise prefill；正常 decode 不跨 forward 复用本次新增的映射缓存。
这减少了可确认的重复工作，不代表已经测得生产端的性能提升。

保留源功能的限制：PP/PCP/DCP 均为 1；不能启用跳过逐层回调的 FULL/staged
SFA 图，也不能使用不兼容的 fused matmul-allreduce。可使用普通 PIECEWISE
边界。关闭 P-node 开关时仍走原有驻留和传输路径。

## 本地验证（2026-09-16）

在 Windows / CPU 上运行生产 Python 方法；NPU kernel、stream/event 使用替身
的测试不代表真实 NPU、HCCL 或跨机 Mooncake 验证。

| 测试组 | 通过 |
| --- | ---: |
| vllm 双 bank 分配、全局 slab/容量、scheduler bootstrap 定向测试 | 34 |
| Ascend forward 内 metadata 复用、P-role 初始化、MTP unpadding | 34 |
| LMCache standalone（含 checkpoint、GLM52、加载协议、release frontier） | 106 |
| LMCache adapter 的 P-node/双 bank/transfer-window 定向测试 | 28 |
| LMCache layerwise retrieve/deferred store 测试 | 12 |
| LMCache shared dense 生命周期/失败保护定向测试 | 16 |
| LMCache-Ascend deferred store 与 prepared/fast transfer CPU 合约 | 6 |

合计 236 个定向用例通过。四仓变更文件通过 Ruff F821/E9 和
`git diff --check`。完整 pre-commit/Ascend `format.sh ci` 未运行成功：
本地缺少 pre-commit。

扩大测试并非全绿，不能据此声称全仓回归通过：

- LMCache shared-CPU 全文件仍有 41 个失败，主要是旧测试的配置/模拟对象字段
  不完整及旧预期。用目标分支迁移前的 cache engine 在同一环境复跑，这些
  用例也失败；本次未改这些无关生产路径。
- LMCache worker/scheduler/save 三文件回归有 21 个已有失败（perf stats、
  cold-resume 模拟对象、旧 indexer slot 预期等），迁移前 adapter 复跑也失败。
- vllm KV utils 全文件中的 10 个用例被本地模型 architecture inspection /
  原生依赖环境阻断；上述 34 个定向用例不依赖此环境。

尚未验证：真实 NPU 算子正确性、TP/DP 多卡与跨机 PD 端到端、完整模型
接受率及性能。服务器上四仓需配套更新，不能只更新其中一个。

后续新增的本地串行验证工具见 [单机三次验证](layerwise_prefill_check.md)：
完整模型 baseline → layerwise prefill P → 加载 P 输出的 D，保存逐层 KV
并汇总数值差异；这不是上述未迁入的旧 profiling 脚本。
