# Prefill bank overlap statistics

四仓 `dev-lmy` 分别基于 fetch 后的 `origin/dsa-two-groups`（vllm）、
`origin/sparse`（vllm-ascend）及 `origin/dev-qzy`（两个 LMCache 仓）。
请配套更新四仓。

在 **P 节点启动服务之前**设置：

```bash
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
export VLLM_ASCEND_PREFILL_OVERLAP_STATS=true
```

bank 数默认是 4，可以在 P 节点启动前设置
`export VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS=8`。
合法值为正整数，非敏感配置；
未开启 P-node 开关时忽略此配置，D 节点保持原路径。

设置为 8 时，每个实际 prefill chunk 的计算开始前，先异步提交各 KV 组的
前 8 层历史 KV 加载；组内层数不足 8 时只提交存在的层。latent 和 indexer
各自按组内层序轮转，indexer 的序号不是模型层号。冷请求第一段没有历史
KV 时无需加载。第 N 层在计算流上等待自己的加载 event；最后一次读取 bank
后记录计算 event，同一 bank 的 FIFO 依次执行 `save N → load N+8`。
这些传输与其他 bank 上的层计算重叠，不为增加 bank 数新增主线程同步。

物理 block 分配、block table、跨 chunk 偏移和调度容量估算均使用相同 bank 数。
总 KV 池仍在启动时预分配，增加 bank 数不会自动扩容；同一长度请求从 2 改为
8 bank，其轮转 KV block 用量约为原来的 4 倍，可容纳的并发或上下文容量相应下降。
传输总量不会因预取窗口变大而减少，是否加速需要实机测量。

启动时按 bank 数校验至少一个 `max-model-len` 请求能否放入 NPU KV 池，
不会强制为 `max-num-seqs` 个满长请求同时保留容量。最终启动日志中的
`Maximum concurrency for ... tokens per request` 表示按满长计算的池容量；
`max-num-seqs` 只是运行请求数量上限。运行时仍按实际需求检查 block，
不足则等待或抢占。LMCache CPU 缓存另行配置，不受这项 NPU 容量校验保障。

第二个变量默认 `false`，合法值为 `true` / `false`，不是敏感配置。
它必须与已有 P-node 开关同时开启才生效。D 节点保持原配置。
统计不要求开启 tensor dump、profiler 或已有的 start-load 调试开关，
也不会修改图、调度、缓存或 DMA 的依赖关系。

## 统计口径

每次实际 chunked prefill 的整个模型 batch 统计一次，支持同批多个 request，
各请求长度、已处理前缀可以不同。日志保留所有请求的真实 token 区间，
不把整批耗时分摊或重复累计给各个请求。
同时检查 connector metadata 和 attention metadata，避免漏掉请求后错误归属计时。
空跑、无可确认请求范围的 batch 和 decode 不统计。

token 区间来自调度器这次实际计算的 `[start, end)`，按 request 自己的逻辑位置计数。
例如每批 4096 token，图上依次显示 `0-4095`、`4096-8191` 等；
末段不足 4096、已有前缀命中、调度器实际切小 batch 时保留真实范围，
不会把它们伪装成完整 4096-token 段。服务端 JSON 的 `token_end` 为开区间终点。

定义：

- `start(N)`：模型第 N 层最后一次 bank 等待结束的 NPU event。
  若该层有 latent 和 indexer 两个等待，取后者的结束点，保证它们都在该起点之前。
  这是 bank 就绪后开始计算的边界，不是 Python 进入 decoder 的主机时间。
- 按实际 `(KV group, physical bank)` 找前后两次使用该 bank 的模型层 A、B。
  latent、indexer 组的 bank 相互独立，各自从组内序号 0 开始，固定按
  `组内层序号 % bank数` 选择。前缀长度、chunk 序号不改变 bank。
- 同批所有请求在同一层使用相同编号的 bank，各自占用独立的物理块。
  consumer gate 逐个等待各请求的 load 完成事件，整段等待只统计一次。
  `banks` 按实际 bank 列出等待、复用窗口和 DMA 时间。
- 每个复用窗口的覆盖率：`1 - wait(group, bank, B) / (start(B) - start(A))`。
  分子是 B 在这个 bank 的 consumer gate 上等待的 NPU 时间。
- 4 bank 时，latent 组通常比较 N 和 N+4，而非 N 和 N+1；indexer 组按自己的
  实际使用顺序比较，稀疏层对应的模型层间隔可能大于 4。
- 同一模型层有多个有效 bank 比值时先对它们取平均，再对模型层取算术平均，
  保证有 indexer 的模型层不会重复加权。`layer_samples` 是有效模型层数，
  `bank_reuse_samples` 是有效 bank 配对数，不用总等待除以总窗口时间重新计算。
- 每个 chunk 独立配对，不跨 chunk。每个 group/bank 的首次使用没有前驱，
  该次等待不计算复用比值，但仍计入 `initial_bank_wait_ms` 和 `all_wait_ms`。
  `first_layer_wait_ms` 继续保留首个模型层的等待。

这个 coverage 的窗口与真实 bank 复用关系一致，用于观察下次消费 bank 时仍剩多少等待。
它不覆盖请求排队等窗口外耗时；启动等待和完整请求时间仍需同时看
`initial_bank_wait_ms`、`all_wait_ms` 和客户端 TTFT。

load/save 的两端 event 记录在该 bank 原有的 FIFO stream 上，且在已有的计算完成依赖之后。
因此传输区间不包含传输流等待计算完成的时间。
`load_ms`、`save_ms`、`dma_ms` 同时输出，按各 bank 区间累加；不同 stream
互相重叠的时间会分别计入各自区间，不能将该总和理解成服务端墙钟时间。
混合 C8 路径的区间也包含传输调用内的打包/搬运 kernel。
event 区间还可能包含主机下发不足时的设备空隙。

每个 TP/DP worker、每段一条 `[PREFILL_OVERLAP]` JSON 日志，包括
request_id、token 范围、覆盖率、有效模型层及 bank 配对数、复用等待、复用窗口累计时间、
首次使用及全层等待，还有各 bank 的 load/save、配对数及覆盖率。
日志异步输出，可能晚于下一批的主机日志。
小于 0 或大于 100% 的异常结果保留原值，`out_of_range_layers` 标记数量，避免隐藏测量问题。
新口径为 `schema_version=2`、`coverage_definition=same_bank_reuse_v1`。
旧 schema 1 是相邻模型层统计，不能由旧汇总恢复同 bank 配对的逐层时间，绘图会跳过并提示；
需要更新 LMCache-Ascend 并重启服务重新统计，不能把旧 coverage 改个名称当作新结果。

## 开销和完整性

主路径只记录 NPU event 和提交轻量统计对象。读取 event、汇总、JSON 序列化和日志
由后台线程执行；只在 `query()` 确认完成后，直接调用 ACL 的
`aclrtEventElapsedTime` 读取时间。特意绕过 torch-npu 的 `Event.elapsed_time()`，
因为其实现包含 `emptyAllNPUStream()` 和 event synchronize。
实现依据见 [torch-npu NPUEvent.cpp](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUEvent.cpp)。
ACL 动态库也在后台延迟加载，使用标准 CANN 环境中的 `libascendcl.so`。
没有为统计新增 `synchronize()`、`wait_event()` 或 `wait_stream()`，不拷贝 tensor 到 CPU。
计时 event 独立于推理所用的 bank/publication event，完成统计后才放回池中复用。

开启计时仍有 event 记录和 Python 下发开销，不能承诺严格零开销；
真实影响需要在 NPU 上比较开关前后的吞吐或 profile。
关闭开关不创建计时 event 或统计线程；空跑及无法确认完整请求范围的 batch 不记录事件。
待收集队列最多 32 批，event 空闲池最多 4096 个。队列满时丢弃统计、保留正常推理，
后续日志的 `dropped_steps` 会显示累计丢弃数量。中断批次标为 `incomplete`，
后台计时错误标为 `collector_error`；绘图排除这些记录。
停止服务时只尝试排空已完成的计时，不等待尚未完成的 NPU 工作；
强制杀进程可能缺少最后几条统计。

## 离线绘图

在任何有 Python 和 matplotlib 的机器上运行，无需导入 vllm、torch 或 NPU 库：

```bash
python3 -m pip install matplotlib
python3 tools/plot_prefill_overlap.py server.log --output overlap.png
```

多个日志可以一起处理，支持带引号的 glob。输出白底图片及同名 CSV。
图包含八个子图：覆盖率、`dma_ms`（save + load）、`save_ms`（卸载）、
`load_ms`（加载）、`all_wait_ms`（包括首层的所有 bank 等待）、
`first_layer_wait_ms`（首层 bank 等待）、`wait_ms`（同 bank 有效复用窗口的等待）、
`interval_ms`（同 bank 复用窗口之和）。横轴都是实际 prefill token 区间。
这些复用窗口会互相重叠，其总和不是模型墙钟时间。
首层和全层等待保留在图中，`initial_bank_wait_ms` 另保留在 CSV；
没有前驱的各 bank 首次使用不计算比值，但不会丢弃等待时间。
若旧日志缺少首层或全层等待字段，对应结果留空，不会用零填补。
覆盖率纵轴为百分比；耗时纵轴为毫秒，默认每个 DP/TP rank 一条曲线。
相同 rank、相同 token 区间的多个样本按有效层数合并，仍保持逐层比值的平均口径。
可使用 `--request-id`、`--tp-rank`、`--dp-rank` 筛选，或者 `--aggregate-ranks` 合成一条曲线。
图中耗时使用每个 worker/chunk 的平均值（时间总和 / `worker_steps`），
避免重复请求或合并多个 rank 后，单纯因为样本数量增多而把耗时画大。

```bash
python3 tools/plot_prefill_overlap.py 'tp4_dp4_logs/*.log' \
  --request-id YOUR_REQUEST_ID --dp-rank 0 --tp-rank 0 --output overlap-tp0.png
```

图和 CSV 保留真实区间及异常覆盖率；重复日志行会去重。
CSV 中的耗时为对应样本的总和，覆盖率不从这些时间总和重新计算。
已有 schema 2 日志可以直接重新绘图，无需重新采集；schema 1 不能用于新覆盖率。

## 本地 CPU 验证

```bash
# LMCache-Ascend 仓
python3 -m pytest --confcutdir=tests/standalone \
  tests/standalone/test_prefill_overlap.py \
  tests/standalone/test_layerwise_prefill_dma.py \
  tests/standalone/test_prefill_mixed_c8_packets.py \
  tests/standalone/test_prefill_failed_load_completion.py

# vllm-ascend 仓
python3 -m pytest --confcutdir=tests/standalone \
  tests/standalone/test_prefill_overlap_plot.py
```

CPU 测试检查 2/4/8 bank 实际复用配对、分组隔离、稀疏组层号、首次使用等待保留、
按模型层平均、真实调度范围、单请求限制、
未完成 event 不复用、队列满时不阻塞，以及计时标记不破坏原有 bank FIFO 依赖。
它们不代替 NPU 性能和硬件计时验证。
