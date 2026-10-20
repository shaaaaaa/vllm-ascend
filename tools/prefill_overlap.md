# Prefill two-bank overlap statistics

四仓 `dev-lmy` 分别基于 fetch 后的 `origin/dsa-two-groups`（vllm）、
`origin/sparse`（vllm-ascend）及 `origin/dev-qzy`（两个 LMCache 仓）。
请配套更新四仓；vllm 本次只有分支，没有代码改动。

在 **P 节点启动服务之前**设置：

```bash
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
export VLLM_ASCEND_PREFILL_OVERLAP_STATS=true
```

第二个变量默认 `false`，合法值为 `true` / `false`，不是敏感配置。
它必须与已有 P-node 开关同时开启才生效。D 节点保持原配置。
统计不要求开启 tensor dump、profiler 或已有的 start-load 调试开关，
也不会修改图、调度、缓存或 DMA 的依赖关系。

## 统计口径

每次实际 chunked prefill 单独统计，要求整个模型 batch 中只有一个 request。
同批多个 request 一律跳过，即使它们处于相同 token 区间。
同时检查 connector metadata 和 attention metadata，避免漏掉无需缓存操作的请求。
空跑、无可确认请求范围的 batch 和 decode 不统计。

token 区间来自调度器这次实际计算的 `[start, end)`，按 request 自己的逻辑位置计数。
例如每批 4096 token，图上依次显示 `0-4095`、`4096-8191` 等；
末段不足 4096、已有前缀命中、调度器实际切小 batch 时保留真实范围，
不会把它们伪装成完整 4096-token 段。服务端 JSON 的 `token_end` 为开区间终点。

定义：

- `start(N)`：模型第 N 层最后一次 bank 等待结束的 NPU event。
  若该层有 latent 和 indexer 两个等待，取后者的结束点，保证它们都在该起点之前。
  这是 bank 就绪后开始计算的边界，不是 Python 进入 decoder 的主机时间。
- `wait(N)`：该模型层在计算流上等待 bank 就绪的 NPU 时间之和。
  indexer 的稀疏组序号会转换为实际模型层号。
- 每层覆盖率：`1 - wait(N+1) / (start(N+1) - start(N))`。
- 每段覆盖率：所有有效相邻层覆盖率的**算术平均**。首层不参与；
  不是 `1 - sum(wait) / sum(interval)`，也不用 DMA 总耗时作分母。

load/save 的两端 event 记录在该 bank 原有的 FIFO stream 上，且在已有的计算完成依赖之后。
因此传输区间不包含传输流等待计算完成的时间。
`load_ms`、`save_ms`、`dma_ms` 同时输出，按各 bank 区间累加；不同 stream
互相重叠的时间会分别计入各自区间，不能将该总和理解成服务端墙钟时间。
混合 C8 路径的区间也包含传输调用内的打包/搬运 kernel。
event 区间还可能包含主机下发不足时的设备空隙。

每个 TP/DP worker、每段一条 `[PREFILL_OVERLAP]` JSON 日志，包括
request_id、token 范围、覆盖率、有效层数、等待时间、层间隔总时间、
首层等待及各 bank 的 load/save 时间。日志异步输出，可能晚于下一批的主机日志。
小于 0 或大于 100% 的异常结果保留原值，`out_of_range_layers` 标记数量，避免隐藏测量问题。

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
关闭开关不创建计时 event 或统计线程；被跳过的多请求 batch 也不记录计时 event。
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
横轴是实际 prefill token 区间，纵轴是覆盖率百分比；默认每个 DP/TP rank 一条曲线。
相同 rank、相同 token 区间的多个样本按有效层数合并，仍保持逐层比值的平均口径。
可使用 `--request-id`、`--tp-rank`、`--dp-rank` 筛选，或者 `--aggregate-ranks` 合成一条曲线。

```bash
python3 tools/plot_prefill_overlap.py 'tp4_dp4_logs/*.log' \
  --request-id YOUR_REQUEST_ID --dp-rank 0 --tp-rank 0 --output overlap-tp0.png
```

图和 CSV 保留真实区间及异常覆盖率；重复日志行会去重。
CSV 中的耗时为对应样本的总和，覆盖率不从这些时间总和重新计算。

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

CPU 测试检查统计公式、首层排除、稀疏组层号、真实调度范围、单请求限制、
未完成 event 不复用、队列满时不阻塞，以及计时标记不破坏原有 bank FIFO 依赖。
它们不代替 NPU 性能和硬件计时验证。
