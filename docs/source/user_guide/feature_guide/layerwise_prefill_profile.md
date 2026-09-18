# 单机 P 节点 layerwise prefill profile

在四仓配套的 `lmy_merge_prefill_layerwise_cache` 分支、Linux Ascend 服务器运行：

```bash
python -u tools/layerwise_prefill_profile.py 2>&1 | tee log.log
```

默认完整 GLM-5.2 权重 `/workspace/models/GLM-5.2-w4a8c8-0723`，TP8、DP1、
8 张卡、MTP1、`gpu_memory_utilization=0.96`。默认只跑 `100k_on`。
需要重新抓同长度 OFF 对照时，加 `--include-off`，按 100k OFF、ON 顺序各跑一次，每次独立加载模型：

```bash
python -u tools/layerwise_prefill_profile.py --include-off 2>&1 | tee log.log
```

| 目录 | 输入 | `LAYERWISE_PREFILL_P_NODE` | max-model-len |
| --- | --- | --- | --- |
| `100k_on`（默认） | 约 100000 tokens | true | 100000 tokens 时为 100352 |
| `100k_off`（显式选择） | 与 100k ON 使用相同 token IDs | false | 与 ON 相同 |
| `10k_on` / `10k_off`（`--case`） | 原短文章，约 9600 tokens | true / false | 16384 |

100k 固定输入位于 `examples/layerwise_prefill/article_summary_100k.txt`：
以原文章组成十二份编号资料副本，文本已在仓库中，不是在运行时循环生成。
它是重复文章构造的性能负载，不是十二次独立试验，也不是新的精度测试。
10k case 仍读取 `examples/layerwise_prefill/article_summary.txt`，不再补齐短文章到恰好 10000 tokens。

脚本先 tokenize 正文，再按 token 预算裁剪**正文**，应用完整 chat template；
边界差异最多调整三次，不进行无上限扩容或整篇二分重编码。
实际长度由服务器模型 tokenizer 确定，允许略短于目标，不裁掉 chat 结束标记；
如果长度不足目标的 95%，直接报错，避免 tokenizer 截断造成无限扩容。
保存原文、实际输入文本与 token IDs，并打印准备耗时和实际 token 数。

两组都使用 eager、4096-token chunked prefill、1024-token LMCache chunk，
保留普通 MTP1 路径，但只生成首个 token，不进行后续 decode，也不跑 D 节点。
OFF/ON 环境只改变该特性开关；不使用八层裁剪、dummy 权重、KV dump、
parity hook 或测试文件 SDK。

## 范围与资源

这是**本地 CPU cache 的 P 节点 profile**，包含 D2H 保存、历史 H2D 回载与
计算重叠。不开 Mooncake/RemoteFill，不用文件模拟远端，也不包含远端持久化等待；
不能拿它代表完整跨机 PD 的端到端耗时。无远端 store 队列，两种模式均设置
`store_async=false`，逐层本地传输仍走原实现的 stream/event。

默认共享 CPU cache 为 16 GiB，需要足够的主机内存和 `/dev/shm` 可用容量。
可用 `--cpu-cache-gb` 修改；脚本不清空其他进程的共享内存。
实际 NPU 容量仍由启动时分配结果确认。100k OFF 不使用逐层复用，完整模型可能因
HBM 容量不足无法启动；默认不运行它，也不会偷偷改成八层或短输入。

模型初始化、初始化阶段的预热与 profiler 导出不在采集范围内；
采集覆盖一次真实请求的全部 prefill chunks、首 token 采样以及实际执行的 MTP。
不额外发送 warmup 请求，避免复用上次请求的 KV；真实首请求的一次性成本会出现在 trace。
`result.json` 中的时间包含 profiler 开销，不能当作无 profiler 的性能结论。

## 输出

启动时打印 `layerwise-profile-xxxx` 结果目录。每组输出：

- `server.log`：完整启动和请求日志，同时输出到终端/`log.log`。
- `engine_options.json`、`environment.json`、`result.json`：参数、首 token 和请求统计。
- `profile/`：8 个 worker 的原始 profiling 数据及导出的 `ASCEND_PROFILER_OUTPUT`。
- `traces.json`：8 个 `trace_view.json` 的完整路径，可在 MindStudio 时间线中打开。

根目录还有 `100k_article_source.txt`、`100k_input.txt`、`100k_prompt.json`，
分别是固定原文副本、实际送入 chat template 的文本、实际 token IDs 和长度。

脚本在每组模型退出后自动解析 profile，导出期间会打印进度阶段，不会重新运行模型。
若仅解析失败，可重新解析已有目录：

```bash
python -u tools/layerwise_prefill_profile.py --analyse-only /path/to/layerwise-profile-xxxx
```

仍可用 `--case 10k_on` 回到短输入，或 `--case 100k_off` 单独跑长输入 OFF。
`--case all` 与 `--include-off` 一样，只跑 100k OFF/ON，不额外运行 10k。
本地 CPU 单测只验证输入、配置、
采集生命周期和进程顺序，真实 NPU/MindStudio 结果需要在服务器上确认。

## 定位长输入卡住的位置

仅凭“整晚未退出”无法区分推理和 profiler 问题。脚本在阶段切换时打印日志，
不增加逐层同步或逐层日志。看最后一条 `[PREFILL_PROFILE]`：

| 最后开始、尚未完成的阶段 | 当前等待 |
| --- | --- |
| `loading tokenizer` / `tokenizing fixed file` | tokenizer 加载 / 固定文本编码 |
| `loading full model` | 模型初始化，还未采集 |
| `profiler start begin` | 启动 profiler |
| `generate begin` | prefill、KV 回载/保存、MTP、首 token |
| `generate complete` 后的 `profiler stop begin` | 推理已完成，正在停止采集/落盘 |
| `model shutdown begin` | 模型进程清理 |
| `exporting ... traces (model has exited)` | NPU 模型已退出，正在离线解析 profile |

输入生成在模型启动前完成；如果日志已经出现 `generate begin`，就不是文本生成循环卡住。
100k 覆盖约 25 个 4096-token prefill chunks，trace 明显大于短输入；导出慢不等于 NPU 死锁。
具体根因仍需结合停住阶段和 worker 日志确认，不能仅凭输入长度判定。

## 检查传输与通信重叠

当前脚本默认只跑 100k ON，`--include-off` 才追加同长度 OFF 对照。
在 ON 的非首个 prefill chunk，查看同一 worker 的 compute、copy 和 HCCL
stream：普通 `o_proj` GEMM 之后，`single_layer_paged_kv_copy` 与它的 TP
all-reduce 应有机会并行；下一层读取对应 bank 前仍必须等 H2D 完成。
不要把 CPU 侧 `AscendCL@hcom_allReduce` API 区间直接当作设备通信执行区间。
比较实际设备时间线及整个 chunk 耗时；这里调整的是提交依赖，硬件资源竞争仍可能
限制实际重叠。第一段没有历史 H2D，最后的发布/源释放等待也仍可能留下空隙。
