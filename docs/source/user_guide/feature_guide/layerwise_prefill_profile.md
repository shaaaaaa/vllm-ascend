# 单机 P 节点 layerwise prefill profile

在四仓配套的 `lmy_merge_prefill_layerwise_cache` 分支、Linux Ascend 服务器运行：

```bash
python -u tools/layerwise_prefill_profile.py 2>&1 | tee log.log
```

默认完整 GLM-5.2 权重 `/workspace/models/GLM-5.2-w4a8c8-0723`，TP8、DP1、
8 张卡、MTP1、`gpu_memory_utilization=0.96`。默认只跑 `100k_on`，
只采集前 3 个和最后 3 个 **compute-prefill chunk**（每个最多 4096 tokens，
不是 1024-token LMCache 存储块）。中间 chunk 正常计算、保存/回载 KV，但不开 profiler。
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

模型初始化、初始化阶段的预热与 profiler 导出不在采集范围内。
100k 的 OFF/ON 都仅抓首尾各 3 个 prefill chunk；10k 仍采集完整请求。
例如输入恰好 100000 tokens 时，共 25 个 chunk：`head` 抓第 1–3 个，
token 范围 `[0, 12288)`；`tail` 抓第 23–25 个，范围 `[90112, 100000)`，
包含最后的不足整块部分、首 token 采样以及这一段实际执行的 MTP。
实际窗口按保存的 prompt token 数计算，不能把约 100k 的输入一律当作 25 个 chunk。

工具在模型加载后通过 worker RPC 安装采集包装，不改正常服务代码，不改变调度或 KV 传输。
每段开始/结束各同步一次 NPU（两段共 4 次），避免未完成的中间工作进入尾段 trace；
不在每层、每个 chunk 或短 kernel 之间额外同步。窗口边界有 profiler 启停及同步开销，
不要用边界空白衡量正常流水线，也不要把分段 profile 当成连续整段计时。
记录所有实际 chunk 的 token 范围，结束后核对固定 4096-token 调度是否与计划一致；
若不一致，保存报告并报错，不把不准确的窗口当作成功。
不额外发送 warmup 请求，避免复用上次请求的 KV；真实首请求的一次性成本会出现在 trace。
`result.json` 中的时间包含 profiler 开销，不能当作无 profiler 的性能结论。

## 输出

启动时打印 `layerwise-profile-xxxx` 结果目录。每组输出：

- `server.log`：完整启动和请求日志，同时输出到终端/`log.log`。
- `engine_options.json`、`environment.json`、`result.json`：参数、首 token 和请求统计。
- `profile/`：8 个 worker 的原始 profiling 数据及导出的 `ASCEND_PROFILER_OUTPUT`。
  100k 每张卡各有 `100k_on_head_...` / `100k_on_tail_...` 两份（OFF 同理）。
- `traces.json`：100k 共 16 个 `trace_view.json` 的完整路径，10k 仍为 8 个。
  可在 MindStudio 时间线中分别打开首段和尾段。
- `capture_plan.json` / `capture_windows.json`：100k 的计划窗口、各 rank 实际执行的
  chunk 编号、token 起止范围（左闭右开）以及是否被采集。中间 chunk 的 `window` 为 `null`。

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
| `generate begin` | prefill、KV 回载/保存、MTP、首 token；100k 中间 chunk 不采集 |
| `.../head` / `.../tail` 的 `profiler start` / `stop begin` | worker 的首段/尾段采集切换 |
| `generate complete` 后的 `profiler stop begin` | 推理已完成，正在停止采集/落盘 |
| `model shutdown begin` | 模型进程清理 |
| `exporting ... traces (model has exited)` | NPU 模型已退出，正在离线解析 profile |

输入生成在模型启动前完成；如果日志已经出现 `generate begin`，就不是文本生成循环卡住。
100k 仍计算约 25 个 4096-token prefill chunks，但仅采集首尾共 6 个；
因此减少采集数据量与解析工作，不会省略中间的模型计算。导出慢不等于 NPU 死锁。
具体根因仍需结合停住阶段和 worker 日志确认，不能仅凭输入长度判定。

## 检查传输与通信重叠

本 profile 脚本默认开启实验性的短 kernel 低优先级回载；生产 connector 的默认值仍是 `0`。
显式设置 `LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=0` 可关闭该实验，设置 `1` 则开启。
只影响 P-node 的逐层传输：本层 SFA 结束后，尽早把本层 D2H 和下一层 H2D
按顺序提交到同一个低优先级 FIFO，不再等 `o_proj` 的 all-reduce 前才提交。
D2H 保留原单 kernel（通常只保存本轮新计算的 4096 token），H2D 按最多 16384 token
拆分；计算仅等待对应层的完成事件，不等待整个队列。D 节点、远端发布路径不变。
需要先重新编译安装 LMCache-Ascend。
脚本会打印实际开关值并写入每个 case 的 `environment.json`，不修改父进程环境。
比较拆分效果时，两次都使用 `100k_on`，只切这个开关：

```bash
LMCACHE_ASCEND_PREFILL_SPLIT_LOAD=0 python -u tools/layerwise_prefill_profile.py --case 100k_on --cpu-cache-gb 32
python -u tools/layerwise_prefill_profile.py --case 100k_on --cpu-cache-gb 32
```

初始化仅一次查询优先级支持范围；不支持则明确报错，不静默用普通流代替。
CANN 8.5 对 A2/A3 的 stream priority 标为预留参数，不能保证服务器支持该实验。
先在 LMCache-Ascend 执行 `python -u tools/check_prefill_split_load.py --probe-only`，
无需加载模型即可确认 API 能力。低优先级不能抢占已执行的 kernel，也不保证只在空闲时运行。
新增开销：每次 D2H/H2D 各两次 record、两次设备侧 stream wait；每个 H2D 分片一次
kernel launch；每轮每组一次保存 metadata 的跨流就绪等待。D2H 不增加 kernel 数量，
没有逐片 CPU 等待、metadata 拷贝或新校验扫描；优先级能力检查只在队列创建时做一次。

当前脚本默认只跑 100k ON，`--include-off` 才追加同长度 OFF 对照。
在 ON 的非首个 prefill chunk，查看同一 worker 的 compute、copy 和 HCCL
stream：开启短 kernel 实验时，传输从 SFA 结束后即可开始，与后续投影及 TP
all-reduce 有机会重叠；关闭实验时保留普通 `o_proj` GEMM 后提交的时点。
下一层读取对应 bank 前仍必须等自己的 H2D 完成。
不要把 CPU 侧 `AscendCL@hcom_allReduce` API 区间直接当作设备通信执行区间。
比较实际设备时间线及整个 chunk 耗时；这里调整的是提交依赖，硬件资源竞争仍可能
限制实际重叠。第一段没有历史 H2D，最后的发布/源释放等待也仍可能留下空隙。
