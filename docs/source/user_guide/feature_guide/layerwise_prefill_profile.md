# 单机 P 节点 layerwise prefill profile

在四仓配套的 `lmy_merge_prefill_layerwise_cache` 分支、Linux Ascend 服务器运行：

```bash
python -u tools/layerwise_prefill_profile.py 2>&1 | tee log.log
```

默认只运行完整 GLM-5.2 的 `80k_on`。加 `--include-off` 才依次运行 `80k_off`、`80k_on`，每组分别启动、关闭模型；两组均为 TP8、DP1、MTP1、`gpu_memory_utilization=0.97`。每组仅采集前 3 个和最后 3 个 **compute-prefill chunk**（每个最多 4096 tokens，不是 1024-token LMCache 存储块）。中间 chunk 仍正常计算和传输 KV，只是不开 profiler。

```bash
python -u tools/layerwise_prefill_profile.py --include-off 2>&1 | tee log.log
```

| case | 输入 | `VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE` | max-model-len |
| --- | --- | --- | --- |
| `80k_on`（默认） | 约 80000 tokens | true | 84000（预留约 4000 输出 tokens） |
| `80k_off`（加 `--include-off`） | 与 ON 完全相同的 token IDs | false | 与 ON 相同 |
| `10k_off` / `10k_on`（`--case`） | 原短文章，约 9600 tokens | false / true | 16384 |

80k 固定源文本在 `examples/layerwise_prefill/article_summary_80k.txt`，由原文章的十份编号资料副本组成；运行时不是循环生成文章。脚本用模型 tokenizer 裁剪正文，应用完整 chat template，并保存 `80k_article_source.txt`、`80k_input.txt`、`80k_prompt.json`（包括实际 token IDs）。实际长度可略短于 80000；若不足目标的 95%，脚本报错，不会悄悄拿短输入当 80k 对照。OFF 和 ON 读取同一份保存的 token IDs。10k case 仍使用 `examples/layerwise_prefill/article_summary.txt`。

两组使用 eager、4096-token chunked prefill、1024-token LMCache chunk。MTP1 保留，但仅生成首个 token，不进行后续 decode，也不跑 D 节点。这是本地 CPU cache 的 P 节点 profile：包含逐层 D2H 保存、历史 H2D 回载和计算，不启动 Mooncake/RemoteFill，不能代表跨机 PD 的端到端耗时。两组均设置 `store_async=false`。

默认共享 CPU cache 为 24 GiB，可用 `--cpu-cache-gb` 修改；需保证主机内存和 `/dev/shm` 有足够空间。NPU 容量以模型启动时的实际分配为准。若启用 OFF 后因 HBM 不足无法启动，脚本会报错并停止，不会改用短输入或八层模型。

模型初始化、初始化预热和 profiler 导出不在采集窗口内。输入恰好 80000 tokens 时，共 20 个 compute-prefill chunk：`head` 为第 1–3 个，范围 `[0, 12288)`；`tail` 为第 18–20 个，范围 `[69632, 80000)`。实际窗口根据保存的 prompt token 数计算。工具在 worker RPC 安装采集包装，不更改正常服务的调度或 KV 传输；每个窗口开始/结束各同步一次 NPU，不在每层或每个 chunk 之间额外同步。窗口边界有 profiler 启停开销，不适合衡量正常流水线。`result.json` 的总时间也包含 profiler 开销。

启动时打印 `layerwise-profile-xxxx` 结果目录。每组目录包含：

- `server.log`、`engine_options.json`、`environment.json`、`result.json`。
- `profile/`：8 个 worker 的原始 profile 和导出的 `ASCEND_PROFILER_OUTPUT`，每张卡有 `80k_off_head_...`、`80k_off_tail_...` 或对应 ON 的两份。
- `traces.json`：80k 每组应有 16 个 `trace_view.json` 路径，可在 MindStudio 分别打开首、尾窗口。
- `capture_plan.json`、`capture_windows.json`：计划窗口和各 rank 实际执行的 chunk 编号、token 范围以及是否采集。中间 chunk 的 `window` 为 `null`。

只跑 ON、只跑 OFF 或回到短输入：

```bash
python -u tools/layerwise_prefill_profile.py --case 80k_on
python -u tools/layerwise_prefill_profile.py --case 80k_off
python -u tools/layerwise_prefill_profile.py --case 10k_on
```

`--case all` 与 `--include-off` 等价，均先跑 80k OFF、再跑 80k ON；不会额外跑 10k。若只需重新解析已有 profile：

```bash
python -u tools/layerwise_prefill_profile.py --analyse-only /path/to/layerwise-profile-xxxx
```

若长输入看似卡住，先看最后一条 `[PREFILL_PROFILE]`：`loading full model` 表示仍在初始化；`generate begin` 表示正在执行 prefill/KV 传输和首 token；`model shutdown begin` 表示退出模型进程；`exporting ... traces` 表示离线解析。80k 的中间 chunk 仍执行，只是不采集，因此不能仅凭 profile 文件数量判断是否卡在计算。

要定位每个 chunk 开头的等待，可单独运行诊断采集：

```bash
python -u tools/layerwise_prefill_profile.py --diagnose-chunk-start 2>&1 | tee log.log
grep -aF '[PREFILL_START]' log.log
```

诊断日志按 `start_load_kv_total`、`materialize_bank_maps`、`retrieve_setup_before_prime`、首两层各组的 `prime_retriever`、`dma_plan`、`second_bank_submit`、`store_chunk_scan` 和 `prime_storer` 拆分主机时间，并记录历史 token 数和已有/新存储 chunk 数。`first_bank_wait_enqueue` 只记录首层设备依赖的非阻塞下发；`first_bank_wait_device.device_wait_ms` 在已有的 chunk-end 同步后读取事件，表示计算流实际等待加载/保存事件的时间。该选项不再额外同步首层计算流，但事件记录和日志仍有诊断开销；默认 profile 和生产路径不开启。

ON 路径在 chunk 准备阶段只提交第 0 层加载；首层 SFA 入口以虚拟 N=-1 触发第 1 层异步加载（不保存或计算虚拟层），后续仍按 N+2 提交。因此 `second_bank_submit` 现在发生在首层 forward 入口，而非 `start_load_kv_total` 内。

`final_load_source_sync` 是末层加载后、释放 H2D 源内存前已有的同步；`store_publish_sync` 是 chunk 结束、发布 CPU KV 前已有的 D2H 同步。两者只增加计时日志，不改变原有同步。旧版诊断的 `first_bank_wait.elapsed_ms` 包含额外的首层 `Event.synchronize()`，可能把先前排队的模型计算算进去；不能据此判断 H2D 等待。

比较重叠时，查看同一 worker 的 compute、copy/DMA 和 HCCL 设备时间线；不要把 CPU 侧 `AscendCL@hcom_allReduce` API 区间直接当作 NPU 通信执行区间。ON 的下一层读取对应 bank 前仍须等待 H2D 完成。提交时机允许与其他工作重叠，但实际效果取决于设备资源竞争。
