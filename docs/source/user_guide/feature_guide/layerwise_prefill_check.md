# 单机三次验证 layerwise prefill cache

在四仓配套的 `lmy_merge_prefill_layerwise_cache` 上运行：

```bash
cd /workspace/lmy/vllm-ascend
python -u tools/layerwise_prefill_check.py 2>&1 | tee log.log
```

默认完整 GLM-5.2 真实权重 `/workspace/models/GLM-5.2-w4a8c8-0723`，
单机 TP8，设备 0–7。不裁层，不用 dummy 权重。默认读取仓库自带的
`examples/layerwise_prefill/article_summary.txt`：中文总结指令，加一份约 8200
英文词的虚构公共图书馆改造评估报告，包含 36 份现场记录。文件可以直接打开
阅读，不再运行时生成、重复追加段落或凑 token。模型路径可用 `--model` 修改，
输入文件可用 `--prompt-file` 修改。

只 tokenize 一次并打印实际 token 数，三次复用同一组 token IDs。
实际长度由模型 tokenizer 决定，不保证恰好 12k；必须超过 4096-token prefill
计算 chunk 和 256-token LMCache chunk。baseline/D 输出最多 4000 token，允许正常 EOS；
这里的单位是 token，不是词。文章的“约 200 字”摘要限制已删除。
`--prompt-tokens` 现在仅是可选的最小长度检查，不会补齐或截断固定输入。

三次运行统一使用 `max_model_len=16384`、`gpu_memory_utilization=0.96`。
实际 prompt 加输出长度超出 16384 时，在启动模型前报错，不静默截断文章。
使用默认 4000-token 输出预算时，prompt 最多 12384 token；精确 token 数必须
由实际 GLM tokenizer 计算，不能用文章英文词数替代。
普通单机服务脚本 `tools/serve_glm52_baseline.sh` 也使用 `16384 / 0.96`；
其 MTP 保留开启，不加 `--enforce-eager`，仍使用本地 CPU 缓存，不启用 NPU 直传。
**该 shell 脚本沿用清空 `/dev/shm/*` 的行为，只能在专用环境运行。**
下述三阶段验证仍使用磁盘交接，不需要 Mooncake 服务。

脚本入口不再先导入 torch，会立即输出启动提示；随后分别打印读取输入、
加载 tokenizer、tokenize、启动各模型阶段的进度。子进程使用无缓冲输出。

## 三次执行

1. **baseline**：P-node 关闭，普通 prefill + decode，`SHRINK_LATENT=2`。
   保存每层实际写入的 NPU latent/indexer KV、LMCache 持久化数据、输出 token 和文本。
2. **prefill**：P-node 开启，双 bank、逐层加载/保存，`SHRINK_LATENT=0`。
   只做 prefill 和第一个 token 的采样，保存每层 KV 与持久化数据。
   与 baseline 对比同一 prompt 的各层结果。
3. **decode**：P-node 关闭，consumer，`SHRINK_LATENT=2`。
   从第二次封存的磁盘 KV 通过正常 LMCache RemoteBackend 加载，不读取 baseline
   的 archive，且不能向 P archive 写数据。允许正常的 prompt 尾部重算，随后 decode。
   记录实际 NPU 加载/写入的 KV 和完整生成结果。

三个模型进程串行启动，每次结束清理本次创建的进程组；不会同时放三份权重，
不会清空 `/dev/shm` 或杀其他服务。默认 LMCache 每次需要 8 GiB 可用共享内存
（`--cpu-cache-gb` 可调）。原始 KV 全量记录占用磁盘，应预留数十 GiB；
长 prompt 的记录规模会继续增长。

## 统计结果

启动时打印结果目录 `layerwise-prefill-...`：

- `prompt.txt`、`prompt.json`：文章、实际 prompt token IDs、摘要校验。
  `prompt.txt` 保留输入文件的完整文本，`prompt.json` 同时记录原文件路径。
- `baseline/output.txt`、`prefill/output.txt`、`decode/output.txt`：模型输出。
  对应 `output.json` 同时记录 token IDs。
- 各阶段 `kv/rank*/`：全部 TP rank、各层、按逻辑 token 位置记录的原始 KV。
  短 decode 写入按每层/分量累计 256 行落盘，结束时刷新尾部；不减少采样行，
  避免 4000-token 输出产生数百万个单行文件。
- `baseline/archive/`、`prefill/archive/`：实际 LMCache 保存的数据。
- `kv_statistics.csv`：逐 rank/层/分量，baseline 与候选值及其绝对值的均值、
  总体方差、最大值；`diff = candidate - baseline` 和 `abs_diff` 的同类统计。
- `kv_chunk_statistics.csv`：按 prefill token 位置区间拆分的上述统计，包含
  `prefill_written`（baseline 写出 vs P 写出）和 `prefill_reloaded`（P 写出 vs P 回灌）。
  两种对比均覆盖所有已记录的 rank/层/分量，按层号自然排序。
- `summary.json`：上述统计、持久化 KV 的逐组/层对比、缓存命中证据、
  输出 token 是否相同及首次分叉的位置。
- `analysis_progress.jsonl`：每项统计完成即追加结果、耗时和分析进程 PID；
  archive 的中间记录是分批统计（`partial: true`），最终逐层合并值以 summary 为准。

D 结束后，离线统计默认使用 **64 个 CPU 分析进程**：按 rank/层/KV 分量并发，
持久化 archive 按文件批次并发。每个进程仅保留当前任务的数据，不把全模型 KV
传回父进程；同一任务的 trace 文件只读取一次，共同 decode 区间为空时不读文件。
每个分析子进程限制为 1 个 PyTorch 计算线程，避免多进程再各自展开大型线程池。
可用 `--analysis-workers N` 调整并发数，`1` 为串行；这不改变 baseline/P/D 的运行顺序。
64 并发面向多核、大内存服务器；相比 4 并发会增加 CPU 内存和磁盘 I/O 压力。

开始统计时即写出 token 对比，`summary.json` 的 `analysis_status: running` 表示
结果尚不完整；统计期间约每 5 秒更新快照和控制台进度，JSONL 则逐任务落盘。
`analysis_status: complete` 表示统计结束，仍需查看 `structural_errors`，不代表精度通过。
分析异常时保存已完成的统计并标记 `failed`。这些改动只影响离线分析，不改变推理路径。

**不使用数值差异阈值，不因 1e-7 或更小差异中断。** NaN/Inf 单独计数，
不会当作相等。缺层、缺 rank、加载失败、缓存命中不足、实际重新算了整个
prompt 等结构性问题会报错，避免“没测到却通过”。

decode 的 KV 只在两次输出的共同 token 前缀内进行数值对比；输出一旦分叉，
后续输入已不同，不再把它们的 KV 差异当作同输入误差。原始记录仍全部保留。
`decode_reloaded` 是实际消费过的历史行：latent sparse top-k 未选中的位置
不冒充已验证，覆盖行数在 CSV 中单独列出。Indexer 按自己的 block table 对齐。
历史行只记录首次消费，当前 token 写入全部记录。

`prefill_reloaded` 的参考值是 **P 自己写出的 KV**，不是 baseline 的 KV；因此可以
单独判断已观测到的 P 历史回灌是否保持数据不变，不把 baseline/P 的计算差异混进来。
这些统计位于 `summary.json` 的 `kv`（整层）和 `kv_by_prefill_chunk`（分区间）中。
`status: compared` 只表示观测行有对应参考值，不代表数值相等；`not_observed` 表示没有
加载记录，统计值留空，绝不当成差异为零。覆盖行数与非有限值计数仍需一起查看。

分区大小优先读取原运行的 `prefill/engine_options.json` 中的 `max_num_batched_tokens`，
其次读取 `run.json`，再读取 baseline 的 engine options；旧结果都未记录时才使用 4096，
并在日志和 summary 中明确标出来源。对于 9565-token prompt、4096-token 预算，
区间为 `[0,4096)`、`[4096,8192)`、`[8192,9565)`。这是按原计算预算划分的
**源 token 位置区间**，不是实际调度 step 或加载时间：例如第三次 forward 中加载的
早期 token，仍属于它自己的源位置区间。

旧探针只保存同一 token 的首次观测加载，后续重复回灌没有留档，不能通过离线分析补出。
P 最后一个区间没有后续 prefill 计算使用时，没有回灌记录是正常的；报告不会宣称它已验证。
控制台新增 `[PREFILL_CHUNK]` 和 `[PREFILL_RELOAD]`，仅打印 rank0 的 latent-nope
逐层摘要；indexer、pe 和其他 rank 的完整均值、方差、最大值及覆盖数在 CSV/JSON 中。
分区统计合并得到整层结果，不重复做一遍整层张量归约，同一任务仍只读取一次原始文件。

无需重新跑模型即可重新汇总：

```bash
python -u tools/layerwise_prefill_check.py --analyse-only --run-dir /实际结果目录 2>&1 | tee log.log
grep -aE '\[PREFILL_(CHUNK|RELOAD)\]' log.log
```

先停止仍在运行的旧统计进程，避免同时读同一批 KV、同时写同一份报告。
重新分析不会启动模型、重新生成 KV 或改写原始 trace/archive 文件。

## 测试边界

三次均关闭 MTP；仅 P 阶段使用 eager，baseline/D 使用普通 **PIECEWISE** 图，
保留 `vllm::mla_forward` 的逐层回调边界。探针安装时检查实际图配置，防止
切到 FULL/staged 后只记录 capture 而漏掉真实 decode。`output.json` 记录实际
`enforce_eager`、生成 token 数、上限和停止原因。
探针有同步和拷贝开销，可能改变重叠时序，不是性能测试，也不能证明不存在
仅在异步时序下出现的竞争。没有修改默认生产执行路径。

磁盘 connector 是测试专用持久化传输，保留 flat KV 的 `valid_tokens` 等元数据，
不持久化进程指针或分配器末尾的对齐填充。旧版 archive 含有对齐填充时，仍校验
整个文件载荷的 checksum，再按 shapes/dtypes 恢复实际 KV 字节，不改写原文件；
避免尾部不足一个 chunk 时，P 批量分配与 D 单次分配的 raw view 长度不同导致加载失败。
截断数据或校验不一致仍会报错，不会补零冒充有效 KV。
**不验证 Mooncake 网络/租约、跨机 PD、MTP 接受率或 FULL/staged 图。**
NPU 直传暂不接入本测试。
当前本地只能跑 CPU 合约回归；完整 NPU 三段执行需要在服务器验证。
