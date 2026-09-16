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
计算 chunk 和 256-token LMCache chunk。输出最多 128 token，允许正常 EOS。
`--prompt-tokens` 现在仅是可选的最小长度检查，不会补齐或截断固定输入。

三次运行统一使用 `max_model_len=16384`、`gpu_memory_utilization=0.96`。
实际 prompt 加输出长度超出 16384 时，在启动模型前报错，不静默截断文章。
普通单机服务脚本 `tools/serve_glm52_baseline.sh` 也使用 `16384 / 0.96`；
其原有 MTP、图模式等其他配置不变。

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
- `baseline/archive/`、`prefill/archive/`：实际 LMCache 保存的数据。
- `kv_statistics.csv`：逐 rank/层/分量，baseline 与候选值及其绝对值的均值、
  总体方差、最大值；`diff = candidate - baseline` 和 `abs_diff` 的同类统计。
- `summary.json`：上述统计、持久化 KV 的逐组/层对比、缓存命中证据、
  输出 token 是否相同及首次分叉的位置。

**不使用数值差异阈值，不因 1e-7 或更小差异中断。** NaN/Inf 单独计数，
不会当作相等。缺层、缺 rank、加载失败、缓存命中不足、实际重新算了整个
prompt 等结构性问题会报错，避免“没测到却通过”。

decode 的 KV 只在两次输出的共同 token 前缀内进行数值对比；输出一旦分叉，
后续输入已不同，不再把它们的 KV 差异当作同输入误差。原始记录仍全部保留。
`decode_reloaded` 是实际消费过的历史行：latent sparse top-k 未选中的位置
不冒充已验证，覆盖行数在 CSV 中单独列出。Indexer 按自己的 block table 对齐。
历史行只记录首次消费，当前 token 写入全部记录。

无需重新跑模型即可重新汇总：

```bash
python tools/layerwise_prefill_check.py --analyse-only --run-dir /实际结果目录
```

## 测试边界

三次均为 **eager、关闭 MTP**，保证 Python 探针观测每次执行且 token 位置一致。
探针有同步和拷贝开销，可能改变重叠时序，不是性能测试，也不能证明不存在
仅在异步时序下出现的竞争。没有修改默认生产执行路径。

磁盘 connector 是测试专用持久化传输，保留 flat KV 的 `valid_tokens` 等元数据，
不持久化进程指针。**不验证 Mooncake 网络/租约、跨机 PD、MTP 接受率或图模式精度。**
当前本地只能跑 CPU 合约回归；完整 NPU 三段执行需要在服务器验证。
