# 单机文件 PD 校验

`layerwise_prefill_file_check.py` 参考 `lmy_merge_prefill_layerwise_cache` 的
同名脚本，执行真实模型和 LMCache。它只在 Mooncake SDK 的字节存取接口替换为
文件，保留生产 connector、chunk key、合并 layer page、KV 分配、DMA、稀疏
索引与加载路径。所有运行参数写在脚本中，不读取外部 LMCache YAML。

## 运行

在安装当前四仓 `prefill_layerwise_cache` 分支的 Linux Ascend 环境运行。
下面显式指定之前四机服务使用的 GLM-5.3，避免误用其他 checkpoint：

```bash
set -o pipefail
python3 tools/layerwise_prefill_file_check.py \
  --model /workspace/models/GLM-5.3-w4a8c8 \
  --run-dir /workspace/layerwise-file-glm53 \
  2>&1 | tee log.log
```

不指定 `--model` 时与当前 profile 脚本一致，默认是
`/workspace/models/GLM-5.2-w4a8c8-0723`。启动首行会打印实际模型路径。
默认 TP8、DP1、4096 compute chunk、1024 LMCache chunk、24 GiB CPU cache、
max-model-len 16384、显存利用率 0.97、FlashComm1=1、MTP=1。
默认固定长文章约 10000 个输入 token，生成 256 个 token。支持
`--devices 0,1,2,3`、`--output-tokens 512`、`--mtp-tokens 0` 等显式覆盖。

复现之前五 token 左右的 completions 请求：

```bash
set -o pipefail
python3 tools/layerwise_prefill_file_check.py \
  --model /workspace/models/GLM-5.3-w4a8c8 \
  --prompt '你好，请介绍一下你自己' --prompt-format raw \
  --output-tokens 512 --run-dir /workspace/layerwise-file-short \
  2>&1 | tee log.log
```

`raw` 不添加 chat template；显式 prompt 只 tokenize 一次，三阶段复用同一组
token IDs。`--prompt-format chat` 则应用模型 chat template。至少需要两个
输入 token，才能验证 D 的缓存命中。短 prompt 和不足 1024 token 的尾块也走
真实存取。测试设置 `temperature=0`、`seed=1024`、`ignore_eos=True`，保证真正
执行指定长度的多 token decode，而不是因为 EOS 提前退出。

## 三个进程阶段

1. `baseline/`：关闭 P 开关，在一个实例里完整跑 prefill + decode，保存原始
   tensor、KV、logits 和输出 token。这里保留原生 OFF 的普通分层 CPU 对象；
   P/D 的文件存储使用生产的合并 layer page，不强制改写 OFF 的缓存布局。
2. `prefill/`：开启 `VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE`，执行 prompt 并生成
   第一个 token。P 使用 `store_async=true`、队列大小 2。请求完成后在测试收尾
   阶段等待后台存储完成，关闭 P 的进程组，然后封存 `store/`。
3. `decode/`：独立新实例，关闭 P 开关。向它提交同一份**原始 prompt**，由真实
   LMCache lookup/load 从文件恢复 KV，再重算最后一个 prompt token、继续生成。
   不把 P 的第一个输出 token 追加到 prompt。要求命中数恰好是 prompt 长度减一，
   并要求实际读取两组 DSA 的文件 payload；不会把重新做完整 prefill 算作成功。

脚本仅停止自己创建的进程组，不清空 `/dev/shm`。每阶段使用自己的 LMCache
实例，D 不复用 P 的共享缓存对象。`--store-gb` 只是文件 SDK 的 setup 参数，
不会申请原生 Mooncake 大内存段；真实字节落到 `store/` 所在磁盘。

## 复用 OFF

```bash
set -o pipefail
python3 tools/layerwise_prefill_file_check.py \
  --off-dir /workspace/layerwise-file-glm53 \
  --run-dir /workspace/layerwise-file-glm53-repeat \
  2>&1 | tee log.log
```

也可以把 `--off-dir` 指向旧目录的 `baseline/`。旧 OFF 已完成时，即使旧 P 或
D 失败，仍可复用。脚本校验 OFF 覆盖、tensor 文件、模型配置与运行参数，恢复
保存的 prompt IDs 和未显式覆盖的参数，只启动新的 P、D，不修改旧目录。
显式给出与旧 OFF 不一致的模型或计算参数会报错。之前
`layerwise_prefill_correctness.py` 的单 token OFF 没有 decode 记录，不能作为
本脚本的完整基线。

## 记录与比较

每阶段的 `tensors/rankN/` 保存完整 `.pt` 与 `index.jsonl`，不采样、不计算
tensor 指纹。记录主模型和 MTP 的模型/各层输入输出、SFA query、逻辑 top-k、
实际写入及消费的 KV、索引映射和采样前 logits。物理 block table/slot 用于解释 KV
来源，不要求两个实例物理地址相同。按实际逻辑位置和输入上下文对齐，排除
声明的 TP padding；不会直接用不同阶段的 step 编号配对。

FlashComm1 的 MTP `positions` 会经过 `reduce_scatter(SUM)`，模型实际收到的
非零位置数值是原位置乘 TP 数。探针从 proposer 的未分片 positions 读取逻辑
位置，同时核对分片后的真实输入是否符合求和结果；文件仍保存原始 tensor，
不会把它除以 TP 后覆盖。KV 和 token 的对齐使用逻辑位置，不从这个求和后的
数值或 CPU 序列长度反推。

`report.json` 是简短总报告，`comparisons.jsonl` 保存逐项原始数据分布、差异
比例、绝对误差、relative L2、RMSE/std 等全量统计。各阶段 `output.json` 保存
全部输出 token，`coverage.json` 保存每个 rank 的完整性信息；
`store-report.json` 检查两组 KV 是否确实从 P 的文件读入 D。

`passed` 只表示覆盖/结构/对齐、输出 token 和新增非有限数检查通过，
`numeric_tolerance_applied=false`。浮点差异是否可接受需根据统计判断。
缺少 tensor、无法对齐、D 没有缓存命中、没有实际文件读取或未执行要求的
MTP 验证，都不会静默报成功。生成 token 已发生分叉后，不把不同输入上下文的
中间变量强行比较为同一个计算。

重新生成报告，不加载模型：

```bash
python3 tools/layerwise_prefill_file_check.py \
  --compare-only /workspace/layerwise-file-glm53 2>&1 | tee compare.log
```

完整 tensor 记录占用大量磁盘，三阶段都保留原始数据。测试统一使用 eager 并
关闭图捕获，确保每次真实 forward 的探针都执行；CPU 回读会改变时序。因此
结果覆盖单请求、单机 TP 的文件恢复及 decode 数值路径，不代表四机 DP/EP、
原生 Mooncake/RDMA、并发 RemoteFill、图回放或无探针时 DMA 竞态已通过验证。
`--rpc-timeout-seconds` 默认 1800，`--stage-timeout-seconds` 默认 21600，包含
完整 tensor I/O 的时间。
