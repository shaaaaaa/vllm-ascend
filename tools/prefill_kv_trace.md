# 分段 KV hash 诊断

本工具用于定位长前缀、chunked prefill、多请求混批时的数据搬运错误。
只把 hash 和坐标写入服务日志，不保存、传输或反序列化 KV tensor 文件。

## 开启与复现

在所有 P、D worker 的启动环境中设置已有开关，然后重启服务：

```bash
export LMCACHE_PREFILL_STORE_DEBUG=1
```

保持模型、图模式、异步调度、Mooncake 和 bank 数配置不变。工具不会强制 eager。
这次优先复现 warmup 本身的乱码，可以只执行并发测试的 round 0：

```bash
python3 tools/multi_turn_kv_reuse_concurrent.py \
  --base-url http://7.150.2.72:8000 \
  --model-name GLM-5.3-falcon \
  --concurrency 16 --rounds 0 2>&1 | tee round0.log
```

此功能依赖 LMCache、LMCache-Ascend、vllm-ascend 三个仓的配套修改。
关闭开关不提交 hash kernel、镜像加载或 hash 回读。

## 检查的边界

每条 `[KVTRACE]` 都包含 request_id、主机、进程、rank、KV group、层和逻辑 token 范围。
同一逻辑范围、同一层、同一数据平面的 hash 才能比较。
若某个 KV group 按配置只由 TP0 保存，所有 D rank 都与该 group 的实际发布 rank 比较，
而不是假定 D rank 1 的源一定是 P rank 1。

| 观测 | 比较对象 | 首次不同所定位的边界 |
| --- | --- | --- |
| `MAP_ACTUAL` | `MAP_EXPECTED` | attention 使用的 bank 块表与请求应使用的 bank 不一致 |
| `P_LOAD` | `P_CPU_LOAD`，或此前实际保存/计算的尾部 | P 历史 KV 加载或不足一个缓存块的尾部恢复 |
| `P_COMPUTE` / `P_HISTORY_TAIL` | 同一 chunk 的 `P_LOAD` | 本层执行期间历史 KV 被改写 |
| `P_SCATTER` | `P_PRODUCED` | 新计算 KV 写入模型缓存的位置或内容 |
| `P_CPU_STORE` | 对应 `P_COMPUTE` | NPU bank 到 CPU 保存结果 |
| `P_CPU_LOAD` | 此请求此前 `P_CPU_STORE` | 此请求的 CPU 保存/复用 |
| `D_CPU` | P 最后一个 chunk 的 `P_FINAL` | P 最终数据到 D 继承的 CPU KV |
| `D_MIRROR` | `D_CPU` | D 独立、分块 H2D 加载 |
| `D_DIRECT` | `P_FINAL` | Mooncake 直写 NPU 的完整 indexer KV |
| `D_SELECTED`（可观测时） | D CPU 源中实际选中的历史行 | 第一轮真实稀疏 KV 传输目标 |

P 对每个 chunk、每层的完整逻辑历史和当前 KV 分段计算 hash，包含非 TP0。
不使用固定头尾采样；C8 indexer 的 key 与 scale 分开检查。
P 一般按 LMCache 存储块分段；最终跨 P/D 的参考按最多 128 token 分段，
并额外计算最后一小段所有前缀的 hash，适配 D 重算最后一个 prompt token 等截断。
非整块尾部保留实际起止位置，allocator padding 不参与比较。

全层 `[KVTRACE]` 使用简单的 `sum16x2-v1` 校验，显示为 16 个十六进制字符。
把原始字节按无符号 16 位整数分组，只计算两个 32 位结果：
`sum(words) XOR valid_bytes` 和 `sum(words * position)`，位置从 1 开始，整数溢出按模 `2^32` 处理。
奇数个字节在最后补一个零字节，补齐字节不计入有效长度。
位置加权用于检查普通求和无法区分的常见错位、重排；不做种子混合、多轮移位或乘法雪崩。
这是低成本数据校验，不是密码学摘要，仍可能发生碰撞。
CPU/NPU 使用相同的算法，
每个请求先校准两个后端；校准失败会报告 `invalid_algorithm`，不能解释为 KV 损坏。
同一请求中已租用且不可变的完整 CPU 源块复用首次 hash；实际 NPU bank 每个 chunk 都重新检查。
P 的 CPU 源 hash 在后台执行，等待该层真实 H2D 完成 event 后才读取源页，
避免阻塞加载主路径和误读提前发布但尚未完成 D2H 的源数据。
每层每段仍覆盖全部字节，不通过抽样降低覆盖范围；单段读回从 16 字节降为 8 字节。
P、D 必须一起更新并重启。分析器支持单独分析旧版 `mix32x4-v1` 日志，
但会拒绝把不同算法的 hash 混合比较；报告为 `invalid_algorithm`。
历史 `[KVH]` 少量 CPU 摘要是独立日志，不参与 `[KVTRACE]` 的比较。

D 在模型首次 decode 前独立逐块加载全部继承的 latent KV，仅用于 hash。
镜像不写入模型缓存；每次只加载一个存储块，读回的只是每段 8 字节校验值。
完整 indexer 若本来就是 Mooncake 直写 NPU，则直接检查真实目标，不为诊断改走 CPU 路径。

## 自动收集与比较

在任意装有本分支工具的机器上执行，替换 request_id、密码和实际路径：

```bash
python3 tools/prefill_kv_trace.py \
  --request-id cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --hosts root@7.150.2.72 root@7.150.3.11 root@7.150.3.60 root@7.150.6.177 \
  --workspace /workspace/fsi_lab/sqh \
  --password '你的密码' --skip-host-key-check \
  --output ./kv-trace-case
```

`--workspace` 可以是包含 vllm-ascend 的工作目录或仓库目录；默认搜集其中的
`tp4_dp4_logs/**/*.log`。路径不同时用 `--log-dir /绝对路径/日志目录`。
若路径在容器内，增加 `--container 容器名`。使用原生 OpenSSH askpass，不需要 sshpass。
远端先按 request UUID 筛选，再 gzip 压缩传输；不拉取整个服务日志。
默认最多等待 120 秒，供后台 hash 队列输出 END；需要时调整 `--wait-seconds`。

本地已有收集结果时，无需再次连接服务器：

```bash
python3 tools/prefill_kv_trace.py \
  --request-id cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --local ./kv-trace-case/*.jsonl.gz --output ./kv-trace-recheck
```

终端只打印汇总与首次差异/缺失位置；完整结果在 `report.json`。
`comparisons.jsonl` 仅写不一致项，`coverage-errors.jsonl` 保留最早的缺失/错误示例，
计数仍覆盖所有段。SQLite 文件仅保存 hash、坐标和日志来源。

## 结果与范围

- `differences`：列出最先观测到不同的搬运边界、rank、层、token 段和来源日志行。
- `incomplete`：缺 rank、层、段、源参考或队列 END，不能当作通过。
- `consistent`：所声明范围内的 KV 搬运检查一致，不代表模型计算和采样精度已通过。
- `invalid_algorithm`：CPU/NPU hash 校准失败，先处理诊断算子的支持问题。

整图回放不会逐次执行图内 Python 探针。D 的全历史独立加载在模型执行前，仍能检查；
但实际稀疏搬运探针可能没有记录，报告会用 `sparse_transfer_observed=false` 明确标记，
不会把独立镜像一致说成已经验证整图最终消费的槽位。MLAPO 融合计算不暴露独立 produced KV 时，
仍检查 bank 与保存数据，但不虚构 `P_PRODUCED` 对照。

全层、全历史 hash 会增加设备读带宽和计算开销，也会改变时序；仅用于定位，不能用于性能测试。
检查结果描述最早观测到的差异边界，不把 hash 一致解释为排除了所有时序或模型计算问题。
本工具的 CPU 测试覆盖原始字节、块表映射、部分块、混合 C8、缺失检查和 SSH 压缩筛选；
NPU 运行仍需在实际服务器验证。

## DEBUG 运行超时

`RPC call to sample_tokens timed out` 表示 EngineCore 没在执行 RPC 的期限内收到结果。
异步调度会先提交模型执行再提交采样，超时时 Worker 可能仍在执行前面的模型工作，
或采样的回读正在等待此前排队的 NPU hash；这条堆栈本身不能证明采样算法出错。
控制该期限的是 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`，当前默认 300 秒。
全历史 DEBUG 复现可以在**容器内的服务启动环境**临时设置更长的期限并重启 P：

```bash
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

这只延长等待期限。即便只读回 hash，完整历史仍需要逐层扫描和计算，不能视为零开销。
同一 DEBUG 开关下，后台 hash 任务连续 60 秒无进展会输出 `stage=STALL`：
包含等待的阶段、group/layer、队列长度，以及主线程和 hash 线程各最多 8 帧的 Python 栈摘要。
不安装信号处理器，不额外同步 NPU；每个停滞任务只输出一次。
分析报告的 `first_stall` 保留首条摘要；任务后来完成时，曾经停滞不等于 hash 不一致。
