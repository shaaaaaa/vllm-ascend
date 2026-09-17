# 单机 P/D：仅将 Mooncake SDK 存取替换为文件

在 `lmy_merge_prefill_layerwise_cache` 分支的 `vllm-ascend` 目录运行：

```bash
python -u tools/layerwise_prefill_file_check.py --output-tokens 256 2>&1 | tee log.log
```

不启动 Mooncake master、holder，不需要安装原生 Mooncake、指定 IP 或 YAML。
默认完整 GLM-5.2 权重、TP8、`max_model_len=16384`、
`gpu_memory_utilization=0.96`，只跑 P、D，不跑 baseline。
P、D（以及可选 baseline）均开启 MTP，`num_speculative_tokens=1`。
只有 P 使用 eager，D 使用 PIECEWISE，捕获大小为 `[1, 2]`，覆盖普通 decode 和 MTP 验证。
不支持多 token MTP：现有 layerwise prefill 协议要求每次 forward 最多经过一次 draft 层。
`--output-tokens` 是上限，不强制生成到该长度，EOS 可以提前结束。

沿用之前要求：脚本在开始模型运行前清理一次 `/dev/shm/*`。
请先停止同机使用共享内存的其他任务；被清理的数据不可恢复。
P/D 之间不会再次清理。

## 验证的位置

```text
P：原逐层 offload → LocalCPU → 原 MooncakestoreConnector 分页/拼接/等待
   → MooncakeDistributedStore.batch_put_from[_multi_buffers]
   → 测试替身写文件，保存原始 key 和原始字节
P 完成所有 store，进程退出
D：原 lookup/加载逻辑 → 原 MooncakestoreConnector
   → MooncakeDistributedStore.batch_get_into[_multi_buffers]
   → 测试替身读取相同 key，写回调用者提供的 CPU/NPU 缓冲区
   → 原来的后续计算
```

不是旧版 `ValidationFileConnector`：没有把整个远端 connector 换掉，
没有绕过 Mooncake key、page 或 multi-buffer 构造。
SDK 的 setup/注册接口也是测试替身，不建立网络连接。
观察原注册函数中的 tensor owner，用其真实 storage 解析传入的地址；
未知地址直接报错，不把 NPU 指针当 CPU 指针解引用。
直接读入 NPU 的路径在测试 SDK 内部使用同步本地 CPU→NPU 拷贝。
P 的 NPU→LocalCPU offload、后续层的 LocalCPU→NPU reload 不变。

父进程给此轮子进程设置私有启动钩子，覆盖 EngineCore、lookup 和 worker；
正常 `.sh` 启动不加载这些代码。原生产代码没有增加分支或检查。

## 输出和判定

开始时会打印结果目录 `layerwise-file-*`，其中：

- `prompt.txt`、`prompt.json`：原文章和实际提交的 token IDs。
- `prefill/output.txt`、`decode/output.txt`：P 的一个输出 token 和 D 的生成文本。
- 各阶段 `server.log`、`engine_options.json`、`lmcache_env.json`：日志和实际配置。
- `store/*.bin`：原始 KV 字节，头部包含原始 key、字节数和 SHA256。
- 各阶段 `store-io-*.jsonl`：实际 SDK 级 put/get/exists 调用。
- `store-sealed.json`：P 完成并退出后的存储清单。
- `summary.json`：put/get 次数、字节数、两个 KV group 的读取证据、模型输出和各阶段 MTP 统计。

日志中的 `[PREFILL_FILE] decode MTP:`、各阶段 `output.json` 的 `mtp` 字段和
`summary.json` 的 `mtp` 字段记录本次请求的 draft 验证次数、draft token 数、
接受 token 数、接受率。计数来自生成前后的 vLLM 指标差值，不包含模型初始化计数。
这些指标统计提交给目标模型验证的 draft，不是 draft forward 的总次数。
P 只生成一个输出 token，通常没有 MTP 验证；若 D 提前 EOS，
也可能没有验证，此时 `verification_observed=false`、接受率为 `null`，
不能认为已覆盖 MTP decode。不会为了凑验证次数强制忽略 EOS。

只有 D 实际通过 SDK get 读到两个 KV group、字节校验和匹配 P、
缓存命中覆盖预期前缀，才认为本轮走到了预期加载路径。
不凭“命中 token 数”一项宣称读回成功。
数值/模型输出是否正确仍需检查生成文本；文件字节一致不等于模型精度已通过。
需要输出基线时可以加 `--with-baseline`；只报告输出差异，不按微小误差中断。
默认没有逐层 tensor dump 或统计全模型 KV 的耗时后处理。

这不验证 Mooncake 原生传输、租约、网络注册、并发 RemoteFill、多请求 DP，
也不适合测性能。文件读写、校验和及本地同步拷贝均有额外测试开销。
