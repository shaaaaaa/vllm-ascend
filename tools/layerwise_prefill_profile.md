# 用 LoCoMo 抓取 OFF / ON profile

在空闲的 Linux Ascend 单机上运行：

```bash
set -o pipefail
python3 tools/layerwise_prefill_profile.py 2>&1 | tee log.log
```

默认依次启动两个独立 API server：关闭、开启 layerwise prefill。
每个 server 就绪后开启 profiler，再执行当前 Python 环境下的命令：

```bash
python /workspace/dataset/benchmark-new/locomo/test_advanced.py --vllm_port 8000 --vllm_ip 127.0.0.1
```

不再构造测试 prompt。输入、请求数量、输出长度由 LoCoMo 脚本决定。
benchmark 在它自身的目录下运行，以兼容相对路径数据集；标准输出/错误分别保存到 OFF、ON 的 `benchmark.log`。
服务启动不计入 profile；整个 benchmark 的 prefill、decode 都会抓取。
测试结束后停止 profiler、退出 server 并离线导出 trace，然后运行另一轮。
若一轮失败，保留日志和结果状态并停止，不自动继续下一轮。

模型默认 `/workspace/models/GLM-5.2-w4a8c8-0723`。
沿用 TP8/DP1、MTP1、eager、FlashComm1=1、显存比例 0.97、max-model-len=84096、
compute chunk=4096、LMCache chunk=1024、CPU 缓存 24 GiB。
参数均在脚本内；无外部配置文件、Mooncake、tensor dump。
这是单机本地缓存的性能抓取，不是四机 P→D 正确性验证。
profiler 本身会增加开销，benchmark 耗时不代表未抓取时的性能。

输出到新的 `layerwise-profile-*/`：

```text
model_info.json
off/server.log
off/benchmark.log
off/engine_options.json
off/environment.json
off/benchmark_command.json
off/result.json
off/profile/
off/traces.json
on/                         # 同样的目录结构
```

可选参数：

```bash
python3 tools/layerwise_prefill_profile.py --case on
python3 tools/layerwise_prefill_profile.py --model /path/to/model --port 8001
python3 tools/layerwise_prefill_profile.py --benchmark-script /path/to/test_advanced.py
python3 tools/layerwise_prefill_profile.py --analyse-only /path/to/layerwise-profile-run
```

默认服务启动超时 1800 秒、benchmark 总时限 7200 秒，可以分别用
`--startup-timeout`、`--benchmark-timeout` 调整；它们不修改 Worker 的执行超时。
脚本沿用启动前清理 `/dev/shm/*` 的行为，OFF server 退出后才会为 ON 清理。
因此请使用没有其他模型实例运行的单机。
