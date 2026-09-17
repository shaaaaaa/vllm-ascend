# 单机顺序验证 Mooncake 持久化

`tools/layerwise_prefill_mooncake_check.py` 使用完整 GLM-5.2、单机 TP8，
`max_model_len=16384`、显存利用率 `0.96`。不开 profiler，不安装 KV debug
hooks，不开 MTP。只有 P 使用 eager，baseline/D 使用普通 PIECEWISE 图。

不需要 YAML 或 `--config`。脚本根据此前的部署配置，在各子进程设置 `LMCACHE_*`
环境变量：chunk size 1024、NUMA interleave、passive writable、lookup timeout
30000 ms、pin timeout 1800 s、两组 page-first/merged-page、Ascend 传输，
transfer timeout 120 s。P 使用 sender、异步保存；D 使用 receiver 和
`persistent_direct_hbm`。

Mooncake master 默认沿用 **`7.150.4.174:58888`，必须已经启动**，本机地址根据到
master 的路由自动识别，不能把 master IP 当成本机 IP。脚本不启动或停止已有
master、不清理 `/dev/shm`。建议使用隔离的测试 master；需要换地址时可选传
`--master IP:端口`，无须编辑配置文件。

```bash
python -u tools/layerwise_prefill_mooncake_check.py 2>&1 | tee log.log
```

脚本顺序执行：

1. 启动独立的 Mooncake CPU 存储进程（默认 8 GiB），一直保留到 D 完成。
2. baseline：关闭 layerwise P offload，生成并保存参考输出；不向 Mooncake 写入。
3. P：启用 layerwise P offload，逐层保存到本地 CPU，异步保存到 Mooncake。
   生成一个 token 后完成最终持久化屏障，关闭 P 及其全部 worker。
4. D：重新创建本地 LMCache，从 Mooncake 加载 P 的 KV 后生成输出。

P/D 的 Mooncake `global_segment_size` 都设为 0，避免数据落在会随 P 退出而
注销的 segment。真正的数据由独立存储进程持有，而不是只保留 master 的元数据。
每轮在文章开头添加唯一标记，避免旧缓存跳过本轮 P 的计算。
本地 CPU cache 和独立存储 segment 默认各 8 GiB，未照搬多机配置中的 95 GB/100 GB。
P 的 NPU 直传开关仍置为 true，以验证 bank-safe warning 和逐层 CPU 保存覆盖确实生效；
不进行对旧 NPU bank 的远端读取。

这不是“保留 P 的 LMCache 不退出”：当前 LocalCPU allocator 隶属于模型 worker，
没有在这里增加独立 LMCache daemon。D 使用全新 CPU cache，可以检查跨进程重载。

日志开头打印结果目录 `layerwise-mooncake-*`，包含：

- `prompt.txt`、`prompt.json`：实际文章及 token IDs。
- `baseline/`、`prefill/`、`decode/` 的 `output.txt`、`output.json`、`server.log`。
- 各阶段实际设置的 `lmcache_env.json` 和 `engine_options.json`；仅是记录，无须提供输入配置文件。
- `summary.json`：P 首 token 与 baseline 的比较、D 首次 token 分歧位置，
  以及 D 命中的持久化前缀长度。

默认最多生成 256 tokens，允许 EOS 提前结束，并非强制生成 256 个。
数值/输出不同只记录，不因此中止；P 命中旧缓存或 D 没有加载到所需缓存时则报错，
防止把重新计算 prompt 当成“PD 验证通过”。逐层 KV 数值比较仍使用原来的
`layerwise_prefill_check.py`；它使用文件归档，不能替代这次 Mooncake 路径验证。

## 验证边界

顺序测试时 D 在 P 退出后才存在，所以没有同时在线的 RemoteFill 握手/提前推送。
通过这个测试只能说明本次输入的 P 持久化和 D 重载/输出路径通过了对照，
**不能推导生产环境剩下的唯一风险是时序问题**。还需覆盖同时在线的 RemoteFill、
跨机网络、不同 TP/DP、并发、MTP、失败/重试及更长上下文。
