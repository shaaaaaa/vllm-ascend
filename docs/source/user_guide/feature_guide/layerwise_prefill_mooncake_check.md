# 单机顺序验证 Mooncake 持久化

`tools/layerwise_prefill_mooncake_check.py` 使用完整 GLM-5.2、单机 TP8，
`max_model_len=16384`、显存利用率 `0.96`。不开 profiler，不安装 KV debug
hooks，不开 MTP。只有 P 使用 eager，baseline/D 使用普通 PIECEWISE 图。
**默认只跑 P → D，跳过 baseline**，避免为当前的持久化/加载排查额外加载一次模型。
需要 baseline 输出对照时，再显式加 `--with-baseline`；它不会自动启用 KV dump。

不需要 YAML 或 `--config`。脚本根据此前的部署配置，在各子进程设置 `LMCACHE_*`
环境变量：chunk size 1024、NUMA interleave、passive writable、lookup timeout
30000 ms、pin timeout 1800 s、两组 page-first/merged-page、Ascend 传输，
transfer timeout 120 s。P 使用 sender、异步保存；D 使用 receiver 和
`persistent_direct_hbm`。

默认**自动启动本机独立 Mooncake master**，客户端连接 `127.0.0.1:<自动选择的空闲端口>`，
不再依赖远端服务。RPC 和 metrics/admin 各用独立空闲端口，不占用其他任务的固定端口。
需要本机已经安装 `mooncake_master` 可执行文件；脚本从 PATH 或 Python 所在目录查找，
不在这些位置时可以传 `--master-bin /path/to/mooncake_master`。找不到会在加载模型前报错。

Ascend 传输仍使用本机网卡 IP，而非 master 的 loopback 地址；默认通过本机路由表
识别地址（不向外部发送探测包），也可用 `--local-hostname 本机IP` 指定。
无需提供 YAML，原来的 `--output-tokens` 参数不变。

只有显式传 `--master IP:端口` 时才连接已有 master，且不会启停它。
脚本启动时会先清理一次 `/dev/shm/*`（等价于 `rm -rf /dev/shm/*`），
再启动 master、holder 和模型。子进程和 P → D 切换时不重复清理。
**删除无法恢复，运行前必须停止同一共享内存命名空间中的其他任务**；
若容器使用 `--ipc=host`，也会影响宿主机共享内存。脚本不会停止其他任务的 Mooncake 进程。

```bash
python -u tools/layerwise_prefill_mooncake_check.py 2>&1 | tee log.log
```

脚本顺序执行：

1. 清理 `/dev/shm/*`；失败即停止。随后自动启动本地 master，等待监听就绪，
   打印 `local master ready: 127.0.0.1:端口`。
2. 启动独立的 Mooncake CPU 存储进程（默认 8 GiB），一直保留到 D 完成。
3. 仅指定 `--with-baseline` 时：先关闭 layerwise P offload，生成并保存 baseline 输出；
   不向 Mooncake 写入。默认跳过此步骤。
4. P：启用 layerwise P offload，逐层保存到本地 CPU，异步保存到 Mooncake。
   生成一个 token 后完成最终持久化屏障，关闭 P 及其全部 worker。
5. D：重新创建本地 LMCache，从 Mooncake 加载 P 的 KV 后生成输出。
6. 输出统计后，先关闭 holder，再关闭本次启动的 master。中途失败也按此顺序清理。

本地 master 异常退出时，launcher 会停止正在运行的模型阶段；不会一直等客户端重连。
这些存活检查在 launcher 中执行，不在模型逐层计算或 KV 传输回调里。

holder 虽然保存的是 CPU 内存，但 `ascend` 传输仍需要 NPU 上下文。脚本在
Mooncake 初始化前选择 `--devices` 中的第一张卡（进程内逻辑编号 0）并初始化
NPU，启动后会打印 `holder Ascend context ready`。这不会加载模型权重，但会有
额外的设备上下文/传输资源占用；该进程存活到 D 完成后才退出。

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
- `master/server.log`、`master/process.json`：本次本地 master 的日志、PID、地址和启动命令。
- `prefill/`、`decode/` 的 `output.txt`、`output.json`、`server.log`；
  `baseline/` 仅在 `--with-baseline` 时创建。
- 各阶段实际设置的 `lmcache_env.json` 和 `engine_options.json`；仅是记录，无须提供输入配置文件。
- `summary.json`：默认记录 `baseline_ran=false`、P 缓存命中数量、D 命中的持久化前缀长度，
  不生成 baseline 对比结论。指定 `--with-baseline` 时，才增加 P 首 token 与 baseline 的比较、
  D 首次 token 分歧位置。

默认最多生成 256 tokens，允许 EOS 提前结束，并非强制生成 256 个。
开启 baseline 对比后，输出不同只记录，不因此中止；P 命中旧缓存或 D 没有加载到所需缓存时则报错，
防止把重新计算 prompt 当成“PD 验证通过”。逐层 KV 数值比较仍使用原来的
`layerwise_prefill_check.py`；它使用文件归档，不能替代这次 Mooncake 路径验证。

## 验证边界

顺序测试时 D 在 P 退出后才存在，所以没有同时在线的 RemoteFill 握手/提前推送。
默认测试检查本次输入的 P 持久化和 D 重载/生成路径，不证明输出数值正确；
只有显式运行 baseline 才会生成输出对比。
**不能推导生产环境剩下的唯一风险是时序问题**。还需覆盖同时在线的 RemoteFill、
跨机网络、不同 TP/DP、并发、MTP、失败/重试及更长上下文。

本地 master 的参数参考 [Mooncake 部署说明](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md)。
