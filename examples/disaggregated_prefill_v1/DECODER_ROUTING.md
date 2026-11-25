# FalconKV D 路由验证

适用固定 RemoteFill 拓扑，每个 endpoint 对应一个引擎和独立 `/metrics`，每台物理机在 inventory 中只用一个地址。D 并发上限须一致。P 仍走原路由。

在 enhanced proxy 原命令上追加下列一组参数。三轮均设 `--request-count-slack 1 --decoder-route-capacity 16`（容量以实际 `--max-num-seqs` 为准）。

| 组别 | 参数 |
|---|---|
| A：原算法 + 同等观测 | `--decoder-route-monitor --decoder-route-policy legacy` |
| B：全体健康 D 中选 | `--decoder-route-policy global` |
| C：稳定跨机双候选 | `--decoder-route-policy dual-home` |

未启用实验时不增加轮询。实验独立并行抓取 D 指标（默认 2 秒一次），可与原有 `--disable-metrics-polling` 一起使用。`ROUTE_RUN` 中的 DP 编号为 decoder 列表下标，`hosts` 为各物理机的下标组。

## 如何保持同一会话的候选对稳定

回放端优先发送 `X-FalconKV-Session-Id: <opaque-session-id>`（最多 256 字节），同一逻辑会话的所有文件、所有轮次使用相同值。此头不代表有多少 KV 命中。若没有，则用客户端明确提供的 `x-lmcache-prefix-affinity` 最短锚点；否则对 chat 使用第一条 user 文本开头作为弱提示。公共开场白、历史截断可能产生碰撞或换键。模型名隔离路由键。对每个宿主机上的端口做 rendezvous hash，结果不依赖 endpoint 列表顺序。缺键的 dual-home 请求回退原算法。

## 何时改投

在候选内优先有效的前缀记录，其次参考近期成功服务过同一键的 D（有界、过期及 engine epoch 检查）。连续两次 waiting 非零视为拥塞，两次为零解除。KV 高水位或即将被预定打满的端点可提前改投。压力采用 `max(proxy_active, running + waiting + 仍在P的预定 + 抓取开始后的D交接数)`；交接边界短暂保守高估。候选对均无余量而其他 D 有空位时允许跳出候选对。D 端缺失、超龄或重复 gauge 不当零处理，回退原算法。这些选择只影响新分配。

## 内网汇总

保存回放开始与停止发新请求的带时区时间，以及单轮客户端 stdout（`bench.log`）。等在途请求结束后再收集本轮 proxy 日志。`ROUTE_DECISION` 记录预定瞬间候选/目标、原因和匿名负载；`--decoder-route-log-every 10` 默认十次打印一次，0 关闭细节。每约一分钟 `ROUTE_WINDOW` 聚合全部决策。`ROUTE_COMPLETE` 指 proxy 观察到的成功非流响应或带终止原因的正常流结束，不保证等于客户端成功数。新增诊断不记录 IP、请求正文和会话 ID。

```bash
python3 summarize_decoder_routes.py --proxy proxy_tp4_dp4.log --bench bench.log \
  --start '2026-11-25T16:00:00+08:00' --end '2026-11-25T17:00:00+08:00'
```

只分享输出 JSON。完成数按 proxy 收到请求的时间选出该轮请求，并按结束时间区分墙钟窗口与 drain。窗口计数只用完整落在时间范围内的 `ROUTE_WINDOW`，边界不按比例臆算。`valid_s` 是全体 D 同时有效的观测秒数，`skew_s` 是一台机的各 D 都满且排队、另一台有未满无队列且 KV 低于阈值的 D 的有效秒数。`null` 表示无足够数据。

客户端延迟来自单轮 stdout 成功行（含 drain）。也支持 JSON 列表或 `results`/`request_results` 数组中显式以 `ttft_ms`、`tpot_ms` 命名的字段。未知 JSON 结构不会猜单位，请改用 stdout。
