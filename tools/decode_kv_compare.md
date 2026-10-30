# 比较两轮 D 端收到的完整 KV

在 D 的服务启动环境设置 `LMCACHE_DECODE_KV_SNAPSHOT_DIR`，例如 `/workspace/fsi_lab/sqh/vllm-ascend/decode-kv-snapshots`。P 不开启。使用正常图模式，关闭旧 STORE_DEBUG、FENCE_DEBUG、AUDIT 后运行多轮测试。等待两个请求各 TP rank 的 `[D_KV_SNAPSHOT] ... complete`。

## 从两台 D 收集并比较

用同一 session 的两轮 response UUID 替换下面的两个参数。只需要 D 机器；传入四台机器也可以，没有对应请求的机器返回空集合。

```bash
python3 tools/decode_kv_compare.py \
  --previous 第一轮response的UUID \
  --current 第二轮response的UUID \
  --hosts root@7.150.3.60 root@7.150.6.177 \
  --workspace /workspace/fsi_lab/sqh/vllm-ascend \
  --password '你的密码' \
  --skip-host-key-check \
  --workers 8 \
  --output ./decode-kv-check
```

保存目录不在 workspace 下时加 `--snapshot-dir /实际绝对路径`；目录只在容器中可见时加 `--container 容器名`。密码也可用 `--password-env 环境变量名`。使用系统 OpenSSH/SSH_ASKPASS，不需要 sshpass。密码不会写进收集清单。

远端通过 gzip level 1 压缩流传输；本地下载完成后解包，自动删除临时压缩包。解包期间需要同时容纳压缩包和原始文件。一次收集可以复用；不要在文件仍写入时收集。

## 只比较已有本地文件

```bash
python3 tools/decode_kv_compare.py \
  --previous 第一轮response的UUID \
  --current 第二轮response的UUID \
  --local ./decode-kv-check/hosts \
  --workers 8 \
  --output ./decode-kv-recheck
```

`--local` 也接受直接复制下来的 `decode-kv-snapshots` 目录，可提供多个目录。不需要访问远端。

## 报告

- `report.json`：公共前缀 token 数、逐字节比较总量、差异总量、首批差异位置。
- `layers.jsonl`：所有层/组件的比较结果，以及当前轮新增区间的统计。
- `received_history_identical`：公共 token 前缀的接收数据完全相同。仍需检查后续 H2D、稀疏映射和计算。
- `history_differs`：至少一个原始字节不同，查看 TP/group/layer/token/byte 定位。它是差异证据，不自动认定差异原因。
- `incomplete`：文件/rank/布局/收集有缺失或不兼容，不能得出一致性结论。

`current_uncompared_range` 是当前轮未与旧快照对比的范围；其浮点非有限值/零字节统计只是线索。不同会话、不同前缀或不同模型不适合作为两轮对照。脚本不会把新增 KV 没有旧参考值解释为正确。
