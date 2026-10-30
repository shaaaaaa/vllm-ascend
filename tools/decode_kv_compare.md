# 比较两轮 D 端收到的完整 KV

在 D 的服务启动环境设置 `LMCACHE_DECODE_KV_SNAPSHOT=1`。无需配置保存目录，会自动在实际加载的 vllm-ascend 根目录下创建 `decode-kv-snapshots`，不受启动命令工作目录影响。P 不开启。使用正常图模式，关闭旧 STORE_DEBUG、FENCE_DEBUG、AUDIT 后运行多轮测试。等待两个请求各 TP rank 的 `[D_KV_SNAPSHOT] ... complete`。

## 在两台 D 离线计算分段校验值并比较（默认）

用同一 session 的两轮 response UUID 替换下面的两个参数。只需要 D 机器；传入四台机器也可以，没有对应请求的机器返回空集合。计算发生在请求完成后，读取已经保存的文件，不加载模型、不调用 NPU、不更改服务端诊断开关。

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

脚本自动在 workspace 或其 `vllm-ascend` 子目录下寻找 `decode-kv-snapshots`，不需要传 `--snapshot-dir`。`workspace` 是执行远端命令所在环境的路径：直接 SSH 时使用宿主机路径；加 `--container 容器名` 时使用容器内路径，例如 `/workspace/sqh/vllm-ascend`。密码也可用 `--password-env 环境变量名`。使用系统 OpenSSH/SSH_ASKPASS，不需要 sshpass。密码不会写进结果文件。

默认处理流程：

1. 读取两轮的元数据和真实 token ID，确定共同前缀，检查所有 TP rank、层和组件是否完整。
2. 在各 D 上对每个 rank、每层、每个组件的共同前缀计算 BLAKE2b-128 校验值。默认每段 4096 tokens，覆盖段内所有原始字节，不是抽样。最后不足一段的部分按两轮相同的边界计算，新增后缀不混入比较。
3. 校验值相同的段不传原始数据。不同的段如果在同一 D 上，直接在该 D 逐字节比较，只返回差异位置和计数。
4. 不同的段分散在两个 D 上时，只下载这些段的两份数据做逐字节比较。默认所有原始下载数据合计不超过 64 MiB，不按请求/rank 分别重复计算额度。元数据、校验清单、差异结果另外传输，均经过 gzip level 1 压缩。

远端只需 Python 3.9 或更新版本的标准库，不必安装模型依赖或更新远端脚本；客户端会发送所需代码。同机逐字节比较可以使用已有 NumPy 加速，无 NumPy 时使用标准库实现。`--workers` 控制每次扫描/比较的文件线程数，默认 4。扫描仍需在 D 本地完整读取共同前缀，因此磁盘读取时间不会消失，但几百 GB 的 KV 不再跨网络传输。

可选参数：

```text
--segment-tokens 4096   每个校验段的 token 数
--max-diff-mib 64       跨机器原始差异下载额度，两份数据合计
--max-diff-mib 0        禁止跨机器原始下载，仍定位所有校验值不同的段
--full-transfer        显式恢复旧模式：下载两轮所有完整文件
```

差异很多、超过下载额度时，脚本仍报告所有不一致段，并把未逐字节比较的数量记为 `unfetched_differing_segments`。`different_bytes` 为 `null`，不会把部分数据的差异数量当成全部；`confirmed_different_bytes` 是已逐字节比较部分的计数。增大额度并使用相同的 `--output` 重跑，可以复用已经下载的差异块。

校验清单保存在 `checksums/`，差异块保存在 `diff-blocks/`。重复原命令会检查远端文件的大小、时间和文件标识，只复用来源未变化的校验结果；下载块还会校验长度和校验值。文件仍在写入、缺失或分析期间变化，会返回 `incomplete`。

相同校验值是校验和一致性证据，有理论碰撞可能，不宣称已把匹配段逐字节传回比较。所有不一致段都会被定位；新后缀、后续 NPU 加载和模型计算仍不在这一结论范围内。

## 只比较已有本地文件

```bash
python3 tools/decode_kv_compare.py \
  --previous 第一轮response的UUID \
  --current 第二轮response的UUID \
  --local ./decode-kv-check/hosts \
  --workers 8 \
  --output ./decode-kv-recheck
```

`--local` 也接受直接复制下来的 `decode-kv-snapshots` 目录，可提供多个目录。不需要访问远端。它需要完整原始快照；默认远端模式产生的校验清单和少量差异块不能作为完整快照传给 `--local`，也不能用于 D 重放。默认远端模式复用结果时，直接重复原命令即可。

## 报告

- `report.json`：共同前缀、校验段数量、实际压缩响应字节数、差异下载额度使用量及差异位置。
- `layers.jsonl`：所有层/组件的比较结果。远端模式包括不一致段和已确认差异计数；完整本地模式还统计新增区间的零值/非有限值。
- `segments.jsonl`：远端模式所有不一致段的 rank/group/layer/component、token 范围，以及是否完成逐字节比较。
- `history_checksums_match`：默认远端模式共同前缀的所有段校验值一致。
- `received_history_identical`：完整本地/完整下载模式已逐字节确认共同前缀一致。
- `history_differs`：至少一个原始字节不同，查看 TP/group/layer/token/byte 定位。它是差异证据，不自动认定差异原因。
- `incomplete`：文件/rank/布局/收集有缺失或不兼容，不能得出一致性结论。

`current_uncompared_range` 是当前轮未与旧快照对比的范围；其浮点非有限值/零字节统计只是线索。不同会话、不同前缀或不同模型不适合作为两轮对照。脚本不会把新增 KV 没有旧参考值解释为正确。

`compressed_response_bytes` 是本次接收的压缩响应内容总量，不包括 SSH 协议本身的开销。远端模式的 `checksummed_bytes_per_snapshot` 是每份快照参与比较的原始字节量，不是网络传输量。`difference_counts_complete` 仅表示所有校验值不同的段是否均已完成逐字节比较。

## 只检查重新计算尾部的数值

知道当前轮的外部缓存命中结束位置时，可以直接指定 `--inspect-start`。例如命中 499712 tokens、当前 prompt 为 500084 tokens，则检查末尾的 372 tokens：

```bash
python3 tools/decode_kv_compare.py \
  --previous dce27753-310a-445c-8ac1-7c42aa22a67a \
  --current 21c69b01-12a2-4976-89ba-144a6270aad0 \
  --hosts root@7.150.3.60 root@7.150.6.177 \
  --workspace /workspace/fsi_lab/sqh/vllm-ascend \
  --password '你的密码' --skip-host-key-check \
  --workers 8 --inspect-start 499712 \
  --output ./compare-s0-r0-r1
```

这个模式只读取选中的尾部，不重新计算整个 500k 前缀的校验值，也不下载原始 KV。它需要 `--hosts`，不能与 `--full-transfer` 混用。不覆盖已有的 `report.json`、`layers.jsonl` 和校验清单，而是写入 `tail_report.json`、`tail_layers.jsonl`。

所有 rank、层、组件分别统计零值比例、全零行、NaN/正负 Inf、均值、标准差、RMS、最小/最大值、相邻相同行数及是否所有行相同。统计根据文件 dtype 和端序解码，支持 BF16、FP16、FP32、FP64、int8 和 uint8；C8 key 和 scale 分别统计，不做反量化。读文件按块进行，内存不会随检查范围无限增长。

两轮共同 token 区间和真正新增的区间单独统计。例如旧轮共 499994 tokens，则 `[499712,499994)` 的 282 tokens 有旧参考，`[499994,500084)` 的 90 tokens 没有。两份快照在同一台 D 时，还在 D 上对相同 token 计算 RMSE、相对 L2、最大/平均绝对误差、余弦相似度及新出现的非有限值；如果分处不同 D，仍返回两边的分布，但明确标记没有逐元素数值对比，不自动传输大块数据。

相对 L2 是差值的 L2 范数除以旧参考的 L2 范数；旧参考范数为零时记为 `null`。非有限值单独计数，误差指标只使用两边都有限的位置。不设置数值误差通过阈值；全零、重复行等是定位线索，未必单独构成错误。正常分布也不能证明 attention、索引映射或模型输出正确。

终端摘要包含首次出现数值差异的层及 RMS/相对 L2，以及最大全局相对 L2。完整逐组件统计保存在 `tail_layers.jsonl`，无需把整个文件贴回。
