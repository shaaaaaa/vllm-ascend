# 四机 PD 原始 tensor 诊断

本功能旁路记录真实四机 PD 服务的数据，保留 Mooncake 和 LMCache 的生产传输路径。
同时更新 `prefill_layerwise_cache` 分支的 vLLM 和 vLLM-Ascend。
开关默认为空；未开启时不安装探针、不写文件、不增加 tensor 读回。

## 1. 修改现有启动 template

在 `templates/run_vllm_lmcache_tp4_dp4.sh.j4` 的 P/D 公共环境变量部分增加：

```bash
# 四台机器使用相同 run 名；每次新实验换一个目录，避免混入旧请求。
export VLLM_ASCEND_PD_TENSOR_DUMP_DIR=/workspace/sqh/vllm-ascend/pd-tensor-dump/case-on
export VLLM_ASCEND_SFA_STAGED_GRAPH=0
export VLLM_ASCEND_SFA_STAGED_MTP_DRAFT_GRAPH=0
# 原始 tensor 写盘较慢，诊断期间放宽单次 worker 执行超时。
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
```

P、D 两条 `vllm serve` 命令均设置以下参数。已有参数直接替换，不要重复追加相冲突的 JSON：

```bash
--enforce-eager \
--compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
--no-async-scheduling \
```

把两边已有 `--additional-config` JSON 中的 `enable_npugraph_ex` 改为 `false`，保留其他字段。
这仅是逐层记录所需的诊断设置，模型、TP/DP/EP、显存利用率、长度、chunk 大小和 MTP 参数保持原值。
记录当前支持的主模型是本分支的 GLM/DeepSeek DSA/SFA 路径；启动会检查层清单，不能覆盖时直接报错。
要求 PP=PCP=DCP=1，支持 TP/DP/EP 以及 FlashComm1；不支持独立 FREE_PAGED latent pool 路径。

仅在 **D 的 serve 命令**增加下面参数（已有 JSON 时合并字段）：

```bash
--override-generation-config '{"max_new_tokens":16}' \
```

它把每个请求最终生成的 token 上限设为 16。P 的代理请求仍保持 `max_tokens=1`。
启用 MTP 时，16 个输出 token 不等于 16 次模型 forward；EOS 也可能使输出提前结束。
不要把 `max-model-len` 或 `max-num-seqs` 改成 16 来限制生成长度。
诊断请求使用 `n=1`、文本 token 输入。

这个诊断开关和 `VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE` 相互独立：

- ON 实验：P 保留原来的 layerwise 开关；D 仍关闭该开关。
- OFF 实验：按你原来的 OFF 配置关闭 layerwise，四台机器仍设置上面的诊断目录，改名 `case-off`。
- 结束诊断：取消 `VLLM_ASCEND_PD_TENSOR_DUMP_DIR`，恢复原来的图和调度参数。

建议先发一条短请求，等请求完成、worker 完成清理后再拉文件。若进程中途退出，未完成的目录仍可分析已有数据，报告不会将它当成完整覆盖。

## 2. 保存的数据和 request ID

每个 worker 各写自己的文件，不共享写入一个索引：

```text
pd-tensor-dump/case-on/
  P或D/<外部request_id>/<host>-dp<dp>-tp<tp>-pid<pid>/
    manifest.json
    calls/0.json ...
    index.jsonl
    tensors/00000000.pt ...
    sampled.jsonl
```

请求目录使用可逆的 URL 编码以避免 request ID 中的路径字符。
`manifest.json` 同时保存外部 ID 和引擎内部 ID。
正常同一次代理转发的 P/D 外部 ID 相同；两个引擎内部生成的随机后缀不同，不能直接匹配。
本修改把已有外部 ID 传到 worker，不改变调度或代理的 ID 生成规则。
代理重试/重选实例可能生成新 ID，单独实验的 OFF/ON 也通常有不同 ID，不自动猜测它们是同一个请求。

逐次保存真实原始 tensor，不计算指纹：

- 每个请求实际进入模型的 token、逻辑 position、该次因果上下文；每个 chunk 独立记录。
- 每层 decoder/SFA 输入输出、残差、attention query、真实 indexer/top-k、logits。
- P/D 当前 token 写入的 KV、indexer 读取的 KV/scale，以及 attention 实际消费的 KV。
- 采样后被接受的 token；未完成 prefill 的丢弃采样和 MTP 的 padding 不计作输出。

`sampled.jsonl` 记录 worker 的接受结果，处于引擎最终按 EOS/停止词/长度裁剪之前；MTP 最后一步可能多产生少量接受 token。
它不是 HTTP 响应文本的副本，服务对外输出仍受上面的 16 token 上限约束。

按请求切分 batch，TP padding 不参与数值比较。保留逻辑 position 与物理映射证据；不会直接拿 P/D 的物理 block 编号作相等判断。
共享 indexer 层记录实际共享 top-k，不额外运行 indexer。
记录覆盖主模型与 MTP target verification，**不覆盖 MTP draft 模型内部**。

原始 tensor 读回和写盘会明显减速、占用大量磁盘，也可能改变异步竞争的时序。此模式用于数据排查，不能代表性能，也不能排除仅在图回放或原始并发时序下出现的问题。

## 3. 拉取四台机器的文件

在有 SSH 访问权限的机器上，从 vllm-ascend 目录执行：

```bash
python3 tools/pd_tensor_collect.py \
  --hosts root@7.150.4.174 root@7.150.5.55 root@7.150.5.81 root@7.150.1.46 \
  --repo-path /workspace/sqh/vllm-ascend \
  --run-id case-on \
  --output ./collected-case-on \
  --analyze-pd
```

`--repo-path` 填实际部署路径；若是 `/workspace/lmy/vllm-ascend`，就替换为该路径。
工具读取 `<repo-path>/pd-tensor-dump/<run-id>`。如果目录只在容器中可见，追加：

```bash
--container vllm-ascend-v0.18.0rc1-lmcache-ascend-v0.4.3-fsi-sqh
```

使用系统 SSH 的密钥/配置和主机校验；可传 `--ssh-port` 和 `--identity-file`。
不会改远端文件。不同机器路径/容器不同时分别运行到不同本地目录，分析脚本支持多个根目录。
`collection.json` 标明每台机器是否成功；失败返回非零状态，重跑同一命令仅重试未完成项。
`--analyze-pd` 在成功拉取四机后自动分析 P→D KV，报告写到 `collected-case-on/report-pd`，执行端需要 CPU PyTorch。
只拉文件时去掉该参数，不需要 PyTorch；文件已拉取后加回该参数也不会重复下载。

## 4. 不需要 OFF：先检查 P → D 的 KV

若采集时没有使用 `--analyze-pd`，可在装有 CPU PyTorch 的机器上单独运行：

```bash
python3 tools/pd_tensor_analyze.py \
  --mode pd-kv \
  --reference ./collected-case-on \
  --candidate ./collected-case-on \
  --output ./report-pd
```

按外部 request ID、相同 TP rank、层和逻辑 token 位置匹配。P/D 的 DP rank 可以不同。
比较 P 算出的 prompt KV 与 D 实际消费的对应 KV，以及 indexer KV；不拿 P 的 prefill 输出与 D 的不同 token 的 decode 输出硬比。
若 D 读到的 prompt KV 与 P 不同，可以缩小到保存/传输/加载/映射这条链路；单靠两端文件不能继续断定是 Mooncake 还是某个搬运算子。
没有被 D 消费的 P KV 单独标记，不当作传输丢失。
如果 P 命中已有 prefix cache、因此本次未计算完整 prompt，该部分会标为缺少 P 参考数据，不会宣称 KV 已全部核验。

## 5. 有 OFF：定位层内计算从哪里开始不同

关闭 layerwise 再采集同一输入，保存到 `case-off` 并拉取到 `collected-case-off`。
如果两次外部 ID 不同，写一个明确映射：

```json
{"cmpl-OFF请求ID-0":"cmpl-ON请求ID-0"}
```

例如文件名 `request-map.json`，然后执行：

```bash
python3 tools/pd_tensor_analyze.py \
  --mode off-on \
  --reference ./collected-case-off \
  --candidate ./collected-case-on \
  --request-map request-map.json \
  --output ./report-off-on
```

同 ID 时省略 `--request-map`。相同角色、TP rank、逻辑 token 和因果上下文才做数值比较；不会因 chunk 编号不同就错配。
一旦生成 token 分叉，后续输入上下文不同的数据标记为不可直接比较，不能将不同输入产生的差异直接归咎于 KV。

`report.json` 是简明汇总，`comparisons.jsonl` 给出每项证据。统计均来自原始 tensor，包括原始均值/标准差/RMS、绝对差、RMSE、相对 L2、新增非有限值。
OFF/ON 还报告 worker 接受的输出 token 序列及首个分叉位置。`first_observed_difference` 指最早观察到差异的计算位置，不直接等于已确认的根因。
数值差异与覆盖完整性分开：`analysis_complete` 表示分析完整，不是精度 PASS；`accuracy_verdict` 明确为 `not_assessed`。
缺 rank、缺文件、未完成、上下文不同会明确列出。先看 P→D KV，再结合 OFF/ON 的最早观察差异与后续误差放大判断位置。
