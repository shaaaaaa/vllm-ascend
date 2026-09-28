# 四机 PD 原始 tensor 诊断

本功能旁路记录真实四机 PD 服务的数据，保留 Mooncake 和 LMCache 的生产传输路径。
同时更新 `prefill_layerwise_cache` 分支的 vLLM 和 vLLM-Ascend。
开关默认为空；未开启时不安装探针、不写文件、不增加 tensor 读回。

## 先做单机预检查

在安装好四个仓库的 Linux Ascend 单机环境中执行，不需要配置文件或 Mooncake 服务：

```bash
set -o pipefail
python3 tools/pd_tensor_smoke.py 2>&1 | tee log.log
```

默认模型是 `/workspace/models/GLM-5.2-w4a8c8-0723`，使用 0–7 卡、TP8/DP1、MTP1，
约 4608 输入 token、4096 compute chunk、1024 LMCache chunk，D 生成 16 token。
`max-model-len=16384`、显存利用率 0.97、CPU 缓存 24 GiB，P 的 FlashComm1=1、D=0；两端均 eager。
模型路径不同时用 `--model /实际模型目录`，例如部署的 GLM-5.3。

默认顺序运行四个独立模型实例：

1. P 关闭 layerwise、开启新抓取开关，运行完整 prefill，保存数据并退出。
2. D 开启抓取，从上一步文件还原 KV，完成 16 token 输出，然后退出。
3. P 开启 layerwise 和抓取，重新运行相同 prompt，使用独立的文件存储并退出。
4. D 从 ON 的文件还原 KV，完成 16 token 输出，然后退出。

本机依次复用同一组卡。仅将 Mooncake SDK 换成已有文件后端，仍走实际 LMCache key/page 和搬运代码。
抓取来自新的 `VLLM_ASCEND_PD_TENSOR_DUMP_DIR` 生产探针，不安装旧 correctness 探针。
单机脚本沿用 profile 的内置参数和环境变量方式，清除继承的 `LMCACHE_CONFIG_FILE`，无需准备 YAML。
LMCache 的 `No LMCache configuration file is set` 提示表示使用环境变量，并非要求补配置文件。
探针在 KV connector 初始化后安装，避免提前读取尚未由 LMCache-Ascend 扩展的配置类。
检查每个 TP 的归档完整性、P 多 chunk 覆盖、D decode 覆盖、D 缓存命中长度 `prompt长度−1`，
以及两个 DSA group 的实际文件读取；最后自动运行 P→D KV 和 OFF/ON 离线比对。
它不覆盖真实 Mooncake 网络、多机 DP、图回放或没有读回时的并发竞争。

结果写到新建的 `pd-tensor-smoke-*/`：

```text
report.json                         # 总体执行/覆盖检查
off/prefill/server.log              # 各实例原始日志
off/decode/server.log
on/prefill/server.log
on/decode/server.log
off/capture/P/0/...                 # 按 request ID / TP 保存的新格式原始 tensor
off/capture/D/0/...
on/capture/P/0/...
on/capture/D/0/...
off/report-pd/report.json           # OFF P→D KV 比较
on/report-pd/report.json            # ON P→D KV 比较
report-off-on/report.json           # OFF/ON 逐层比较
```

`smoke_passed=true` 表示执行、覆盖、传输和输出 token 检查通过，不等于所有浮点 tensor 逐 bit 相等或无并发 bug。
原始 tensor 占用较多磁盘并且明显减速；没有安装模型/NPU 的机器仅能运行 CPU 回归或 `--dry-run`。
不会启动后再清理全局 `/dev/shm`；实例退出时仅关闭自己的 LMCache 资源。

可选命令：

```bash
# 只验证 ON 的 P、D，减少两次模型加载
python3 tools/pd_tensor_smoke.py --case on 2>&1 | tee log.log

# 很短的 request，用来单独覆盖不足一个 LMCache chunk 的情况
python3 tools/pd_tensor_smoke.py --case on --prompt '你好，请介绍一下你自己' 2>&1 | tee log.log

# 不加载模型，检查即将使用的参数
python3 tools/pd_tensor_smoke.py --dry-run

# 重跑已有文件分析，不重新启动模型
python3 tools/pd_tensor_smoke.py --analyze-only /实际/pd-tensor-smoke-目录
```

## 1. 修改现有启动 template

在 `templates/run_vllm_lmcache_tp4_dp4.sh.j4` 的 P/D 公共环境变量部分增加：

```bash
# 四台机器使用相同 run 名；每次新实验换一个目录，避免混入旧请求。
export VLLM_ASCEND_PD_TENSOR_DUMP_DIR=/workspace/sqh/vllm-ascend/pd-tensor-dump/case-on
# 默认已是 64，可省略：每个固定 4096-token 段只抓少量行。
export VLLM_ASCEND_PD_TENSOR_DUMP_MAX_TOKENS=64
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

默认使用固定位置抽样，避免长 prompt 全量读回/写盘：

- KV 按 **request 内绝对位置**划分 4096-token 段，每段保存前最多 64 行。
  例如长度 8192 保存 `[0,64)`、`[4096,4160)`；长度 8200 再保存 `[8192,8200)`。
  P 的当前 KV、历史 indexer KV 和 D 实际消费的 KV 使用同一组位置，跨 chunk 调度也不重新编号。
  attention 只保存上述位置中实际被选中的 KV；没有消费的位置不会伪造记录。
- 输入输出/query/top-k 按每个 TP 实际可见的段保存前最多 64 行；D 当前计算的一两行也保留。
  因此不会因 TP 分片从 1024 开始，或 decode position 不落在 KV 采样窗口内，而完全漏掉模型计算。
- 截取发生在 KV gather 和 CPU 读回之前；hidden/head/vocab 维度保留完整。
  完整 token/position/因果上下文等小型逻辑元数据、MTP 接受判断和最终 token 不截断。
- `index.jsonl` 每条记录的 `row_capture.segments` 给出段起点、实际 `length`、
  `source_rows`、`saved_rows` 和省略范围；不足 4096 或不足 64 都按实际数量记录。
  `source_indices` 给出采样行在原始张量中的下标。物理槽别名也受每段行数上限约束。
- 四台机器默认一致，无需新增参数即可生效；`VLLM_ASCEND_PD_TENSOR_DUMP_MAX_TOKENS=0`
  才恢复全量。自定义值须在 0–4096，P、D 应保持一致。固定 4096 是诊断分段，
  不改变模型的调度 chunk，也不改变 LMCache/Mooncake 传输数据。

`sampled.jsonl` 记录 worker 的接受结果，处于引擎最终按 EOS/停止词/长度裁剪之前；MTP 最后一步可能多产生少量接受 token。
它不是 HTTP 响应文本的副本，服务对外输出仍受上面的 16 token 上限约束。

按请求切分 batch，TP padding 不参与数值比较。保留逻辑 position 与物理映射证据；不会直接拿 P/D 的物理 block 编号作相等判断。
共享 indexer 层记录实际共享 top-k，不额外运行 indexer。
记录覆盖主模型、MTP target verification 和 **MTP draft 模型内部**：输入 token、
原始 positions、分片前逻辑 positions、输入/输出 hidden states、各层 SFA/indexer、
当前及实际消费的 KV、采样前 logits 和草稿 token。记录通过 `model=main/mtp` 区分，
MTP call 的 `parent_call` 指向主模型调用，`sampled.jsonl` 仍只记录主模型接受的输出。
MTP 被拒绝草稿形成的 padding 按实际采样位置排除，数量记在 call 的 `rejected_query_rows`。
MTP 位置 p 消费主模型位置 p+1 的 token；PD KV 对比排除依赖生成 token 的最后一个
prompt 位置。`report.json` 的 `counts_by_model` 和 `first_difference_by_model`
分别报告主模型和 MTP 的差异。旧 dump 没有 MTP 数据，需要重新抓取才能验证 MTP。

同一个开关还记录 `AscendRejectionSampler` 的实际输入输出，归在对应主模型 call 的
`kind=rejection` 下：被验证的草稿 token、处理前 target/bonus logits、应用采样约束后的
target logits、bonus token、内核输出和 sampler 输出（含原始 padding）。调用元数据保存
温度等采样参数以及草稿对应的位置。随机采样还记录实际使用的 uniform 随机数、target
概率、可选 draft 概率和 recovered token；探针不会再执行一遍采样或额外消耗随机数。

收集器的 `--analyze-pd` 和直接运行分析脚本都会生成 `rejection.jsonl`，逐 request/call
列出草稿、主模型 argmax、接受数量、首个拒绝下标（从 0 开始）、后续未被采用的草稿及
补回/bonus token。`report.json` 的 `rejection_sampling.workers` 给出各 worker 的接受率；
同一请求的 TP 副本分别列出，不把多个 rank 累加成多个请求。计数排除未完成 prefill
而被丢弃的采样，处于 HTTP 的 EOS/长度裁剪之前。

`temperature=0` 时，离线工具还检查输出是否符合“连续接受与 target argmax 相同的草稿，
首个不同处采用 target token；全部相同则追加 bonus token”，记录
`greedy_decision_consistent` 和 `greedy_inconsistent_calls`。该判断只在离线进行，不中断
在线请求。随机采样的原始证据保留，但不套用 greedy 判断。旧 dump 没有这些输入，需
更新四台机器代码并用原有 dump 开关重新抓取，无需新增环境变量。

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
四个 IP 直接一起传入，不用先判断哪个 P/哪个 D 接到了请求，也不需要指定机器配对关系。
脚本从 manifest 识别 P/D，按外部 request ID 和 TP rank 自动匹配，允许 P/D 位于不同 DP。
没有本轮目录或目录为空的机器标为 `empty`，下次收集会重试；SSH/容器/权限错误、损坏文件仍报错。
所有机器都没有数据时返回失败；只有 P 或只有 D、缺 TP rank、重复请求记录时，分析报告明确标为不完整。

密码登录只需现有的 Python 和系统 OpenSSH，不需要安装 `sshpass`、`paramiko` 或图形界面的 askpass 程序。给命令追加：

```bash
--password
```

它会隐藏输入，一次输入供四台机器使用。不同密码用 `--password-per-host` 逐台输入。
自动化可用 `--password-env PD_SSH_PASSWORD` 从已有环境变量读取；密码不写入命令行、日志或 collection.json。
脚本通过 [OpenSSH 原生 SSH_ASKPASS 接口](https://man.openbsd.org/ssh.1#SSH_ASKPASS) 提供密码；临时回调文件不含密码，退出时自动删除。
Linux 使用 `/bin/sh` 回调，Windows 使用当前 Python；密码只保存在内存及 SSH 子进程环境中，父进程环境不变。
默认密码模式要求机器已存在于 SSH known_hosts；第一次连接先用普通 `ssh user@host` 确认主机身份。
需要跳过时，collect 或 cleanup 命令追加 `--skip-host-key-check`，本次连接不校验 SSH 服务器身份，
也不读取或写入 known_hosts；密码参数保持不变，不修改系统 SSH 配置。
远端（使用 `--container` 时为容器内）需要有 `python3` 和 `tar`。

不会改远端文件。不同机器路径/容器不同时分别运行到不同本地目录，分析脚本支持多个根目录。
`collection.json` 标明每台机器的状态；失败返回非零状态，重跑同一命令仅重试未完成/空机器。
`--analyze-pd` 在四机收集完成（允许部分机器为空）后自动分析 P→D KV，报告写到 `collected-case-on/report-pd`，执行端需要 CPU PyTorch。
只拉文件时去掉该参数，不需要 PyTorch；文件已拉取后加回该参数也不会重复下载。

### 清理四机 dump，复用 template 的目录

等本轮请求结束并收集完成后执行。停止发送请求期间清理，不要与正在写盘的抓取或收集并发：

```bash
python3 tools/pd_tensor_cleanup.py \
  --hosts root@7.150.4.174 root@7.150.5.55 root@7.150.5.81 root@7.150.1.46 \
  --repo-path /workspace/sqh/vllm-ascend \
  --run-id case-on \
  --password
```

与 collect 使用相同的 `--container`、`--ssh-port`、`--identity-file` 和密码选项。
这会删除四台机器的 `<repo-path>/pd-tensor-dump/case-on`；不存在的目录直接跳过。
保留 dump 父目录、其他 run、模型、Mooncake 数据和本地已收集文件。
若要清空整个 dump 目录，把 `--run-id case-on` 替换为 `--all-runs`。
可追加 `--dry-run` 查看删除范围而不删除。每台机器各输出一行结果，有失败会返回非零状态并继续尝试其他机器。

下一轮继续使用 template 中相同的 `case-on`，新请求会重新创建远端目录。
**本地 `--output` 每轮换新目录**（如 `./collected-case-on-02`），避免 collector 复用上一轮已成功下载的文件。

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
数值差异与覆盖完整性分开：`analysis_complete` 表示已采集数据的分析完成，不是全量覆盖或精度 PASS；`accuracy_verdict` 明确为 `not_assessed`。
`capture_sampling` 汇总抽样范围；`outside_reference_sample` / `outside_candidate_sample`
表示对应位置未被抽样，不算文件丢失，也不算比较通过。数值统计仅适用于真正对齐并加载的行。
缺 rank、缺文件、未完成、上下文不同会明确列出。先看 P→D KV，再结合 OFF/ON 的最早观察差异与后续误差放大判断位置。

如果同一次 attention 把一个逻辑 token 映射到多个物理槽，探针保留采样范围内多个槽的原始 KV，
`kv_consumed` 的 `positions` 可以重复，`physical_slots` 与 tensor 行一一对应。
每个请求、每个 worker 最多打印一次 `[PD_DUMP] ... kv_alias_rows=...; within captured rows`；
这只表示存在多个槽，不表示它们内容一定不同。
探针不因数值差异或 NaN 中止模型；越界、缺失映射等采集错误仍会报错。

离线 `report.json` 的 `kv_aliases` 汇总同一逻辑 token 的多个副本是否相同，
`first_conflict` 给出请求、TP rank、call、层、位置、物理槽和原始 tensor 文件。
`comparisons.jsonl` 的 `kv_aliases_different` / `kv_aliases_equal` 包含完整差异与分布统计，
两边相同位置的 NaN 作为相同非有限值报告，不误称内容差异。
P→D 比较会核对 D 已保存的每个副本；OFF/ON 也会独立检查已保存的别名冲突。未保存的行或副本不作正确性结论。
发现冲突仍需结合逻辑 top-k、实际 sparse indices 和 P 端 KV 判断是映射、加载还是数据异常，不能直接归因于 Mooncake。
