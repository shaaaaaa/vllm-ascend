# Long-prefix multi-turn KV reuse

向已经启动的 vLLM 服务或 PD proxy 发送一条预先写好的约 500k-token 输入，
取得真实模型输出，再继续 50 轮短问题。每轮严格使用：

```text
本轮输入 = 上轮完整输入 + 上轮模型实际输出 + 本轮预先写好的问题及角色分隔符
```

默认是 **1 次初始化 + 50 次追加，共 51 次请求**，全部串行。
使用 `/v1/completions` 直接追加文本，保留回答中的空格、换行和思考文本，
不重新构造 chat template，不做摘要、截断、随机采样或运行时补长前缀。
文本前缀保持不变；重新分词时，拼接边界附近的少量 token 仍可能变化，
不能仅凭字符相同就断言 token 或服务端缓存命中数完全相同。

## 固定输入

- `tools/multi_turn_kv_assets/prefix_500k.txt`：约 2.49 MB 的原创英文合成城市治理档案，
  加上中文阅读说明与首轮问题。人物、地点、数字和事件均为虚构。
- `tools/multi_turn_kv_assets/questions_50.txt`：50 个预先写好的中文短问题，每行一个。
- `tools/multi_turn_kv_assets/manifest.json`：文件大小与预先测得的参考 token 数。

前缀使用公开 [GLM-5 tokenizer](https://huggingface.co/zai-org/GLM-5/blob/main/tokenizer.json)
在开发时计数，`add_special_tokens=False` 时为 **499964 token**。
这个参考计数不代表部署的 GLM-5.3 tokenizer 必然相同。
脚本运行时直接读取这份完整文本，实际长度以服务端返回的 `usage.prompt_tokens` 为准；
不下载 tokenizer，不调用模型生成测试资料，也不会为凑长度而改写文件。

## 运行

在 vllm-ascend 仓库运行，客户端只需要 Python 3.9+ 标准库：

```bash
python3 tools/multi_turn_kv_reuse.py \
  --base-url http://7.150.4.174:8000 \
  --model-name GLM-5.3-falcon \
  --rounds 50 \
  --max-tokens 128 \
  --output-dir multi-turn-on \
  2>&1 | tee multi-turn-on.log
```

`--base-url` 填实际服务地址；若走 PD proxy，填 proxy 的 IP 和端口。
末尾带 `/v1` 也可以。`--model-name` 对应服务的 `--served-model-name`，
省略时取 `/v1/models` 返回的第一个模型。
默认每次最多输出 128 token，适合测试长历史、短增量；需要长回答可改为 `--max-tokens 4000`。
默认 HTTP socket timeout 为 3600 秒，可通过 `--timeout` 修改。
服务的 `max-model-len` 必须容纳初始前缀、历轮输出和问题以及当前输出预算。

`--rounds 0` 只发初始化请求，`--rounds 3` 发初始化和前三个固定问题。
支持 `--prefix-file` 和 `--questions-file` 指定已有文本；不会生成缺失的问题。
输出目录必须是新的或空的。请求失败后立即停止，不自动重试或跳过该轮。
第一轮也可能命中先前运行留下的前缀，脚本不主动清理服务端缓存，
不向固定前缀插入随机 run ID；每轮 request ID 则独立生成并打印，方便关联服务日志。

## 结果

每轮开始打印 round 和 request ID，完成后打印一条 JSON 指标，包括：

- 输入、输出 token 数，以及服务报告的 `cached_tokens`；
- TTFT、请求总耗时和近似平均 token 间隔；
- 输入字符数、本轮增加的字符数、结束原因。

TTFT 从开始发送 HTTP 请求到收到第一个非空文本增量，包含上传、排队等时间。
平均 token 间隔按首末文本增量时间差除以 `completion_tokens - 1` 估计；
MTP 或网络缓冲可能在一个增量里返回多个 token，因此它不是逐 token 的设备计时。
客户端不把流式消息数当成 token 数。
若服务或 proxy 没有报告缓存量，对应项为 `null`，不能当作命中数为零。
准确命中范围仍需结合服务端日志检查。

输出目录包含：

```text
initial_prompt.txt       初始完整输入，只保存一份
questions.txt           预写问题的副本
metrics.csv             逐轮指标
summary.json            状态、运行参数、全部已完成轮次的指标
turns/000/              初始化请求
turns/001/ ... 050/     后续每轮
  added_input.txt       本轮追加的问题及角色分隔符；初始化为空
  output.txt            本轮实际输出，保留空白
  request.json          请求参数、request ID、输入字符数；不重复保存完整前缀
  response.json         合并后的回答和 usage
  events.jsonl          服务端原始流式 JSON 事件
```

要重建第 N 轮输入，从 `initial_prompt.txt` 开始，按顺序追加第 0 至 N-1 轮的
`output.txt` 和其后一轮的 `added_input.txt` 即可。
初始化轮编号为 0，50 个后续问题编号为 1 至 50。
失败轮的部分输出和已收到的事件会保留，但不会用于发送下一轮。
同一份固定前缀与问题可用于分别开启、关闭 layerwise 后运行；
后续历史使用各自模型的实际输出，因此若回答不同，后续请求也会不同。

本地 HTTP 模拟测试验证 51 次请求的真实回答累积、原样保存空白、流式 usage、
服务模型名、失败即停和固定资料文件。实际 NPU 性能与缓存命中需要在目标服务上运行。

```bash
python3 -m pytest --confcutdir=tests/standalone \
  tests/standalone/test_multi_turn_kv_reuse.py -q -o addopts= -o log_cli=false
```

## 按对话轮次画 overlap 图

P 节点启动服务前开启统计，随后按上面的命令运行客户端并保留 `multi-turn-on` 目录：

```bash
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
export VLLM_ASCEND_PREFILL_OVERLAP_STATS=true
```

测试结束后，在绘图机器运行：

```bash
python3 -m pip install matplotlib
python3 tools/plot_multi_turn_overlap.py \
  --hosts root@7.150.4.174 root@7.150.5.55 \
  --remote-log-dir /workspace/fsi_lab/sqh/vllm-ascend/tp4_dp4_logs \
  --run-dir multi-turn-on \
  --ssh-password '你的SSH密码' \
  --skip-host-key-check \
  --aggregate-ranks \
  --output multi-turn-overlap.png
```

`--hosts` 填所有 P 机器的 IP 或 `user@IP`，`--remote-log-dir` 填这些机器上日志的
实际绝对目录。脚本并发收集各机器上该目录中的 `P_dp*.log`，包括每台机器的两个
DP 日志，按 request ID 自动匹配轮次。开始采集时会打印实际目录和文件名匹配模式。
也可用 `--repo-path /仓库路径`，但这只是固定取 `<repo-path>/tp4_dp4_logs`，不会自动
搜索仓库或其他目录；不要把仓库目录当作已经确认的日志目录。
文件名不同，用 `--log-pattern 'P*.log'`，也可重复指定该参数。
若日志在容器内而 SSH 登录宿主机，加 `--container 容器名`。

`--ssh-password '密码'` 直接传各机器共用的 SSH 密码，运行中不会再询问密码。
原有 `--password` 仍可提示输入一次共用密码，输入不回显；密码不同改用
`--password-per-host`。也可以用 `--password-env PD_SSH_PASSWORD` 从同名环境变量
读取密码，或不传密码选项，沿用 SSH 密钥。仅使用系统 OpenSSH 和临时 `SSH_ASKPASS`
回调，不依赖 sshpass；密码不会再写入远端 SSH 命令、临时脚本或采集结果，完成后删除临时回调。
示例中的 `--skip-host-key-check` 沿用之前跳过主机校验的用法；已有 known_hosts 时可去掉。
SSH 端口和私钥可通过 `--ssh-port`、`--identity-file` 指定。

远端只读取匹配日志中的 `[PREFILL_OVERLAP]` 行，先打包压缩，再传回本地，
不改服务日志、不触发 tensor dump。每次创建新快照，保存在图片旁的
`multi-turn-overlap-logs-*/`（可用 `--collect-dir` 指定一个尚不存在的目录）；
各主机分目录保留过滤后的日志、压缩包和采集清单。服务仍写日志时，仅读取打开文件时
已有的字节，跳过最后未写完的行，建议 50 轮结束后采集。
默认每台机器最多等待 600 秒，可用 `--ssh-timeout` 调整。任意机器连接、认证或路径
失败都会保留结果并停止绘图，避免把收集失败当作无数据；正常读取到日志但没有统计行的
空闲节点允许为空，缺少轮次仍会在绘图诊断中提示。

已有本地日志仍可直接画图：

```bash
python3 tools/plot_multi_turn_overlap.py 'P1_logs/P_dp*.log' 'P2_logs/P_dp*.log' \
  --run-dir multi-turn-on --aggregate-ranks --output multi-turn-overlap.png
```

`logs` 可以传多个 P 节点日志文件或 glob；DP 路由可能把不同轮次发到不同实例，
应收集所有相关 P 节点的日志。客户端目录只需 `summary.json`，或者没有该文件时
使用 `metrics.csv`；不需要复制 500k 文本或 tensor dump。脚本使用客户端 request ID
和响应 ID 匹配服务端内部 ID，接受 vLLM 添加的 `cmpl-` / `chatcmpl-` 前缀及内部后缀。
不按日志出现顺序猜轮次，也不会把另一组测试的请求混进来。

图为白底，横轴是轮次：0 是初始化长前缀，1–50 是后续追加。
八个子图分别显示 coverage、save+load、save、load、含首层的总 bank wait、
首层 bank wait、参与相邻层覆盖率计算的 bank wait、layer start interval。
服务日志中原有的 `first_layer_wait_ms`、`all_wait_ms` 现在保留到逐轮 CSV 和逐 chunk CSV。
`all_wait_ms` 包含首层，`wait_ms` 保持原有相邻层统计口径；首层等待不会被丢弃。
每个 chunk 都有自己的首层等待，同一轮先累加各 chunk 的首层等待，再按 worker 取平均。
首层没有上一层起点，因此不能凭空计算相邻层覆盖率，其等待时间单独展示。
已有服务日志可以直接重新绘图，无需重跑推理；仅有旧版汇总 CSV 无法还原首层等待，
要读取原始 `[PREFILL_OVERLAP]` 日志。缺失字段留空，不视为零。
若第 0 轮远大于后续轮次，用同一命令加 `--start-round 1`，只看 50 轮追加；
也可以用 `--end-round 10` 截取区间。

同一轮有多个 prefill chunk 时，先累加它们的耗时。coverage 按 `layer_samples`
对服务端已计算的逐层覆盖率加权平均，仍然是逐层比例的平均，不改成总等待除以总间隔。
默认每个 DP/TP worker 一条曲线；`--tp-rank 0` / `--dp-rank 0` 可以筛选。
`--aggregate-ranks` 合成一条曲线：coverage 平均所有实际观测的层，耗时则对各 worker
的整轮累计值取平均。图中时间单位为 ms / worker / round，不是每个 chunk 的平均，
也不是请求墙钟时间；bank 之间重叠的传输时间仍分别相加。

输出同名 `.csv`（逐轮汇总）、`.chunks.csv`（逐 chunk 原始计数/耗时）和 `.json`
（匹配及缺失情况）。没有统计的轮次留空、曲线断开并给出提示，不会补成 0 或 100%。
多请求同批仍按服务端既有规则跳过；统计队列丢失、失败或不完整记录不会被当成完整数据。
某个 worker 缺少部分 chunk 仍可能形成部分汇总，需结合 JSON 提示及 chunks CSV 检查；
图只代表实际记录到的样本。
