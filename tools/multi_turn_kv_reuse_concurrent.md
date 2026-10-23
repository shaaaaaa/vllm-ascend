# Concurrent long-prefix multi-turn KV reuse

并发版本使用独立入口 `multi_turn_kv_reuse_concurrent.py` 和
`plot_multi_turn_overlap_concurrent.py`。原来的单并发测试及按轮次绘图入口仍保留，
参见 [单并发测试说明](multi_turn_kv_reuse.md)。本页命令全部指向并发版本。

向已启动的 vLLM 服务或 PD proxy 发请求。每个独立会话先用固定长文章预热，
等待所有会话的预热请求完整结束，再开始测量。后续各会话独立推进：一个会话收到
自己的完整回答，就发送自己的下一轮，不等待其他会话。每个会话最多一个在途请求，
可同时出现会话 0 在第 2 轮、会话 1 在第 5 轮；只有预热阶段有全局屏障。

每个会话自己的输入始终是：

```text
本轮输入 = 上轮完整输入 + 上轮实际输出 + 本轮预写问题及角色分隔符
```

会话开头加入固定的 `session=编号; timestamp_ns=时间戳; run=运行编号`。
不同会话、不同运行的前缀不同，防止互相复用整篇文章的缓存；同一会话的标记
在所有轮次保持不变，不会为隔离会话而破坏它自己的历史 KV 复用。
使用 `/v1/completions`，保留回答的原始空白，不重新套 chat template、摘要或截断历史。

## 运行

```bash
python3 tools/multi_turn_kv_reuse_concurrent.py \
  --base-url http://7.150.4.174:8000 \
  --model-name GLM-5.3-falcon \
  --concurrency 4 \
  --context-tokens 200k \
  --tokenizer /workspace/models/GLM-5.3-w4a8c8 \
  --rounds 50 \
  --max-tokens 128 \
  --output-dir multi-turn-c4-200k \
  2>&1 | tee multi-turn-c4-200k.log
```

- `--base-url` 是服务或 PD proxy 地址，末尾带 `/v1` 也可。
- `--model-name` 对应服务的 `--served-model-name`；省略时通过 `/v1/models` 发现模型。
- `--concurrency` 默认 1；4 表示 4 个独立会话。默认每个会话先预热一次，再测 50 轮，
  共发出 `4 × (1 + 50)` 个请求，指标中只有 `4 × 50` 个后续请求。
- `--context-tokens 200k` 是文章前 200,000 个真实 token，`k` / `m` 使用十进制。
  需要 `--tokenizer` 指定与服务一致的本地目录或 `tokenizer.json`，以及 Python 的
  `tokenizers` 包。不下载模型，不加载权重，也不依赖 proxy 提供 `/tokenize` 接口。
- 截短文章后追加固定阅读问题；会话标记和阅读问题会增加少量 token。
  真实初始长度以预热响应的 `prompt_tokens` 为准。文章不足目标长度时不补长。
- 该长度只控制初始文章。后续保留全部历史，随着回答和问题逐轮增长，不反复截断到 200k。
- 不传 `--context-tokens` 时使用完整预写文章，约 500k token，不需要 tokenizer。
- `--rounds` 是预热后的测量轮数，默认 50；`--rounds 0` 只预热。
- `--max-tokens` 默认 128，需要长回答可改为 4000。HTTP socket timeout 默认 3600 秒，
  可用 `--timeout` 修改。服务上下文上限需容纳文章、所有历史以及当前回答预算。

所有会话共用已提交的 `tools/multi_turn_kv_assets/prefix_500k.txt` 和
`questions_50.txt`，运行时不生成文章或问题。原始前缀约 2.49 MB，参考 GLM-5
分词器为 499964 token；GLM-5.3 的实际数量以服务响应为准。
可用 `--prefix-file`、`--questions-file` 换已有文本；问题不足时发送请求前就报错。
输出目录必须为空或不存在。请求不自动重试；任何预热失败都不进入测量阶段。
全部预热完成只保证已完成初始请求，不锁定缓存驻留，也不保证不同后端一定共享缓存；
检查后续响应的 `cached_tokens`。缺失缓存量记为 null，不能当作 0。

## 输出和计时

默认在每条回复完整接收后打印原文，预热和后续轮次都包含，不截断、不逐 token 混排。
每段以 `[MULTI_TURN_OUTPUT] ... begin` / `end` 包围，标明 `session`、`round`、
`phase` 和 `request_id`，方便检查乱码和对应请求。上面的 `tee` 命令会一起保存输出。
打印发生在响应计时结束后；原始回答仍保存在对应轮次的 `output.txt`，
下一轮使用的历史不受打印格式影响。打印文本便于人工检查，不自动判定精度正确。

`metrics.csv` 和 `summary.json` 的 `rounds` 只包含已完成的后续请求，轮次为 1–50。
每条有 `session_id`、并发数、request/response ID、输入输出 token 数、缓存量、TTFT
和总耗时。每收到一个完整响应就写盘；中断测试可使用已经完成的请求绘图。
`summary.json` 的 `warmups` 单独保留预热结果用于检查，预热不会计入曲线和分位数。
`sessions` 分别记录各会话的 `completed_followup_rounds`；根级同名字段是所有会话
进度的最小值，只表示大家都已完成到哪一轮，不用于阻止快会话继续发送。

TTFT 从开始发送 HTTP 请求到收到第一个非空文本增量，包含上传、排队、模型计算、
PD 交接和网络时间。它不是纯 NPU prefill 时间。平均 token 间隔按首末文本增量间隔
除以 `completion_tokens - 1` 估计；MTP、流式合并和网络缓冲会影响该估计。

```text
initial_prompt.txt           截取后的公共文章/阅读问题，只存一份
questions.txt                固定问题副本
summary.json                 状态、配置、会话标记、预热和测量结果
metrics.csv                  仅后续请求，每条包含 session_id
sessions/000/                并发大于 1 时，每会话独立目录
  session_prefix.txt         该会话固定的时间戳/编号标记
  turns/000/                 预热，仅存档
  turns/001/ ... 050/        后续轮次
    added_input.txt          本轮增加的问题
    output.txt               本轮实际输出
    request.json             请求元数据，不重复保存整篇文章
    response.json            完整响应及 usage
    events.jsonl             原始流式事件
```

并发为 1 时，`session_prefix.txt`、`turns/` 直接放在输出根目录。
重建任一会话输入：固定会话标记 + 公共初始文章 + 此会话之前的实际回答和追加问题。
重新分词时拼接边界附近少量 token 可能变化，字符相同不保证 token 前缀完全相同。

## 只画 TTFT，不读服务端日志

```bash
python3 tools/plot_multi_turn_overlap_concurrent.py \
  --run-dir multi-turn-c4-200k \
  --ttft-only \
  --output multi-turn-c4-ttft.png
```

绘图需要 matplotlib，不需要 vLLM、NPU、SSH 或服务端统计开关。
读取 `summary.json`，不存在时读取 `metrics.csv`。预热轮 0 始终排除。
并发为 1 时画带数值的曲线；并发大于 1 时，每个后续轮画一个箱体：
箱体为 P25–P75，中线 P50，须线 P1–P99，菱形是均值；标注均值、P50、P99 和
有效样本数 `n/并发数`。分位数在已完成会话样本间做线性插值；4 路只有 4 个样本，
P99 接近最大值，并不代表有大量尾延迟样本。缺失 TTFT 不补 0，不把 TP rank 当作会话。
轮次是各会话自己的对话深度，同轮样本可能来自不同墙钟时段，不代表同一同步 batch。
会话完成全部指定轮次后退出，因此测试收尾阶段的实际在途并发数会降低。
输出同名 CSV，含均值 `ttft_s`、P50、P99、样本数等。中断轮可显示例如 `n=3/4`。

截取后续第 1–10 轮，加 `--start-round 1 --end-round 10`。
只完成 10 轮但参数写 50，不会补出后面没有完成的轮次。

## 同时画 overlap

P 节点启动前开启：

```bash
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
export VLLM_ASCEND_PREFILL_OVERLAP_STATS=true
```

更新 LMCache-Ascend 后重启 P 服务，多请求 batch 也输出一份整体统计。
所有请求的真实 token 区间保留在日志 `requests`，不把 batch 时间分摊或重复计给各请求。
不同请求轮转相位不同时，按组内序号模 bank 数配对整个 batch 的复用窗口；4 bank
比较前后相隔 4 个组内层的等待门，不能按重叠的物理 bank 集合误配成相邻层。
每个有效统计 batch 读取一次 CPU 时钟，记录 `batch_start_time_ns` 作为横轴定位，
表示主线程开始提交此 batch 的时间，不是 NPU 实际开始执行的时间。
coverage、save、load、wait 的耗时仍取 NPU event；不增加 host synchronize，
不增加 NPU 计时事件，也不改变传输或计算依赖。多台 P 机器的时钟需要对齐。

```bash
python3 tools/plot_multi_turn_overlap_concurrent.py \
  --hosts root@7.150.4.174 root@7.150.5.55 \
  --remote-log-dir /workspace/fsi_lab/sqh/vllm-ascend/tp4_dp4_logs \
  --run-dir multi-turn-c4-200k \
  --ssh-password '你的SSH密码' \
  --skip-host-key-check \
  --aggregate-ranks \
  --time-bin-seconds 1 \
  --output multi-turn-c4-overlap.png
```

收集所有 P 机器的日志。默认匹配 `P_dp*.log`；用 `--log-pattern` 换文件名，
也可用 `--container` 从容器内收集。`--repo-path` 只取固定的 `<repo>/tp4_dp4_logs`，
不是目录搜索。原生 OpenSSH/SSH_ASKPASS，不需要 sshpass；可使用
`--password-env PD_SSH_PASSWORD` 或密钥替代显式密码。
远端先过滤统计行再压缩传输，每次保存在图片旁的 `multi-turn-overlap-logs-*/`。

已有本地日志可以直接画图，不再次访问远端：

```bash
python3 tools/plot_multi_turn_overlap_concurrent.py \
  'multi-turn-overlap-logs-*/host-*/snapshot/logs/*.log' \
  --run-dir multi-turn-c4-200k --aggregate-ranks --output multi-turn-c4-overlap.png
```

overlap 和各 bank 时间的横轴改为时间，以首个匹配的测量 batch 为 0 秒。
不按对话轮次归类，不标注请求 ID；一个 batch 可以包含任意多个不同轮次的请求。
只要包含选中测量范围内的任一请求，就保留整个 batch 一次，不把指标分摊给请求，
也不因跨轮而跳过。混有范围外请求时，整批仍保留并在 JSON 诊断计数。
只有预热或无关请求的 batch 排除。`--start-round` / `--end-round` 筛选 TTFT 的轮次，
以及 overlap 所含的测量请求范围；混合 batch 无法进一步拆分。

不加 `--aggregate-ranks` 时，每个 worker 的每个 batch 是一个时间数据点。
加上时，在固定时间窗口内汇总所有观测 worker 的 batch，默认每 1 秒一个均值点；
可用 `--time-bin-seconds 5` 改为 5 秒。coverage 按有效模型层样本数加权平均已有
逐层比值，不用总等待除以总窗口重新计算。save/load/wait 是该窗口内每个 worker
batch 的平均耗时；没有数据的窗口不补点，不把整批耗时摊到经过的每一秒。
不同 stream 的传输时间分别累计，并非墙钟时间。首层 bank 等待仍保留。

TTFT 继续单独按对话轮次画，并发大于 1 时画箱体，与 overlap 的时间横轴独立。

图含 coverage、save+load、save、load、全层 bank wait、首层 bank wait、复用 wait
及 TTFT。输出时间数据 `.csv`、逐 batch 的 `.chunks.csv`、逐轮 TTFT 的
`.ttft.csv` 和匹配/缺失情况 `.json`。原始 batch 统计保留，不会被时间窗口均值替代。
缺失服务端统计留空，不补 0 或 100%；客户端 TTFT 独立使用完整会话样本。
旧 schema 1 无法恢复同 bank 复用口径，需重新采集。schema 2 旧日志没有 batch 时间戳
时，可使用日志行里完整的年月日时分秒作为近似横轴，并显式提示这是后台输出时间，
不能当成 batch 执行时间；需保持日志与分析机器的时区一致。没有任何时间戳则跳过并
报告缺失，不用 batch 编号或轮次伪造时间。更新 LMCache-Ascend 并重启 P 后使用新时间戳。

## 本地验证

```bash
python3 -m pytest --confcutdir=tests/standalone \
  tests/standalone/test_multi_turn_kv_reuse_concurrent.py \
  tests/standalone/test_multi_turn_overlap_plot_concurrent.py -q -o addopts=
```

本地 HTTP 模拟验证并发、仅预热屏障、后续独立推进、隔离历史、真实 token 截取和失败处理。
真实 NPU 性能与缓存命中仍需在目标服务测试。
