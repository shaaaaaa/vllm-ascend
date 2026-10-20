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
