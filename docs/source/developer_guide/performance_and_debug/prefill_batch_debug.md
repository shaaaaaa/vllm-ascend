# P 节点 batch 成员日志

在 P 的启动环境加入，重启 P 服务：

```bash
export VLLM_ASCEND_PREFILL_BATCH_DEBUG=true
```

默认关闭，只在需要记录的 P 节点设置；D 不设置。与 STORE_DEBUG、FENCE_DEBUG、AUDIT 和 overlap 统计互相独立，无需开启它们。关闭 layerwise prefill cache 时也可以保留本开关作对照。

每个 DP/PP 的 TP0 在实际 main-model forward 提交前记录一次，不记录空调度或启动时的 dummy batch。数据来自模型实际 input batch，而不是可能省略无缓存操作请求的 connector 列表。支持正常图回放和异步调度，不读取 NPU tensor、不创建 event、不加同步。代价仅为开启时整理 CPU 元数据和写日志。

## 日志示例

```text
[P_BATCH] b=host:1234:7 dp=2 pp=0 n=2 mixed=1 stage=submit model=main q=4096 ubatch=0:4096
[P_BATCH] b=host:1234:7 dp=2 pp=0 n=2 mixed=1 i=0 req='A' span=[20000,22000) prompt=30000 flat=[0,2000) mode=prefill last_prefill=0
[P_BATCH] b=host:1234:7 dp=2 pp=0 n=2 mixed=1 i=1 req='B' span=[10000,12096) prompt=20000 flat=[2000,4096) mode=prefill last_prefill=0
```

- `b`：主机、进程和递增 batch 序号组成的标识。相同 `b` 的请求属于同一次提交。
- `n` / `mixed`：本 batch 请求数；大于一时 `mixed=1`。
- `span=[start,end)`：本次实际计算的请求内逻辑 token 区间，包含 start、不包含 end；start 也是本次计算前的历史长度，并不等同于外部缓存命中数。
- `prompt`：完整 prompt 长度。`last_prefill=1` 表示本次计算覆盖 prompt 末尾。
- `flat`：该请求在拼接 batch 中的 token 区间，不是请求自己的 token 位置。
- `mode`：本次计算处于 prefill、decode，或跨越 prompt 末尾。单 token 的 prefill 尾段仍标为 prefill。
- `ubatch`：若启用 DBO，列出子 batch 的 token 区间，可与 `flat` 交叉判断是否在同一个子 batch。

日志记录主干模型的提交，不能当作 NPU 完成通知；MTP 后续 draft 不重复记成新的 main-model batch。`mixed=1` 表明存在同批请求，是关联线索，本身不证明乱码由混批引起。

## 查看某个异常请求

```bash
grep -H -F 'response的UUID' tp4_dp4_logs/P_dp*.log | grep -F '[P_BATCH]'
```

拿到其中的完整 `b=主机:进程:序号` 后查看所有同批成员：

```bash
grep -H -F 'b=host:1234:7 ' tp4_dp4_logs/P_dp*.log
```

全程保留服务器完整 request ID，包括 PD 的后缀，因此客户端 response UUID 可以直接用于匹配。若出现 `[P_BATCH] ... error=...`，表示该次元数据日志不完整，不能按单请求批次解释。
