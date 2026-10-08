#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send two article-reading turns with an unchanged first-turn text prefix.

python3 tools/two_turn_kv_reuse.py --base-url http://7.150.4.174:8000
Uses only the Python standard library and an already running inference service.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.tokenize_available = True

    def request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def count_tokens(self, model: str, prompt: str) -> int | None:
        if not self.tokenize_available:
            return None
        try:
            result = self.request("/tokenize", {"model": model, "prompt": prompt})
        except urllib.error.HTTPError as exc:
            if exc.code not in (404, 405):
                raise
            self.tokenize_available = False
            print("[TWO_TURN] /tokenize unavailable; using approximate article lengths", flush=True)
            return None
        return int(result["count"])


def article(paragraphs: int, turn: int) -> str:
    """Construct a readable article with distinct cases, figures and conclusions."""
    themes = (
        (
            "公共交通与城市更新",
            (
                "公交优先并不只意味着增加车辆。线路需要连接就业、教育和医疗设施，"
                "换乘规则也应让居民容易理解。评价方案时，必须同时考察等候、步行和换乘时间。",
                "道路空间是一种有限的公共资源。设置公交专用道之后，应比较人员通行量，"
                "而不是只统计汽车数量。步行环境、无障碍设施和安全过街同样影响公共交通的吸引力。",
                "老城区的更新不能以搬迁居民作为唯一手段。修复社区道路、补充公共服务和改善"
                "建筑节能，可以逐步提升生活质量。实施顺序应考虑居民承受能力与商户正常经营。",
                "交通票价体现了效率与公平之间的权衡。低收入家庭、学生和老年人的出行需求"
                "不完全相同，补贴方案应公开成本，并检查优惠是否真正覆盖需要帮助的人。",
                "交通数据能够帮助发现拥堵原因，但匿名化和访问权限必须在采集前确定。"
                "单次客流变化不足以证明政策有效，还应观察工作日、周末和季节之间的差异。",
                "政策试点需要可检验的目标。居民反馈、服务可靠性、交通事故和财政负担"
                "应共同进入评估，不能用一项漂亮的平均数掩盖某些社区的明显损失。",
            ),
        ),
        (
            "城市低碳转型与公共治理",
            (
                "城市能源系统需要同时满足可靠、可负担和低排放三个目标。屋顶光伏可以"
                "提供清洁电力，但建筑遮挡、维护责任和电网容量会影响实际收益，必须逐项核实。",
                "建筑改造往往比新建示范项目更难。保温、照明和空调升级需要结合建筑使用"
                "习惯，并明确节能收益如何分配。租户与业主之间的利益差异不能靠宣传解决。",
                "社区绿地既能改善热环境，也能提供日常活动空间。树种选择、灌溉用水和"
                "后续养护都应计入预算，避免把短期建设规模误当作长期生态成效。",
                "海绵城市设施需要与既有排水系统协同。透水铺装、雨水花园和调蓄池能够"
                "分担常见降雨，但极端暴雨仍需要应急预案，设计能力和维护记录应向公众公开。",
                "垃圾分类的效果取决于完整链条。居民分类之后，运输、分拣和资源利用环节"
                "必须保持一致。评估不能只统计投放点数量，还要查看污染率和真实回收去向。",
                "企业减排应采用清晰的核算边界。供应链、运输与生产环节的排放要避免"
                "重复计算，采购绿色电力也不能代替必要的工艺改进和能源效率提升。",
                "低碳政策可能给不同群体带来不同负担。对设备改造提供补助时，应考虑小微"
                "企业现金流和老旧社区条件，政策执行需要留下申诉渠道与调整空间。",
                "公共决策需要持续学习。试点结果应公开基线、测量方法和不确定性，"
                "跨城市复制经验时必须重新检查气候、产业结构、人口密度和财政条件。",
            ),
        ),
    )
    title, topics = themes[turn - 1]
    sections = [f"《{title}：实践观察与政策讨论》\n"]
    for index in range(paragraphs):
        district = index + 1
        topic = topics[index % len(topics)]
        sections.append(
            f"第{district}项观察：{topic}"
            f"某市第{district}片区进行了{12 + index % 18}个月的试点，"
            f"涉及{800 + district * 37}户居民和{20 + index % 45}家商户。"
            f"项目组把实施前后的结果与邻近片区比较，发现平均指标改善约{5 + index % 16}%，"
            "但不同年龄和收入群体的反馈并不一致。访谈说明，设施的可达性、管理人员的响应"
            "以及持续维护，比项目启动时的宣传更能影响居民体验。研究人员建议先解决服务"
            "最薄弱的环节，保留阶段性复核机制，再决定是否扩大规模。\n"
        )
    return "\n".join(sections)


def make_input(client: Client, model: str, target: int, turn: int, run_id: str) -> tuple[str, int | None]:
    """Size only this turn's new input; the second turn's history is added later."""
    if turn == 1:
        opening = f"用户（阅读编号 {run_id}）：\n请解读下面的文章，概括核心观点、主要证据与政策建议。\n\n"
        ending = "\n\n请用简洁的中文回答。\n助手：\n"
    else:
        opening = "\n\n用户（第二轮）：\n请继续解读下面的新文章，并结合前面的文章和你的回答比较两者的观点。\n\n"
        ending = "\n\n请指出共同原则、主要差异与值得进一步验证的问题，用简洁的中文回答。\n助手：\n"
    corpus = article(max(40, target // 40), turn)
    chars = min(len(corpus), math.ceil(target * 1.4))
    count = None
    for _ in range(3):
        text = opening + corpus[:chars] + ending
        count = client.count_tokens(model, text)
        if count is None or abs(count - target) <= target * 0.08:
            return text, count
        chars = min(len(corpus), max(1, round(chars * target / count)))
    text = opening + corpus[:chars] + ending
    return text, client.count_tokens(model, text)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def complete(client: Client, model: str, prompt: str, max_tokens: int, root: Path, turn: int) -> dict[str, Any]:
    """Save the exact request and response, including the server's cache usage."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    save_json(root / f"round{turn}_request.json", payload)
    (root / f"round{turn}_input.txt").write_text(prompt, encoding="utf-8", newline="")
    started = time.perf_counter()
    response = client.request("/v1/completions", payload)
    elapsed = time.perf_counter() - started
    save_json(root / f"round{turn}_response.json", response)
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0].get("text"), str) or not choices[0]["text"]:
        raise RuntimeError(f"Round {turn} returned no text; inspect round{turn}_response.json")
    output = choices[0]["text"]  # Preserve spaces/newlines exactly for the next prompt.
    (root / f"round{turn}_output.txt").write_text(output, encoding="utf-8", newline="")
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    stats = {
        "request_id": response.get("id"),
        "seconds": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "finish_reason": choices[0].get("finish_reason"),
    }
    print(f"\n[TWO_TURN] round={turn} {json.dumps(stats, ensure_ascii=False)}\n{output}\n", flush=True)
    return {"text": output, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Service or PD proxy address")
    parser.add_argument("--model", help="Default: first model returned by /v1/models")
    parser.add_argument("--first-tokens", type=int, default=3000)
    parser.add_argument("--second-tokens", type=int, default=8000, help="New second-turn input, excluding history")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens per turn")
    parser.add_argument("--timeout", type=float, default=600, help="HTTP timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="New or empty artifact directory")
    args = parser.parse_args()
    if min(args.first_tokens, args.second_tokens, args.max_tokens, args.timeout) <= 0:
        parser.error("Token counts and timeout must be positive")
    run_id = uuid.uuid4().hex[:12]
    root = args.output_dir or Path(f"two-turn-kv-{time.strftime('%Y%m%d-%H%M%S')}-{run_id}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error("--output-dir must be empty")
    client = Client(args.base_url, args.timeout)
    model = args.model
    if model is None:
        models = client.request("/v1/models").get("data", [])
        if not models:
            raise RuntimeError("No models returned by /v1/models; specify --model")
        model = models[0]["id"]
    print(f"[TWO_TURN] model={model} artifacts={root.resolve()}", flush=True)
    first_input, first_count = make_input(client, model, args.first_tokens, 1, run_id)
    second_added, added_count = make_input(client, model, args.second_tokens, 2, run_id)
    (root / "round2_added_input.txt").write_text(second_added, encoding="utf-8", newline="")
    print(f"[TWO_TURN] first_input_tokens={first_count} second_new_input_tokens={added_count}", flush=True)
    first = complete(client, model, first_input, args.max_tokens, root, 1)
    second_input = first_input + first["text"] + second_added
    second = complete(client, model, second_input, args.max_tokens, root, 2)
    save_json(
        root / "summary.json",
        {
            "model": model,
            "second_input_composition": "round1_input + round1_actual_output + round2_added_input",
            "first_input_tokens": first_count,
            "second_new_input_tokens": added_count,
            "round1": first["stats"],
            "round2": second["stats"],
        },
    )
    print(
        "[TWO_TURN] cached_tokens=null means this endpoint did not report cache usage; check server logs.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"[TWO_TURN] HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
