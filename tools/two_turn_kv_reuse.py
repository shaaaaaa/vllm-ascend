#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send two article-reading turns with an unchanged first-turn text prefix.

python3 tools/two_turn_kv_reuse.py --base-url http://7.150.4.174:8000 --model-name GLM-5.3-falcon
Uses only the Python standard library and an already running inference service.
Reads both prewritten articles from tools/two_turn_kv_articles before sending requests.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
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
            print("[TWO_TURN] /tokenize unavailable; keeping the prewritten articles unchanged", flush=True)
            return None
        return int(result["count"])


def make_input(article_text: str, turn: int, run_id: str) -> str:
    """Wrap a prewritten article without generating, trimming or resizing it."""
    if turn == 1:
        opening = f"用户（阅读编号 {run_id}）：\n请解读下面的文章，概括核心观点、主要证据与政策建议。\n\n"
        ending = "\n\n请用简洁的中文回答。\n助手：\n"
    else:
        opening = "\n\n用户（第二轮）：\n请继续解读下面的新文章，并结合前面的文章和你的回答比较两者的观点。\n\n"
        ending = "\n\n请指出共同原则、主要差异与值得进一步验证的问题，用简洁的中文回答。\n助手：\n"
    return opening + article_text + ending


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def complete(client: Client, model: str, prompt: str, max_tokens: int, root: Path, turn: int) -> dict[str, Any]:
    """Save the exact request and response, including the server's cache usage."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    save_json(root / f"round{turn}_request.json", payload)
    (root / f"round{turn}_input.txt").write_bytes(prompt.encode("utf-8"))
    started = time.perf_counter()
    response = client.request("/v1/completions", payload)
    elapsed = time.perf_counter() - started
    save_json(root / f"round{turn}_response.json", response)
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0].get("text"), str) or not choices[0]["text"]:
        raise RuntimeError(f"Round {turn} returned no text; inspect round{turn}_response.json")
    output = choices[0]["text"]  # Preserve spaces/newlines exactly for the next prompt.
    (root / f"round{turn}_output.txt").write_bytes(output.encode("utf-8"))
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
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model",
        help="Served model name, e.g. GLM-5.3-falcon; default: first model returned by /v1/models",
    )
    articles = Path(__file__).resolve().parent / "two_turn_kv_articles"
    parser.add_argument("--first-article", type=Path, default=articles / "round1.txt", help="Prewritten first article")
    parser.add_argument(
        "--second-article", type=Path, default=articles / "round2.txt", help="Prewritten second article"
    )
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum output tokens per turn (default: 4000)")
    parser.add_argument("--timeout", type=float, default=600, help="HTTP timeout in seconds")
    parser.add_argument("--output-dir", type=Path, help="New or empty artifact directory")
    args = parser.parse_args()
    if min(args.max_tokens, args.timeout) <= 0:
        parser.error("Maximum output tokens and timeout must be positive")
    first_article = args.first_article.read_text(encoding="utf-8")
    second_article = args.second_article.read_text(encoding="utf-8")
    if not first_article.strip() or not second_article.strip():
        parser.error("Both article files must contain text")
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
            raise RuntimeError("No models returned by /v1/models; specify --model-name")
        model = models[0]["id"]
    print(f"[TWO_TURN] model={model} artifacts={root.resolve()}", flush=True)
    first_input = make_input(first_article, 1, run_id)
    second_added = make_input(second_article, 2, run_id)
    first_count = client.count_tokens(model, first_input)
    added_count = client.count_tokens(model, second_added)
    (root / "round2_added_input.txt").write_bytes(second_added.encode("utf-8"))
    print(f"[TWO_TURN] first_input_tokens={first_count} second_new_input_tokens={added_count}", flush=True)
    first = complete(client, model, first_input, args.max_tokens, root, 1)
    second_input = first_input + first["text"] + second_added
    second = complete(client, model, second_input, args.max_tokens, root, 2)
    save_json(
        root / "summary.json",
        {
            "model": model,
            "first_article": str(args.first_article.resolve()),
            "second_article": str(args.second_article.resolve()),
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
