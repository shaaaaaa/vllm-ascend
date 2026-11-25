# SPDX-License-Identifier: Apache-2.0
"""Exercise standard chat requests, preserved history and both benchmark clients."""

import json
import re
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import prefill_kv_replay  # noqa: E402
from multi_turn_chat import CHAT_FORMAT, NATIVE_FORMAT, AssistantStream, restore_native_tokens  # noqa: E402


def render(messages):
    text = "[gMASK]<sop>"
    for message in messages:
        text += "<|" + message["role"] + "|>"
        if message["role"] == "assistant":
            text += "<think>" + message.get("reasoning", "") + "</think>"
        text += message["content"]
    return [ord(c) for c in text + "<|assistant|><think>"]


def answer(session, turn, mode):
    message = dict(role="assistant", content=f" \r\n真实回答 {session}/{turn}。\n\n")
    if mode != "content":
        message["reasoning"] = f" \n思考 {session}/{turn}。\r\n"
    if mode in ("reasoning", "empty"):
        message["content"] = ""
    if mode == "empty":
        message.pop("reasoning", None)
    return message


@contextmanager
def server(concurrency=1, mode="mixed", hold=False):
    state = dict(requests=[], paths=[], completed=set())
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency, timeout=10)
    advanced = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["paths"].append(self.path)
            assert self.path == "/v1/chat/completions"
            assert "prompt" not in body and "add_special_tokens" not in body
            assert "stop" not in body and "chat_template" not in body
            assert body["chat_template_kwargs"] == {"clear_thinking": False}
            assert body["temperature"] == 0
            match = re.search(r"-s(\d+)-", body["request_id"])
            session = int(match[1]) if match else 0
            turn = int(body["request_id"].rsplit("-", 1)[1])
            with lock:
                state["requests"].append((session, turn, body))
                if turn:
                    assert len({s for s, t in state["completed"] if t == 0}) == concurrency
                    assert (session, turn - 1) in state["completed"]
            if not turn:
                barrier.wait()
            if hold and session == 1 and turn == 2:
                advanced.set()
            if hold and session == 0 and turn == 1:
                assert advanced.wait(10), "new round waited for other sessions"
                state["independent_advance"] = True
            message = answer(session, turn, mode)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            identifier = "chatcmpl-" + str(uuid.uuid4())

            def send(delta, finish=None, usage=None):
                event = dict(id=identifier, choices=[dict(index=0, delta=delta, finish_reason=finish)])
                if usage is not None:
                    event["usage"] = usage
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()

            send(dict(role="assistant"))
            if "reasoning" in message:
                key = "reasoning" if turn % 2 else "reasoning_content"
                for part in (message["reasoning"][:3], message["reasoning"][3:]):
                    send({key: part})
            for part in (message["content"][:3], message["content"][3:]):
                send(dict(content=part))
            with lock:
                state["completed"].add((session, turn))
            send(
                {},
                finish="length" if turn % 2 else "stop",
                usage=dict(
                    prompt_tokens=len(render(body["messages"])),
                    completion_tokens=8 if mode != "empty" else 0,
                    prompt_tokens_details=dict(cached_tokens=4 if turn else 0),
                ),
            )
            self.wfile.write(b"data: [DONE]\n\n")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}", state
        finally:
            http.shutdown()
            worker.join(timeout=5)


def invoke(tmp_path, url, concurrent, *extra):
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("固定文章。\n助手：\n", encoding="utf-8")
    questions = tmp_path / "questions.txt"
    questions.write_text("依据？\n结论？\n下一步？\n", encoding="utf-8")
    output = tmp_path / "run"
    script = "multi_turn_kv_reuse_concurrent.py" if concurrent else "multi_turn_kv_reuse.py"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(REPO / "tools" / script),
        "--base-url",
        url,
        "--model-name",
        "glm",
        "--prefix-file",
        str(prefix),
        "--questions-file",
        str(questions),
        "--rounds",
        "3",
        "--output-dir",
        str(output),
    ]
    if concurrent:
        command += ["--concurrency", "3"]
    return subprocess.run(command + list(extra), capture_output=True, text=True, encoding="utf-8", timeout=30), output


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "reasoning", "empty", "content"])
def test_chat_keeps_messages_and_accepts_normal_stops(tmp_path, concurrent, mode):
    concurrency = 3 if concurrent else 1
    with server(concurrency, mode) as (url, state):
        result, root = invoke(tmp_path, url, concurrent, "--tokenizer-url", "http://127.0.0.1:1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(state["requests"]) == 4 * concurrency
    assert set(state["paths"]) == {"/v1/chat/completions"}
    assert (root / "initial_prompt.txt").read_text(encoding="utf-8") == "固定文章。"
    assert not list(root.rglob("prompt_token_delta.json"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    for session in range(concurrency):
        folder = root / "sessions" / f"{session:03d}" if concurrent else root
        previous = []
        for _, turn, payload in sorted(row for row in state["requests"] if row[0] == session):
            messages = payload["messages"]
            assert messages[: len(previous)] == previous
            if turn:
                assert messages[-2] == answer(session, turn - 1, mode)
                assert messages[-1]["role"] == "user"
            turn_dir = folder / "turns" / f"{turn:03d}"
            saved = json.loads((turn_dir / "assistant_message.json").read_text(encoding="utf-8"))
            assert saved == answer(session, turn, mode)
            expected = saved.get("reasoning", "") + saved["content"]
            assert (turn_dir / "output.txt").read_bytes().decode() == expected
            metadata = json.loads((turn_dir / "request.json").read_text(encoding="utf-8"))
            assert metadata["prompt_format"] == CHAT_FORMAT
            previous = messages
        if concurrent:
            replay = prefill_kv_replay.restore_turn(root, session, 3)
            assert replay["payload"]["messages"] == previous
            assert "prompt" not in replay["payload"]
            assert replay["previous_messages"] == previous[:-2]
    rows = summary["rounds"] + summary.get("warmups", [])
    assert all((r["ttft_s"] is None) == (mode == "empty") for r in rows)


def test_chat_sessions_advance_independently(tmp_path):
    with server(3, hold=True) as (url, state):
        result, _ = invoke(tmp_path, url, True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert state["independent_advance"]


def test_chat_replay_prepares_original_messages_with_reasoning(tmp_path, monkeypatch):
    with server(3) as (url, state):
        result, root = invoke(tmp_path, url, True)
    assert result.returncode == 0, result.stderr
    startup = tmp_path / "P.log"
    startup.write_text("[KV_FILE_REPLAY] enabled role=P\n", encoding="utf-8")
    monkeypatch.setattr(prefill_kv_replay, "live_contract", lambda paths: dict(tp_size=4))

    def http(url, path, body=None):
        if path == "/v1/models":
            return dict(data=[dict(id="glm", root="/models/glm")])
        assert path == "/tokenize"
        assert body["chat_template_kwargs"] == {"clear_thinking": False}
        return dict(tokens=render(body["messages"]))

    monkeypatch.setattr(prefill_kv_replay, "http_json", http)
    args = prefill_kv_replay.parser().parse_args(
        [
            "--run-dir",
            str(root),
            "--session",
            "0",
            "--round",
            "3",
            "--p-logs",
            str(startup),
            "--local-hostname",
            "127.0.0.1",
        ]
    )
    out = tmp_path / "replay"
    out.mkdir()
    restored, plan = prefill_kv_replay.prepare(args, out)
    sent = next(body for session, turn, body in state["requests"] if session == 0 and turn == 3)
    assert restored["payload"]["messages"] == sent["messages"]
    assert "prompt" not in restored["payload"]
    assert plan["previous_prompt_tokens"] == len(render(sent["messages"][:-2]))


def test_content_with_embedded_thinking_tags_is_preserved():
    stream = AssistantStream(True)
    text = "<think>未经解析的思考</think>正文"
    assert stream.take(dict(delta=dict(content=text))) == text
    assert stream.message() == dict(role="assistant", content=text)


def test_old_token_archive_remains_readable(tmp_path):
    for turn, (prefix, ids) in enumerate(((0, [1, 2]), (2, [3, 4]))):
        folder = tmp_path / "turns" / f"{turn:03d}"
        folder.mkdir(parents=True)
        (folder / "prompt_token_delta.json").write_text(
            json.dumps(dict(prefix_tokens=prefix, token_count=prefix + len(ids), token_ids=ids))
        )
        (folder / "request.json").write_text(
            json.dumps(dict(prompt_format=NATIVE_FORMAT, prompt_token_count=prefix + len(ids)))
        )
    assert restore_native_tokens(tmp_path, 1) == [1, 2, 3, 4]
