#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standard chat messages and response preservation for the multi-turn clients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CHAT_FORMAT = "native-messages-v1"
# Read-only compatibility for archives from the earlier token-list client.
NATIVE_FORMAT = "native-chat-v1"
TEMPLATE_KWARGS = (("enable_thinking", False), ("clear_thinking", False))


def add_chat_arguments(parser) -> None:
    parser.add_argument(
        "--prompt-format",
        choices=("native", "legacy"),
        default="native",
        help="Standard chat completions (default), or the previous raw-text benchmark",
    )
    # Accept old command lines; standard chat needs no separate tokenizer URL.
    parser.add_argument("--tokenizer-url", help=argparse.SUPPRESS)


def initial_user_text(text: str) -> str:
    """Remove only the benchmark's old trailing assistant cue, not article text."""
    for suffix in ("\r\n助手：\r\n", "\r\n助手：\n", "\n助手：\r\n", "\n助手：\n"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def read_assistant(directory: Path) -> dict:
    message = json.loads((directory / "assistant_message.json").read_text(encoding="utf-8"))
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        raise ValueError(f"Invalid assistant message in {directory}")
    if "reasoning" in message and not isinstance(message["reasoning"], str):
        raise ValueError(f"Invalid assistant reasoning in {directory}")
    return message


class AssistantStream:
    """Keep chat reasoning and visible content separate, including truncated turns."""

    def __init__(self, chat: bool):
        self.chat = chat
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.has_reasoning = False

    def take(self, choice: dict) -> str:
        if not self.chat:
            text = choice.get("text", "")
            if not isinstance(text, str):
                raise RuntimeError("Completion text must be a string")
            return text
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        reasoning = delta.get("reasoning")
        if reasoning is None:
            reasoning = delta.get("reasoning_content")
        if not isinstance(text, str) or (reasoning is not None and not isinstance(reasoning, str)):
            raise RuntimeError("Chat content and reasoning must be strings")
        self.content.append(text)
        if reasoning is not None:
            self.has_reasoning = True
            self.reasoning.append(reasoning)
        # Count the first reasoning OR content fragment for client TTFT.
        return (reasoning or "") + text

    def message(self) -> dict:
        result = dict(role="assistant", content="".join(self.content))
        if self.has_reasoning:
            result["reasoning"] = "".join(self.reasoning)
        return result


class NativeDialogue:
    def __init__(self, initial: str):
        self.messages = [dict(role="user", content=initial)]
        self.last_assistant: dict | None = None

    def append(self, question: str) -> None:
        if self.last_assistant is None:
            raise RuntimeError("Previous assistant response is missing")
        self.messages.extend((self.last_assistant, dict(role="user", content=question)))
        self.last_assistant = None

    def prepare(self, payload: dict, directory: Path) -> dict:
        prepared = {key: value for key, value in payload.items() if key != "prompt"}
        prepared.update(messages=list(self.messages), chat_template_kwargs=dict(clear_thinking=False))
        metadata_path = directory / "request.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            prompt_format=CHAT_FORMAT,
            prompt_source="initial user + prior assistant_message.json + added_input.txt",
            chat_template_kwargs=prepared["chat_template_kwargs"],
        )
        write_json(metadata_path, metadata)
        return prepared


def complete_chat(client, dialogue: NativeDialogue, payload: dict, directory: Path):
    text, stats = client.complete(dialogue.prepare(payload, directory), directory)
    dialogue.last_assistant = read_assistant(directory)
    return text, stats


def restore_native_tokens(directory: Path, turn: int) -> list[int]:
    """Read old token-list archives; new chat requests do not create token deltas."""
    tokens: list[int] = []
    for current in range(turn + 1):
        folder = directory / "turns" / f"{current:03d}"
        delta = json.loads((folder / "prompt_token_delta.json").read_text(encoding="utf-8"))
        metadata = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        added = delta.get("token_ids")
        if (
            delta.get("prefix_tokens") != len(tokens)
            or not isinstance(added, list)
            or any(type(value) is not int or value < 0 for value in added)
        ):
            raise ValueError(f"Invalid native token delta at round {current}")
        tokens.extend(added)
        if len(tokens) != delta.get("token_count") or len(tokens) != metadata.get("prompt_token_count"):
            raise ValueError(f"Native prompt token count differs at round {current}")
    return tokens
