#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Review every recorded multi-turn answer, without deciding KV correctness."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

MIN_LATIN_WORDS = 40
MIN_REPEATED_ANSWER_CHARS = 24
MIN_COPIED_ANSWER_CHARS = 48


def _normalized(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).lower()


def review_answer(text: str, question: str = "", previous_output: str = "") -> dict:
    """Return reproducible text symptoms, not a model-accuracy verdict."""
    symptoms = []
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) >= MIN_LATIN_WORDS:
        common = Counter(words).most_common(3)
        if common[0][1] / len(words) >= 0.15 and sum(count for _, count in common) / len(words) >= 0.45:
            symptoms.append("repetitive_latin_output")
    if "\ufffd" in text:
        symptoms.append("replacement_characters")
    turns = list(re.finditer(r"(?:^|\n)\s*用户\s*[：:]", text))
    if turns:
        symptoms.append("generated_user_turn")
        first_answer = _normalized(text[: turns[0].start()])
        continuation = _normalized(text[turns[0].end() :])
        if len(first_answer) >= MIN_REPEATED_ANSWER_CHARS and first_answer[:MIN_REPEATED_ANSWER_CHARS] in continuation:
            symptoms.append("repeated_answer_prefix")
        if question:
            invented = re.findall(r"(?:^|\n)\s*用户\s*[：:]\s*(.*?)(?=\n\s*助手\s*[：:]|$)", text, re.S)
            if any(_normalized(value) == _normalized(question) for value in invented):
                symptoms.append("repeated_question")
    if previous_output:
        previous = _normalized(previous_output)
        if len(previous) >= MIN_COPIED_ANSWER_CHARS and previous[:MIN_COPIED_ANSWER_CHARS] in _normalized(text):
            symptoms.append("copies_previous_output")
    severe = {"repetitive_latin_output", "replacement_characters"}
    repeated = {"repeated_answer_prefix", "repeated_question", "copies_previous_output"}
    priority = (
        "degenerate_output"
        if severe.intersection(symptoms)
        else "repeated_output"
        if repeated.intersection(symptoms)
        else "review_continuation"
        if turns
        else "unflagged"
    )
    return dict(priority=priority, symptoms=symptoms, generated_user_turns=len(turns))


def read_client_log(path: Path) -> list[dict]:
    """Keep all answers and metrics, including incomplete pasted log fragments."""
    records: dict[str, dict] = {}
    active = None
    content: list[str] = []

    def finish(complete: bool) -> None:
        nonlocal active, content
        if active is not None:
            records[active].update(output="\n".join(content), output_complete=complete)
        active, content = None, []

    marker = re.compile(r"^\[MULTI_TURN_OUTPUT\] round=(\d+) session=(\d+) phase=(\S+) request_id=(\S+) (begin|end)$")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = marker.match(line)
        if match:
            turn, session, phase, request_id, action = match.groups()
            if action == "begin":
                finish(False)
                active = request_id
                content = []
                records.setdefault(request_id, {}).update(
                    request_id=request_id, round=int(turn), session_id=int(session), phase=phase
                )
            elif active == request_id:
                finish(True)
            continue
        if active is not None:
            if not line.startswith("[MULTI_TURN] "):
                content.append(line)
                continue
            finish(False)
        if line.startswith("[MULTI_TURN] {"):
            try:
                row = json.loads(line[len("[MULTI_TURN] ") :])
            except ValueError:
                continue
            if row.get("request_id"):
                records.setdefault(row["request_id"], {}).update(row)
    finish(False)
    previous: dict[int, dict] = {}
    result = sorted(records.values(), key=lambda row: (row["session_id"], row["round"]))
    for row in result:
        old = previous.get(row["session_id"], {})
        review = review_answer(row.get("output", ""), row.get("question", ""), old.get("output", ""))
        row.update(review)
        row["previous_suspect_request_id"] = (
            old.get("response_id") or old.get("request_id")
            if old.get("priority") in ("degenerate_output", "repeated_output")
            else None
        )
        row["output_present"] = "output" in row
        previous[row["session_id"]] = row
    return result
