# SPDX-License-Identifier: Apache-2.0
"""Output symptoms must not silently filter requests or imply KV corruption."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from multi_turn_output_review import read_client_log, review_answer


def block(session, turn, text, *, metrics=True, complete=True):
    row = dict(
        request_id=f"client-{session}-{turn}", response_id=f"server-{session}-{turn}", session_id=session, round=turn
    )
    row["question"] = "主要矛盾呢？"
    label = f"round={turn:02d} session={session} phase=followup request_id={row['request_id']}"
    return (
        ("[MULTI_TURN] " + json.dumps(row) + "\n" if metrics else "")
        + f"[MULTI_TURN_OUTPUT] {label} begin\n{text}"
        + (f"\n[MULTI_TURN_OUTPUT] {label} end\n" if complete else "")
    )


def test_nonrepeating_invented_dialogue_is_not_labelled_corrupt():
    row = review_answer("应先完成试点。\n用户：为什么？\n助手：这样可以核实真实成本。")
    assert row["priority"] == "review_continuation"
    assert row["symptoms"] == ["generated_user_turn"]


def test_repeated_question_and_partial_answer_are_marked():
    answer = "有限资源与服务需求存在冲突，必须同时衡量公平、效率以及长期运行维护成本。"
    row = review_answer(answer + "\n用户：主要矛盾呢？\n助手：" + answer[:29], "主要矛盾呢？")
    assert row["priority"] == "repeated_output"
    assert "repeated_answer_prefix" in row["symptoms"]
    assert "repeated_question" in row["symptoms"]


def test_degenerate_english_and_copied_previous_output():
    answer = " request order not" * 40
    row = review_answer(answer, previous_output=answer)
    assert row["priority"] == "degenerate_output"
    assert "copies_previous_output" in row["symptoms"]


def test_log_parser_keeps_missing_ids_and_unfinished_output(tmp_path):
    path = tmp_path / "client.log"
    path.write_text(block(0, 1, "正常答案", metrics=False) + block(1, 1, "不完整", complete=False), encoding="utf-8")
    rows = read_client_log(path)
    assert len(rows) == 2
    assert rows[0].get("response_id") is None
    assert rows[0]["output_complete"] is True
    assert rows[1]["output_complete"] is False
    assert rows[1]["output"] == "不完整"


def test_rounds_are_ordered_with_history_of_anomalies(tmp_path):
    path = tmp_path / "client.log"
    bad = "request order not " * 40
    path.write_text(block(1, 2, "恢复连贯的答案") + block(1, 1, bad), encoding="utf-8")
    rows = read_client_log(path)
    assert [row["round"] for row in rows] == [1, 2]
    assert rows[1]["previous_suspect_request_id"] == "server-1-1"


def test_different_rounds_do_not_count_as_answer_repetition():
    row = review_answer("第一项是改善教育。第二项是改善医疗。第三项是改善住房。")
    assert row["priority"] == "unflagged"


def test_truncated_output_does_not_swallow_next_request(tmp_path):
    path = tmp_path / "client.log"
    path.write_text(block(0, 1, "截断的内容", complete=False) + "\n" + block(1, 1, "正常答案"), encoding="utf-8")
    rows = read_client_log(path)
    assert len(rows) == 2
    assert rows[0]["output_complete"] is False
    assert rows[1]["response_id"] == "server-1-1"
    assert rows[1]["output"] == "正常答案"
