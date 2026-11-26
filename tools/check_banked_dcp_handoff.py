# SPDX-License-Identifier: Apache-2.0
"""Qualify one cold completion through four-bank DCP prefill and compact graph decode.

Install BankedDcpWitness as a worker extension in the participating engine(s).
The two RPCs bracket a single qualification request. No hooks remain for timing.
"""

import argparse
import json
import random
from uuid import uuid4


class BankedDcpWitness:
    def arm_banked_dcp_witness(self, check_id):
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.forward_context import get_forward_context

        runner = self.model_runner
        if hasattr(self, "_banked_dcp_witness"):
            raise RuntimeError("A handoff check is already armed on this worker")
        # Restore the class method after the check, or a pre-existing instance override.
        original = runner._model_forward
        override = vars(runner).get("_model_forward")
        tp = get_tp_group()
        report = {"rank": tp.rank_in_group, "tp": tp.world_size, "world": runner.dcp_size, "prefill": {}, "decode": {}}

        def observe(*args, **kwargs):
            ctx = get_forward_context()
            graph = getattr(runner, "_sfa_full_graph", None)
            before = getattr(graph, "replay_count", 0)
            result = original(*args, **kwargs)
            if ctx.staged_sfa_graph_dummy_run:
                return result
            metadata = ctx.attn_metadata
            if not isinstance(metadata, dict):
                return result
            for meta in metadata.values():
                batch = getattr(meta, "prefill_dcp", None)
                if batch is not None and getattr(getattr(meta, "sfa_dcp", None), "owner_rows", None) is None:
                    for req, (start, end) in batch.coverage.items():
                        # Retain chunk intervals once, independent of layer count.
                        chunks = report["prefill"].setdefault(req, [])
                        chunks.append([start, end, len(batch.sources), batch.count])
                    break
            else:
                if runner.layerwise_prefill_p_node and runner.dcp_size == 1:
                    inputs = runner.input_batch
                    banks = len(inputs.layerwise_prefill_block_tables)
                    for i, req in enumerate(inputs.req_ids):
                        start, stop = int(inputs.num_computed_tokens_cpu[i]), int(runner.seq_lens.np[i])
                        report["prefill"].setdefault(req, []).append([start, stop, banks, None])
            if graph is not None and graph.replay_count > before:
                route = ctx.staged_sfa_route
                # Only accept an actual request with a positive CPU-backed frontier and
                # owner rows in the metadata that the completed replay consumed.
                if any(getattr(getattr(m, "sfa_dcp", None), "owner_rows", None) is not None for m in metadata.values()):
                    for row, (req, frontier) in enumerate(
                        zip(runner.input_batch.req_ids, route.frontiers, strict=True)
                    ):
                        if frontier > 0:
                            placement = runner._dcp_indexer_owners[req]
                            record = report["decode"].setdefault(
                                req,
                                {
                                    "owner": int(placement[0]),
                                    "epoch": int(placement[1]),
                                    "replays": 0,
                                },
                            )
                            if tuple(placement) != (record["owner"], record["epoch"]):
                                record["placement_changed"] = True
                            record.update(
                                frontier=int(frontier),
                                replays=record["replays"] + 1,
                                computed=int(runner.input_batch.num_computed_tokens_cpu[row]),
                            )
            return result

        runner._model_forward = observe
        self._banked_dcp_witness = check_id, override, report

    def finish_banked_dcp_witness(self, check_id):
        state = getattr(self, "_banked_dcp_witness", None)
        if state is None or state[0] != check_id:
            return None
        _, override, report = state
        del self._banked_dcp_witness
        if override is None:
            del self.model_runner._model_forward
        else:
            self.model_runner._model_forward = override
        return report


def verify(reports, phase, request_id, prompt_len, expected_dcp=None):
    if not reports or any(not isinstance(r, dict) for r in reports):
        raise ValueError(f"No {phase} worker evidence")
    ranks = sorted(reports, key=lambda r: r["rank"])
    world = len(ranks)
    dcp = world if expected_dcp is None else expected_dcp
    if world not in (2, 4, 8) or [(r["rank"], r["tp"], r["world"]) for r in ranks] != [
        (i, world, dcp) for i in range(world)
    ]:
        raise ValueError(f"Incomplete {phase} DCP group")
    rows = []
    for rank in ranks:
        records = rank[phase]
        # vLLM may append an engine-local random suffix after the prompt index.
        matches = [v for req, v in records.items() if req == request_id + "-0" or req.startswith(request_id + "-0-")]
        if len(matches) != 1:
            raise ValueError(f"{phase} rank {rank['rank']} expected one observation for {request_id}: {records}")
        rows.append(matches[0])
    if any(row != rows[0] for row in rows):
        raise ValueError(f"{phase} ranks disagree: {rows}")
    if phase == "prefill":
        end = 0
        for start, stop, banks, historical_columns in rows[0]:
            exchange_ok = (
                historical_columns is None
                if dcp == 1
                else (historical_columns is not None and (historical_columns > 0) == (start > 0))
            )
            if start != end or stop <= start or banks != 4 or not exchange_ok:
                raise ValueError(f"Invalid four-bank prefill or prefix exchange: {rows[0]}")
            end = stop
        if end != prompt_len or len(rows[0]) < 2:
            raise ValueError(f"Expected multiple cold prefill chunks covering {prompt_len} tokens: {rows[0]}")
    else:
        r = rows[0]
        if (
            not 0 < r["frontier"] <= prompt_len
            or not 0 <= r["owner"] < world
            or r["epoch"] <= 0
            or r.get("placement_changed")
            or r["replays"] < 2
            or r["computed"] <= prompt_len
        ):
            raise ValueError(f"Invalid compact graph placement: {rows[0]}")
    return rows[0]


def main():
    from contextlib import ExitStack

    import requests

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:8080", help="Completion service or P/D proxy base URL")
    cli.add_argument("--prefiller", default="http://127.0.0.1:7910", help="Prefill worker RPC base URL")
    cli.add_argument("--decoder", default="http://127.0.0.1:7920", help="Decode worker RPC base URL")
    cli.add_argument("--model", default="glm", help="Served model name")
    cli.add_argument("--prefill-dcp", type=int, choices=(1, 2, 4, 8), help="Expected sender DCP size (default: TP)")
    cli.add_argument("--prompt-tokens", type=int, default=60000)
    cli.add_argument("--output-tokens", type=int, default=16)
    cli.add_argument("--seed", type=int, help="Match input IDs between arms using fresh cache namespaces")
    args = cli.parse_args()
    if args.prompt_tokens < 2 or args.output_tokens < 3:
        cli.error("A positive history and at least three output tokens are required")
    p, d = (url.rstrip("/") for url in (args.prefiller, args.decoder))

    def post(url, body):
        response = requests.post(url, json=body, timeout=(10, 1800))
        response.raise_for_status()
        return response.json()

    check_id = uuid4().hex

    def rpc(url, method):
        return post(url + "/collective_rpc", {"method": method, "args": [check_id], "timeout": 120})["results"]

    reports = {}
    with ExitStack() as cleanup:
        for url in dict.fromkeys((p, d)):
            cleanup.callback(lambda url=url: reports.update({url: rpc(url, "finish_banked_dcp_witness")}))
            rpc(url, "arm_banked_dcp_witness")
        # A unique prefix prevents a preceding run from supplying a cold-cache hit.
        rng = random.Random(uuid4().int if args.seed is None else args.seed)
        response = post(
            args.url.rstrip("/") + "/v1/completions",
            {
                "model": args.model,
                "prompt": [rng.randrange(256, 8192) for _ in range(args.prompt_tokens)],
                "temperature": 0,
                "ignore_eos": True,
                "max_tokens": args.output_tokens,
                "return_token_ids": True,
                "add_special_tokens": False,
            },
        )
    usage = response["usage"]
    if usage["prompt_tokens"] != args.prompt_tokens or usage["completion_tokens"] != args.output_tokens:
        raise ValueError(f"Unexpected completion coverage: {usage}")
    req = response["id"]
    tokens = response["choices"][0]["token_ids"]
    if len(tokens) != args.output_tokens:
        raise ValueError("Incomplete generated token IDs")
    print(
        json.dumps(
            {
                "request": req,
                "usage": usage,
                "generated_token_ids": tokens,
                "prefill_chunks": verify(reports[p], "prefill", req, args.prompt_tokens, args.prefill_dcp),
                "decode": verify(reports[d], "decode", req, args.prompt_tokens),
            }
        )
    )


if __name__ == "__main__":
    main()
