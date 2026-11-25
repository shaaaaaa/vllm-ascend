# SPDX-License-Identifier: Apache-2.0
"""Read existing raw snapshot files on demand; no Torch/Mooncake/NPU runtime."""

from __future__ import annotations

import base64
import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from kv_replay_snapshot import PREFIX, snapshot_metadata, validate_tp_copies

COPY_BYTES = 4 * 1024 * 1024
TAG = "lmcache.tag.file_replay"


class SnapshotFiles:
    def __init__(self, plan: dict):
        self.tokens, self.rows, self.comparisons = snapshot_metadata(plan)
        self.role = "D" if plan.get("replay_mode") == "decode_receive" else "PD"
        schema = plan["contract"]["schema"]
        self.chunk = schema["chunk_size"]
        self.manifest = dict(
            role=self.role,
            model=plan["model_path"],
            signature=plan["contract"]["signature"],
            groups=[count for count, _ in schema["runtime_kv_groups"]],
            chunk_size=self.chunk,
            tokens=self.tokens,
            request_id=plan["previous"],
        )
        self.nonce = uuid.uuid4().hex
        self.completed = set()
        self.layer_completions = {}
        self.readers = {}
        self.role_completed = {"P": set(), "D": set()}
        self.bytes_sent = 0
        self.errors = []
        self.lock = threading.Lock()

    def page(self, group: int, start: int, end: int, layer: int | None = None) -> list:
        if group not in (0, 1) or start < 0 or start % self.chunk or end != min(start + self.chunk, len(self.tokens)):
            raise ValueError("Invalid snapshot page range")
        if not start < end <= len(self.tokens):
            raise ValueError("Snapshot page exceeds the selected historical frontier")
        if layer is not None and not 0 <= layer < self.manifest["groups"][group]:
            raise ValueError("Invalid snapshot layer")
        return [
            (row, path)
            for (g, layer_id, _), (row, path) in sorted(self.rows.items())
            if g == group and (layer is None or layer_id == layer)
        ]

    def stats(self) -> dict:
        with self.lock:
            expected = 2 * ((len(self.tokens) + self.chunk - 1) // self.chunk)
            return dict(
                source="raw_snapshot_files",
                request_id=self.manifest["request_id"],
                completed_pages=len(self.completed),
                expected_pages=expected,
                groups=sorted({group for group, _, _ in self.completed}),
                bytes_sent=self.bytes_sent,
                errors=list(self.errors),
                role_completed_pages={role: len(pages) for role, pages in self.role_completed.items()},
                readers=[dict(info, completed_pages=len(pages)) for info, pages in self.readers.values()],
            )

    def handler(self):
        source = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(120)

            def do_POST(self):
                prefix = "/" + source.nonce + "/ack/"
                try:
                    if not self.path.startswith(prefix):
                        raise ValueError("Invalid acknowledgement")
                    parts = list(map(int, self.path[len(prefix) :].split("/")))
                    group, start, end = parts[:3]
                    layer = parts[3] if len(parts) == 4 else None
                    if len(parts) not in (3, 4):
                        raise ValueError("Invalid acknowledgement")
                    source.page(group, start, end, layer)
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 1024:
                        raise ValueError("Missing reader identity; update the LMCache file reader")
                    reader = json.loads(self.rfile.read(length))
                    if (
                        reader.get("role") not in source.role
                        or reader.get("role") not in ("P", "D")
                        or not isinstance(reader.get("host"), str)
                        or type(reader.get("pid")) is not int
                        or type(reader.get("rank")) is not int
                    ):
                        raise ValueError("Invalid reader identity")
                    reader["ip"] = self.client_address[0]
                    reader_key = tuple(reader[k] for k in ("host", "ip", "pid", "role", "rank"))
                except (ValueError, TypeError, AttributeError):
                    self.send_error(404)
                    return
                with source.lock:
                    identity = (group, start, end)
                    info, pages = source.readers.setdefault(reader_key, (reader, set()))
                    if layer is None:
                        pages.add(identity)
                    else:
                        done = source.layer_completions.setdefault((reader_key, identity), set())
                        done.add(layer)
                        if len(done) == source.manifest["groups"][group]:
                            pages.add(identity)
                    if identity in pages:
                        source.completed.add(identity)
                        source.role_completed[info["role"]].add(identity)
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                prefix = "/" + source.nonce + "/"
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                path = self.path[len(prefix) :]
                if path in ("manifest", "stats"):
                    data = json.dumps(source.manifest if path == "manifest" else source.stats()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                try:
                    parts = path.split("/")
                    layer = None
                    if parts[0] == "page":
                        _, g, a, b = parts
                    elif parts[0] == "layer":
                        _, g, layer_text, a, b = parts
                        layer = int(layer_text)
                    else:
                        raise ValueError("Invalid file endpoint")
                    group, start, end = int(g), int(a), int(b)
                    rows = source.page(group, start, end, layer)
                except ValueError:
                    self.send_error(404)
                    return
                size = (end - start) * sum(row["row_bytes"] for row, _ in rows)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                try:
                    # Files already contain one planar component per file.
                    # Stream layer-major slices directly, never create another
                    # full KV copy or place a page into a native cache store.
                    for row, file_path in rows:
                        with file_path.open("rb") as stream:
                            stream.seek(start * row["row_bytes"])
                            remaining = (end - start) * row["row_bytes"]
                            while remaining:
                                data = stream.read(min(COPY_BYTES, remaining))
                                if not data:
                                    raise OSError("Snapshot file truncated during replay")
                                self.wfile.write(data)
                                remaining -= len(data)
                    self.wfile.flush()
                    with source.lock:
                        source.bytes_sent += size
                except (OSError, ValueError) as error:
                    self.close_connection = True
                    with source.lock:
                        if len(source.errors) < 8:
                            source.errors.append(str(error))

        return Handler


def serve(plan: dict) -> None:
    print(PREFIX + json.dumps(dict(status="validating_files", native_transport=False)), flush=True)
    source = SnapshotFiles(plan)
    validate_tp_copies(source.comparisons, plan["workers"])
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    release = Path(plan["release_file"])
    release.parent.mkdir(parents=True, exist_ok=True)
    if release.exists():
        raise ValueError("File replay release marker already exists")
    host = plan["file_host"]
    if not host or any(c in host for c in "/@?#"):
        raise ValueError("Invalid advertised snapshot file host")
    with ThreadingHTTPServer(("0.0.0.0", plan.get("file_port", 0)), source.handler()) as server:
        server.daemon_threads = True
        server.timeout = 0.5
        url = f"http://{host}:{server.server_port}/{source.nonce}"
        print(
            PREFIX
            + json.dumps(
                dict(
                    status="ready",
                    source="files",
                    file_url=url,
                    role=source.role,
                    tokens=len(source.tokens),
                    native_transport=False,
                )
            ),
            flush=True,
        )
        deadline = time.monotonic() + plan["hold_seconds"]
        while not stopped.is_set() and not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("File replay lease expired")
            server.handle_request()


def configure_plan(args, plan: dict) -> None:
    host = args.local_hostname
    if not host and args.snapshot_host:
        host = args.snapshot_host.rsplit("@", 1)[-1]
    if not host:
        raise ValueError("Local file replay requires --local-hostname reachable from P/D")
    if not 0 <= args.file_port <= 65535:
        raise ValueError("file-port must be between 0 and 65535")
    plan.update(file_host=host, file_port=args.file_port)


def attach_source(payload: dict, owner, output: Path) -> None:
    info = owner.file_info
    options = dict(url=info["file_url"], role=info["role"])
    tag = base64.urlsafe_b64encode(json.dumps(options, separators=(",", ":")).encode()).decode().rstrip("=")
    transfer = dict(payload.get("kv_transfer_params") or {})
    if any(k.startswith("lmcache.tag.") for k in transfer):
        raise ValueError("Original request carries extra cache tags; refusing a mismatched file replay")
    transfer[TAG] = tag
    payload["kv_transfer_params"] = transfer
    payload["cache_salt"] = "file-replay-" + urlsplit(info["file_url"]).path.strip("/")
    from multi_turn_kv_reuse_concurrent import save_json

    save_json(output / "request.json", payload)


def source_stats(owner, output: Path) -> dict:
    from multi_turn_kv_reuse_concurrent import save_json

    opener = build_opener(ProxyHandler({}))
    with opener.open(owner.file_info["file_url"] + "/stats", timeout=30) as response:
        stats = json.load(response)
    save_json(output / "file_reads.json", stats)
    required_role = "P" if owner.file_info["role"] == "PD" else "D"
    if (
        stats["errors"]
        or stats["role_completed_pages"][required_role] != stats["expected_pages"]
        or stats["groups"] != [0, 1]
    ):
        raise RuntimeError(f"File replay did not read the entire selected KV prefix: {stats}")
    for reader in stats["readers"]:
        print(PREFIX + json.dumps(dict(status="reader_completed", **reader)), flush=True)
    return stats


def require_enabled(paths, role: str) -> None:
    marker = f"[KV_FILE_REPLAY] enabled role={role}"

    def found(path):
        with path.open(encoding="utf-8", errors="replace") as stream:
            return any(marker in line for line in stream)

    if not any(found(path) for path in paths):
        raise ValueError(
            f"{role} file replay is not enabled in the supplied startup logs. Update LMCache, set "
            "LMCACHE_FILE_REPLAY_ENABLED=true and restart this service once before replay."
        )
