# SPDX-License-Identifier: Apache-2.0
"""Copy only transport settings from live local workers to the replay seed."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

# These are existing HCCL/Mooncake settings, not vLLM feature flags. Never copy
# the whole worker environment (credentials, rank, device visibility, preload,
# model/debug settings, etc.) into the independent seed process.
TRANSPORT_ENV_KEYS = (
    "HCCL_INTRA_ROCE_ENABLE",
    "HCCL_INTRA_PCIE_ENABLE",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "HCCL_RDMA_TIMEOUT",
    "HCCL_RDMA_RETRY_CNT",
    "ASCEND_BUFFER_POOL",
    "ASCEND_ENABLE_USE_FABRIC_MEM",
    "ASCEND_AUTO_CONNECT",
    "ASCEND_USE_ASYNC_TRANSFER",
    "ASCEND_USE_SHORT_CONNECTION",
    "ASCEND_CONNECT_TIMEOUT",
    "ASCEND_TRANSFER_TIMEOUT",
    "ASCEND_RDMA_TC",
    "ASCEND_RDMA_SL",
    "ASCEND_THREAD_POOL_SIZE",
    "ASCEND_GLOBAL_RESOURCE_CONFIG",
)


def read_transport_environment(
    paths: list[Path], proc_root: Path = Path("/proc"), *, skip_stopped: bool = False
) -> dict:
    """Find the last cache-initializing worker per rank in the selected D logs.

    Read /proc inside the same host/container as those logs. The source is the
    currently running process, not shell defaults or a possibly stale template.
    """
    workers = {}
    for path in paths:
        for line in path.open(encoding="utf-8", errors="replace"):
            if "Mooncake payload layout:" not in line:
                continue
            match = re.search(r"\((Worker[^ ()]*) pid=(\d+)\)", line)
            if match:
                workers[(str(path), match[1])] = int(match[2])
    if not workers:
        raise ValueError("Cannot locate live service worker PIDs in transport logs; use the current startup logs")
    settings = None
    visible_devices = set()
    pids = []
    for pid in sorted(set(workers.values())):
        try:
            raw = (proc_root / str(pid) / "environ").read_bytes()
        except FileNotFoundError as error:
            if skip_stopped:
                continue
            raise ValueError(
                f"Cannot read live service worker {pid} environment; use the current startup logs"
            ) from error
        except OSError as error:
            raise ValueError(
                f"Cannot read live service worker {pid} environment; select current transport logs "
                "and run metadata collection in the serving container"
            ) from error
        env = dict(item.split("=", 1) for item in raw.decode(errors="surrogateescape").split("\0") if "=" in item)
        current = {key: env.get(key) for key in TRANSPORT_ENV_KEYS}
        if settings is not None and current != settings:
            different = [key for key in TRANSPORT_ENV_KEYS if current[key] != settings[key]]
            raise ValueError(
                f"Service workers disagree on transport settings: {different}; select matching transport logs"
            )
        settings = current
        pids.append(pid)
        visible = env.get("ASCEND_RT_VISIBLE_DEVICES")
        if visible:
            try:
                visible_devices.update(int(value.strip()) for value in visible.split(","))
            except ValueError as error:
                raise ValueError(f"Unsupported D device visibility: {visible!r}") from error
    if not pids:
        raise ValueError("Cannot locate live service workers in transport logs; run seed in the serving container")
    return dict(environment=settings, source_pids=pids, d_visible_devices=sorted(visible_devices))


def read_prefill_seed_transport(patterns: list[str]) -> dict:
    """Read seed-host settings, not the remote P host's IP/interface settings.

    P replay is launched on the client but its seed runs beside the snapshots.
    Logs copied from other hosts must not be supplied here. Ignore stopped
    workers left in old local logs, while checking all remaining live workers.
    """
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    return read_transport_environment(paths, skip_stopped=True)


def apply_transport_environment(config: dict, seed_device: int, *, check_device_overlap: bool = True) -> dict:
    """Apply the captured values before importing torch_npu or native Mooncake.

    Null means unset on D: remove a stale shell value on seed as well. Preserve
    seed's own device visibility; copying D's visibility can select the same
    physical device and cause an unsupported ADXL self-device connection.
    """
    env = config["environment"]
    if set(env) != set(TRANSPORT_ENV_KEYS) or any(
        value is not None and not isinstance(value, str) for value in env.values()
    ):
        raise ValueError("Invalid captured transport environment; update replay tools on client and snapshot host")
    if type(seed_device) is not int or seed_device < 0:
        raise ValueError("seed-device must be a nonnegative device index")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    physical_device = seed_device
    if visible:
        try:
            physical_device = int(visible.split(",")[seed_device].strip())
        except (ValueError, IndexError) as error:
            raise ValueError(f"seed-device {seed_device} is outside seed visibility {visible!r}") from error
    # D replay reads from a known co-located D endpoint. P replay gets settings
    # from local workers but they are not necessarily its consuming P/D workers;
    # their visibility is not a valid device exclusion list for that route.
    if check_device_overlap and physical_device in config["d_visible_devices"]:
        raise ValueError(
            f"Seed physical device {physical_device} overlaps target D devices {config['d_visible_devices']}; "
            "set --seed-device to a different visible device for cross-process ADXL transfer"
        )
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return dict(
        source_pids=config["source_pids"],
        seed_device=seed_device,
        seed_physical_device=physical_device,
        seed_visibility=visible,
        d_visible_devices=config["d_visible_devices"],
        environment={key: os.environ[key] for key in TRANSPORT_ENV_KEYS if key in os.environ},
    )
