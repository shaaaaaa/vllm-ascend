import threading
from collections.abc import Iterator
from contextlib import contextmanager


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.registered_buffers: dict[int, int] = {}
        self._temporary_refcounts: dict[int, int] = {}
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        """Register additional regions, idempotently.

        Live split handoff adds request-scoped host regions after the vLLM NPU
        cache was registered.  The previous process-wide boolean silently
        skipped those regions, which made a CPU destination unsafe.
        """
        if len(ptrs) != len(sizes):
            raise ValueError("Mooncake pointer and size counts must match")
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            for ptr, size in zip(ptrs, sizes):
                if ptr <= 0 or size <= 0:
                    raise ValueError("Mooncake memory regions must be positive")
                end = ptr + size
                containing = next(
                    (
                        (base, registered_size)
                        for base, registered_size in self.registered_buffers.items()
                        if base <= ptr and end <= base + registered_size
                    ),
                    None,
                )
                if containing is not None:
                    continue
                overlapping = any(
                    ptr < base + registered_size and base < end
                    for base, registered_size in self.registered_buffers.items()
                )
                if overlapping:
                    raise RuntimeError(
                        "Mooncake memory region partially overlaps an existing registration"
                    )
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
                self.registered_buffers[ptr] = size

    @contextmanager
    def temporary_registration(
        self, ptrs: list[int], sizes: list[int]
    ) -> Iterator[None]:
        """Lease request-scoped regions without releasing process KV buffers."""
        if len(ptrs) != len(sizes):
            raise ValueError("Mooncake pointer and size counts must match")
        leased_bases: list[int] = []
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, (
                "Transfer engine must be initialized"
            )
            try:
                for ptr, size in zip(ptrs, sizes):
                    if ptr <= 0 or size <= 0:
                        raise ValueError(
                            "Mooncake memory regions must be positive"
                        )
                    end = ptr + size
                    containing = next(
                        (
                            base
                            for base, registered_size
                            in self.registered_buffers.items()
                            if base <= ptr
                            and end <= base + registered_size
                        ),
                        None,
                    )
                    if containing is not None:
                        if containing in self._temporary_refcounts:
                            self._temporary_refcounts[containing] += 1
                            leased_bases.append(containing)
                        continue
                    if any(
                        ptr < base + registered_size and base < end
                        for base, registered_size
                        in self.registered_buffers.items()
                    ):
                        raise RuntimeError(
                            "Mooncake memory region partially overlaps an "
                            "existing registration"
                        )
                    ret_value = self.transfer_engine.register_memory(ptr, size)
                    if ret_value != 0:
                        raise RuntimeError(
                            "Mooncake memory registration failed."
                        )
                    self.registered_buffers[ptr] = size
                    self._temporary_refcounts[ptr] = 1
                    leased_bases.append(ptr)
            except Exception:
                self._release_temporary_locked(reversed(leased_bases))
                raise
        try:
            yield
        finally:
            with self.register_buffer_lock:
                self._release_temporary_locked(reversed(leased_bases))

    def _release_temporary_locked(self, bases) -> None:
        for base in bases:
            refs = self._temporary_refcounts.get(base)
            if refs is None:
                continue
            if refs > 1:
                self._temporary_refcounts[base] = refs - 1
                continue
            ret_value = self.transfer_engine.unregister_memory(base)
            if ret_value != 0:
                raise RuntimeError("Mooncake memory unregistration failed.")
            self._temporary_refcounts.pop(base, None)
            self.registered_buffers.pop(base, None)

global_te = GlobalTE()
