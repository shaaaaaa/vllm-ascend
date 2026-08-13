import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.registered_buffers: dict[int, int] = {}
        self._adopted_buffers: dict[int, int] = {}
        self._adopted_leases: dict[int, int] = {}
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
            # DSA shared-pool caches expose the same slab as one index view and
            # smaller latent views.  Register containing views first so nested
            # aliases are idempotent regardless of connector dictionary order.
            regions = sorted(
                zip(ptrs, sizes, strict=True),
                key=lambda region: (-region[1], region[0]),
            )
            registered: list[int] = []
            try:
                for ptr, size in regions:
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
                    registered.append(ptr)
            except Exception:
                failures = []
                for base in reversed(registered):
                    if self.transfer_engine.unregister_memory(base) != 0:
                        failures.append(base)
                    else:
                        self.registered_buffers.pop(base, None)
                if failures:
                    raise RuntimeError(
                        "Mooncake memory registration rollback failed for "
                        f"{len(failures)} region(s)."
                    )
                raise

    def adopt_registered_buffer(
        self,
        ptr: int,
        size: int,
        register: Callable[[], int] | None = None,
    ) -> bool:
        """Track a region registered by another owner of this native engine."""
        if ptr <= 0 or size <= 0:
            raise ValueError("Mooncake memory regions must be positive")
        end = ptr + size
        with self.register_buffer_lock:
            containing = next(
                (
                    (base, registered_size)
                    for base, registered_size in self.registered_buffers.items()
                    if base <= ptr and end <= base + registered_size
                ),
                None,
            )
            if containing is not None:
                base, registered_size = containing
                raise RuntimeError(
                    "Mooncake registration already has another owner"
                )
            if any(
                ptr < base + registered_size and base < end
                for base, registered_size in self.registered_buffers.items()
            ):
                raise RuntimeError(
                    "Mooncake memory region partially overlaps an existing registration"
                )
            if register is not None and register() != 0:
                raise RuntimeError("Mooncake memory registration failed")
            self.registered_buffers[ptr] = size
            self._adopted_buffers[ptr] = 1
            return True

    def release_adopted_buffer(
        self,
        ptr: int,
        size: int,
        unregister: Callable[[], int] | None = None,
    ) -> None:
        """Forget one externally owned region after its owner unregisters it."""
        with self.register_buffer_lock:
            refs = self._adopted_buffers.get(ptr)
            if refs is None:
                raise RuntimeError("Mooncake registration is not externally owned")
            if self._adopted_leases.get(ptr, 0):
                raise RuntimeError("Adopted Mooncake registration is in use")
            if self.registered_buffers.get(ptr) != size:
                raise RuntimeError("Adopted Mooncake registration size changed")
            if unregister is not None and unregister() != 0:
                raise RuntimeError("Mooncake memory unregistration failed")
            self._adopted_buffers.pop(ptr, None)
            self.registered_buffers.pop(ptr, None)

    @contextmanager
    def temporary_registration(
        self, ptrs: list[int], sizes: list[int]
    ) -> Iterator[None]:
        """Lease request-scoped regions without releasing process KV buffers."""
        if len(ptrs) != len(sizes):
            raise ValueError("Mooncake pointer and size counts must match")
        leased_bases: list[int] = []
        adopted_bases: list[int] = []
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
                        if containing in self._adopted_buffers:
                            self._adopted_leases[containing] = (
                                self._adopted_leases.get(containing, 0) + 1
                            )
                            adopted_bases.append(containing)
                        elif containing in self._temporary_refcounts:
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
                try:
                    self._release_temporary_locked(reversed(leased_bases))
                finally:
                    self._release_adopted_leases_locked(
                        reversed(adopted_bases)
                    )
                raise
        try:
            yield
        finally:
            with self.register_buffer_lock:
                try:
                    self._release_temporary_locked(reversed(leased_bases))
                finally:
                    self._release_adopted_leases_locked(
                        reversed(adopted_bases)
                    )

    def _release_temporary_locked(self, bases) -> None:
        failures: list[int] = []
        releases: dict[int, int] = {}
        for base in bases:
            releases[base] = releases.get(base, 0) + 1
        for base, release_count in releases.items():
            refs = self._temporary_refcounts.get(base)
            if refs is None:
                continue
            if refs > release_count:
                self._temporary_refcounts[base] = refs - release_count
                continue
            ret_value = self.transfer_engine.unregister_memory(base)
            if ret_value != 0:
                # Keep the native registration visible, but no request owns a
                # lease now. A later temporary use will retry unregistration.
                self._temporary_refcounts[base] = 0
                failures.append(base)
                continue
            self._temporary_refcounts.pop(base, None)
            self.registered_buffers.pop(base, None)
        if failures:
            raise RuntimeError(
                "Mooncake memory unregistration failed for "
                f"{len(failures)} region(s)."
            )

    def _release_adopted_leases_locked(self, bases) -> None:
        for base in bases:
            refs = self._adopted_leases[base]
            if refs > 1:
                self._adopted_leases[base] = refs - 1
            else:
                self._adopted_leases.pop(base, None)

global_te = GlobalTE()
