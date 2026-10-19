# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator references for raw CPU pointers consumed by asynchronous graphs."""

from typing import Any


class SFASourceLease:
    """Retain actual allocations, not just Python views, until device completion.

    Construct only when a source snapshot changes. TensorMemoryObj reference
    counts also cover LayerPageMemoryObj and passive shared-slab views. On TP
    rank 0 the reference retains the real shared allocation. Proxy objects with
    no-op reference counts and tensor-only views cannot provide this guarantee.
    Compact sources instead retain explicit service lease holds locally.
    """

    def __init__(self, sources: tuple[Any, ...]) -> None:
        self.sources = sources
        self.owners: list[Any] = []
        self.leases: list[Any] = []
        if not any(source is not None for source in sources):
            return
        # LMCache is optional for other attention backends; import on binding.
        from lmcache.v1.memory_management import TensorMemoryObj

        try:
            owners: dict[int, Any] = {}
            lease_ids: set[int] = set()
            for source in sources:
                if source is None:
                    continue
                owner_lease = getattr(source, "owner_lease", None)
                if owner_lease is not None:
                    if id(owner_lease) not in lease_ids:
                        held = owner_lease.retain()
                        self.leases.append(held)
                        held.mark_imported()
                        lease_ids.add(id(owner_lease))
                    continue
                prepared_owners = getattr(source, "graph_owners", None)
                if prepared_owners is not None:
                    owners.update((id(owner), owner) for owner in prepared_owners)
                    continue
                for layer in source.layers:
                    if len(layer.memory_objs) != layer.chunk_ptrs_npu.numel():
                        raise RuntimeError("Async full SFA requires an allocator owner for every CPU source chunk")
                    for owner in layer.memory_objs:
                        if not isinstance(owner, TensorMemoryObj):
                            raise RuntimeError(
                                "Async full SFA requires ref-counted TensorMemoryObj sources, not proxies"
                            )
                        owners[id(owner)] = owner
            for owner in owners.values():
                if not owner.is_valid():
                    raise RuntimeError("Cannot retain an invalid full SFA source allocation")
                owner.ref_count_up()
                self.owners.append(owner)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release only this lease's references, after its event has completed."""
        while self.owners:
            self.owners[-1].ref_count_down()
            self.owners.pop()
        while self.leases:
            self.leases[-1].release(proof="fenced")
            self.leases.pop()
        self.sources = ()
