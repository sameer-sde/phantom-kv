"""Phase 5, target 1: vLLM's SimpleCPUOffloadConnector, checked by Phantom.

SimpleCPUOffloadConnector keeps a second, larger KV cache in CPU memory: full
GPU blocks are stored there, and a later request whose prefix was evicted from
the GPU cache can load it back instead of recomputing it. The decisions (which
GPU block is stored into which CPU block, which CPU blocks a request may load,
when a block may be reused or evicted, what happens on preemption, abort and
reset) are made by ~1,500 lines of scheduler-side code,
vllm/v1/simple_kv_offload/manager.py. That code runs here UNMODIFIED.

Only the copy engine is replaced. vLLM's SimpleCPUOffloadWorker copies blocks
with CUDA streams and DMA, which the CPU backend does not have.
PhantomOffloadWorker executes exactly the same instructions (the block lists
in SimpleCPUOffloadMetadata) with plain tensor copies, and reports completions
through the same channels (finished_recving for loads, worker metadata for
stores). Like real DMA, a copy reads its source when it executes, not when it
was requested: with PHANTOM_OFFLOAD_DELAY=n a transfer completes n steps
after it was issued, so any GPU block the manager lets go of too early can be
overwritten in between.

Phantom's tags make every loaded block self-describing: if the manager loads
the wrong CPU block, a stale one, or one whose store never completed, the
next read of that slot is a violation.

    PHANTOM_OFFLOAD=1             install PhantomOffloadWorker (vllm_plugin.py)
    PHANTOM_OFFLOAD_DELAY=n       steps until a transfer completes (default 0)
    PHANTOM_OFFLOAD_FAULT=store-before-compute
                                  planted bug, the class of vLLM #45704: a store
                                  reads the GPU block before this step's
                                  forward pass has written it

Only imported inside the vLLM environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

FAULTS = ("store-before-compute",)

# Cumulative counters, read by the harness (same process: in-process engine).
STATS: dict[str, int] = {
    "load_events": 0, "blocks_loaded": 0,
    "store_events": 0, "blocks_stored": 0,
    "flushes": 0, "faults_injected": 0,
}


def pending_transfers() -> int:
    """Transfers issued but not yet completed, across all live workers."""
    return sum(len(w._ops) for w in _WORKERS)


_WORKERS: list["Any"] = []


def install() -> None:
    """Make vLLM's SimpleCPUOffloadConnector use PhantomOffloadWorker."""
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        simple_cpu_offload_connector as conn,
    )

    conn.SimpleCPUOffloadWorker = _worker_class()


def _worker_class():
    import torch

    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    @dataclass
    class _Op:
        kind: str  # "load" | "store"
        event: int
        src: list[int]
        dst: list[int]
        due: int
        snapshot: dict[str, Any] | None = field(default=None)  # store data, if taken early

    class PhantomOffloadWorker(SimpleCPUOffloadWorker):
        """SimpleCPUOffloadWorker with plain tensor copies instead of CUDA DMA."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._fault = os.environ.get("PHANTOM_OFFLOAD_FAULT") or None
            if self._fault is not None and self._fault not in FAULTS:
                raise ValueError(f"unknown PHANTOM_OFFLOAD_FAULT {self._fault!r}; "
                                 f"choose from {FAULTS}")
            self._tick = 0
            self._ops: list[_Op] = []
            self._done_loads: set[int] = set()
            self._early: dict[str, Any] | None = None
            _WORKERS.append(self)

        @staticmethod
        def _delay() -> int:
            return int(os.environ.get("PHANTOM_OFFLOAD_DELAY") or 0)

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
            if not kv_caches:
                return
            self.gpu_kv_caches = dict(kv_caches)
            first = next(iter(kv_caches.values()))
            self.device = first.device
            bytes_per_block = sum(t[0].numel() * t.element_size() for t in kv_caches.values())
            self.num_cpu_blocks = self._resolve_num_cpu_blocks(self.cpu_capacity_bytes,
                                                               bytes_per_block)
            # zeros: a load from a CPU block nothing was ever stored into
            # reads as "unwritten" to Phantom
            self.cpu_kv_caches = {
                name: torch.zeros((self.num_cpu_blocks,) + tuple(t.shape[1:]), dtype=t.dtype)
                for name, t in kv_caches.items()}

        def bind_connector_metadata(self, metadata) -> None:  # type: ignore[override]
            super().bind_connector_metadata(metadata)
            if (self._fault == "store-before-compute" and metadata.store_event >= 0
                    and metadata.store_gpu_blocks):
                # BUG (planted): read the GPU blocks now, before this step's
                # forward pass has written them
                self._early = {name: t[metadata.store_gpu_blocks].clone()
                               for name, t in self.gpu_kv_caches.items()}
                STATS["faults_injected"] += 1
            else:
                self._early = None

        def start_load_kv(self) -> None:
            md = self._connector_metadata
            if md is not None and md.load_cpu_blocks:
                self._ops.append(_Op("load", md.load_event, list(md.load_cpu_blocks),
                                     list(md.load_gpu_blocks), self._tick + self._delay()))

        def wait_for_save(self) -> None:
            md = self._connector_metadata
            if md is not None and md.store_gpu_blocks:
                self._ops.append(_Op("store", md.store_event, list(md.store_gpu_blocks),
                                     list(md.store_cpu_blocks), self._tick + self._delay(),
                                     snapshot=self._early))

        def _execute(self, op: _Op) -> None:
            if op.kind == "load":
                for name, gpu in self.gpu_kv_caches.items():
                    gpu[op.dst] = self.cpu_kv_caches[name][op.src]
                self._done_loads.add(op.event)
                STATS["load_events"] += 1
                STATS["blocks_loaded"] += len(op.dst)
            else:
                for name, gpu in self.gpu_kv_caches.items():
                    data = op.snapshot[name] if op.snapshot is not None else gpu[op.src]
                    self.cpu_kv_caches[name][op.dst] = data
                self._completed_store_events[op.event] = 1
                STATS["store_events"] += 1
                STATS["blocks_stored"] += len(op.dst)

        def _complete(self, everything: bool = False) -> None:
            still = []
            for op in self._ops:  # in issue order, like the FIFO copy streams
                if everything or op.due <= self._tick:
                    self._execute(op)
                else:
                    still.append(op)
            self._ops = still

        def get_finished(self, finished_req_ids: set[str]):  # type: ignore[override]
            self._complete()
            self._tick += 1
            md = self._connector_metadata
            finished: set[str] = set()
            for ev in sorted(self._done_loads):
                reqs = md.load_event_to_reqs.get(ev) if md is not None else None
                if reqs:
                    finished.update(reqs)
            self._done_loads.clear()
            in_flight = {(o.kind, o.event) for o in self._ops}
            self._pending_load_event_indices = {
                e for e in self._pending_load_event_indices if ("load", e) in in_flight}
            self._pending_store_event_indices = {
                e for e in self._pending_store_event_indices if ("store", e) in in_flight}
            return None, finished or None

        def handle_preemptions(self, kv_connector_metadata) -> None:  # type: ignore[override]
            if kv_connector_metadata.need_flush:
                STATS["flushes"] += 1
                self._complete(everything=True)

        def drain(self) -> None:
            """Let every in-flight transfer finish (time passing with no steps)."""
            self._complete(everything=True)

    return PhantomOffloadWorker
