"""Phase 5, target 2: vLLM's OffloadingConnector (tiered KV offloading).

vLLM has a second, larger offloading system next to SimpleCPUOffloadConnector
(target 1): `OffloadingConnector`. Its scheduler side
(vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py, ~2,600
lines) plus `CPUOffloadingManager` decide what is stored, in which CPU
chunk, what a request may load, sync vs async loads, deferred stores,
preemption flushes, failed transfers and resets. All of that runs UNMODIFIED.

vLLM's offloading API is pluggable by design: an `OffloadingSpec` provides the
scheduler-side manager and the worker-side `OffloadingWorker` that moves the
bytes. `PhantomCPUOffloadingSpec` is vLLM's own `CPUOffloadingSpec` (same
manager, same chunk sizing) with one change: `get_worker()` returns
`PhantomOffloadWorker`, plain tensor copies instead of CUDA streams, which the
CPU backend does not have. It is selected through vLLM's documented
out-of-tree spec mechanism (`spec_name` + `spec_module_path`).

Like real DMA, a copy reads its source when it executes. Loads and stores run
on separate FIFO queues with their own delays, so a load can overtake a slower
store; expedited (sync) loads and flush waits complete immediately, as
`wait()` requires.

Faults (PHANTOM_OFFLOAD2_FAULT):
    store-reported-early  planted bug: a store is reported successful when
                          submitted, while the copy runs later
    load-fails            environment fault vLLM must handle: every 3rd load
                          fails (TransferResult.success=False); the request
                          must recover, and Phantom checks its output
    store-fails           environment fault: every 3rd store fails; the
                          manager must never let a request load those chunks

Only imported inside the vLLM environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from .offload_map import chunk_pairs

FAULTS = ("store-reported-early", "load-fails", "store-fails")

STATS: dict[str, int] = {
    "load_jobs": 0, "blocks_loaded": 0, "loads_failed": 0, "sync_loads": 0,
    "store_jobs": 0, "blocks_stored": 0, "stores_failed": 0,
    "flush_waits": 0, "faults_injected": 0,
}
TRACE: list[tuple] = []
TRACE_ON = False
_WORKERS: list["PhantomOffloadWorker"] = []


def pending_transfers() -> int:
    return sum(len(w._loads) + len(w._stores) for w in _WORKERS)


def _delay(kind: str) -> int:
    return int(os.environ.get(f"PHANTOM_OFFLOAD2_{kind.upper()}_DELAY") or 0)


@dataclass
class _Job:
    job_id: int
    kind: str  # "load" | "store"
    gpu_blocks: list[int]
    cpu_chunks: list[int]
    first_index: int  # logical block index of gpu_blocks[0] (GPULoadStoreSpec.block_indices)
    due: int
    fail: bool = False
    reported: bool = False  # result already handed to vLLM (store-reported-early)
    extra: dict = field(default_factory=dict)


class PhantomOffloadWorker(OffloadingWorker):
    """OffloadingWorker with tensor copies: one CPU buffer per canonical tensor,
    shaped (num_chunks, blocks_per_chunk * page_bytes)."""

    def __init__(self, kv_caches: CanonicalKVCaches, blocks_per_chunk: int, num_chunks: int):
        if len(kv_caches.group_data_refs) != 1:
            raise NotImplementedError("PhantomOffloadWorker supports one KV cache group")
        self.bpc = blocks_per_chunk
        self.gpu = [t.tensor for t in kv_caches.tensors]  # (num_blocks, page_bytes) int8
        self.page = [t.page_size_bytes for t in kv_caches.tensors]
        # zeros: a chunk nothing was ever stored into reads as "unwritten" to Phantom
        self.cpu = [torch.zeros((num_chunks, blocks_per_chunk * p), dtype=torch.int8)
                    for p in self.page]
        self.fault = os.environ.get("PHANTOM_OFFLOAD2_FAULT") or None
        if self.fault is not None and self.fault not in FAULTS:
            raise ValueError(f"unknown PHANTOM_OFFLOAD2_FAULT {self.fault!r}; choose from {FAULTS}")
        self._tick = 0
        self._loads: list[_Job] = []
        self._stores: list[_Job] = []
        self._results: list[TransferResult] = []
        self._n = {"load": 0, "store": 0}
        _WORKERS.append(self)

    # ------------------------------------------------------------ submission
    def _job(self, job_id: int, kind: str, gpu_spec: GPULoadStoreSpec,
             cpu_spec: LoadStoreSpec) -> _Job:
        assert len(gpu_spec.group_sizes) == 1
        self._n[kind] += 1
        fail = (self.fault == f"{kind}-fails" and self._n[kind] % 3 == 0)
        if fail:
            STATS["faults_injected"] += 1
        return _Job(job_id, kind, [int(b) for b in gpu_spec.block_ids],
                    [int(c) for c in np.asarray(cpu_spec.block_ids)],
                    int(gpu_spec.block_indices[0]), self._tick + _delay(kind), fail)

    def _log(self, phase: str, job: _Job) -> None:
        if TRACE_ON:
            TRACE.append((self._tick, phase, job.kind, job.job_id, len(job.gpu_blocks),
                          (tuple(job.gpu_blocks), tuple(job.cpu_chunks), job.first_index)))

    def submit_store(self, job_id: int, src_spec: GPULoadStoreSpec,
                     dst_spec: LoadStoreSpec) -> bool:
        job = self._job(job_id, "store", src_spec, dst_spec)
        self._stores.append(job)
        self._log("issue", job)
        if self.fault == "store-reported-early":
            # BUG (planted): report success now; the copy still runs later
            self._results.append(TransferResult(job_id, True))
            job.reported = True
            STATS["faults_injected"] += 1
        return True

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec,
                    expedited: bool = False) -> bool:
        job = self._job(job_id, "load", dst_spec, src_spec)
        if expedited:
            job.due = self._tick  # a sync load must not queue behind regular loads
            STATS["sync_loads"] += 1
            self._execute(job)
        else:
            self._loads.append(job)
        self._log("issue", job)
        return True

    # ------------------------------------------------------------- execution
    def _pairs(self, job: _Job):
        return chunk_pairs(job.gpu_blocks, job.cpu_chunks, job.first_index, self.bpc)

    def _execute(self, job: _Job) -> None:
        self._log("done", job)
        if job.fail:
            STATS["loads_failed" if job.kind == "load" else "stores_failed"] += 1
        else:
            for t, (gpu, cpu, page) in enumerate(zip(self.gpu, self.cpu, self.page)):
                for gb, chunk, sub in self._pairs(job):
                    window = slice(sub * page, (sub + 1) * page)
                    if job.kind == "load":
                        gpu[gb, :page] = cpu[chunk, window]
                    else:
                        cpu[chunk, window] = gpu[gb, :page]  # read now, like DMA
            key = "blocks_loaded" if job.kind == "load" else "blocks_stored"
            STATS[key] += len(job.gpu_blocks)
        STATS["load_jobs" if job.kind == "load" else "store_jobs"] += 1
        if not job.reported:
            self._results.append(TransferResult(job.job_id, not job.fail))

    def _advance(self, everything: bool = False, only: set[int] | None = None) -> None:
        for queue in (self._loads, self._stores):  # loads first when both are due
            if only is not None:
                # waiting on a job of a FIFO stream also completes everything
                # queued before it in that stream
                last = max((i for i, j in enumerate(queue) if j.job_id in only), default=-1)
                for _ in range(last + 1):
                    self._execute(queue.pop(0))
                continue
            while queue and (everything or queue[0].due <= self._tick):
                self._execute(queue.pop(0))  # FIFO within a direction

    def get_finished(self) -> list[TransferResult]:
        self._advance()
        self._tick += 1
        out, self._results = self._results, []
        return out

    def wait(self, job_ids: set[int]) -> None:
        STATS["flush_waits"] += 1
        self._advance(only=set(job_ids))

    def drain(self) -> None:
        self._advance(everything=True)


class PhantomCPUOffloadingSpec(CPUOffloadingSpec):
    """vLLM's CPUOffloadingSpec (same manager and sizing), with a copy engine
    that runs without CUDA."""

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:  # type: ignore[override]
        if not self._worker:
            self._worker = PhantomOffloadWorker(kv_caches, self.blocks_per_chunk,
                                                self.num_chunks)  # type: ignore[assignment]
        return self._worker  # type: ignore[return-value]
