"""PhantomFaultConnector: vLLM's ExampleConnector with planted transfer faults.

Selected with
    KVTransferConfig(kv_connector="PhantomFaultConnector",
                     kv_connector_module_path="phantomkv.vllm_fault_connector", ...)
and a fault chosen by the environment variable PHANTOM_PD_FAULT:

  (unset)      behaves exactly like ExampleConnector
  swap         on load, serve ANOTHER prompt's KV of the same size: the
               decode engine silently receives someone else's context
  silent-drop  on load, report success but write nothing: the decode engine
               reads whatever its slots held before
  duplicate    (async loads only) the KV lands once, correctly, and then a
               second, stale copy of the same write lands 6 steps later, like
               a retried RDMA write; harmless while the request still owns
               its blocks, corrupting once they have been freed and reused

Both are silent: shapes match and nothing raises, so vLLM itself cannot
notice. Faults only affect loads (decode side); saves are untouched.
Every injected fault is appended to $PHANTOM_PD_FAULT_LOG (JSON lines) so a
run can prove the fault actually fired.

Only imported by vLLM inside the vLLM environment.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import ExampleConnector

FAULTS = ("swap", "silent-drop", "duplicate")  # duplicate: async loads only


class PhantomFaultConnector(ExampleConnector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fault = os.environ.get("PHANTOM_PD_FAULT") or None
        if self._fault is not None and self._fault not in FAULTS:
            raise ValueError(f"unknown PHANTOM_PD_FAULT {self._fault!r}; choose from {FAULTS}")
        self._fault_log = os.environ.get("PHANTOM_PD_FAULT_LOG")
        self._loading = False
        # Decode engine in seeded PD fuzzing: never save. Otherwise a request
        # computed locally in one run would have "remote" KV on the next run of
        # the same seed, and replays (and shrinking) would not be deterministic.
        self._no_save = os.environ.get("PHANTOM_PD_NO_SAVE") == "1"

    # ------------------------------------------------------------ helpers
    def _record(self, **event: Any) -> None:
        if self._fault_log:
            with open(self._fault_log, "a") as f:
                f.write(json.dumps({"fault": self._fault, **event}) + "\n")

    def _swap_target(self, path: str) -> str:
        """Another prompt's file for the same layer with the same byte size
        (so the tensor shape matches and nothing raises). Deterministic."""
        folder, fname = os.path.split(path)
        if not os.path.exists(path):
            return path
        size = os.path.getsize(path)
        candidates = sorted(
            d for d in os.listdir(self._storage_path)
            if os.path.join(self._storage_path, d) != folder
            and os.path.exists(os.path.join(self._storage_path, d, fname))
            and os.path.getsize(os.path.join(self._storage_path, d, fname)) == size)
        if not candidates:
            return path
        # rotate: pick the candidate right after our own folder name
        own = os.path.basename(folder)
        target = next((c for c in candidates if c > own), candidates[0])
        return os.path.join(self._storage_path, target, fname)

    # ----------------------------------------------------------- overrides
    def _generate_filename_debug(self, layer_name, token_ids, mm_hashes):  # type: ignore[override]
        path = super()._generate_filename_debug(layer_name, token_ids, mm_hashes)
        if self._loading and self._fault == "swap":
            target = self._swap_target(path)
            if target != path:
                self._record(layer=layer_name, served=os.path.basename(os.path.dirname(target)),
                             instead_of=os.path.basename(os.path.dirname(path)))
            return target
        return path

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if self._no_save:
            return
        super().save_kv_layer(*args, **kwargs)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:  # type: ignore[override]
        if self._fault == "silent-drop":
            meta = self._get_connector_metadata()
            n = sum(1 for r in getattr(meta, "requests", []) if not r.is_store)
            if n:
                self._record(dropped_loads=n)
            return  # BUG (planted): claim the load happened, write nothing
        self._loading = True
        try:
            super().start_load_kv(forward_context, **kwargs)
        finally:
            self._loading = False


# ------------------------------------------------------------------ async loads
from dataclasses import dataclass, field  # noqa: E402

import safetensors.torch  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E402
    ExampleConnectorMetadata,
    ReqMeta,
)


@dataclass
class PhantomAsyncMetadata(ExampleConnectorMetadata):
    # (scheduler request id, what to load) for loads starting this step
    async_loads: list[tuple[str, ReqMeta]] = field(default_factory=list)


class PhantomAsyncFaultConnector(PhantomFaultConnector):
    """ExampleConnector with ASYNCHRONOUS loads, for abort-mid-transfer tests.

    A load is reported as async (`get_num_new_matched_tokens` -> (n, True)), so
    vLLM parks the request in WAITING_FOR_REMOTE_KVS with its blocks allocated.
    The KV then lands PHANTOM_PD_ASYNC_DELAY engine steps later (default 3),
    like an RDMA read in flight: one-sided, written into the blocks it was
    given at start, regardless of who owns them by then. Completion is
    reported through get_finished(), as real async connectors do.

    Correct vLLM keeps an aborted request's blocks until that completion
    arrives (scheduler.finish_requests: delay_free_blocks for
    WAITING_FOR_REMOTE_KVS). Every load start and landing is logged to
    PHANTOM_PD_FAULT_LOG; the planted transfer faults (swap, silent-drop)
    apply to async loads too.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending: dict[str, tuple[list[int], list[int]]] = {}  # scheduler side
        self._inflight: list[tuple[int, str, ReqMeta, bool]] = []  # worker side; bool = duplicate
        self._done: set[str] = set()
        self._tick = 0
        self._kv: dict[str, Any] = {}

    # ---------------------------------------------------------- scheduler side
    def get_num_new_matched_tokens(self, request, num_computed_tokens):  # type: ignore[override]
        n, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        return n, bool(n)  # every external hit is loaded asynchronously

    def update_state_after_alloc(self, request, blocks, num_external_tokens):  # type: ignore[override]
        if num_external_tokens > 0:
            self._pending[request.request_id] = (
                list(request.prompt_token_ids or []), list(blocks.get_block_ids()[0]))

    def build_connector_meta(self, scheduler_output):  # type: ignore[override]
        base = super().build_connector_meta(scheduler_output)
        meta = PhantomAsyncMetadata(requests=base.requests)
        for rid, (tokens, block_ids) in self._pending.items():
            meta.async_loads.append(
                (rid, ReqMeta.make_meta(tokens, block_ids, self._block_size, False, [])))
        self._pending.clear()
        return meta

    # ------------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches):  # type: ignore[override]
        super().register_kv_caches(kv_caches)
        self._kv = dict(kv_caches)

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:  # type: ignore[override]
        super().start_load_kv(forward_context, **kwargs)  # sync loads (none in async mode)
        meta = self._get_connector_metadata()
        self._tick += 1
        # read per load: seeded PD fuzzing (phantomkv/vllm_pd.py) sets a delay per run
        delay = int(os.environ.get("PHANTOM_PD_ASYNC_DELAY") or 3)
        for rid, rm in getattr(meta, "async_loads", []):
            self._inflight.append((self._tick + delay, rid, rm, False))
            self._record(event="started", req=rid, tick=self._tick, delay=delay)
        still = []
        for land_at, rid, rm, dup in self._inflight:
            if land_at > self._tick:
                still.append((land_at, rid, rm, dup))
            elif dup:
                self._land(rid, rm, duplicate=True)  # not reported: vLLM never asked for it
            else:
                self._land(rid, rm)
                self._done.add(rid)
                if self._fault == "duplicate":  # planted: a stale copy of this write lands later
                    still.append((self._tick + 6, rid, rm, True))
                    self._record(event="dup-started", req=rid, tick=self._tick)
        self._inflight = still

    def _land(self, rid: str, rm: ReqMeta, duplicate: bool = False) -> None:
        n = len(rm.slot_mapping)
        dropped = self._fault == "silent-drop"  # planted: report done, write nothing
        if n and not dropped:
            blocks = rm.slot_mapping // self._block_size
            offsets = rm.slot_mapping % self._block_size
            self._loading = True  # lets the planted swap fault pick the file
            try:
                for layer_name, kv in self._kv.items():
                    path = self._generate_filename_debug(layer_name, rm.token_ids, rm.mm_hashes)
                    src = safetensors.torch.load_file(path)["kv_cache"].to(kv.device)
                    kv[blocks, :, offsets] = src  # one-sided: no ownership check, like RDMA
            finally:
                self._loading = False
        self._record(event="dup-landed" if duplicate else "landed", req=rid, tokens=n,
                     tick=self._tick, dropped=dropped,
                     blocks=sorted(set((rm.slot_mapping // self._block_size).tolist())))

    def get_finished(self, finished_req_ids):  # type: ignore[override]
        done, self._done = self._done, set()
        return None, (done or None)
