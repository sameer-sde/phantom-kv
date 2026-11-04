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

FAULTS = ("swap", "silent-drop")


class PhantomFaultConnector(ExampleConnector):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fault = os.environ.get("PHANTOM_PD_FAULT") or None
        if self._fault is not None and self._fault not in FAULTS:
            raise ValueError(f"unknown PHANTOM_PD_FAULT {self._fault!r}; choose from {FAULTS}")
        self._fault_log = os.environ.get("PHANTOM_PD_FAULT_LOG")
        self._loading = False

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
