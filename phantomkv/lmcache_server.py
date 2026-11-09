"""A CPU-only LMCache server for one Phantom campaign (Phase 5, target 3).

The harness starts one server per campaign, clears its cache before every
run (so a replay of a seed starts from the same empty store), and can
restart it in the middle of a run (a fault vLLM must survive). LMCache must
be installed in the same Python environment (`lmcache` next to the
interpreter).
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request


class LMCacheServer:
    def __init__(self, port: int = 5555, http_port: int = 8090, l1_gb: float = 1.0,
                 chunk_size: int = 32, log: str = "lmcache_server.log") -> None:
        self.port, self.http_port, self.l1_gb, self.chunk_size = port, http_port, l1_gb, chunk_size
        self.log = os.path.abspath(log)
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        exe = os.path.join(os.path.dirname(sys.executable), "lmcache")
        self.exe = exe if os.path.exists(exe) else shutil.which("lmcache")
        if not self.exe:
            raise RuntimeError("the `lmcache` command is not installed in this environment")
        atexit.register(self.stop)

    def _cmd(self) -> list[str]:
        return [self.exe, "server", "--port", str(self.port), "--http-port", str(self.http_port),
                "--l1-size-gb", str(self.l1_gb), "--chunk-size", str(self.chunk_size),
                "--eviction-policy", "LRU",
                # no GPU: vLLM's worker does the copies ("engine-driven")
                "--supported-transfer-mode", "engine_driven"]

    def _up(self, port: int) -> bool:
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return True
        except OSError:
            return False

    def start(self, timeout: float = 120.0) -> None:
        env = dict(os.environ, LMCACHE_DISABLE_BANNER="1")
        with open(self.log, "a") as out:
            self.proc = subprocess.Popen(self._cmd(), stdout=out, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"LMCache server exited (code {self.proc.returncode}); "
                                   f"see {self.log}")
            if self._up(self.port) and self._up(self.http_port):
                return
            time.sleep(0.5)
        raise RuntimeError(f"LMCache server did not come up in {timeout:.0f}s; see {self.log}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def restart(self) -> None:
        """Kill and restart: every stored chunk is gone; vLLM must cope."""
        self.stop()
        self.start()
        self.restarts += 1

    def clear(self) -> bool:
        """Empty every tier, including objects held by readers (between runs)."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.http_port}/cache/clear",
            data=json.dumps({"tier": "all", "force": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return 200 <= r.status < 300
        except Exception:
            return False

    def counts(self) -> dict[str, int]:
        """Stores and retrievals the server has logged so far."""
        try:
            text = open(self.log).read()
        except OSError:
            return {"stores": 0, "retrievals": 0}
        return {"stores": text.count("Stored "), "retrievals": text.count("Retrieved ")}
