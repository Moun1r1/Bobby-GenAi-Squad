"""dgx_monitor — a backend service that watches the DGX in REALTIME via NVIDIA's own tooling (nvidia-smi) plus plain
host metrics, so we can be SURE the box is safe BEFORE launching training and never crash it again.

One lightweight SSH poll gathers GPU (util / VRAM / temp / power + per-process usage) · CPU (load / cores / RAM) ·
STORAGE (disk) · the running DOCKER sessions. An nvidia-smi *query* does NO GPU compute, so polling is cheap; results
are cached on a short interval and served both as a snapshot and an SSE realtime stream. `is_safe()` is the pre-train
gate — the resource management that stops a run from starving the shared DGX.
"""
import logging
import os
import subprocess
import threading
import time

_log = logging.getLogger(__name__)


def _ssh_cmd(host: str):
    """Build the ssh invocation from DGX_* env (user / key), non-interactive + no host-key prompt so it works from a
    fresh container the same way `ssh spark` works on the host."""
    cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    key = os.environ.get("DGX_KEY")
    if key:
        cmd += ["-i", key, "-o", "IdentitiesOnly=yes"]
    user = os.environ.get("DGX_USER")
    return cmd, (f"{user}@{host}" if user else host)


class DgxMonitor:
    def __init__(self, host: str = "localhost", worker: str = "ga_worker"):
        self.host = host
        self.worker = worker
        self._cache: dict = {}
        self._ts = 0.0
        self._lock = threading.Lock()

    _POLL = (
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit "
        "--format=csv,noheader,nounits 2>/dev/null; echo '@@PROC@@'; "
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null; "
        "echo '@@CPU@@'; cat /proc/loadavg; nproc; free -m 2>/dev/null | awk '/Mem:/{print $2\",\"$3\",\"$7}'; "
        "echo '@@DISK@@'; df -Ph / 2>/dev/null | tail -1; "
        "echo '@@DOCKER@@'; docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}' 2>/dev/null"
    )

    def _poll(self) -> dict:
        try:
            if os.environ.get("DGX_LOCAL"):                 # backend co-located on the DGX → poll the worker locally
                worker = os.environ.get("DGX_WORKER", self.worker)
                # `; true` so the trailing `docker ps` (no docker INSIDE the worker) can't fail the whole poll —
                # GPU/RAM/disk are already captured; the docker-sessions section is simply empty here.
                p = subprocess.run(["docker", "exec", worker, "bash", "-lc", self._POLL + "\ntrue"],
                                   capture_output=True, text=True, timeout=15)
            else:
                cmd, target = _ssh_cmd(self.host)
                p = subprocess.run(cmd + [target, self._POLL], capture_output=True, text=True, timeout=15)
        except Exception as e:
            _log.warning("dgx poll failed: %s", e)          # log server-side; do NOT leak the trace to clients
            return {"ok": False, "error": "dgx poll failed", "ts": time.time()}
        if p.returncode != 0:
            _log.warning("dgx poll returncode=%s stderr=%s", p.returncode, (p.stderr or "").strip()[:200])
            return {"ok": False, "error": "dgx poll unavailable", "ts": time.time()}
        sec = {"gpu": "", "PROC": "", "CPU": "", "DISK": "", "DOCKER": ""}
        cur = "gpu"
        for line in p.stdout.splitlines():
            m = {"@@PROC@@": "PROC", "@@CPU@@": "CPU", "@@DISK@@": "DISK", "@@DOCKER@@": "DOCKER"}.get(line.strip())
            if m:
                cur = m
                continue
            sec[cur] += line + "\n"
        out: dict = {"ok": True, "host": self.host, "ts": time.time()}
        # GPU
        g = [x.strip() for x in (sec["gpu"].strip().splitlines()[0].split(",") if sec["gpu"].strip() else [])]
        if len(g) >= 6:
            def _n(v, f=float):
                try:
                    return f(v)
                except Exception:
                    return None
            mu, mt = _n(g[2], int), _n(g[3], int)
            out["gpu"] = {"name": g[0], "util_pct": _n(g[1], int), "mem_used_mb": mu, "mem_total_mb": mt,
                          "mem_free_mb": (mt - mu) if (mu is not None and mt is not None) else None,
                          "temp_c": _n(g[4], int), "power_w": _n(g[5]), "power_limit_w": _n(g[6]) if len(g) > 6 else None}
        # GPU processes
        procs = []
        for ln in sec["PROC"].strip().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 3:
                procs.append({"pid": parts[0], "name": parts[1], "mem_mb": parts[2]})
        out["gpu_procs"] = procs
        # CPU / RAM
        cpu = sec["CPU"].strip().splitlines()
        if cpu:
            la = cpu[0].split()
            cores = int(cpu[1]) if len(cpu) > 1 and cpu[1].isdigit() else None
            ram = cpu[2].split(",") if len(cpu) > 2 else []
            out["cpu"] = {"load1": float(la[0]) if la else None, "load5": float(la[1]) if len(la) > 1 else None,
                          "cores": cores,
                          "load_pct": round(100 * float(la[0]) / cores) if (la and cores) else None}
            if len(ram) == 3:
                out["ram"] = {"total_mb": int(ram[0]), "used_mb": int(ram[1]), "free_mb": int(ram[2])}
        # DISK
        d = sec["DISK"].strip().split()
        if len(d) >= 5:
            out["disk"] = {"size": d[1], "used": d[2], "avail": d[3], "use_pct": int(d[4].rstrip("%"))}
        # DOCKER sessions
        out["docker"] = [{"name": a, "image": b, "status": c}
                         for ln in sec["DOCKER"].strip().splitlines() if ln
                         for a, b, c in [(ln.split("|") + ["", "", ""])[:3]]]
        out["worker_up"] = any(s["name"] == self.worker for s in out["docker"])
        return out

    def snapshot(self, max_age: float = 4.0) -> dict:
        """Cached realtime snapshot — refreshes only if older than max_age, so many viewers = one poll."""
        with self._lock:
            if time.time() - self._ts > max_age or not self._cache:
                self._cache = self._poll()
                self._ts = time.time()
            return self._cache

    def is_safe(self, min_free_mb: int = 16000, max_util: int = 92, max_disk_pct: int = 95) -> dict:
        """The pre-train GATE. On the GB10 GPU + CPU share ONE unified memory pool and nvidia-smi reports VRAM as
        [N/A], so the memory headroom that matters is SYSTEM RAM free (prefer it; fall back to discrete VRAM on GPUs
        that report it). Are there enough free memory + GPU/disk headroom to train without starving the shared DGX?"""
        s = self.snapshot()
        if not s.get("ok"):
            return {"safe": False, "reason": f"monitor unreachable ({s.get('error')})"}
        g = s.get("gpu") or {}
        ram = s.get("ram") or {}
        vram_free = g.get("mem_free_mb")
        # unified-memory box → RAM free is the real signal; discrete GPU → VRAM free
        free = ram.get("free_mb") if vram_free is None else vram_free
        mem_src = "unified-RAM" if vram_free is None else "VRAM"
        util = g.get("util_pct")
        disk = (s.get("disk") or {}).get("use_pct")
        reasons = []
        if free is not None and free < min_free_mb:
            reasons.append(f"{mem_src} free {free}MB < {min_free_mb}MB")
        if util is not None and util > max_util:
            reasons.append(f"GPU util {util}% > {max_util}%")
        if disk is not None and disk > max_disk_pct:
            reasons.append(f"disk {disk}% > {max_disk_pct}%")
        return {"safe": not reasons, "reason": "; ".join(reasons) or "ok",
                "mem_source": mem_src, "mem_free_mb": free, "gpu_util_pct": util, "disk_pct": disk}


_MON = None


def get_monitor(host: str = "localhost") -> DgxMonitor:
    global _MON
    if _MON is None:
        import os
        _MON = DgxMonitor(host=os.environ.get("DGX_HOST", host))
    return _MON
