import json
import os
import signal
import subprocess
import time
from typing import List, Optional


def _now() -> float:
    return time.time()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _kill_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


class JobRegistry:
    """Filesystem-backed named-job store rooted at `root`. Every job is `<root>/<name>.{json,exit,log}`."""

    _TERMINAL = ("done", "failed", "killed", "timeout")

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _paths(self, name: str):
        base = os.path.join(self.root, name)
        return base + ".json", base + ".exit", base + ".log"

    def _load(self, name: str) -> Optional[dict]:
        meta, _, _ = self._paths(name)
        try:
            return json.load(open(meta))
        except (OSError, ValueError):
            return None

    def _save(self, j: dict) -> dict:
        meta, _, _ = self._paths(j["name"])
        json.dump(j, open(meta, "w"))
        return j

    def _reconcile(self, j: dict) -> dict:
        """Derive current status from the world (sentinel / pid / timeout). No daemon updates this — reads do."""
        if j.get("status") in self._TERMINAL:
            return j
        _, exitf, _ = self._paths(j["name"])
        if os.path.exists(exitf):                                  # the wrapper finished and wrote its code
            code = open(exitf).read().strip()
            j["exit"] = int(code) if code.lstrip("-").isdigit() else -1
            j["status"] = "done" if j["exit"] == 0 else "failed"
            j["ended"] = _now()
        elif not _pid_alive(j["pid"]):                             # process vanished without a sentinel → crash/reboot
            j["status"] = "failed"
            j["exit"] = -1
            j["ended"] = _now()
        elif j.get("timeout") and _now() - j["submitted"] > j["timeout"]:
            _kill_group(j["pid"])
            j["status"] = "timeout"
            j["exit"] = -9
            j["ended"] = _now()
        return self._save(j)

    # ── public API ─────────────────────────────────────────────────────────────────────────────────
    def submit(self, name: str, cmd: str, timeout: Optional[float] = 3600, cwd: Optional[str] = None,
               env: Optional[dict] = None) -> dict:
        """Start `cmd` as the named job (idempotent). If `name` is already RUNNING, return the running job unchanged."""
        cur = self.status(name)
        if cur and cur["status"] == "running":
            return cur                                             # idempotent: don't duplicate a live job
        meta, exitf, logf = self._paths(name)
        for p in (meta, exitf):
            if os.path.exists(p):
                os.remove(p)
        wrapped = f"( {cmd} ) ; printf $? > {exitf!s}"             # daemonless exit sentinel
        with open(logf, "wb") as lo:
            proc = subprocess.Popen(["bash", "-lc", wrapped], stdout=lo, stderr=subprocess.STDOUT,
                                    cwd=cwd, env=({**os.environ, **env} if env else None),
                                    start_new_session=True)        # detached process group → survives our exit
        return self._save({"name": name, "pid": proc.pid, "cmd": cmd, "status": "running",
                           "submitted": _now(), "timeout": timeout})

    def status(self, name: str) -> Optional[dict]:
        j = self._load(name)
        return self._reconcile(j) if j else None

    def wait(self, name: str, poll: float = 0.5, timeout: Optional[float] = None) -> Optional[dict]:
        """Block until the job reaches a terminal state (or `timeout`). The scheduler's 'Wait' state — NOT an agent
        loop. Returns the final job dict (or None if the job doesn't exist)."""
        t0 = _now()
        while True:
            j = self.status(name)
            if not j or j["status"] != "running":
                return j
            if timeout is not None and _now() - t0 > timeout:
                return j
            time.sleep(poll)

    def logs(self, name: str, tail: int = 0) -> str:
        _, _, logf = self._paths(name)
        try:
            data = open(logf).read()
        except OSError:
            return ""
        return "\n".join(data.splitlines()[-tail:]) if tail else data

    def cancel(self, name: str) -> Optional[dict]:
        j = self.status(name)
        if j and j["status"] == "running":
            _kill_group(j["pid"])
            j["status"] = "killed"
            j["exit"] = -15
            j["ended"] = _now()
            return self._save(j)
        return j

    def list(self) -> List[dict]:
        out: List[dict] = []
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".json"):
                j = self.status(fn[:-5])
                if j:
                    out.append(j)
        return out
