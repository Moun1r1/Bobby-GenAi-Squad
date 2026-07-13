"""agent_tools — READ-ONLY investigation tools + a tool-use loop, so a generative agent can look at a codebase
the way Claude does (grep / read / ls / find) instead of being fed static material and guessing.

Why: the biggest confabulation source was agents claiming "X is missing/broken" from a partial view. Give them the
ability to VERIFY — grep for the symbol, read the actual body — and the "it's missing" class of error goes away,
because the agent can check before it claims.

Safety: strictly read-only, sandboxed to a root directory. No writes, no arbitrary shell, no path escape. Tools
run via subprocess with fixed argv (no shell=True) or pure Python, with output + time caps.
"""
import difflib
import json
import os
import re
import subprocess
import time


class ReadOnlyTools:
    """grep / read / ls / find, confined to `root`. Every path is validated to stay inside root; nothing writes."""

    def __init__(self, root: str, max_out: int = 6000, timeout: int = 15):
        self.root = os.path.realpath(root)
        self.max_out = max_out
        self.timeout = timeout

    def _safe(self, path: str) -> str:
        p = os.path.realpath(os.path.join(self.root, path or "."))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"path escapes sandbox: {path}")
        return p

    def _cap(self, text: str) -> str:
        return text if len(text) <= self.max_out else text[: self.max_out] + "\n… (truncated)"

    def grep(self, pattern: str, path: str = ".") -> str:
        target = self._safe(path)
        try:
            r = subprocess.run(
                ["grep", "-rnI", "--exclude-dir=__pycache__", "--exclude-dir=.git", "-e", pattern, target],
                capture_output=True, text=True, timeout=self.timeout, cwd=self.root)
        except subprocess.TimeoutExpired:
            return "(grep timed out)"
        out = r.stdout or "(no matches)"
        # make paths relative to root for readability
        out = out.replace(self.root + os.sep, "")
        lines = out.splitlines()
        if len(lines) > 80:
            out = "\n".join(lines[:80]) + f"\n… ({len(lines)-80} more matches)"
        return self._cap(out)

    def read(self, path: str, lines: str = "") -> str:
        """Read a file — PAGED, never silently truncated. If the requested span exceeds the window it returns whole
        lines up to the cap and tells the caller the exact next range to continue with, so the agent's long-horizon
        loop pages through the ENTIRE file across turns and always ends up with the full code (no lost lines)."""
        p = self._safe(path)
        if not os.path.isfile(p):
            return f"(not a file: {path})"
        with open(p, "r", errors="ignore") as f:
            content = f.readlines()
        n = len(content)
        start, end = 1, n
        lines = (lines or "").strip("[]() ")
        try:
            if lines and "-" in lines:
                a, b = lines.split("-", 1)
                start, end = int(a or 1), (int(b) if b else n)
            elif lines:
                start = end = int(lines)
        except ValueError:
            pass                                            # malformed range → whole file
        start = max(1, start); end = min(n, end)
        if start > end:
            return "(empty range)"
        out, size, last = [], 0, start - 1                  # page by WHOLE lines so we never cut a line mid-way
        for i in range(start, end + 1):
            line = f"{i:>5}  {content[i-1]}"
            if size + len(line) > self.max_out and out:
                break
            out.append(line); size += len(line); last = i
        body = "".join(out) or "(empty range)"
        if last < end:                                      # more remains → hand back the exact continuation range
            body += (f"\n[paged: showed lines {start}-{last} of {n}. Continue with "
                     f"read('{path}', lines='{last + 1}-{end}') to get the rest — do this until you have the whole file.]")
        return body

    def ls(self, path: str = ".") -> str:
        p = self._safe(path)
        if os.path.isfile(p):
            return f"(file, {os.path.getsize(p)} bytes)"
        entries = sorted(os.listdir(p))
        return "\n".join(("%s/" % e if os.path.isdir(os.path.join(p, e)) else e) for e in entries if e != "__pycache__")

    def find(self, name: str) -> str:
        hits = []
        for r, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for fn in files:
                if name in fn:
                    hits.append(os.path.relpath(os.path.join(r, fn), self.root))
        return "\n".join(sorted(hits)[:60]) or "(not found)"

    def run_json(self, name: str, args: dict) -> str:
        """Dispatch a NATIVE tool call (name + JSON args from the model's function-calling)."""
        try:
            if name == "grep":
                return self.grep(args.get("pattern", ""), args.get("path", "."))
            if name == "read":
                return self.read(args.get("path", ""), str(args.get("lines", "")))
            if name == "ls":
                return self.ls(args.get("path", "."))
            if name == "find":
                return self.find(args.get("name", ""))
            return f"(unknown read-only tool: {name})"
        except Exception as e:
            return f"(tool error: {e})"


# NATIVE OpenAI-style tool schemas (the endpoint speaks these — no hand-rolled text protocol).
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "grep", "description": "search files under the repo for a text/regex "
     "pattern; returns file:line matches", "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string"}, "path": {"type": "string", "description": "optional subpath"}},
        "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read", "description": "read a file (optional line range like '40-80'). "
     "Long files are PAGED: the result ends with the exact next range to continue with — keep calling until you have "
     "the whole file, so you never work from a truncated view.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
        "lines": {"type": "string", "description": "optional, e.g. 40-80"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "ls", "description": "list a directory",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "find", "description": "locate a file by name substring",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
]

# read-only tools PLUS copy/write/edit/run in an isolated sandbox — so a verdict comes from a real RUN (0d), and an
# engine change is tested by COPYING the real file in and EDITING it (not rewriting from memory). Writes/edits get a
# live compile check; the model self-selects these — nothing scripts it.
SANDBOX_SCHEMAS = TOOL_SCHEMAS + [
    {"type": "function", "function": {"name": "copy_in", "description": "copy a REAL repo file into the sandbox so "
     "you can edit a full, exact copy (e.g. copy core.py in, then edit one method to test an engine change) — no "
     "need to rewrite the file from memory.", "parameters": {"type": "object", "properties": {
        "src": {"type": "string", "description": "repo path to copy"}, "dst": {"type": "string",
        "description": "optional sandbox destination"}}, "required": ["src"]}}},
    {"type": "function", "function": {"name": "write", "description": "write/create a file in the sandbox (NEVER the "
     "real repo). Set append=true to assemble a long file across calls. Python files get a live compile check.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
        "append": {"type": "boolean", "description": "append instead of overwrite"}},
        "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "surgically replace an exact old_string with "
     "new_string in a sandbox file (like a code editor) — fix or extend long code WITHOUT rewriting the whole file. "
     "old_string must be unique unless replace_all. Python files get a live compile check.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
        "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
        "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "run", "description": "run a python script inside the sandbox and "
     "return [exit·ms] + stdout/stderr. The real package is importable and sandbox files shadow it, so you can run a "
     "modified engine. Time- and output-capped.", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "args": {"type": "string", "description": "optional CLI args"}},
        "required": ["path"]}}},
    {"type": "function", "function": {"name": "pip_install", "description": "create (once) an isolated .venv in the "
     "sandbox and install packages into it — your own environment for experiments (e.g. pytest, numpy).",
     "parameters": {"type": "object", "properties": {"packages": {"type": "string", "description": "space-separated"}},
        "required": ["packages"]}}},
    {"type": "function", "function": {"name": "test", "description": "run pytest on a sandbox path; returns the "
     "pass/fail summary + tracebacks. pip_install('pytest') first if needed.", "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "optional, default all"}}}}},
    {"type": "function", "function": {"name": "diff", "description": "unified diff of an edited sandbox file vs its "
     "original in the real repo — see exactly what change is under test.", "parameters": {"type": "object",
        "properties": {"sandbox_path": {"type": "string"}, "repo_path": {"type": "string"}},
        "required": ["sandbox_path", "repo_path"]}}},
    {"type": "function", "function": {"name": "tree", "description": "list the files you've written in the sandbox",
     "parameters": {"type": "object", "properties": {}}}},
]


class SandboxTools(ReadOnlyTools):
    """Read-only over the real repo (inherited) PLUS write+run confined to an ISOLATED sandbox dir. This is the
    TOOLS lever for 0d/0e: an agent can script an experiment (or copy+modify an engine file) and actually RUN it, so
    a verification/experiment verdict comes from evidence that RAN — not a prompt rubric. Writes NEVER touch the real
    repo; runs are time- and output-capped, no shell. The agent self-selects when to use these — nothing scripts it."""

    def __init__(self, repo_root: str, sandbox_root: str, max_out: int = 6000, timeout: int = 15,
                 run_timeout: int = 30, depth: int = 1):
        super().__init__(repo_root, max_out=max_out, timeout=timeout)
        self.sandbox = os.path.realpath(sandbox_root)
        os.makedirs(self.sandbox, exist_ok=True)
        self.run_timeout = run_timeout
        self.depth = depth                             # recursion budget: a script this agent RUNs inherits depth-1,
        #                                                so a sub-lab spawned from here can't fork-bomb (see rd_lab).
        self.last_gain = None                          # verdict from the most recent confirm_gain that ACTUALLY ran
        self.schemas = SANDBOX_SCHEMAS                 # investigate() reads this → offers write/run to the model

    def _safe_sandbox(self, path: str) -> str:
        p = os.path.realpath(os.path.join(self.sandbox, path or "."))
        if p != self.sandbox and not p.startswith(self.sandbox + os.sep):
            raise ValueError(f"path escapes sandbox: {path}")
        return p

    def write(self, path: str, content: str, append: bool = False) -> str:
        """Write into the sandbox. `append=True` assembles a long file across MULTIPLE calls, so a script bigger than
        one turn's token budget is never truncated — the long-horizon way to produce full code."""
        p = self._safe_sandbox(path)
        os.makedirs(os.path.dirname(p) or self.sandbox, exist_ok=True)
        with open(p, "a" if append else "w") as f:
            f.write(content or "")
        verb = "appended to" if append else "wrote"
        return f"{verb} {os.path.relpath(p, self.sandbox)} (now {os.path.getsize(p)} bytes)" + self._pycheck(p)

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Surgical edit like Claude's Edit tool: replace an exact `old_string` with `new_string` in a sandbox file,
        so you fix/extend long code WITHOUT re-emitting the whole file. `old_string` must be unique unless
        replace_all. This is the long-horizon way to change big files a piece at a time."""
        p = self._safe_sandbox(path)
        if not os.path.isfile(p):
            return f"(not in sandbox: {path} — write it first)"
        with open(p, "r", errors="ignore") as f:
            s = f.read()
        cnt = s.count(old_string)
        if cnt == 0:
            return "(old_string not found — copy it exactly from a read, including whitespace)"
        if cnt > 1 and not replace_all:
            return f"(old_string appears {cnt}× — not unique; add surrounding context to disambiguate, or set replace_all=true)"
        s = s.replace(old_string, new_string) if replace_all else s.replace(old_string, new_string, 1)
        with open(p, "w") as f:
            f.write(s)
        return f"edited {os.path.relpath(p, self.sandbox)} ({cnt if replace_all else 1} replacement, now {len(s)} bytes)" + self._pycheck(p)

    def copy_in(self, src: str, dst: str = "") -> str:
        """Copy a REAL repo file into the sandbox so you can EDIT a full, exact copy (0e: test an engine change
        without rewriting core.py from memory). Source is the read-only repo; destination is the sandbox."""
        sp = self._safe(src)                                # read-only repo source
        if not os.path.isfile(sp):
            return f"(not a file: {src})"
        with open(sp, "r", errors="ignore") as f:
            data = f.read()
        dp = self._safe_sandbox(dst or os.path.basename(src))
        os.makedirs(os.path.dirname(dp) or self.sandbox, exist_ok=True)
        with open(dp, "w") as f:
            f.write(data)
        return f"copied {src} → sandbox/{os.path.relpath(dp, self.sandbox)} ({len(data)} bytes) — now edit it in place"

    def _pycheck(self, p: str) -> str:
        """Live post-write/edit build check: syntax-compile a .py so broken code is caught the instant it's written
        (the 'code that builds in' guardrail), not later at run time."""
        if not p.endswith(".py"):
            return ""
        try:
            r = subprocess.run(["python3", "-m", "py_compile", p], capture_output=True, text=True, timeout=self.timeout)
        except Exception:
            return ""
        if r.returncode == 0:
            return "  ✓ compiles"
        tail = (r.stderr or "").strip().splitlines()
        return "  ✗ does NOT compile: " + (tail[-1][:200] if tail else "SyntaxError")

    def run(self, path: str, args: str = "") -> str:
        p = self._safe_sandbox(path)
        if not os.path.isfile(p):
            return f"(not in sandbox: {path})"
        try:
            t0 = time.perf_counter()
            r = subprocess.run([self._py(), p] + (args.split() if args else []), capture_output=True, text=True,
                               timeout=self.run_timeout, cwd=self.sandbox, env=self._env())
            ms = int((time.perf_counter() - t0) * 1000)
        except subprocess.TimeoutExpired:
            return f"(run timed out after {self.run_timeout}s)"
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        gm = re.search(r'GAIN\s*\{.*?"verdict"\s*:\s*"(WIRE|MARGINAL|DELETE|DEFER)"', out)
        if gm:                                          # a confirm_gain ACTUALLY ran → record its real verdict
            self.last_gain = gm.group(1).upper()
        return f"[exit {r.returncode} · {ms}ms]\n" + self._cap(out.strip() or "(no output)")

    # -- own isolated environment + structured test/observation bench (self-selected tools) --------------
    def _env(self) -> dict:
        # sandbox shadows the real package (0e); the repo's parent makes `bobby_squad` importable
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([self.sandbox, os.path.dirname(self.root), env.get("PYTHONPATH", "")])
        env["RD_DEPTH"] = str(self.depth - 1)          # a lab spawned by this run inherits a SMALLER depth budget →
        #                                                bounded recursion; RD_TRACE (if set) flows through so all
        #                                                nested labs append to ONE observable ledger.
        return env

    def _py(self) -> str:
        venv_py = os.path.join(self.sandbox, ".venv", "bin", "python")
        return venv_py if os.path.isfile(venv_py) else "python3"

    def pip_install(self, packages: str) -> str:
        """Create (once) an isolated .venv inside the sandbox and pip-install packages into it — the agent gets its
        OWN environment for experiments (pytest, numpy, …) without touching anything else."""
        venv_dir = os.path.join(self.sandbox, ".venv")
        if not os.path.isdir(venv_dir):
            v = subprocess.run(["python3", "-m", "venv", venv_dir], capture_output=True, text=True, timeout=120)
            if v.returncode != 0:
                return f"(venv create failed: {(v.stderr or '')[:200]})"
        pkgs = packages.split() if isinstance(packages, str) else list(packages)
        if not pkgs:
            return "(no packages given)"
        try:
            r = subprocess.run([os.path.join(venv_dir, "bin", "pip"), "install", "-q", *pkgs],
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return "(pip install timed out)"
        return self._cap(((r.stdout or "") + (r.stderr or "")).strip() or f"installed {' '.join(pkgs)} (exit {r.returncode})")

    def test(self, path: str = ".") -> str:
        """Run pytest on a sandbox path and return the pass/fail summary + tracebacks — a structured verdict, not raw
        stdout. Falls back cleanly if pytest isn't installed (use pip_install('pytest'))."""
        tp = self._safe_sandbox(path)
        try:
            r = subprocess.run([self._py(), "-m", "pytest", "-q", tp], capture_output=True, text=True,
                               timeout=self.run_timeout * 4, cwd=self.sandbox, env=self._env())
        except subprocess.TimeoutExpired:
            return f"(tests timed out after {self.run_timeout * 4}s)"
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if "No module named pytest" in out:
            return "(pytest not installed — call pip_install('pytest') first)"
        return f"[pytest exit {r.returncode}]\n" + self._cap(out or "(no output)")

    def diff(self, sandbox_path: str, repo_path: str) -> str:
        """Unified diff of an edited sandbox file vs its ORIGINAL in the real repo — observe EXACTLY what change is
        under test (essential when testing an engine change via copy_in→edit)."""
        sp, rp = self._safe_sandbox(sandbox_path), self._safe(repo_path)
        if not (os.path.isfile(sp) and os.path.isfile(rp)):
            return "(need an existing sandbox file and a real repo file)"
        a = open(rp, errors="ignore").read().splitlines()
        b = open(sp, errors="ignore").read().splitlines()
        d = list(difflib.unified_diff(a, b, fromfile="repo/" + repo_path, tofile="sandbox/" + sandbox_path, lineterm=""))
        return self._cap("\n".join(d)) if d else "(identical — no change yet)"

    def tree(self) -> str:
        """List the files the lab has written in the sandbox (observe your own artifacts)."""
        hits = []
        for r, dirs, files in os.walk(self.sandbox):
            dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__")]
            for fn in files:
                hits.append(os.path.relpath(os.path.join(r, fn), self.sandbox))
        return "\n".join(sorted(hits)[:100]) or "(sandbox empty)"

    def run_json(self, name: str, args: dict) -> str:
        try:
            if name == "write":
                return self.write(args.get("path", ""), args.get("content", ""), bool(args.get("append")))
            if name == "edit":
                return self.edit(args.get("path", ""), args.get("old_string", ""), args.get("new_string", ""),
                                 bool(args.get("replace_all")))
            if name == "copy_in":
                return self.copy_in(args.get("src", ""), args.get("dst", ""))
            if name == "run":
                return self.run(args.get("path", ""), str(args.get("args", "")))
            if name == "pip_install":
                return self.pip_install(args.get("packages", ""))
            if name == "test":
                return self.test(args.get("path", "."))
            if name == "diff":
                return self.diff(args.get("sandbox_path", ""), args.get("repo_path", ""))
            if name == "tree":
                return self.tree()
        except Exception as e:
            return f"(sandbox error: {e})"
        return super().run_json(name, args)

DGX_SCHEMAS = SANDBOX_SCHEMAS + [
    {"type": "function", "function": {"name": "dgx", "description": "Run a shell command on the DGX GPU host over SSH "
     "— REAL GPU compute (nvidia-smi, docker run --gpus all …, docker exec <container> python train.py, etc.). Returns "
     "stdout/stderr + exit. For a LONG or CONTINUOUS job set background=true: it launches DETACHED and returns a job "
     "id you poll with dgx_logs — so training runs while you keep working.", "parameters": {"type": "object",
     "properties": {"cmd": {"type": "string"}, "background": {"type": "boolean", "description": "launch detached for a "
     "long/continuous job"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "dgx_push", "description": "Copy a file from your sandbox to the DGX "
     "working dir so you can run it on the GPU.", "parameters": {"type": "object", "properties": {"path": {"type":
     "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "dgx_pull", "description": "Copy a file (result / checkpoint / metrics / "
     "log) from the DGX working dir back into your sandbox.", "parameters": {"type": "object", "properties": {"path":
     {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "dgx_logs", "description": "Tail a background DGX job's log (by id) to "
     "OBSERVE a continuous run's progress.", "parameters": {"type": "object", "properties": {"job": {"type": "string"},
     "lines": {"type": "integer"}}, "required": ["job"]}}},
]


class DgxTools(SandboxTools):
    """Generic REMOTE GPU compute on the DGX over SSH — reusable for ANY training/experiment (build a model, train a
    memory-optimizer, run a continuous pipeline). Extends the sandbox with dgx / dgx_push / dgx_pull / dgx_logs. Every
    remote action streams a structured `dgx` OBSERVABILITY event through the SAME observer the Agent uses, so remote
    work is watched live and captured. Host/dir from DGX_HOST / DGX_DIR (default ssh 'spark', ~/ga_dgx). The agent
    self-selects when to reach for the GPU — nothing scripts it. Not gemma-specific: any compute the SELF calls for."""

    def __init__(self, repo_root: str, sandbox_root: str, host=None, worker=None, image=None, workdir=None,
                 observer=None, dgx_timeout: int = 240, **kw):
        super().__init__(repo_root, sandbox_root, **kw)
        self.host = host or os.environ.get("DGX_HOST", "spark")
        # ISOLATION: a dedicated GPU-enabled WORKER container — never the host. Reuses an NVIDIA training-stack image so
        # CUDA/pytorch is already there. Everything runs INSIDE it, so a bad command can't harm the DGX.
        self.worker = worker or os.environ.get("DGX_WORKER", "ga_worker")
        self.image = image or os.environ.get("DGX_IMAGE", "")   # "" → auto-detect a LOCAL torch image (never pull)
        self.workdir = workdir or os.environ.get("DGX_WORKDIR", "/workspace")
        self.observer = observer
        self.dgx_timeout = dgx_timeout
        self._ensured = False
        self.schemas = DGX_SCHEMAS

    def _emit(self, **d):                                  # observability → the engine's event stream
        if self.observer:
            try:
                self.observer({"kind": "dgx", "host": self.host, "worker": self.worker, **d})
            except Exception:
                pass

    def _ssh(self, cmd: str, timeout: int):
        import subprocess
        try:
            p = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", self.host, cmd],
                               capture_output=True, text=True, timeout=timeout)
            return p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired:
            return 124, "(timed out)"

    def ensure_worker(self) -> str:
        """Create the dedicated GPU WORKER container if it isn't running (idempotent). Reuses the NVIDIA training-stack
        image with `--gpus all`; a persistent named volume holds the workdir. All compute runs INSIDE this container, so
        the DGX host is never touched. First call may PULL the image (slow)."""
        _rc, out = self._ssh(f"docker ps --format '{{{{.Names}}}}' | grep -qx {self.worker} && echo UP || echo DOWN", 25)
        if "UP" in out:
            self._ensured = True
            return "worker up"
        # REUSE an NVIDIA training stack ALREADY on the DGX (never pull — that hangs). Detect a LOCAL image that has a
        # cuda/torch runtime; prefer a known-good one. If none, error clearly rather than pulling 20GB.
        if not self.image:
            _rc, imgs = self._ssh("docker images --format '{{.Repository}}:{{.Tag}}'", 25)
            local = [ln.strip() for ln in imgs.splitlines() if ln.strip() and "<none>" not in ln]
            kw = ("pytorch", "torch", "cuda", "cu13", "cu12", "cu11", "nvcr", "nemo", "tensorrt", "jax", "vllm", "sglang", "azr")
            cand = [im for im in local if any(k in im.lower() for k in kw)]
            # prefer OUR purpose-built image (framework + full DPO/transformer/encoder stack), then known-good bases
            pref = ([im for im in local if im.startswith("ga_worker:")] or [im for im in cand if "azr:" in im]
                    or [im for im in cand if "sglang" in im] or cand)
            if not pref:
                return ("(no local NVIDIA/torch image found on the DGX to reuse — set DGX_IMAGE to one, or `docker pull` "
                        "an NVIDIA pytorch image on the DGX first; refusing to auto-pull 20GB)")
            self.image = pref[0]
        # RESOURCE MANAGEMENT (the DGX is SHARED — never starve it): cap container RAM/CPUs, and — critically — stop
        # JAX/torch from grabbing the whole GPU. JAX preallocates ~75% of VRAM by default; on a shared box that OOMs
        # everything else. These env force lazy, fractional GPU use so a training run can't crash the DGX.
        mem = os.environ.get("DGX_WORKER_MEM", "48g")
        cpus = os.environ.get("DGX_WORKER_CPUS", "12")
        frac = os.environ.get("DGX_GPU_FRACTION", "0.35")
        # mount foundation-model weights read-only so the worker can TRAIN real models (Gemma/Qwen/…) with no download.
        models = os.environ.get("DGX_MODELS", "/home/moums/models")
        mnt = f"-v {models}:/models:ro " if models else ""
        create = (f"docker rm -f {self.worker} 2>/dev/null; "
                  f"docker run -d --name {self.worker} --gpus all --restart unless-stopped "
                  f"--memory={mem} --memory-swap={mem} --cpus={cpus} --shm-size=8g "
                  f"-e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION={frac} "
                  f"-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_DEVICE_ORDER=PCI_BUS_ID "
                  f"{mnt}-v {self.worker}_data:{self.workdir} -w {self.workdir} {self.image} sleep infinity")
        rc, out = self._ssh(create, 600)                   # may pull the NVIDIA image (large, slow) — one-time
        self._ensured = rc == 0
        self._emit(action="worker_create", image=self.image, exit=rc)
        if rc != 0:
            return (f"(could not start GPU worker from image {self.image}: {out.strip()[-200:]} — set DGX_IMAGE to an "
                    "NVIDIA pytorch image available on the DGX, e.g. an arm64 tag)")
        return f"worker '{self.worker}' up (GPU, image {self.image})"

    def _in_worker(self, cmd: str, timeout: int, detached: bool = False):
        import shlex
        inner = f"cd {self.workdir} && {cmd}"
        flag = "-d" if detached else ""
        return self._ssh(f"docker exec {flag} {self.worker} bash -lc {shlex.quote(inner)}", timeout)

    def dgx(self, cmd: str, background: bool = False) -> str:
        import re
        import shlex
        if not self._ensured and "up" not in self.ensure_worker():
            return self.ensure_worker()                    # surface the setup error
        try:
            if background:
                job = (re.sub(r"[^a-z0-9]+", "_", (cmd or "").lower()).strip("_")[:24]) or "job"
                launch = f"nohup bash -lc {shlex.quote(cmd)} > {job}.log 2>&1 & echo JOB $!"
                rc, _out = self._in_worker(launch, 40, detached=True)
                self._emit(action="launch", cmd=(cmd or "")[:200], job=job, exit=rc)
                return f"[background job '{job}' launched IN worker '{self.worker}' — poll with dgx_logs('{job}')] [exit {rc}]"
            rc, out = self._in_worker(cmd, self.dgx_timeout)
            tail = (out.strip().splitlines()[-1][:120] if out.strip() else "")
            self._emit(action="run", cmd=(cmd or "")[:200], exit=rc, out=tail)
            return self._cap(out) + f"\n[exit {rc} · in worker {self.worker}]"
        except Exception as e:
            return f"(dgx error: {e})"

    def dgx_push(self, path: str) -> str:
        import subprocess
        src = self._safe_sandbox(path)
        if not os.path.exists(src):
            return f"(not in sandbox: {path} — write it first)"
        if not self._ensured:
            self.ensure_worker()
        try:
            stage = f"/tmp/{self.worker}_stage"
            self._ssh(f"mkdir -p {stage}/{os.path.dirname(path)}".rstrip("/"), 20)
            p = subprocess.run(["scp", "-q", src, f"{self.host}:{stage}/{path}"], capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                return f"(push scp failed: {p.stderr.strip()[:160]})"
            rc, out = self._ssh(f"docker exec {self.worker} mkdir -p {self.workdir}/{os.path.dirname(path)}; "
                                f"docker cp {stage}/{path} {self.worker}:{self.workdir}/{path}", 60)
            self._emit(action="push", path=path, exit=rc)
            return f"pushed {path} → worker {self.worker}:{self.workdir}/{path}" if rc == 0 else f"(docker cp failed: {out.strip()[:160]})"
        except Exception as e:
            return f"(push error: {e})"

    def dgx_pull(self, path: str) -> str:
        import subprocess
        dst = self._safe_sandbox(path)
        os.makedirs(os.path.dirname(dst) or self.sandbox, exist_ok=True)
        try:
            stage = f"/tmp/{self.worker}_stage"
            rc, out = self._ssh(f"mkdir -p {stage}/{os.path.dirname(path)}; "
                                f"docker cp {self.worker}:{self.workdir}/{path} {stage}/{path}", 60)
            if rc != 0:
                return f"(docker cp out failed: {out.strip()[:160]})"
            p = subprocess.run(["scp", "-q", f"{self.host}:{stage}/{path}", dst], capture_output=True, text=True, timeout=180)
            self._emit(action="pull", path=path, exit=p.returncode)
            return f"pulled {path} → sandbox ({os.path.getsize(dst)} bytes)" if p.returncode == 0 else f"(pull scp failed: {p.stderr.strip()[:160]})"
        except Exception as e:
            return f"(pull error: {e})"

    def dgx_logs(self, job: str, lines: int = 40) -> str:
        try:
            _rc, out = self._in_worker(f"tail -n {int(lines)} {job}.log 2>/dev/null", 30)
            self._emit(action="logs", job=job)
            return self._cap(out) or f"(no log yet for job '{job}')"
        except Exception as e:
            return f"(logs error: {e})"

    def run_json(self, name: str, args: dict) -> str:
        if name == "dgx":
            return self.dgx(args.get("cmd", ""), bool(args.get("background")))
        if name == "dgx_push":
            return self.dgx_push(args.get("path", ""))
        if name == "dgx_pull":
            return self.dgx_pull(args.get("path", ""))
        if name == "dgx_logs":
            return self.dgx_logs(args.get("job", ""), int(args.get("lines", 40) or 40))
        return super().run_json(name, args)


# Behavior-neutral principle, not a script of what to produce (0c): verify/ground before claiming; run when useful.
_SYSTEM = ("You have tools to investigate real code and, when a claim needs proof, to WRITE and RUN a script in a "
           "sandbox. Ground before you claim: never assert something exists/works/breaks without checking it against "
           "the real code or an actual run.")


def investigate(llm, task: str, tools: "ReadOnlyTools", max_rounds: int = 6, max_tokens: int = 500, on_event=None):
    """Tool-use loop using the model's NATIVE function-calling: the model returns structured tool_calls, we run them
    and feed the results back as tool messages, until it answers with content (no tool_calls). The tool object
    supplies its own schemas (read-only, or read+sandbox-run), so an agent that CAN run experiments self-selects to.
    Returns (final_answer, trace) where trace is the list of (tool_name, args_dict) actually run. `llm` needs `.chat`."""
    schemas = getattr(tools, "schemas", None) or TOOL_SCHEMAS
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": task}]
    trace = []
    for _rnd in range(max_rounds):
        msg = llm.chat(messages, tools=schemas, max_tokens=max_tokens)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip(), trace
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for tc in calls[:6]:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            name = fn.get("name", "")
            if on_event:
                on_event("tool", name=name, args={k: str(v)[:50] for k, v in args.items()})   # live: what it's doing
            result = tools.run_json(name, args)
            trace.append((name, args))
            if on_event:
                on_event("tool_done", name=name, out=result.splitlines()[0][:80] if result else "")
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    # exploration budget spent — force a final answer with tools OFF so we never return a bare tool-call turn
    messages.append({"role": "user", "content": "Stop investigating — no more tool calls. Based on everything you "
                                                "READ above, give your final answer now."})
    final = llm.chat(messages, tools=None, max_tokens=max(max_tokens, 700))
    return (final.get("content") or "").strip(), trace
