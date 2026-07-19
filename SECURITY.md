# Security Policy

## Supported versions

Bobby is a research codebase under active development. Security fixes are applied to the
`main` branch and the most recent tagged release.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| latest tagged release | ✅ |
| older tags | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's **[Private vulnerability reporting](https://github.com/Moun1r1/Bobby-Self-Organizing-Agent-Squad/security/advisories/new)**
(Security → *Report a vulnerability*).

Please include:
- a description of the issue and its impact,
- steps to reproduce (a minimal PoC if possible),
- affected paths / versions.

You can expect an acknowledgement within **72 hours** and a status update within **7 days**.
Coordinated disclosure is appreciated — we will agree on a disclosure timeline once a fix is ready.

## Scope & known risk areas

Bobby executes model-proposed logic and shells out to infrastructure. Reviewers and reporters should
pay particular attention to:

- **The `algo` distillation sandbox** (`bobby_squad/burn_in.py`, `make_codeplugin`) — runs
  LLM-authored code under *restricted builtins*, which is a **demo sandbox, not true isolation**. Do not
  run untrusted code paths on shared infrastructure; the intended isolation is a copy-on-write execution
  plane (deferred).
- **The Studio backend** (`studio/backend/`) — a FastAPI service that shells out over SSH to a training
  host and controls Docker. Do not expose it to untrusted networks; treat `DGX_*` credentials as secrets.
- **Dependency supply chain** — automated updates are configured in
  [`.github/dependabot.yml`](.github/dependabot.yml); CodeQL / Dependabot alerts gate the repository.

## Handling of secrets

Never commit API keys, SSH keys, or `.env` files. Runtime configuration is via environment variables
(`BOBBY_LLM_URL`, `BOBBY_EMBED_URL`, `DGX_*`, …). Errors are logged server-side; exception details and
stack traces are **not** returned to HTTP clients.
