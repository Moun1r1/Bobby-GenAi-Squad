# Cross-Project Knowledge Map

_Built by one persistent-self agent (the served model) streaming 25 project units through a bounded window (size 2) with compaction between each. Pinned prompt stayed ≤ 4689 tokens; a naive keep-everything agent would have reached ~40075 tokens by the end (**8.5× more**). The complete index below lived in the pinned tier the whole time._

## Top-value summary (synthesized from the complete pinned index)

**TOP-VALUE SUMMARY**

1.  **hermes**: ASan+Debug default build config; strict RN version pinning (README.md, CLAUDE.md)
2.  **prometheus**: Strict flag vs. file config separation; oklog/run actor lifecycle model (internal_architecture.md)
3.  **terraform**: Deadlock-free promise concurrency in `internal/promising`; provider mirror migration (internal/promising/README.md)
4.  **tokio**: Isolated feature flag testing crate; 32-bit atomic compatibility verification (tests-build/README.md)
5.  **polars**: Zero-dependency import strategy; strict Arrow array equality adherence (crates/polars-buffer/README.md)
6.  **ollama**: Integrated benchmarking with TTFT tracking; Go+React desktop dev workflow (cmd/bench/README.md)
7.  **langchain**: `uv` monorepo structure with editable installs; `standard-tests` for integration validation (libs/README.md)
8.  **httpx**: Optional CLI integration; strict timeout enforcement across sync/async APIs (docs/api.md)
9.  **svelte**: Async DOM update test patterns; back-to-front list insertion optimization (packages/svelte/README.md)
10. **redis**: External server test mode with granular tag filtering; srandmember distribution validation (tests/README.md)
11. **neovim**: CMake build system with XDG/shada integration; vterm adoption strategy (BUILD.md)
12. **opencode**: Multi-platform distribution matrix (npm/brew/scoop); VS Code extension debugging workflow (README.md)

**DEAD/LOW-VALUE**
*   **airflow**: No proven findings; generic CLI patterns.
*   **bevy**: No proven findings; standard ECS patterns.
*   **cargo**: No proven findings; standard credential architecture.
*   **core**: No proven findings; standard type testing.
*   **deno**: No proven findings; standard runtime assembly.
*   **elasticsearch**: No proven findings; standard Docker pipeline.
*   **etcd**: No proven findings; standard CLI/benchmark tools.
*   **fastapi**: No proven findings; standard type-hint design.
*   **flask**: No proven findings; standard WSGI patterns.
*   **grafana**: No proven findings; standard Docker setup.
*   **react**: No proven findings; standard CI/benchmarking.

**CROSS-PROJECT REUSABLE PRIMITIVES**
*   **Build/Config Isolation**: `tokio` (feature flag testing), `prometheus` (flag vs. file config), `hermes` (ASan defaults).
*   **Concurrency Models**: `terraform` (deadlock-free promises), `prometheus` (actor lifecycle), `tokio` (work-stealing).
*   **Testing Infrastructure**: `redis` (external server/tag filtering), `langchain` (standard-tests), `bevy` (compile-fail utils).
*   **Distribution/Install**: `opencode` (multi-platform matrix), `terraform` (provider mirrors), `ollama` (Go+React dev workflow).
*   **Architecture Patterns**: `polars` (zero-dep imports), `httpx` (optional CLI), `svelte` (async DOM updates).

---

## Full per-project index (pinned store, verbatim)

[airflow]
PROJECT: airflow
WHAT: Apache Airflow workflow orchestration platform with SDK, CLI, and provider ecosystem
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, CLAUDE.md, generated/README.md
REUSE: CLI command patterns, breeze shim setup, naming conventions (Dag vs DAG), test execution via uv/breeze

[bevy]
PROJECT: bevy
WHAT: Rust data-oriented game engine with ECS architecture, timekeeping, and compile-fail test infrastructure
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, tools/compile_fail_utils/README.md, crates/bevy_time/README.md, crates/bevy_ecs/README.md
REUSE: compile-fail test setup patterns (ui_test annotations, .stderr generation), data-driven ECS architecture

[cargo]
PROJECT: cargo
WHAT: Rust package manager and build system handling dependency resolution, compilation, and registry authentication
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, credential/README.md, credential/cargo-credential-1password/README.md, credential/cargo-credential-wincred/README.md
REUSE: credential provider architecture, platform-specific secret storage integrations (1password, Wincred, Keychain, libsecret), build dependency requirements

[core]
PROJECT: vuejs/core
WHAT: Vue 3 core framework with migration build, SFC playground, and comprehensive TypeScript type testing infrastructure
VALUE: high
STATUS: active
PROVEN: none
KEY: packages-private/dts-test/README.md, packages-private/dts-built-test/README.md, packages/vue-compat/README.md
REUSE: dual-mode type validation (source vs built), SFC playground dev setup, Vue 2/3 migration build limitations

[deno]
PROJECT: deno
WHAT: JavaScript/TypeScript/WASM runtime built on V8, Rust, and Tokio with secure defaults and CLI tooling
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, CLAUDE.md, tools/README.md
REUSE: Rust-based runtime assembly (cli/runtime/ext), dprint/dlint tooling, feature-branch git workflow

[elasticsearch]
PROJECT: elasticsearch
WHAT: Elasticsearch search engine with Docker distribution pipeline, packaging tests, and migrating documentation system
VALUE: high
STATUS: active
PROVEN: none
KEY: distribution/docker/README.md, qa/packaging/README.md, docs/README.md
REUSE: multi-flavor Docker build pipeline (UBI/Wolfi/ESS), abstract test class pattern for cross-distribution packaging tests, dual-mode docs (asciidoc 8.x vs markdown 9.x)

[etcd]
PROJECT: etcd
WHAT: Distributed reliable key-value store using Raft consensus, with CLI tools and robustness testing infrastructure
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, etcdutl/README.md, tools/benchmark/README.md
REUSE: etcdutl offline data directory operations (defrag, snapshot restore), official benchmark tooling, robustness testing patterns

[fastapi]
PROJECT: fastapi
WHAT: High-performance Python web framework for building APIs using standard Python type hints and OpenAPI standards
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, docs/zh-hant/docs/benchmarks.md, docs/zh-hant/docs/python-types.md
REUSE: type-hint driven API design, OpenAPI/JSON Schema compatibility, Starlette/Pydantic performance architecture

[flask]
PROJECT: flask
WHAT: Lightweight WSGI web framework with minimal core, extensible architecture, and community-driven extension ecosystem
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, examples/celery/README.md, src/flask/sansio/README.md
REUSE: sansio IO-free core pattern for alternative implementations, Celery async task integration example

[grafana]
PROJECT: grafana
WHAT: Open-source observability platform for querying, visualizing, and alerting on metrics across diverse data sources
VALUE: high
STATUS: active
PROVEN: none
KEY: README.md, packaging/docker/README.md, devenv/README.md
REUSE: Docker image build history (Alpine/Ubuntu base shifts), devenv setup scripts for data sources and SMTP testing

[hermes]
PROJECT: hermes
WHAT: Facebook's JavaScript engine for React Native, featuring ahead-of-time static optimization and compact bytecode
VALUE: high
STATUS: active
PROVEN: WIRE: ASan+Debug with -O1 is the default development build configuration for catching memory bugs; Hermes releases are strictly version-locked to specific React Native versions to prevent crashes
KEY: README.md, CLAUDE.md, tools/hermes-parser/js/CLAUDE.md
REUSE: ASan+Debug build defaults, strict RN version pinning rule, WASM parser build prerequisite for JS packages

[httpx]
PROJECT: httpx
WHAT: Python HTTP/1.1 and HTTP/2 client with sync/async APIs, CLI, and WSGI/ASGI transport support
VALUE: high
STATUS: active
PROVEN: WIRE: Integrated CLI is an optional dependency; strict timeouts applied everywhere; supports direct WSGI/ASGI transport
KEY: README.md, docs/api.md, docs/exceptions.md, docs/async.md
REUSE: requests-compatible API patterns, exception hierarchy design, optional CLI integration, WSGI/ASGI transport abstraction

[langchain]
PROJECT: langchain
WHAT: Python monorepo for LLM application framework with core primitives, classic implementation, and partner integrations
VALUE: high
STATUS: active
PROVEN: WIRE: Monorepo uses uv with editable installs and per-package pyproject.toml; standard-tests provides shared test suite for integrations
KEY: README.md, CLAUDE.md, libs/README.md
REUSE: uv monorepo structure, editable install pattern, standard-tests for integration validation

[llama.cpp]
PROJECT: llama.cpp
WHAT: C/C++ LLM inference engine with quantization, server, and multimodal support
VALUE: high
STATUS: active
PROVEN: WIRE: Hugging Face cache migration standardizes model storage; multimodal support integrated into llama-server
KEY: README.md, tools/quantize/README.md, tools/ui/README.md
REUSE: GGUF quantization pipeline (F32/BF16 to Q4_K_M), HF cache standardization, SvelteKit WebUI architecture

[neovim]
PROJECT: neovim
WHAT: Aggressively refactored Vim editor with embedded terminal, async job control, and cross-language API
VALUE: high
STATUS: active
PROVEN: WIRE: CMake-based build system with convenience Makefile wrapper; XDG base directories and shared data (shada) are core architectural features
KEY: README.md, BUILD.md, src/nvim/vterm/README.md
REUSE: CMake build patterns, vterm adoption strategy, XDG/shada integration

[ollama]
PROJECT: ollama
WHAT: Local LLM runner with REST API, CLI, and desktop app supporting multimodal models and benchmarking
VALUE: high
STATUS: active
PROVEN: WIRE: Integrated benchmarking tool supports multi-model comparison, TTFT tracking, and controlled prompt token length; Go-based desktop app with React UI and hot-reload dev mode
KEY: cmd/bench/README.md, app/README.md, README.md
REUSE: benchmark tool CLI patterns (benchstat/CSV output, warmup epochs), Go+React desktop app dev workflow, multimodal prompt handling

[opencode]
PROJECT: opencode
WHAT: Open source AI coding agent with CLI, VS Code extension, and GitHub Action integration
VALUE: high
STATUS: active
PROVEN: WIRE: Multi-platform distribution via npm, brew, scoop, choco, nix, and mise; VS Code extension uses separate workspace debugging workflow with automatic tsc/esbuild watchers
KEY: README.md, sdks/vscode/README.md, github/README.md
REUSE: Cross-platform install patterns, VS Code extension dev workflow, GitHub Action comment-triggered workflow syntax

[polars]
PROJECT: polars
WHAT: High-performance DataFrame query engine written in Rust with Python, Rust, Node.js, and R frontends
VALUE: high
STATUS: active
PROVEN: WIRE: Zero required dependencies for lightweight imports; strict Arrow specification adherence for array equality; internal sub-crate architecture (core, expr, plan)
KEY: README.md, crates/polars-buffer/README.md, crates/polars-expr/README.md, crates/polars-plan/README.md, crates/polars-arrow/src/README.md
REUSE: Zero-dependency import strategy, internal sub-crate modularity, strict Arrow equality operator pattern

[prometheus]
PROJECT: prometheus
WHAT: CNCF systems and service monitoring platform with multi-dimensional time series model and autonomous single-server architecture
VALUE: high
STATUS: active
PROVEN: WIRE: Strict separation of flag-based (restart-required) and file-based (hot-reload) configuration; actor-like concurrency model using oklog/run for component lifecycle coordination
KEY: documentation/internal_architecture.md, documentation/examples/custom-sd/README.md, README.md
REUSE: flag vs file config distinction, oklog/run actor model, file_sd adapter pattern for custom service discovery

[react]
PROJECT: react
WHAT: JavaScript UI library with automated release CI, benchmarking suite, and Rust compiler port in monorepo structure
VALUE: high
STATUS: active
PROVEN: WIRE: Automated CI cron publishes prereleases to canary/experimental channels; benchmarking supports local/remote comparison with optional build skip
KEY: scripts/bench/README.md, scripts/release/README.md, CLAUDE.md
REUSE: automated prerelease CI workflow, benchmarking skip-build optimization, monorepo layout with separate compiler crate

[redis]
PROJECT: redis
WHAT: In-memory data store with C-based core, multi-platform build system, and rigorous test suite with external server support
VALUE: high
STATUS: active
PROVEN: WIRE: Test suite supports external server mode with granular tag-based filtering (cluster, tls, large-memory); build system targets specific Ubuntu/Debian/Alpine/AlmaLinux/Rocky versions
KEY: README.md, tests/README.md, utils/srandmember/README.md
REUSE: external test server configuration, tag-based test filtering, srandmember distribution validation utility

[svelte]
PROJECT: svelte
WHAT: Compiler-based frontend framework with reactive runes, migrating test suite to new runtime with distinct DOM update timing
VALUE: high
STATUS: active
PROVEN: WIRE: New runtime updates DOM asynchronously (after tick), requiring test adjustments; list insertion order optimized back-to-front; CSS minification replaced by unused style commenting
KEY: README.md, packages/svelte/README.md, packages/svelte/tests/README.md, documentation/docs/02-runes/02-$state.md
REUSE: async DOM update test patterns, back-to-front list insertion optimization, rune-based reactive state syntax

[terraform]
PROJECT: terraform
WHAT: Infrastructure-as-code engine with execution planning, resource graphing, and provider ecosystem management
VALUE: high
STATUS: active
PROVEN: WIRE: terraform-bundle is deprecated in favor of built-in `terraform providers mirror` and local registry mirrors (v0.13+); deadlock-free promise implementation in `internal/promising` guarantees no self-dependency deadlocks
KEY: tools/terraform-bundle/README.md, internal/promising/README.md, website/README.md
REUSE: provider distribution migration pattern (bundle to mirror), deadlock-free promise concurrency model, documentation centralization to web-unified-docs

[tokio]
PROJECT: tokio
WHAT: Rust asynchronous runtime with work-stealing scheduler, OS reactor, and non-blocking I/O primitives
VALUE: high
STATUS: active
PROVEN: WIRE: Feature flag combination testing isolated in separate crate to bypass cargo limitations; no-atomic-u64 target spec ensures compatibility with 32-bit architectures
KEY: README.md, tests-build/README.md, target-specs/README.md, docs/contributing/README.md
REUSE: isolated feature flag testing pattern, 32-bit atomic compatibility verification, cargo feature workaround

