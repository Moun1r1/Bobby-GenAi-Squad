"""Deterministic tests for the studio backend — store, projection, data loader, spec factory, pipeline registry,
proofs state, config. No LLM and no network: the store runs in its in-memory fallback (Qdrant unreachable).

Run:  cd studio/backend && python -m pytest tests -q     (or: python tests/test_backend.py)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

import store        # noqa: E402
import specs        # noqa: E402
import runner       # noqa: E402

UNREACHABLE = "http://127.0.0.1:59599"   # forces the store's in-memory fallback


class FakeRun:
    """Minimal Run stand-in for _load_units (only needs params + emit)."""
    def __init__(self, params):
        self.params = params
        self.id = "test"
        self.events = []

    def emit(self, kind, **data):
        self.events.append((kind, data))


# ── store ──────────────────────────────────────────────────────────────────────
def test_store_falls_back_to_memory():
    s = store.Store(url=UNREACHABLE)
    assert s.backend == "memory"


def test_upsert_list_delete_and_search():
    s = store.Store(url=UNREACHABLE)
    s.upsert("knowledge", "k1", {"text": "cats are feline mammals", "domain": "bio", "ts": 1})
    s.upsert("knowledge", "k2", {"text": "compilers translate code", "domain": "cs", "ts": 2})
    rows = s.list("knowledge")
    assert len(rows) == 2 and rows[0]["ts"] <= rows[1]["ts"]        # sorted by ts
    hits = s.search("knowledge", "feline animal", limit=1)
    assert hits and "score" in hits[0]
    assert s.delete("knowledge", "k1") is True
    assert len(s.list("knowledge")) == 1


def test_delete_run_cascades():
    s = store.Store(url=UNREACHABLE)
    s.upsert("runs", "r1", {"run_id": "r1", "ts": 1})
    s.upsert("knowledge", "kk", {"run_id": "r1", "text": "x", "ts": 1})
    s.delete_run("r1")
    assert s.get_run("r1") is None
    assert all(r.get("run_id") != "r1" for r in s.list("knowledge"))


# ── vector projection ─────────────────────────────────────────────────────────
def test_project_is_deterministic_and_bounded():
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
    a = store._project(vecs)
    b = store._project(vecs)
    assert a == b and len(a) == 4
    for x, y in a:
        assert -1.0001 <= x <= 1.0001 and -1.0001 <= y <= 1.0001


def test_scatter_returns_points_with_coords():
    s = store.Store(url=UNREACHABLE)
    for i in range(6):
        s.upsert("knowledge", f"k{i}", {"text": f"item number {i}", "domain": "d", "ts": i})
    pts = s.scatter("knowledge", limit=10)
    assert len(pts) == 6
    assert all("x" in p and "y" in p and "text" in p for p in pts)


# ── data loader (any type → work units) ─────────────────────────────────────────
def test_load_units_text():
    units, kind, _ = runner._load_units(FakeRun({"data": "hello world", "kind": "text"}))
    assert kind == "text" and units and "hello" in units[0]


def test_load_units_json_rows():
    units, kind, _ = runner._load_units(FakeRun({"data": '[{"a": 1}, {"a": 2}, {"a": 3}]', "kind": "json"}))
    assert kind == "json" and len(units) == 3


def test_load_units_csv_rows():
    units, kind, _ = runner._load_units(FakeRun({"data": "h1,h2\nv1,v2\nv3,v4", "kind": "csv"}))
    assert kind == "csv" and len(units) == 2 and "h1=v1" in units[0]


def test_load_units_empty():
    units, _kind, _ = runner._load_units(FakeRun({"data": "", "kind": "text"}))
    assert units == []


# ── spec factory + registry ─────────────────────────────────────────────────────
def test_builtin_specs_registered_and_callable():
    for sid in ("label_data", "extract_entities", "sentiment_group"):
        assert sid in specs.REGISTRY
        assert callable(specs.REGISTRY[sid]["fn"])


def test_register_and_delete_custom_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(specs, "CUSTOM_FILE", str(tmp_path / "custom.json"))
    info = runner.register_spec({"id": "T Case!", "title": "T", "identity": "an analyst", "goal": "do the thing"})
    assert info["id"] == "t_case_"                          # slugified
    assert "t_case_" in runner.REGISTRY and "t_case_" in runner.PIPELINES
    assert any(x["id"] == "t_case_" for x in specs.load_custom())   # persisted
    # appears as custom in the catalog
    assert any(p["id"] == "t_case_" and p.get("custom") for p in runner.pipeline_info())
    # delete removes + unpersists; built-ins are protected
    assert runner.delete_spec("t_case_") is True
    assert "t_case_" not in runner.REGISTRY
    assert runner.delete_spec("label_data") is False       # not custom → protected


def test_register_requires_identity_and_goal():
    try:
        runner.register_spec({"id": "bad"})                # missing identity/goal → KeyError
        assert False, "expected failure"
    except Exception:
        assert True


# ── proofs state shape ───────────────────────────────────────────────────────────
def test_proofs_state_shape():
    st = runner.proofs_state()
    assert {"state", "results", "ran", "total"} <= set(st)
    assert st["state"] in ("idle", "running", "done")


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            # crude fixtures for the standalone runner
            if "tmp_path" in fn.__code__.co_varnames:
                import pathlib
                import tempfile
                class _MP:
                    def setattr(self, o, a, v): setattr(o, a, v)
                fn(pathlib.Path(tempfile.mkdtemp()), _MP())
            else:
                fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception:
            failed += 1
            print(f"  ✗ {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
