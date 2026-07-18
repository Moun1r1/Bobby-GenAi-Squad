import ast
import hashlib
from typing import Optional, Set


class _Canon(ast.NodeTransformer):
    """Alpha-rename every identifier (Name ids + arg names) to a canonical token by first-seen order, so renaming a
    variable/param doesn't change the fingerprint. Attribute names, method/class names, and literals are preserved
    (they carry functional meaning — the interface, the ops called)."""

    def __init__(self):
        self._map = {}

    def _tok(self, ident: str) -> str:
        if ident not in self._map:
            self._map[ident] = f"v{len(self._map)}"
        return self._map[ident]

    def visit_Name(self, node: ast.Name):
        node.id = self._tok(node.id)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._tok(node.arg)
        node.annotation = None                                     # annotations are cosmetic for functional identity
        return node


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]


def fingerprint(src: str) -> Optional[str]:
    """A 16-hex functional fingerprint, invariant to docstrings/comments/whitespace/identifier spelling. None if the
    source doesn't parse (a non-building candidate is not a 'duplicate' — let the build/proof reject it)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    _strip_docstrings(tree)
    tree = _Canon().visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=False)                  # structural, no line/col noise
    return hashlib.sha256(dump.encode()).hexdigest()[:16]


class AstDedup:
    """A running set of functional fingerprints. `is_dup(src)` → True if a functional twin was already added."""

    def __init__(self):
        self.seen: Set[str] = set()

    def is_dup(self, src: str) -> bool:
        fp = fingerprint(src)
        return fp is not None and fp in self.seen

    def add(self, src: str) -> Optional[str]:
        fp = fingerprint(src)
        if fp is not None:
            self.seen.add(fp)
        return fp

    def __len__(self) -> int:
        return len(self.seen)
