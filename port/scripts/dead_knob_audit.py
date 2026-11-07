#!/usr/bin/env python3
"""Find the NEXT config knob that exists, validates, and does nothing.

The defect this instrument hunts has cost this project weeks: a field on
`ModelConfig`/`TrainConfig` (or a sub-config) that is declared, defaulted,
validated — and read by no code that can change a run, or read only to be
overwritten later. Confirmed instances: `muon_lr_mult`, `fused_residual`,
`use_msa`, `doc_masking` (0.06 % masked), `seq_len=512`.

Two passes, one verdict per field:

* **static** — AST, not grep. Every read of a config field outside
  `config.py` is located: attribute access (`cfg.d_model`), `getattr(cfg,
  "d_model")`, `cfg["d_model"]`, `cfg.get("d_model")`. Receivers are resolved
  through parameter annotations, one-hop aliases (`tcfg = run.train`), class
  attributes (`self.cfg = cfg`), `RunConfig.model`/`.train`, and a small name
  fallback for unannotated code (`cfg`, `run`, `model.cfg`).
* **runtime** (`--runtime PRESET`) — install an attribute hook on the config
  classes and run the real trainer for two tiny CPU steps. Every field read
  from a non-`config.py` frame is recorded, so the static verdict is checked
  against a live run.

Verdicts:

  LIVE       read in a non-test module, on a path that can affect a run
  LOG-ONLY   read only to build a log/report line
  DEAD       never read outside `config.py` (validators do not count)
  SHADOWED   read, and a later write to the same state slot drops the value

Two extra evidence flags are printed but are not verdicts:

  init-only  every behavioural read is inside `__init__`/`__post_init__`
             (the `muon_lr_mult` shape: a later step overwrites the effect)
  guarded    `config.py` refuses a non-default value loudly, so it cannot lie
             silently even though it reaches no computation

Exit codes: 0 clean, 1 findings (`DEAD`/`SHADOWED`), 2 `PRESET-USES-DEAD` —
a preset explicitly sets a dead field to a non-default value. 2 is the gate
that would have caught `muon_lr_mult=8` in a preset.

Usage:

    python scripts/dead_knob_audit.py
    python scripts/dead_knob_audit.py --runtime mirror_control
    python scripts/dead_knob_audit.py --json /tmp/audit.json --markdown /tmp/audit.md

The static pass is offline and takes a second; `--runtime` imports torch and
runs the trainer on CPU, so it is opt-in.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

PORT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = ("src/dormouse/config.py", "src/dormouse/post/rl.py")
DEFAULT_ROOTS = ("src/dormouse",)
DEFAULT_PRESET_GLOB = "configs/**/*.toml"
DEFAULT_CLASSES = ("ModelConfig", "TrainConfig", "RLConfig")

#: Unannotated receivers this codebase uses for a config object. `cfg` is the
#: dominant one (700+ reads); `run` is the resolved `RunConfig`.
NAME_HINTS: dict[str, str] = {"cfg": "*", "run": "RunConfig"}

#: Receivers that are structurally a config even when the type does not resolve
#: (`model.cfg`, `self.model.cfg`, `stored.model`, ...).
RECEIVER_FALLBACK = re.compile(
    r"(^|\.)(cfg|mcfg|tcfg)$"
    r"|\.cfg$"
    r"|^(run|stored|self\.run)\.(model|train)$"
)

#: Enclosing-function names that build a log line rather than a computation.
#: `_eval_line` is deliberately NOT here: it reads `eval_depths` to decide
#: whether to run the extra depth forwards, so its reads change the run.
LOG_FUNC_SUFFIXES = ("_field",)
LOG_CALL_SUFFIXES = (".print", ".log", ".info", ".warning", ".error", ".debug", ".report")
LOG_CALL_NAMES = {"print", "log", "_log", "log_line", "report", "warn", "info", "debug"}

#: A write that captures a config value: `self.x = cfg.f`, `self.x = int(cfg.f)`,
#: `self.x = getattr(cfg, "f", d)`. Anything that computes with it is not a
#: capture and cannot be "overwritten without the knob".
PURE_CALLS = {"float", "int", "bool", "str", "tuple", "list", "getattr", "max", "min"}
NOT_PURE = (
    ast.BinOp,
    ast.Compare,
    ast.IfExp,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Starred,
)

#: Construction: a write from a helper called here is still `__init__`'s work.
CONSTRUCTION = re.compile(
    r"^(__init__|__post_init__|__new__|_*init|_*build|_*setup|_*configure|_*make|from_)"
)

INIT_FUNCS = {"__init__", "__post_init__", "__new__"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    cls: str
    path: str
    line: int
    annotation: str
    default: str


@dataclass
class ReadSite:
    field: str
    path: str
    line: int
    func: str
    kind: str
    log_only: bool
    orphan: bool = False

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass
class WriteSite:
    path: str
    line: int
    func: str
    slot: str
    taint: frozenset[str]
    direct_field: str | None = None
    pure: bool = False
    aug: bool = False
    self_ref: bool = False
    reset: bool = False

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass
class FieldReport:
    spec: FieldSpec
    verdict: str
    reads: list[ReadSite] = dc_field(default_factory=list)
    plumbing: list[ReadSite] = dc_field(default_factory=list)
    shadow_writes: list[WriteSite] = dc_field(default_factory=list)
    aliased_to: list[str] = dc_field(default_factory=list)
    bridged_to: list[str] = dc_field(default_factory=list)
    guarded: list[str] = dc_field(default_factory=list)
    init_only: bool = False
    presets: list[str] = dc_field(default_factory=list)
    preset_non_default: list[str] = dc_field(default_factory=list)
    runtime: list[str] = dc_field(default_factory=list)

    @property
    def orphan_note(self) -> str:
        seen = sorted({r.path for r in self.reads if r.orphan})
        return ", ".join(_short(p) for p in seen[:3])


# ---------------------------------------------------------------------------
# field extraction (AST only: no import, so a temp module needs no package)
# ---------------------------------------------------------------------------


def _class_decorated_dataclass(node: ast.ClassDef) -> bool:
    for d in node.decorator_list:
        name = ast.unparse(d)
        if name in ("dataclass", "dataclasses.dataclass") or name.startswith("dataclass("):
            return True
    return False


def load_fields(
    paths: list[Path], wanted: set[str] | None = None
) -> tuple[dict[str, FieldSpec], set[str]]:
    """Return (field name -> spec, class -> fields, class set).

    Only the config classes are reported: the wrapper (`RunConfig`) and the
    dataclasses that are runtime values (`Sample`, `RLAdvantage`) are not
    knobs. `wanted` names the roots; a sub-config referenced by a field
    annotation is pulled in transitively.
    """
    parsed: dict[str, tuple[Path, ast.ClassDef]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _class_decorated_dataclass(node):
                parsed[node.name] = (path, node)
    roots = set(wanted or ())
    missing = roots - parsed.keys()
    if missing:
        raise SystemExit(f"config class(es) not found: {sorted(missing)}")
    keep = set(roots)
    for _ in range(3):
        for cls in list(keep):
            path, node = parsed[cls]
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ast.unparse(stmt.annotation)))
                    keep |= toks & parsed.keys()

    specs: dict[str, FieldSpec] = {}
    classes: set[str] = set()
    for cls in sorted(keep):
        path, node = parsed[cls]
        classes.add(cls)
        for stmt in node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            name = stmt.target.id
            if name in specs:
                # A duplicate name across two config classes makes the verdict
                # ambiguous; the caller is told rather than guessed.
                print(
                    f"WARNING: field {name!r} is declared in both "
                    f"{specs[name].cls} and {cls}; reports are per name.",
                    file=sys.stderr,
                )
            specs[name] = FieldSpec(
                name=name,
                cls=cls,
                path=str(path),
                line=stmt.lineno,
                annotation=ast.unparse(stmt.annotation),
                default=ast.unparse(stmt.value) if stmt.value is not None else "<required>",
            )
    return specs, classes


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def _walk_scope(nodes: list[ast.stmt]):
    """Yield descendants of one scope, not entering nested defs/classes."""

    def walk(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield child
            yield from walk(child)

    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield n
        yield from walk(n)


def _is_log_call(node: ast.Call) -> bool:
    name = _call_name(node)
    short = name.rsplit(".", 1)[-1]
    return (
        name in LOG_CALL_NAMES
        or short in LOG_CALL_NAMES
        or name.endswith(LOG_CALL_SUFFIXES)
        or short in ("print", "warn", "log", "info", "debug")
    )


def _walk_scope_ctx(nodes: list[ast.stmt]):
    """Like `_walk_scope`, carrying whether the node sits inside a log call."""

    def walk(node: ast.AST, in_log: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            child_log = in_log or (isinstance(child, ast.Call) and _is_log_call(child))
            yield child, child_log
            yield from walk(child, child_log)

    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield n, False
        yield from walk(n, False)


def _walk_all(nodes: list[ast.stmt]):
    """Yield every node, entering nested scopes (used to find defs)."""

    def walk(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            yield child
            yield from walk(child)

    for n in nodes:
        yield n
        yield from walk(n)


def _call_name(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:  # noqa: BLE001 - unparse cannot fail on a parsed tree
        return ""


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f"..." with one constant? not a key
        return None
    return None


def _pure_capture(value: ast.AST) -> bool:
    for node in ast.walk(value):
        if isinstance(node, NOT_PURE):
            return False
        if isinstance(node, ast.Call) and _call_name(node) not in PURE_CALLS:
            return False
    return True


def _is_reset(value: ast.AST) -> bool:
    """`self.cache = None` / `= ()` / `= 0`: a reset, not an authoritative value."""
    if isinstance(value, ast.Constant):
        return value.value in (None, False, 0, "", 0.0)
    return isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)) and not list(
        ast.iter_child_nodes(value)
    )


def module_names(roots: list[Path]) -> dict[Path, str]:
    """Map each scanned file to `dormouse.model.loopblock`-style module name."""
    out: dict[Path, str] = {}
    for root in roots:
        top = root.name
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).with_suffix("")
            parts = list(rel.parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            out[path] = ".".join([top, *parts]) if parts else top
    return out


def reachable_modules(trees: dict[Path, ast.Module], names: dict[Path, str]) -> set[str]:
    """Modules reachable from the two entry points a run has: `cli` and package init."""
    known = set(names.values())
    if not any(m in known for m in ("dormouse.cli", "dormouse")):
        return known  # not the dormouse package: no entry point to walk from
    by_name = {names[p]: p for p in trees}

    def edges(path: Path) -> list[str]:
        mod = names[path]
        pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
        out: list[str] = []
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.Import):
                out += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg
                    for _ in range(node.level - 1):
                        base = base.rsplit(".", 1)[0] if "." in base else base
                else:
                    base = ""
                modname = ".".join(x for x in (base, node.module or "") if x)
                out.append(modname)
                out += [f"{modname}.{a.name}" if modname else a.name for a in node.names]
        return [m for m in out if m in known]

    todo = [m for m in ("dormouse.cli", "dormouse") if m in known]
    seen: set[str] = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in by_name:
            todo.extend(m for m in edges(by_name[cur]) if m not in seen)
    return seen


class Scanner:
    """Resolves config-typed expressions, then records reads and writes."""

    def __init__(
        self,
        specs: dict[str, FieldSpec],
        config_classes: set[str],
        class_attrs: dict[str, dict[str, set[str]]],
        returns: dict[str, set[str]],
    ) -> None:
        self.specs = specs
        self.config_classes = config_classes
        self.class_attrs = class_attrs
        self.returns = returns
        self.reads: list[ReadSite] = []
        self.writes: list[WriteSite] = []
        self.unknown_receivers: dict[str, int] = {}
        self.path = ""

    # -- type resolution ---------------------------------------------------

    def annotation_classes(self, node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        text = ast.unparse(node)
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        return toks & self.class_attrs.keys()

    def resolve(self, node: ast.AST, env: dict[str, set[str]]) -> set[str]:
        if isinstance(node, ast.Name):
            return env.get(node.id, set())
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value, env)
            out: set[str] = set()
            for cls in base:
                out |= self.class_attrs.get(cls, {}).get(node.attr, set())
            return out
        if isinstance(node, ast.Call):
            return self.returns.get(_call_name(node), set())
        return set()

    def _receiver_classes(self, node: ast.AST, env: dict[str, set[str]]) -> set[str]:
        got = self.resolve(node, env)
        if got:
            return got
        text = ast.unparse(node)
        if RECEIVER_FALLBACK.search(text):
            return set(self.config_classes)
        return set()

    # -- env building ------------------------------------------------------

    def _seed_params(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, set[str]]:
        env: dict[str, set[str]] = {}
        args = fn.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if a is None:
                continue
            ann = self.annotation_classes(a.annotation)
            if ann:
                env[a.arg] = ann
            elif a.annotation is None and a.arg in NAME_HINTS:
                hint = NAME_HINTS[a.arg]
                env[a.arg] = set(self.config_classes) if hint == "*" else {hint}
        return env

    def _build_env(self, nodes: list[ast.stmt], base: dict[str, set[str]]) -> dict[str, set[str]]:
        env = {k: set(v) for k, v in base.items()}
        for _ in range(6):
            changed = False
            for node in _walk_all(nodes):
                targets: list[ast.expr] = []
                value: ast.AST | None = None
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets, value = [node.target], node.value
                if not targets or value is None:
                    continue
                classes = self.annotation_classes(
                    node.annotation if isinstance(node, ast.AnnAssign) else None
                ) or self.resolve(value, env)
                if not classes:
                    continue
                for t in targets:
                    if isinstance(t, ast.Name) and env.get(t.id) != classes:
                        env[t.id] = set(classes)
                        changed = True
            if not changed:
                break
        return env

    # -- scanning ----------------------------------------------------------

    def scan_module(self, path: Path, tree: ast.Module) -> None:
        self.path = str(path)
        module_env: dict[str, set[str]] = {}
        # Class attributes first: `self.cfg = cfg` makes `self.cfg` a config in
        # every method of the class.
        attrs: dict[str, dict[str, set[str]]] = {}
        methods: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods[node.name] = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        for _ in range(4):
            changed = False
            for cls, fns in methods.items():
                known = attrs.setdefault(cls, {})
                for fn in fns:
                    env = self._build_env(fn.body, {**module_env, **self._seed_params(fn)})
                    for node in _walk_all(fn.body):
                        if not isinstance(node, ast.Assign):
                            continue
                        classes = self.resolve(node.value, env)
                        if not classes:
                            continue
                        for t in node.targets:
                            if (
                                isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                                and known.get(t.attr) != classes
                            ):
                                known[t.attr] = set(classes)
                                changed = True
            if not changed:
                break
        for cls, known in attrs.items():
            self.class_attrs.setdefault(cls, {}).update(known)

        self._scan_scope(tree.body, module_env, None, "<module>")

    def _scan_scope(
        self,
        nodes: list[ast.stmt],
        base_env: dict[str, set[str]],
        cls_name: str | None,
        func_name: str,
    ) -> None:
        env = self._build_env(nodes, base_env)
        if cls_name:
            env["self"] = {cls_name}
        taint = self._build_taint(nodes, env)
        for node, in_log in _walk_scope_ctx(nodes):
            self._record_reads(node, env, func_name, in_log)
            self._record_write(node, env, taint, cls_name, func_name)
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = {**env, **self._seed_params(node)}
                self._scan_scope(node.body, inner, cls_name, node.name)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._scan_scope(
                            item.body, {**env, **self._seed_params(item)}, node.name, item.name
                        )

    def _build_taint(self, nodes: list[ast.stmt], env: dict[str, set[str]]) -> dict[str, set[str]]:
        taint: dict[str, set[str]] = {}
        for _ in range(4):
            changed = False
            for node in _walk_all(nodes):
                if isinstance(node, ast.Assign):
                    got = self._fields_in(node.value, env, taint)
                    if not got:
                        continue
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            merged = taint.get(t.id, set()) | got
                            if merged != taint.get(t.id):
                                taint[t.id] = merged
                                changed = True
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    if isinstance(node.target, ast.Name):
                        got = self._fields_in(node.value, env, taint)
                        if got:
                            merged = taint.get(node.target.id, set()) | got
                            if merged != taint.get(node.target.id):
                                taint[node.target.id] = merged
                                changed = True
            if not changed:
                break
        return taint

    def _fields_in(
        self, node: ast.AST, env: dict[str, set[str]], taint: dict[str, set[str]]
    ) -> set[str]:
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out |= taint.get(sub.id, set())
            field = self._read_field(sub, env)
            if field:
                out.add(field)
        return out

    def _read_field(self, node: ast.AST, env: dict[str, set[str]]) -> str | None:
        def owns(attr: str, classes: set[str]) -> bool:
            return bool(classes) and ("*" in classes or self.specs[attr].cls in classes)

        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr not in self.specs:
                return None
            classes = self._receiver_classes(node.value, env)
            if owns(node.attr, classes):
                return node.attr
            if not classes:
                text = ast.unparse(node)
                self.unknown_receivers[text] = self.unknown_receivers.get(text, 0) + 1
            return None
        if isinstance(node, ast.Call) and _call_name(node) == "getattr" and len(node.args) >= 2:
            key = _const_str(node.args[1])
            if key in self.specs:
                classes = self._receiver_classes(node.args[0], env)
                if owns(key, classes):
                    return key
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "pop") and len(node.args) >= 1:
                key = _const_str(node.args[0])
                if key in self.specs:
                    classes = self._receiver_classes(node.func.value, env)
                    if owns(key, classes):
                        return key
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = _const_str(node.slice)
            if key in self.specs:
                classes = self._receiver_classes(node.value, env)
                if owns(key, classes):
                    return key
        return None

    def _record_reads(
        self, node: ast.AST, env: dict[str, set[str]], func_name: str, in_log: bool
    ) -> None:
        field = self._read_field(node, env)
        if not field:
            return
        self.reads.append(
            ReadSite(
                field=field,
                path=self.path,
                line=node.lineno,
                func=func_name,
                kind=type(node).__name__,
                log_only=in_log or self._is_log(func_name),
            )
        )

    def _is_log(self, func_name: str) -> bool:
        return func_name not in INIT_FUNCS and func_name.endswith(LOG_FUNC_SUFFIXES)

    def _record_write(
        self,
        node: ast.AST,
        env: dict[str, set[str]],
        taint: dict[str, set[str]],
        cls_name: str | None,
        func_name: str,
    ) -> None:
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value
        if value is None:
            return
        got = frozenset(self._fields_in(value, env, taint))
        for target in targets:
            if not isinstance(target, (ast.Attribute, ast.Subscript)):
                continue
            slot = ast.unparse(target)
            direct: str | None = None
            base = target.value
            kind = target.attr if isinstance(target, ast.Attribute) else _const_str(target.slice)
            if kind in self.specs:
                classes = self._receiver_classes(base, env)
                if classes and ("*" in classes or self.specs[kind].cls in classes):
                    direct = kind
            base_text = ast.unparse(base)
            self_ref = any(
                (
                    isinstance(n, ast.Attribute)
                    and n.attr == kind
                    and ast.unparse(n.value) == base_text
                )
                or (
                    isinstance(n, ast.Subscript)
                    and kind is not None
                    and _const_str(n.slice) == kind
                    and ast.unparse(n.value) == base_text
                )
                for n in ast.walk(value)
            )
            self.writes.append(
                WriteSite(
                    path=self.path,
                    line=node.lineno,
                    func=f"{cls_name}.{func_name}" if cls_name else func_name,
                    slot=slot,
                    taint=got,
                    direct_field=direct,
                    pure=_pure_capture(value),
                    aug=isinstance(node, ast.AugAssign),
                    self_ref=self_ref,
                    reset=_is_reset(value),
                )
            )


# ---------------------------------------------------------------------------
# class-attribute and return-type pre-pass (cross-module)
# ---------------------------------------------------------------------------


def build_type_index(
    trees: dict[Path, ast.Module], config_classes: set[str]
) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]]]:
    returns: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.returns is not None
            ):
                toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ast.unparse(node.returns)))
                hit = toks & config_classes
                if hit:
                    returns[node.name] = hit
    # Class attributes: `self.cfg = cfg` in DormouseModel.__init__ makes
    # `model.cfg` resolvable, and an annotated dataclass field
    # (`RunConfig.model: ModelConfig`) makes `run.model` resolvable. One pass
    # per level; two levels cover `self.outer.cfg`.
    attrs: dict[str, dict[str, set[str]]] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            known = attrs.setdefault(node.name, {})
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ast.unparse(stmt.annotation)))
                    hit = toks & config_classes
                    if hit:
                        known[stmt.target.id] = hit
    for _ in range(3):
        changed = False
        for tree in trees.values():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                known = attrs.setdefault(node.name, {})
                for fn in node.body:
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    env: dict[str, set[str]] = {}
                    for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
                        toks = (
                            set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ast.unparse(a.annotation)))
                            if a.annotation
                            else set()
                        )
                        if toks & config_classes:
                            env[a.arg] = toks & config_classes
                        elif a.annotation is None and a.arg in NAME_HINTS:
                            hint = NAME_HINTS[a.arg]
                            env[a.arg] = set(config_classes) if hint == "*" else {hint}
                    for sub in ast.walk(fn):
                        if not isinstance(sub, ast.Assign):
                            continue
                        classes: set[str] = set()
                        if isinstance(sub.value, ast.Name):
                            classes = env.get(sub.value.id, set())
                        elif isinstance(sub.value, ast.Attribute):
                            base = (
                                env.get(sub.value.value.id, set())
                                if isinstance(sub.value.value, ast.Name)
                                else set()
                            )
                            for cls in base:
                                classes |= attrs.get(cls, {}).get(sub.value.attr, set())
                        if not classes:
                            continue
                        for t in sub.targets:
                            if (
                                isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                                and known.get(t.attr) != classes
                            ):
                                known[t.attr] = set(classes)
                                changed = True
        if not changed:
            break
    return attrs, returns


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def load_presets(glob: str, specs: dict[str, FieldSpec]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    import glob as globmod

    pattern = Path(glob)
    if pattern.is_absolute():
        files = sorted(Path(p) for p in globmod.glob(glob, recursive=True))
    else:
        files = sorted(PORT.glob(glob))
    for path in files:
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:  # a broken preset is its own defect
            print(f"WARNING: preset {path} does not parse: {e}", file=sys.stderr)
            continue
        flat: dict[str, object] = {}

        def flatten(table: dict, prefix: str = "", into: dict = flat):
            for k, v in table.items():
                if isinstance(v, dict):
                    flatten(v, prefix, into)
                else:
                    into[k] = v

        flatten(raw)
        key = str(path.relative_to(PORT)) if path.is_relative_to(PORT) else str(path)
        out[key] = {k: v for k, v in flat.items() if k in specs}
    return out


def _is_default(value: object, spec: FieldSpec) -> bool:
    text = spec.default.strip()
    if text in ("True", "False", "None"):
        return value is {"True": True, "False": False, "None": None}[text]
    try:
        return value == ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False


# ---------------------------------------------------------------------------
# static report
# ---------------------------------------------------------------------------


def static_audit(
    config_paths: list[Path],
    roots: list[Path],
    preset_glob: str,
    classes: set[str] | None = None,
) -> tuple[dict[str, FieldReport], list[str], Scanner]:
    specs, config_classes = load_fields(config_paths, classes)
    trees: dict[Path, ast.Module] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path.name == "config.py" and path in config_paths:
                continue
            if "tests" in path.parts:
                continue
            trees[path] = ast.parse(path.read_text(), filename=str(path))
    config_trees = {p: ast.parse(p.read_text(), filename=str(p)) for p in config_paths}
    class_attrs, returns = build_type_index({**trees, **config_trees}, config_classes)
    scanner = Scanner(specs, config_classes, class_attrs, returns)
    for path, tree in trees.items():
        scanner.scan_module(path, tree)
    names = module_names(roots)
    reachable = reachable_modules(trees, names)
    path_module = {str(p): names[p] for p in names}
    for site in scanner.reads:
        site.orphan = path_module.get(site.path, "") not in reachable

    plumbing: dict[str, list[ReadSite]] = {name: [] for name in specs}
    config_scanner = Scanner(specs, config_classes, class_attrs, returns)
    for path, tree in config_trees.items():
        config_scanner.scan_module(path, tree)
    for site in config_scanner.reads:
        plumbing[site.field].append(site)

    aliases: dict[str, set[str]] = {name: set() for name in specs}
    # `__post_init__: self.a = self.b` — b's value lives on as a; and
    # `if self.a: self.b = ...` — the branch makes a an alias of b.
    for _path, tree in config_trees.items():
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != "__post_init__":
                    continue
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        reads = {
                            n.attr
                            for n in ast.walk(node.value)
                            if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name)
                            and n.value.id == "self"
                        }
                        targets = [
                            t.attr
                            for t in node.targets
                            if isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                            and t.attr in specs
                        ]
                        for r in reads & specs.keys():
                            aliases[r].update(targets)
                    elif isinstance(node, ast.If):
                        reads = {
                            n.attr
                            for n in ast.walk(node.test)
                            if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name)
                            and n.value.id == "self"
                            and n.attr in specs
                        }
                        targets = {
                            t.attr
                            for stmt in node.body
                            for t in ast.walk(stmt)
                            if isinstance(t, ast.Attribute)
                            and isinstance(t.ctx, ast.Store)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                            and t.attr in specs
                        }
                        for r in reads:
                            aliases[r].update(targets - {r})
    # `rl_config(tcfg) -> RLConfig(quality=tcfg.rl_quality)` — a bridge into a
    # sub-config class; the source field's value lives on as the target field.
    bridges: dict[str, set[str]] = {name: set() for name in specs}
    for _path, tree in config_trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _call_name(node)
            target_cls = None
            for c in config_classes:
                if callee.endswith(c):
                    target_cls = c
                    break
            if target_cls is None:
                continue
            for kw in node.keywords:
                for sub in ast.walk(kw.value):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.attr in specs
                    ):
                        bridges[sub.attr].add(f"{target_cls}.{kw.arg}")

    shadow: dict[str, list[WriteSite]] = {name: [] for name in specs}
    direct: dict[str, list[WriteSite]] = {name: [] for name in specs}
    groups: dict[tuple[str, str, str], dict[str, list[WriteSite]]] = {}

    def _group(w: WriteSite) -> dict[str, list[WriteSite]]:
        key = (w.path, w.func.split(".")[0], w.slot)
        return groups.setdefault(key, {"derived": [], "plain": []})

    for w in scanner.writes:
        if w.direct_field:
            direct[w.direct_field].append(w)
            continue
        if w.taint:
            # A value captured from a knob into a state slot. A computed value
            # (`self.pos = f(self.pos, cfg.seq_len)`) is not a capture: rewriting
            # it later is not shadowing the knob.
            if w.pure:
                _group(w)["derived"].append(w)
            continue
        # A plain write only counts if it REPLACES the slot (`+=` accumulates a
        # cursor) and is not construction work, and does not read the slot.
        func = w.func.split(".")[-1]
        if w.aug or w.self_ref or CONSTRUCTION.match(func) or w.reset:
            continue
        _group(w)["plain"].append(w)
    for _key, g in groups.items():
        for d in g["derived"]:
            plain = [p for p in g["plain"] if p.func != d.func]
            if not plain:
                continue
            for f in d.taint:
                shadow[f].extend([d, *plain])

    reports: dict[str, FieldReport] = {}
    for name, spec in specs.items():
        reads = [r for r in scanner.reads if r.field == name]
        rep = FieldReport(spec=spec, verdict="DEAD", reads=reads, plumbing=plumbing[name])
        rep.aliased_to = sorted(aliases[name])
        rep.bridged_to = sorted(bridges[name])
        non_log = [r for r in reads if not r.log_only]
        reachable_read = [r for r in non_log if not r.orphan]
        if reachable_read:
            rep.verdict = "LIVE"
        elif non_log:
            rep.verdict = "DEAD"  # read only by a module nothing imports
        elif reads:
            rep.verdict = "LOG-ONLY"
        reports[name] = rep

    # An alias (`share_u` -> `share_u_mode`) or a bridge
    # (`rl_quality` -> `RLConfig.quality`) carries the value to a live field.
    # Resolve to a fixpoint: a bridge of an alias must also count.
    for _ in range(4):
        live = {n for n, r in reports.items() if r.verdict == "LIVE"}
        changed = False
        for rep in reports.values():
            if rep.verdict == "LIVE":
                continue
            if any(a in live for a in rep.aliased_to) or any(
                b.split(".")[-1] in live for b in rep.bridged_to
            ):
                rep.verdict = "LIVE"
                changed = True
        if not changed:
            break

    for name, rep in reports.items():
        if rep.verdict == "LIVE":
            if direct[name]:
                shadow[name].extend(direct[name])
            if shadow[name]:
                rep.verdict = "SHADOWED"
                rep.shadow_writes = shadow[name]
            reachable_read = [r for r in rep.reads if not r.log_only and not r.orphan]
            rep.init_only = bool(reachable_read) and all(
                r.func in INIT_FUNCS for r in reachable_read
            )

    guards = collect_guards(config_trees, specs)
    for name, rep in reports.items():
        rep.guarded = sorted(guards.get(name, []))

    findings = [
        f"{r.verdict}: {n}" for n, r in sorted(reports.items()) if r.verdict in ("DEAD", "SHADOWED")
    ]
    return reports, findings, scanner


def collect_guards(
    config_trees: dict[Path, ast.Module], specs: dict[str, FieldSpec]
) -> dict[str, list[str]]:
    """Fields a `loud(...)` refusal names: a non-default value cannot lie silently."""
    out: dict[str, list[str]] = {}
    for path, tree in config_trees.items():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                if not _call_name(node).endswith("loud"):
                    continue
                for arg in node.args:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            for name in specs:
                                if re.search(rf"\b{re.escape(name)}\b", sub.value):
                                    out.setdefault(name, []).append(f"{path}:{node.lineno}")
    return out


# ---------------------------------------------------------------------------
# runtime pass
# ---------------------------------------------------------------------------


def runtime_audit(
    preset: str, config_paths: list[Path], steps: int = 2, workdir: Path | None = None
) -> dict[str, list[str]]:
    """Run `preset` on CPU for a few tiny steps and record config field reads."""
    sys.path.insert(0, str(PORT / "src"))
    from dormouse.config import ModelConfig, TrainConfig, load_run
    from dormouse.model import DormouseModel
    from dormouse.post.rl import RLConfig
    from dormouse.train.loop import train

    tmp = Path(tempfile.mkdtemp(prefix="dead-knob-", dir=workdir))
    try:
        corpus = tmp / "corpus.bin"
        corpus.write_bytes((bytes(range(16)) * 512)[:8192])
        over = {
            "steps": steps,
            "batch": 1,
            "seq_len": 64,
            "eval": False,
            "log_every": 1,
            "ckpt_every": 10**9,
            "json_log": "",
            "out_dir": str(tmp),
            "ckpt_name": "audit",
            "data": str(corpus),
        }
        run = load_run(preset, train_overrides=over)

        records: dict[str, list[str]] = {}
        field_names: dict[type, set[str]] = {}
        orig_getattr: dict[type, object] = {}
        src_root = str(PORT / "src")
        config_files = {str(p) for p in config_paths}

        def hook(cls: type):
            names = set(field_names[cls])
            prev = cls.__getattribute__

            def __getattribute__(self, name, _prev=prev, _names=names):
                if name in _names:
                    frame = sys._getframe(1)
                    file = frame.f_code.co_filename
                    if file.startswith(src_root) and file not in config_files:
                        rel = str(Path(file).relative_to(PORT))
                        records.setdefault(name, []).append(f"{rel}:{frame.f_lineno}")
                return _prev(self, name)

            orig_getattr[cls] = prev
            cls.__getattribute__ = __getattribute__  # type: ignore[method-assign]

        for cls in (ModelConfig, TrainConfig, RLConfig):
            import dataclasses as _dc

            field_names[cls] = {f.name for f in _dc.fields(cls)}
            hook(cls)
        try:
            model = DormouseModel(run.model, device="cpu")
            train(model, run, log=lambda *a, **k: None, device=None)
        finally:
            for cls, prev in orig_getattr.items():
                cls.__getattribute__ = prev  # type: ignore[method-assign]
        return {k: sorted(set(v)) for k, v in records.items()}
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def print_report(
    reports: dict[str, FieldReport],
    presets: dict[str, dict[str, object]],
    *,
    verbose: bool,
) -> list[str]:
    dead_preset: list[str] = []
    for name, rep in sorted(reports.items()):
        users = [p for p, keys in presets.items() if name in keys]
        rep.presets = users
        if users:
            rep.preset_non_default = [
                p
                for p, keys in presets.items()
                if name in keys and not _is_default(keys[name], rep.spec)
            ]
    for name, rep in sorted(reports.items()):
        if rep.verdict == "LIVE" and not (verbose or rep.init_only):
            continue
        flags = []
        if rep.init_only:
            flags.append("init-only")
        if rep.guarded:
            flags.append("guarded@" + ",".join(rep.guarded))
        if rep.aliased_to:
            flags.append("alias->" + ",".join(rep.aliased_to))
        if rep.bridged_to:
            flags.append("bridge->" + ",".join(rep.bridged_to))
        if rep.presets:
            flags.append("presets=" + ",".join(rep.presets))
        print(
            f"{rep.verdict:9s} {name:26s} {rep.spec.cls}.{rep.spec.name} "
            f"{rep.spec.path}:{rep.spec.line} {' '.join(flags)}".rstrip()
        )
        if rep.verdict in ("DEAD", "SHADOWED", "LOG-ONLY") or verbose:
            for r in rep.reads:
                mark = "log" if r.log_only else ("orphan" if r.orphan else "read")
                print(f"    {mark:6s} {r.where} {r.kind} in {r.func}")
            for w in rep.shadow_writes:
                print(f"    WRITE {w.where} {w.slot} in {w.func} taint={sorted(w.taint)}")
            for p in rep.plumbing:
                print(f"    config {p.where} {p.func} (validation/plumbing)")
    for name, rep in sorted(reports.items()):
        if rep.verdict == "DEAD" and rep.preset_non_default:
            dead_preset.append(name)
            print(
                f"PRESET-USES-DEAD: {name} "
                f"({', '.join(f'{p}={presets[p][name]!r}' for p in rep.preset_non_default)})"
            )
    for name, rep in sorted(reports.items()):
        if rep.verdict == "SHADOWED":
            print(f"SHADOWED: {name}")
    return dead_preset


def _short(path: str) -> str:
    try:
        return str(Path(path).relative_to(PORT.parent))
    except ValueError:
        return path


def write_markdown(
    path: Path,
    reports: dict[str, FieldReport],
    presets: dict[str, dict[str, object]],
    dead_preset: list[str],
    runtime: dict[str, list[str]] | None = None,
    runtime_unread: list[str] | None = None,
) -> None:
    lines = [
        "# Dead-knob audit",
        "",
        "Generated by `port/scripts/dead_knob_audit.py`. One row per config field.",
        "",
        "| field | class | type | default | decl | verdict | evidence | presets |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rep in sorted(reports.items()):
        ev = ""
        if rep.verdict == "SHADOWED":
            ev = "; ".join(
                f"write {_short(w.path)}:{w.line} `{w.slot}`" for w in rep.shadow_writes[:3]
            )
        elif rep.verdict == "LOG-ONLY":
            ev = "; ".join(_short(r.path) + f":{r.line}" for r in rep.reads[:3])
        elif rep.verdict == "DEAD":
            if rep.orphan_note:
                ev = "read only in unreachable module: " + rep.orphan_note
            else:
                ev = (
                    "; ".join(_short(p.path) + f":{p.line}" for p in rep.plumbing[:3])
                    or "no reads outside config.py"
                )
        elif rep.init_only:
            ev = "init-only: " + "; ".join(_short(r.path) + f":{r.line}" for r in rep.reads[:3])
        elif rep.reads:
            ev = _short(rep.reads[0].path) + f":{rep.reads[0].line}"
            if rep.orphan_note:
                ev += " (orphan: " + rep.orphan_note + ")"
        if rep.aliased_to:
            ev += ("; " if ev else "") + "alias->" + ",".join(rep.aliased_to)
        if rep.bridged_to:
            ev += ("; " if ev else "") + "bridge->" + ",".join(rep.bridged_to)
        if rep.guarded:
            ev = (ev + "; " if ev else "") + "guarded"
        lines.append(
            f"| `{name}` | {rep.spec.cls} | `{rep.spec.annotation}` | `{rep.spec.default}` | "
            f"{_short(rep.spec.path)}:{rep.spec.line} | **{rep.verdict}** | {ev} | "
            f"{', '.join(rep.presets)} |"
        )
    lines += ["", "## PRESET-USES-DEAD", ""]
    lines += [f"- `{n}`" for n in dead_preset] or ["none"]
    if runtime is not None:
        lines += ["", "## Runtime pass", ""]
        lines.append(
            f"- {len(runtime)} fields read at {sum(len(v) for v in runtime.values())} sites"
        )
        read_live = [
            n for n in runtime if reports.get(n) and reports[n].verdict in ("DEAD", "LOG-ONLY")
        ]
        lines += [f"- RUNTIME-READS-{reports[n].verdict}: `{n}`" for n in read_live]
        lines += ["", "## PRESET-SETS-UNREAD", ""]
        lines += [f"- `{n}`" for n in (runtime_unread or [])] or ["none"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help=f"config module(s) with @dataclass fields (default: {', '.join(DEFAULT_CONFIGS)})",
    )
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="source root(s) to scan (default: src/dormouse)",
    )
    ap.add_argument("--preset-glob", default=DEFAULT_PRESET_GLOB)
    ap.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=None,
        help=f"config dataclass(es) to audit (default: {', '.join(DEFAULT_CLASSES)})",
    )
    ap.add_argument("--runtime", nargs="?", const="mirror_control", default=None)
    ap.add_argument("--runtime-steps", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    config_paths = [
        Path(p) if Path(p).is_absolute() else PORT / p for p in (args.config or DEFAULT_CONFIGS)
    ]
    roots = [Path(p) if Path(p).is_absolute() else PORT / p for p in (args.root or DEFAULT_ROOTS)]
    reports, findings, scanner = static_audit(
        config_paths, roots, args.preset_glob, set(args.classes or DEFAULT_CLASSES)
    )
    presets = load_presets(args.preset_glob, {n: r.spec for n, r in reports.items()})

    dead_preset = print_report(reports, presets, verbose=args.verbose)

    runtime: dict[str, list[str]] = {}
    runtime_unread: list[str] = []
    if args.runtime:
        print(f"\n-- runtime: preset {args.runtime!r}, {args.runtime_steps} tiny CPU steps --")
        runtime = runtime_audit(args.runtime, config_paths, steps=args.runtime_steps)
        sites = sum(len(v) for v in runtime.values())
        print(f"runtime read {len(runtime)} config fields at {sites} sites")
        for name in sorted(runtime) if args.verbose else ():
            rep = reports.get(name)
            static_verdict = rep.verdict if rep else "UNKNOWN"
            print(f"  {name:26s} {len(runtime[name]):3d} reads  static={static_verdict}")
        missed = [
            n for n in runtime if reports.get(n) and reports[n].verdict in ("DEAD", "LOG-ONLY")
        ]
        for name in missed:
            print(f"RUNTIME-READS-{reports[name].verdict}: {name} at {runtime[name][0]}")
        key = next(
            (k for k in presets if Path(k).stem == Path(args.runtime).stem),
            None,
        )
        if key:
            runtime_unread = [
                n
                for n, value in presets[key].items()
                if n not in runtime and not _is_default(value, reports[n].spec)
            ]
            for name in runtime_unread:
                print(
                    f"PRESET-SETS-UNREAD: {name}={presets[key][name]!r} "
                    f"(runtime never read it on {key})"
                )
        if (
            "d_model" in runtime
            and reports.get("d_model")
            and reports["d_model"].verdict in ("LIVE", "SHADOWED")
        ):
            n = len(runtime["d_model"])
            print(f"AGREE: d_model static={reports['d_model'].verdict} runtime={n} reads")
        else:
            print("DISAGREE: d_model is not live on both passes")

    if args.json:
        payload = {
            "fields": {
                name: {
                    "class": rep.spec.cls,
                    "decl": f"{rep.spec.path}:{rep.spec.line}",
                    "default": rep.spec.default,
                    "annotation": rep.spec.annotation,
                    "verdict": rep.verdict,
                    "reads": [
                        f"{r.where} {r.func}" + (" (log)" if r.log_only else "") for r in rep.reads
                    ],
                    "config_reads": [f"{p.where} {p.func}" for p in rep.plumbing],
                    "shadow_writes": [f"{w.where} {w.slot}" for w in rep.shadow_writes],
                    "init_only": rep.init_only,
                    "guarded": rep.guarded,
                    "aliased_to": rep.aliased_to,
                    "bridged_to": rep.bridged_to,
                    "presets": rep.presets,
                    "preset_non_default": rep.preset_non_default,
                    "runtime": runtime.get(name, []),
                }
                for name, rep in sorted(reports.items())
            },
            "unknown_receivers": dict(
                sorted(scanner.unknown_receivers.items(), key=lambda kv: -kv[1])[:40]
            ),
            "preset_uses_dead": dead_preset,
            "runtime_unread": runtime_unread,
        }
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    if args.markdown:
        write_markdown(
            args.markdown, reports, presets, dead_preset, runtime or None, runtime_unread
        )
        print(f"wrote {args.markdown}")

    counts = {
        v: sum(1 for r in reports.values() if r.verdict == v)
        for v in ("LIVE", "LOG-ONLY", "DEAD", "SHADOWED")
    }
    print(f"\n{len(reports)} fields: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    if dead_preset:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
