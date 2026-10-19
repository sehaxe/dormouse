"""The code jail: an AST allowlist, then a runtime allowlist in a child process.

The screen is a WHITELIST, not a blacklist. A `Name` load must be a standard
name an algorithmic solution needs ([`_ALLOWED_BUILTINS`]) or a name the
submission binds itself; a call must be a bare name from that set or an
attribute without a leading underscore; an import must be [`_ALLOWED_MODULES`]
(`math` only); every attribute whose name starts with `_` is refused. That is
what closes the routes a regex could not see: `os.open`, `pathlib`, `io`,
`getattr(__builtins__, ...)`, `().__class__.__bases__`, `eval`/`exec`/`compile`
and `open` are all refused by NAME, with a reason that names the offending
token — never by a pattern that a string builder can walk around.

The child process is the SECOND line: the forbidden modules are removed from
`sys.modules` and refused by an import hook, `open`/`input` deny all I/O, and
the submission runs with an allowlist of builtins (`__import__` is a guarded
math-only import; `getattr`, `eval`, `open` do not exist in its namespace). The
harness that prints the test results runs with the real builtins but its `_B`
and `_results` names are assigned AFTER the submission's globals, so a
submission cannot shadow either.

`run_code` returns `None` when every test passed, otherwise the refusal/failure
reason (the caller wraps it in a `Verdict`). The reason never carries the raw
response.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

#: The exec jail's wall clock and CPU limit, in seconds (2602.11210's shape).
TIMEOUT_S = 5.0

#: The screen's size cap: a submission larger than this is refused before it is
#: parsed (a parser is CPU too, and the task budget is 256 B).
MAX_CODE_CHARS = 16_384

__all__ = ["MAX_CODE_CHARS", "TIMEOUT_S", "run_code"]


#: Names the screen refuses even when the program binds them itself. The
#: prompt's list is the floor; the rest are the same class (process, file,
#: native, serialization) and an algorithmic solution needs none of them.
_FORBIDDEN_NAMES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "pathlib",
        "path",
        "io",
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "importlib",
        "sys",
        "builtins",
        "pickle",
        "marshal",
        "inspect",
        "gc",
        "signal",
        "resource",
        "pty",
        "fcntl",
        "mmap",
        "tempfile",
        "sqlite3",
        "http",
        "urllib",
        "requests",
        "httpx",
        "ftplib",
        "smtplib",
        "asyncio",
        "multiprocessing",
        "threading",
        "socketserver",
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "dir",
        "globals",
        "locals",
        "memoryview",
        "help",
        "exit",
        "quit",
    }
)

#: The standard names an algorithmic solution needs. A call to a bare name NOT
#: in this set and NOT defined by the submission is refused — the allowlist is
#: the screen, [`_FORBIDDEN_NAMES`] only sharpens the reason.
_ALLOWED_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "Exception",
        "IndexError",
        "KeyError",
        "NotImplementedError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
)

#: The one importable module. It has no file/process surface; its dunder
#: attributes are refused by the attribute rule.
_ALLOWED_MODULES = frozenset({"math"})


class _BoundNames(ast.NodeVisitor):
    """Every name the submission defines (assignments, defs, params, targets).

    A Name load is allowed when it is a standard allowlisted name or one of
    these; everything else — `os`, `pathlib`, `__builtins__` — has no binding
    to resolve to.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._target(elt)
        elif isinstance(node, ast.Starred):
            self._target(node.value)

    def _args(self, args: ast.arguments) -> None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self.names.add(arg.arg)
        if args.vararg is not None:
            self.names.add(args.vararg.arg)
        if args.kwarg is not None:
            self.names.add(args.kwarg.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)
        self._args(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._args(node.args)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._target(node.target)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._target(node.target)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.names.update(node.names)


class _JailCheck(ast.NodeVisitor):
    """The static allowlist. The FIRST refusal is the one reported."""

    def __init__(self, bound: set[str]) -> None:
        self.bound = bound
        self.violation: str | None = None

    def _fail(self, token: str, why: str) -> None:
        if self.violation is None:
            self.violation = f"refused: {token!r} {why}"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in _ALLOWED_MODULES:
                self._fail(alias.name, "is not an importable module in the sandbox")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module.split(".")[0] not in _ALLOWED_MODULES:
            self._fail(module or "relative import", "is not an importable module in the sandbox")
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                self._fail(alias.name, "is not an allowed math name")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id.startswith("_"):
                self._fail(node.id, "is not an allowed name")
            elif node.id in _FORBIDDEN_NAMES:
                self._fail(node.id, "is not allowed in the sandbox")
            elif node.id not in _ALLOWED_BUILTINS and node.id not in self.bound:
                self._fail(node.id, "is not in the sandbox allowlist")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Children first, so `().__class__.__bases__` names `__class__` (the
        # inner escape) rather than the outer attribute on it.
        self.generic_visit(node)
        if node.attr.startswith("_"):
            self._fail(node.attr, "is not an allowed attribute")

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Name):
            if func.id.startswith("_") or func.id in _FORBIDDEN_NAMES:
                self._fail(func.id, "is not allowed in the sandbox")
            elif func.id not in _ALLOWED_BUILTINS and func.id not in self.bound:
                self._fail(func.id, "is not in the sandbox allowlist")
        elif isinstance(func, ast.Attribute):
            if func.attr.startswith("_"):
                self._fail(func.attr, "is not an allowed attribute")
        else:
            self._fail(type(func).__name__, "is not an allowed call target")


def screen_code(code: str) -> str | None:
    """The AST allowlist: `None` when the submission may run, else a reason."""
    if len(code) > MAX_CODE_CHARS:
        return (
            f"refused: the submission is {len(code)} characters, over the "
            f"{MAX_CODE_CHARS}-character limit"
        )
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"refused: the submission is not valid Python ({exc.msg})"
    bound = _BoundNames()
    bound.visit(tree)
    check = _JailCheck(bound.names)
    check.visit(tree)
    return check.violation


#: Child-side jail: the second line behind the static screen. An import hook and
#: `sys.modules` deletion refuse the modules the screen names even if reached
#: dynamically; `open`/`input` refuse every file/terminal read; the submission
#: runs with an allowlist of builtins, so `__import__` (guarded, math-only),
#: `getattr`, `eval` and `open` do not exist in its namespace at all. The
#: harness that prints the test results is executed with the real builtins but
#: its own `_B`/`_results` names are assigned AFTER the user globals, so a
#: submission cannot shadow either.
_SANDBOX_PREAMBLE = """\
import builtins as _builtins
import sys as _sys

_BANNED = %r
_ALLOWED_MODULES = %r
for _m in list(_sys.modules):
    if _m.split(".")[0] in _BANNED:
        del _sys.modules[_m]


class _RefuseImport:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BANNED:
            raise ImportError("sandbox: import of " + repr(name) + " is refused")
        return None


_sys.meta_path.insert(0, _RefuseImport())
_real_import = _builtins.__import__


def _guarded_import(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError("sandbox: import of " + repr(name) + " is refused")
    return _real_import(name, *a, **k)


def _deny_file_io(*_a, **_k):
    raise PermissionError("sandbox: file I/O is refused by the verifier jail")


_builtins.open = _deny_file_io
_builtins.input = _deny_file_io

_ALLOWED = %r
_SAFE = {_n: getattr(_builtins, _n) for _n in _ALLOWED if hasattr(_builtins, _n)}
_SAFE["__build_class__"] = _builtins.__build_class__
_SAFE["__import__"] = _guarded_import
_globals = {"__name__": "__main__", "__builtins__": _SAFE}
exec(compile(%r, "<submission>", "exec"), _globals)
_results = []
_harness_globals = dict(_globals)
_harness_globals["__builtins__"] = _builtins
_harness_globals["_B"] = _builtins
_harness_globals["_results"] = _results
exec(compile(%r, "<tests>", "exec"), _harness_globals)
"""


class _Timeout(Exception):
    """The jailed program exceeded [`TIMEOUT_S`]."""


class _SandboxError(Exception):
    """The jailed program exited non-zero; carries the code, never the stderr."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _sandbox_limits() -> None:  # runs in the child, before exec
    os.setsid()
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT_S), int(TIMEOUT_S)))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _extract_code(response: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return fence.group(1) if fence else response


def _harness(tests) -> str:
    lines = []
    for call, _ in tests:
        lines.append(f"_results.append(_B.repr({call}))")
    lines.append("_B.print('\\n'.join(_results))")
    return "\n".join(lines) + "\n"


def _run_sandbox(program: str, *, timeout: float) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", "-"],
                input=program,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "HOME": workdir},
                preexec_fn=_sandbox_limits,
            )
        except subprocess.TimeoutExpired as exc:
            raise _Timeout from exc
    if proc.returncode != 0:
        raise _SandboxError(proc.returncode)
    return proc.stdout


def run_code(response: str, tests, match: str = "exact") -> str | None:
    """Run one submission against `tests`; `None` when all passed, else a reason."""
    code = _extract_code(response)
    bad = screen_code(code)
    if bad is not None:
        return bad
    program = _SANDBOX_PREAMBLE % (
        tuple(sorted(_FORBIDDEN_NAMES)),
        tuple(sorted(_ALLOWED_MODULES)),
        tuple(sorted(_ALLOWED_BUILTINS)),
        code,
        _harness(tests),
    )
    try:
        out = _run_sandbox(program, timeout=TIMEOUT_S)
    except _Timeout:
        return f"timed out after {TIMEOUT_S:g}s in the sandbox"
    except _SandboxError as exc:
        return f"program exited with code {exc.code} under the sandbox"
    lines = out.strip().splitlines() if out.strip() else []
    if len(lines) != len(tests):
        return f"expected {len(tests)} result lines, got {len(lines)}"
    for (call, expected), got in zip(tests, lines, strict=True):
        if not _answer_matches(str(expected), got, match):
            return f"test {call!r} failed (expected {expected!r})"
    return None


def _answer_matches(expected: str, got: str, match: str) -> bool:
    if match == "regex":
        return re.search(expected, got) is not None
    return got.strip() == expected.strip()
