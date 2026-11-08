# Your first hour

Sixty minutes, CPU only, no GPU, no corpus, no checkpoint. At the end you will
have run the suite's gate, watched a real test go red on a one-line change and
go green again, and made one real documentation change that removes a claim
nothing can check.

Every command below was run against the live Python lane on 2026-11-08, except
the fresh-clone steps and the full-suite run — the one command here that needs a
quiet machine, and it was not re-run while other work held this box. The
commands are the source of truth; the outputs quoted here are what they printed
then, and a number that has moved is a reason to run the command, not a reason
to trust this page. If a command here disagrees with the repository, the
repository wins — open an issue against this page.

## 0:00–0:15 — Read three things

| file | what you are looking for |
|---|---|
| [`README.md`](../../README.md) | the top, **Results**, and **What is measured, not assumed**: what the project claims, and how each claim is marked |
| [`CONTRIBUTING.md`](../../CONTRIBUTING.md) | the gates you will run at 0:55, and what this project refuses |
| [`AGENTS.md` §1](../../AGENTS.md) | the rules in full — LOUD/COUNTED/SILENT, A/B or death, claims name their evidence, one heavy thing at a time. About a hundred lines; read all of §1 |

You are not memorising them. You are learning the two shapes everything here
takes: **a claim carries its evidence**, and **a gate that cannot fail is a
defect**.

The Python lane (`port/`) is the live one. `crates/` is the Rust/burn lane,
frozen since 2026-11-07 — read it if you like, do not send it new work.

## 0:15–0:25 — Set up (CPU only)

```bash
git clone https://github.com/sehaxe/dormouse.git
cd dormouse/port
uv sync                       # CPU-only; the card extra is `uv sync --extra gpu`
.venv/bin/python -c "import dormouse; print(dormouse.__file__)"
```

The import must print a path under `dormouse/port/src/`. The first `uv sync`
downloads `torch`; it is the slowest step of the hour and needs no card.

Which branch: the live lane is `backup/main-local-2026-11-08`; `origin/main` is
a different, Rust-era lineage until the owner merges them. Open pull requests
against the live lane. [`README.md`](../../README.md) says which is which.

## 0:25–0:35 — Smoke: a number pinned in two places

```bash
cd port
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_param_count.py
```

Expect `6 passed`. It is slower than the other tests in this hour because the
`b1` preset is instantiated on the `meta` device — 1.07 billion parameters,
every module built, no storage behind them.

Read the test's docstring while it runs. It does something this repository does
everywhere: it prices the model twice, by hand (`arithmetic`) and off the
instantiated module tree (`built`), and asserts the two agree with a third,
independent constant (`SMALL_PORT_TOTAL = 9_476_993`). A preset that drifts from
the pinned total fails here, which is what you are about to prove.

## 0:35–0:45 — Break it on purpose, then fix it

Open `port/configs/small.toml`, find `max_iter = 4` (line 50), and change it to
`max_iter = 5`. Run the same test.

Expected failure (abridged from a real run; `...` is the elision):

```
____________ test_small_arithmetic_matches_the_model_and_the_preset ____________

    def test_small_arithmetic_matches_the_model_and_the_preset():
        ...
>       assert sum(want.values()) == SMALL_PORT_TOTAL == param_count(model)
E       AssertionError: assert 9477761 == 9476993
E        +  where 9477761 = sum(dict_values([196608, 65600, 1536, 4608, ...]))
```

The two counts now differ by 768, which is `d_model`: one more loop iteration
adds one `iter_embed` row of 768 weights. The test is not telling you the number
is wrong; it is telling you the preset's shape and the published total are now
two different claims. That is the whole point of the pin.

Restore the file and watch it go green:

```bash
git status --short                  # exactly one file, M configs/small.toml
git restore configs/small.toml
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_param_count.py
```

`6 passed`. You have now seen the gate red on the change and green after it —
the only order in which a test means anything here (`AGENTS.md` §1: "every fix
lands with a gate that fails on the un-fixed code").

## 0:45–0:55 — Your first real change: a number with no artifact

The Layout block in `port/README.md` says:

```
tests/             167 files, 2,194 collected tests, CPU-true by default, CUDA gates opt-in
```

Check it:

```bash
cd /path/to/dormouse                # repository root
git ls-files 'port/tests/*.py' | wc -l
```

On this lane at `eb285e3` (2026-11-08) that printed **169**, not 167. The count
is already wrong, and it has rotted before: it said `117` until `26f0e96`
corrected it to `167 files, 2,194 collected tests`, which has now rotted again.
Nothing in the repository checks it, so nothing will stop it rotting a third
time — and the collected-test number moves every time anyone adds a test.

The fix is not a new number. It is deleting the number and keeping what the line
is for:

```diff
-tests/             167 files, 2,194 collected tests, CPU-true by default, CUDA gates opt-in
+tests/             pytest, CPU-true by default, CUDA gates opt-in
```

Then:

```bash
python3 tools/check_doc_refs.py     # still green; you added no path
git diff -- port/README.md
```

This is a real first PR: one line, no code, and it removes a claim that cannot
name its artifact. Commit it in the repository's style:

```
docs(readme): delete a test count nothing checks
```

In the pull request body, give the artifact for the claim that the number was
wrong: the command above and its output, and the date. That is what makes it a
correction rather than a preference.

## 0:55–1:00 — The gate and the pull request

From the repository root:

```bash
bash scripts/gate_fast.sh
```

It ends in `GATE PASSED`. The steps are: `ruff check`; format on the files
changed since `origin/main`; every path a document names exists; no preset sets
a config field nothing reads; the claims ledger (report-only); `cargo check`
(prints `SKIP` when Rust is not installed). If it fails, the failing step is the
task — unless the file it names is one you did not touch, in which case report
it with `file:line` and leave it alone.

The behaviour gate is separate, and it is the same suite CI runs:

```bash
cd port && CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q
```

Then `git status --short` must show exactly the one file you meant to change.
Fill [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md);
its checklist is the gate list you have just run, plus the one thing this hour
taught: **a number in the body names its artifact**.

## What you should take away

1. A test is only a gate if you have seen it fail. Break the thing it pins once;
   it costs two minutes and it is the difference between a test and decoration.
2. A number without a config, a commit, a date and a log path (or a test) is a
   liability, not documentation — and that applies to a README sentence.
3. `gate_fast` checks static claims; `pytest` checks behaviour. A pull request
   needs both, and neither is optional.
4. The rest is in [`docs/guides/first-pr.md`](first-pr.md) (where each rule
   lives) and [`docs/guides/ab-testing.md`](ab-testing.md) (how a mechanism is
   judged). If you want to propose a mechanism, the issue template asks for its
   removal arm before it asks for its design.
