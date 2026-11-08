# Release: the dormouse Python package

The artifact is a pure-Python wheel built from [`port/`](../../port) (no
compiled extensions; `torch` is a dependency, FLA an optional `gpu` extra).
The Rust crates are not part of this release.

The version has one home: `version` in [`port/pyproject.toml`](../../port/pyproject.toml).
The installed metadata carries it and `dormouse.__version__` reads it back, so a
tag, a wheel and `dormouse --version` cannot disagree. The history is in
[`port/CHANGELOG.md`](../../port/CHANGELOG.md).

## What must be true before a release

Every box needs evidence; "looks fine" is not a gate.

- [ ] `version` bumped in `port/pyproject.toml`, and `port/CHANGELOG.md` has a
      dated section for it with no unreleased work left in it.
- [ ] Tree clean on the release commit (`git status --short` empty), lock
      committed, `uv sync --frozen` succeeds.
- [ ] Full CPU suite green **on that commit** (CUDA tests skip; a release does
      not claim GPU verification).
- [ ] Lint and format clean over the paths CI checks.
- [ ] Doc-reference and dead-knob gates green.
- [ ] `uv build` produces both artifacts; the wheel contains
      `dormouse/configs/*.toml`, the dist-info version matches `pyproject.toml`,
      and the `dormouse` console script is declared.
- [ ] The wheel runs from outside a checkout: `--version` and
      `dormouse info small` succeed in a throwaway environment.
- [ ] The annotated tag `vX.Y.Z` is created on the release commit.

## Build it

```bash
cd port
uv sync --frozen
uv build                     # dist/dormouse-X.Y.Z.tar.gz + .whl
```

`uv build` builds the sdist first and the wheel from it, so a preset that is
missing from the sdist fails the build rather than shipping a broken wheel.
The presets are force-included by
`[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`; the loader
looks for that copy last (`dormouse.config.PACKAGE_PRESET_DIR`).

Inspect before trusting:

```bash
cd port
python - <<'PY'
import glob, tomllib, zipfile

wheel = glob.glob("dist/dormouse-*.whl")[0]
names = zipfile.ZipFile(wheel).namelist()
configs = [n for n in names if n.startswith("dormouse/configs/") and n.endswith(".toml")]
version = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
metadata = zipfile.ZipFile(wheel).read(f"dormouse-{version}.dist-info/METADATA").decode()

assert configs, "no presets in the wheel"
assert f"Version: {version}" in metadata, metadata[:200]
assert not [n for n in names if n.endswith(".pyc") or "__pycache__" in n]
print(f"{len(configs)} presets, version {version}, {len(names)} files")
PY
```

## Gates (the same commands CI runs)

```bash
cd port
PYTHONPATH=$PWD/src uv run python -m pytest -q -p no:logging
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests

cd ..
python3 tools/check_doc_refs.py
cd port
PYTHONPATH=src uv run python scripts/dead_knob_audit.py \
    --preset-glob 'configs/**/*.toml' > /tmp/dead_knobs.txt 2>&1
test "$?" -ne 2                    # exit 2 = the audit crashed, 1 = findings
```

The release gate on version, `--version` and both preset layouts is
`tests/test_release.py`; it builds nothing, so run it first when iterating:

```bash
cd port && uv run pytest tests/test_release.py -q
```

## Install the wheel outside the checkout

This is the check the source tree cannot give you: no cwd `configs/`, no
`port/src` on the path.

```bash
cd port
wheel=$(ls -t dist/dormouse-*.whl | head -1)
uv venv /tmp/dormouse-release
uv pip install --python /tmp/dormouse-release/bin/python "$wheel"
cd /tmp
/tmp/dormouse-release/bin/dormouse --version
/tmp/dormouse-release/bin/dormouse info small | head -5
```

For an offline install (no dependency download), add `--no-deps`; that proves
the file installs but cannot import, because `torch` is a runtime dependency.
To exercise a wheel without a network at all, unpack it and put it in front of
an environment that already has `torch`:

```bash
cd port
wheel=$(ls -t dist/dormouse-*.whl | head -1)
python3 -m zipfile -e "$wheel" /tmp/dormouse-wheel
cd /tmp && PYTHONPATH=/tmp/dormouse-wheel python3 -m dormouse.cli --version
```

## Tag it

```bash
VERSION=$(python3 -c 'import tomllib;print(tomllib.load(open("port/pyproject.toml","rb"))["project"]["version"])')
git tag -a "v$VERSION" -m "dormouse v$VERSION"
```

Pushing the tag is an outward action: a maintainer does it after the checklist.
Nothing in the repo publishes to PyPI yet (see below).

## What this process still does not cover

- **No PyPI publish.** `uv publish` needs credentials/trusted publishing; no
  workflow owns it yet. The tag is the release today.
- **No GitHub Release job.** The changelog is the notes; nothing attaches the
  wheel to a release page.
- **Project metadata is thin.** `pyproject.toml` declares no `license`,
  `authors`, `readme`, `classifiers` or `urls`, so the wheel's metadata is
  minimal.
- **CPU gates only on the release machine.** A green release says the CPU path
  is good; the CUDA/FLA arms were not run.
- **Python is pinned to 3.12 only** (`>=3.12,<3.13`); that is a support
  statement, not an accident.
