"""The one lane guard every measurement script shares.

WHY THIS FILE EXISTS
--------------------
Thirty-three scripts each carried their own copy of `preflight()` and ten a copy
of `skip()` — 628 lines of identical logic whose failure modes had to be fixed
thirty-three times. On 2026-10-20 the fix was: no `torch.cuda` call in the
parent, because `torch.cuda.is_available()` initialises a CUDA context and a
`subprocess` child then inherits a context it cannot use — every child died with
`CUDA-capable device(s) is/are busy or unavailable` while a manual child ran
fine. Two copies had the fix; thirty-one did not.

ONE GPU PROCESS AT A TIME (AGENTS 1.5). `preflight` returns the reason to skip
or `None`; `skip` prints it loudly and exits 0, because nothing was measured and
the code is not at fault.

DATA ON THIS BOX (AGENTS 2.6). `CORPUS` and `EVAL` are the two files every
pretraining measurement scores on. `data_missing` names the missing one instead
of letting a run die three layers down in the byte loader.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The filtered corpus and the mixed eval tail (the same pair `configs/*.toml`
# name). Kept here so a script that forgets them fails at the guard, not inside
# the stream after a minute of model construction.
CORPUS = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"
EVAL = "/home/sehaxe/dormouse/data_local/eval_mix/eval_mix.bin"

# A card holding more than this already has an owner: the desktop alone sits at
# ~0.7 GiB, a model at 5-13 GiB. The ceiling is deliberately low because the
# cost of a false SKIP is one rerun and the cost of a false GO is a corrupted
# measurement (two GPU processes collided once and wrote garbage into a
# checkpoint, AGENTS 1.5).
CARD_CEILING_GIB = 2.0


def _compute_app_gib(line: str) -> float:
    """MiB held by one `--query-compute-apps` row, 0.0 when unparsable.

    The row is `pid, name, used_memory`; a process name may contain a comma
    (`/usr/bin/python3, ...` does not, but be strict) so parse from the right.
    """
    tail = line.rsplit(",", 1)[-1].strip()
    try:
        return float(tail) / 1024.0
    except ValueError:
        return 0.0


def preflight(device: str, ceiling_gib: float = CARD_CEILING_GIB) -> str | None:
    """`None` when the card is ours, else the reason to skip.

    NO torch CUDA call here, on purpose: `torch.cuda.is_available()` and
    `mem_get_info()` initialise a CUDA context in the PARENT, and `subprocess`
    then forks a child that inherits a context it cannot use. The card is read
    from `nvidia-smi` instead (measured 2026-10-20).

    `ceiling_gib` is a parameter because twelve benchmark scripts need a tighter
    budget than a trainer (they measure milliseconds, so a card holding 1.5 GiB
    of desktop already distorts them). One default, one override, no copies.
    """
    if device == "cpu":
        print("[preflight] device=cpu: the card checks do not apply", flush=True)
        return None
    # No `pgrep`: it matches any shell whose COMMAND LINE contains the pattern,
    # and a driver launched from a heredoc carries that text (measured twice,
    # 2026-10-22). The only authoritative answer is the driver's own list of
    # compute processes: a pid holding VRAM is a training job, a `bash -c` is not.
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    busy = [line for line in apps.splitlines() if line.strip() and _compute_app_gib(line) >= 0.5]
    if busy:
        return "a compute process holds the card:\n" + "\n".join(busy)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    first = smi.stdout.strip().splitlines()[0] if smi.stdout.strip() else ""
    if not first:
        return "device=cuda but nvidia-smi reports no device"
    try:
        used_mib = float(first.split(",")[0])
    except ValueError:
        return f"nvidia-smi's memory row is unreadable: {first!r}"
    used = used_mib / 1024
    if used >= ceiling_gib:
        return f"the card already holds {used:.2f} GiB (the preflight's ceiling is {ceiling_gib})"
    print(f"[preflight] no dormouse process; card holds {used:.2f} GiB", flush=True)
    return None


def data_missing(*paths: str) -> str | None:
    """`None` when every path exists, else the reason to skip (names the path)."""
    missing = [p for p in paths or (CORPUS, EVAL) if not Path(p).exists()]
    if not missing:
        return None
    return "missing data file(s):\n" + "\n".join(f"  {p}" for p in missing)


def skip(why: str) -> int:
    """A loud SKIP: the reason, and no table. Exit 0 — the code is not at fault."""
    print(
        f"\n=== SKIP: {why}\n"
        f"=== ONE GPU PROCESS AT A TIME (AGENTS 1.5): nothing was measured, so there\n"
        f"=== is no number to quote. Re-run when the card is free.\n",
        flush=True,
    )
    return 0


def guard(device: str, *data: str) -> int | None:
    """The one call a `main` makes.

    Returns 0 when the run must not proceed (already printed), else `None`:

        rc = guard(args.device)
        if rc is not None:
            return rc
        ... run ...

    NOT `rc = guard(...) or run()`: `guard` returns 0 on a skip, and `0 or run()`
    would run the measurement anyway. The same trap this module exists to prevent.
    """
    why = preflight(device) or data_missing(*data)
    if why:
        return skip(why)
    return None


if __name__ == "__main__":  # the smallest thing that fails if this breaks
    assert preflight("cpu") is None, "cpu must never consult the card"
    assert data_missing(CORPUS) is None, f"the corpus is missing: {CORPUS}"
    assert "missing data file" in (data_missing("/nonexistent.bin") or ""), (
        "a missing path must be named"
    )
    assert guard("cpu") is None, "the guard must pass on a box that has the data"
    print("lane guard: ok")
    sys.exit(0)
