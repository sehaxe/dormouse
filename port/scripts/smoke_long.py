"""Long-context training smoke: ms/step and peak GiB, segmented vs one-shot.

    uv run --extra gpu python scripts/smoke_long.py

What it measures, and why these are the numbers: `segment_len` is the enabler
for training on windows longer than one forward fits in VRAM, and the two
questions it has to answer are (1) does a segmented training step cost more than
a one-shot one (the checkpoint recomputes each segment's activations in the
backward — that is the trade, so the measurement is fwd+bwd and not the forward
alone), and (2) how much VRAM does it save at the shapes where the one-shot
forward does not fit at all.

Shape: `small` as shipped (the dense blend; with `moe_topk > 0` segmentation needs
`moe_capacity = 0` — the dropless arm — see `config._validate_segment`), batch 2,
seq 2048 / 8192 / 16384 / 32768, the SAME window for both arms at each shape,
`segment_len = 4096` (16k in 4 segments; the 8k cut was the first measurement).
The one-shot arm is attempted where it fits and COUNTED as OOM where it does
not — never silently skipped.

The counters are printed per arm (`loop_segments`, `kda=…`) because the number a
reader needs is "which path produced this step time", not just the time.

PREFLIGHT, and it is the whole reason this file is safe to run: one GPU process
at a time (AGENTS §1.5) — no `dormouse` process alive (the `pgrep` pattern uses a
bracket so it cannot match this script's own command line, §2.4.6) and the card
holding < 2 GiB. On either, the script prints a SKIP banner naming what it found
and exits 0: a smoke that cannot measure is not a failure of the code under test,
and a partial silent run would be read as one. **If it skips, the numbers are not
taken — say so, do not quote a table from a busy card.**
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.config import load_run
from dormouse.model import DormouseModel
from dormouse.train import kda_field, open_streams

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

#: The port's data on this box (AGENTS §2.6) — the LOCAL copy (the /mnt drive
#: may not be mounted; the 19.9 GB filtered corpus was copied here).
DATA = "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"

DEV = "cuda"
BATCH = 2
SEQ_LENS = (8192, 16384, 32768)
SEGMENT = 4096
WARM, MEASURE = 2, 3


def arm(seq_len: int, segment_len: int) -> dict:
    """One arm at one shape: a real resolved run, then `WARM + MEASURE` steps.

    The run goes through `load_run`, so the config refusals (`segment_len` vs
    `seq_len`, the arms it cannot ride) are the ones a real launch would hit.
    """
    run = load_run(
        "small",
        model_overrides={"max_seq_len": max(SEQ_LENS)},
        train_overrides={"batch": BATCH, "seq_len": seq_len, "segment_len": segment_len},
    )
    # A preset carries no data path (the CLI always passes --data); the smoke
    # names the corpus itself. Without this `open_streams` refuses loudly.
    run.train.data = DATA
    model = DormouseModel(run.model, device=DEV)
    stream, _ = open_streams(run)
    x, y = stream.next_batch()
    x, y = x.to(DEV), y.to(DEV)
    name = "segmented" if segment_len else "one-shot"
    try:
        times: list[float] = []
        counters = None
        for step in range(WARM + MEASURE):
            model.zero_grad(set_to_none=True)
            if step == WARM:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _, loss, counters = model(x, y, segment_len=segment_len)
            assert loss is not None
            loss.backward()
            torch.cuda.synchronize()  # the step's own wall clock, not the enqueue
            times.append(time.perf_counter() - t0)
        ms = statistics.median(times[WARM:]) * 1e3
        peak = torch.cuda.max_memory_allocated() / 2**30
        out = {
            "arm": name,
            "seq": seq_len,
            "segment": segment_len,
            "ms": ms,
            "peak": peak,
            "loss": float(loss.detach()),
            "kda": kda_field(counters),
            "segments": counters.get("loop_segments", 0),
            "oom": False,
        }
    except RuntimeError as e:
        # OOM is COUNTED here, never a silent skip; anything else is a real
        # failure and propagates (a smoke that swallows a broken forward is the
        # defect this whole file's preflight exists to avoid).
        if "out of memory" not in str(e).lower():
            raise
        out = {
            "arm": name,
            "seq": seq_len,
            "segment": segment_len,
            "ms": float("nan"),
            # A lower bound: what the arm had already reached when it died.
            "peak": torch.cuda.max_memory_allocated() / 2**30,
            "loss": float("nan"),
            "kda": "",
            "segments": 0,
            "oom": True,
            "why": str(e).splitlines()[0],
        }
        torch.cuda.empty_cache()
    print(
        f"[{name} s{seq_len}] "
        + (
            f"OOM: {out['why']}"
            if out["oom"]
            else f"{out['ms']:.0f} ms/step warm  peak {out['peak']:.2f} GiB  "
            f"loss {out['loss']:.3f}  {out['kda']}  loop_segments={out['segments']}"
        ),
        flush=True,
    )
    del model, x, y
    torch.cuda.empty_cache()
    return out


def main() -> int:
    busy = subprocess.run(
        ["pgrep", "-af", "dormouse sf[t]"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = preflight("cuda") or (f"a dormouse process is alive:\n{busy}" if busy else None)
    if why is not None:
        return skip(why)
    arms: list[dict] = []
    for seq_len in SEQ_LENS:
        arms.append(arm(seq_len, SEGMENT))
        arms.append(arm(seq_len, 0))
    print("\n| seq | arm | segment_len | ms/step | peak GiB | loss | kda | segments |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        ms = "OOM" if a["oom"] else f"{a['ms']:.0f}"
        peak = f"{a['peak']:.2f}"
        loss = "—" if a["oom"] else f"{a['loss']:.3f}"
        print(
            f"| {a['seq']} | {a['arm']} | {a['segment']} | {ms} | {peak} | {loss} | "
            f"{a['kda']} | {a['segments']} |"
        )
    print(
        "\nA segmented step pays the backward's recompute of every segment; what it "
        "buys is the one-shot row that reads OOM.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
