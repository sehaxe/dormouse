#!/usr/bin/env -S uv run --python /home/sehaxe/dormouse/port/.venv/bin/python --script
"""The MoE package's price: 299M-total / 100M-active bank vs the 95M dense control.

THE QUESTION (plan T4). A routed bank costs TOTAL parameters at checkpoint size
but only ACTIVE parameters per token. This measures whether that package pays in
STEP TIME and VRAM — i.e. whether it forces the batch down — at the live recipe's
shape, or whether a 3x parameter bank is free at the same batch.

ARMS (each `--steps` optimizer steps, median + min, same seed, the live recipe):

    control      `configs/flagship3_dense.toml` as it stands: block_form
                 "transformer", 12 unique blocks, GELU FFN at d_ffn=3072 — 95.0M
    control-kda  the SAME 12-layer stack on block_form="kda" with
                 `ffn_form="dense"` at d_ffn=2880 (~95M): the bank's own block,
                 dense — this is the arm that separates "the bank costs" from
                 "the KDA block costs"
    bank         block_form="kda", `ffn_form="bank"`, n_experts=8, topk=2,
                 rank=832 (=d_model), d_ffn=1664 — 299.2M total, 99.7M active
                 per token. Dispatch: `moe_kernel="batched"` + `moe_capacity=0.0`
                 (DROPLESS: the cap is this batch's largest group, not
                 `capacity x the mean`), and `--moe-capacities 1.0,1.25,2.0`
                 sweeps the capacity rule so the waste is a number per row.

WHY `rank` IS THE SIZING KNOB, NOT `d_ffn`. The bank's expert is a rank-`rank`
dual-latent GLU plus ONE shared `down [f, d]`; the shared down and the shared
expert are always on, so total/active is NOT `E/topk`. At the preset's `rank=64`
the ratio is 1.34 over every `d_ffn` (a 299M-total bank would be 222M ACTIVE),
and no `d_ffn` reaches 299M/100M. At `rank=d_model=832` nine expert copies run
against three active ones and the ratio is 3.0 by construction, which is what
makes `299M total / 100M active` expressible at all. `--check` asserts the two
counts and `--rank-scan` prints the whole table without a GPU.

TWO LAYOUT FIXES THIS HARNESS NOW MEASURES (2026-11-10, after the first run's
b24/b16 OOM): the bank applies the router weight in RANK SPACE before `@ v`
instead of scaling a `[n·k, f]` fp32 product, and the padded bank's
`capacity x mean` rows are replaced by the dropless arm's `largest group` cap.
Both are layout changes with identical math, gated in
`tests/test_moe_batched.py` (1e-5 fp32, 1e-2 relative bf16).

PROTOCOL. Real corpus bytes (`data_local/best/fwedu_train.bin`, the flat
`ByteStream` the trainer reads, not a mix plan), seq 1024, grad_accum 1, the
recipe's `wsd_lr` warmup in TOKENS (`warmup_tokens = 19e6`, 1% of the 1.9 GB
budget) so every arm ramps identically, `opt=mix`, `muon_lr_mult=12`,
`ns_dtype=bf16`, `muon_state_dtype=bf16`, `torch_compile=true` with the trainer's
own wrapper (`dynamic=False, mode="max-autotune-no-cudagraphs"`), one
`invalidate_banks_all` per step (the trainer's seam). Times are per-step CUDA
events (no per-step host sync, so the CPU can still run ahead as it does in the
trainer); VRAM is `torch.cuda.max_memory_allocated` after the timed region.

GPU DISCIPLINE. The script RE-EXECS itself through `scripts/gpu_lock.sh` (the
repo's ONE gateway: flock on /tmp/dormouse-gpu.lock + an idle-card check +
`timeout`) unless already inside it. A busy card is exit 75 and a timeout 124 —
both mean nothing was measured. No guard is bypassed here.

    uv run --python port/.venv/bin/python port/scripts/bench_moe_package.py
    uv run --python port/.venv/bin/python port/scripts/bench_moe_package.py --check
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from dormouse.config import RunConfig, load_run
from dormouse.instrument import Counters
from dormouse.model.experts import SparseMoE, invalidate_banks_all
from dormouse.model.model import DormouseModel
from dormouse.model.qat_cache import invalidate_qat_cache_all
from dormouse.train.data import ByteStream, warmup_steps
from dormouse.train.lr import wsd_lr
from dormouse.train.optim import MixOptimizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data_local/best/fwedu_train.bin"
PRESET = "configs/flagship3_dense.toml"
SEQ = 1024
#: The live recipe (`scripts/train_budget.sh`): 1% of the 1.9 GB budget in tokens.
WARMUP_TOKENS = 19_000_000
STEPS = 20
WARM = 3
LOCK_ENV = "DORMOUSE_BENCH_LOCKED"
LOCK = ROOT / "scripts" / "gpu_lock.sh"
EXIT_BUSY, EXIT_TIMEOUT = 75, 124

#: The KDA 12-layer stack (`configs/arch_moe8.toml`'s depth: 12 bodies, 12 passes).
KDA = {
    "block_form": "kda",
    "loop_blocks": 12,
    "max_iter": 12,
    "n_layers": 0,
    "d_model": 832,
    "n_heads": 13,
    "head_dim": 64,
    "vocab": 256,
    "max_seq_len": SEQ,
    "bf16": True,
}
#: 299.15M total / 99.67M active, counted (`--check` pins both).
#: `moe_kernel="batched"` + `moe_capacity=0.0` is the DROPLESS bank: its cap is
#: THIS batch's largest group, not `capacity x the mean`, so no expert computes
#: rows the router did not assign. The 2026-11-10 OOM was measured with the
#: padded arm at `capacity=2.0` (2x the mean) on top of the old `[n·k, f]` fp32
#: router-weight product; both are gone (`SparseMoE._bank_dispatch`).
BANK = dict(
    KDA,
    ffn_form="bank",
    n_experts=8,
    moe_topk=2,
    moe_kernel="batched",
    moe_capacity=0.0,
    rank=832,
    d_ffn=1664,
    shared_rank=192,
)
#: ~95.1M total: the bank's block, FFN dense.
KDA_DENSE = dict(KDA, ffn_form="dense", n_experts=1, moe_topk=0, d_ffn=2880)

ARMS = {
    "control": {},
    "control-kda": KDA_DENSE,
    "bank": BANK,
}


def active_params(model: torch.nn.Module) -> tuple[int, int]:
    """`(total, active)` — active = total minus the routed experts a token never runs.

    Counted off the module tree, not inferred from a formula: every `SparseMoE`
    owns `len(experts)` copies and runs `topk` of them per token, so the rest of
    each copy is dead weight for this token.
    """
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for m in model.modules():
        if isinstance(m, SparseMoE) and len(m.experts) > 1 and int(m.topk) > 0:
            per = sum(p.numel() for p in m.experts[0].parameters())
            inactive += (len(m.experts) - int(m.topk)) * per
    return total, total - inactive


def the_run(model_overrides: dict, batch: int, steps: int) -> RunConfig:
    """The bench's RunConfig: the live recipe's train half plus the arm's model half."""
    train = dict(
        batch=batch,
        seq_len=SEQ,
        grad_accum=1,
        steps=steps,
        eval=False,
        log_every=10**9,
        ckpt_every=10**9,
        json_log="",
        data="",
        eval_data="",
        mix_plan="",
        lr=3e-4,
        opt="mix",
        muon_lr_mult=12,
        ns_dtype="bf16",
        muon_state_dtype="bf16",
        torch_compile=True,
    )
    run = load_run(PRESET, train_overrides=train, model_overrides=model_overrides)
    return dataclasses.replace(
        run, model=dataclasses.replace(run.model, max_seq_len=max(SEQ, run.model.max_seq_len))
    )


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _graphs() -> int:
    """`torch.compile`'s unique-graph count — a recompile per step shows up here."""
    stats = torch._dynamo.utils.counters.get("stats") or {}
    return int(stats.get("unique_graphs", 0))


def _optimizer(model: torch.nn.Module, run) -> MixOptimizer:
    c = run.train
    return MixOptimizer(
        model,
        lr=c.lr,
        wd=c.wd,
        opt=c.opt,
        ns_steps=c.ns_steps,
        ns_kind=c.ns_kind,
        colrow_3d=c.muon_colrow_3d,
        muon_state_dtype=c.muon_state_dtype,
        ns_dtype=c.ns_dtype,
        muon_variant=c.muon_variant,
        muon_sf=c.muon_sf,
        muon_sf_wd=c.muon_sf_wd,
        eps=c.adam_eps,
        muon_lr_mult=c.muon_lr_mult,
    )


def measure(name: str, overrides: dict, batch: int, *, device, steps: int, warm: int, seed: int):
    """One arm: build, warm, then `steps` timed steps. Returns a row (status `ok`/`OOM`)."""
    _free(device)
    torch._dynamo.reset()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    run = the_run(overrides, batch, warm + steps)
    model: DormouseModel | None = None
    row = {"arm": name, "batch": batch, "status": "ok"}
    try:
        model = DormouseModel(run.model, device=device).train()
        total, active = active_params(model)
        opt = _optimizer(model, run)
        graphs0 = _graphs()
        target: torch.nn.Module = model
        if run.train.torch_compile:
            target = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
        stream = ByteStream(DATA, batch, SEQ)
        warm_steps = warmup_steps(WARMUP_TOKENS, batch, SEQ)
        counter = Counters()
        for step in range(warm + steps):
            lr = wsd_lr(
                step,
                total_steps=warm + steps,
                peak_lr=run.train.lr,
                warmup_steps=warm_steps,
                decay_steps=0,
                final_frac=run.train.min_lr_frac,
            )
            opt.set_lr(lr)
            x, y = stream.next_batch()
            x, y = x.to(device), y.to(device)
            if step == warm:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _logits, loss, step_counters = target(x, targets=y)
            opt.zero_grad(set_to_none=True)
            assert loss is not None, "a training step must carry targets"
            loss.backward()
            opt.step()
            # The trainer's per-step seam (`train/loop.py`), after the update landed.
            invalidate_banks_all(target)
            invalidate_qat_cache_all(target)
            end.record()
            if step >= warm:
                row.setdefault("events", []).append((start, end))
            # OUTSIDE the event window: the trainer passes no `counters` (the
            # forward builds its own), and a dict that grows between calls makes
            # dynamo re-guard the graph. Merging after `end.record()` keeps the
            # measured step the trainer's step.
            counter.merge(step_counters)
        torch.cuda.synchronize(device)
        times = sorted(s.elapsed_time(e) for s, e in row.pop("events"))
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        row.update(
            total=total,
            active=active,
            ms_median=statistics.median(times),
            ms_min=times[0],
            peak_gib=peak,
            steady_gib=torch.cuda.memory_allocated(device) / 2**30,
            bytes_per_step=batch * SEQ,
            counters=dict(counter),
            graphs=_graphs() - graphs0,
        )
        stream.close()
        del opt, target
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except RuntimeError as e:
        # `torch.compile`'s own autotune allocation surfaces as an
        # `InductorError` wrapping the OOM, not as `torch.cuda.OutOfMemoryError`
        # (measured at b14). Only the OOM spelling is absorbed: any other
        # RuntimeError is a real failure and must stop the run.
        if "out of memory" not in str(e).lower():
            raise
        row["status"] = "OOM"
    del model
    _free(device)
    return row


def _table(rows: list[dict]) -> str:
    head = (
        "| arm | batch | ms/step (median) | ms/step (min) | peak GiB | steady GiB "
        "| B/step | params total | params active | graphs |"
    )
    out = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            out.append(
                f"| {r['arm']} | {r['batch']} | {r['status']} | — | — | — | "
                f"{r['batch'] * SEQ:,} | — | — | — |"
            )
            continue
        out.append(
            f"| {r['arm']} | {r['batch']} | {r['ms_median']:.1f} | {r['ms_min']:.1f} "
            f"| {r['peak_gib']:.2f} | {r['steady_gib']:.2f} | {r['bytes_per_step']:,} "
            f"| {r['total'] / 1e6:.2f}M | {r['active'] / 1e6:.2f}M | {r['graphs']} |"
        )
    return "\n".join(out)


def _check() -> int:
    """No GPU: the parameter counts this report quotes, asserted from the module tree."""
    for name, overrides, want_total, want_active in (
        ("control", {}, 95_035_200, None),
        ("control-kda", KDA_DENSE, 95_119_692, None),
        ("bank", BANK, 299_152_716, 99_672_396),
    ):
        run = the_run(overrides, 8, 1)
        model = DormouseModel(run.model, device="meta")
        total, active = active_params(model)
        print(f"{name:12s} total={total / 1e6:8.2f}M active={active / 1e6:8.2f}M")
        assert abs(total - want_total) / want_total < 0.01, f"{name}: total drifted"
        if want_active is not None:
            assert abs(active - want_active) / want_active < 0.02, f"{name}: active drifted"
            assert total > 2 * active, f"{name}: the package is not 3x"
    print("check: the bank is 3.00x the control's total at its active size")
    return 0


def _inherited_lock_fd() -> bool:
    """True when an ancestor `gpu_lock.sh` already opened the lock in our fd table.

    `gpu_lock.sh` does `exec 9>"$LOCK"` and `flock -w 1800 9`, so every child
    inherits fd 9 pointing at the lock file. Calling the script under the
    wrapper AND having it re-exec through the wrapper again would then block on
    a lock the SAME process tree holds (measured: an infinite wait, no table).
    """
    try:
        fds = list(Path("/proc/self/fd").iterdir())
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(fd) == str(LOCK):
                return True
        except OSError:
            continue
    return False


def _rank_scan(target: float = 299e6) -> int:
    """No GPU: for each `rank`, the `d_ffn` that lands ~299M total and its ACTIVE cost.

    The sizing question the report answers: `rank` is what moves the bank's
    total/active ratio, so this walks the latent width and bisects `d_ffn`.
    """
    print(f"{'rank':>5} {'d_ffn':>7} {'total':>9} {'active':>9} {'ratio':>6}")
    for rank in (64, 128, 256, 384, 512, 640, 832):
        lo, hi, best = 256, 65536, None
        while lo <= hi:
            mid = max(256, (lo + hi) // 2 // 64 * 64)
            over = {**BANK, "rank": rank, "d_ffn": mid}
            total, active = active_params(DormouseModel(the_run(over, 8, 1).model, device="meta"))
            if best is None or abs(total - target) < abs(best[1] - target):
                best = (mid, total, active)
            if total < target:
                lo = mid + 64
            else:
                hi = mid - 64
        mid, total, active = best
        print(f"{rank:5d} {mid:7d} {total / 1e6:8.1f}M {active / 1e6:8.1f}M {total / active:6.2f}")
    return 0


def _under_lock(argv: list[str]) -> int | None:
    """Re-exec through `scripts/gpu_lock.sh`; `None` means the lock is already held."""
    if os.environ.get(LOCK_ENV) == "1" or _inherited_lock_fd():
        return None
    if not LOCK.is_file():
        print(f"loud: {LOCK} is missing — no guard, no GPU job", file=sys.stderr)
        return 2
    env = {**os.environ, "GPU_TIMEOUT": os.environ.get("GPU_TIMEOUT", "3600")}
    proc = subprocess.run(
        ["bash", str(LOCK), "env", f"{LOCK_ENV}=1", sys.executable, str(Path(__file__)), *argv],
        env=env,
    )
    if proc.returncode in (EXIT_BUSY, EXIT_TIMEOUT):
        print(
            f"\n=== nothing was measured (exit {proc.returncode}: the card was held or the "
            f"run timed out). Quote no number from this run.\n",
            file=sys.stderr,
        )
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="control,control-kda,bank")
    ap.add_argument("--batch-sizes", default="24,16")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--warm", type=int, default=WARM)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true", help="no GPU: assert the two param counts")
    ap.add_argument("--rank-scan", action="store_true", help="no GPU: the sizing table")
    ap.add_argument(
        "--moe-kernel",
        default="batched",
        choices=("padded", "batched", "triton", "naive"),
        help="the bank arm's dispatch (default: dropless `batched`)",
    )
    ap.add_argument(
        "--moe-capacities",
        default="0.0",
        help="comma list of moe_capacity for the bank arm; 0.0 = DROPLESS",
    )
    a = ap.parse_args(argv)

    if a.check:
        return _check()
    if a.rank_scan:
        return _rank_scan()
    if a.device == "cuda":
        locked = _under_lock(argv if argv is not None else sys.argv[1:])
        if locked is not None:
            return locked
    device = torch.device(a.device)
    if not DATA.is_file():
        print(f"loud: the corpus is missing: {DATA}", file=sys.stderr)
        return 2
    names = [n.strip() for n in a.arms.split(",") if n.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        print(f"--arms: unknown {unknown}. Known: {', '.join(ARMS)}", file=sys.stderr)
        return 2
    batches = [int(b) for b in a.batch_sizes.split(",") if b.strip()]
    caps = [float(c) for c in a.moe_capacities.split(",") if c.strip()]
    #: `(label, model overrides)` — the bank arm expands into one row per capacity
    #: so a sweep is ONE command and every row names its own cap in the label.
    plan: list[tuple[str, dict]] = []
    for name in names:
        if name != "bank":
            plan.append((name, ARMS[name]))
            continue
        for cap in caps:
            label = "bank" if cap == 0.0 else f"bank-c{cap:g}"
            plan.append((label, {**BANK, "moe_kernel": a.moe_kernel, "moe_capacity": cap}))

    free_mib = (torch.cuda.mem_get_info()[0] / 2**20) if device.type == "cuda" else 0.0
    print(
        f"device {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}; "
        f"{free_mib:,.0f} MiB free of {torch.cuda.mem_get_info()[1] / 2**30:.2f} GiB; "
        f"steps={a.steps} warm={a.warm} seq={SEQ} seed={a.seed}"
    )
    print(f"corpus {DATA} ({DATA.stat().st_size:,} B); warmup_tokens={WARMUP_TOKENS:,}")
    print(f"bank: moe_kernel={a.moe_kernel} capacities={caps} (0.0 = dropless)")
    rows: list[dict] = []
    for batch in batches:
        for name, overrides in plan:
            row = measure(
                name,
                overrides,
                batch,
                device=device,
                steps=a.steps,
                warm=a.warm,
                seed=a.seed,
            )
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"[{name} b{batch}] {row['ms_median']:.1f} ms/step (min "
                    f"{row['ms_min']:.1f}) peak {row['peak_gib']:.2f} GiB "
                    f"{row['bytes_per_step'] / (row['ms_median'] / 1e3):,.0f} B/s "
                    f"params {row['total'] / 1e6:.2f}M/{row['active'] / 1e6:.2f}M "
                    f"graphs {row['graphs']}",
                    flush=True,
                )
                print(f"  counters {row['counters']}", flush=True)
            else:
                print(f"[{name} b{batch}] {row['status']} — nothing measured", flush=True)
    print("\n" + _table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
