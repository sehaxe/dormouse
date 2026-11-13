#!/usr/bin/env python3
"""One step of `flagship3_dense`: per-kernel CUDA time + per-op MEMORY TRAFFIC.

`kernel_profile.py` answers "which region", `profile_phase.py` answers "which
kernel". Neither answers the question a fusion has to start from: **which op
moves how many bytes, and how many of those bytes are a separate pass that a
fused kernel could delete**. This file answers exactly that, on the live form
(`block_form=transformer`, b8 x s1024), with two instruments run on separate
steps so they cannot perturb each other:

* `torch.profiler` (CPU+CUDA, `record_shapes`) → the top kernels by self device
  time, i.e. what the card is actually busy with;
* a `TorchDispatchMode` census → for every aten op the step runs, the bytes it
  READS + WRITES. A view (`data_ptr` shared with an input) costs 0; everything
  else costs `sum(in bytes) + sum(out bytes)`, the same convention
  `scripts/pass_census.py` and `docs/records/reviews/2026-11-08-kernel-audit.md` use, so
  the totals here are comparable to the audit's 52.5 GB/step seam.

Both run after 20 warm steps. `--fused` installs
`dormouse.model.fused_pointwise.patch()` first, so the same two instruments
measure the fused seam; that is how the "after" column of the A/B is produced.

    # profile + census, baseline
    uv run python scripts/bench_pointwise_fusion.py --mode profile --json /tmp/px_base.json
    # 150-step A/B timing, baseline vs fused
    uv run python scripts/bench_pointwise_fusion.py --mode ab --steps 150 --json /tmp/px_ab.json

Synthetic tokens (`torch.randint`, as `profile_phase.py` does): timing and
traffic at this shape do not depend on the token VALUES (MoD routing is off at
`mod_capacity=1.0`, the NaN firewall takes the same branch), and a synthetic
batch cannot contaminate a checkpoint. This script writes no checkpoint, no
eval and no run-registry entry; it is a bench, not a train.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from dormouse.cli import _device, _model  # noqa: E402
from dormouse.config import load_run, validate_train  # noqa: E402
from dormouse.train.optim import MixOptimizer  # noqa: E402

#: The card's measured HBM ceiling (docs/records/reviews/2026-11-08-kernel-audit.md §3).
BW_BYTES_S = 1.9e12

#: A fusion deletes PASSES. These are the ops that are a pass over a tensor:
#: one elementwise kernel each, plus the row-wise norms (memory bound at this
#: width) and the cast/copy family. Everything else (GEMM, attention, collectives)
#: is transport or compute, not a fusible pass — classified separately so the
#: seam total cannot be inflated by a GEMM.
_PASS_OPS = {
    "aten::_to_copy",
    "aten::copy_",
    "aten::add",
    "aten::add_",
    "aten::sub",
    "aten::sub_",
    "aten::mul",
    "aten::mul_",
    "aten::div",
    "aten::div_",
    "aten::neg",
    "aten::abs",
    "aten::exp",
    "aten::log",
    "aten::tanh",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::gelu",
    "aten::gelu_backward",
    "aten::silu",
    "aten::rsqrt",
    "aten::sqrt",
    "aten::pow",
    "aten::rsub",
    "aten::clamp",
    "aten::clamp_",
    "aten::where",
    "aten::nan_to_num_",
    "aten::fill_",
    "aten::zero_",
    "aten::lerp_",
    "aten::cat",
    "aten::native_layer_norm",
    "aten::native_layer_norm_backward",
    "aten::rms_norm",
    "aten::_foreach_mul_",
    "aten::_foreach_norm",
    "aten::_foreach_add_",
}
#: Metadata only: these return a view (or a stride-0 alias) and move no bytes.
#: Counting them is how a census reports 52 GB of "traffic" on a step that moves
#: 20: `view` alone re-describes the SAME storage ~300 times per step, and a
#: `data_ptr`-based aliasing check cannot see it on meta (every ptr is 0) — so
#: the names are listed instead of inferred.
_META_OPS = {
    "aten::view",
    "aten::_unsafe_view",
    "aten::detach",
    "aten::transpose",
    "aten::t",
    "aten::expand",
    "aten::as_strided",
    "aten::slice",
    "aten::select",
    "aten::narrow",
    "aten::permute",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::split",
    "aten::split_with_sizes",
    "aten::chunk",
    "aten::unbind",
    "aten::alias",
    "aten::reshape",  # a real copy shows up as clone/_to_copy on the same call
}
_GEMM_PREFIX = ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::_scaled_mm")
_ATTN_PREFIX = ("aten::_scaled_dot_product", "aten::scaled_dot_product", "aten::softmax")
_OPT_PREFIX = ("aten::_foreach", "aten::lerp_", "aten::adamw")


def _tensors(x):
    """Every tensor inside `x` (walks tuples/lists/dicts; the dispatch args)."""
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


def _opname(func) -> str:
    """`aten.add.Tensor` -> `aten::add` (the name the op sets are written in)."""
    parts = str(func).split(".")
    return f"{parts[0]}::{parts[1]}" if len(parts) > 1 else str(func)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class TrafficCensus(TorchDispatchMode):
    """Per-op bytes for one step, split by phase (`model` / `optimizer`)."""

    def __init__(self, by_shape: bool = False) -> None:
        self.phase = "model"
        self.by_shape = by_shape
        self.rows: dict[str, dict] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        out = func(*args, **(kwargs or {}))
        name = _opname(func)
        if name in _META_OPS:
            return out
        ins = list(_tensors(args)) + list(_tensors(kwargs or {}))
        outs = list(_tensors(out))

        # Storage identity, not `data_ptr`: on META every data_ptr is 0, so a
        # ptr-keyed dedupe collapses every operand into one and undercounts an
        # elementwise op by ~2x (measured 2026-11-08; the first census run of
        # this file reported `add` as one read).
        def sid(t: torch.Tensor) -> int:
            return t.untyped_storage()._cdata

        in_sid = {sid(t) for t in ins}
        read, seen = 0, set()
        for t in ins:
            if t.numel() and sid(t) not in seen:
                seen.add(sid(t))
                read += _nbytes(t)
        write, seen = 0, set()
        for t in outs:
            # an in-place op (`copy_`, `nan_to_num_`, ...) WRITES its own input:
            # same storage, real traffic. A view (`view`, `detach`) writes
            # nothing and is skipped by name above.
            if t.numel() and (name.endswith("_") or sid(t) not in in_sid) and sid(t) not in seen:
                seen.add(sid(t))
                write += _nbytes(t)
        key = name
        if self.by_shape and (outs or ins):
            t0 = outs[0] if outs else ins[0]
            src = ins[0].dtype if ins else None
            key = f"{name}|{tuple(t0.shape)}|{src}->{t0.dtype}"
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = {
                "op": name,
                "key": key,
                "launches": 0,
                "read": 0,
                "write": 0,
                "shapes": [],
            }
        row["launches"] += 1
        row["read"] += read
        row["write"] += write
        if len(row["shapes"]) < 3:
            sig = ",".join(
                f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}" for t in (outs or ins)[:2]
            )
            if sig and sig not in row["shapes"]:
                row["shapes"].append(sig)
        return out


def _tensor_class(key: str) -> str:
    """Which family a census row's tensor belongs to, from its own key.

    `attn` = the [heads, t, t] score matrix (only the MATH SDPA path makes one;
    flash never materialises it), `head` = per-head q/k/v work (RoPE), `act` =
    the [b, t, d] residual/activation stream. A pointwise seam lives in `act`.
    """
    parts = key.split("|")
    if len(parts) < 2:
        return "act"
    if ", 1024, 1024)" in parts[1]:
        return "attn"
    if parts[1].startswith("(8, 13,"):
        return "head"
    return "act"


def _class(op: str) -> str:
    if op in _META_OPS:
        return "meta"
    if op.startswith(_GEMM_PREFIX):
        return "gemm"
    if op.startswith(_ATTN_PREFIX):
        return "attention"
    if op in _PASS_OPS or op.startswith(_OPT_PREFIX):
        return "pass"
    return "other"


def build(a: argparse.Namespace):
    over = {
        "batch": a.batch,
        "seq_len": a.seq,
        "eval": False,
        "eval_every": 10**9,
        "ckpt_every": 10**9,
        "doc_masking": False,
        "mix_plan": "",
        "data": "",
    }
    if a.fast:
        # The FAST rung of `scripts/throughput_ladder.sh` (`muon_fast`): the
        # precision rungs only, so both arms pay the same optimizer.
        over |= {"ns_dtype": "bf16", "ns_steps": 3, "muon_state_dtype": "bf16"}
        over |= {"muon_lr_mult": 12.0, "lr": 3e-4}
    run = load_run(a.preset, train_overrides=over)
    validate_train(run.train)
    dev = _device(a.device)
    model = _model(run).to(dev).train()
    if a.mode == "meta":
        model = model.to("meta")
        ids = torch.randint(0, run.model.vocab, (a.batch, a.seq), device="meta")
        return run, torch.device("meta"), model, None, ids
    c = run.train
    opt = MixOptimizer(
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
        muon_chunk_elems=c.muon_chunk_elems,
        muon_lr_mult=c.muon_lr_mult,
        muon_lr_mode=c.muon_lr_mode,
        eps=c.adam_eps,
        tsct_expert_lr=run.model.tsct_expert_lr,
    )
    ids = torch.randint(0, run.model.vocab, (a.batch, a.seq), device=dev)
    return run, dev, model, opt, ids


def one_step(model, opt, ids, *, census: TrafficCensus | None = None) -> torch.Tensor:
    opt.zero_grad(set_to_none=True)
    logits, loss, _c = model(ids, targets=ids)
    loss.backward()
    if census is not None:
        census.phase = "optimizer"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        census.phase = "model"
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="profile", choices=("profile", "ab", "ab2", "meta"))
    ap.add_argument("--windows", type=int, default=4, help="ab2: alternating windows/arm")
    ap.add_argument("--preset", default="configs/flagship3_dense.toml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--no-fast", dest="fast", action="store_false")
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--by-shape", action="store_true", help="census rows keyed by shape too")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    if a.fused:
        from dormouse.model import fused_pointwise as fp

        fp.patch()
        print(f"fused seam installed: {fp.describe()}")

    run, dev, model, opt, ids = build(a)
    n = run.model
    print(
        f"{a.preset} block_form={n.block_form} d={n.d_model} ffn={n.d_ffn} "
        f"layers={n.n_layers} b={a.batch} s={a.seq} fast={a.fast} fused={a.fused}"
    )
    if a.mode == "meta":
        # CPU-only cross-check of the byte table: meta tensors carry the real
        # shapes and dtypes, so the census arithmetic is exact without the card
        # (the method `docs/records/reviews/2026-11-08-kernel-audit.md` §2 used). No
        # timing is read here, and SDPA falls back to the math backend on meta,
        # so the attention rows are NOT the card's — they are excluded from the
        # pointwise seam by `_class` anyway.
        model.to("meta")
        ids = torch.randint(0, run.model.vocab, (a.batch, a.seq), device="meta")
        census = TrafficCensus(by_shape=a.by_shape)
        with census:
            logits, loss, _c = model(ids, targets=ids)
            loss.backward()
        _report_census(census, a)
        return 0

    for _ in range(a.warmup):
        one_step(model, opt, ids)
    torch.cuda.synchronize(dev)

    if a.mode == "ab":
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        times: list[float] = []
        for _ in range(a.steps):
            ev0.record()
            one_step(model, opt, ids)
            ev1.record()
            torch.cuda.synchronize(dev)
            times.append(ev0.elapsed_time(ev1))
        times.sort()
        med = statistics.median(times)
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        bps = a.batch * a.seq
        out = {
            "mode": "ab",
            "fused": a.fused,
            "fast": a.fast,
            "batch": a.batch,
            "seq": a.seq,
            "steps": a.steps,
            "ms_median": med,
            "ms_p95": p95,
            "ms_min": times[0],
            "kb_s": bps / med,
            "ms_all": times,
        }
        print(
            f"AB fused={a.fused} n={a.steps} median {med:.2f} ms  p95 {p95:.2f}  "
            f"min {times[0]:.2f}  -> {bps / med:.1f} kB/s"
        )
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        return 0

    if a.mode == "ab2":
        # Paired A/B in ONE process: the same weights, the same optimizer state,
        # alternating windows. A separate-process A/B carries clock/allocator
        # drift of the same order as a 1% effect; this one differences that out.
        from dormouse.model import fused_pointwise as fp2

        bps = a.batch * a.seq
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)

        def window(n: int) -> list[float]:
            ts: list[float] = []
            for _ in range(n):
                ev0.record()
                one_step(model, opt, ids)
                ev1.record()
                torch.cuda.synchronize(dev)
                ts.append(ev0.elapsed_time(ev1))
            return ts

        base: list[float] = []
        fused: list[float] = []
        per: list[tuple[float, float]] = []
        fp2.unpatch()  # ab2 owns the arm switch; --fused is not used here
        fp2.patch()
        one_step(model, opt, ids)  # JIT + allocator warm, UNTIMED
        fp2.unpatch()
        for i in range(a.windows):
            if i % 2 == 0:
                tb = window(a.steps)
                fp2.patch()
                one_step(model, opt, ids)  # untimed, same reason
                tf = window(a.steps)
                fp2.unpatch()
            else:
                fp2.patch()
                one_step(model, opt, ids)
                tf = window(a.steps)
                fp2.unpatch()
                tb = window(a.steps)
            base += tb
            fused += tf
            per.append((statistics.median(tb), statistics.median(tf)))
        mb, mf = statistics.median(base), statistics.median(fused)
        out = {
            "mode": "ab2",
            "batch": a.batch,
            "seq": a.seq,
            "steps_per_arm": a.steps * a.windows,
            "ms_median_base": mb,
            "ms_median_fused": mf,
            "delta_ms": mb - mf,
            "delta_pct": 100 * (mb - mf) / mb,
            "windows": per,
            "kb_s_base": bps / mb,
            "kb_s_fused": bps / mf,
            "calls": fp2.CALLS,
        }
        print(
            f"AB2 paired: base {mb:.2f} ms ({bps / mb:.1f} kB/s)  "
            f"fused {mf:.2f} ms ({bps / mf:.1f} kB/s)  "
            f"delta {mb - mf:+.2f} ms ({100 * (mb - mf) / mb:+.2f}%)  "
            f"windows {[(round(x, 1), round(y, 1)) for x, y in per]}"
        )
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        return 0

    # --- profile: kernel time on one step, traffic on the next ---------------
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one_step(model, opt, ids)
    torch.cuda.synchronize(dev)

    # `key_averages()` holds BOTH activities: a CUDA kernel row AND the CPU op
    # row that launched it, and the CPU row carries its kernels' device time.
    # Summing both double counts (318 ms on a 145 ms step, measured). Kernels
    # only, by `device_type`, is the honest device total.
    try:
        from torch.profiler import DeviceType

        rows = [
            r
            for r in prof.key_averages()
            if getattr(r, "device_type", None) == DeviceType.CUDA and r.self_device_time_total > 0
        ]
    except Exception:
        rows = [
            r
            for r in prof.key_averages()
            if r.self_device_time_total > 0
            and not r.key.startswith(("aten::", "c10::", "autograd::"))
        ]
    rows.sort(key=lambda r: r.self_device_time_total, reverse=True)
    gpu_ms = sum(r.self_device_time_total for r in rows) / 1e3
    print(f"\n  top kernels by self CUDA time (total {gpu_ms:.1f} ms):")
    for r in rows[: a.top]:
        if r.self_device_time_total <= 0:
            continue
        print(
            f"    {r.self_device_time_total / gpu_ms * 100:5.1f}%  "
            f"{r.self_device_time_total / 1e3:7.2f} ms  {r.count:6d}x  {r.key[:62]}"
        )

    census = TrafficCensus(by_shape=a.by_shape)
    with census:
        one_step(model, opt, ids, census=census)
    torch.cuda.synchronize(dev)
    _report_census(census, a, gpu_ms=gpu_ms)

    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {
                    "mode": "profile",
                    "fused": a.fused,
                    "fast": a.fast,
                    "gpu_ms": gpu_ms,
                    "traffic_bytes": sum(r["read"] + r["write"] for r in census.rows.values()),
                    "ops": sorted(census.rows.values(), key=lambda r: -(r["read"] + r["write"])),
                },
                indent=1,
            )
        )
    return 0


def _report_census(census: TrafficCensus, a: argparse.Namespace, gpu_ms: float = 0.0) -> None:
    ops = sorted(census.rows.values(), key=lambda r: -(r["read"] + r["write"]))
    total = sum(r["read"] + r["write"] for r in ops if _class(r["op"]) != "meta")
    by_group: dict[str, int] = defaultdict(int)
    for r in ops:
        by_group[_class(r["op"])] += r["read"] + r["write"]
    print(f"\n  traffic per step: {total / 1e9:.2f} GB  ({total / 1.9e12 * 1e3:.1f} ms @1.9TB/s)")
    for g, b in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f"    {g:10s} {b / 1e9:7.2f} GB  {100 * b / total:5.1f}%  {b / 1.9e12 * 1e3:6.1f} ms")
    if a.by_shape:
        cls: dict[str, int] = defaultdict(int)
        for r in ops:
            if _class(r["op"]) != "meta":
                cls[_tensor_class(r.get("key", r["op"]))] += r["read"] + r["write"]
        print("\n  by tensor family:")
        for g, b in sorted(cls.items(), key=lambda kv: -kv[1]):
            print(f"    {g:10s} {b / 1e9:7.2f} GB  {b / 1.9e12 * 1e3:6.1f} ms")
    print(f"\n  top ops by bytes ({a.top}):")
    print(f"    {'op':40s} {'MB':>9s} {'read':>8s} {'write':>8s} {'n':>6s}  per-call MB")
    for r in ops[: a.top]:
        b = r["read"] + r["write"]
        per = b / max(r["launches"], 1) / 1e6
        print(
            f"    {r['op'][:40]:40s} {b / 1e6:9.1f} {r['read'] / 1e6:8.1f} "
            f"{r['write'] / 1e6:8.1f} {r['launches']:6d}  {per:8.2f}  "
            f"{' | '.join(r['shapes'])[:44]}"
        )

    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {
                    "mode": "census",
                    "fused": a.fused,
                    "gpu_ms": gpu_ms,
                    "traffic_bytes": total,
                    "groups": dict(by_group),
                    "ops": ops,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    sys.exit(main())
