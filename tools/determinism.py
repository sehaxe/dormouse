#!/usr/bin/env python3
"""Cross-process determinism check for a dormouse checkpoint.

WHAT THIS IS FOR. `AGENTS.md` 3.7 and `docs/protocols/AB-PROTOCOL.md` carry a
cross-process reproducibility claim for `--seed`. This is the check behind the
replacement number, and it is a standalone script because the property is
cross-PROCESS: you cannot test it by building a model twice inside one process
(Device::seed does not rewind a consumed stream, and Device::flex() hands back
a shared device -- AGENTS.md 3.7).

    # compare two checkpoints that a trainer wrote in two separate processes
    tools/determinism.py check A/m.bin B/m.bin
    tools/determinism.py check A/m.bin B/m.bin --seed-differs   # expect a FAIL

    # drive the experiment end to end (needs a built trainer + a free GPU)
    tools/determinism.py run --steps 0 --reps 3

READING THE OUTPUT. Two numbers, because one of them lies.

  differing slots   a COUNT of f32 slots that are not bit-equal. Useless on its
                    own: a tensor whose last bit moved counts the same as one
                    that moved by 1.0. Same-seed pairs sit at 4-13%.
  relFro            ||A-B||_F / ||A||_F, the distance that actually means
                    something. Same seed ~1e-08, different seed ~1.4.

  The measured separation between those two is 1.1e8x, which is the answer to
  "is 3 seeds per arm implementable" (AGENTS.md 1.2): yes, and the check below
  is what keeps it that way.

THE PARSER IS SELF-VALIDATING, and that is not decoration. A burnpack record
stores its tensor `data_offsets` RELATIVE to the data section, and solving for
the section start is easy to get wrong; get it wrong and 53 of 54 tensors still
read as plausible f32 data, so the failure is silent. base=0 in particular
reads every tensor 1408 floats early and produces a clean-looking table of
wrong numbers (it reports only ~400 absurd values, all in the first tensor,
where the window lands on the record header). The assertion in `tensors()`
rejects any base that leaves a value >= 1e6, so a wrong solve raises instead of
lying. See docs/research/2026-09-30-cross-process-repro-renamed.md.
"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys
import time

import numpy as np

MAGIC = b"DMCK\x00\x02\x00\x00"
ABSURD = 1e6          # no parameter init produces this; see tensors()
# Thresholds. Same-seed must agree far below anything an A/B resolves; a
# different seed must differ hugely. 1e8x of measured headroom on both sides.
REL_FRO_SAME_SEED_MAX = 1e-5
REL_FRO_DIFF_SEED_MIN = 1e-1


# ---------------------------------------------------------------- the record
def _cbor(buf, p):
    """Minimal CBOR reader -> (value, next_pos). Enough for the record metadata."""
    ib = buf[p]
    mt, ai = ib >> 5, ib & 0x1F
    p += 1
    if ai < 24:
        val = ai
    elif ai == 24:
        val = buf[p]; p += 1
    elif ai == 25:
        val = int.from_bytes(buf[p:p + 2], "big"); p += 2
    elif ai == 26:
        val = int.from_bytes(buf[p:p + 4], "big"); p += 4
    elif ai == 27:
        val = int.from_bytes(buf[p:p + 8], "big"); p += 8
    else:
        raise ValueError(f"indefinite/unsupported ai={ai} at {p - 1}")
    if mt == 0:
        return val, p
    if mt == 3:
        return buf[p:p + val].decode(), p + val
    if mt == 4:
        out = []
        for _ in range(val):
            v, p = _cbor(buf, p)
            out.append(v)
        return out, p
    if mt == 5:
        out = {}
        for _ in range(val):
            k, p = _cbor(buf, p)
            v, p = _cbor(buf, p)
            out[k] = v
        return out, p
    if mt == 7:
        return {20: False, 21: True, 22: None}.get(ai, f"simple({ai})"), p
    raise ValueError(f"CBOR major type {mt} ai={ai} at {p - 1}")


SECTIONS = ("model", "optim", "teacher")


def sections(path):
    b = open(path, "rb").read()
    if b[:8] != MAGIC:
        raise SystemExit(f"{path}: not a DMCK v2 checkpoint ({b[:8]!r})")
    step, mlen, olen, tlen, flags = struct.unpack("<QQQQQ", b[8:48])
    if 48 + mlen + olen + tlen != len(b):
        # The container declares every section's length. This is the ONLY place
        # a truncated tail is visible: the base solve inside tensors() shrinks
        # by exactly the amount the record is short, so a short record solves
        # cleanly and the plausibility assertion still passes (verified).
        raise SystemExit(
            f"{path}: header declares {48 + mlen + olen + tlen} bytes, "
            f"file has {len(b)} -- truncated or not fully written")
    return b, {"step": step, "flags": flags, "size": len(b),
               "model": b[48:48 + mlen], "optim": b[48 + mlen:48 + mlen + olen],
               "teacher": b[48 + mlen + olen:48 + mlen + olen + tlen]}


def tensors(rec):
    """-> [(name, shape, start, end)] with ABSOLUTE byte offsets, self-checked.

    Layout: b"NRUB" | u16 ver | u32 size, then a CBOR map
    {"tensors": {name -> {dtype, shape, data_offsets, param_id}}} in [0, base),
    then the f32 data, ending at the record end. The declared offsets are
    relative to the data section, which starts at

        base = len(record) - (sum(sizes) + sum(internal gaps))

    The internal gaps are the padding runs between consecutive declared ranges
    (208 + 252 + 252 bytes on this model). Forget them and the solve lands 712
    bytes high, every tensor reads misaligned, and nothing raises -- which is
    why the plausibility assertion at the end exists.
    """
    if rec[:4] != b"NRUB":
        raise SystemExit(f"not a burnpack record: {rec[:4]!r}")
    declared = struct.unpack("<I", rec[6:10])[0]
    body, meta_end = _cbor(rec, 10)
    if declared != meta_end - 10:
        raise SystemExit(f"metadata length {declared} != parsed {meta_end - 10}: "
                         "the record is truncated or not a v2 layout")
    raw = []
    for name, m in body["tensors"].items():
        if m["dtype"] != "F32":
            raise SystemExit(f"{name}: {m['dtype']} is not F32; widen this reader")
        s, e = m["data_offsets"]
        shape = list(m["shape"])
        n, exp = (e - s) // 4, int(np.prod(shape)) if shape else 1
        if n != exp:
            raise SystemExit(f"{name}: (end-start)/4={n} != prod(shape)={exp}")
        raw.append((name, shape, s, e))
    srt = sorted((s, e) for _, _, s, e in raw)
    gaps = sum(max(0, srt[i][0] - srt[i - 1][1]) for i in range(1, len(srt)))
    base = len(rec) - (sum(e - s for _, _, s, e in raw) + gaps)
    if base < meta_end:
        raise SystemExit(f"base {base} overlaps the metadata ending at {meta_end}")
    if base - meta_end > 4096:
        raise SystemExit(f"base {base} is {base - meta_end} B past the metadata end "
                         f"({meta_end}); the gap solve is wrong")
    out = [(n, sh, s + base, e + base) for n, sh, s, e in raw]
    for name, _, s, e in out:
        a = np.frombuffer(rec, dtype="<f4", count=(e - s) // 4, offset=s)
        f = a[np.isfinite(a)]
        if f.size == 0 or np.abs(f).max() >= ABSURD:
            raise SystemExit(
                f"base {base} is WRONG: {name} has max|v|={np.abs(f).max():.3e}. "
                "A wrong base does not raise on its own -- this is that check.")
    return out


def load(path, section="model"):
    rec = sections(path)[1][section]
    out = {}
    for name, shape, s, e in tensors(rec):
        a = np.frombuffer(rec, dtype="<f4", count=(e - s) // 4, offset=s)
        out[name] = a.reshape(shape) if shape else a
    return out


# ------------------------------------------------------------- the comparison
def distance(a, b):
    """-> (differing_slots, total_slots, relFro, max_abs_delta, nan_slots)"""
    assert set(a) == set(b), set(a) ^ set(b)
    nd = tot = 0
    num = den = 0.0
    mx = 0.0
    nan = 0
    for k in sorted(a):
        x = a[k].astype(np.float64).ravel()
        y = b[k].astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        nan += int((~m).sum())
        d = x[m] - y[m]
        nd += int((d != 0).sum())
        tot += int(m.sum())
        num += float((d * d).sum())
        den += float((x[m] * x[m]).sum())
        mx = max(mx, float(np.abs(d).max()) if d.size else 0.0)
    return nd, tot, (num / den) ** 0.5, mx, nan


def report(label_a, pa, label_b, pb, seed_differs):
    a, b = load(pa), load(pb)
    nd, tot, rf, mx, nan = distance(a, b)
    print(f"  {label_a} vs {label_b}   ({tot:,} comparable slots)")
    print(f"    differing slots : {nd:,}  ({100 * nd / tot:.4f}%)")
    print(f"    relFro          : {rf:.6e}")
    print(f"    max |delta|     : {mx:.6e}")
    print(f"    non-finite      : {nan}")
    return rf


def check(pa, pb, seed_differs=False):
    print(f"check {pa}\n   vs {pb}")
    rf = report("A", pa, "B", pb, seed_differs)
    if seed_differs:
        ok = rf >= REL_FRO_DIFF_SEED_MIN
        print(f"    gate: different seed must differ (relFro >= "
              f"{REL_FRO_DIFF_SEED_MIN:g}) -> {'PASS' if ok else 'FAIL'}")
    else:
        ok = rf <= REL_FRO_SAME_SEED_MAX
        print(f"    gate: same seed must agree (relFro <= "
              f"{REL_FRO_SAME_SEED_MAX:g}) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ the run
def run(args):
    """Drive N zero-step trainer runs, each in its own process, and compare."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # A worktree (AGENTS.md 1.6) has its own target/ only if it was built; in
    # practice the build lives in the shared checkout, so look there too. The
    # binary is the instrument here, not the thing under test, so a trainer
    # from a neighbouring checkout is fine -- say which one, because "which
    # binary" is a question a reproducibility number has to answer.
    main_root = root
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute",
                                 "--git-common-dir"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        main_root = os.path.dirname(common)      # <checkout>/.git -> <checkout>
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass
    cands = [args.bin] if args.bin else [
        os.path.join(root, "target", "release", "train"),
        os.path.join(main_root, "target", "release", "train"),
    ]
    train = next((c for c in cands if c and os.path.exists(c)), None)
    if train is None:
        raise SystemExit("no trainer found; tried " + ", ".join(str(c) for c in cands)
                         + " -- pass --bin")
    print(f"trainer: {train}")
    corpus = args.corpus
    if not os.path.isdir(corpus):
        raise SystemExit(f"no corpus dir at {corpus}")
    out = args.out
    os.makedirs(out, exist_ok=True)
    paths, epochs = [], []
    for i in range(args.reps):
        run_dir = os.path.join(out, f"r{i}")
        os.makedirs(run_dir, exist_ok=True)
        cmd = [train, "--data", corpus, "--preset", args.preset,
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--batch", str(args.batch), "--seq-len", str(args.seq_len),
               "--no-kda", "--no-engram",
               "--ckpt-dir", run_dir, "--ckpt-name", "m"]
        t0 = time.time()
        # A trainer launch goes under a MemoryMax scope (AGENTS.md 2.4). The
        # guard is best-effort: a user bus is not always there (containers, CI).
        scope = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=40G"]
        if subprocess.run(scope + ["true"], capture_output=True).returncode != 0:
            scope = []
            print("note: systemd-run unavailable, running without a memory cap")
        r = subprocess.run(scope + cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"trainer failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        paths.append(os.path.join(run_dir, "m.bin"))
        epochs.append(int(t0))
        print(f"  rep {i}: ok in {time.time() - t0:.0f}s (step {args.steps})")
    print(f"\n{len(paths)} runs, seed {args.seed}, {args.steps} steps each\n")
    worst = 0.0
    rc = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            gap = int(epochs[j] - epochs[i])
            print(f"pair {i} vs {j}   gap {gap}s")
            if check(paths[i], paths[j]) != 0:
                rc = 1
            worst = max(worst, 0.0)
            print()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare two checkpoints from two processes")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--seed-differs", action="store_true",
                   help="the two runs used different --seed; assert they differ")
    r = sub.add_parser("run", help="drive the zero-step experiment end to end")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--preset", default="small")
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--seq-len", type=int, default=128)
    r.add_argument("--corpus", default=os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "dormouse-determinism", "corpus"))
    r.add_argument("--out", default=os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "dormouse-determinism", "runs"))
    r.add_argument("--bin", default=None)
    args = ap.parse_args()
    if args.cmd == "check":
        return check(args.a, args.b, args.seed_differs)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
