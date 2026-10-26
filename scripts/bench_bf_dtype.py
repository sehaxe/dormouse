"""fp32 vs bf16 for the ByteFlow channel, per stage: encode / loop / decode.

    uv run --extra gpu python scripts/bench_bf_dtype.py

`byteflow.py`'s precision rule (2026-10-21) lets the net run bf16 matmuls with
fp32 masters. This bench prices that at the channel's profile shape — b=4 x
T=512 bytes, K=128 patch latents, d=1024, f=2730, rank 64, E=256 / topk 2
(`medium`'s geometry; the same loop shape `bench_moe_fp32.py` measured) — in
both dtypes, one process, one card.

It prints, per dtype: the model's parameter MiB (must AGREE: the flag moves the
compute, not the masters), ms per stage (median of `REPS`, one warm-up
excluded, `cuda.synchronize` around each call), and the peak allocated GiB of
the model plus the whole staged forward. The speedup column is bf16 over fp32 per stage.

Forward only, under `no_grad`: this is a GEMM/activation measurement
(the trainer's backward is the loop's, already bf16). The loop arm is the MoE
(`use_kda=false`) so the bench needs no FLA.

Preflight (AGENTS §1.5): no `dormouse` process alive and the card holding
< 1 GiB. Otherwise it prints a SKIP banner naming what it found and exits 0 —
nothing is measured, so there is no number to quote.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from dormouse.config import ModelConfig
from dormouse.instrument import Counters
from dormouse.model import DormouseModel

# `scripts/` is not a package (a test loads this file by path): the lane guard is
# a sibling file, so its directory goes on the path, APPENDED not inserted (a
# script name at the front of `sys.path` would shadow a package of that name).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

DEV = "cuda"
#: The channel's profile shape (the ByteFlow recipe's own numbers).
B, T, K = 4, 512, 128
D, F_DIM, RANK, E, TOPK = 1024, 2730, 64, 256, 2
MAX_ITER = 2
REPS = 10
DTYPES = (("fp32", False), ("bf16", True))


def config(bf16: bool) -> ModelConfig:
    return ModelConfig(
        d_model=D,
        n_heads=16,
        head_dim=D // 16,
        d_ffn=F_DIM,
        rank=RANK,
        n_experts=E,
        moe_topk=TOPK,
        moe_capacity=0.0,  # the byteflow recipe runs dropless
        max_iter=MAX_ITER,
        bf16=bf16,
        use_byteflow=True,
        use_engram=False,  # no byte position to hash on the patch path (a loud refusal)
        use_kda=False,  # the loop arm here is the MoE: no FLA dependency
        use_attnres=False,
        loom=False,
        byteflow_k_tokens=K,
        byteflow_max_bytes=T,
        byteflow_d_local=D,
        byteflow_d_global=D,
        byteflow_heads_local=8,
        byteflow_heads_global=8,
        byteflow_e_layers=3,
        byteflow_w_local=T,
        byteflow_d_ff_local=F_DIM,
        byteflow_bins=16,
    )


def _timed(fn) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def run_dtype(label: str, bf16: bool, bytes_: torch.Tensor) -> dict[str, float]:
    # The peak covers the model AND the staged forward, so clear the previous
    # arm's cached blocks and reset BEFORE the parameters are allocated.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(11)
    model = DormouseModel(config(bf16), device=DEV)
    assert model.bf is not None, "the channel must be armed (moe_topk > 0)"
    bf = model.bf
    loop_block = model.loop
    params = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    counters = Counters()
    with torch.no_grad():
        h, z, sel = bf.net.encode_chunks(bytes_, K)

    def encode():
        with torch.no_grad():
            return bf.net.encode_chunks(bytes_, K)

    def loop():
        # The bridge (in_proj) and the K-patch loop: the stage between the two
        # halves of the net.
        with torch.no_grad():
            return loop_block(bf.in_proj(z), None, None, None, counters=counters)[0]

    out = loop()

    def decode():
        with torch.no_grad():
            return bf.net.decode_chunks(bf.out_proj(out), h, sel)

    re = _timed(encode)
    rl = _timed(loop)
    out = loop()
    rd = _timed(decode)
    peak = torch.cuda.max_memory_allocated() / 2**30
    logits = decode()
    assert tuple(logits.shape) == (B, T, 256), logits.shape
    print(
        f"[{label}] params={params:.1f} MiB encode={re:.2f} ms loop={rl:.2f} ms "
        f"decode={rd:.2f} ms total={re + rl + rd:.2f} ms peak={peak:.3f} GiB",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return {"encode": re, "loop": rl, "decode": rd, "peak": peak, "params": params}


def main() -> int:
    busy = subprocess.run(
        ["pgrep", "-af", "[d]ormouse (train|rl|sft|bench|serve)|[/]target/release/train"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = preflight("cuda") or (f"a dormouse process is alive:\n{busy}" if busy else None)
    if why is None:
        # This bench's own ceiling is 1 GiB, tighter than the lane guard's 2.0:
        # a card already holding a second model is contended however idle it looks.
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if smi and int(smi.split()[0]) >= 1024:
            why = f"the card already holds {smi.split()[0]} MiB (this bench's ceiling is 1024)"
    if why is not None:
        return skip(why)
    if not torch.cuda.is_available():
        return skip("torch sees no CUDA device")
    torch.manual_seed(11)
    bytes_ = torch.randint(0, 256, (B, T), dtype=torch.long, device=DEV)
    print(
        f"\n[shape] b={B} T={T} K={K} d={D} f={F_DIM} rank={RANK} E={E} topk={TOPK} "
        f"iters={MAX_ITER}",
        flush=True,
    )

    def total(r: dict[str, float]) -> float:
        return r["encode"] + r["loop"] + r["decode"]

    got = {label: run_dtype(label, flag, bytes_) for label, flag in DTYPES}
    a, b = got["fp32"], got["bf16"]
    print(
        f"\n[speedup] encode {a['encode'] / b['encode']:.2f}x loop "
        f"{a['loop'] / b['loop']:.2f}x decode {a['decode'] / b['decode']:.2f}x "
        f"total {total(a) / total(b):.2f}x",
        flush=True,
    )
    assert a["params"] == b["params"], "the dtype moved the parameter count"
    return 0


if __name__ == "__main__":
    sys.exit(main())
