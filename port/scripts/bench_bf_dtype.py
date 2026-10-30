"""ByteFlow codec and dtype cost, per stage: encode / loop / decode.

    uv run --extra gpu python scripts/bench_bf_dtype.py

Two axes:

* **codec** — `attention` (the ported ByteFlow net, the default) vs `conv`
  (`model/convcodec.py`, strided-conv patchify + low-rank unpatchify). The
  point of the codec arm is `convcodec.py`'s MAC table: the attention encoder
  scales with BYTES, so at 8 bytes/patch it costs the same while the loop HALVES
  and its share of the step nearly doubles. This bench is the wall-time check
  of the arithmetic on the card.
* **dtype** — `byteflow.py`'s precision rule (2026-10-21) runs bf16 matmuls
  with fp32 masters. The parameter MiB must AGREE across a codec's two dtypes:
  the flag moves the compute, not the masters.

Shape: b=4 x T=512 bytes, d=1024, f=2730, rank 64, E=256 / topk 2 (`medium`'s
loop geometry, the same shape `bench_moe_fp32.py` measured), and K = T /
bytes-per-patch for BOTH codecs — 4 and 8 bytes/patch, the two compression
points the codec decision is made at.

It prints, per (codec, bytes/patch, dtype): ms per stage (median of `REPS`,
one warm-up excluded, `cuda.synchronize` around each call), the model's
parameter MiB, and peak allocated GiB. Then the codec table: attention vs conv
per stage at the same K, so the loop line is a control (it must not move) and
the encode/decode lines are the measurement.

Forward only, under `no_grad`: this is a GEMM/activation measurement
(the trainer's backward is the loop's, already bf16). The loop arm is the MoE
(`use_kda=false`) so the bench needs no FLA.

Preflight (AGENTS §1.5): no `dormouse` process alive and the card holding
fewer MiB than `BF_BENCH_MAX_MIB` (default 1024). Otherwise it prints a SKIP
banner naming what it found and exits 0 — nothing is measured, so there is no
number to quote.
"""

from __future__ import annotations

import os
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
B, T = 4, 512
D, F_DIM, RANK, E, TOPK = 1024, 2730, 64, 256, 2
MAX_ITER = 2
REPS = 10
DTYPES = (("fp32", False), ("bf16", True))
#: The codec decision's two compression points (K = T // bytes_per_patch).
BYTES_PER_PATCH = (4, 8)
CODECS = ("attention", "conv")


def config(bf16: bool, bpp: int, codec: str) -> ModelConfig:
    # `byteflow_patch_bytes` is the conv patchify's stride: under the attention
    # codec it is inert (and `validate` refuses a non-default value), so only
    # the conv arm sets it.
    conv = {"byteflow_patch_bytes": bpp} if codec == "conv" else {}
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
        byteflow_k_tokens=T // bpp,
        byteflow_max_bytes=T,
        byteflow_d_local=D,
        byteflow_d_global=D,
        byteflow_heads_local=8,
        byteflow_heads_global=8,
        byteflow_e_layers=3,
        byteflow_w_local=T,
        byteflow_d_ff_local=F_DIM,
        byteflow_bins=16,
        byteflow_codec=codec,
        **conv,
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


def run_arm(codec: str, bpp: int, label: str, bf16: bool, bytes_: torch.Tensor) -> dict[str, float]:
    # The peak covers the model AND the staged forward, so clear the previous
    # arm's cached blocks and reset BEFORE the parameters are allocated.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(11)
    model = DormouseModel(config(bf16, bpp, codec), device=DEV)
    assert model.bf is not None, "the channel must be armed (moe_topk > 0)"
    bf = model.bf
    loop_block = model.loop
    k = bf.k
    params = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    counters = Counters()
    with torch.no_grad():
        h, z, sel = bf.net.encode_chunks(bytes_, k)

    def encode():
        with torch.no_grad():
            return bf.net.encode_chunks(bytes_, k)

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
        f"[{codec}/{bpp}B/{label}] params={params:.1f} MiB encode={re:.2f} ms "
        f"loop={rl:.2f} ms decode={rd:.2f} ms total={re + rl + rd:.2f} ms "
        f"peak={peak:.3f} GiB",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return {"encode": re, "loop": rl, "decode": rd, "peak": peak, "params": params}


def _share(stage: dict[str, float]) -> float:
    """Codec's share of the staged forward (encode + decode, loop excluded)."""
    return (
        100
        * (stage["encode"] + stage["decode"])
        / sum(stage[k] for k in ("encode", "loop", "decode"))
    )


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
        # The ceiling is about a second MODEL, not about a desktop: this box
        # drives a display from the same card, so ~1.6 GiB is the compositor and
        # is not contention. Raise it explicitly when that is the case.
        ceiling = int(os.environ.get("BF_BENCH_MAX_MIB", "1024"))
        if smi and int(smi.split()[0]) >= ceiling:
            why = (
                f"the card already holds {smi.split()[0]} MiB (ceiling {ceiling}; "
                f"raise BF_BENCH_MAX_MIB if this is the desktop compositor, not a model)"
            )
    if why is not None:
        return skip(why)
    if not torch.cuda.is_available():
        return skip("torch sees no CUDA device")
    torch.manual_seed(11)
    bytes_ = torch.randint(0, 256, (B, T), dtype=torch.long, device=DEV)
    print(
        f"\n[shape] b={B} T={T} d={D} f={F_DIM} rank={RANK} E={E} topk={TOPK} "
        f"iters={MAX_ITER} bytes/patch={BYTES_PER_PATCH}",
        flush=True,
    )

    got: dict[tuple[str, int, str], dict[str, float]] = {}
    for bpp in BYTES_PER_PATCH:
        for codec in CODECS:
            for label, flag in DTYPES:
                got[codec, bpp, label] = run_arm(codec, bpp, label, flag, bytes_)
            a, b = got[codec, bpp, "fp32"], got[codec, bpp, "bf16"]
            assert a["params"] == b["params"], f"the dtype moved {codec}/{bpp}B's parameter count"
            print(
                f"[dtype {codec}/{bpp}B] encode {a['encode'] / b['encode']:.2f}x "
                f"loop {a['loop'] / b['loop']:.2f}x decode {a['decode'] / b['decode']:.2f}x",
                flush=True,
            )

    print("\n[codec] attention / conv, same K (the loop is the control):", flush=True)
    for bpp in BYTES_PER_PATCH:
        for label, _ in DTYPES:
            att, conv = got["attention", bpp, label], got["conv", bpp, label]
            print(
                f"[codec {bpp}B/{label}] encode {att['encode']:.2f}/{conv['encode']:.2f} ms "
                f"({att['encode'] / conv['encode']:.1f}x) "
                f"loop {att['loop']:.2f}/{conv['loop']:.2f} ms "
                f"({att['loop'] / conv['loop']:.1f}x) "
                f"decode {att['decode']:.2f}/{conv['decode']:.2f} ms "
                f"({att['decode'] / conv['decode']:.1f}x) "
                f"codec share att={_share(att):.1f}% conv={_share(conv):.1f}%",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
