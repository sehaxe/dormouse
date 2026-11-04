"""Per-step audit of the TRAINING step: host reads, python loops, allocations.

    python scripts/step_audit.py                 # the CPU tiny preset (no GPU)
    python scripts/step_audit.py --gpu           # the real launcher, audited
    python scripts/step_audit.py --census-only   # the loop inventory

`--tiny` runs the same tiny model/stream the CPU suite uses and prints one row
per step: allocations + requested bytes (+ the CUDA allocator's peak where CUDA
exists), host reads split by device, and every instrumented loop with its
instance/iteration/module counts. It then prints the static loop census and
FAILS (exit 1) on a loop in the step's files that is neither instrumented nor
excused with a reason.

`--gpu` is the owner's benchmark: it runs the real CLI (same argv construction
as `scripts/sync_audit.py`, the m400 resume recipe) under the same patches, with
`log_every` from `--log-every` (default 100). The rows are per step, so
"non-report steps have ZERO device reads" is a reading, not a ratio. GPU only —
it needs the m400 checkpoint under `checkpoints/m400`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse import stepaudit  # noqa: E402
from dormouse.config import ModelConfig, RunConfig, TrainConfig  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.train import train  # noqa: E402

#: Mirrors `tests/conftest.py::TINY` (d=64, E=4, rank=8). Kept as a literal so
#: the script runs from a source checkout without importing the test package.
TINY = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=64,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
)
SEQ, BATCH = 64, 2

#: Host reads a CPU-only run may take on a non-report step. Everything in this
#: list is a path that does not exist on CUDA: the stock optimizer's CPU
#: fallback (foreach is CUDA-only, so `_single_tensor_adam` calls `.item()` once
#: per parameter) and the MoE reference dispatcher (`experts._naive_dispatch`,
#: the sibling lane's CPU path — it `.tolist()`s the unique expert ids; on CUDA
#: the triton kernel runs instead).
CPU_ONLY_FRAMES = stepaudit.CPU_ONLY_FRAMES


def _tiny_run(tmp: Path, steps: int, log_every: int) -> tuple[DormouseModel, RunConfig]:
    path = tmp / "corpus.bin"
    path.write_bytes(bytes(range(16)) * (4096 // 16 + 1))
    cfg = TrainConfig(
        steps=steps,
        batch=BATCH,
        seq_len=SEQ,
        lr=0.01,
        log_every=log_every,
        ckpt_every=10**6,
        ckpt_name="stepaudit",
        out_dir=str(tmp / "ckpts"),
        data=str(path),
    )
    model = ModelConfig(**TINY)
    return DormouseModel(model), RunConfig(source="tiny", model=model, train=cfg)


def _unexpected_reads(audit: stepaudit.StepAudit) -> list[str]:
    bad: list[str] = []
    for rec in audit.records:
        if rec.report:
            continue
        for frame, n in rec.frames.items():
            if any(marker in frame for marker in CPU_ONLY_FRAMES):
                continue
            bad.append(f"step {rec.index}: {n}x {frame}")
    return bad


def run_tiny(steps: int, log_every: int) -> int:
    torch.manual_seed(1)
    audit = stepaudit.install()
    with tempfile.TemporaryDirectory() as td:
        model, run = _tiny_run(Path(td), steps, log_every)
        try:
            train(model, run, log=lambda _line: None)
        finally:
            stepaudit.uninstall()
    print(audit.summary())

    bad = _unexpected_reads(audit)
    if bad:
        print("\nUNEXPECTED host reads on non-report steps (not a CUDA-only path):")
        for line in bad:
            print("  " + line)

    sites = stepaudit.census()
    print(f"\nloop census: {len(sites)} sites")
    for site in sites:
        if site.status == "UNMARKED":
            print(f"  UNMARKED {site.file}:{site.line} {site.kind} {site.function}")
    if not bad and not stepaudit.unmarked():
        print("gates: no unmarked loops, no unexpected non-report reads on a CPU-only path")
        return 0
    if stepaudit.unmarked():
        print(
            "Escape: wrap each UNMARKED loop in stepaudit.iterations(...), or add a "
            "reason to stepaudit.EXCUSED_FUNCTIONS if it is not part of a step."
        )
    return 1


def run_gpu(args, steps: int, log_every: int) -> int:
    import re
    import shutil

    from dormouse import cli

    # The same recipe `scripts/sync_audit.py` measured the 12 syncs with; the
    # checkpoint IS the resume identity, so its name/out_dir fields are renamed.
    sets = [
        "steps=" + str(steps),
        "batch=8",
        "seq_len=1024",
        "max_seq_len=1024",
        "grad_accum=1",
        "lr=5e-4",
        "warmup_tokens=126000000",
        "decay_frac=0.4",
        "min_lr_frac=0.01",
        "use_msa=false",
        "use_csa=true",
        "csa_chunk=8",
        "csa_coarse_chunk=256",
        "use_mhc=true",
        "use_attnres=true",
        "share_u=true",
        "use_engram=true",
        "engram_ram=false",
        "attn_latent_rank=256",
        "moe_kernel=triton",
        "moe_capacity=0.0",
        "doc_masking=false",
        "data=/home/sehaxe/dormouse/data_local/best/fwedu.bin",
        "mix_plan=configs/mix_v7.toml",
        "eval=false",
        "log_every=" + str(log_every),
        "eval_data=/home/sehaxe/dormouse/data_local/mix4/web_eval.bin",
        "eval_batches=8",
        "ckpt_every=100000",
        "layerrope=true",
        "layerrope_form=paper",
        "layerrope_index=iteration",
        "layerrope_init=paper",
        "rank=64",
        "n_experts=512",
        "seed=1",
        "opt_graph=false",
        "iter_checkpoint=false",
        "timers=true",
        "ckpt_name=stepaudit",
        "out_dir=/tmp/opencode/stepaudit",
    ]
    argv = ["train", args.config, "--resume"]
    for kv in sets:
        argv += ["--set", kv]
    sys.argv = ["dormouse"] + argv
    out = Path("/tmp/opencode/stepaudit")
    out.mkdir(parents=True, exist_ok=True)
    src = Path("/home/sehaxe/dormouse/checkpoints/m400")
    (out / "stepaudit.pt").write_bytes((src / "m400.pt").read_bytes())
    cfg = (src / "m400.config.toml").read_text()
    for key, val in (("ckpt_name", '"stepaudit"'), ("out_dir", '"/tmp/opencode/stepaudit"')):
        cfg = re.sub(rf"^{key} = .*$", f"{key} = {val}", cfg, flags=re.M)
    (out / "stepaudit.config.toml").write_text(cfg)
    shutil.rmtree(out / "stepaudit.ngram", ignore_errors=True)

    audit = stepaudit.install()
    try:
        cli.main()
    except SystemExit:
        pass
    finally:
        stepaudit.uninstall()
    print("\n" + audit.summary())
    rep = audit.report()
    print(
        f"\nper step: {rep['ms_step']:.1f} ms allocs={rep['allocs'] / max(rep['steps'], 1):.1f} "
        f"req={rep['alloc_bytes'] / max(rep['steps'], 1) / 1e6:.2f} MB "
        f"peak={rep['peak_bytes']} device reads={rep['device_reads']}"
    )
    bad = _unexpected_reads(audit)
    if bad:
        print("unexpected non-report reads:")
        for line in bad[:20]:
            print("  " + line)
    print(
        "\nthe owner's number: the `ms/step` on the `--timers` log lines, with the "
        "per-step alloc/peak rows above as the cost side of the same window."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="the CPU tiny preset (default)")
    ap.add_argument("--gpu", action="store_true", help="run the real m400 recipe on CUDA")
    ap.add_argument("--census-only", action="store_true", help="loop inventory, no run")
    ap.add_argument(
        "--steps", type=int, default=None, help="total steps (default 4 tiny / 150040 gpu)"
    )
    ap.add_argument("--log-every", type=int, default=None, help="default 3 tiny / 100 gpu")
    ap.add_argument("--config", default="configs/m400.toml", help="--gpu config")
    a = ap.parse_args()
    if a.census_only:
        for site in stepaudit.census():
            print(f"{site.status:8} {site.file}:{site.line:5} {site.kind:5} {site.function}")
        return 0 if not stepaudit.unmarked() else 1
    if a.gpu:
        return run_gpu(a, a.steps or 150040, a.log_every or 100)
    return run_tiny(a.steps or 4, a.log_every or 3)


if __name__ == "__main__":
    raise SystemExit(main())
