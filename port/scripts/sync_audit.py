"""Count host syncs per training step, and name the caller.

    python scripts/sync_audit.py --steps 3 --batch 8 --seq 1024

Why: one step is ~320 ms for 8192 bytes on a 45 TFLOP/s card and the cost does
NOT move with the batch/window split (b1x8192 == b8x1024), which rules out
launch count and shape effects. A host round trip (~20-50 us, and it drains the
stream) is the remaining explanation. This patches `Tensor.item/.tolist/.cpu/
__bool__` and then runs the REAL CLI, so the data path, device moves and the
trainer are exactly the ones that are slow.

`scripts/step_audit.py` is the successor: same read patch (plus `__float__`/
`__int__`, which this one misses), and it also counts allocations and python
loops per step. Keep this script for the run-level sync ratio it was built for.
"""

from __future__ import annotations

import collections
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

CALLS: collections.Counter = collections.Counter()
FRAMES: collections.Counter = collections.Counter()
STEP: list[int] = [0]


def _blame() -> str:
    for fr in reversed(traceback.extract_stack()[:-2]):
        f = fr.filename
        if "site-packages/torch" in f or "/torch/" in f:
            continue
        return f"{Path(f).name}:{fr.lineno}:{fr.name}"
    return "?"


def _wrap(name: str, fn):
    def inner(self, *a, **kw):
        if self.is_cuda:
            CALLS[name] += 1
            FRAMES[_blame()] += 1
        return fn(self, *a, **kw)

    return inner


def main() -> int:
    torch.Tensor.item = _wrap("item", torch.Tensor.item)
    torch.Tensor.tolist = _wrap("tolist", torch.Tensor.tolist)
    torch.Tensor.cpu = _wrap("cpu", torch.Tensor.cpu)
    torch.Tensor.__bool__ = _wrap("bool", torch.Tensor.__bool__)

    from dormouse import cli

    # every flag the real launcher passes: a resume compares the snapshot, and a
    # missing `data` is a loud refusal before the first step
    sets = [
        "steps=150040",
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
        "log_every=1",
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
        "ckpt_name=syncaudit",
        "out_dir=/tmp/opencode/syncaudit",
    ]
    argv = ["train", "configs/m400.toml", "--resume"]
    for kv in sets:
        argv += ["--set", kv]
    sys.argv = ["dormouse"] + argv
    Path("/tmp/opencode/syncaudit").mkdir(parents=True, exist_ok=True)
    # the resume needs a copy of the pair under the audit name

    src = Path("/home/sehaxe/dormouse/checkpoints/m400")
    dst = Path("/tmp/opencode/syncaudit")
    (dst / "syncaudit.pt").write_bytes((src / "m400.pt").read_bytes())
    # the snapshot IS the resume identity: rename the fields that name the run,
    # or the gate (correctly) refuses
    cfg = (src / "m400.config.toml").read_text()
    for key, val in (
        ("ckpt_name", '"syncaudit"'),
        ("out_dir", '"/tmp/opencode/syncaudit"'),
    ):
        cfg = re.sub(rf"^{key} = .*$", f"{key} = {val}", cfg, flags=re.M)
    (dst / "syncaudit.config.toml").write_text(cfg)
    try:
        cli.main()
    except SystemExit:
        pass

    n = 40
    print(f"\nhost syncs (cuda tensors), over the run, divided by {n} steps:")
    for k, v in CALLS.most_common():
        print(f"  {k:8} {v / n:9.2f}/step   total {v}")
    print("\ntop frames:")
    for k, v in FRAMES.most_common(20):
        print(f"  {v / n:9.2f}/step  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
