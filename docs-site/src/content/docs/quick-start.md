---
title: Quick start
description: Train the live recipe, run the gates, export the file — every command that matters.
---

Everything runs on one machine. A shape-check arm takes about twenty-five
minutes; a full pass over the corpus takes a night.

## Install

```bash
git clone https://github.com/sehaxe/dormouse && cd dormouse/port
uv sync                      # python 3.12 + torch + triton, pinned
```

The GPU must be sm_120 or newer (RTX 50-series, B-series). A CPU-only box runs
the whole test suite; it just cannot train.

## Train

```bash
PYTHONPATH=src .venv/bin/python -m dormouse.cli train configs/arch_gated_swiglu.toml \
  --data ../data_local/best/fwedu_train.bin \
  --eval ../data_local/best/fwedu_eval.bin \
  --set mix_plan=configs/mix_refetch.toml \
  --set steps=4880 --set batch=24 --set seq_len=1024 \
  --set opt=mix --set muon_lr_mult=12 --set lr=3e-4 \
  --set ns_dtype=bf16 --set muon_state_dtype=bf16 \
  --set dense_qat=ternary --set factor_qat_anneal_steps=2000
```

That is the 120 MB shape-check arm. Drop `steps` to get the full corpus run.

:::caution[Muon's scale]
`muon_lr_mult=12` is not optional. Muon's update is ~12× smaller than AdamW's on
these shapes; without the multiplier the matrix half of the model trains at 8–52%
of the intended step and every number is garbage. The CLI prints a banner about
it at every start.
:::

:::note[The format is a training flag]
`dense_qat=ternary` puts the *forward pass* on three levels from the first step,
and `factor_qat_anneal_steps` ramps the projection from full precision to hard
ternary over that window. The optimizer still holds one bf16 master per weight —
a three-valued number cannot absorb a gradient — but the network always computes
in ternary. Nothing is quantized after training.
:::

## Gate

```bash
cd .. && bash scripts/gate_fast.sh        # ruff, dead knobs, doc refs
cd port && PYTHONPATH=src .venv/bin/python -m pytest -q -m "not slow and not pins"
```

- `-m "not pins"` — 2500 tests, the ones that gate a commit.
- `-m pins` — 82 exact-fingerprint tests. They compare sha256 of logits and
  state dicts, so they break on any legitimate change; they report, they do not
  gate a PR.

## Export

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
from dormouse.config import load_preset
from dormouse.model import DormouseModel
from dormouse.model.export import export_qat

sd = torch.load('checkpoints/run/port.pt', map_location='cpu', weights_only=False)
model = DormouseModel(load_preset('configs/arch_gated_swiglu.toml'))
model.load_state_dict(sd.get('model', sd))
report = export_qat(model, 'artifact.pt', mode='ternary')
print(report.table())
PY
```

The export packs each weight to 2-bit codes plus one fp16 scale per tensor.
Reloading reproduces the trained quantized network to 0.15% of the quantizer's
own error, so the file is the network rather than an approximation of it.
