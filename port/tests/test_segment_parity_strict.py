"""The segmented forward's LOSS pin: |Δloss| <= 1e-6 against the one-shot run.

`tests/test_longctx.py::test_segmented_logits_are_the_one_shot_logits` already
holds the parity claim at **1e-4 absolute on logits AND loss**. The logits half
is right at that tolerance and must stay: the measured move is 1.07–1.16e-6
absolute, i.e. **2e-7 relative** to a 5.6 logit scale — pure BLAS reassociation,
so a 1e-6 absolute logit gate would be a flake generator, not an invariant.

The LOSS is a different contract. It is the scalar every BPB number on record
is read from, and it is a sum over the same per-byte CE *before* the logits'
reassociation, so it is exact (measured: 0.0 at 4 segments, 4.77e-7 at 2). This
file pins the tighter claim the report quotes, and pins the `loop_segments`
counter so a segmented call that silently took the one-shot branch cannot pass
by being trivially equal.

CPU, fp32, the same 64-byte fixture as `test_longctx.py`.
"""

from __future__ import annotations

import pytest
from conftest import SEQ, tiny_model

from dormouse.instrument import Counters

#: Two legal cuts of the 64-byte window: 4 segments and 2.
CUTS = [16, 32]


@pytest.mark.parametrize("seg", CUTS)
def test_segmented_loss_is_the_one_shot_loss_to_1e6(seg, byte_tensor):
    import torch

    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()  # the SAME fixture `test_longctx.py` parity uses

    _, ref_loss, _ = model(ids, targets, counters=Counters())
    _, loss, c = model(ids, targets, counters=Counters(), segment_len=seg)
    dl = abs(float(loss.detach()) - float(ref_loss.detach()))
    print(f"[segment parity] seg={seg} segments={c['loop_segments']} dloss={dl:.3e}")

    assert c["loop_segments"] == SEQ // seg, "the segmented branch did not run"
    assert dl <= 1e-6, f"segmented loss differs by {dl:.3e} (gate 1e-6)"
