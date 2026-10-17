"""Segmented training with carried KDA state — the long-context enabler.

The claim this file gates: **segmentation is a MEMORY decision, not a model
change.** The training forward cuts the `[b, seq_len]` window into
`segment_len`-byte segments, carries the KDA state across the cuts (per
iteration slot — the loop calls the attention arm once per iteration) and lets
the gradient ride that state, so a window longer than one forward fits in VRAM
can be trained without truncated BPTT. The logits must therefore BE the one-shot
logits to fp32 round-off, and the state must be demonstrably load-bearing.

Every gate here is CPU and tiny: what has to be true is the SEMANTICS (the
logits match, the carry is real, the flag refuses what it cannot ride), and none
of it needs a GPU. The FUSED arm's own carry
(`fla.ops.kda.chunk_kda(initial_state=..., output_final_state=True)`,
`chunk.py:185-186`) is CUDA-only, so the one gate that runs it is skipped on a
CPU box — the SDPA fallback carries its exact K/V context instead, which is what
makes the CPU gates meaningful rather than a second implementation's opinion.

Read the refusals as the port's rule 1.1: a `segment_len` that quietly did
nothing, or that cut the window in a way that changed the objective or the
attention pattern, would show the flag on while the run reported it.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest
import torch
from conftest import SEQ, tiny_cfg, tiny_model, tiny_run

from dormouse.config import load_run
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.kda import KDAAttention
from dormouse.train import train

#: Two legal cuts of the 64-byte window (4 and 2 segments).
SEG, HALF = 16, 32

#: `find_spec`, not an import: the fused arm's own gate (the port's other CUDA
#: tests use the same pattern — a warmup call at collection time would be GPU
#: work in the wrong place).
_FLA_INSTALLED = importlib.util.find_spec("fla") is not None
_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and _FLA_INSTALLED),
    reason="needs a CUDA device AND FLA (`uv sync --extra gpu`)",
)


def _blank_second_segment(ids: torch.Tensor, seg: int = SEG) -> torch.Tensor:
    """The same window with segment 1 replaced by one constant byte.

    What is left for segment 1 to know about segment 0 is the carried state and
    nothing else — which is exactly what the load-bearing gate measures.
    """
    out = ids.clone()
    out[:, seg:] = 0
    return out


# --------------------------------------------------------------------- off is off


def test_off_is_byte_identical_and_never_enters_the_branch(byte_tensor):
    """`segment_len=0` (the default) is the one-shot forward, bit for bit.

    The flag's default must not move a number: `torch.equal`, not `allclose`,
    and the segmented branch must not be entered at all (`loop_segments` is the
    counter that only the segmented path bumps).
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    logits_a, loss_a, c_a = model(ids, targets)
    logits_b, loss_b, c_b = model(ids, targets, segment_len=0)
    assert torch.equal(logits_a, logits_b), "the default path must not have moved"
    assert torch.equal(loss_a, loss_b)
    assert "loop_segments" not in c_a and "loop_segments" not in c_b
    assert c_b["iter"] == model.cfg.max_iter
    print(f"[longctx off] logits identical, counters={c_b.line()}")


def test_one_segment_reproduces_the_one_shot_forward_bit_for_bit(byte_tensor):
    """The pin the default rests on: `segment_len = t` IS the one-shot forward.

    A single full-length segment takes the same branch every other segment does
    (`state=None` → the arm's `is_causal` call, `drop=0` → the arm's own memory
    read, one `checkpoint` around one window), so a difference here would mean
    the segmented machinery changed the model before it ever cut anything. The
    config layer refuses `segment_len >= seq_len` for a RUN (it buys nothing);
    this calls the loop block directly, which is where the degenerate case has
    to be exact.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    h0 = model.embed(ids)
    a_out, a_rec, a_c = model.loop(h0, ids, targets, model.lm_head, counters=Counters())
    b_out, b_rec, b_c = model.loop(
        h0, ids, targets, model.lm_head, counters=Counters(), segment_len=SEQ
    )
    assert torch.equal(a_out, b_out), "the readout must be the same tensor values"
    assert torch.equal(a_rec, b_rec), "the Σ per-iteration CE must be the same scalar"
    assert b_c["loop_segments"] == 1 and b_c["iter"] == a_c["iter"] == model.cfg.max_iter
    print(f"[longctx 1seg] equal; {b_c.line()}")


# ------------------------------------------------------- (a) the parity gate


@pytest.mark.parametrize("seg", [SEG, HALF])
def test_segmented_logits_are_the_one_shot_logits(seg, byte_tensor):
    """The logits of a segmented forward == the one-shot logits (fp32, 1e-4 abs).

    This is the whole license for the feature: the trainer's loss code, the
    eval's meaning and every number on record stay valid because the function
    did not change, only where its activations live.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    ref_logits, ref_loss, _ = model(ids, targets)
    logits, loss, c = model(ids, targets, segment_len=seg)
    assert logits.shape == ref_logits.shape == (ids.numel(), model.cfg.vocab)
    d = float((logits - ref_logits).abs().max())
    dl = abs(float(loss) - float(ref_loss))
    print(
        f"[longctx parity] seg={seg} segments={c['loop_segments']} "
        f"max|dlogits|={d:.3e} dloss={dl:.3e}"
    )
    assert c["loop_segments"] == SEQ // seg
    assert d <= 1e-4, f"segmented logits differ by {d}"
    assert dl <= 1e-4, f"segmented loss differs by {dl}"


# -------------------------------------------------- (b) gradient into segment 0


def test_gradient_reaches_the_first_segment(byte_tensor):
    """Segment 0 is trained, not a warm-up: its bytes move the loss and it gets grad.

    Both halves matter. The loss moving proves the window is one objective (the
    segmented forward is not "segment 1's loss with segment 0 prefixed"); the
    non-zero gradient on segment 0's own embedding rows proves the backward
    reaches the first checkpointed window through the carried state.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    _, loss_a, _ = model(ids, targets, segment_len=SEG)
    other = ids.clone()
    other[:, :SEG] = (other[:, :SEG] + 7) % 251
    _, loss_b, _ = model(other, targets, segment_len=SEG)
    moved = abs(float(loss_a) - float(loss_b))
    print(f"[longctx seg0] loss moves {moved:.3e} when segment 0's bytes change")
    assert moved > 1e-6, "segment 0's bytes must move the loss"

    model.zero_grad()
    _, loss, _ = model(ids, targets, segment_len=SEG)
    loss.backward()
    grad = model.embed.weight.grad
    rows = ids[0, :SEG].unique()
    assert grad is not None and bool(torch.isfinite(grad).all())
    g0 = float(grad[rows].abs().sum())
    print(f"[longctx seg0] |grad| on segment 0's {len(rows)} embedding rows = {g0:.3e}")
    assert g0 > 0.0, "no gradient reached segment 0's inputs"


# ------------------------------------------------------ (c) the carry is real


def test_a_blank_second_segment_still_reads_the_carried_state(byte_tensor, monkeypatch):
    """With segment 1 blank, the carried run differs from a no-carry run.

    The no-carry arm is built by patching the arm's stateful call to always be
    handed `state=None` — every segment then starts a fresh sequence, which is
    exactly the truncated-BPTT shape this feature exists to replace. If the
    carried state were decorative, the two would be the same logits.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    blank = _blank_second_segment(ids)
    carried = model(blank, targets, segment_len=SEG)[0].reshape(*ids.shape, -1)

    real = KDAAttention.forward_state

    def dropped(self, x, state=None, counters=None, *, carry=False):
        return real(self, x, None, counters, carry=carry)

    monkeypatch.setattr(KDAAttention, "forward_state", dropped)
    truncated = model(blank, targets, segment_len=SEG)[0].reshape(*ids.shape, -1)
    monkeypatch.undo()

    delta = float((carried[:, SEG:] - truncated[:, SEG:]).abs().max())
    print(f"[longctx carry] blank segment 1: carried vs no-carry max|d|={delta:.3e}")
    assert delta > 1e-3, "the carried state must change segment 1's logits"


def test_segment_zero_moves_segment_one_through_the_carry(byte_tensor):
    """Perturb segment 0 ONLY: segment 1's inputs are identical, its logits move.

    The only path between the two is the carried state, so this is the direct
    statement of "context from segment 0 reaches segment 1" — and it is the
    property truncated BPTT does not have.
    """
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    other = ids.clone()
    other[:, :SEG] = (other[:, :SEG] + 13) % 251
    a = model(ids, targets, segment_len=SEG)[0].reshape(*ids.shape, -1)
    b = model(other, targets, segment_len=SEG)[0].reshape(*ids.shape, -1)
    delta = float((a[:, SEG:] - b[:, SEG:]).abs().max())
    print(f"[longctx carry] segment 1 logits move {delta:.3e} on a segment-0-only change")
    assert delta > 1e-4, "segment 0 must reach segment 1 through the carried state"


def test_segmented_is_exact_on_the_host_tables_too(byte_tensor):
    """The RAM-offload substrate takes the same left-context slice, and stays exact.

    `_memory` hands the extended ids to the arm's PUBLIC `forward`, which is the
    same entry point on both substrates (`_forward_host` differs only in how the
    `emb` rows are obtained, and both run one `_read`), so the segmented host
    forward must be the segmented in-VRAM forward — and `engram_ram` is the flag
    the long-context recipe actually launches with, which is why this is not an
    academic combination.
    """
    from dormouse.model.engram import enable_host_tables

    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    with torch.no_grad():
        ref = model(ids, targets)[0]
        seg_vram = model(ids, targets, segment_len=SEG)[0]
        assert enable_host_tables(model) == 1
        seg_host = model(ids, targets, segment_len=SEG)[0]
        one_host = model(ids, targets)[0]
    d = float((seg_host - ref).abs().max())
    d1 = float((one_host - seg_host).abs().max())
    print(
        f"[longctx host] seg_host vs seg_vram equal; "
        f"seg_host vs one-shot max|d|={d:.3e}; one-shot host vs segmented host max|d|={d1:.3e}"
    )
    assert torch.equal(seg_vram, seg_host), "the substrate must not be a variable"
    assert d <= 1e-4, "the segmented host forward must be the one-shot forward's values"
    assert d1 <= 1e-4, "and the host substrate must not change the segmented parity"


# ------------------------------------------------------------ the trainer seam


def test_the_trainer_passes_the_flag_and_counts_exactly_once(tmp_path):
    """`train.segment_len` reaches the forward, and the counters stay exact.

    The counter is the load-bearing half: a segment's body runs inside
    `checkpoint(use_reentrant=False)`, which runs it AGAIN in the backward, so
    counting into the run's own object would report `2·steps·T·segments`
    iterations on the log line. `iter` here is the exact count, which is what
    fails if that ever regresses.
    """
    run = tiny_run(tmp_path, segment_len=HALF, steps=2, log_every=2)
    # `tiny_cfg` ships the sparse dispatch, which `segment_len` refuses (below);
    # `small` — the long-context recipe's own preset — is the dense blend.
    run = dataclasses.replace(run, model=tiny_cfg(moe_topk=0))
    lines: list[str] = []
    model = DormouseModel(run.model)
    report = train(model, run, log=lines.append)
    assert torch.isfinite(torch.tensor(report["loss"])), lines
    expected = 2 * run.model.max_iter * (SEQ // HALF)
    assert report["counters"]["iter"] == expected, report["counters"].line()
    assert report["counters"]["loop_segments"] == 2 * (SEQ // HALF)
    logged = [ln for ln in lines if ln.startswith("step=")]
    assert logged and f"loop_segments={2 * (SEQ // HALF)}" in logged[-1], logged
    print(f"[longctx train] {logged[-1]}")


# ----------------------------------------------------------------- (d) refusals


def _refused(**over) -> str:
    """`load_run` with the overrides, returning the refusal's message."""
    model_over = over.pop("model_overrides", None)
    with pytest.raises(LoudError) as e:
        load_run("small", train_overrides=over, model_overrides=model_over)
    return str(e.value)


def test_segment_len_at_or_above_the_window_is_refused():
    """Pointless: the window is one segment, so the flag buys nothing."""
    msg = _refused(seq_len=512, segment_len=512)
    assert "segment_len=512" in msg and "seq_len=512" in msg and "one segment" in msg
    assert "Escape" in msg
    msg = _refused(seq_len=512, segment_len=1024)
    assert "segment_len=1024" in msg and "Escape" in msg


def test_a_negative_segment_len_is_refused():
    """0 is the OFF value; a negative is a typo, never a silent off."""
    msg = _refused(seq_len=512, segment_len=-64)
    assert "segment_len=-64" in msg and "0 is the OFF value" in msg and "Escape" in msg


def test_a_non_divisor_is_refused_rather_than_given_a_short_tail():
    """An uneven cut would re-weight the loss — refused, not rounded down.

    The per-iteration CE is a MEAN per segment, so a short last segment's bytes
    would count for more than the others' and the segmented run would be a
    different objective wearing a memory flag. Allowing it would need a weighted
    mean; refusing it is the honest form of "same loss".
    """
    msg = _refused(seq_len=512, segment_len=300)
    assert "segment_len=300" in msg and "does not divide seq_len=512" in msg
    assert "MEAN per segment" in msg and "Escape" in msg


def test_use_byteflow_is_refused():
    """The channel runs the loop over K patch latents: no byte window to cut."""
    msg = _refused(
        seq_len=512,
        segment_len=128,
        model_overrides={
            "use_byteflow": True,
            "bf16": False,
            "use_engram": False,
            "byteflow_k_tokens": 8,
        },
    )
    assert "segment_len=128" in msg and "use_byteflow" in msg and "Escape" in msg


def test_use_msa_is_refused():
    """The block-sparse arm selects blocks INSIDE a window: a cut changes it."""
    msg = _refused(seq_len=512, segment_len=128, model_overrides={"use_msa": True})
    assert "segment_len=128" in msg and "use_msa" in msg and "attention" in msg


def test_rand_depth_is_refused():
    """Two unvalidated axes in one run: the result would be attributable to neither."""
    msg = _refused(seq_len=512, segment_len=128, rand_depth=True)
    assert "segment_len=128" in msg and "rand_depth" in msg and "Escape" in msg


def test_the_sparse_moe_dispatch_is_refused():
    """The finding this feature came with: the capacity drop is batch-compositional.

    `_permute` keeps `pos < ceil(moe_capacity · tokens · k / E)` rows per expert,
    so WHICH tokens get an expert depends on how many other tokens share their
    forward — cutting the window changes that set. Measured on the tiny shape:
    `moe_capacity=1.0` → 0.39 on the readout (a different model, silently),
    `moe_capacity=2.0` → no drop happened there, which is not a promise a
    collapsing router keeps. Hence a refusal and not a counter.
    """
    msg = _refused(seq_len=512, segment_len=128, model_overrides={"moe_topk": 2})
    assert "segment_len=128" in msg and "moe_topk=2" in msg
    assert "DROPS" in msg and "Escape" in msg


def test_the_sparse_moe_dispatch_is_refused_at_the_loop_seam_too(byte_tensor):
    """The same refusal where the segmentation is, for a hand-built model."""
    torch.manual_seed(0)
    model = tiny_model(moe_topk=2)
    ids, targets = byte_tensor()
    h0 = model.embed(ids)
    with pytest.raises(AssertionError) as e:
        model.loop(h0, ids, targets, model.lm_head, segment_len=SEG)
    assert "moe_topk>0" in str(e.value) and "Escape" in str(e.value)


def test_a_legal_cut_resolves():
    """The refusals are not a blanket ban: 1024 at 2048 is the recipe's own cut."""
    run = load_run("small", train_overrides={"seq_len": 512, "segment_len": 128})
    assert run.train.segment_len == 128 and run.train.seq_len == 512
    assert run.diff_keys(run) == []


def test_the_loop_block_refuses_a_segment_longer_than_its_window(byte_tensor):
    """The runtime seam: a hand-built call cannot silently run a pointless cut."""
    torch.manual_seed(0)
    model = tiny_model(moe_topk=0)
    ids, targets = byte_tensor()
    h0 = model.embed(ids)
    with pytest.raises(LoudError) as e:
        model.loop(h0, ids, targets, model.lm_head, segment_len=SEQ + 1)
    assert "segment_len=65" in str(e.value) and "Escape" in str(e.value)
    with pytest.raises(LoudError) as e:
        model.loop(h0, ids, targets, model.lm_head, segment_len=48)
    assert "does not divide the window 64" in str(e.value)


def test_use_msa_is_refused_at_the_loop_seam_too(byte_tensor):
    """`validate` refuses the pair for a run; the loop block holds it at runtime.

    A model built by hand never passes `validate`, so the refusal has to exist
    where the segmentation does — the `_forward_channel` aux assert is the same
    pattern.
    """
    torch.manual_seed(0)
    model = tiny_model(use_msa=True, msa_topk=2, msa_block=4)
    ids, targets = byte_tensor()
    h0 = model.embed(ids)
    with pytest.raises(AssertionError) as e:
        model.loop(h0, ids, targets, model.lm_head, segment_len=SEG)
    assert "use_msa" in str(e.value)


def test_the_model_refuses_segment_len_on_the_byteflow_shapes(byte_tensor):
    """Option A/B have no byte window: the refusal lives at the model seam too."""
    torch.manual_seed(0)
    cfg = tiny_cfg(
        use_byteflow=True,
        use_engram=False,
        byteflow_d_local=16,
        byteflow_d_global=16,
        byteflow_k_tokens=8,
        byteflow_e_layers=2,
        byteflow_g_layers=2,
        byteflow_heads_local=2,
        byteflow_heads_global=2,
        byteflow_w_local=8,
        byteflow_d_ff_local=32,
        byteflow_d_ff_global=32,
        byteflow_bins=8,
        byteflow_max_bytes=64,
    )
    model = DormouseModel(cfg)
    ids, targets = byte_tensor(2, 64)
    with pytest.raises(LoudError) as e:
        model(ids, targets, segment_len=SEG)
    assert "use_byteflow" in str(e.value) and "segment_len=16" in str(e.value)


# ------------------------------------------------------ the fused arm (CUDA only)


@_cuda
def test_the_fused_arm_carries_the_state_and_trains_the_first_segment():
    """The real arm, on a real card: same logits, and the carry is a graph edge.

    `chunk_kda` takes `initial_state` and returns `final_state` (fp32,
    `[b, H, head_dim, head_dim]`), and its backward takes `dht` and returns
    `dh0` — so this asserts the two things the CPU gate cannot: the fused
    segmented forward is the fused one-shot forward to fp32 round-off, and
    segment 0 receives gradient THROUGH the carried state.
    """
    dev = "cuda"
    torch.manual_seed(0)
    cfg = tiny_cfg(bf16=False, moe_topk=0)
    model = DormouseModel(cfg, device=dev)
    ids = (torch.arange(2 * SEQ, dtype=torch.long).view(2, SEQ) % 251).to(dev)
    targets = ((ids + 1) % 251).to(dev)
    ref, ref_loss, ref_c = model(ids, targets)
    assert ref_c["kda_fla"] == cfg.max_iter, "the fused arm must be the one running"
    logits, loss, c = model(ids, targets, segment_len=SEG)
    d = float((logits - ref).abs().max())
    print(f"[longctx cuda] seg={SEG} kda_fla={c['kda_fla']} max|dlogits|={d:.3e}")
    assert c["kda_fla"] == cfg.max_iter * (SEQ // SEG), c.line()
    assert c["loop_segments"] == SEQ // SEG
    assert d <= 1e-4 and abs(float(loss) - float(ref_loss)) <= 1e-4
    model.zero_grad()
    _, loss, _ = model(ids, targets, segment_len=SEG)
    loss.backward()
    grad = model.embed.weight.grad
    rows = ids[0, :SEG].unique()
    assert grad is not None and bool(torch.isfinite(grad).all())
    assert float(grad[rows].abs().sum()) > 0.0, "the fused carry is not a graph edge"
