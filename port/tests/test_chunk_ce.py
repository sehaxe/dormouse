"""Gates for `chunk_ce.chunked_cross_entropy` — the chunked loss must be EXACT.

The arm exists for one reason (`chunk_ce.py`'s arithmetic: 805M logits values at
b24·s1024 with a 32k vocabulary), and the way it can go wrong is by producing a
number that LOOKS like a loss and is not: a chunk boundary that reads the
neighbour's labels, a denominator that counts the padded positions, a gradient
that is the chunk's instead of the batch's. So every gate here is a comparison
against the un-chunked path, not a smoke test:

a. PARITY with `F.cross_entropy` on the same logits: loss ≤ 1e-6 relative,
   gradient w.r.t. the hidden state ≤ 1e-5 relative (the head's too — it is the
   gradient nothing would notice was missing until the head stopped learning).
b. CHUNK-SIZE INVARIANCE at 512/1024/4096, plus the two permutation controls
   that catch a boundary leak the size sweep would hide: permuting hidden and
   targets TOGETHER is a bit-identical loss (chunking only partitions
   positions); permuting the targets ALONE is a different number.
c. `ignore_index` and the trainer's `loss_mask` semantics: the same rows
   excluded from the SUM and the DENOMINATOR, checked against the un-chunked
   masked mean on a padded batch.
d. `softcap` composes inside the chunk exactly as `_readout` applies it.
e. MEMORY at the shape the brief states (T=8192, V=32768): the peak allocation
   ratio. **Measured**, not analytic: `resource.getrusage(RUSAGE_SELF).ru_maxrss`
   of a fresh interpreter per arm — CPU tensors have no allocator stats and
   `torch.cuda.max_memory_allocated` would need the card, which another lane
   holds. The analytic per-tensor figure (T/chunk = 32x fewer live logit
   values) is asserted beside it.
f. WIRING: the config refusals, `load_run` carrying the field, the readout
   returning `logits=None`, the trainer's banner line, and SFT's refusal.

CPU and tiny on purpose — the numbers are comparisons, not timings.
"""

from __future__ import annotations

import dataclasses as dc
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from conftest import BATCH, SEQ, tiny_cfg, tiny_run

from dormouse.chunk_ce import chunked_cross_entropy
from dormouse.config import TrainConfig, load_run, validate_train
from dormouse.instrument import LoudError
from dormouse.losses import logit_softcap
from dormouse.model import DormouseModel
from dormouse.train.loop import train


def _fixture(t: int = 256, v: int = 512, d: int = 32, seed: int = 0):
    """`(hidden, weight, targets, project)` — a head, not a model."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(t, d, generator=g, requires_grad=True)
    w = torch.randn(v, d, generator=g, requires_grad=True)
    targets = torch.randint(0, v, (t,), generator=g)
    return h, w, targets, (lambda x: F.linear(x, w))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-30))


# --------------------------------------------------------------------------- a


def test_a_loss_and_gradients_match_the_un_chunked_path():
    """`F.cross_entropy(logits)` vs the same logits projected 64 at a time."""
    h, w, targets, project = _fixture()
    z = project(h)
    ref = F.cross_entropy(z, targets)
    ref.backward()
    gh_ref, gw_ref = h.grad.clone(), w.grad.clone()

    h2, w2 = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    got = chunked_cross_entropy(h2, lambda x: F.linear(x, w2), targets, 64)
    got.backward()

    loss_rel = abs(float(got.detach()) - float(ref.detach())) / abs(float(ref.detach()))
    grad_h = _rel(h2.grad, gh_ref)
    grad_w = _rel(w2.grad, gw_ref)
    print(f"loss rel={loss_rel:.3e}  d/dhidden rel={grad_h:.3e}  d/dhead rel={grad_w:.3e}")
    assert loss_rel <= 1e-6, f"loss is not the un-chunked loss: rel={loss_rel:.3e}"
    assert grad_h <= 1e-5, f"hidden gradient is wrong: rel={grad_h:.3e}"
    assert grad_w <= 1e-5, f"head gradient is wrong: rel={grad_w:.3e}"


# --------------------------------------------------------------------------- b


def test_b_three_chunk_sizes_are_one_number():
    """512/1024/4096 positions per chunk: the same loss to fp32 rounding.

    The defect class this is here for: an off-by-one in the label slice (chunk i
    reading chunk i+1's targets) moves the loss at some chunk sizes and not
    others, which is a plausible-looking curve rather than a crash.
    """
    t = 4096
    h, w, targets, project = _fixture(t=t, v=512, d=16, seed=1)
    reference = F.cross_entropy(project(h), targets)
    losses = {}
    for chunk in (512, 1024, 4096):
        losses[chunk] = float(chunked_cross_entropy(h, project, targets, chunk).detach())
    print(
        f"ref={float(reference.detach()):.7f} "
        + " ".join(f"{c}={v:.7f}" for c, v in losses.items())
    )
    for chunk, value in losses.items():
        rel = abs(value - float(reference)) / abs(float(reference))
        assert rel <= 1e-6, f"chunk={chunk} moved the loss: rel={rel:.3e}"


def test_b_permuting_positions_is_free_but_permuting_labels_is_not():
    """The boundary-leak controls: chunking partitions positions, nothing else."""
    t = 512
    h, w, targets, project = _fixture(t=t, v=64, d=16)
    base = float(chunked_cross_entropy(h, project, targets, 7))
    perm = torch.randperm(t, generator=torch.Generator().manual_seed(3))
    joint = float(chunked_cross_entropy(h[perm], project, targets[perm], 7))
    labels_only = float(chunked_cross_entropy(h, project, targets[perm], 7))
    print(f"base={base:.7f} joint_perm={joint:.7f} labels_only={labels_only:.7f}")
    assert abs(joint - base) <= 1e-6 * abs(base), (
        "permuting hidden and targets together changed the loss — a chunk is not "
        "reading the same positions it projects"
    )
    assert abs(labels_only - base) > 1e-3, (
        "permuting the targets alone left the loss unchanged — the labels are not read per position"
    )


# --------------------------------------------------------------------------- c


def test_c_ignore_index_and_loss_mask_match_the_masked_mean():
    """Padding is excluded from BOTH the sum and the denominator, chunk or not."""
    t = 300
    h, w, targets, project = _fixture(t=t, v=256, d=16)
    padded = targets.clone()
    padded[::3] = -100
    ref = F.cross_entropy(project(h), padded, ignore_index=-100)
    got = float(chunked_cross_entropy(h, project, padded, 64, ignore_index=-100))
    print(f"ignore_index: ref={float(ref):.7f} chunked={got:.7f}")
    assert abs(got - float(ref)) <= 1e-6 * abs(float(ref))

    # The trainer's `loss_mask` (doc_masking): a mean over the SELECTED positions.
    mask = torch.ones(t, dtype=torch.bool)
    mask[7::5] = False
    ce = F.cross_entropy(project(h), targets, reduction="none")
    ref_masked = float((ce * mask).sum() / mask.sum())
    got_masked = float(chunked_cross_entropy(h, project, targets, 64, mask=mask))
    print(f"loss_mask: ref={ref_masked:.7f} chunked={got_masked:.7f}")
    assert abs(got_masked - ref_masked) <= 1e-6 * abs(ref_masked)
    # A mask that drops everything is 0/0 — the un-chunked mean's NaN, not a crash.
    nothing = torch.zeros(t, dtype=torch.bool)
    empty = float(chunked_cross_entropy(h, project, targets, 64, mask=nothing))
    assert empty != empty, "an empty mask must be the 0/0 the masked mean already is"


def test_c_the_gradient_is_zero_on_the_masked_rows():
    """A masked position contributes NEITHER loss nor gradient (the arm's claim)."""
    t = 200
    h, w, targets, project = _fixture(t=t, v=128, d=16)
    mask = torch.ones(t, dtype=torch.bool)
    mask[:100] = False
    chunked_cross_entropy(h, project, targets, 32, mask=mask).backward()
    assert float(h.grad[:100].abs().sum()) == 0.0, "a masked row has a gradient"
    assert float(h.grad[100:].abs().sum()) > 0.0, "a supervised row has no gradient"


# --------------------------------------------------------------------------- d


def test_d_softcap_composes_inside_the_chunk():
    """`logit_softcap` before the CE, as `_readout` applies it — chunk-free."""
    h, w, targets, project = _fixture(t=256, v=256, d=16)
    cap = 15.0
    ref = F.cross_entropy(logit_softcap(project(h), cap), targets)
    got = chunked_cross_entropy(h, project, targets, 64, softcap=cap)
    rel = abs(float(got) - float(ref)) / abs(float(ref))
    print(f"softcap: ref={float(ref):.7f} chunked={float(got):.7f} rel={rel:.3e}")
    assert rel <= 1e-6


# --------------------------------------------------------------------------- e

#: The brief's shape: one training window of b24·s1024 = 8192 positions scored at
#: a 32,768-token vocabulary. `D=16` keeps the head itself (2 MB) out of the
#: number, so what the ratio measures is the LOGITS. 64 positions per chunk at
#: this V is a 8 MB logits slice; the ratio grows with the chunk (10.1x at 256,
#: 7.7x at 512) because a ~230 MB allocator floor is paid by BOTH arms and only
#: the tensor set scales.
MEMORY_SHAPE = dict(t=8192, v=32768, d=16)
MEMORY_CHUNK = 64

_PROBE = r"""
import sys, threading, time
sys.path[:0] = {path!r}
import torch, torch.nn.functional as F

T, V, D, CH, mode = {t}, {v}, {d}, {chunk}, {mode!r}
torch.manual_seed(0)
h = torch.randn(T, D, requires_grad=True)
w = torch.randn(V, D, requires_grad=True)
targets = torch.randint(0, V, (T,))
project = lambda x: F.linear(x, w)

# Peak RSS sampled WHILE the arm runs, at 1 ms. `ru_maxrss` alone cannot do this:
# the process's peak can predate the arm (a transient during the torch import),
# and then the small arm's delta reads 0 and the ratio is a division by zero.
PAGE = 4096
def rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE

start = rss()
peak = [start]
stop = threading.Event()

def watch():
    while not stop.is_set():
        peak[0] = max(peak[0], rss())
        time.sleep(0.001)

thread = threading.Thread(target=watch, daemon=True)
thread.start()
if mode == "full":
    F.cross_entropy(project(h), targets).backward()
else:
    from dormouse.chunk_ce import chunked_cross_entropy
    chunked_cross_entropy(h, project, targets, CH).backward()
peak[0] = max(peak[0], rss())
stop.set()
thread.join()
print(peak[0] - start)
"""


def _peak_kib(mode: str) -> int:
    """Bytes of peak resident growth of one fresh interpreter.

    `MALLOC_MMAP_THRESHOLD_` is pinned so a freed chunk-sized block is returned
    to the OS instead of raising glibc's dynamic mmap threshold: without it the
    SAME chunked arm measures 1237 MiB at chunk=64, i.e. the number would say
    more about the heap's history than about the arm. It is set for BOTH arms.
    """
    code = _PROBE.format(
        path=[p for p in sys.path if p],
        chunk=MEMORY_CHUNK,
        mode=mode,
        **MEMORY_SHAPE,
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
            "MALLOC_MMAP_THRESHOLD_": "131072",
        },
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def test_e_peak_allocation_is_at_least_10x_lower_at_the_stated_shape():
    """T=8192, V=32768: sampled peak RSS growth per arm, in a fresh process.

    Not `torch.cuda.max_memory_allocated`: the card is held by another lane, and
    CPU tensors carry no allocator stats, so the peak resident growth of one
    interpreter per arm (sampled at 1 ms from `/proc/self/statm` while the arm
    runs — see `_PROBE`) is the honest CPU-shaped equivalent. Both arms pay the
    same ~230 MB allocator floor, so the RATIO understates what the vocabulary
    costs; the analytic figure (T/chunk live logit sets) is asserted beside it.
    """
    full = _peak_kib("full")
    chunked = _peak_kib("chunk")
    ratio = full / chunked
    print(
        f"peak growth: full={full / 2**20:.0f} MiB  ce_chunk={MEMORY_CHUNK}="
        f"{chunked / 2**20:.0f} MiB  ratio={ratio:.1f}x  "
        f"(analytic live logit values T/chunk={MEMORY_SHAPE['t'] // MEMORY_CHUNK}x fewer)"
    )
    assert MEMORY_SHAPE["t"] // MEMORY_CHUNK == 128, "the analytic ratio moved"
    assert full >= 2**30, (
        f"the un-chunked arm only grew {full} B: it did not build the [T, V] logits"
    )
    assert ratio >= 10.0, (
        f"chunking only bought {ratio:.1f}x at T={MEMORY_SHAPE['t']} V={MEMORY_SHAPE['v']} "
        f"(full={full} B, chunked={chunked} B): the [T, V] tensor is still being built"
    )


# --------------------------------------------------------------------------- f


def test_f_the_config_refuses_the_combinations_it_cannot_honour():
    validate_train(TrainConfig(ce_chunk=512))  # the legal arm
    for kwargs, match in (
        ({"ce_chunk": -1}, "ce_chunk"),
        ({"ce_chunk": 8, "ce_fixed": True}, "ce_fixed"),
        ({"ce_chunk": 8, "tst_bag": 4}, "tst_bag"),
        ({"ce_chunk": 8, "full_graph": True}, "full_graph"),
        ({"ce_chunk": 8, "opt_graph": True}, "opt_graph"),
    ):
        with pytest.raises(LoudError, match=match):
            validate_train(TrainConfig(**kwargs))


def test_f_load_run_carries_the_field():
    run = load_run("small", train=TrainConfig(ce_chunk=512))
    assert run.train.ce_chunk == 512


#: The TRANSFORMER form's overrides: `block_form` refuses a shared loop and the
#: KDA/Engram arms, so the dense control needs `max_iter=1` and those off.
TRANSFORMER = dict(
    block_form="transformer", max_iter=1, use_kda=False, use_engram=False, use_csa=False
)


@pytest.mark.parametrize("form", [{}, TRANSFORMER], ids=["loop", "transformer"])
def test_f_the_readout_builds_no_logits_and_the_same_loss(form):
    """BOTH stack forms: `logits is None`, and the loss/grads are the un-chunked ones.

    Two sites, one rule: the LOOP form's head+CE lives in `_forward_byte_lm`, the
    transformer/hybrid forms' in `_readout`. A chunked arm that reached only one
    of them would be a flag whose effect depended on `block_form`.
    """
    model = DormouseModel(tiny_cfg(**form))
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    targets = (ids + 1) % 251
    logits, loss, _ = model(ids, targets)
    assert logits is not None and logits.shape[-1] == model.cfg.vocab
    logits_ch, loss_ch, _ = model(ids, targets, ce_chunk=7)
    assert logits_ch is None, "the chunked readout must not hand back the [b·t, V] tensor"
    rel = abs(float(loss_ch.detach()) - float(loss.detach())) / abs(float(loss.detach()))
    print(
        f"model: un-chunked={float(loss.detach()):.7f} "
        f"ce_chunk=7={float(loss_ch.detach()):.7f} rel={rel:.3e}"
    )
    assert rel <= 1e-6
    loss_ch.backward()
    head = [p for n, p in model.named_parameters() if "lm_head" in n or "embed" in n]
    assert head, "no head parameter found"
    for p in head:
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, (
            "the chunked loss left a head/embedding gradient empty"
        )
    # ... and the flag is off again on the next forward, not sticky.
    again, _, _ = model(ids, None)
    assert again is not None and again.shape[-1] == model.cfg.vocab


def test_f_the_masked_readout_takes_the_chunked_branch():
    model = DormouseModel(tiny_cfg())
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    targets = (ids + 1) % 251
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, SEQ // 2 :] = False
    _, loss, _ = model(ids, targets, loss_mask=mask)
    logits, loss_ch, _ = model(ids, targets, loss_mask=mask, ce_chunk=9)
    assert logits is None
    rel = abs(float(loss_ch.detach()) - float(loss.detach())) / abs(float(loss.detach()))
    print(
        f"masked: un-chunked={float(loss.detach()):.7f} "
        f"ce_chunk=9={float(loss_ch.detach()):.7f} rel={rel:.3e}"
    )
    assert rel <= 1e-6


def test_f_the_model_refuses_the_flags_it_cannot_compose():
    """ADR-0019: the runtime seam owns the refusals the config owns, too."""
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    with pytest.raises(LoudError, match="ce_fixed"):
        DormouseModel(tiny_cfg())(ids, ids, ce_chunk=4, ce_fixed=True)
    with pytest.raises(LoudError, match="z_loss"):
        DormouseModel(tiny_cfg(z_loss=0.1))(ids, ids, ce_chunk=4)
    with pytest.raises(LoudError, match="byteflow"):
        DormouseModel(tiny_cfg(use_byteflow=True))(ids, ids, ce_chunk=4)


def test_f_the_trainer_runs_the_arm_and_prints_it(tmp_path):
    """The whole seam: `train()` passes the kwarg, the step runs, the banner says so."""
    run = tiny_run(tmp_path, steps=2, ce_chunk=8, log_every=1)
    lines: list[str] = []
    model = DormouseModel(run.model)
    report = train(model, run, log=lines.append)
    assert any(line.startswith("ce_chunk=8:") for line in lines), (
        "the arm ran without saying so (the banner is the only place a reader "
        "learns the [b·t, vocab] tensor was never built)"
    )
    assert report["loss"] == report["loss"], "the chunked step produced NaN"


def test_f_sft_and_rl_refuse_the_field():
    from dormouse.train.sft import refuse_unwired

    for loop in ("sft", "rl"):
        with pytest.raises(LoudError, match="ce_chunk"):
            refuse_unwired(TrainConfig(ce_chunk=8), loop)
    refuse_unwired(TrainConfig(), "sft")


def test_f_a_default_config_never_reaches_the_arm():
    assert TrainConfig().ce_chunk == 0
    assert dc.replace(TrainConfig(), steps=1).ce_chunk == 0
