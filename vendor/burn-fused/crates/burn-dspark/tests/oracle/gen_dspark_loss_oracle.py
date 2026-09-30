#!/usr/bin/env python3
"""Generate the DSpark-loss golden fixture by RUNNING DeepSeek's own loss.

ON A CLEAN VENV, this is the whole recipe, and every package in it is
REQUIRED -- the list is not "the obvious ones":

    $ uv venv --python 3.12 /tmp/opencode/oracle-venv
    $ VIRTUAL_ENV=/tmp/opencode/oracle-venv uv pip install \
          --index-url https://download.pytorch.org/whl/cpu torch numpy pyyaml transformers
    $ git clone https://github.com/deepseek-ai/DeepSpec.git   # need not be run again
    $ /tmp/opencode/oracle-venv/bin/python gen_dspark_loss_oracle.py \
          --deepspec /path/to/DeepSpec > ../fixtures/dspark_loss_oracle.txt

WHY `pyyaml` AND `transformers` ARE ON THAT LINE, since omitting them is the
one way this fails with an error that does not name them. `torch numpy` gets
you a torch and nothing else, and the import that follows walks the upstream
package: `deepspec/modeling/dspark/__init__.py` -> `common.py`, and from there
`deepspec/utils/config.py` (`import yaml`) and
`deepspec/modeling/dspark/gemma4/modeling.py` (`from transformers.cache_utils
import Cache`). Both raise a bare `ModuleNotFoundError: No module named
'yaml'` / `'transformers'` with a traceback that points at DeepSpec's import
graph and not at the missing dependency. Measured on this box 2026-09-30: with
`torch numpy` only, the generator dies before it emits a single case.

WHAT RAN, AND WHY IT IS TIER (a)
---------------------------------
`deepspec/modeling/dspark/loss.py::compute_dspark_loss` -- the loss DeepSeek
AI ships for DSpark, executed on CPU, unmodified, at

    repo    https://github.com/deepseek-ai/DeepSpec
    commit  005e03b81cec38b7da6399833d609ee89a2587f2  (2026-07-09)
    file    deepspec/modeling/dspark/loss.py          sha256 2e91efcaff780eec...
    paper   arXiv:2607.05147

The same file is vendored byte-identically at
`crates/dormouse-core/tests/oracle/dspark_loss.py` (verified by sha256, and
re-verified here at fetch time). The copy next to this generator exists so the
gate never needs a network.

WHY A PROCESS-GROUP IS NEEDED, because it is a real API fact
-------------------------------------------------------------
`compute_dspark_loss` calls `dist.get_world_size()` and `add_metric(...)`, and
`add_metric`'s default `dp_sum` reduction calls `dist.get_backend()`. So the
loss is NOT a pure function of its six tensors in a bare interpreter -- it needs
an initialised process group. A single-rank gloo group is enough and changes no
number: every `all_reduce` in `_all_reduce_loss_denominators` is over one rank.

Two dtype contracts the call site has to get right, both learned by running it:
  * `block_keep_mask` must be BOOL. `loss.py:137` does
    `valid_blocks = block_keep_mask & valid_pred_tokens`, and `&` is undefined
    for f32: torch raises `NotImplementedError: "bitwise_and_cpu" not
    implemented for 'Float'`. A float mask is not a slower path, it is a crash.
  * `eval_mask` must be FLOAT (it is multiplied by the decay weights).

THE TWO UPSTREAM NUMBERS, and what each is for
-----------------------------------------------
1. the official `compute_dspark_loss` scalar, at the OFFICIAL config's alphas
   (`config/dspark/dspark_qwen3_4b.py`: ce 0.1, l1 0.9, confidence 1.0,
   gamma 4.0) -- what `dspark_loss`'s three components must reconstruct;
2. the official per-term numerators/denominators, recomputed here from the
   vendored source so the test can attribute a disagreement to ONE term
   instead of only seeing a total.

AND THE OFFICIAL GRADIENTS, which are the only part of the reference a
value-level comparison is blind to
----------------------------------------------------------------------------
`compute_dspark_loss` returns a live scalar, so `loss.backward()` gives
DeepSeek's OWN `d/d(draft_logits)` and `d/d(confidence_pred)` by autograd, on
the same pinned source, with no transcription of ours anywhere in the path.
They are emitted as `grad_draft` / `grad_conf` and compared against ours in
`dspark_loss_gradients_agree_with_the_official_loss`.

This is the half of the reference a value oracle cannot see, and the reason
matters: `loss.py:145` is `confidence_targets = accept_rate_3d.detach()`. An
implementation that leaves that target attached computes a bit-identical
FORWARD loss and a drafter gradient that differs by 20-101%. A fixture of
values is not able to tell the two apart, so the fixture carries gradients.

SELF-VALIDATING, but NOT by finite differences, and the reason is the
single most useful thing this generator learned on 2026-09-30
----------------------------------------------------------------------------
A central difference of `out_official` CANNOT check `grad_draft`, and this is
not a tuning problem. `loss.py:145` is `confidence_targets = accept_rate_3d
.detach()`: the acceptance rate is computed from `draft_logits`, so upstream's
FORWARD loss is a function of the drafter's own logits, while upstream's
GRADIENT deliberately is not -- the detach is a stop-gradient and the returned
scalar is not differentiable-consistent with the graph that produced it.
Measured on `main`, d/d(draft[0]):

    autograd through DeepSeek's own code        -0.00598767
    ce + l1 terms only (target detached)         -0.00598767   <- identical
    ce + l1 + a LIVE confidence target           -0.00991132
    central difference of the forward scalar     -0.00991344   <- ~= the live one

The central difference converges, stably over h in [1e-3, 0.2], to the LIVE
target's gradient. So a finite-difference check of the reference gradient is
not merely loose, it is checking a different function, and it would report the
correct stop-gradient as broken. Two consequences, both load-bearing:

  * the reference gradient here is admissible ONLY because it is autograd on
    the pinned source -- there is no finite-difference route to it;
  * the FIX in `src/lib.rs` is not a choice between two equally defensible
    gradients. Ours was the LIVE one: we were training a term upstream does
    not train, worth 20.5-100.9% of the drafter gradient by case.

So the checks that are actually run, each with one job:

  * `grad_draft` EXACTLY INVARIANT to the presence of a confidence head. The
    conf term contributes nothing to the drafter's gradient when its target is
    detached, so `grad_draft` with a head and `grad_draft` without one must be
    the same tensor BIT FOR BIT. Exact, tolerance-free, and it is the property
    under test: any implementation that leaks `c_star` into the graph fails it.
  * `grad_draft` CROSS-CHECKED against this file's own transcription of the
    three formulas, differentiated (tier (b) under a tier-(a) number, the same
    arrangement `selfcheck` uses for the values). It cannot replace the
    reference; it catches a mis-shaped, mis-indexed or transposed recording,
    which nothing downstream would.
  * `grad_conf` FINITE-DIFFERENCE CHECKED. This one is valid, because nothing
    on the path from `confidence_pred` to the output is detached. A central
    difference here is a real derivative and it agrees.

NINE SIGNIFICANT DIGITS
-----------------------
`%.9g`, never `%g`. An f32 needs 9 to round-trip exactly; `f"{v:g}"` is six and
silently truncated every column of the rmsnorm fixture to 0.49975 overnight,
failing a test on CORRECT code.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# The official alphas, from the official config, NOT chosen here.
# deepseek-ai/DeepSpec@005e03b config/dspark/dspark_qwen3_4b.py:35-40
CE_ALPHA = 0.1
L1_ALPHA = 0.9
CONF_ALPHA = 1.0
GAMMA = 4.0

DEEPSPEC_REPO = "https://github.com/deepseek-ai/DeepSpec"
DEEPSPEC_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
TORCH_VERSION = torch.__version__

# The test's own bound. f32 CE over a 32-wide vocabulary: burn computes
# `log_softmax` then `gather`+`neg`, torch computes `cross_entropy`, whose CPU
# kernel is a different reduction order. log2(32) * u = 5 * 5.96e-8 = 3e-7, and
# the three terms are summed with different associativity, so ~2e-6 is the
# honest f32 envelope. 1e-5 is 5x that and still ~4 orders of magnitude below
# the smallest margin any case below actually has.
TOL_REL = 1e-5

# The Rust gate's bound on GRADIENTS, which live on a different scale from the
# loss and need their own. Gradients here are O(1e-2 .. 1e-3) per element, so
# a pure relative test is fine but a pure absolute one is not; both are carried
# and the test asserts `|ours - ref| <= GRAD_ABS + GRAD_REL * |ref|`.
#
# DERIVED, not tuned to pass: the f32 envelope was MEASURED by running this
# comparison with the detach in place (worst element over all 8 cases and all
# 462 coordinates: see `research/reviews/verify-tails-2026-09-30.md`), and these
# are the next power of ten above it. The margin in the OTHER direction is
# also measured, so the bound's job is visible: with the `detach()` REMOVED
# again, d/d(draft_logits) is wrong by 20.5% - 100.9% in L2 by case. A bound
# that discriminates 21% needs no more than 1e-2; this one is 1e-4.
GRAD_ABS = 1e-7
GRAD_REL = 1e-4


def g9(v) -> str:
    return "%.9g" % float(np.float32(v))


def fmt(a) -> str:
    return " ".join(g9(v) for v in np.asarray(a, dtype=np.float32).ravel())


def build_cases():
    """Cases chosen so a WRONG term is visible in a TERM, not only in a total.

    The official loss has three terms (CE, L1/TV, confidence BCE) and three
    weight conventions (the per-position decay w_k, the mask, and the
    `+1e-6` in the denominators). Each case below breaks one of them.
    """
    rng = np.random.default_rng(20260930)
    cases = []

    # 1. main: ordinary logits, a partially-masked block. The everyday case.
    #
    # WHY num_anchors == 1 EVERYWHERE, which is a shape decision and not a
    # convenience. Upstream's decay weight is indexed by the position WITHIN a
    # block (`_build_loss_weight_mask`: `arange(block_size)`, broadcast over
    # `[B, A, K]`), so its weight for anchor a, step s is exp(-s/gamma) --
    # the same for every anchor. Our `dspark_loss` takes a single `[B, L, V]`
    # and indexes `position_weights` by the FLATTENED position, which is
    # exp(-(a*K+s)/gamma). Those agree only when A == 1. With A > 1 the two
    # are different functions and a test would be measuring my reshape rather
    # than our formula, so the fixture holds A at 1 and the reshape is the
    # identity. The multi-anchor decay is upstream's business, tested there.
    b, a, k, v = 2, 1, 7, 32
    cases.append((
        "main",
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        (rng.random((b, a, k)) > 0.25).astype(np.float32),
        (rng.random((b, a)) > 0.3),
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.standard_normal((b, a, k)).astype(np.float32),
    ))

    # 2. all_masked_off: eval_mask is ALL ZERO. The official code divides by
    #    `ce_loss_den + 1e-6` -- i.e. by 1e-6, not by a clamped 1.0. A
    #    `clamp_min(1.0)` on the denominator is the single most likely
    #    transcription slip in this function and this case is the one that
    #    sees it: the official answer is ~0 and a clamped denominator returns
    #    an exactly-0 numerator over 1.0, which agrees -- so this case is
    #    NOT discriminating on its own. It is here to pin the no-NaN
    #    contract (0/1e-6 = 0, not 0/0 = NaN).
    cases.append((
        "all_masked_off",
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        np.zeros((b, a, k), dtype=np.float32),
        np.ones((b, a), dtype=bool),
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.standard_normal((b, a, k)).astype(np.float32),
    ))

    # 3. tiny_mask: exactly ONE supervised position in the whole block. The
    #    weighted denominator is then w_0 * 1 = exp(0) = 1.0 for that position
    #    and 0 elsewhere, so the decay weights cannot be cancelled: a loss
    #    that forgot `loss_decay_gamma` entirely still returns a finite number
    #    here, and the two differ in the CE and L1 terms by the ratio of the
    #    mean weight to w_0 = 1. It is also the case where `+1e-6` is 1e-6
    #    RELATIVE to a denominator of 1.0 and is therefore visible.
    m = np.zeros((b, a, k), dtype=np.float32)
    m[0, 0, 0] = 1.0
    cases.append((
        "tiny_mask",
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        m,
        np.ones((b, a), dtype=bool),
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.standard_normal((b, a, k)).astype(np.float32),
    ))

    # 4. saturated_conf: confidence logits pushed to +/-40, where sigmoid is
    #    1.0 and 0.0 to f32 precision. The official term is
    #    `binary_cross_entropy_with_logits` -- the STABLE form, which takes
    #    LOGITS and is finite at |logit| = 40 (loss ~ 0 for a correct sign,
    #    ~40 for a wrong one). A transcription that exponentiates first and
    #    clamps the probability to [1e-7, 1-1e-7] returns ~16.1 instead of
    #    ~0.0. This is the case that separates the two forms.
    conf = np.full((b, a, k), 40.0, dtype=np.float32)
    conf[..., ::2] = -40.0
    cases.append((
        "saturated_conf",
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        (rng.random((b, a, k)) > 0.25).astype(np.float32),
        np.ones((b, a), dtype=bool),
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        conf,
    ))

    # 5. one_hot_logits: draft and target agree exactly (aligned_target_logits
    #    == draft_logits), so the L1/TV term is EXACTLY 0 and
    #    accept_rate_3d is exactly 1. A ceiling case: any implementation that
    #    adds a floor to the TV term, or clamps accept_rate below 1, is
    #    visible here and nowhere else.
    dl = rng.standard_normal((b, a, k, v)).astype(np.float32)
    cases.append((
        "aligned_identical",
        dl.copy(),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        (rng.random((b, a, k)) > 0.25).astype(np.float32),
        np.ones((b, a), dtype=bool),
        dl.copy(),
        np.zeros((b, a, k), dtype=np.float32),
    ))

    # 6. no_confidence_head: confidence_pred = None. Upstream then SKIPS the
    #    whole confidence branch (`has_confidence` is False), so the term is
    #    absent rather than zero-weighted. An implementation that always
    #    computes a confidence term and multiplies it by `conf_alpha` returns
    #    a different TOTAL here while every other term agrees.
    cases.append((
        "no_confidence_head",
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        (rng.random((b, a, k)) > 0.25).astype(np.float32),
        np.ones((b, a), dtype=bool),
        rng.standard_normal((b, a, k, v)).astype(np.float32),
        None,
    ))

    # 7. big_logits: |logits| ~ 30, so softmax underflows to a one-hot in f32
    #    and the CE is ~0 while the L1 term saturates at 2.0. A reduction that
    #    subtracts a max computed in f16, or that normalises in f16, is
    #    visible here and invisible on `main`.
    big = (30.0 * rng.standard_normal((b, a, k, v))).astype(np.float32)
    cases.append((
        "big_logits",
        big,
        rng.integers(0, v, (b, a, k)).astype(np.int64),
        (rng.random((b, a, k)) > 0.25).astype(np.float32),
        np.ones((b, a), dtype=bool),
        (30.0 * rng.standard_normal((b, a, k, v))).astype(np.float32),
        rng.standard_normal((b, a, k)).astype(np.float32),
    ))

    # 8. block7_exact: the OFFICIAL shape's own block size (7) and vocab (32),
    #    one anchor, full mask -- i.e. the smallest legal block with no
    #    masking at all, where the decay weights exp(-k/4) are the ONLY thing
    #    distinguishing the correct answer from an undecayed one.
    dl = rng.standard_normal((1, 1, 7, 32)).astype(np.float32)
    cases.append((
        "block7_exact",
        dl,
        rng.integers(0, 32, (1, 1, 7)).astype(np.int64),
        np.ones((1, 1, 7), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        rng.standard_normal((1, 1, 7, 32)).astype(np.float32),
        rng.standard_normal((1, 1, 7)).astype(np.float32),
    ))

    return cases


def attribute_terms(dl, tid, em, conf, at, bkm):
    """Per-term numerators/denominators, using upstream's own formulas.

    Transcribed from deepspec/modeling/dspark/loss.py at 005e03b: the decay
    mask (_build_loss_weight_mask), the weighted CE (_collect_local_terms),
    the L1 term (_compute_local_l1_term) and the confidence term
    (binary_cross_entropy_with_logits against accept_rate_3d). Every one of
    these is a line-for-line restatement, NOT a re-derivation, and the TOTAL
    they produce is cross-checked against the official scalar below -- so a
    transcription slip here shows up as a mismatch with the official number
    rather than as a silently wrong attribution.
    """
    b, a, k, v = dl.shape
    pos = torch.arange(k, dtype=torch.float32).view(1, 1, -1)
    decay = torch.exp(-pos / GAMMA)
    w = torch.from_numpy(em) * decay                       # loss_weight_mask
    flat_w = w.reshape(-1)
    ce_num = (F.cross_entropy(
        torch.from_numpy(dl).reshape(-1, v), torch.from_numpy(tid).reshape(-1),
        reduction="none") * flat_w).sum()
    ce_den = flat_w.sum()
    if at is not None:
        pd = torch.softmax(torch.from_numpy(dl).float(), -1)
        pt = torch.softmax(torch.from_numpy(at).float(), -1)
        l1_num = ((pd - pt).abs().sum(-1) * w).sum()
        l1_den = w.sum()
    else:
        l1_num = l1_den = torch.zeros(())
    if conf is not None and at is not None:
        pd = torch.softmax(torch.from_numpy(dl).float(), -1)
        pt = torch.softmax(torch.from_numpy(at).float(), -1)
        accept = (1.0 - 0.5 * (pd - pt).abs().sum(-1)).clamp_(0.0, 1.0)
        c_num = (F.binary_cross_entropy_with_logits(
            torch.from_numpy(conf).float(), accept, reduction="none") * w).sum()
        c_den = w.sum()
    else:
        c_num = c_den = torch.zeros(())
    return {k: float(v_) for k, v_ in dict(
        ce_num=ce_num, ce_den=ce_den, l1_num=l1_num, l1_den=l1_den,
        conf_num=c_num, conf_den=c_den,
    ).items()}


def selfcheck(name, terms, official):
    """Refuse to emit a fixture the test would only reject."""
    problems = []
    ce = terms["ce_num"] / (terms["ce_den"] + 1e-6)
    l1 = terms["l1_num"] / (terms["l1_den"] + 1e-6) if terms["l1_den"] > 0 else 0.0
    cf = terms["conf_num"] / (terms["conf_den"] + 1e-6) if terms["conf_den"] > 0 else 0.0
    total = CE_ALPHA * ce + L1_ALPHA * l1 + CONF_ALPHA * cf
    if abs(total - official) > 1e-4 * max(1.0, abs(official)):
        problems.append(
            "%s: my per-term attribution gives %.9g but the OFFICIAL scalar is "
            "%.9g -- the transcription is wrong, refusing to emit"
            % (name, total, official))
    return problems, {"ce": ce, "l1": l1, "conf": cf, "total": total}


# ── the gradient, and the check that keeps it honest ────────────────────────────

# Central-difference step for the `grad_conf` check. 0.05 is a compromise
# between the f32 noise floor (which scales as 1/h) and the curvature of a
# sigmoid in the logit (which scales as h). Valid HERE and only here: nothing
# on the path from `confidence_pred` to the output is detached.
FD_H = 0.05
FD_REL = 5e-3
FD_ABS = 2e-5
# How many coordinates to probe, spread deterministically over the flattened
# tensor so a wrong indexing cannot hide in the un-sampled 99%.
FD_PROBES = 24

# Cross-check tolerances for the transcription-vs-autograd comparison. f32
# reductions in a different order, so this is an envelope, not an equality.
XC_REL = 1e-3
XC_ABS = 1e-5


def official_scalar(dl, tid, em, bkm, at, conf):
    """`compute_dspark_loss` on a fresh `DSparkForwardOutput`, as a float.

    No `requires_grad` anywhere: this is the pure scalar the `grad_conf`
    finite difference walks, and it must not be reading anything out of a
    graph.
    """
    from deepspec.modeling.dspark.common import DSparkForwardOutput
    from deepspec.modeling.dspark.loss import compute_dspark_loss
    import torch.distributed as dist
    assert dist.is_initialized(), "the process group must exist; see the docstring"
    o = DSparkForwardOutput(
        draft_logits=torch.from_numpy(dl),
        target_ids=torch.from_numpy(tid),
        eval_mask=torch.from_numpy(em),
        block_keep_mask=torch.from_numpy(bkm),
        confidence_pred=None if conf is None else torch.from_numpy(conf),
        aligned_target_logits=torch.from_numpy(at),
    )
    return float(compute_dspark_loss(
        outputs=o, loss_decay_gamma=GAMMA, ce_loss_alpha=CE_ALPHA,
        l1_loss_alpha=L1_ALPHA, confidence_head_alpha=CONF_ALPHA))


def grads_of(dl, tid, em, bkm, at, conf):
    """DeepSeek's own d/d(draft_logits) and d/d(confidence_pred), by autograd.

    The loss function is the pinned, unmodified, vendored source; this is
    `Tensor.backward()` on its return value. There is no formula of ours in
    here, which is the whole reason the gradient it produces is admissible as
    a reference.
    """
    from deepspec.modeling.dspark.common import DSparkForwardOutput
    from deepspec.modeling.dspark.loss import compute_dspark_loss

    dl_t = torch.from_numpy(dl).clone().requires_grad_(True)
    conf_t = None if conf is None else torch.from_numpy(conf).clone().requires_grad_(True)
    o = DSparkForwardOutput(
        draft_logits=dl_t,
        target_ids=torch.from_numpy(tid),
        eval_mask=torch.from_numpy(em),
        block_keep_mask=torch.from_numpy(bkm),
        confidence_pred=conf_t,
        aligned_target_logits=torch.from_numpy(at),
    )
    loss = compute_dspark_loss(
        outputs=o, loss_decay_gamma=GAMMA, ce_loss_alpha=CE_ALPHA,
        l1_loss_alpha=L1_ALPHA, confidence_head_alpha=CONF_ALPHA)
    loss.backward()
    return (dl_t.grad.numpy().copy(),
            None if conf_t is None else conf_t.grad.numpy().copy())


def transcribed_grads(dl, tid, em, conf, at):
    """The same three formulas, differentiated -- TIER (b), under a tier (a).

    `attribute_terms` already exists for the values and is already
    cross-checked against the official scalar; this is that transcription with
    `requires_grad` on, so a recorded gradient can be checked for shape, index
    and scale. It does NOT replace the reference: both sides agreeing with a
    transcription proves nothing about the reference. What it catches is a
    recording that is wrong in a way the reference number itself cannot show,
    because the reference number and this one are stored in the same fixture
    row and would agree in the fixture's bytes.
    """
    b, a, k, v = dl.shape
    d = torch.from_numpy(dl).clone().requires_grad_(True)
    c = None if conf is None else torch.from_numpy(conf).clone().requires_grad_(True)
    pos = torch.arange(k, dtype=torch.float32).view(1, 1, -1)
    w = torch.from_numpy(em) * torch.exp(-pos / GAMMA)
    den = w.reshape(-1).sum() + 1e-6

    ce = (F.cross_entropy(d.reshape(-1, v), torch.from_numpy(tid).reshape(-1),
                          reduction="none") * w.reshape(-1)).sum()
    # One graph, one backward, both leaves: `tv` is shared between the L1 term
    # and the acceptance rate, exactly as upstream shares it.
    tv = (torch.softmax(d, -1) - torch.softmax(torch.from_numpy(at), -1)).abs().sum(-1)
    total = CE_ALPHA * ce / den + L1_ALPHA * (tv * w).sum() / den
    if c is not None:
        accept = (1.0 - 0.5 * tv).clamp(0.0, 1.0).detach()
        cf = (F.binary_cross_entropy_with_logits(c, accept, reduction="none") * w).sum()
        total = total + CONF_ALPHA * cf / den
    total.backward()
    return d.grad.numpy().copy(), (None if c is None else c.grad.numpy().copy())


def check_grads(name, dl, tid, em, bkm, at, conf, g_draft, g_conf):
    """The three checks, each with exactly one job. See the docstring.

    Returns `(problems, report)` where `report` is a one-line-per-check
    summary for stderr, so a green run still says how close each check was.
    """
    problems, report = [], []
    x_draft, x_conf = transcribed_grads(dl, tid, em, conf, at)

    # (1) EXACT invariance of grad_draft to the presence of a head. This is the
    # property the detach creates, and it is a bit-for-bit comparison: no
    # tolerance is offered and none should be.
    headless, _ = grads_of(dl, tid, em, bkm, at, None)
    if conf is not None and not np.array_equal(headless, g_draft):
        n = int((headless != g_draft).sum())
        worst = float(np.abs(headless - g_draft).max())
        problems.append(
            "%s: grad_draft differs at %d/%d coordinates (worst %g) between a head "
            "and NO head, but loss.py:145 detaches the acceptance target, so the "
            "confidence term must contribute NOTHING to the drafter's gradient"
            % (name, n, g_draft.size, worst))
    report.append("invariance %s" % ("exact" if not any("grad_draft differs" in p for p in problems) else "VIOLATED"))

    # (2) The transcription, differentiated, must agree in scale and index.
    for what, ref, mine in (("draft", g_draft, x_draft), ("conf", g_conf, x_conf)):
        if ref is None:
            continue
        bad = np.abs(ref - mine) > (XC_ABS + XC_REL * np.abs(ref))
        if bad.any():
            i = int(np.argmax(np.abs(ref - mine) / (XC_ABS + XC_REL * np.abs(ref))))
            problems.append(
                "%s: grad_%s disagrees with this file's own transcription at %d "
                "coordinates; worst at [%d] ref %.9g vs transcription %.9g -- the "
                "recording is mis-shaped, mis-indexed or mis-scaled"
                % (name, what, int(bad.sum()), i, ref.reshape(-1)[i], mine.reshape(-1)[i]))
        report.append("xcheck %s worst %.2e" % (what, float(np.abs(ref - mine).max())))

    # (3) grad_conf by central difference. Valid, unlike grad_draft, because
    # nothing between confidence_pred and the output is detached.
    if conf is not None and g_conf is not None:
        flat = conf.reshape(-1)
        n = flat.size
        step = 1 + (n - 1) // FD_PROBES
        checked, worst = 0, 0.0
        for idx in range(0, n, step):
            if checked >= FD_PROBES:
                break
            p, m = flat.copy(), flat.copy()
            p[idx] += FD_H
            m[idx] -= FD_H
            fp = official_scalar(dl, tid, em, bkm, at, p.reshape(conf.shape))
            fm = official_scalar(dl, tid, em, bkm, at, m.reshape(conf.shape))
            fd = (fp - fm) / (2.0 * FD_H)
            an = float(g_conf.reshape(-1)[idx])
            if abs(fd - an) > FD_ABS + FD_REL * abs(an):
                problems.append(
                    "%s: d/d(conf[%d]) finite-difference %.9g but autograd %.9g"
                    % (name, idx, fd, an))
            worst = max(worst, abs(fd - an))
            checked += 1
        report.append("conf-vs-FD worst %.2e (%d probes)" % (worst, checked))

    return problems, "  ".join(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepspec", required=True,
                    help="path to a clone of deepseek-ai/DeepSpec")
    ap.add_argument("--dump", action="store_true",
                    help="also re-extract upstream/ from the clone")
    args = ap.parse_args()

    root = os.path.abspath(args.deepspec)
    loss_py = os.path.join(root, "deepspec/modeling/dspark/loss.py")
    if not os.path.isfile(loss_py):
        sys.exit("no %s -- pass --deepspec <clone of %s>" % (loss_py, DEEPSPEC_REPO))
    sha = hashlib.sha256(open(loss_py, "rb").read()).hexdigest()

    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # The upstream package, imported UNMODIFIED and only after the clone's
    # root is on sys.path.
    sys.path.insert(0, root)
    from deepspec.modeling.dspark.common import DSparkForwardOutput
    from deepspec.modeling.dspark.loss import compute_dspark_loss

    rows = []
    fd_report = []
    for c in build_cases():
        name, dl, tid, em, bkm, at, conf = c
        o = DSparkForwardOutput(
            draft_logits=torch.from_numpy(dl),
            target_ids=torch.from_numpy(tid),
            eval_mask=torch.from_numpy(em),
            block_keep_mask=torch.from_numpy(bkm),
            confidence_pred=None if conf is None else torch.from_numpy(conf),
            aligned_target_logits=torch.from_numpy(at),
        )
        official = float(compute_dspark_loss(
            outputs=o, loss_decay_gamma=GAMMA, ce_loss_alpha=CE_ALPHA,
            l1_loss_alpha=L1_ALPHA, confidence_head_alpha=CONF_ALPHA))
        terms = attribute_terms(dl, tid, em, conf, at, bkm)
        problems, attributed = selfcheck(name, terms, official)
        # The reference's OWN gradients, by autograd on the pinned source, plus
        # the three checks that are actually valid for a stop-gradient.
        g_draft, g_conf = grads_of(dl, tid, em, bkm, at, conf)
        g_problems, g_report = check_grads(name, dl, tid, em, bkm, at, conf,
                                           g_draft, g_conf)
        problems += g_problems
        fd_report.append("%s: %s" % (name, g_report))
        if problems:
            sys.stderr.write("\n".join("FIXTURE PROBLEM: " + p for p in problems) + "\n")
            sys.exit("refusing to emit")
        rows.append((name, dl, tid, em, bkm, at, conf, official, terms, attributed,
                     g_draft, g_conf))

    if args.dump:
        here = os.path.dirname(os.path.abspath(__file__))
        up = os.path.join(here, "upstream")
        os.makedirs(up, exist_ok=True)
        for src, dst in (("deepspec/modeling/dspark/loss.py", "deepspec_loss.py"),
                         ("deepspec/modeling/dspark/common.py", "deepspec_common.py")):
            with open(os.path.join(root, src), encoding="utf-8") as a, \
                 open(os.path.join(up, dst), "w", encoding="utf-8") as b:
                b.write("# VERBATIM from %s at %s\n# repo: %s\n# do not edit; "
                        "re-extract with gen_dspark_loss_oracle.py --dump\n\n"
                        % (src, DEEPSPEC_SHA, DEEPSPEC_REPO))
                b.write(a.read())

    print("# DSpark loss oracle fixture -- GENERATED, do not hand-edit.")
    print("# generator: tests/oracle/gen_dspark_loss_oracle.py")
    print("# upstream: %s" % DEEPSPEC_REPO)
    print("# commit:   %s  (2026-07-09)" % DEEPSPEC_SHA)
    print("# file:     deepspec/modeling/dspark/loss.py::compute_dspark_loss")
    print("# sha256:   %s" % sha)
    print("# torch:    %s   (CPU, gloo world_size=1)" % TORCH_VERSION)
    print("# config:   config/dspark/dspark_qwen3_4b.py -- ce=%g l1=%g conf=%g gamma=%g"
          % (CE_ALPHA, L1_ALPHA, CONF_ALPHA, GAMMA))
    print("# numbers:  9 significant digits (%.9g) of the f32 the reference produced")
    print("#")
    print("# out_official is what DeepSeek's own compute_dspark_loss returned.")
    print("# out_ce/out_l1/out_conf are the same three terms recomputed from the")
    print("# SAME vendored primitives, so a disagreement is attributable to one term.")
    print("# grad_draft/grad_conf are the SAME function's OWN autograd gradients,")
    print("# d/d(draft_logits) and d/d(confidence_pred), cross-checked here against")
    print("# a central difference of out_official -- they are the only half of the")
    print("# reference a value comparison cannot see.")
    print("")
    print("meta.cases: %s" % " ".join(r[0] for r in rows))
    print("meta.upstream_sha256: %s" % sha)
    print("meta.upstream_commit: %s" % DEEPSPEC_SHA)
    print("meta.torch_version: %s" % TORCH_VERSION)
    print("meta.ce_alpha: %s" % g9(CE_ALPHA))
    print("meta.l1_alpha: %s" % g9(L1_ALPHA))
    print("meta.conf_alpha: %s" % g9(CONF_ALPHA))
    print("meta.gamma: %s" % g9(GAMMA))
    print("meta.tol_rel: %s" % g9(TOL_REL))
    print("meta.grad_abs: %s" % g9(GRAD_ABS))
    print("meta.grad_rel: %s" % g9(GRAD_REL))
    print("meta.grad_fd_rel: %s" % g9(FD_REL))
    print("")
    for (name, dl, tid, em, bkm, at, conf, official, terms, attr,
         g_draft, g_conf) in rows:
        print("case.%s.dims: %s" % (name, " ".join(str(v) for v in dl.shape)))
        print("case.%s.draft_logits: %s" % (name, fmt(dl)))
        print("case.%s.target_ids: %s" % (name, " ".join(str(int(v)) for v in np.asarray(tid).ravel())))
        print("case.%s.eval_mask: %s" % (name, fmt(em)))
        print("case.%s.block_keep_mask: %s" % (name, " ".join("1" if v else "0" for v in np.asarray(bkm).ravel())))
        print("case.%s.aligned_target_logits: %s" % (name, fmt(at)))
        if conf is None:
            print("case.%s.confidence_pred: none" % name)
        else:
            print("case.%s.confidence_pred: %s" % (name, fmt(conf)))
        print("case.%s.out_official: %s" % (name, g9(official)))
        print("case.%s.out_ce: %s" % (name, g9(attr["ce"])))
        print("case.%s.out_l1: %s" % (name, g9(attr["l1"])))
        print("case.%s.out_conf: %s" % (name, g9(attr["conf"])))
        for t in ("ce_num", "ce_den", "l1_num", "l1_den", "conf_num", "conf_den"):
            print("case.%s.%s: %s" % (name, t, g9(terms[t])))
        print("case.%s.grad_draft: %s" % (name, fmt(g_draft)))
        if g_conf is None:
            print("case.%s.grad_conf: none" % name)
        else:
            print("case.%s.grad_conf: %s" % (name, fmt(g_conf)))
        print("")
    sys.stderr.write("emitted %d cases; every guard satisfied\n" % len(rows))
    sys.stderr.write("gradient checks (a stop-gradient is not finite-differenceable; "
                     "see the docstring):\n")
    for line in fd_report:
        sys.stderr.write("  %s\n" % line)


if __name__ == "__main__":
    main()
