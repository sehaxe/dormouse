"""The `residual_dtype="bf16"` arm: OFF pins, the measured drift, the census.

The arm stores the loop's residual stream at half the bytes
(`config.RESIDUAL_DTYPES`). This file judges it against its own named rule, one
gate per claim:

1. **OFF is clean main** — `residual_dtype="fp32"` (the default, no flag) must
   produce the same logits, loss and parameter gradients as `main` did before
   the field existed. The pins below were captured on a detached clean worktree
   of `main` (`f09eea5`) BEFORE the edit, at `OMP_NUM_THREADS=1` /
   `torch.set_num_threads(1)`.
2. **ON stores what the census says, and nothing else** — the three branch
   writes, the mHC lanes, `ctx`, `normed`, `y`, `step_out` are bf16; the lane
   norm (`xn`), the Sinkhorn, the KDA decay and the final norm stay fp32. The
   spy records the actual dtypes at the seams, so the flag cannot say one thing
   and store another.
3. **The drift is MEASURED, not asserted away** — 200 CPU steps, same seed,
   same fresh-batch stream, fp32 vs bf16. The pins are this machine's reading;
   the gate reports the curve and fails when the drift leaves its recorded
   band.
4. **A mutation that drops an fp32-required tensor into bf16 reddens a named
   gate** — the lane norm without its `.float()` raises at the hyper-network's
   matmul (the seam enforces the census); and a census row moved from
   `required` to `free` changes the pinned total.
5. **The census's arithmetic cannot drift** — every row's bytes are recomputed
   as `prod(shape)·itemsize` and compared to the pinned per-iteration totals.
6. **Refusals name the fp32-only kernel and carry `Escape:`**, both at
   `config.validate` and at the runtime seam a hand-built model reaches.
7. **No host sync** — `item`/`bool`/`float` raise through the ON forward.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
import torch

from dormouse.config import ModelConfig, load_run, validate
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model import mhc as mhc_mod
from dormouse.model.loopblock import LoopBlock
from dormouse.train.optim import MixOptimizer, clip_grads

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ffn=128,
    vocab=256,
    max_seq_len=32,
    max_iter=2,
    rank=8,
    n_experts=4,
    moe_topk=2,
    engram_rows=64,
    engram_dim=8,
    bf16=False,
    shared_rank=16,
    use_kda=True,
    use_engram=False,
    use_msa=False,
    use_csa=False,
)
#: mHC + AttnRes + LayerRoPE-paper + Engram: the composition the measured run
#: uses, at d=64.
A = dict(
    BASE,
    use_mhc=True,
    mhc_streams=2,
    use_attnres=True,
    layerrope=True,
    layerrope_form="paper",
    layerrope_index="iteration",
    layerrope_init="paper",
    use_engram=True,
)
#: The other residual operator (ReZero) plus CSA's additive branch.
B = dict(BASE, use_csa=True, csa_chunk=8, csa_kv_dim=16, csa_topk=4, attn_kind="kda")

#: `main f09eea5`, captured on a clean detached worktree BEFORE this branch
#: edited anything: sha256 over (logits, loss, every parameter gradient), then
#: over the state dict. `torch.set_num_threads(1)`, `torch.manual_seed(0)`.
PIN_A = "ece5e68a37482137d054f77f1c5feed51f3e0d215dccd9212ba23cd2195d6dcb"
PIN_A_SD = "d03d7c2222ed7c26079ee1d6eb614b75f85a0264ff1c78740468ebb45b8700dd"
PIN_B = "fed14353e2f84b9c2526ad1e87d0d2e04f98e8f7fea253ba0ac3709e92346f41"
PIN_B_SD = "3db871895d7f5f7c8e3b0bf57303a08802c98e04ad8d96716e7ddee8d40c5be5"

#: The 200-step measurement (this box, CPU, 1 thread, `tiny` b2 x s64,
#: mHC+AttnRes+Engram, fresh batch every step, `MixOptimizer lr=1e-3`).
#: `max_rel` is the worst per-step |Δloss|/|fp32 loss|; `final` is the signed
#: final-loss delta; `last20` the mean |Δloss| over the last 20 steps.
DRIFT_STEPS = 200
GATE_MAX_REL = 0.0007322474101372813
GATE_FINAL_DELTA = 0.00010442733764648438
GATE_LAST20_MEAN = 0.0006329298248291015
#: 25 % on the pin. The measurement is deterministic at one thread; the band is
#: for a scheduler that interleaves the reduction differently, and it is still
#: far below the fp32 seed spread (max per-step 7.7e-2, measured) — the arm is
#: judged against its own control here, not against a different initialisation.
DRIFT_BAND = 1.25

#: The precision census's own arithmetic, for the two configs the gate names.
CENSUS_M400 = (921_141_252, 644_268_036)
CENSUS_TINY_A = (543_492, 412_036)


def _cfg(**over) -> ModelConfig:
    return ModelConfig(**{**A, **over})


def _pinned(fn):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return fn()
    finally:
        torch.set_num_threads(threads)


def _digest(cfg: ModelConfig, ids: torch.Tensor) -> tuple[str, str]:
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cpu").to("cpu").train()
    model.zero_grad(set_to_none=True)
    logits, loss, _ = model(ids, targets=ids, counters=Counters())
    loss.backward()
    grad = hashlib.sha256()
    for t in (logits, loss.reshape(1)):
        grad.update(t.detach().contiguous().numpy().tobytes())
    for name, p in sorted(model.named_parameters()):
        if p.grad is not None:
            grad.update(name.encode())
            grad.update(p.grad.detach().contiguous().numpy().tobytes())
    state = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        state.update(name.encode())
        state.update(p.detach().contiguous().numpy().tobytes())
    return grad.hexdigest(), state.hexdigest()


def _load_census():
    name = "precision_census_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "precision_census.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves the module's string annotations through
    # `sys.modules`, so the module must be registered before `exec_module`.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _m400_cfg() -> ModelConfig:
    raw = tomllib.loads((ROOT / "configs" / "m400.toml").read_text())
    return ModelConfig(**raw)


# --------------------------------------------------------------------------- #
# 0. OFF is clean main — the pins were taken before the edit
# --------------------------------------------------------------------------- #
def test_off_is_bit_identical_to_clean_main() -> None:
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    grad, state = _pinned(lambda: _digest(_cfg(), ids))
    assert grad == PIN_A, (
        "the residual-bf16 arm's OFF path moved: `residual_dtype` must be a "
        f"route, not a formula. got {grad}"
    )
    assert state == PIN_A_SD, f"the OFF state dict moved: got {state}"


def test_off_csa_is_bit_identical_to_clean_main() -> None:
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    grad, state = _pinned(lambda: _digest(ModelConfig(**B), ids))
    assert grad == PIN_B, f"the CSA + ReZero OFF path moved: got {grad}"
    assert state == PIN_B_SD, f"the CSA OFF state dict moved: got {state}"


# --------------------------------------------------------------------------- #
# 1. ON stores exactly what the census says
# --------------------------------------------------------------------------- #
def _seam_dtypes(cfg: ModelConfig) -> dict[str, set[torch.dtype]]:
    """One forward, recording the dtypes at the residual seams.

    `_step_post` receives the three branch writes, the body's normed read and
    the state; `_iter_ce` receives the readout. `MHCMix._x_norm` is the
    fp32-REQUIRED reduction the census names.
    """
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cpu").train()
    seen: dict[str, set[torch.dtype]] = {
        k: set() for k in ("h", "streams", "attn", "branch", "normed_mlp", "ctx", "step_out", "xn")
    }
    orig_post = LoopBlock._step_post
    orig_ce = LoopBlock._iter_ce
    orig_xn = mhc_mod.MHCMix._x_norm

    def post(self, h, streams, attn, branch, normed_mlp, *rest):
        seen["h"].add(h.dtype)
        if streams is not None:
            seen["streams"].add(streams.dtype)
        for name, t in (("attn", attn), ("branch", branch), ("normed_mlp", normed_mlp)):
            seen[name].add(t.dtype)
        # ctx is the 9th positional argument after normed_mlp.
        seen["ctx"].add(rest[3].dtype)
        return orig_post(self, h, streams, attn, branch, normed_mlp, *rest)

    def ce(self, step_out, *rest):
        seen["step_out"].add(step_out.dtype)
        return orig_ce(self, step_out, *rest)

    def xn(self, x_streams):
        out = orig_xn(self, x_streams)
        seen["xn"].add(out.dtype)
        return out

    LoopBlock._step_post = post
    LoopBlock._iter_ce = ce
    mhc_mod.MHCMix._x_norm = xn
    try:
        ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
        model(ids, targets=ids, counters=Counters())[0]
    finally:
        LoopBlock._step_post = orig_post
        LoopBlock._iter_ce = orig_ce
        mhc_mod.MHCMix._x_norm = orig_xn
    return seen


def test_on_stores_the_residual_in_bf16() -> None:
    off = _seam_dtypes(_cfg(residual_dtype="fp32"))
    on = _seam_dtypes(_cfg(residual_dtype="bf16"))
    free = ("h", "streams", "attn", "branch", "normed_mlp", "ctx", "step_out")
    assert all(off[k] == {torch.float32} for k in free), off
    assert all(on[k] == {torch.bfloat16} for k in free), on
    # The REQUIRED reduction is fp32 in both arms: the flag is storage, not a
    # licence to run the lane norm in bf16.
    assert off["xn"] == {torch.float32} and on["xn"] == {torch.float32}, (off, on)


# --------------------------------------------------------------------------- #
# 2. The drift: measured over 200 steps, pinned, not "should be fine"
# --------------------------------------------------------------------------- #
def _drift_curve(residual_dtype: str) -> list[float]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _drift_curve_t1(residual_dtype)
    finally:
        torch.set_num_threads(threads)


def _drift_curve_t1(residual_dtype: str) -> list[float]:
    torch.manual_seed(1)
    run = load_run(
        "tiny",
        model_overrides={
            "use_mhc": True,
            "mhc_streams": 2,
            "use_attnres": True,
            "use_engram": True,
            "residual_dtype": residual_dtype,
        },
        train_overrides={"steps": DRIFT_STEPS, "batch": 2, "seq_len": 64},
    )
    model = DormouseModel(run.model, device=torch.device("cpu")).train()
    opt = MixOptimizer(model, lr=1e-3, wd=0.0, ns_steps=5)
    data = torch.Generator().manual_seed(1234)
    losses: list[float] = []
    for _ in range(DRIFT_STEPS):
        x = torch.randint(0, 256, (2, 64), generator=data)
        y = torch.randint(0, 256, (2, 64), generator=data)
        opt.zero_grad()
        _logits, loss, _counters = model(x, y)
        loss.backward()
        if run.train.grad_clip > 0:
            clip_grads(model.parameters(), run.train.grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def test_on_vs_off_drift_is_the_measured_200_step_value() -> None:
    fp32 = _drift_curve("fp32")
    bf16 = _drift_curve("bf16")
    rel = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(fp32, bf16, strict=True)]
    max_rel = max(rel)
    final_delta = bf16[-1] - fp32[-1]
    last20 = sum(abs(a - b) for a, b in zip(fp32[-20:], bf16[-20:], strict=True)) / 20
    print(
        f"\nresidual bf16, {DRIFT_STEPS} steps: max per-step rel {max_rel:.6e}, "
        f"final Δ {final_delta:+.6e}, last-20 mean |Δ| {last20:.6e} "
        f"(fp32 final {fp32[-1]:.6f}, bf16 final {bf16[-1]:.6f})"
    )
    assert max_rel <= GATE_MAX_REL * DRIFT_BAND, (
        f"the residual-bf16 drift left its measured band: max per-step rel "
        f"{max_rel:.6e} > {GATE_MAX_REL:.6e} × {DRIFT_BAND} — the flag now costs "
        f"more than the pin says"
    )
    assert abs(final_delta) <= GATE_FINAL_DELTA * DRIFT_BAND, (
        f"final loss moved {final_delta:+.6e}, past the pinned "
        f"{GATE_FINAL_DELTA:.6e} × {DRIFT_BAND}"
    )
    assert last20 <= GATE_LAST20_MEAN * DRIFT_BAND, (
        f"last-20 mean |Δloss| {last20:.6e}, past the pinned {GATE_LAST20_MEAN:.6e} × {DRIFT_BAND}"
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in bf16), "a loss went non-finite"


# --------------------------------------------------------------------------- #
# 3. The mutation that drops a required tensor into bf16 reddens its gate
# --------------------------------------------------------------------------- #
def test_dropping_the_fp32_lane_norm_reddens_the_dtype_seam() -> None:
    """`xn` is the census's `required` row: bf16 storage + bf16 norm dies here."""
    cfg = _cfg(residual_dtype="bf16")
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cpu")
    ids = torch.arange(96, dtype=torch.long).reshape(3, 32) % 251
    # The clean arm runs: the required tensor is widened as the census says.
    model(ids, targets=ids, counters=Counters())[0]

    orig = mhc_mod.MHCMix._x_norm

    def no_widen(self, x_streams):
        b, t = x_streams.shape[:2]
        flat = x_streams.reshape(b, t, self.n_streams * self.d_model)
        return flat / torch.sqrt(flat.pow(2).mean(-1, keepdim=True) + mhc_mod.EPS)

    mhc_mod.MHCMix._x_norm = no_widen
    try:
        with pytest.raises(RuntimeError, match="same dtype|dtype"):
            model(ids, targets=ids, counters=Counters())
    finally:
        mhc_mod.MHCMix._x_norm = orig
    # Restored, the same forward runs again.
    model(ids, targets=ids, counters=Counters())[0]


def test_a_required_census_row_moved_to_bf16_reddens_the_pinned_total() -> None:
    pc = _load_census()
    rows = pc.tensors(_cfg(), 2, 32)
    before, after = pc.total(rows), pc.total(rows, after=True)
    assert (before, after) == CENSUS_TINY_A, ((before, after), CENSUS_TINY_A)
    required = next(r for r in rows if r.name.startswith("xn"))
    assert required.required, "the census must mark the lane norm fp32-required"
    mutated = [dataclasses.replace(r, required=False) if r is required else r for r in rows]
    assert pc.total(mutated) == before, "the fp32 total must not depend on `required`"
    assert pc.total(mutated, after=True) != after, (
        "flipping a REQUIRED row to bf16 must change the projected total; "
        "otherwise the policy is not what the census prices"
    )


# --------------------------------------------------------------------------- #
# 4. The census's arithmetic is recomputed, and pinned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "cfg", "pin"),
    [
        ("m400 b8 x s1024", _m400_cfg(), CENSUS_M400),
        ("tiny-A b2 x s32", _cfg(), CENSUS_TINY_A),
    ],
)
def test_the_census_bytes_are_recomputed_and_pinned(label, cfg, pin) -> None:
    pc = _load_census()
    b, s = (8, 1024) if label.startswith("m400") else (2, 32)
    rows = pc.tensors(cfg, b, s)
    assert rows, "the census produced no tensors"
    for r in rows:
        assert r.nbytes == r.elements * pc.ITEMSIZE[r.dtype], f"{r.name}: row bytes drift"
        assert r.after in pc.ITEMSIZE, f"{r.name}: unknown projected dtype {r.after}"
    recomputed = sum(r.elements * pc.ITEMSIZE[r.dtype] for r in rows)
    recomputed_after = sum(r.elements * pc.ITEMSIZE[r.after] for r in rows)
    assert (pc.total(rows), pc.total(rows, after=True)) == (recomputed, recomputed_after)
    assert (recomputed, recomputed_after) == pin, (
        f"{label}: the census totals moved — got {(recomputed, recomputed_after)}, "
        f"pinned {pin}. A tensor edit that does not recompute goes red here."
    )
    assert any(r.required for r in rows) and any(not r.required for r in rows)


# --------------------------------------------------------------------------- #
# 5. Refusals name the fp32-only kernel, and carry `Escape:`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("flag", "kernel"),
    [
        ("fused_residual", "fused_residual"),
        ("fused_epilogue", "fused_epilogue"),
        ("fused_kda_tail", "fused_kda_tail"),
    ],
)
def test_bf16_refuses_the_fp32_only_kernels_loudly(flag, kernel) -> None:
    cfg = _cfg(residual_dtype="bf16", **{flag: True})
    with pytest.raises(LoudError) as e:
        validate(cfg)
    msg = str(e.value)
    assert "Escape:" in msg, msg
    assert kernel in msg, f"the refusal does not name {kernel}: {msg}"
    # A hand-built model skips `validate`; the runtime seam must hold too.
    with pytest.raises(LoudError) as e2:
        DormouseModel(cfg, device="cpu")
    assert "Escape:" in str(e2.value) and kernel in str(e2.value)


def test_an_unknown_residual_dtype_is_loud() -> None:
    with pytest.raises(LoudError) as e:
        validate(_cfg(residual_dtype="bf16x"))
    assert "Known: fp32, bf16" in str(e.value) and "Escape:" in str(e.value)
    with pytest.raises(LoudError):
        DormouseModel(_cfg(residual_dtype="bf16x"), device="cpu")


# --------------------------------------------------------------------------- #
# 6. The changed forward adds no host sync
# --------------------------------------------------------------------------- #
def test_the_bf16_residual_forward_touches_no_host(monkeypatch) -> None:
    """`item`/`bool`/`float` raise through the ON forward.

    `tolist` is deliberately NOT patched: the CPU MoE's `_naive_dispatch` has a
    pre-existing `.tolist()` outside this task. The residual seam and the arms
    it widens add none.
    """

    def boom(*_a, **_k):
        raise AssertionError("host sync in the residual-bf16 forward")

    # `Tensor.float()`/`.bool()` are dtype casts, not host reads. `tolist` is
    # left unpatched: the CPU MoE's `_naive_dispatch` has a pre-existing
    # `.tolist()` (`experts.py:2263`, outside this task) — the residual seam
    # and the arms it widens add no host read of their own.
    for name in ("item", "__bool__", "__float__"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    cfg = _cfg(residual_dtype="bf16", use_engram=False)
    torch.manual_seed(0)
    model = DormouseModel(cfg, device="cpu").train()
    ids = torch.arange(32, dtype=torch.long).reshape(1, 32) % 251
    logits, loss, _ = model(ids, targets=ids, counters=Counters())
    loss.backward()
    # `.numpy()`, not `bool(...)`: the tensor primitives are still patched.
    assert bool(torch.isfinite(logits).all().numpy())
