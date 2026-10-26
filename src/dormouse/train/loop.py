"""The trainer loop: WSD + Muon/AdamW, the NaN firewall, checkpoints, counters.

The instrument layer is day-one code here, not an addition (readiness plan §7):
every log line carries the arm counters and the step timing, every eval line
carries the BYTE COUNT it scored (`eval_batches × batch × seq_len`, AGENTS
§2.6), and the two failure modes that cost the burn side real runs are LOUD:

* **a NaN step is skipped, COUNTED** (`nan_skipped`), and two consecutive
  windows in which EVERY step was non-finite stop the run — the plan's
  `nan_windows ≥ 2` hard error (Phase 1 item 1), whose absence is why "a
  drowning-NaN run currently exits 0" (plan §1.2).
* **a resume refuses a differing resolved config**, progress keys exempt
  (ADR-0021). Extending a run is legal; changing what the model computes under
  one step count is not.
"""

from __future__ import annotations

import dataclasses
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..config import PROGRESS_KEYS, RunConfig, TrainConfig, validate_train
from ..instrument import Counters, LoudError, StepTimer, loud
from ..model import kda
from ..model.engram import (
    Engram,
    enable_host_tables,
    host_engrams,
    host_state,
    host_step_all,
    load_host_state,
)
from ..model.experts import invalidate_banks_all
from ..model.qat_cache import invalidate_qat_cache_all
from ..model.tsct import set_mm_dtype
from .data import ByteStream, distinct_scored_bytes, eval_window_bytes, open_streams, warmup_steps
from .graph import FullGraph, OptimizerGraph, full_graph_refusal
from .lr import anneal_alpha, wsd_lr
from .mixture import MixtureStream, open_mixture
from .optim import MixOptimizer, clip_grads, grad_norm


def bpb(loss: float) -> float:
    """Bits per byte from a natural-log CE (burn's `dormouse_train::bpb`)."""
    return loss / math.log(2)


def _host_scalar(value: torch.Tensor) -> float:
    """The ONE declared host read of a device scalar (log / report cadence).

    Everything the step decides is device-resident; this is the seam where the
    already-decided numbers cross to the host, so the 0-sync rule has a single
    auditable place to be checked (`tests/test_nan_firewall.py` patches THIS
    function to allow the cadence and to fail on any other read).
    """
    return float(value)


def kda_field(counters: Counters) -> str:
    """`kda=<arm>,n=<calls>` — which attention arm produced this window."""
    if counters.get("attn_full_forward"):
        return f"kda=full,n={counters['attn_full_forward']}"
    n = counters.get("kda_sdpa", 0) + counters.get("kda_fla", 0)
    if counters.get("kda_fla"):
        arm = "fla"
    elif counters.get("kda_sdpa"):
        arm = "sdpa"
    else:
        arm = "off"
    return f"kda={arm},n={n}"


def mtp_field(model: torch.nn.Module) -> str:
    """`mtp=<term>` — the MTP objective's weighted value, on the log cadence.

    The forward leaves its scalar DETACHED on the model (`last_mtp`) and never
    touches the host (ADR-0018 rule 2: no `.item()` in a step). This is the one
    declared read, once per log window, and it is what makes the arm visible in
    the log rather than only in the parameter count. Empty when the arm never
    ran (weight 0, a decode forward, or a step before the first training
    forward) — a zero would claim a term that was not added.
    """
    value = getattr(model, "last_mtp", None)
    return "" if value is None else f"mtp={float(value):.4f}"


def halt_field(model: torch.nn.Module) -> str:
    """`halt=<BCE>` — the halt objective's value, on the log cadence.

    The forward leaves the term DETACHED on the model (`last_halt`) and never
    reads it; this is the one declared read, once per log window (ADR-0018
    rule 2 — the same shape as `mtp_field`). Empty when the arm never ran
    (weight 0, a decode forward, a pure eval): a zero would claim a term that
    was not added.
    """
    value = getattr(model, "last_halt", None)
    return "" if value is None else f"halt={float(value):.4f}"


def ckpt_paths(cfg: TrainConfig) -> tuple[Path, Path]:
    base = Path(cfg.out_dir)
    return base / f"{cfg.ckpt_name}.pt", base / f"{cfg.ckpt_name}.config.toml"


def ngram_path(cfg: TrainConfig) -> Path:
    """The host tables' sibling of a checkpoint: `<ckpt_name>.ngram`.

    A sibling file, not a key inside the checkpoint, for the reason the burn side
    learned (`7adda92`, AGENTS §3.3): the `.ngram` sidecar did not travel with
    the weights and a `.best` checkpoint turned out to be a different model. Here
    the pairing is checkable — the load refuses a missing sibling, the snapshot
    drift check refuses an `engram_ram` flip, and `load_host_state` refuses a
    shape mismatch — which is what that entry's "no test, no stamp" complaint
    asked for, minus the `train_step` stamp.
    """
    weights, _ = ckpt_paths(cfg)
    return weights.with_suffix(".ngram")


def host_device_type(model: torch.nn.Module) -> str:
    """The device the MODEL lives on — the one input to the host-path refusal.

    Its own function because it is the seam the CPU suite drives (`engram_ram`
    is a CUDA-budget rule; the host path's NUMERICS are device-agnostic and CPU-
    observable, since the tables ARE CPU memory). Nothing else reads it.
    """
    return next(model.parameters()).device.type


def host_tables(model: torch.nn.Module, bias_correction: bool = True) -> int:
    """Switch every Engram to host tables; LOUD on a run that cannot have them.

    Called BEFORE the optimizer is built (`train`), which is what keeps the
    tables out of a parameter group: on this path they are plain CPU tensors,
    not parameters at all.

    Two refusals, no fallback — both cases would show the flag on while the
    thing the flag buys (VRAM) was never saved:

    * a non-CUDA model has no VRAM to save: the per-forward row transfers are
      pure cost against the in-VRAM table, and burn's `--engram-ram` is a
      VRAM-budget decision (`cfg.rs:50`);
    * a model with no Engram to host (`use_engram=false`) — `load_run` refuses
      that pair too, but a hand-built model reaches here without passing it.

    `bias_correction` is `TrainConfig.engram_bias_correction`: the host update's
    formula, not a substrate choice (`enable_host_tables` sets it).
    """
    dev = host_device_type(model)
    if dev != "cuda":
        loud(
            f"engram_ram on a {dev} model: host tables buy VRAM by keeping the n-gram "
            f"rows and their momentum in host RAM, and a {dev} run has no VRAM to buy — "
            f"the per-forward row transfers would be pure cost against the in-VRAM "
            f"table. Escape: engram_ram=false, or run on cuda."
        )
    n = enable_host_tables(model, bias_correction=bias_correction)
    if n == 0:
        loud(
            "engram_ram=true but the model has no Engram to host (use_engram=false?): "
            "the flag would host nothing while the run reported it on. Escape: "
            "use_engram=true, or engram_ram=false."
        )
    return n


def load_ngram(cfg: TrainConfig) -> dict:
    """The `<ckpt_name>.ngram` sibling, read as CPU state (tensors + ints only)."""
    return torch.load(ngram_path(cfg), map_location="cpu", weights_only=True)


def _checkpoint_half(sd: dict, rung: str) -> dict:
    """One model half normalized for the file: prefix stripped, rung packed.

    The ONE normalization `save_ckpt` applies to BOTH copies an EMA run
    writes (`model` raw, `model_ema` shadow), so the two cannot disagree about
    a storage rung or a `torch.compile` prefix.
    """
    from ..model.tsct import is_factor_key, pack_tsct

    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if rung and rung in ("fp8", "nvfp4", "ternary"):
        sd = {k: (pack_tsct(v, rung) if is_factor_key(k) else v) for k, v in sd.items()}
    return sd


class _ModelHalf(dict):
    """The model half of a checkpoint, plus the parts that are NOT `state_dict` keys.

    `moe` is the LFB-bias section (`save_ckpt`). The biases are non-persistent
    buffers on purpose — the optimizer must not see them and `state_dict()` must
    not grow a routing integrator — so a strict `load_state_dict` cannot carry
    them, while `ck["model"]` must stay exactly `state_dict()`-shaped (that is
    what every direct `load_state_dict(ck["model"])` caller and the prefix test
    assume). The attribute is how the ONE loader reaches them.
    """

    #: `{module name: lfb_bias}`; `None` is a file written before the section.
    moe: dict[str, torch.Tensor] | None = None
    #: A training load refuses a file whose controller state is missing when
    #: `moe_lb_coef != 0`; a reader prints the COUNTED line instead (`load_model_half`).
    refuse_missing_moe: bool = False


def _lfb_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """The LFB modules by CHECKPOINT name: `torch.compile`'s `_orig_mod.` stripped."""
    return {
        name.removeprefix("_orig_mod."): m
        for name, m in model.named_modules()
        if getattr(m, "lfb_bias", None) is not None
    }


def lfb_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """`{module name: lfb_bias}` — the routing controller's integrator OUTPUT.

    `lfb_bias` is not reconstructible from the persistent `lfb_counts`
    (`b_e -= coef·sign(L_e − mean L)` is the integrator's state, not a function
    of the cumulative load), so it travels in the checkpoint's `moe` section.
    """
    return {
        name: m.lfb_bias.detach().to("cpu", copy=True) for name, m in _lfb_modules(model).items()
    }


def _with_moe(sd: dict, moe: dict | None, *, refuse: bool) -> _ModelHalf:
    """Wrap a model half and attach the `moe` section (may be `None` = pre-section file)."""
    half = _ModelHalf(sd)
    half.moe = None if moe is None else {str(k): v for k, v in moe.items()}
    half.refuse_missing_moe = refuse
    return half


def _carries_lfb_counts(sd: dict) -> bool:
    """Whether this model half carries the LFB integrator's persistent counter."""
    return any(k == "lfb_counts" or k.endswith(".lfb_counts") for k in sd)


def _refuse_missing_lfb(where: str, coef: float) -> None:
    """REFUSE, never guess: a checkpoint with `lfb_counts` but no `lfb_bias`.

    The prejudice is `load_host_state`'s (a `.ngram` file without `counts`): the
    missing half IS controller state that no later step can recover, so a zero
    default silently trains — or serves — a different router under the same step
    count. Measured on the 4+resume repro at `moe_lb_coef=0.01`: 50 of 51
    tensors differ; at `0.0` the run is bit-exact.
    """
    loud(
        f"{where} carries `lfb_counts` but no `lfb_bias`, and this run has "
        f"moe_lb_coef={coef:g}. The bias is the routing controller's OUTPUT "
        f"(`b_e -= coef·sign(L_e − mean L)`); it cannot be derived from the "
        f"cumulative counts, so starting it at zero would train a DIFFERENT router "
        f"from the one this step count claims (measured: 50/51 tensors differ after "
        f"4+4 steps at coef=0.01, bit-exact at coef=0). Escape: resume this file on "
        f"the commit that wrote it, set moe_lb_coef=0 (the controller then never "
        f"moves the bias), or start a new ckpt_name."
    )


def _restore_lfb(model: torch.nn.Module, moe: dict[str, torch.Tensor], *, where: str) -> int:
    """Copy a checkpoint's `moe` section into the model's LFB buffers, LOUD on a mismatch."""
    live = _lfb_modules(model)
    missing = sorted(set(live) - set(moe))
    unknown = sorted(set(moe) - set(live))
    if missing or unknown:
        loud(
            f"{where}: the `moe` section names {len(moe)} LFB module(s) but this "
            f"model has {len(live)} (missing={missing[:4]}, unknown={unknown[:4]}). "
            f"Escape: load with the config/preset the file was written from, or "
            f"start a new ckpt_name."
        )
    with torch.no_grad():
        for name, module in live.items():
            src = moe[name]
            if tuple(src.shape) != tuple(module.lfb_bias.shape):
                loud(
                    f"{where}: `{name}.lfb_bias` is {tuple(src.shape)} in the file "
                    f"but {tuple(module.lfb_bias.shape)} in this model. Escape: the "
                    f"config the file was written from."
                )
            module.lfb_bias.copy_(
                src.to(device=module.lfb_bias.device, dtype=module.lfb_bias.dtype)
            )

    return len(live)


def save_ckpt(
    run: RunConfig,
    model: torch.nn.Module,
    opt: MixOptimizer,
    step: int,
    *,
    ema: dict | None = None,
) -> tuple[Path, Path]:
    """Weights + optimizer state, plus the config snapshot beside them.

    A storage rung (`model.quant` ∈ fp8/nvfp4) packs the factor matrices on
    the way out: masters stay fp32 in memory, the FILE carries the compressed
    record (`model_quant_rung` names it), and `load_ckpt` unpacks back to
    fp32. The bank/checkpoint is the rung's whole contract — the forward path
    within a run never changes numerics.

    `ema` (the shadow of `TrainConfig.ema_decay`) is written beside the raw
    training copy under the clearly named `model_ema` key, packed by the SAME
    rule. It is the checkpoint's SERVED model half — `load_model_half` prefers
    it — while `model` stays the RAW iterate a resume continues (the optimizer
    moments belong to those weights). An EMA checkpoint therefore carries two
    weight copies; that is the cost of a legal resume, stated here rather than
    discovered from a 2x file size.
    """
    weights, snapshot = ckpt_paths(run.train)
    weights.parent.mkdir(parents=True, exist_ok=True)
    rung = run.model.quant if run.model.quant != "none" else ""
    # A torch.compile'd model is an OptimizedModule whose keys carry a
    # `_orig_mod.` prefix. Strip it on the way out (audit 2026-10-17,
    # CRITICAL 2): every reader — load_ckpt, eval, decode, serve — loads into
    # the UNWRAPPED model, so a prefixed checkpoint is unresumable and
    # un-evaluable. The prefix is a wrapper artifact, never part of the file
    # format.
    payload = {
        "model": _checkpoint_half(model.state_dict(), rung),
        "opt": opt.state_dict(),
        "step": step,
        "model_quant_rung": rung or None,
        # The LFB bias is the other half of the routing controller (`lfb_counts`
        # rides in `model` as a persistent buffer); it is non-persistent because
        # `state_dict()` is the OPTIMIZER's view of the model. A file that
        # restored only the counts would restart the integrator at zero — the
        # measured 50/51-tensor divergence of a 4+resume at moe_lb_coef=0.01.
        "moe": lfb_state(model),
    }
    if ema is not None:
        payload["model_ema"] = _checkpoint_half(ema, rung)
    tmp = weights.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(weights)  # atomic: a crash never leaves a half-written ckpt
    if run.train.engram_ram:
        # The host tables are not parameters, so they are not in `sd`: they are
        # the sibling file, written atomically like the weights. A checkpoint
        # that restored the model without its memory would be a different
        # network at the same step count (AGENTS §3.3).
        ng = ngram_path(run.train)
        tmp_ng = ng.with_suffix(".ngram.tmp")
        torch.save(host_state(model), tmp_ng)
        tmp_ng.replace(ng)
    snapshot.write_text(run.snapshot_toml())
    return weights, snapshot


def unpack_state_dict(sd: dict, rung: str | None) -> dict:
    """Model half of a checkpoint → fp32 masters. Packed records unpacked;
    a rung that does not name a storage level is a loud refusal.

    The ONE normalization every reader shares: the `_orig_mod.` prefix a
    torch.compile'd run wrote before the strip landed (audit CRITICAL 2) is
    dropped here, so `load_ckpt`, `eval`, `decode` and `serve` all load into the
    UNWRAPPED model.
    """
    from ..model.tsct import is_factor_key, unpack_tsct

    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if rung is None:
        return sd
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and v.get("rung"):
            assert is_factor_key(k), f"packed record at a non-factor key {k}"
            out[k] = unpack_tsct(v)
        else:
            out[k] = v
    return out


def load_model_half(weights: Path) -> dict:
    """A checkpoint FILE's model half → fp32 masters, storage rung unpacked.

    The loader `eval`, `decode` and `serve` share with `load_ckpt` (audit MAJOR
    6, 2026-10-17): those three read the file with a raw `torch.load` +
    `load_state_dict`, so a checkpoint written with `quant = fp8|nvfp4|ternary`
    — a config `validate` accepts and `save_ckpt` packs — died in the load with
    a `RuntimeError` naming neither the rung nor the escape.

    An EMA checkpoint keeps its SERVED copy under `model_ema` and its RAW
    training copy under `model`; this reader PREFERS the former, so eval,
    decode, serve and chat all score the averaged model while `load_ckpt`'s
    resume still continues the raw iterate and its optimizer state.

    The `moe` section (the LFB biases) is attached to the returned half and the
    ONE loader applies it. A file written before the section existed CANNOT be
    served with the router it trained (the bias is not derivable), so that is
    COUNTED here rather than silent; a training load of the same file REFUSES
    (`_refuse_missing_lfb`).
    """
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    half = ck.get("model_ema")
    if half is None:
        half = ck["model"]
    sd = unpack_state_dict(half, ck.get("model_quant_rung"))
    moe = ck.get("moe")
    if moe is None and _carries_lfb_counts(sd):
        print(
            f"router: {weights} carries lfb_counts but no lfb_bias (a file written "
            f"before the LFB section existed); serving it with a ZERO bias, i.e. a "
            f"different router than a moe_lb_coef>0 run trained. Escape: resume it "
            f"on the commit that wrote it (a resume refuses), or serve a file "
            f"written by this build."
        )
    return _with_moe(sd, moe, refuse=False)


def require_ngram(cfg: TrainConfig) -> None:
    """LOUD unless the host tables' `<ckpt_name>.ngram` sibling exists.

    ONE definition of the refusal: `load_ckpt` (a resume) and every CLI loader
    (`load_model_state`) call it, so an `engram_ram` checkpoint that lost its
    sidecar fails the same way everywhere. On that path the tables ARE this file
    — the model half carries no `engram.tables` parameter (`Engram.to_host`
    deletes it) — so loading without it would silently seed fresh rows and use a
    different network under the same step count (`7adda92`, AGENTS §3.3).
    """
    if ngram_path(cfg).is_file():
        return
    weights, _ = ckpt_paths(cfg)
    loud(
        f"{weights} is a host-table run (train.engram_ram=true) and its sibling "
        f"{ngram_path(cfg)} is missing. On that path the tables and their momentum "
        f"ARE the memory's state — the model half carries no `engram.tables` "
        f"parameter — so loading without them would restart the memory from scratch "
        f"and use a different network under this step count. Escape: restore the "
        f".ngram, start fresh under a new ckpt_name, or set engram_ram=false."
    )


def init_cfg(cfg: TrainConfig, path: Path, sd: dict, model: torch.nn.Module) -> TrainConfig:
    """The cfg that describes the INIT FILE: its substrate and its sidecar path.

    `load_model_state` reads the substrate from `cfg.engram_ram` and finds the
    `.ngram` sibling through `cfg.out_dir`/`cfg.ckpt_name`, so an init load
    cannot use the SFT run's own cfg — that names the sidecar THIS run will
    write, not the init file's. The substrate is read from the FILE, not from a
    flag: a host-table checkpoint's model half carries no `engram.tables`
    (`Engram.to_host` deletes the parameter), which is the file's own statement
    of where its memory lives.

    The run's `engram_ram` must AGREE with the file's, and a disagreement is
    LOUD, because the flag decides three things a mismatch would corrupt: the
    host switch happens BEFORE the optimizer exists (a switch after it leaves
    the tables in a parameter group that then steps a deleted tensor), the loop
    steps the tables only when the flag is on, and `save_ckpt` writes the
    `.ngram` only when it is on.
    """
    tables_in_file = any(k.endswith("engram.tables") for k in sd)
    has_engram = any(isinstance(m, Engram) for m in model.modules())
    host = has_engram and not tables_in_file
    if host and not cfg.engram_ram:
        loud(
            f"init-from: {path}'s model half carries no `engram.tables`: it "
            f"is a host-table run (its rows are the sibling `{path.stem}.ngram`), "
            f"or it is not this model's checkpoint at all. This run is configured "
            f"engram_ram=false, so the memory could not be loaded: a host-table "
            f"load switches the model to host tables after the optimizer exists, "
            f"and the run would read a memory it never steps and never saves. "
            f"Escape: --set engram_ram=true (the pretrain's substrate), or an init "
            f"checkpoint whose model half carries the tables."
        )
    if tables_in_file and cfg.engram_ram:
        loud(
            f"init-from: {path} carries its Engram in the model half "
            f"(`engram.tables`) but this run is configured engram_ram=true. "
            f"Switching to host tables would DROP the checkpoint's memory — the "
            f"host path starts from the model's own rows and this run's `.ngram` "
            f"sibling does not exist yet. Escape: --set engram_ram=false, or "
            f"re-save the init as a host-table run (with its `.ngram` sibling)."
        )
    return dataclasses.replace(cfg, engram_ram=host, out_dir=str(path.parent), ckpt_name=path.stem)


def init_controller(run: RunConfig, model: torch.nn.Module, path: Path, *, log=print) -> RunConfig:
    """`--init-from` carries the checkpoint's LFB coefficient, LOUD.

    The LFB bias is routing-CONTROLLER state: `SparseMoE.forward` adds it to the
    router logits unconditionally and `_lfb_step` moves it by `±moe_lb_coef` per
    routed forward from the CUMULATIVE counts. It is neither a parameter nor part
    of the model half (`lfb_bias` is a non-persistent buffer), so **no learning
    rate freezes it** — the reason an `lr=1e-12` run still degrades. A cold
    integrator started over a converged router has no feedback: the top-k is
    pinned by weights the bias does not reach until it is several logits tall,
    the sign never flips, and the bias ramps linearly. Measured on this seam
    (mix1 + `train medium`'s `moe_lb_coef=0.01`, weights frozen): ±4.8 after
    ~480 forwards, routing onto experts the model never selects, loss
    1.275 → 1.500 → 11 425 785 by step 300 (`logs/diag_lr0.log`). The same
    stream with `moe_lb_coef=0` reads 1.21-1.49 for all 400 steps.

    `--init-from` deliberately has no config-drift gate (that is `resume`'s job;
    the QAT seam exists to change the forward), so before this the PRESET's
    coefficient silently decided the controller — `train medium` says 0.01
    while mix1 trained at 0. The init file's snapshot is the authority, as it is
    for the engram substrate: carry its coefficient, LOUD. A checkpoint with no
    snapshot cannot say what controller it trained with, so a run asking for a
    live one is refused rather than ramped over a converged router;
    `moe_lb_coef=0` is the escape (a controller that never moves the bias).
    """
    snap = path.with_suffix(".config.toml")
    if not snap.is_file():
        if run.model.moe_lb_coef != 0.0:
            loud(
                f"init-from: {path} has no {snap.name} snapshot, so the LFB "
                f"coefficient it trained with is unknown, and this run has "
                f"moe_lb_coef={run.model.moe_lb_coef:g}. The bias is controller state "
                f"(not a parameter, not in the model half), so no lr freezes it: a "
                f"cold integrator over a converged router winds up (measured: loss "
                f"1.275 -> 11 425 785 by step 300 with the weights frozen). Escape: "
                f"--set moe_lb_coef=0 leaves the routing exactly the checkpoint's, or "
                f"use an init checkpoint that carries its snapshot."
            )
        return run
    stored = RunConfig.from_snapshot(snap.read_text())
    coef = float(stored.model.moe_lb_coef)
    if coef == float(run.model.moe_lb_coef):
        return run
    log(
        f"init: moe_lb_coef {run.model.moe_lb_coef:g} -> {coef:g} from {snap.name} — "
        f"the LFB bias is routing-controller state that is not a parameter and not "
        f"in the model half, so a cold integrator over this checkpoint's router "
        f"winds up (measured loss 1.275 -> 11 425 785 by step 300 with the weights "
        f"frozen). The checkpoint's own recipe decides it."
    )
    model_cfg = dataclasses.replace(run.model, moe_lb_coef=coef)
    for m in model.modules():
        if hasattr(m, "lfb_coef"):
            m.lfb_coef = coef
    if hasattr(model, "cfg"):
        model.cfg = model_cfg
    return dataclasses.replace(run, model=model_cfg)


def load_model_state(model: torch.nn.Module, cfg: TrainConfig, sd: dict, *, strict: bool = True):
    """The ONE way a checkpoint's model half reaches a model.

    `train`'s resume, `eval`, `decode`, `serve`, `scripts/chat.py` and the two
    decode benches all call this, so "how an `engram_ram` checkpoint is loaded"
    is defined once. `sd` is the already-unpacked model half (`load_model_half`
    or `unpack_state_dict`), so a storage rung is the caller's read.

    The host-table rule, and why it is not a plain `load_state_dict`: on that
    path the tables are not a parameter (`Engram.to_host` deletes
    `engram.tables`), so the model must be switched to host tables BEFORE the
    load — otherwise a fresh in-VRAM model refuses `Missing key(s):
    "loop.engram.tables"` — and the sibling `<ckpt>.ngram` is restored after.
    `enable_host_tables` (not `host_tables`) is the call: the CUDA-only refusal
    is a training VRAM-budget rule, and a load must work on `--device cpu`.
    A missing sibling is LOUD (`require_ngram`), never freshly seeded rows.

    The LFB rule is the same shape: `lfb_bias` is a non-persistent buffer, so
    it travels in the checkpoint's `moe` section and is copied here — after
    `load_state_dict`, because the buffers are the live module's. A half that
    says it came from a TRAINING load and carries `lfb_counts` without the
    bias is refused when the controller is live (`_refuse_missing_lfb`);
    a reader (`load_model_half`) has already COUNTED the zero-bias line.
    """
    if cfg.engram_ram:
        require_ngram(cfg)
        enable_host_tables(model, bias_correction=cfg.engram_bias_correction)  # idempotent
    moe = getattr(sd, "moe", None)
    refuse = bool(getattr(sd, "refuse_missing_moe", False))
    result = model.load_state_dict(sd, strict=strict)
    if moe is not None:
        _restore_lfb(model, moe, where=f"{len(moe)}-module `moe` section")
    elif refuse and _carries_lfb_counts(sd):
        # The EFFECTIVE coefficient is the model's (the preset's, or the
        # snapshot's after `init_controller` rewrote it) — not the CLI's guess.
        coef = max((float(m.lfb_coef) for m in _lfb_modules(model).values()), default=0.0)
        if coef != 0.0:
            _refuse_missing_lfb("the checkpoint", coef)
    if cfg.engram_ram:
        load_host_state(model, load_ngram(cfg))
    return result


def load_ckpt(run: RunConfig) -> tuple[dict, RunConfig]:
    """Load a checkpoint, refusing a config that differs from the snapshot.

    The exemption list is the progress keys, and nothing else: they decide WHEN
    we look (how many steps, how often to log/save/evaluate) and WHERE the
    telemetry goes, never what the model computes. `lr` is NOT exempt — a
    different lr is a different run.
    """
    weights, snapshot = ckpt_paths(run.train)
    if not weights.is_file() or not snapshot.is_file():
        loud(
            f"resume: {weights} and {snapshot} must both exist (the snapshot is "
            f"what makes a resume checkable). Escape: start without --resume, or "
            f"fix ckpt_name/out_dir."
        )
    if run.train.engram_ram:
        require_ngram(run.train)
    stored = RunConfig.from_snapshot(snapshot.read_text())
    drift = run.diff_keys(stored)
    if drift:
        loud(
            f"resume refused: {snapshot} was written by a config that differs in "
            f"{drift}. Only the progress keys {list(PROGRESS_KEYS)} may change on a "
            f"resume (ADR-0021) - everything else decides what the model computes, "
            f"so continuing under one step count would be a different experiment. "
            f"Escape: resolve the run with the stored values, or use a new ckpt_name."
        )
    ck = dict(torch.load(weights, map_location="cpu", weights_only=False))
    # Storage rungs: the model half unpacks to fp32 masters (eval/decode/
    # resume all ride the same path) — and the `_orig_mod.` prefix of a
    # pre-strip torch.compile file is dropped there too, so ONE function is the
    # whole reader contract (audit CRITICAL 2 + MAJOR 6). `quant` on the LOAD
    # side is allowed to differ from the save side — the file's rung is what
    # describes the values, and a checkpoint is legal evidence for a bf16 run
    # starting from fp8-rounded init; the snapshot gate above handles intent.
    ck["model"] = _with_moe(
        unpack_state_dict(ck["model"], ck.get("model_quant_rung")),
        ck.get("moe"),
        refuse=True,
    )
    # The EMA copy travels beside the raw one (`save_ckpt`). It is unpacked by
    # the same rule, and a file without it is a legal resume: the run starts
    # the shadow from the loaded weights (COUNTED in `train`'s log).
    if ck.get("model_ema") is not None:
        ck["model_ema"] = unpack_state_dict(ck["model_ema"], ck.get("model_quant_rung"))
    return ck, stored


def _sample_depth(seed: int, step: int, max_iter: int) -> int:
    """The ADAPTIVE DEPTH arm's depth for one step: `1 + ((step·2654435761 + seed) % max_iter)`.

    A PURE function of `(seed, step)`: same arguments, same `T`, in any process
    and before or after a resume. Not a draw from a global RNG, because burn's
    RNG is never seeded — a sampled depth would differ between two runs of one
    config and a resume would continue a different objective (ADR-0021, the
    defect that fixed DSpark's stride). `2654435761` is the 32-bit golden-ratio
    odd constant (Knuth's multiplicative hash), so consecutive steps land on
    different residues in `1..=max_iter`.
    """
    assert max_iter >= 1, f"max_iter must be >= 1, got {max_iter}"
    return 1 + ((step * 2654435761 + seed) % max_iter)


def _shadow_state(model: torch.nn.Module, initial: dict | None = None) -> dict:
    """The EMA shadow: one tensor per PARAMETER, keyed by the live name.

    `initial` is a checkpoint's `model_ema` half (already unpacked, CPU, keys
    WITHOUT the `_orig_mod.` prefix `torch.compile` adds). Keys are matched
    after stripping that prefix on both sides, and a set or shape mismatch is
    LOUD: a shadow that names a different tree would make every eval score a
    network the run never trained. `None` starts the shadow AT the current
    weights (a fresh run, or a resume of a file written with `ema_decay=0`).

    The shadow covers `named_parameters()` and nothing else — buffers and the
    host Engram tables are not weights of the checkpoint's model half, and
    `ema_decay`'s docstring names that boundary.
    """
    live = list(model.named_parameters())
    if initial is not None:
        want = [n.removeprefix("_orig_mod.") for n, _ in live]
        missing = sorted(set(want) - set(initial))
        extra = sorted(set(initial) - set(want))
        if missing or extra:
            loud(
                f"ema: the checkpoint's `model_ema` does not name this parameter "
                f"tree — missing {missing[:5]}{' ...' if len(missing) > 5 else ''}, "
                f"unexpected {extra[:5]}{' ...' if len(extra) > 5 else ''}. The "
                f"shadow is a second copy of the SAME model; evaluating a mismatched "
                f"one would score a network this run never trained. Escape: resume "
                f"with the checkpoint's own config, or start a fresh ckpt_name."
            )
    out: dict[str, torch.Tensor] = {}
    for name, p in live:
        src = None if initial is None else initial[name.removeprefix("_orig_mod.")]
        if src is not None and tuple(src.shape) != tuple(p.shape):
            loud(
                f"ema: `model_ema` {name} has shape {tuple(src.shape)}, the model "
                f"expects {tuple(p.shape)}. Escape: the checkpoint's config, or a "
                f"fresh ckpt_name."
            )
        out[name] = (
            p.detach().clone()
            if src is None
            else src.detach().to(device=p.device, dtype=p.dtype).clone()
        )
    return out


@contextmanager
def _shadow_params(model: torch.nn.Module, shadow: dict | None):
    """Evaluate with the shadow: swap `.data` for the eval, restore after.

    A reference swap, not a copy: an eval is a declared cadence, but copying a
    1B-parameter shadow in and out of the live tensors twice per eval is the
    kind of cost that makes an arm look slow for no reason. `.data` assignment
    does not disturb the optimizer's parameter objects (the objects, their
    `.grad` and their identity are untouched), which is why the switch back is
    exact and the next step's graph sees the RAW weights it built.
    """
    if shadow is None:
        yield
        return
    live = list(model.named_parameters())
    saved = {name: p.data for name, p in live}
    for name, p in live:
        p.data = shadow[name]
    try:
        yield
    finally:
        for name, p in live:
            p.data = saved[name]


def _ema_update(model: torch.nn.Module, shadow: dict, decay: float, flag_dev: torch.Tensor) -> None:
    """`shadow <- d·shadow + (1-d)·param`, ONCE per optimizer step.

    `flag_dev` is the firewall's 0/1 device flag, so a NaN-skipped step leaves
    the shadow EXACTLY where it was (`lerp` with weight 0) and no host branch on
    a device value exists. `lerp_` is the in-place form of the recurrence; it
    reads the parameter AFTER the step (or after the graph replay), which is
    why `opt_graph` is legal with this arm — the EMA is outside the capture.
    """
    weight = flag_dev.to(torch.float32) * (1.0 - decay)
    for name, p in model.named_parameters():
        s = shadow[name]
        s.lerp_(p.detach(), weight.to(s.dtype))


def _eval_loss(
    model: torch.nn.Module,
    stream: ByteStream,
    batches: int,
    device: torch.device | str | None,
    depth: int | None,
    counters: Counters | None = None,
) -> tuple[float, int, float, list[float]]:
    """One held-out pass at ONE loop depth: (loss, bytes, fin, per_iter_mean).

    `loss` is the headline (Σ CE_i + CE_final)/(T+1) — kept for continuity but
    DEPTH-CONFOUNDED (its term count changes with T; found 2026-10-16).
    `fin` is the CE of the mean readout alone — the number comparable across
    depths and the one generation reads. `per_iter_mean` averages the
    per-iteration CEs over the batch — the deep-supervision shape.

    `counters`, when given, receives THIS eval's own arm counters (the eval
    forward's `engram_rows`/`engram_arms` — the `engram=` field the eval line
    prints). They are the eval's, not the run's: a training counter would say
    the memory ran during an eval that ran without it, which is exactly the
    blind spot of AGENTS §3.2's retracted arm.
    """
    stream.rewind()
    total = 0.0
    scored = 0
    fin_total = 0.0
    per_iter_acc: list[float] = []
    was_training = model.training
    model.eval()
    kw = {} if depth is None else {"depth": depth}
    ev_counters = Counters()
    with torch.no_grad():
        for _ in range(batches):
            x, y = stream.next_batch()
            if device is not None:
                x, y = x.to(device), y.to(device)
            # `pure=True`: the held-out number is the PREDICTIVE CE alone.
            # The aux terms (DSpark/SIGReg) are training signals; without this
            # an aux arm reads ~w*aux_term worse by construction and the A/B
            # compares two different quantities (measured on the first aux A/B
            # run: control 3.045 vs fb 3.469 while the arm was fine). Counters
            # are not bumped on this path.
            _, loss, _ = model(x, y, pure=True, breakdown=True, counters=ev_counters, **kw)
            assert loss is not None, "an eval batch must carry targets"
            total += float(loss)
            scored += int(x.numel())
            bd = getattr(model, "last_breakdown", None)
            if bd is not None:
                fin_total += bd["fin"]
                per = bd["per_iter"]
                if not per_iter_acc:
                    per_iter_acc = [0.0] * len(per)
                for i, v in enumerate(per):
                    per_iter_acc[i] += v
    n = max(batches, 1)
    if getattr(stream, "eval_tail_capped", None) is not None:
        # The tail is capped below the requested window, so the stream wrapped
        # and `scored` above counts duplicate rows. The printed figure is the
        # DISTINCT bytes those windows scored (the cap's own COUNTED line
        # names the requested/granted sizes).
        scored = distinct_scored_bytes(stream, batches)
    model.train(was_training)
    if counters is not None:
        counters.merge(ev_counters)
    return (
        total / n,
        scored,
        fin_total / n,
        [v / n for v in per_iter_acc],
    )


def evaluate(
    model: torch.nn.Module,
    stream: ByteStream,
    batches: int,
    device: torch.device | str | None = None,
    *,
    depths: bool = False,
    counters: Counters | None = None,
) -> tuple[float, int] | dict[int, tuple[float, int]]:
    """Mean held-out CE over `batches` windows + the BYTES those windows scored.

    `rewind()` first, always: an eval that inherits the stream position is not
    comparable with the previous one (AGENTS §2.6). No autodiff — burn's eval
    runs never-backwarded graphs to OOM on the card.

    `depths=True` is the ADAPTIVE DEPTH arm's measurement, the plan's
    `--eval-depths`: one pass per depth `1..=max_iter` and a dict
    `{T: (ce, bytes)}` back — the depth-efficiency curve. It trains nothing
    (no optimizer, no `no_grad` gap) and needs a model that carries its own
    `cfg.max_iter`.
    """
    if depths:
        return {
            d: _eval_loss(model, stream, batches, device, d, counters)
            for d in range(1, model.cfg.max_iter + 1)
        }
    return _eval_loss(model, stream, batches, device, None, counters)


def _eval_line(
    model: torch.nn.Module,
    stream: ByteStream,
    cfg: TrainConfig,
    max_iter: int,
    device: torch.device | str | None,
    step: int,
) -> tuple[str, float, int]:
    """The eval line, carrying the depth curve when the config asks for it.

    `cfg.rand_depth` turns the curve on (the arm must show its own curve);
    `cfg.eval_depths` requests it on a fixed-depth run. The line reports the
    held-out loss at `max_iter` as its headline number, so the arm's curve and
    the fixed-depth control are read off the same line.
    """
    curve = None
    eng = Counters()
    if cfg.rand_depth or cfg.eval_depths:
        curve = evaluate(model, stream, cfg.eval_batches, device=device, depths=True, counters=eng)
        ev, scored, fin, per_iter = curve[max_iter]
    else:
        ev, scored, fin, per_iter = evaluate(
            model, stream, cfg.eval_batches, device=device, counters=eng
        )
    # `fin` = the CE of the final mean readout — the DEPTH-COMPARABLE number
    # (the headline `loss` averages T+1 terms and is not, see `_eval_loss`).
    # `engram=<rows>/<arms>` is burn's field (`lib.rs:1751`, the `ENGRAM_KEYS` /
    # `ENGRAM` probe deltas) over THIS eval's own forwards: `engram=0/<n>` is how
    # a reader learns the eval ran a memory-disabled forward, which is the defect
    # that retracted the Engram arm's numbers (AGENTS §3.2). It is printed on
    # both substrates — the host path counts the same rows in `Engram._read`.
    line = (
        f"eval step={step} loss={ev:.4f} bpb={bpb(ev):.3f} fin={fin:.4f} "
        f"bpb_fin={bpb(fin):.3f} over {scored} B "
        f"({cfg.eval_batches} x {cfg.batch} x {cfg.seq_len}) "
        f"engram={eng.get('engram_rows', 0)}/{eng.get('engram_arms', 0)}"
    )
    capped = getattr(stream, "eval_tail_capped", None)
    if capped is not None:
        line += f" eval_tail_capped={capped[0]}/{capped[1]}"

    if per_iter:
        line += " periter=" + ",".join(f"{v:.4f}" for v in per_iter)
    if curve is not None:
        line += " depths=" + ",".join(f"{d}:{curve[d][2]:.4f}(fin)" for d in sorted(curve))
    return line, ev, scored


def train(
    model: torch.nn.Module,
    run: RunConfig,
    *,
    stream: ByteStream | None = None,
    eval_stream: ByteStream | None = None,
    resume: bool = False,
    init_from: str | None = None,
    log=print,
    device: torch.device | str | None = None,
) -> dict:
    """Run the training loop. Returns a report dict (final loss, counters, step).

    `device` is the batch seam: the stream hands out CPU tensors (its bytes come
    from an mmap, and it is the caller who knows where the model lives), so a
    GPU run names the device here and gets its windows transferred. `None` is
    the CPU path, unchanged.
    """
    cfg = run.train
    # The hand-built path skips `load_run`; the train-half refusals are ONE
    # definition (`config.validate_train`) and this is the second caller.
    validate_train(cfg)
    if cfg.tf32:
        # The knob the b1 opt bench measured (2026-10-16): 1399 -> 1201 ms/step.
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")
    # The 2-D GEMM COMPUTE mode (`tsct.set_mm_dtype`), the `tf32` knob's twin: a
    # process-wide torch setting, applied once at the top of a run. LOUD when it
    # is `fp8`, because on torch 2.13 that lane is FORWARD-ONLY: the training
    # forward's operands require grad and `_scaled_mm` has no backward, so every
    # training forward takes the COUNTED bf16 fallback and a reader has to know
    # that before reading the loss curve.
    set_mm_dtype(cfg.mm_dtype)
    kda.set_disable_recompute(cfg.kda_disable_recompute)
    if cfg.kda_disable_recompute:
        log(
            "kda_disable_recompute=true: FLA's chunk_kda SAVES its backward "
            "intermediates instead of recomputing them (their docstring: for small "
            "training at the cost of memory). Same math, more VRAM, less recompute; "
            "default false = every number on record."
        )
    if cfg.mm_dtype == "fp8":
        log(
            "mm_dtype=fp8: the 2-D GEMMs run e4m3 `torch._scaled_mm` where autograd "
            "is off. torch 2.13's `_scaled_mm` has NO backward (measured: "
            "`RuntimeError: derivative for aten::_scaled_mm is not implemented`), so "
            "this run's TRAINING forward falls back to bf16 COUNTED (`mm_fp8_fallback` "
            "on the log line, the reason on `counters.why`) while its no-grad eval "
            "forwards run fp8. Escape: mm_dtype=bf16 (the default)."
        )
    device = None if device is None else torch.device(device)
    if stream is None:
        if run.train.mix_plan:
            # The staged mixture (train/mixture.py): N sources interleaved by
            # per-stage weights, epoch-capped LOUD. Its counter is MICRO-BATCHES:
            # the loop draws `grad_accum` windows per optimizer step, so the
            # schedule and the epoch cap both take `steps x grad_accum` (passing
            # `steps` ran the mix `grad_accum`x fast — stages at opt steps 18/23
            # instead of 70/90 — and under-counted bytes by the same factor).
            micro_steps = run.train.steps * run.train.grad_accum
            stream = open_mixture(
                run.train.mix_plan,
                steps=micro_steps,
                batch=run.train.batch,
                seq_len=run.train.seq_len,
                seed=run.train.seed,
                max_epochs=run.train.max_epochs,
                doc_masking=run.train.doc_masking,
            )
            _, eval_stream = open_streams(run, eval_only=True)
            log(
                f"mix_plan={run.train.mix_plan}: {len(stream.plan.sources)} sources, "
                f"{len(stream.plan.stages)} stage(s), steps={micro_steps} micro-batches "
                f"({run.train.steps} x grad_accum={run.train.grad_accum}), "
                f"max_epochs={run.train.max_epochs}"
            )
        else:
            stream, eval_stream = open_streams(run)
    if cfg.doc_masking:
        # The arm's own line: which reader ran and what it does. The row shape
        # is the whole mechanism, so a reader does not have to infer it from the
        # flag (a packed row that still glued documents would be the defect).
        log(
            "doc_masking=true: one document per training row (whole document, or a "
            "contiguous chunk of one when it is longer than seq_len), tail-padded; "
            "padding and cross-document targets are excluded from the loss. The eval "
            "stream stays flat (the held-out window is the comparable number)."
        )
    model.train()
    if "ternary_s3t" in (run.model.factor_qat, run.model.dense_qat):
        # The ADR-0021 caveat, said where a reader of the log meets it. The
        # projector has no step index, so the S3T mask is a process-local
        # generator's stream (seeded from `factor_qat_seed`): reproducible
        # within a run, NOT continued by a resume. A degradation a reader
        # cannot see is a defect (`tsct.factor_qat_project`).
        log(
            f"ternary_s3t (factor_qat={run.model.factor_qat}, "
            f"dense_qat={run.model.dense_qat}): the S3T Bernoulli mask is a "
            f"process-local stream seeded {run.model.factor_qat_seed} — a resumed "
            f"run restarts it, so its masks differ from an unbroken run's. "
            f"Reproducible within one run only."
        )
    if cfg.engram_ram:
        # BEFORE the optimizer exists: on the host path the tables are not
        # parameters, so no group, no grad clip and no checkpoint half ever
        # carries them (`Engram.to_host` deletes the parameter).
        n = host_tables(model, cfg.engram_bias_correction)
        log(
            f"engram_ram: {n} table set(s) in host RAM, VRAM holds the batch's unique "
            f"rows, bias_correction={'on' if cfg.engram_bias_correction else 'off'}"
        )
    opt = MixOptimizer(
        model,
        lr=cfg.lr,
        wd=cfg.wd,
        opt=cfg.opt,
        ns_steps=cfg.ns_steps,
        ns_kind=cfg.ns_kind,
        colrow_3d=cfg.muon_colrow_3d,
        # The two optimizer rungs the bench named (2026-10-17). Without these
        # kwargs a `--set muon_state_dtype=bf16` would be INERT in a train run
        # — the SILENT class the repo's rules forbid (the wiring gap the
        # optimizer lane's own report flagged). Both default fp32 = the
        # pre-rung behaviour, so a resume of an older snapshot is unaffected.
        muon_state_dtype=cfg.muon_state_dtype,
        ns_dtype=cfg.ns_dtype,
        # The two opt-in direction / schedule arms (2026-10-17). Both default
        # off (`muon` / false), so a resume of an older snapshot and every
        # number on record are unaffected; the snapshot pins them, so a flip on
        # a resume is refused by `diff_keys`.
        muon_variant=cfg.muon_variant,
        muon_sf=cfg.muon_sf,
        muon_sf_wd=cfg.muon_sf_wd,
    )
    if cfg.optim_8bit:
        # The device firewall multiplies the FIRST MOMENT and the weight-decay
        # factor before the step; bitsandbytes' AdamW8bit keeps its state in
        # 8-bit blocks and its `step` has no seam between that state and the
        # parameter write, so the flag could only be honoured by a host branch
        # on the device value — the sync this path exists to remove. LOUD, and
        # the escape is the plain AdamW this arm was a VRAM lever against
        # (the follow-up is a snapshot-gated bnb arm, ADR-0019 style).
        loud(
            "optim_8bit=true: bitsandbytes' AdamW8bit has no device-flag path "
            "(its update cannot be multiplied by the NaN firewall's flag), so a "
            "non-finite step could not be skipped without a host read. Escape: "
            "optim_8bit=false (plain AdamW keeps the device firewall), or gate "
            "this arm on its own host-read path first."
        )

    def sf(mode: str) -> None:
        """The schedule-free switch, a no-op when the arm is off.

        `sf_switch` is LOUD on an optimizer whose arm is off (a silent no-op is
        the ADR-0019 failure class), so the trainer asks the CONFIG, not the
        optimizer, and one helper keeps every call site identical.
        """
        if cfg.muon_sf:
            opt.sf_switch(mode)

    sf("train")
    if cfg.muon_sf:
        log(
            "muon_sf: the params hold Y=(1-beta1)Z+beta1 X during every step, X is "
            "what each eval and checkpoint sees, and the Muon-group LR is CONSTANT "
            "after warmup (the AdamW half keeps WSD - the hybrid deviation "
            "`optim.Muon.sf_switch` documents). Weight decay is "
            f"{cfg.muon_sf_wd:g} AT Z (Lemma 3.1 needs wd>0 there; `config.validate` "
            "refuses 0); the per-step TSCT retraction is OFF "
            "(muon_sf_retract=false - the only legal setting, S3). A NaN-skipped step "
            "advances none of Z/X/s."
        )
    tokens_per_step = cfg.batch * cfg.seq_len * cfg.grad_accum
    start = 0
    resumed_ema: dict | None = None
    if init_from is not None:
        # Fine-tune init: WEIGHTS from a foreign checkpoint, no config gate and no
        # optimizer state. The deliberate difference from `resume`, and the reason
        # this exists: a resume must prove the config did not move (ADR-0021), so
        # it cannot express "same weights, different forward" — which is exactly
        # what a QAT-finish is (`factor_qat=none` bf16 checkpoint -> ternary
        # forward). Weights leave their recorded context, so it is LOUD.
        if resume:
            loud(
                "train: --init-from and --resume together name two different sources "
                "of weights. Escape: --init-from to seed a new run from a foreign "
                "checkpoint (fresh optimizer, fresh snapshot), or --resume to continue "
                "THIS run."
            )
        path = Path(init_from)
        if not path.is_file():
            loud(f"train --init-from: {path} does not exist. Escape: the .pt of the run you mean.")
        foreign = torch.load(path, map_location="cpu", weights_only=False)
        # An EMA checkpoint's SERVED half is `model_ema` (the same preference
        # `load_model_half` makes): a fine-tune starts from the averaged model a
        # reader would serve, not from the raw iterate.
        half = foreign.get("model_ema")
        if half is None:
            half = foreign.get("model", foreign)
        sd = unpack_state_dict(half, foreign.get("model_quant_rung"))
        # The LFB coefficient is the CHECKPOINT's (init_controller), and it
        # decides whether a missing `moe` section is a refusal: resolving the
        # controller before the load is what lets the escape it documents
        # (`moe_lb_coef=0` carried from the snapshot) actually apply.
        run = init_controller(run, model, path, log=log)
        sd = _with_moe(sd, foreign.get("moe"), refuse=True)
        # The ONE loader: an `engram_ram` checkpoint has no `engram.tables` in the
        # model half and needs its `.ngram` sibling, which `load_model_state`
        # restores (a missing sibling is its LOUD refusal, never fresh rows).
        load_model_state(model, init_cfg(cfg, path, sd, model), sd)
        log(
            f"init: weights from {path} (step {foreign.get('step', '?')}), "
            "fresh optimizer and snapshot — the forward may differ from the one "
            "that trained them (this is the QAT-finish seam)"
        )
    elif resume:
        ck, _stored = load_ckpt(run)
        load_model_state(model, cfg, ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        resumed_ema = ck.get("model_ema") if cfg.ema_decay > 0.0 else None
        if start:
            # A resumed run must continue on UNSEEN bytes: the burn failure
            # this prevents is re-reading (and memorizing) the corpus head.
            # `grad_accum` micro-batches were drawn per optimizer step, and
            # `MixtureStream.skip_bytes` replays exactly that many draws.
            stream.skip_bytes(start * (cfg.seq_len + 1) * cfg.batch * cfg.grad_accum)
            if isinstance(stream, MixtureStream):
                assert stream.step == start * cfg.grad_accum, (
                    f"resume skipped {stream.step} micro-batches, expected "
                    f"{start} x grad_accum={cfg.grad_accum}"
                )
        # The checkpoint's model half is the X iterate (a checkpoint is written
        # with the eval switch on, below), and so is the optimizer's own `x`, so
        # the params must go back to Y before the next forward. The switch is an
        # ABSOLUTE write from the optimizer's state, which is why this is the
        # same call as the post-construction one and cannot double-lerp.
        sf("train")
        log(f"resume: step={start} opt_steps={opt.steps}")

    warm = warmup_steps(cfg.warmup_tokens, cfg.batch * cfg.grad_accum, cfg.seq_len)
    decay = int(round(cfg.decay_frac * cfg.steps))
    if cfg.torch_compile:
        if cfg.opt_graph:
            loud(
                "torch_compile=true with opt_graph=true: a recompile attempted while "
                "a CUDA graph capture is active aborts the capture, and torch.compile "
                "is neither needed nor wanted under a graph (the graph already removes "
                "the host dispatch). Escape: torch_compile=false, or opt_graph=false."
            )
        # torch.compile(dynamic=True) — the spike's measured winner at the b1
        # shape (273.5 vs 357.3 ms/step, 1.20x). Wrapped AFTER the resume
        # load: an OptimizedModule's state_dict carries a `_orig_mod.` prefix,
        # so loading into the wrapped model would rename every key. The
        # inductor cache persists in ~/.cache/torch — the warmup is once per
        # machine, not per run.
        model = torch.compile(model, dynamic=True)
        log("torch_compile: model wrapped (first steps pay the warmup)")
    # The EMA shadow is built AFTER the resume load AND after torch.compile:
    # it must clone the RESOLVED weights, and its keys must be the live
    # (possibly `_orig_mod.`-prefixed) names `_shadow_params` swaps. `init_from`
    # deliberately starts a FRESH shadow (a fine-tune is a new trajectory), so
    # only a resume restores the stored copy.
    ema = None
    if cfg.ema_decay > 0.0:
        ema = _shadow_state(model, resumed_ema)
        source = (
            "restored from model_ema"
            if resumed_ema is not None
            else "started at the loaded weights"
        )
        log(
            f"ema_decay={cfg.ema_decay:g}: shadow of {len(ema)} parameter tensors "
            f"{source}; eval and the served checkpoint use it, the raw iterate stays "
            f"the training copy (`model_ema` vs `model` in the file)"
        )
    anneal_steps = cfg.factor_qat_anneal_steps
    if anneal_steps > 0 and run.model.factor_qat == "none" and run.model.dense_qat == "none":
        loud(
            f"factor_qat_anneal_steps={anneal_steps} with factor_qat='none' and "
            f"dense_qat='none': there is no quantizer to anneal toward (the ONE "
            f"alpha drives both paths). Escape: set factor_qat "
            f"(ternary|binary|mxfp4|ternary_s3t) or dense_qat, or leave the ramp "
            f"at 0."
        )
    total = Counters()
    window = Counters()
    nan_skipped = 0
    drowning = 0
    window_steps = 0
    timer = StepTimer()
    last_loss = float("nan")
    #: The DEVICE firewall's state (ADR-0018 rule 2). `flag_dev` is the 1/0
    #: integer the step's writes are multiplied by; `last_loss_dev` is the last
    #: FINITE loss, carried on the device so the log cadence is the only host
    #: read of it. Both are persistent storage filled IN PLACE: a CUDA graph
    #: captured later reads this tensor's storage on every replay, so
    #: reassigning the name would freeze the capture-time value (the same
    #: defect `graph.set_lr` documents for the learning rate).
    dev = next(model.parameters()).device
    flag_dev = torch.ones((), dtype=torch.int32, device=dev)
    last_loss_dev = torch.full((), float("nan"), device=dev)
    #: The loss-print decision, separate from the UPDATE flag: the printed
    #: number is the last FINITE loss (`last_loss_dev`'s contract), and a
    #: micro-batch can have a finite loss with a non-finite gradient. ON by
    #: default so a one-micro step hands `torch.where` the same condition the
    #: pre-arm code did.
    loss_ok_dev = torch.ones((), dtype=torch.bool, device=dev)
    #: Host-substrate Engrams, resolved once: this arm's update needs the flag
    #: on the HOST and pays an existing per-step D2H for its row gradients, so
    #: it is the one declared exception to "no per-step device read".
    hosts = host_engrams(model)
    #: The telemetry sink: every event (log cadence, evals, checkpoints) gets
    #: a JSON row. A dict writer that never raises on non-JSON values.
    sink = None
    if cfg.json_log:
        sink = open(cfg.json_log, "a", encoding="utf-8")

        def _row(**kw) -> None:
            import json

            sink.write(
                json.dumps({k: v for k, v in kw.items() if v is not None}, default=str) + "\n"
            )
            sink.flush()

        _row(
            event="run",
            source=run.source,
            steps=cfg.steps,
            batch=cfg.batch,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            opt=cfg.opt,
            grad_accum=cfg.grad_accum,
            ema_decay=cfg.ema_decay,
            resumed=start,
        )

    def opt_stage() -> None:
        """ONE optimizer stage: the grad clip, then the mixed step.

        Defined as one callable so the CUDA-graph arm captures EXACTLY this and
        nothing about the loop (`train/graph.py`), and so the eager and captured
        paths cannot drift into two definitions of "the stage".

        The clip is `clip_grads` (optim.py): ONE `_foreach_norm` + ONE fused
        `_foreach_mul_` — the stock per-tensor walk measured 252 ms of a
        small-form step (2920 norm calls, profile 2026-10-16).

        `flag_dev` is read INSIDE the captured region only as a device tensor,
        which is what keeps the stage capture-safe: the loop fills its storage
        in place before the replay, exactly like `graph.set_lr`'s learning
        rate. No `.item()`, no `bool(tensor)`, no host branch on it.
        """
        if cfg.grad_clip > 0:
            clip_grads(model.parameters(), cfg.grad_clip)
        opt.step(flag_dev)

    #: The CUDA-graph arm (`train/graph.py`, `train.opt_graph`). `None` is the
    #: default path and it is untouched: the LR stays a host float in the group
    #: dict, `opt_stage()` is called directly, and `zero_grad` keeps
    #: `set_to_none=True`. A refusal or a capture failure falls back to EAGER for
    #: the rest of the run, COUNTED as `opt_graph_fallback` + the reason
    #: (ADR-0019) — this arm must never kill a run.
    #:
    #: Built HERE, after the resume load, on purpose: `Optimizer.load_state_dict`
    #: REPLACES `param_groups` with the checkpoint's copies (`torch/optim/
    #: optimizer.py`), which carry a plain float `lr` and the checkpoint's own
    #: `capturable`. A wrapper built before the load would hold LR tensors no
    #: group points at any more — every replay would read a stale value.
    graph = None
    if cfg.opt_graph:
        graph = OptimizerGraph(opt, opt_stage, counters=total, log=log, allow_frozen=True)
        log(
            f"opt_graph: the optimizer stage is captured after {graph.warmup_steps} "
            f"eager steps and replayed with ONE launch per step; the LR lives in a "
            f"device tensor this loop fills IN PLACE (a host float would be baked at "
            f"capture and freeze the WSD schedule), and zero_grad runs "
            f"set_to_none=False so the captured ops keep their pinned .grad buffers."
        )

    def micro_step(x, y, kw, n_micro: int, first: bool):
        """ONE micro-batch: forward + the DEVICE firewall + backward + the gate.

        The ONE definition of the step body, called by the eager accumulation
        loop AND recorded verbatim by the `full_graph` capture — two copies of
        this would be two definitions of "a training step" and would drift
        (`train/graph.py`'s own rule). Returns `(counters, contribution,
        grads)`: the HOST counter merge and the loss accumulator stay with the
        caller, because they are the parts a capture cannot carry.
        """
        logits, loss, step_counters = model(x, y, **kw)
        assert logits.shape[-1] == run.model.vocab, "logits width != vocab"
        # THE DEVICE FIREWALL (ADR-0018 rule 2). No host read decides this
        # step: `loss_finite` and the gradient norm stay on the device, the
        # masked loss makes backward's seed exactly zero, and the resulting
        # 0-dim 1/0 flag is what every write below is multiplied by. The host
        # sees the loss and the skip count at the log cadence only.
        #
        # `torch.where` on the scalar, not `mask_fill`: on a healthy step the
        # value AND the gradient are bit-identical to `loss.backward()` (the
        # where's backward passes the root gradient through unchanged), which
        # is what keeps a healthy run's sha256 equal to HEAD's.
        assert loss is not None, "a training step must carry targets"
        loss_d = loss.detach()
        loss_finite = torch.isfinite(loss_d).all()
        step_loss = torch.where(loss_finite, loss, loss_d.new_zeros(()))
        if n_micro > 1:
            # The MEAN over micro-batches, not their sum: the accumulated
            # gradient must be the wide batch's MEAN gradient, which is what
            # makes `grad_accum=k, batch=b` match `batch=b*k`.
            step_loss = step_loss / n_micro
        if step_loss.requires_grad:
            # A real training loss always does. A model that returns a bare
            # tensor (the firewall's own NaN fixture) has nothing to backprop
            # through, which is the old code's skip branch by another route.
            step_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        # A non-finite GRADIENT disables the step too (the burn 2026-09-27
        # lesson: a masked loss scalar does not stop a NaN that came from an
        # ACTIVATION — `NaN * 0` is still NaN through the Jacobian). One vector
        # norm answers "any" without a host read; the flag is the AND of both.
        raw_norm = grad_norm(model.parameters())
        if raw_norm.device != loss_d.device:
            raw_norm = raw_norm.to(loss_d.device)
        finite = loss_finite & torch.isfinite(raw_norm)
        if n_micro > 1:
            # Sanitize THIS micro-batch's contribution on arrival — a
            # non-finite activation must not poison the accumulator — and AND
            # its flag into the group's. If ANY micro fails, the final multiply
            # below zeroes the earlier contributions too.
            flag_dev.mul_(finite.to(torch.int32))
            loss_ok_dev.logical_and_(loss_finite)
        else:
            flag_dev.copy_(finite.to(torch.int32))
            loss_ok_dev.copy_(loss_finite)
        # Sanitize on the device AND gate: every non-finite gradient becomes 0
        # (so a moment cannot be poisoned), then every gradient is multiplied
        # by the flag — `*1.0` is the identity on a healthy step, `*0` makes the
        # skipped step's direction exactly zero (a masked loss alone does not: a
        # NaN that came from an ACTIVATION would still be in the Jacobian).
        gate = finite.to(torch.float32)
        for g in grads:
            g.nan_to_num_(0.0, 0.0, 0.0)
            g.mul_(gate.to(g.dtype))
        return step_counters, step_loss.detach(), grads

    #: ---- `full_graph`: ONE graph for the whole step --------------------
    #: `cfg.full_graph` (`train/graph.py::FullGraph`): forward + backward +
    #: firewall + `zero_grad` + the optimizer stage, captured once and replayed
    #: with ONE launch per step. The refusals are LOUD and come FIRST, before a
    #: single step runs, so a config that cannot be captured says so instead of
    #: silently staying eager.
    full = None
    if cfg.full_graph:
        why = full_graph_refusal(cfg, run.model)
        if why is not None:
            loud(f"full_graph: {why}")
        if dev.type != "cuda":
            # UP FRONT, not at capture: this arm binds the LR as DEVICE tensors
            # in its constructor, and a run that is refused only when the first
            # capture fires has already moved the optimizer's LR arithmetic
            # (fp32 tensor vs Python float) for every warmup step — i.e. it is a
            # different program from the eager one it claims to fall back to.
            loud(
                f"full_graph on a {dev.type} model: a CUDA graph is a CUDA-only "
                f"mechanism, and this arm binds the LR as device tensors before the "
                f"first step, so a late fallback would not be the eager run. Escape: "
                f"--device cuda, or full_graph=false."
            )
        _captured: dict = {}

        def full_body() -> None:
            """The captured region. A capture EXECUTES it, so a host read here
            would surface at capture time (and `tests/test_full_graph.py` traps
            `Tensor.item` to prove the region is clean)."""
            opt.zero_grad(set_to_none=False)
            flag_dev.fill_(1)
            loss_ok_dev.fill_(True)
            loss_sum = last_loss_dev.new_zeros(())
            step_counters, contribution, _grads = micro_step(full.x, full.y, {}, 1, True)
            _captured["counters"] = step_counters
            loss_sum = loss_sum + contribution
            torch.where(loss_ok_dev, loss_sum, last_loss_dev, out=last_loss_dev)
            opt_stage()

        def full_counters() -> None:
            """Re-merge the counters the forward wrote.

            The forward's arm counters are Python (`Counters.count`), so a
            replay runs only its DEVICE ops and a COUNTED arm would go silent
            after the capture step (the same rule as `OptimizerGraph._replay`).
            Called on EVERY path — the eager warmup and a COUNTED fallback need
            it just as much, or the arm's counters vanish whenever it is on.
            """
            captured = _captured.get("counters")
            if captured:
                total.merge(captured)
                window.merge(captured)

        full = FullGraph(
            opt,
            full_body,
            batch=cfg.batch,
            seq_len=cfg.seq_len,
            device=dev,
            counters=total,
            log=log,
            # The trainer's setting: one structurally dead gain
            # (`loop.residual_scale` under `use_mhc`) must not disable the whole
            # arm — that refusal is why `opt_graph` measured +0 without ever
            # capturing. `graph._check_participants` catches the real hazard.
            allow_frozen=True,
            on_counters=full_counters,
        )
        log(
            f"full_graph: the WHOLE step (forward + backward + firewall + zero_grad + "
            f"the optimizer stage) is captured after {full.warmup_steps} eager steps and "
            f"replayed with ONE launch per step. The inputs live in a static "
            f"{cfg.batch}x{cfg.seq_len} device buffer this loop fills with `copy_`; the "
            f"LR is a device tensor filled IN PLACE; every `.grad` is pre-allocated and "
            f"zeroed in place. Refused arms (grad_accum>1, rand_depth, segment_len, "
            f"use_byteflow, halt, use_msa, engram_ram, jepa, dspark, act_quant, QAT, EMA) "
            f"are LOUD before the first step."
        )
    for step in range(start, cfg.steps):
        lr = wsd_lr(
            step, total_steps=cfg.steps, peak_lr=cfg.lr, warmup_steps=warm, decay_steps=decay
        )
        muon_lr = lr
        if opt.muon is not None:
            # The SF arm's whole point: the Muon group's LR is CONSTANT after
            # warmup, so the arm is horizon-free where the AdamW half is not —
            # `decay_steps=0` gives exactly the warmup ramp and then the peak,
            # from the same schedule function. The hybrid (SF-Muon + WSD-AdamW)
            # is a deliberate deviation from the paper's full-model SF, named in
            # `optim.Muon`'s docstring with full SF-AdamW as the follow-up.
            muon_lr = (
                wsd_lr(
                    step,
                    total_steps=cfg.steps,
                    peak_lr=cfg.lr,
                    warmup_steps=warm,
                    decay_steps=0,
                )
                if cfg.muon_sf
                else lr
            )

        if graph is not None:
            # The graph arm: `group["lr"]` IS a 0-dim device tensor the captured
            # ops read, so it is filled IN PLACE. `group["lr"] = lr` would replace
            # that tensor with a float no replay ever reads again — the silent
            # freeze `dion/cuda_graph.py:70-78` documents. `muon_sf` is refused at
            # capture (`train/graph.py`) and the run falls back to EAGER, but
            # `set_lr` writes EVERY group, so the Muon group would follow WSD for
            # the whole run unless the SF LR is written back after it — a silent
            # non-arm (the 2026-10-21 SF re-run's config carries the battery's
            # `opt_graph=true`). No capture can read this host float: the refusal
            # above is the only path an SF run takes.
            graph.set_lr(lr)
            if opt.muon is not None and cfg.muon_sf:
                for group in opt.muon.param_groups:
                    group["lr"] = muon_lr
        else:
            for group in opt.adamw.param_groups:
                group["lr"] = lr
            if opt.muon is not None:
                for group in opt.muon.param_groups:
                    group["lr"] = muon_lr

        # The ANNEALED factor_qat arm: burn-spectral's measured soft-to-hard
        # ramp (`ste_ternary_annealed`, "converges better than hard STE from
        # step one"). `anneal_alpha` is a PURE function of the step, so a
        # resume lands on the same alpha. `anneal = 0` (default) leaves every
        # module's constant alpha untouched.
        if anneal_steps > 0:
            model.set_factor_qat_alpha(anneal_alpha(step, anneal_steps))

        # The ADAPTIVE DEPTH arm: T is the pure `(seed, step)` sample, and the
        # loss divides by the depth actually run (`model.forward`/`loopblock`).
        # `None` (the default and every other config) keeps the call unchanged.
        # With `grad_accum > 1` the depth is sampled ONCE per optimizer step and
        # shared by every micro-batch — the same objective one wide forward
        # would run, and the reason all micro-batches read the same `step`.
        depth = _sample_depth(cfg.seed, step, run.model.max_iter) if cfg.rand_depth else None
        kw = {} if depth is None else {"depth": depth}
        if cfg.segment_len:
            # The SEGMENTED training forward: the window is cut into
            # `segment_len`-position segments and the KDA state is carried across
            # them as a graph edge (`loopblock._forward_segmented`), so a window
            # longer than one forward fits in VRAM can be trained. Same logits
            # shape, same loss. The EVAL path does not pass it: an eval saves no
            # activations, so it needs no segmentation, and leaving it one-shot
            # keeps the held-out number comparable with the record.
            kw["segment_len"] = cfg.segment_len
        # The graph arm needs STABLE `.grad` buffers: `set_to_none=True` swaps
        # each `.grad` for a fresh tensor every step, which breaks the buffers
        # capture pinned (`dion/cuda_graph.py:170-181`). `False` is a no-op for
        # a param whose grad is still None, so the FIRST backward allocates it
        # — which is why the wrapper demands at least one eager step.
        if full is not None:
            # ---- the CAPTURED step: one replay does all of it --------------
            # `doc_masking` is refused with `full_graph` in `validate`, so this
            # read is always the flat one.
            x, y = stream.next_batch()
            full.fill(x, y)
            full.set_lr(lr)
            full.step()
        else:
            # ---- the EAGER step (and `opt_graph`'s stage takeover) ---------
            opt.zero_grad(set_to_none=graph is None)
            # The group's firewall flag starts healthy and is ANDed by every
            # micro-batch below (`grad_accum > 1`); a one-micro step overwrites
            # it exactly as the pre-arm code did.
            flag_dev.fill_(1)
            loss_ok_dev.fill_(True)
            loss_sum = last_loss_dev.new_zeros(())
            n_micro = cfg.grad_accum
            for _micro in range(n_micro):
                if cfg.doc_masking:
                    # DOCUMENT MASKING (off by default): one document per row
                    # and a `[b, t]` mask, so padding and cross-document targets
                    # are not scored. `micro_step` forwards the mask through
                    # `kw`; the masked CE is a mean over selected positions, so
                    # a micro-batch with fewer scored bytes contributes less —
                    # the same semantics one wide masked batch would have.
                    x, y, loss_mask = stream.next_batch_masked()
                else:
                    x, y = stream.next_batch()
                    loss_mask = None
                if device is not None:
                    x, y = x.to(device), y.to(device)
                    if loss_mask is not None:
                        loss_mask = loss_mask.to(device)
                micro_kw = kw if loss_mask is None else {**kw, "loss_mask": loss_mask}
                step_counters, contribution, grads = micro_step(
                    x, y, micro_kw, n_micro, _micro == 0
                )
                total.merge(step_counters)
                window.merge(step_counters)
                loss_sum = loss_sum + contribution
            if n_micro > 1:
                # THE GROUP'S DECISION: one multiply, no host branch. Healthy
                # (`*1`) changes no bit; a single non-finite micro-batch zeroes
                # the whole accumulated direction, which is the same "skip the
                # step" the one-micro firewall makes.
                gate = flag_dev.to(torch.float32)
                for g in grads:
                    g.mul_(gate.to(g.dtype))
            # The last FINITE loss, carried on the DEVICE: the host reads it
            # when the cadence fires and never before. Same bits as the old
            # per-step `float(loss.detach())`.
            torch.where(loss_ok_dev, loss_sum, last_loss_dev, out=last_loss_dev)
            if graph is not None:
                # ONE launch: the captured stage IS `opt_stage()` (the grad clip
                # plus the mixed step), and the wrapper redoes the host-side step
                # counters a recording cannot carry. A skipped step replays too:
                # the flag multiplies its writes to zero, which is what makes the
                # same recording serve both outcomes with no host branch.
                graph.step()
            else:
                opt_stage()
        timer.tick()
        # COUNTED as a device counter, drained with the window's other counters
        # at the log cadence (the `Counters.count_device` 0-sync rule). OUTSIDE
        # any capture: `1 - flag_dev` is a device op on a persistent tensor, and
        # the flag it reads was written by the replay above.
        window.count_device("nan_skipped", 1 - flag_dev)
        total.count_device("nan_skipped", 1 - flag_dev)
        window_steps += 1
        if ema is not None:
            # ONCE per OPTIMIZER STEP (not per micro-batch), after the update
            # landed. `opt_graph` is legal here because this runs EAGERLY after
            # the replay, outside the capture.
            _ema_update(model, ema, cfg.ema_decay, flag_dev)
        # The parameters just moved — and a CUDA-graph replay moves them
        # WITHOUT bumping `_version` (measured 2026-10-19). `_banks` keys its
        # cache on `_version`, so with `opt_graph = true` the key froze at
        # capture and the bank served the step-5 build for a whole run
        # (experts frozen in the forward, drifting in memory, checkpoint
        # unreproducible — the medbig2/b1night defect). One call per
        # TRAINING step keeps the cache's real job (one build per step,
        # reused by the loop's iterations) and nothing more. The QAT
        # projection cache is keyed on the TENSOR, not `_version` (the same
        # graph-replay trap), so the trainer invalidates it here beside the
        # banks: one quantize per trained step, reused by every forward.
        invalidate_banks_all(model)
        invalidate_qat_cache_all(model)
        if opt.counters:
            # The Muon arms' own counters (ADR-0019): `ns_triton` when the
            # fused Newton–Schulz chain ran, `ns_triton_fallback` + the reason
            # when it could not. Merged and CLEARED here, once per step that
            # runs the optimizer, so the log line's counter field is where a
            # reader learns which chain ran. The clear is what makes the next
            # step's reading this step's, skipped or not.
            total.merge(opt.counters)
            window.merge(opt.counters)
            opt.counters.clear()
        if hosts:
            # The host-table arm, the ONE place the flag reaches the host: this
            # substrate already copies its row gradients D2H every step, so the
            # flag rides a read that exists and the table update has to know
            # `0` (drop the leaves, move nothing) or `1` (the normal update).
            # `scale=0` is the OLD skip branch exactly. With `grad_accum > 1`
            # the group's pending leaves are ALL its micro-batches, and this
            # one call is the old "once per step" contract (the leaves already
            # carry the 1/k loss scale, so the summed row gradient is the wide
            # batch's mean).
            updates, rows = host_step_all(model, lr, _host_scalar(flag_dev))
            total.count("engram_host_updates", updates)
            window.count("engram_host_updates", updates)
            total.count("engram_host_update_rows", rows)
            window.count("engram_host_update_rows", rows)

        if (step + 1) % cfg.log_every == 0:
            # The 0-sync rule: every DEVICE-side counter this window gathered
            # is read NOW, in one batched GPU→CPU transfer, at the declared
            # cadence — never per pass. The last loss is the one other read,
            # and it happens here and nowhere else.
            drained = window.drain()
            total.drain()
            nan_window = int(drained.get("nan_skipped", 0))
            if window_steps and nan_window >= window_steps:
                drowning += 1
            elif window_steps:
                drowning = 0
            window_steps = 0
            nan_skipped = int(total.get("nan_skipped", 0))
            last_loss = _host_scalar(last_loss_dev)
            parts = [
                f"step={step + 1}",
                f"loss={last_loss:.4f}",
                f"lr={lr:.3e}",
                kda_field(window),
                halt_field(model),
                mtp_field(model),
                window.line(),
            ]
            # Telemetry reads the interval BEFORE any opt-in timer report —
            # and the ORDER is load-bearing: `report()` resets the mark, so a
            # read AFTER it measures ~0 s and the JSONL's `tokens_per_s` reads
            # 1e7-1e8 (the optimizer lane measured exactly that on 2026-10-17).
            # This comment said "before" while the code did "after"; fixed.
            interval_s = timer.window_s()
            if cfg.timers:
                parts.append(timer.report(cfg.log_every * tokens_per_step))
            # The rate's denominator must be THIS WINDOW's seconds, not the
            # run's whole wall time: until the 2026-10-17 fix nothing reset the
            # mark when `--timers` was off, so `interval_s` was the CUMULATIVE
            # elapsed and `tokens_per_s` was the monotone-decaying illusion (the
            # trainer's 22k→0.9k that three real runs and the noeval bisect all
            # showed, while every in-process probe stayed flat). One mark reset
            # is the whole fix.
            timer.mark = time.perf_counter()
            rate = (cfg.log_every * tokens_per_step) / interval_s if interval_s > 0 else None
            log(" ".join(p for p in parts if p))
            if sink is not None:
                _row(
                    event="log",
                    step=step + 1,
                    loss=last_loss,
                    lr=lr,
                    tokens_per_s=rate,
                    counters=dict(window),
                    nan_skipped=nan_skipped,
                )
            window = Counters()
            if drowning >= 2:
                raise LoudError(
                    f"two consecutive windows of {cfg.log_every} steps were entirely "
                    f"non-finite (steps {step + 1 - 2 * cfg.log_every + 1}..{step + 1}); "
                    f"nan_skipped={nan_skipped}. This is the readiness plan's "
                    f"nan_windows>=2 hard error (docs/decisions/"
                    f"2026-10-15-1b-readiness-plan.md §3 Phase 1 item 1): a "
                    f"drowning run must stop for a human, not exit 0. Escape: lower "
                    f"lr, check the data, or bisect the arms."
                )

        if cfg.eval and eval_stream is not None and (step + 1) % cfg.eval_every == 0:
            # SF: the held-out number must be scored on the RETURNED iterate X,
            # not on the gradient-evaluation point Y (Alg 1 line 18 / the
            # reference's `eval()`); the params go back to Y for the next step.
            sf("eval")
            with _shadow_params(model, ema):
                line, ev, scored = _eval_line(
                    model, eval_stream, cfg, run.model.max_iter, device, step + 1
                )
            sf("train")
            log(line)
            if sink is not None:
                _row(event="eval", step=step + 1, loss=ev, bpb=bpb(ev), scored=scored)

        if (step + 1) % cfg.ckpt_every == 0:
            # A checkpoint is the model a reader would serve, so it is written
            # at X too — and on a resume the model half plus the optimizer's own
            # `x` are the same iterate, which is what makes the switch back to Y
            # an absolute, double-lerp-free write.
            sf("eval")
            weights, snapshot = save_ckpt(run, model, opt, step, ema=ema)
            sf("train")
            log(f"ckpt step={step + 1} -> {weights} ({snapshot})")

    # The report is a declared cadence: drain whatever the last (partial)
    # window gathered, and read the last finite loss. Both are host reads, and
    # both happen HERE — never inside a step.
    total.drain()
    nan_skipped = int(total.get("nan_skipped", 0))
    last_loss = _host_scalar(last_loss_dev)
    doc_counts = stream.doc_counts() if cfg.doc_masking else {}
    if cfg.doc_masking:
        log(f"doc_masking: {doc_counts}")
    if cfg.eval and eval_stream is not None and cfg.steps % cfg.eval_every != 0:
        sf("eval")
        with _shadow_params(model, ema):
            line, _ev, _scored = _eval_line(
                model, eval_stream, cfg, run.model.max_iter, device, cfg.steps
            )
        sf("train")
        log(line)
    return {
        "steps": cfg.steps,
        "loss": last_loss,
        "nan_skipped": nan_skipped,
        "counters": total,
        "opt_steps": opt.steps,
        # The shadow this run trained (`None` at `ema_decay=0`). A test and a
        # post-mortem read the same tensor dict the checkpoint writes; it is a
        # declared report value, not a hot-path export.
        "ema": ema,
        "eval_window_bytes": eval_window_bytes(cfg),
        "doc_counts": doc_counts,
    }
