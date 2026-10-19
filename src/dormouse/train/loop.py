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

import math
import time
from pathlib import Path

import torch

from ..config import PROGRESS_KEYS, RunConfig, TrainConfig
from ..instrument import Counters, LoudError, StepTimer, loud
from ..model import kda
from ..model.engram import (
    enable_host_tables,
    host_engrams,
    host_state,
    host_step_all,
    load_host_state,
)
from ..model.experts import invalidate_banks_all
from ..model.tsct import set_mm_dtype
from .data import ByteStream, open_streams, warmup_steps
from .graph import OptimizerGraph
from .lr import wsd_lr
from .mixture import open_mixture
from .optim import MixOptimizer, clip_grads


def bpb(loss: float) -> float:
    """Bits per byte from a natural-log CE (burn's `dormouse_train::bpb`)."""
    return loss / math.log(2)


def kda_field(counters: Counters) -> str:
    """`kda=<arm>,n=<calls>` — which attention arm produced this window."""
    n = counters.get("kda_sdpa", 0) + counters.get("kda_fla", 0)
    if counters.get("kda_fla"):
        arm = "fla"
    elif counters.get("kda_sdpa"):
        arm = "sdpa"
    else:
        arm = "off"
    return f"kda={arm},n={n}"


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


def lfb_path(cfg: TrainConfig) -> Path:
    """The LFB controller's sibling of a checkpoint: `<ckpt_name>.lfb`.

    `lfb_bias` is model state — the next resume continues the integrator from it
    — but it is deliberately NOT a persistent `state_dict` buffer: the export
    artifact's key set is pinned by the demo engine (`docs-site/public/demo/
    engine.js` refuses an unknown key) and by `tests/test_export.py`, which have
    no LFB. The checkpoint still carries it, in this sibling, and
    `load_model_state` restores it for every reader (resume, eval, decode,
    serve) — the same sidecar contract as `<ckpt_name>.ngram` (audit 2026-10-19
    M8: the counts persisted while the bias they integrate did not).
    """
    weights, _ = ckpt_paths(cfg)
    return weights.with_suffix(".lfb")


def _lfb_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """`{buffer name: cpu copy}` for every LFB bias in `model`.

    The `_orig_mod.` prefix of a `torch.compile` wrapper is stripped, like the
    rest of the checkpoint (`save_ckpt`): every reader loads the unwrapped
    model.
    """
    return {
        name.removeprefix("_orig_mod."): buf.detach().to("cpu").clone()
        for name, buf in model.named_buffers()
        if name.rpartition(".")[2] == "lfb_bias"
    }


def _restore_lfb(
    model: torch.nn.Module, cfg: TrainConfig, *, log=print, resume: bool = False
) -> None:
    """Restore the LFB bias from `<ckpt_name>.lfb` when the checkpoint has one.

    A checkpoint written before the sidecar existed cannot supply it, and the
    mark depends on WHO is loading (review round 2 R2-2):

    * a TRAINING RESUME (`resume=True`) with the controller live
      (`moe_lb_coef > 0`) is LOUD: the next optimizer step would integrate from
      a zero bias under the checkpoint's own step count — a different network.
      The escape is the sidecar, or an explicit `moe_lb_coef=0` (the controller
      never MOVES the bias again);
    * every reader — eval/decode/serve/export/bench, an SFT/RL `--init-from`,
      and a resume with `moe_lb_coef == 0` — loads with the controller zeroed
      and a COUNTED `lfb: no sidecar, controller zeroed` line. There is no
      objective to continue, and a LOUD here would brick every pre-sidecar
      artifact for every reader (the failure this scoping fixes).

    `moe_lb_coef=0` is a break-glass, not a no-op: `SparseMoE.forward` adds
    `lfb_bias` to the logits UNCONDITIONALLY (`model/experts.py`), so a
    pre-sidecar checkpoint whose stored run had moved the bias resumes with a
    different routing than the one it was saved in. The COUNTED line says so.

    `lfb_counts` is reset here for every load. It is the sign rule's working
    set, not controller state: the bias is the integral a resume continues.
    `load_state_dict` restores the persistent counts too, and keeping them
    would make the first post-load update diff against the WHOLE run (mix1
    carries 11.8e9 accumulated counts at step 80k) instead of the steps since
    the load — the controller's time constant would grow with the run and
    freeze. Resetting is consistent with the restored bias: the bias survives,
    the counts restart at zero.

    A sidecar naming a buffer this model does not have, or a shape mismatch,
    is LOUD: silently dropping controller state is the class this sidecar fixes.
    """
    path = lfb_path(cfg)
    buffers = dict(model.named_buffers())
    if not any(name.rpartition(".")[2] == "lfb_bias" for name in buffers):
        return  # no LFB in this model: nothing to restore, nothing to report
    for name, buf in buffers.items():
        if name.rpartition(".")[2] == "lfb_counts":
            buf.zero_()
    if not path.is_file():
        coef = float(getattr(getattr(model, "cfg", None), "moe_lb_coef", 0.0))
        if coef > 0 and resume:
            loud(
                f"LFB sidecar {path} is missing but this model runs "
                f"moe_lb_coef={coef}: a training resume would continue the routing "
                f"controller from a zero bias under the checkpoint's step count — a "
                f"different network. "
                f"Escape: restore the `{path.name}` sibling, or resume with "
                f"moe_lb_coef=0 if the controller is meant to be off."
            )
        log(
            f"lfb: no sidecar, controller zeroed — routing differs from the stored "
            f"step ({path}, moe_lb_coef={coef})"
        )
        return

    state = torch.load(path, map_location="cpu", weights_only=True)
    missing = sorted(k for k in state if k not in buffers)
    if missing:
        loud(
            f"LFB sidecar {path} names buffer(s) {missing} this model does not "
            f"have. Escape: load the matching checkpoint, or delete the stale "
            f"`{path.name}` if it belongs to a different run."
        )
    with torch.no_grad():
        for name, value in state.items():
            target = buffers[name]
            if tuple(target.shape) != tuple(value.shape):
                loud(
                    f"LFB sidecar {path}: {name} has shape {tuple(value.shape)} "
                    f"but the model's buffer is {tuple(target.shape)}. Escape: "
                    f"the sidecar does not belong to this model."
                )
            target.copy_(value.to(target.device))


def host_device_type(model: torch.nn.Module) -> str:
    """The device the MODEL lives on — the one input to the host-path refusal.

    Its own function because it is the seam the CPU suite drives (`engram_ram`
    is a CUDA-budget rule; the host path's NUMERICS are device-agnostic and CPU-
    observable, since the tables ARE CPU memory). Nothing else reads it.
    """
    return next(model.parameters()).device.type


def host_tables(model: torch.nn.Module) -> int:
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
    """
    dev = host_device_type(model)
    if dev != "cuda":
        loud(
            f"engram_ram on a {dev} model: host tables buy VRAM by keeping the n-gram "
            f"rows and their momentum in host RAM, and a {dev} run has no VRAM to buy — "
            f"the per-forward row transfers would be pure cost against the in-VRAM "
            f"table. Escape: engram_ram=false, or run on cuda."
        )
    n = enable_host_tables(model)
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


def save_ckpt(
    run: RunConfig,
    model: torch.nn.Module,
    opt: MixOptimizer,
    step: int,
    *,
    counters: Counters | None = None,
) -> tuple[Path, Path]:
    """Weights + optimizer state, plus the config snapshot beside them.

    A storage rung (`model.quant` ∈ fp8/nvfp4) packs the factor matrices on
    the way out: masters stay fp32 in memory, the FILE carries the compressed
    record (`model_quant_rung` names it), and `load_ckpt` unpacks back to
    fp32. The bank/checkpoint is the rung's whole contract — the forward path
    within a run never changes numerics.

    The LFB routing bias is written to the `<ckpt_name>.lfb` sibling, not into
    the model half: it is model state (the next resume continues from it) but
    a non-persistent buffer (`lfb_path` says why). `load_model_state` restores
    it for every reader.

    `counters` receives the storage rung's COUNTED degradations: NVFP4 blocks
    whose stored e4m3 scale rounded to zero (`nvfp4_zero_scale_blocks`) decode
    to zeros, and a save that did not surface the count left the audit's
    "COUNTED, never SILENT" half of the prescale as a test-only tally (review
    round 1 M3). The trainer passes a fresh `Counters` per checkpoint and
    prints it on the checkpoint line.
    """
    from ..model.tsct import is_factor_key, pack_tsct

    weights, snapshot = ckpt_paths(run.train)
    weights.parent.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    # A torch.compile'd model is an OptimizedModule whose keys carry a
    # `_orig_mod.` prefix. Strip it on the way out (audit 2026-10-17,
    # CRITICAL 2): every reader — load_ckpt, eval, decode, serve — loads into
    # the UNWRAPPED model, so a prefixed checkpoint is unresumable and
    # un-evaluable. The prefix is a wrapper artifact, never part of the file
    # format.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    rung = run.model.quant if run.model.quant != "none" else ""
    if rung and rung in ("fp8", "nvfp4", "ternary"):
        sd = {
            k: (pack_tsct(v, rung, counters=counters) if is_factor_key(k) else v)
            for k, v in sd.items()
        }
    tmp = weights.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": sd,
            "opt": opt.state_dict(),
            "step": step,
            "model_quant_rung": rung or None,
        },
        tmp,
    )
    tmp.replace(weights)  # atomic: a crash never leaves a half-written ckpt
    lfb = _lfb_state(model)
    if lfb:
        # LFB's bias is model state but not a persistent state-dict buffer —
        # `lfb_path` says why (the export artifact's pinned key set). Written
        # for EVERY checkpoint, zeros included, so a reused `ckpt_name` can
        # never inherit a stale controller from an older run.
        lp = lfb_path(run.train)
        tmp_lfb = lp.with_suffix(".lfb.tmp")
        torch.save(lfb, tmp_lfb)
        tmp_lfb.replace(lp)
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
    """
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    return unpack_state_dict(ck["model"], ck.get("model_quant_rung"))


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


def load_model_state(
    model: torch.nn.Module,
    cfg: TrainConfig,
    sd: dict,
    *,
    strict: bool = True,
    log=print,
    resume: bool = False,
):
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

    The LFB bias is restored from the `<ckpt>.lfb` sibling after the load when
    that file exists (`_restore_lfb`): it is a non-persistent buffer (see
    `lfb_path`), so a plain `load_state_dict` would reset the routing
    controller a resume is supposed to continue. `resume=True` is the
    TRAINER's resume only: a missing sidecar with the controller live is LOUD
    there, and every reader (or a resume with `moe_lb_coef=0`) gets the COUNTED
    zeroed line instead of a refusal.
    """
    if cfg.engram_ram:
        require_ngram(cfg)
        enable_host_tables(model)  # idempotent: 0 when the model is already host
    result = model.load_state_dict(sd, strict=strict)
    if cfg.engram_ram:
        load_host_state(model, load_ngram(cfg))
    _restore_lfb(model, cfg, log=log, resume=resume)
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
    # The controller-off escape (`_restore_lfb`): a pre-sidecar checkpoint has
    # no bias to restore, and `moe_lb_coef=0` is the documented break-glass —
    # the controller never moves the bias again. It is NOT "the same function":
    # `SparseMoE.forward` adds a stored nonzero bias unconditionally, but a
    # missing sidecar loads a zero one, so routing differs from the stored step
    # (the COUNTED line says so). Only that one key, only to 0, and only when
    # the sidecar really is absent; every other model/train change is refused.
    if run.model.moe_lb_coef == 0 and not lfb_path(run.train).is_file():
        drift = [k for k in drift if k != "model.moe_lb_coef"]
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
    ck["model"] = unpack_state_dict(ck["model"], ck.get("model_quant_rung"))
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


def _eval_batches(stream: ByteStream, requested: int) -> int:
    """How many WHOLE eval windows the stream's byte range holds, capped at `requested`.

    One `next_batch` reads `batch × (seq_len + 1)` bytes (the next-byte shift)
    and scores `batch × seq_len` of them, so that read is the denominator a
    range is worth: `span // (batch × (seq_len + 1))`. The old denominator
    (`batch × seq_len`) let the last window overrun the range and wrap into its
    head — the 160 B / 640 B wrap review round 1 measured (M5). A range can
    still be large enough for `ByteStream` (which requires `batch × seq_len + 1`
    bytes) yet hold no whole window: then this returns 0 and the line says so.
    `open_streams` sizes the no-`--eval` tail in READS
    (`eval_batches × batch × (seq_len + 1)`, capped at 1 MiB), so the cap is
    the only reason a shortfall remains; the eval scores the windows that fit
    and the line names what is missing (COUNTED — audit 2026-10-19 M5, review
    round 2 R2-1).
    """
    per_window = stream.batch * (stream.seq_len + 1)
    assert per_window > 0, "an eval window has no bytes"
    return max(0, min(requested, (stream.hi - stream.lo) // per_window))


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

    The byte count is what the pass SCORED, not the window the config asked
    for: when the eval range is shorter than the requested window (the
    no-`--eval` tail is capped at 1 MiB), the loop runs only the windows the
    range holds, so the `over N B` field cannot claim more text than the tail
    contains (audit 2026-10-19 M5). `_eval_line` names the shortfall.

    `counters`, when given, receives THIS eval's own arm counters (the eval
    forward's `engram_rows`/`engram_arms` — the `engram=` field the eval line
    prints). They are the eval's, not the run's: a training counter would say
    the memory ran during an eval that ran without it, which is exactly the
    blind spot of AGENTS §3.2's retracted arm.

    With no WHOLE window in the range the pass scores 0 B and returns NaN loss:
    there is no measurement to report, and a fabricated 0.0 would read as a
    perfect score (`_eval_line` names the bytes left unscored).
    """
    stream.rewind()
    batches = _eval_batches(stream, batches)
    if batches == 0:
        return float("nan"), 0, float("nan"), []
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
                for j, v in enumerate(per):
                    per_iter_acc[j] += v
    n = batches
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
    *,
    batch: int | None = None,
    seq_len: int | None = None,
    parallel=None,
) -> tuple[str, float, int]:
    """The eval line, carrying the depth curve when the config asks for it.

    `cfg.rand_depth` turns the curve on (the arm must show its own curve);
    `cfg.eval_depths` requests it on a fixed-depth run. The line reports the
    held-out loss at `max_iter` as its headline number, so the arm's curve and
    the fixed-depth control are read off the same line.

    `batch`/`seq_len` are the shape the eval ACTUALLY scored (a distributed
    rank's micro-batch), and they are printed for the same reason the config's
    are: a BPB without its window is not a number. `parallel` adds the
    `dist=r/P` field — in a fleet only rank 0 evaluates, and the line must say
    so rather than look like a fleet-wide measurement.
    """
    b = cfg.batch if batch is None else batch
    t = cfg.seq_len if seq_len is None else seq_len
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
    #
    # The byte field is what the pass SCORED, and it names the shortfall when
    # the range could not hold the window the config asked for (the no-`--eval`
    # tail is capped at 1 MiB; audit 2026-10-19 M5). A line that printed
    # `eval_batches × batch × seq_len` here overstated the evidence by the
    # wrapped bytes.
    ran = _eval_batches(stream, cfg.eval_batches)
    # The WINDOW is the shape the rank actually scored (`b`/`t`), not the
    # config's: on a fleet a rank runs its own micro-batch, and a window that
    # quoted the config would describe another rank's measurement.
    window = cfg.eval_batches * b * t
    batches_field = f"{ran} x {b} x {t}"
    if ran != cfg.eval_batches:
        batches_field += f", requested {cfg.eval_batches}"
    line = (
        f"eval step={step} loss={ev:.4f} bpb={bpb(ev):.3f} fin={fin:.4f} "
        f"bpb_fin={bpb(fin):.3f} over {scored} B ({batches_field}) "
        f"engram={eng.get('engram_rows', 0)}/{eng.get('engram_arms', 0)}"
    )
    if scored < window:
        held = stream.hi - stream.lo
        line += (
            f" shortfall: requested {window} B over {cfg.eval_batches} batch(es), "
            f"the eval stream holds {held} B; scored {scored} B"
            + (f" over {ran} batch(es)" if ran != cfg.eval_batches else "")
            + f", left over {held - scored} B"
        )
        if ran == 0:
            line += (
                f" (no whole window fits: one read is {b} x "
                f"({t} + 1) = {b * (t + 1)} B)"
            )
    if parallel is not None:
        line += f" dist={parallel.rank}/{parallel.world_size}"

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
    log=print,
    device: torch.device | str | None = None,
    parallel=None,
) -> dict:
    """Run the training loop. Returns a report dict (final loss, counters, step).

    `device` is the batch seam: the stream hands out CPU tensors (its bytes come
    from an mmap, and it is the caller who knows where the model lives), so a
    GPU run names the device here and gets its windows transferred. `None` is
    the CPU path, unchanged.

    `parallel` is the distributed seam (`train/distributed.py`). `None` is the
    single-process path and every line below is byte-identical to its history:
    the rank's shape is the config's, the loss is untouched, the NaN decision
    is local. With a `Parallel`, four things change and nothing else — the
    stream is the rank's own byte range and micro-batch, the loss is scaled by
    `world_size · t_rank / Σt` before backward (so DDP's mean-of-means lands on
    the global token mean), the NaN decision is one collective (a rank that
    skips backward would hang the reducer), and only rank 0 evaluates,
    checkpoints and logs.
    """
    cfg = run.train
    #: The model shape's effective batch. `parallel=None` keeps the config's
    #: exactly; a rank runs its own capacity-profile entry (stage 3).
    batch, seq_len = (
        (cfg.batch, cfg.seq_len)
        if parallel is None
        else (
            parallel.shape.batch,
            parallel.shape.seq_len,
        )
    )
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
            # per-stage weights, epoch-capped LOUD. The eval side is unchanged
            # (the same held-out file and window), so BPB stays comparable.
            stream = open_mixture(
                run.train.mix_plan,
                steps=run.train.steps,
                batch=batch,
                seq_len=seq_len,
                seed=run.train.seed,
                max_epochs=run.train.max_epochs,
            )
            _, eval_stream = open_streams(run, batch=batch, seq_len=seq_len)
            log(
                f"mix_plan={run.train.mix_plan}: {len(stream.plan.sources)} sources, "
                f"{len(stream.plan.stages)} stage(s), max_epochs={run.train.max_epochs}"
            )
        else:
            stream, eval_stream = open_streams(run, batch=batch, seq_len=seq_len)
        if parallel is not None:
            # The rank's own byte range: disjoint per rank, and with equal shapes
            # the fleet's union at step s is the contiguous block one process at
            # Σbatch reads at step s — the equal-batch gate's whole basis.
            stream.skip_bytes(parallel.byte_offset())
    model.train()
    if parallel is not None:
        # Refusals BEFORE any state is built, then FSDP2's shard (it must come
        # before the optimizer so the optimizer state is sharded with it).
        parallel.refuse_incompatible(run, resume=resume)
        model = parallel.shard(model)
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
        n = host_tables(model)
        log(f"engram_ram: {n} table set(s) in host RAM, VRAM holds the batch's unique rows")
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
    )
    if cfg.optim_8bit:
        opt.enable_bnb8bit()

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
            "`optim.Muon.sf_switch` documents). A NaN-skipped step advances none of "
            "Z/X/s."
        )
    #: The GLOBAL tokens one step consumes. A fleet property: the loss scale,
    #: the LR warmup and the logged tok/s are all defined on it, so every rank
    #: must agree — a per-rank warmup count would put the ranks on different
    #: schedules and silently train N different runs.
    tokens_per_step = cfg.batch * cfg.seq_len if parallel is None else parallel.global_tokens
    start = 0
    if resume:
        ck, stored = load_ckpt(run)
        load_model_state(model, cfg, ck["model"], log=log, resume=True)
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        if start:
            # A resumed run must continue on UNSEEN bytes: the burn failure
            # this prevents is re-reading (and memorizing) the corpus head.
            # A mixture's position is a STEP-INDEX question, not a byte-count
            # one: its stage schedule is a function of the run's budget, and
            # `steps` is an exempt progress key, so an extended run must
            # replay the prefix under the checkpoint's stored budget or it
            # assigns already-taken steps to different sources and moves every
            # cursor by the wrong amount (review round 3 R3-4). A fleet's rank
            # adds its own byte offset on the byte path (stage 3).
            skip_to_step = getattr(stream, "skip_to_step", None)
            if skip_to_step is not None:
                skip_to_step(start, budget=stored.train.steps)
            else:
                rank_off = 0 if parallel is None else parallel.byte_offset()
                stream.skip_bytes(start * (cfg.seq_len + 1) * cfg.batch + rank_off)
        # The checkpoint's model half is the X iterate (a checkpoint is written
        # with the eval switch on, below), and so is the optimizer's own `x`, so
        # the params must go back to Y before the next forward. The switch is an
        # ABSOLUTE write from the optimizer's state, which is why this is the
        # same call as the post-construction one and cannot double-lerp.
        sf("train")
        log(f"resume: step={start} opt_steps={opt.steps}")

    if parallel is not None:
        # DDP AFTER the optimizer and after the resume load: the wrapper
        # broadcasts rank 0's parameters, and the optimizer already holds those
        # parameter objects — wrapping earlier would load a `module.`-prefixed
        # state dict into the raw module's keys.
        model = parallel.replicate(model)
        log(parallel.line())

    warm = (
        warmup_steps(cfg.warmup_tokens, cfg.batch, cfg.seq_len)
        if parallel is None
        else warmup_steps(cfg.warmup_tokens, 1, tokens_per_step)
    )
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
    window_finite: list[bool] = []
    timer = StepTimer()
    last_loss = float("nan")
    #: The telemetry sink: every event (log cadence, evals, checkpoints) gets
    #: a JSON row. A dict writer that never raises on non-JSON values.
    sink = None
    if cfg.json_log and (parallel is None or parallel.is_main):
        # ONE writer: N ranks appending to one JSONL would interleave partial
        # lines. Rank 0's file is the run's telemetry, like its log.
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
        """
        if cfg.grad_clip > 0:
            clip_grads(model.parameters(), cfg.grad_clip)
        opt.step()

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
        graph = OptimizerGraph(opt, opt_stage, counters=total, log=log)
        log(
            f"opt_graph: the optimizer stage is captured after {graph.warmup_steps} "
            f"eager steps and replayed with ONE launch per step; the LR lives in a "
            f"device tensor this loop fills IN PLACE (a host float would be baked at "
            f"capture and freeze the WSD schedule), and zero_grad runs "
            f"set_to_none=False so the captured ops keep their pinned .grad buffers."
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
            # capture (`train/graph.py`), so the Muon group's LR is this value.
            graph.set_lr(lr)
        else:
            for group in opt.adamw.param_groups:
                group["lr"] = lr
            if opt.muon is not None:
                for group in opt.muon.param_groups:
                    group["lr"] = muon_lr

        # The ANNEALED factor_qat arm: burn-spectral's measured soft-to-hard
        # ramp (`ste_ternary_annealed`, "converges better than hard STE from
        # step one"). alpha = min(1, step/anneal) is a PURE function of the
        # step, so a resume lands on the same alpha. `anneal = 0` (default)
        # leaves every module's constant alpha untouched.
        if anneal_steps > 0:
            model.set_factor_qat_alpha(min(1.0, step / anneal_steps))

        x, y = stream.next_batch()
        if device is not None:
            x, y = x.to(device), y.to(device)
        # The ADAPTIVE DEPTH arm: T is the pure `(seed, step)` sample, and the
        # loss divides by the depth actually run (`model.forward`/`loopblock`).
        # `None` (the default and every other config) keeps the call unchanged.
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
        logits, loss, step_counters = model(x, y, **kw)
        assert logits.shape[-1] == run.model.vocab, "logits width != vocab"
        total.merge(step_counters)
        window.merge(step_counters)
        timer.tick()

        # ONE host read per step, not two (audit MAJOR 4): `float(loss)` and the
        # finiteness test share a single D2H transfer. The full device-side
        # firewall (mask on the device, count at log cadence) needs an optimizer
        # that accepts a device `scale` — the follow-up the audit names — and is
        # not smuggled in here: the skip semantics below are the burn-proven
        # ones and must not drift silently.
        lf = float(loss.detach()) if loss is not None else float("nan")
        finite = math.isfinite(lf)
        if parallel is not None:
            # The NaN decision is the FLEET's, not this rank's: a rank that
            # skipped backward would never enter the reducer and the peers
            # would block on a collective that never comes. MIN over the flag;
            # a rank whose own loss was finite but whose peer's was not has its
            # lost step COUNTED (`nan_forced_skip`), because that is a real
            # cost of a peer's divergence and a reader should see it.
            agreed = parallel.all_finite(finite)
            if finite and not agreed:
                total.count("nan_forced_skip")
                window.count("nan_forced_skip")
            finite = agreed
        window_finite.append(finite)
        if finite:
            # The graph arm needs STABLE `.grad` buffers: `set_to_none=True` swaps
            # each `.grad` for a fresh tensor every step, which breaks the buffers
            # capture pinned (`dion/cuda_graph.py:170-181`). `False` is a no-op for
            # a param whose grad is still None, so the FIRST backward allocates it
            # — which is why the wrapper demands at least one eager step.
            opt.zero_grad(set_to_none=graph is None)
            if parallel is None:
                loss.backward()
            else:
                # DDP averages the ranks' gradients; the objective is the mean
                # over the GLOBAL batch's tokens. Scaling this rank's loss by
                # `world_size · t_rank / Σt` makes the reducer's division land
                # on `Σ (t_r/Σt)·g_r` exactly (stage 3's normalization).
                (loss * parallel.loss_scale).backward()
            if graph is not None:
                # ONE launch: the captured stage IS `opt_stage()` (the grad clip
                # plus the mixed step), and the wrapper redoes the host-side step
                # counters that a recording cannot carry.
                graph.step()
            else:
                opt_stage()
            # The parameters just moved — and a CUDA-graph replay moves them
            # WITHOUT bumping `_version` (measured 2026-10-19). `_banks` keys its
            # cache on `_version`, so with `opt_graph = true` the key froze at
            # capture and the bank served the step-5 build for a whole run
            # (experts frozen in the forward, drifting in memory, checkpoint
            # unreproducible — the medbig2/b1night defect). One call per
            # TRAINING step keeps the cache's real job (one build per step,
            # reused by the loop's iterations) and nothing more.
            invalidate_banks_all(model)
            if opt.counters:
                # The Muon arms' own counters (ADR-0019): `ns_triton` when the
                # fused Newton–Schulz chain ran, `ns_triton_fallback` + the reason
                # when it could not. Merged and CLEARED here, once per step that
                # runs the optimizer, so the log line's counter field is where a
                # reader learns which chain ran — and so a NaN-skipped step (which
                # never reaches this branch) cannot leave a stale count behind.
                total.merge(opt.counters)
                window.merge(opt.counters)
                opt.counters.clear()
            if cfg.engram_ram:
                # The host update runs ONCE per TRAINING step, after the
                # optimizer — never per forward (the loop calls the memory once
                # per iteration, and `host_step` sums those), and never on a
                # NaN-skipped step (that branch does not call `opt.step`
                # either). `rows` counts the rows this step touched; it is a
                # CPU count already, so the 0-sync rule holds.
                updates, rows = host_step_all(model, lr)
                total.count("engram_host_updates", updates)
                window.count("engram_host_updates", updates)
                total.count("engram_host_update_rows", rows)
                window.count("engram_host_update_rows", rows)
            last_loss = lf
        else:
            # Skip the step entirely (a masked loss scalar does not stop NaN
            # gradients from reaching the weights - the burn bug of
            # 2026-09-27), COUNTED, and never call any cleanup: the pool went
            # bad after NaN on the burn stack.
            nan_skipped += 1
            total.count("nan_skipped")
            window.count("nan_skipped")
            # A skipped step calls NEITHER the stage nor `graph.step()`: the graph
            # replay advances only on a real optimizer step, exactly as the
            # optimizer's own step count does (`train/graph.py`).
            opt.zero_grad(set_to_none=graph is None)
            # A skipped step never reaches `host_step_all`, so the host-path
            # leaves it built would sit in `Engram._pending` until the next REAL
            # step (and their rows be re-deduped there) — growth proportional to
            # the NaN count, and the burn record has NaN FLOODS (AGENTS §2.3).
            # Dropped here, once, in the one place that knows the step did not
            # train. The in-VRAM path never appends, so this is a no-op there.
            for m in host_engrams(model):
                m.drop_pending()

        if (step + 1) % cfg.log_every == 0:
            # The 0-sync rule: every DEVICE-side counter this window gathered
            # is read NOW, in one batched GPU→CPU transfer, at the declared
            # cadence — never per pass.
            window.drain()
            total.drain()
            if window_finite and not any(window_finite):
                drowning += 1
            elif window_finite:
                drowning = 0
            window_finite = []
            if parallel is not None:
                # The line's loss is the FLEET's (token-weighted mean of the
                # ranks' local means), reduced at the log cadence on every rank
                # — a fleet line that printed rank 0's micro-batch loss would be
                # the eval-window defect in another costume.
                last_loss = parallel.fleet_loss(last_loss)
            parts = [
                f"step={step + 1}",
                f"loss={last_loss:.4f}",
                f"lr={lr:.3e}",
                kda_field(window),
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
            if parallel is None or parallel.is_main:
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
            # A fleet evaluates on rank 0 only (`dist=r/P` on the line): its
            # window is the rank's micro-batch, printed, and the other ranks
            # wait at the next step's collective.
            if parallel is None or parallel.is_main:
                sf("eval")
                line, ev, scored = _eval_line(
                    model,
                    eval_stream,
                    cfg,
                    run.model.max_iter,
                    device,
                    step + 1,
                    batch=batch,
                    seq_len=seq_len,
                    parallel=parallel,
                )
                sf("train")
                log(line)
                if sink is not None:
                    _row(event="eval", step=step + 1, loss=ev, bpb=bpb(ev), scored=scored)

        if (step + 1) % cfg.ckpt_every == 0 and (parallel is None or parallel.is_main):
            # A checkpoint is the model a reader would serve, so it is written
            # at X too — and on a resume the model half plus the optimizer's own
            # `x` are the same iterate, which is what makes the switch back to Y
            # an absolute, double-lerp-free write. ONE writer (rank 0) and the
            # UNWRAPPED module: DDP prefixes every state-dict key with `module.`.
            sf("eval")
            # A fresh tally per save: the checkpoint line names a file's own
            # zeroed blocks, not a run-cumulative number a reader cannot tie to
            # the file (`save_ckpt`'s `counters`).
            ck_counts = Counters()
            weights, snapshot = save_ckpt(
                run,
                parallel.raw(model) if parallel is not None else model,
                opt,
                step,
                counters=ck_counts,
            )
            sf("train")
            ck_line = f"ckpt step={step + 1} -> {weights} ({snapshot})"
            if ck_counts:
                total.merge(ck_counts)
                ck_line += " " + ck_counts.line()
            log(ck_line)

    if (
        cfg.eval
        and eval_stream is not None
        and cfg.steps % cfg.eval_every != 0
        and (parallel is None or parallel.is_main)
    ):
        sf("eval")
        line, _ev, _scored = _eval_line(
            model,
            eval_stream,
            cfg,
            run.model.max_iter,
            device,
            cfg.steps,
            batch=batch,
            seq_len=seq_len,
            parallel=parallel,
        )
        sf("train")
        log(line)
    return {
        "steps": cfg.steps,
        "loss": last_loss,
        "nan_skipped": nan_skipped,
        "counters": total,
        "opt_steps": opt.steps,
        "eval_window_bytes": cfg.eval_batches * batch * seq_len,
        "rank": parallel.rank if parallel is not None else 0,
        "world_size": parallel.world_size if parallel is not None else 1,
    }
