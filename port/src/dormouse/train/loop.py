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
from pathlib import Path

import torch

from ..config import PROGRESS_KEYS, RunConfig, TrainConfig
from ..instrument import Counters, LoudError, StepTimer, loud
from .data import ByteStream, eval_window_bytes, open_streams, warmup_steps
from .lr import wsd_lr
from .optim import MixOptimizer


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


def save_ckpt(
    run: RunConfig, model: torch.nn.Module, opt: MixOptimizer, step: int
) -> tuple[Path, Path]:
    """Weights + optimizer state, plus the config snapshot beside them."""
    weights, snapshot = ckpt_paths(run.train)
    weights.parent.mkdir(parents=True, exist_ok=True)
    tmp = weights.with_suffix(".pt.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
        },
        tmp,
    )
    tmp.replace(weights)  # atomic: a crash never leaves a half-written ckpt
    snapshot.write_text(run.snapshot_toml())
    return weights, snapshot


def load_ckpt(run: RunConfig) -> tuple[dict, RunConfig]:
    """Load a checkpoint, refusing a config that differs from the snapshot.

    The exemption list is the progress keys, and nothing else: they decide WHEN
    we look (how many steps, how often to log/save/evaluate), never what the
    model computes. `lr` is NOT exempt — a different lr is a different run.
    """
    weights, snapshot = ckpt_paths(run.train)
    if not weights.is_file() or not snapshot.is_file():
        loud(
            f"resume: {weights} and {snapshot} must both exist (the snapshot is "
            f"what makes a resume checkable). Escape: start without --resume, or "
            f"fix ckpt_name/out_dir."
        )
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
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    return ck, stored


def evaluate(
    model: torch.nn.Module,
    stream: ByteStream,
    batches: int,
    device: torch.device | str | None = None,
) -> tuple[float, int]:
    """Mean held-out CE over `batches` windows + the BYTES those windows scored.

    `rewind()` first, always: an eval that inherits the stream position is not
    comparable with the previous one (AGENTS §2.6). No autodiff — burn's eval
    runs never-backwarded graphs to OOM on the card.
    """
    stream.rewind()
    total = 0.0
    scored = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, y = stream.next_batch()
            if device is not None:
                x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            assert loss is not None, "an eval batch must carry targets"
            total += float(loss)
            scored += int(x.numel())
    model.train(was_training)
    return total / max(batches, 1), scored


def train(
    model: torch.nn.Module,
    run: RunConfig,
    *,
    stream: ByteStream | None = None,
    eval_stream: ByteStream | None = None,
    resume: bool = False,
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
    device = None if device is None else torch.device(device)
    if stream is None:
        stream, eval_stream = open_streams(run)
    model.train()
    opt = MixOptimizer(model, lr=cfg.lr, wd=cfg.wd, opt=cfg.opt)
    tokens_per_step = cfg.batch * cfg.seq_len
    start = 0
    if resume:
        ck, _stored = load_ckpt(run)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"]) + 1
        if start:
            # A resumed run must continue on UNSEEN bytes: the burn failure
            # this prevents is re-reading (and memorizing) the corpus head.
            stream.skip_bytes(start * (cfg.seq_len + 1) * cfg.batch)
        log(f"resume: step={start} opt_steps={opt.steps}")

    warm = warmup_steps(cfg.warmup_tokens, cfg.batch, cfg.seq_len)
    decay = int(round(cfg.decay_frac * cfg.steps))
    total = Counters()
    window = Counters()
    nan_skipped = 0
    drowning = 0
    window_finite: list[bool] = []
    timer = StepTimer()
    last_loss = float("nan")
    for step in range(start, cfg.steps):
        lr = wsd_lr(
            step, total_steps=cfg.steps, peak_lr=cfg.lr, warmup_steps=warm, decay_steps=decay
        )
        for group in opt.adamw.param_groups:
            group["lr"] = lr
        if opt.muon is not None:
            for group in opt.muon.param_groups:
                group["lr"] = lr

        x, y = stream.next_batch()
        if device is not None:
            x, y = x.to(device), y.to(device)
        logits, loss, step_counters = model(x, y)
        assert logits.shape[-1] == run.model.vocab, "logits width != vocab"
        total.merge(step_counters)
        window.merge(step_counters)
        timer.tick()

        finite = loss is not None and bool(torch.isfinite(loss))
        window_finite.append(finite)
        if finite:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            last_loss = float(loss.detach())
        else:
            # Skip the step entirely (a masked loss scalar does not stop NaN
            # gradients from reaching the weights - the burn bug of
            # 2026-09-27), COUNTED, and never call any cleanup: the pool went
            # bad after NaN on the burn stack.
            nan_skipped += 1
            total.count("nan_skipped")
            window.count("nan_skipped")
            opt.zero_grad(set_to_none=True)

        if (step + 1) % cfg.log_every == 0:
            if window_finite and not any(window_finite):
                drowning += 1
            elif window_finite:
                drowning = 0
            window_finite = []
            parts = [
                f"step={step + 1}",
                f"loss={last_loss:.4f}",
                f"lr={lr:.3e}",
                kda_field(window),
                window.line(),
            ]
            if cfg.timers:
                parts.append(timer.report(cfg.log_every * tokens_per_step))
            log(" ".join(p for p in parts if p))
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
            ev, scored = evaluate(model, eval_stream, cfg.eval_batches, device=device)
            log(
                f"eval step={step + 1} loss={ev:.4f} bpb={bpb(ev):.3f} over {scored} B "
                f"({cfg.eval_batches} x {cfg.batch} x {cfg.seq_len})"
            )

        if (step + 1) % cfg.ckpt_every == 0:
            weights, snapshot = save_ckpt(run, model, opt, step)
            log(f"ckpt step={step + 1} -> {weights} ({snapshot})")

    if cfg.eval and eval_stream is not None and cfg.steps % cfg.eval_every != 0:
        ev, scored = evaluate(model, eval_stream, cfg.eval_batches, device=device)
        log(
            f"eval step={cfg.steps} loss={ev:.4f} bpb={bpb(ev):.3f} over {scored} B "
            f"({cfg.eval_batches} x {cfg.batch} x {cfg.seq_len})"
        )
    return {
        "steps": cfg.steps,
        "loss": last_loss,
        "nan_skipped": nan_skipped,
        "counters": total,
        "opt_steps": opt.steps,
        "eval_window_bytes": eval_window_bytes(cfg),
    }
