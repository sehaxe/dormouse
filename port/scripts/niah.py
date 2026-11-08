#!/usr/bin/env -S uv run --script
"""S-NIAH-8: exact 8-byte retrieval through the state-carrying path (gates A-D).

WHAT THIS IS. `docs/research/2026-11-08-long-context-eval.md` §4 pre-registers the
cheapest honest 1M protocol. This script is its RUNNER for gate D (retrieval),
its controls, and the gate C rank probe; gates A (artifact) and B (state carry)
are preflights it refuses on, because a number from a failed precondition is a
measurement of a different function:

* **Gate A** — `use_csa=true`, `max_seq_len >= context`, and (with weights) a
  live `csa_gate` in >=1 iteration slot. A 1K dense checkpoint cannot run this
  test; saying so IS the gate.
* **Gate B** — 2 segments must be the 1-pass forward on the same bytes. The
  repo's segmented forward (`LoopBlock._forward_segmented`) carries
  `(kda_state, csa_state)` between its cuts; if the last 8 prediction logits move
  by more than `1e-5` (fp32) / `1e-3` (bf16), the 1M number is a different
  function and the protocol stops.
* **Gate C** — rank of the needle's fine chunk in each layer's score row. When
  the best-layer rank is chance (~`context/csa_chunk`), the selector is the
  bottleneck; the protocol says report that and do NOT spend the 100-item run.
* **Gate D** — exact match of all 8 bytes at `--items` disjoint haystacks, plus
  the four controls. Scoring is teacher-forced argmax at the 8 target positions
  (the report's §4.1 rule): no sampling, no LLM judge, deterministic.

THE TASK. Each item is `--context` bytes of model input: a synthetic haystack
with ONE `The pass key is <KEY>.` needle at a distance, and a final
`The pass key is ` question. The key is aligned to a `csa_chunk` boundary so it
is exactly one fine chunk, as §4.1 requires. `--dry-run` builds the items, prints
the arithmetic (§4.5's formula) and the exact command it would run, and needs no
checkpoint: tonight nothing can run gate D (no trained CSA checkpoint exists),
and the dry run is the proof that the runner is ready for one.

`ponytail:` the state-carrying prefill has no public entry that RETURNS its
state. `decode._chunked_prefill` reaches `LoopBlock._window` directly and names
the missing `LoopBlock.prefill_state` hook; this runner uses the public
`model(ids, segment_len=...)` segmented forward, which carries the pair
INTERNALLY (enough for gates B-D, whose targets are logits) but exposes no state
for a decode loop that spits its own bytes. Add `prefill_state` when a caller
needs the state itself; do not grow a second forward here.

    PYTHONPATH=port/src port/.venv/bin/python port/scripts/niah.py --dry-run
    PYTHONPATH=port/src port/.venv/bin/python port/scripts/niah.py \\
        --ckpt-name <csa_run> --context 1048576 --controls
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dormouse.cli import _device, _model, _snapshot_run  # noqa: E402
from dormouse.config import ModelConfig, validate  # noqa: E402
from dormouse.instrument import loud  # noqa: E402
from dormouse.model.csa import (  # noqa: E402
    CompressedSelectiveAttention,
    total_state_bytes,
)
from dormouse.train.loop import load_model_half, load_model_state  # noqa: E402

# `scripts/` is not a package (the test loads this file by path): the lane guard
# is a sibling file, so its directory goes on the path APPENDED, never inserted
# (a script named like a package would shadow it).
sys.path.append(str(Path(__file__).resolve().parent))
from _lane import preflight, skip  # noqa: E402

CUE = b"The pass key is "
SEP = b". "
KEY_LEN = 8
#: The bytes a max-distance item spends besides the haystack: two cues and the key
#: twice (the prompt's needle and the scored answer), so `distance <= context - this`.
DISTANCE_SLACK = 2 * (KEY_LEN + len(CUE))
#: A 64-symbol printable alphabet, as §4.1 says. 256 is a multiple of 64, so
#: `bytes.translate` gives a uniform draw without a Python-level loop.
ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."
TRANSLATE = bytes(ALPHABET[b % len(ALPHABET)] for b in range(256))
#: The report's §4.2 gate-A stop rule; the user-facing half of every refusal.
ESCAPE = "train a CSA model at >=8K; see .bulba/plan.md"
#: The report's §4.3 bars, as fractions of the item count.
BAR_D = 0.95
BAR_NO_NEEDLE = 0.03
BAR_CSA_OFF = 0.10
BAR_1K = 0.95
#: Control name -> the argparse dest `--control-*` writes.
CONTROLS = {
    "no-needle": "control_no_needle",
    "shuffled-key": "control_shuffled",
    "csa-off": "control_csa_off",
    "1k": "control_1k",
}


@dataclass(frozen=True)
class Item:
    """One S-NIAH-8 item: `scored = prompt + key` is the model's input window."""

    seed: int
    prompt: bytes
    scored: bytes
    key: bytes
    key_at: int
    distance: int
    needle: bool

    def target_rows(self) -> slice:
        """The 8 logit rows that predict the 8 answer bytes (`row p` -> byte p+1)."""
        return slice(-9, -1)


def _haystack(seed: int, n: int) -> bytes:
    """`n` printable bytes, a pure function of `(seed, n)` — apart from the key RNG."""
    return random.Random(seed ^ 0x5EED).randbytes(n).translate(TRANSLATE)


def _drawn_key(seed: int) -> bytes:
    """The key stream is separate from the haystack stream, so a control can keep
    the haystack fixed while it changes (or removes) the key."""
    return random.Random(seed ^ 0xA5A5A5A5).randbytes(KEY_LEN).translate(TRANSLATE)


def build_item(
    seed: int,
    context: int,
    *,
    distance: int | None = None,
    key: bytes | None = None,
    needle: bool = True,
    align: int = 8,
) -> Item:
    """One `--context`-byte item; `key` lives exactly one aligned fine chunk.

    The prompt is `hay(p) + CUE + key + ". " + hay(f) + CUE`, so `distance` is
    the bytes between the key's end and the question's start: `f + 2`. When
    `distance is None` the needle goes as early as the chunk alignment allows
    (the protocol's furthest point). An explicit distance is honoured to the
    alignment, and the ACTUAL value is what `Item.distance` and the run print —
    a rounded distance that the output does not name is a different experiment.

    `needle=False` (the no-needle control) keeps the SAME drawn key as a needle
    item with this seed but leaves it out of the text, so a model that emits it
    anyway is counted as hallucinating the removed key, not guessed correctly.
    """
    if context < 96:
        loud(f"context={context} is too short for the {len(CUE)}-byte cue pair and a key")
    if align not in (4, 8, 16, 32):
        loud(f"align={align}: a CSA chunk is one of 4/8/16/32 (csa.py's CHUNKS)")
    prompt_len = context - KEY_LEN
    k = _drawn_key(seed) if key is None else key
    if len(k) != KEY_LEN:
        loud(f"key has {len(k)} bytes, S-NIAH-8 needs exactly {KEY_LEN}")
    if not needle:
        prompt = _haystack(seed, prompt_len - len(CUE)) + CUE
        return Item(seed, prompt, prompt + k, k, -1, -1, False)
    d_max = context - DISTANCE_SLACK
    if distance is None:
        # Smallest prefix that lands the key on an `align` boundary.
        p = (-len(CUE)) % align
        p = min(p, d_max - 2)
    else:
        if not 2 <= distance <= d_max:
            loud(
                f"distance={distance}: the prompt has room for 2..{d_max} bytes "
                f"between the key and the question at context={context}. Escape: "
                f"a distance in that range, or --distance omitted for the maximum."
            )
        p_target = d_max - distance
        p = p_target - ((p_target + len(CUE)) % align)
        if p < 0:
            loud(
                f"distance={distance} is too close to the maximum {d_max} for a "
                f"{align}-byte key alignment. Escape: ask for <= {d_max - align} "
                f"bytes, or omit --distance."
            )
    actual = d_max - p
    filler = actual - len(SEP)
    hay = _haystack(seed, p + filler)
    prompt = hay[:p] + CUE + k + SEP + hay[p:] + CUE
    key_at = p + len(CUE)
    assert len(prompt) == prompt_len, (len(prompt), prompt_len)
    assert prompt[key_at : key_at + KEY_LEN] == k, "the needle is not where it is placed"
    assert key_at % align == 0, f"key_at={key_at} is not {align}-aligned"
    assert prompt[key_at + KEY_LEN + actual :] == CUE, "the question does not end the prompt"
    return Item(seed, prompt, prompt + k, k, key_at, actual, True)


def effective_segment(context: int, segment: int) -> int:
    """`min(segment, context)`, refusing a cut the repo's segmented forward cannot make.

    `LoopBlock._forward_segmented` asserts `t % segment_len == 0` (its loss
    divisor and its state carry are written for equal segments). A silent
    round-down would measure a different number of bytes per item than the one
    printed, so this is LOUD.
    """
    if segment <= 0 or context <= 0:
        loud(f"segment={segment} and context={context}: both must be positive")
    seg = min(segment, context)
    if context % seg:
        loud(
            f"segment={segment} (effective {seg}) does not divide context={context}: "
            f"the repo's segmented forward requires t % segment_len == 0. Escape: a "
            f"power-of-two --segment <= --context."
        )
    return seg


# --- gates A and the segmented preflight -----------------------------------


def artifact_refusal(cfg: ModelConfig, context: int) -> str | None:
    """Gate A's config half: why a 1M retrieval number cannot be measured, or None.

    `use_csa=false` means the checkpoint is not the network this task is about —
    the report's own audit found the on-disk 95M is a 1K dense transformer, and
    bolting anything onto it would measure that. `max_seq_len < context` means
    the snapshot's own declared window cannot hold the pass.
    """
    if not cfg.use_csa:
        return (
            f"gate A: use_csa=false in the snapshot — a {context}-byte S-NIAH-8 number "
            f"would measure a different network (the 1K dense control, or a CSA branch "
            f"nothing trained). Escape: {ESCAPE}."
        )
    if cfg.max_seq_len < context:
        return (
            f"gate A: max_seq_len={cfg.max_seq_len} < context={context} bytes — the "
            f"snapshot declares a {cfg.max_seq_len}-byte window, so a {context}-byte pass "
            f"is outside its own config. Escape: {ESCAPE}, with max_seq_len >= the "
            f"context you want to measure; or a context <= {cfg.max_seq_len} (the 1K "
            f"positive control)."
        )
    return None


def segmented_refusal(cfg: ModelConfig) -> str | None:
    """Why the SEGMENTED path cannot be the model's own forward on this config, or None.

    These are the repo's own refusals (`_forward_segmented`, `config.validate`),
    lifted to the preflight so the runner dies with the escape instead of an
    AssertionError three layers down. A cut that changes the forward would make
    gate B compare two different functions and "pass" for the wrong reason.
    """
    if cfg.csa_hierarchical:
        return (
            "gate B/D: csa_hierarchical=true — the segmented carry refuses this pair "
            "(`loopblock._forward_segmented`: forward_state and forward already differ "
            "1.7e-2 on this arm), so the 1M number would be a different selection. "
            "Escape: csa_hierarchical=false (the flat fine scan), or segment_len=0 and "
            "no state carry."
        )
    if not cfg.csa_fine:
        return (
            "gate C: csa_fine=false — the fine 8-byte level does not run, so there is no "
            "needle chunk to rank and no exact-8-byte readout. Escape: csa_fine=true; the "
            "coarse-only arm is a separate control with its own bar."
        )
    if cfg.moe_topk > 0 and cfg.moe_capacity > 0:
        return (
            "gate B: moe_topk>0 at moe_capacity>0 — the drop set is a function of the "
            "window's composition (measured 0.39 on the readout), so a segment cut "
            "changes which tokens reach an expert and the segmented pass is not the "
            "one-shot pass. Escape: moe_capacity=0 (dropless), moe_topk=0, or "
            "segment_len=0."
        )
    return None


def gate_a_live(model: torch.nn.Module) -> str | None:
    """Gate A's weights half: a trained CSA gate must exist and be live in >=1 slot.

    `sigmoid(csa_gate)` is a multiplier, so an untrained branch still reads 0.5
    at a zero gate; the report's check is the one this prints — a gate that is
    EXACTLY zero everywhere means the branch contributes nothing, and the
    retrieval number is KDA's alone.
    """
    gates = [
        (name, float(torch.sigmoid(p.detach().float()).abs().max()))
        for name, p in model.named_parameters()
        if name.endswith("csa_gate")
    ]
    if not gates:
        return (
            "gate A: the loaded model has no `csa_gate` parameter, so no CSA branch is "
            f"live under these weights. Escape: {ESCAPE}."
        )
    live = [g for g in gates if g[1] > 0.01]
    if not live:
        return (
            f"gate A: |sigmoid(csa_gate)| <= 0.01 in all {len(gates)} slots — the CSA "
            f"branch is untrained and cannot be the mechanism under test. Escape: {ESCAPE}."
        )
    print(
        f"gate A: PASS · use_csa=true · max_seq_len={model.cfg.max_seq_len} · "
        f"csa_gate live in {len(live)}/{len(gates)} slots "
        f"(max |sigmoid|={max(g[1] for g in gates):.3f})",
        flush=True,
    )
    return None


# --- gate B: the carry is the 1-pass forward -------------------------------


def _ids(raw: bytes, dev: torch.device) -> torch.Tensor:
    """Bytes -> `[1, t]` long ids without a 1M-element Python list."""
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long).view(1, -1).to(dev)


@torch.no_grad()
def carry_gate(model: torch.nn.Module, item: Item, segment: int) -> tuple[float, int]:
    """`(max|dlogits|, segments)` for 1 pass vs 2 segments of the same 2·segment bytes.

    The comparison is the last 8 PREDICTION rows (not the last 8 input rows): a
    mismatch there is exactly a different answer to the question, which is what
    the claim reads. Logits are fp32 by the model's head rule; the tolerance is
    the caller's (`1e-5` fp32, `1e-3` bf16 — the report's §4.2).
    """
    probe = item.scored[: 2 * segment]
    assert len(probe) == 2 * segment, (len(probe), segment)
    dev = next(model.parameters()).device
    ids = _ids(probe, dev)
    ref, _, _ = model(ids)
    got, _, counters = model(ids, segment_len=segment)
    segments = int(counters.get("loop_segments", 0))
    if segments != 2:
        loud(
            f"gate B: the segmented pass reported loop_segments={segments}, expected 2 — "
            f"the carry path did not run as the gate assumes. Escape: inspect "
            f"loopblock._forward_segmented; do not quote a carry number from this run."
        )
    delta = float((ref - got).abs().max())
    return delta, segments


# --- gate C: rank of the needle chunk in the selector's score row ----------


def _capture_select(module: CompressedSelectiveAttention, into: dict[str, tuple], name: str) -> str:
    """Wrap ONE CSA module's selector so the last query's score row is kept.

    `_select` is the fine level's own selector (`_attend` calls it after masking);
    `csa_fused` swaps the same call for `_select_fused`, so the probe follows the
    config instead of silently capturing nothing. The capture is `[N]` per call,
    overwritten per segment, so at 1M the memory is 12 rows, not a 2 GiB score
    matrix. This observes the model's arithmetic; it does not re-implement
    `_level_scores`.
    """
    attr = "_select_fused" if module.fused else "_select"
    orig = getattr(module, attr)

    def proxy(scores, v, allowed, k_eff, width):
        with torch.no_grad():
            into[name] = (
                scores[0, -1].detach().float().cpu(),
                allowed[0, -1].detach().bool().cpu(),
            )
        return orig(scores, v, allowed, k_eff, width)

    setattr(module, attr, proxy)  # type: ignore[method-assign]
    return attr


@torch.no_grad()
def rank_items(
    model: torch.nn.Module, items: list[Item], cfg: ModelConfig, segment: int
) -> list[int | None]:
    """Best-layer flat rank of the needle chunk per item (`None` = not selectable).

    The report's gate C: best-layer rank <= `csa_topk` in >=90% of items, median
    <= topk. A rank near `context/csa_chunk` is chance, and that is the selector
    being the bottleneck rather than the readout.
    """
    modules = [m for m in model.modules() if isinstance(m, CompressedSelectiveAttention)]
    if not modules:
        loud("gate C: the model has no CSA module to probe")
    captured: dict[str, tuple] = {}
    patched: list[tuple[CompressedSelectiveAttention, str]] = []
    for i, module in enumerate(modules):
        patched.append((module, _capture_select(module, captured, f"csa[{i}]")))
    ranks: list[int | None] = []
    try:
        for item in items:
            if not item.needle:
                loud("gate C: a no-needle item has no chunk to rank")
            captured.clear()
            score_item(model, item, segment)
            key_chunk = item.key_at // cfg.csa_chunk
            best: int | None = None
            for scores, allowed in captured.values():
                if key_chunk >= scores.shape[-1] or not bool(allowed[key_chunk]):
                    continue
                s = scores[key_chunk]
                rank = 1 + int(((scores > s) & allowed).sum())
                best = rank if best is None else min(best, rank)
            ranks.append(best)
    finally:
        for module, attr in patched:
            module.__dict__.pop(attr, None)
    return ranks


# --- scoring (gate D) ------------------------------------------------------


@torch.no_grad()
def score_item(model: torch.nn.Module, item: Item, segment: int) -> tuple[int, bytes, int]:
    """`(exact bytes of 8, predicted bytes, segments run)` — teacher-forced argmax.

    The input is `prompt + key`, so row `p` predicts byte `p+1` and the 8 rows
    `[-9:-1]` are the 8 greedy decisions for the answer. Feeding the true prefix
    is the report's §4.1 rule: greedy, deterministic, and it cannot turn a
    retrieval failure into a sampling accident.
    """
    if len(item.scored) % segment:
        loud(f"item {len(item.scored)} B is not a multiple of segment={segment}")
    dev = next(model.parameters()).device
    ids = _ids(item.scored, dev)
    logits, _, counters = model(ids, segment_len=segment)
    rows = logits.reshape(ids.shape[1], -1)[item.target_rows()]
    pred = rows.argmax(-1).detach().cpu().tolist()
    hits = sum(int(p == k) for p, k in zip(pred, item.key, strict=True))
    segments = int(counters.get("loop_segments", 0))
    want = ids.shape[1] // segment
    if segments != want:
        loud(
            f"score: loop_segments={segments}, expected {want} — the item was not run "
            f"through the state-carrying segmented forward. Escape: inspect the run."
        )
    return hits, bytes(pred), segments


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """The 95% lower bound a claim may quote (the report's §4.4 table)."""
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def report(label: str, hits: list[int], bar: float, *, below: bool) -> bool:
    """One summary line; `below` is the control whose PASS is a low count."""
    n = len(hits)
    exact = sum(1 for h in hits if h == KEY_LEN)
    per_byte = sum(hits) / (KEY_LEN * n) if n else 0.0
    rate = exact / n if n else 0.0
    ok = rate <= bar if below else rate >= bar
    cmp = "<=" if below else ">="
    print(
        f"{label}: exact={exact}/{n} ({100 * rate:.0f}%) per-byte={100 * per_byte:.2f}% "
        f"wilson95-lower={100 * wilson_lower(exact, n):.1f}% bar={cmp}{bar:.2f} "
        f"-> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def run_items(model: torch.nn.Module, items: list[Item], segment: int, label: str) -> list[int]:
    """Score every item, one printed line each — no silent aggregation."""
    hits: list[int] = []
    for i, item in enumerate(items):
        h, pred, _ = score_item(model, item, segment)
        hits.append(h)
        print(
            f"[{label} i={i} seed={item.seed}] d={item.distance} B exact={h}/8 "
            f"pred={pred.decode('latin-1')!r} key={item.key.decode('latin-1')!r}",
            flush=True,
        )
    return hits


# --- controls --------------------------------------------------------------


def rotate_key(key: bytes) -> bytes:
    """A key that differs from `key` in the first byte — the shuffled-key control."""
    out = bytearray(key)
    assert out[0] in ALPHABET, "the key alphabet is the printable one"
    out[0] = ALPHABET[(ALPHABET.index(out[0]) + 1) % len(ALPHABET)]
    return bytes(out)


def no_needle_items(seed: int, context: int, n: int, align: int) -> list[Item]:
    """The key is drawn and then left OUT of the text; the target is that removed key."""
    return [build_item(seed + i, context, needle=False, align=align) for i in range(n)]


def shuffled_items(seed: int, context: int, n: int, align: int, distance: int | None) -> list[Item]:
    """Same haystack, rotated key: only the bytes ON THE PAGE can score."""
    out = []
    for i in range(n):
        base = build_item(seed + i, context, distance=distance, align=align)
        out.append(
            build_item(seed + i, context, distance=distance, key=rotate_key(base.key), align=align)
        )
    return out


def csa_off_model(run, ckpt_dir: Path, ckpt_name: str, dev: torch.device):
    """The SAME non-CSA weights with the branch removed, or a LOUD refusal.

    The checkpoint was trained with CSA present, so the file carries CSA tensors;
    `use_csa=false` builds no module for them. That is the control's point (the
    branch is not load-bearing), and `strict=False` + the counts printed here is
    how the unused tensors are NAMED instead of silently dropped. A MISSING
    non-CSA key is different: the file cannot express the off model at all.
    """
    if not run.model.use_csa:
        loud(
            "control csa-off: the snapshot is already use_csa=false, so there is no "
            "branch to remove and the control is the main run. Escape: a CSA snapshot."
        )
    off = dataclasses.replace(
        run.model,
        use_csa=False,
        csa_coarse_chunk=0,
        csa_mid_chunk=0,
        csa_fine=True,
        csa_fused=False,
        csa_hierarchical=False,
    )
    validate(off)
    model = _model(dataclasses.replace(run, model=off)).to(dev)
    half = load_model_half(ckpt_dir / f"{ckpt_name}.pt")
    result = load_model_state(model, run.train, half, strict=False)
    if result.missing_keys:
        loud(
            f"control csa-off: the snapshot cannot express use_csa=false — "
            f"{len(result.missing_keys)} non-CSA key(s) are missing, e.g. "
            f"{result.missing_keys[:3]}. Escape: the same recipe trained with "
            f"use_csa=false (its own checkpoint), or drop --control-csa-off."
        )
    if result.unexpected_keys:
        print(
            f"control csa-off: {len(result.unexpected_keys)} CSA tensor(s) in the file "
            f"are unused by the off model (e.g. {result.unexpected_keys[:3]})",
            flush=True,
        )
    return model


# --- dry run ---------------------------------------------------------------


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def dry_run(a: argparse.Namespace, segment: int) -> int:
    """Build the items, print §4.5's arithmetic and the exact real command. Exit 0.

    No checkpoint is read: the design numbers (chunk, kv width, slots, coarse
    chunk) are flags, because tonight no CSA snapshot exists to read them from.
    The real command is printed with a placeholder name on purpose — the run it
    names will hit gate A first, which is the correct next step.
    """
    align = a.csa_chunk
    ckpt_dir = a.ckpt_dir or Path("checkpoints") / (a.ckpt_name or "longctx1")
    items = [
        build_item(a.seed + i, a.context, distance=a.distance, align=align) for i in range(a.items)
    ]
    for item in items:
        assert len(item.scored) == a.context
    first = items[0]
    segments = a.context // segment
    fine_chunks = math.ceil(a.context / a.csa_chunk)
    flops = 0.5 * a.context * fine_chunks * a.csa_kv_dim * 2 * a.slots
    slot_bytes = total_state_bytes(
        a.context,
        a.csa_kv_dim,
        a.csa_chunk,
        csa_coarse_kv_dim=a.csa_coarse_kv_dim,
        csa_coarse_chunk=a.csa_coarse_chunk,
    )
    controls = "all" if a.controls else ",".join(c for c in CONTROLS if _control_on(a, c))
    print(
        f"S-NIAH-8 dry run · seed={a.seed} · context={a.context} B · items={a.items} · "
        f"segment={segment} B · segments/item={segments} · total segments={a.items * segments}",
        flush=True,
    )
    print(
        f"item[0]: key_at={first.key_at} distance={first.distance} B "
        f"key={first.key.decode('latin-1')!r} prompt={len(first.prompt)} B "
        f"scored={len(first.scored)} B",
        flush=True,
    )
    print(
        f"arithmetic (§4.1/§4.5): fine_chunks=ceil({a.context}/{a.csa_chunk})={fine_chunks} · "
        f"scores={a.context * fine_chunks} · FLOPs/pass="
        f"0.5*context*fine_chunks*kv*2*slots={flops:.3e} "
        f"(kv={a.csa_kv_dim}, slots={a.slots})",
        flush=True,
    )
    print(
        f"state (report §4.5): {_fmt_bytes(slot_bytes)}/slot × {a.slots} slots = "
        f"{_fmt_bytes(slot_bytes * a.slots)} (fine chunk={a.csa_chunk}, "
        f"coarse={a.csa_coarse_chunk})",
        flush=True,
    )
    tol = "1e-5 fp32 / 1e-3 bf16"
    rank = f"C rank {a.rank_items} items"
    if a.rank_items == 0:
        rank += " (NOT MEASURED)"
    print(
        f"gates: A artifact (would load {a.ckpt_name or '<a CSA snapshot>'} and refuse on "
        f"use_csa=false / max_seq_len<context) · B carry {2 * segment} B 1-pass vs 2×"
        f"{segment}, tol {tol} · {rank} · D {a.items} items",
        flush=True,
    )
    print(
        f"controls: {controls or 'none'} (bars: D>=95%, no-needle<=3%, shuffled>=95%, "
        f"csa-off<=10%, 1K>=95%)",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ckpt-dir",
        str(ckpt_dir),
        "--ckpt-name",
        a.ckpt_name or "longctx1",
        "--context",
        str(a.context),
        "--segment",
        str(a.segment),
        "--items",
        str(a.items),
        "--seed",
        str(a.seed),
        "--device",
        a.device,
        "--csa-chunk",
        str(a.csa_chunk),
        "--csa-kv-dim",
        str(a.csa_kv_dim),
        "--csa-coarse-chunk",
        str(a.csa_coarse_chunk),
        "--csa-coarse-kv-dim",
        str(a.csa_coarse_kv_dim),
        "--slots",
        str(a.slots),
    ]
    if a.distance is not None:
        cmd += ["--distance", str(a.distance)]
    if a.controls:
        cmd.append("--controls")
    else:
        cmd += [f"--{CONTROLS[c].replace('_', '-')}" for c in CONTROLS if _control_on(a, c)]
    print(f"command: PYTHONPATH=port/src {shlex.join(cmd)}", flush=True)
    if not a.ckpt_name:
        print(
            "note: gate A will refuse until a CSA checkpoint exists — that refusal is "
            "the correct first outcome, not a failure of this script (.bulba/plan.md).",
            flush=True,
        )
    return 0


# --- the real path ---------------------------------------------------------


def _control_on(a: argparse.Namespace, name: str) -> bool:
    return a.controls or bool(getattr(a, CONTROLS[name]))


def _items_for(a: argparse.Namespace, context: int, align: int) -> list[Item]:
    return [
        build_item(a.seed + i, context, distance=a.distance, align=align) for i in range(a.items)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S-NIAH-8: exact 8-byte retrieval through the state-carrying path"
    )
    ap.add_argument("--context", type=int, default=1 << 20, help="scored bytes per item")
    ap.add_argument("--segment", type=int, default=8192, help="bytes per carried segment")
    ap.add_argument("--items", type=int, default=100)
    ap.add_argument(
        "--distance", type=int, default=None, help="key-to-question bytes (default: max)"
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rank-items", type=int, default=10, help="gate C items (0 = not measured)")
    ap.add_argument("--controls", action="store_true", help="run ALL four controls")
    ap.add_argument("--control-no-needle", action="store_true")
    ap.add_argument("--control-shuffled", action="store_true")
    ap.add_argument("--control-csa-off", action="store_true")
    ap.add_argument("--control-1k", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build items, print arithmetic, exit 0")
    ap.add_argument("--artifact", action="store_true", help="gate A only (config + live gate)")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csa-chunk", type=int, default=8, help="dry-run design + key alignment")
    ap.add_argument("--csa-kv-dim", type=int, default=64, help="dry-run design")
    ap.add_argument("--csa-coarse-chunk", type=int, default=256, help="dry-run design")
    ap.add_argument("--csa-coarse-kv-dim", type=int, default=64, help="dry-run design")
    ap.add_argument("--slots", type=int, default=12, help="dry-run design: iterations/layers")
    a = ap.parse_args(argv)

    if a.items <= 0:
        loud("--items must be positive: a rate needs at least one item")
    if a.rank_items < 0:
        loud("--rank-items is a count (0 = not measured)")
    if a.context < 96:
        loud(f"--context {a.context} B is too short for the cue pair and an 8-byte key")
    if a.dry_run and a.artifact:
        loud("--dry-run and --artifact are different modes: dry-run needs no checkpoint")
    if a.dry_run:
        return dry_run(a, effective_segment(a.context, a.segment))
    if not a.ckpt_name:
        loud(
            "the real path needs --ckpt-name: the artifact is the first gate and there is "
            "no preset to guess it from. Escape: --dry-run (no checkpoint), or name the "
            "snapshot."
        )
    ckpt_dir = a.ckpt_dir or Path("checkpoints") / a.ckpt_name
    run = _snapshot_run(ckpt_dir, a.ckpt_name)
    why = artifact_refusal(run.model, a.context)
    if why is not None:
        loud(why)
    if a.artifact:
        model = _model(run)
        load_model_state(model, run.train, load_model_half(ckpt_dir / f"{a.ckpt_name}.pt"))
        why = gate_a_live(model)
        if why is not None:
            loud(why)
        return 0
    segment = effective_segment(a.context, a.segment)
    why = segmented_refusal(run.model)
    if why is not None:
        loud(why)

    why = preflight(a.device)
    if why is not None:
        return skip(why)
    dev = _device(a.device)
    model = _model(run).to(dev)
    load_model_state(model, run.train, load_model_half(ckpt_dir / f"{a.ckpt_name}.pt"))
    why = gate_a_live(model)
    if why is not None:
        loud(why)

    align = run.model.csa_chunk
    items = _items_for(a, a.context, align)
    print(
        f"S-NIAH-8 · {a.ckpt_name} · seed={a.seed} · context={a.context} B · "
        f"items={a.items} · segment={segment} B · segments/item={a.context // segment} · "
        f"distance={items[0].distance} B · score=teacher-forced argmax",
        flush=True,
    )

    # Gate B: a failed carry makes every later number a different function.
    tol = 1e-3 if run.model.bf16 else 1e-5
    delta, segments = carry_gate(model, items[0], segment)
    print(
        f"gate B: 1 pass vs 2×{segment} B segments · max|dlogits|={delta:.3e} · "
        f"tol={tol:.0e} ({'bf16' if run.model.bf16 else 'fp32'} logits) · "
        f"segments={segments}",
        flush=True,
    )
    if delta > tol:
        loud(
            f"gate B: the 2-segment pass is not the 1-pass forward (max|dlogits|="
            f"{delta:.3e} > {tol:.0e}). The 1M number would be a different function, so "
            f"the protocol stops here. Escape: fix the carry, or do not claim 1M "
            f"retrieval."
        )

    # Gate C: a chance rank means the selector, not the readout, is the wall.
    if a.rank_items:
        probe = items[: a.rank_items]
        ranks = rank_items(model, probe, run.model, segment)
        known = [r for r in ranks if r is not None]
        topk = run.model.csa_topk
        within = sum(1 for r in known if r <= topk)
        med = statistics.median(known) if known else float("inf")
        rate = within / len(ranks) if ranks else 0.0
        print(
            f"gate C: flat best-layer rank<=topk({topk}) in {within}/{len(ranks)} items · "
            f"median={med} · ranks={ranks}",
            flush=True,
        )
        if rate < 0.9 or med > topk:
            print(
                "gate C: FAIL — the selector is the bottleneck (chance rank is "
                f"~{math.ceil(a.context / run.model.csa_chunk)}). Per the protocol, gate D "
                "is NOT run; report the rank, not a retrieval rate.",
                flush=True,
            )
            return 0
    else:
        print(
            "gate C: NOT MEASURED (--rank-items 0) — no claim may cite the rank probe", flush=True
        )

    # Gate D: the retrieval number and its controls.
    main_hits = run_items(model, items, segment, "main")
    report(f"gate D {a.ckpt_name}", main_hits, BAR_D, below=False)

    if any(_control_on(a, c) for c in CONTROLS):
        seed = a.seed
        if _control_on(a, "no-needle"):
            hits = run_items(
                model, no_needle_items(seed, a.context, a.items, align), segment, "no-needle"
            )
            report(
                "control no-needle (hallucinated the removed key)", hits, BAR_NO_NEEDLE, below=True
            )
        if _control_on(a, "shuffled"):
            hits = run_items(
                model,
                shuffled_items(seed, a.context, a.items, align, a.distance),
                segment,
                "shuffled-key",
            )
            report("control shuffled-key", hits, BAR_D, below=False)
        if _control_on(a, "csa-off"):
            off = csa_off_model(run, ckpt_dir, a.ckpt_name, dev)
            hits = run_items(off, items, segment, "csa-off")
            report(
                "control csa-off (bar: a PASS here kills the 1M claim)",
                hits,
                BAR_CSA_OFF,
                below=True,
            )
        if _control_on(a, "1k"):
            context_1k = 1024
            seg_1k = effective_segment(context_1k, segment)
            dist_1k = None if a.distance is None else min(a.distance, context_1k - DISTANCE_SLACK)
            items_1k = [
                build_item(seed + i, context_1k, distance=dist_1k, align=align)
                for i in range(a.items)
            ]
            hits = run_items(model, items_1k, seg_1k, "positive-1k")
            report(
                f"control positive {context_1k} B (a failure here is the task, not length)",
                hits,
                BAR_1K,
                below=False,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
