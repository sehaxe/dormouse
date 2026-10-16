"""Presets and the run config, ported from the burn-side seam (ADR-0005/ADR-0021).

A preset is a FLAT TOML file of model keys — `configs/small.toml`,
`configs/b1.toml` — loaded by name or by explicit path. There is no registry in
Python either: adding a preset is adding a file.

Three rules, all of them the burn side's, and all of them load-time:

1. **Known keys only.** An unknown key is a LOUD error naming the key and the
   escape ("remove the key or port the mechanism"). It is not dropped: a key
   that reaches no code is a config that lies about what it runs.
2. **A missing key takes the class default.** Every field has one, so a sparse
   preset is legal — the defaults are the schema, not a second file.
3. **A key whose mechanism is not ported refuses when set NON-DEFAULT.** The
   field carries `supported = False` in its metadata and its off/default value;
   a preset that turns it on is refused with the mechanism named. This is the
   port's honest answer to the burn schema's 40-odd arms: the day-one skeleton
   ships the three mechanisms nothing in the ecosystem provides plus the
   instrument layer, and everything else is a dependency that has not landed.

`RunConfig` is the snapshot: the resolved model + train config plus provenance,
written next to a checkpoint as `<ckpt_name>.config.toml`. A resume refuses a
snapshot whose non-progress keys differ (ADR-0021) — extending a run is a legal
resume, changing what the model computes is not.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .instrument import LoudError, loud

#: `port/configs/`, reached from this file (`port/src/dormouse/config.py`).
PRESET_DIR = Path(__file__).resolve().parents[2] / "configs"

#: Metadata key marking a field whose mechanism is not ported yet.
UNSUPPORTED = "unsupported_mechanism"

#: Train keys a resume is allowed to change: they decide WHEN we look, never
#: what the model computes. Exactly the burn list (cfg.rs `PROGRESS_KEYS`).
PROGRESS_KEYS = ("steps", "log_every", "ckpt_every", "eval", "eval_every")

#: The `moe_kernel` literals — the expert-pass arms `experts._dispatch` routes
#: between. One list, read by `validate`: a name that is not here is LOUD.
MOE_KERNELS = ("padded", "triton", "naive")
#: The `expert_act` literals — the 2026-10-16 activation tournament arms, each
#: with its formula source in `experts.expert_glu`:
#: `silu` (control, SwiGLU), `situ` (Kimi K3, arXiv 2607.24653 Eq. 12),
#: `memglu` (closed-tail gate, arXiv 2608.07323). A new arm lands WITH its
#: gold test (`tests/test_expert_act.py`) before this tuple grows.
EXPERT_ACTS = ("silu", "situ", "memglu")


def _unsupported(default: Any, mechanism: str) -> Any:
    """A field whose mechanism has not been ported: refuses when non-default."""
    return field(default=default, metadata={UNSUPPORTED: mechanism})


@dataclass
class ModelConfig:
    """The model half of a preset — flat, every key a field, defaults == small.

    Keys that exist on the burn side whose mechanism is NOT in this port carry
    `_unsupported(...)`: they load at their default and refuse loudly when a
    preset sets them. The refusal is the ADR-0019 answer to a config field with
    no code behind it — the burn history's `use_mhc` reached a trainer that
    never read it, and no test could see that.
    """

    # --- geometry -------------------------------------------------------
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    d_ffn: int = 2048
    vocab: int = 256
    max_seq_len: int = 512
    max_iter: int = 4
    rank: int = 64
    norm_eps: float = 1e-3

    # --- precision ------------------------------------------------------
    #: bf16 activations/weights with fp32 masters (`docs/architecture/bf16-plan.md`).
    #: The forward matmuls run in bf16; the softmax, the loss and the TSCT
    #: latents stay fp32. Default True: the port's init path is the bf16 plan,
    #: unlike burn's `small.toml`, which still ships fp32 (listed in the port
    #: report as a deliberate deviation).
    bf16: bool = True
    #: TSCT factor format. `none` resolves to bf16 factors at init; every
    #: ladder rung (fp8 → NVFP4 → LittleBit) is a REFUSAL, not a fallback.
    quant: str = "none"

    # --- arms -----------------------------------------------------------
    use_kda: bool = True
    use_tsct: bool = True
    use_engram: bool = True
    use_attnres: bool = False
    #: LOOM (`2610.01153`, 2026-10-01; scanned in
    #: `docs/research/2026-10-16-new-tech-scan.md` §2.1, whose row names our
    #: own `max_iter 4` / operating depth 2 as exactly the regime that paper
    #: measures as stuck at two loops). A REAL arm, not an `_unsupported` stub:
    #: `LoopBlock` applies one learned per-iteration gate to the RMS-normalized
    #: write — per-loop router conditioning (the `T` passes of a weight-shared
    #: body are otherwise indistinguishable) plus the variance control the
    #: normalized write carries. Zero-init, so `loom = true` at init IS
    #: `loom = false` and pure-CE parity holds; the arm enables the scan's
    #: 8-12-loop claim ONLY if its A/B beats a fixed-depth control. It is a
    #: transcription of the SCAN's summary of the paper — the paper's full
    #: recipe (bounded residual updates, input re-injection) may differ, so
    #: re-derive before any multi-loop default.
    loom: bool = False
    use_gr: bool = _unsupported(False, "Gated Residual (GR)")
    use_mhc: bool = _unsupported(False, "Manifold-Constrained Hyper-Connections")
    use_mor: bool = _unsupported(False, "Mixture-of-Recursions router")
    use_situ: bool = _unsupported(False, "SiTU-GLU expert activation")
    use_byteflow: bool = False

    # --- ByteFlow (arXiv 2603.03583): the byte-compression architecture -----
    # A REAL arm, not an `_unsupported` stub: `byteflow.py` ports the crate's
    # five stages and `DormouseModel` wires them (see that module's docstring
    # for the Option A/B/C design decision). Every default below is
    # `ByteFlowConfig::default()` from `vendor/dormouse-fused/crates/
    # burn-byteflow/src/net.rs:290-310` — the crate's BFlowNet-600M numbers,
    # copied, not guessed. burn's own `schema.rs` ships a reduced
    # `byteflow_9m` build (d_local 96 / d_global 768 / K 128) for its presets;
    # that is a preset choice, and the port's presets all carry
    # `use_byteflow = false`, so the schema is the crate's.
    byteflow_d_local: int = 512
    byteflow_d_global: int = 1536
    #: Global sequence length K after chunking — the compression knob.
    byteflow_k_tokens: int = 3200
    #: Encoder depth E; reused by the decoder (paper §3.5).
    byteflow_e_layers: int = 6
    #: Global depth G (paper Table 5: `[6, 20]` layers at 600M).
    byteflow_g_layers: int = 20
    byteflow_heads_local: int = 8
    byteflow_heads_global: int = 12
    #: Sliding window `w_local` of the local encoder/decoder.
    byteflow_w_local: int = 512
    byteflow_d_ff_local: int = 1408
    byteflow_d_ff_global: int = 4096
    #: Shared upsampling bins B (paper §3.4).
    byteflow_bins: int = 16
    #: Noise variance ε² of the coding rate, eq. (11).
    byteflow_eps2: float = 0.5
    #: Score the chunker with the EXACT log-det rate instead of Appendix B's
    #: L2 fast path (the default: pure tensor ops, no host sync).
    byteflow_logdet: bool = False
    #: Byte-sequence length the RoPE tables are built for.
    byteflow_max_bytes: int = 8192

    # --- MoE ------------------------------------------------------------
    n_experts: int = 3
    #: Experts per position. `0` is the DENSE blend (burn's default), `k` is
    #: the sparse top-k dispatch. Above `n_experts` is refused by validate.
    moe_topk: int = 0
    #: Tokens per expert in the capacity-bound permute:
    #: `ceil(capacity * tokens * topk / n_experts)`, never below `topk`.
    moe_capacity: float = 1.0
    #: Loss-Free Balancing step (arXiv:2408.15664): before the top-k decision
    #: the scores read `probs + b_e`, and after the forward `b_e` moves by
    #: `-coef · sign(load_e − mean load)` — a bias, NOT an aux loss (no
    #: interference gradients; the field converged here away from Switch-style
    #: losses). 0.0 = off. The update is on-device per routed forward.
    moe_lb_coef: float = 0.0
    #: Which expert-pass arm the routed forward runs, and the A/B switch
    #: (`docs/research/2026-10-15-tsct-super-kernel.md` step 2):
    #: `padded` = the cap-padded `[E, cap, ·]` bank of bmms — the reference and
    #: the fallback every other arm is compared against; `triton` = the unpadded
    #: grouped GEMM (`model/moe_kernels.py`, one program per ragged row-block,
    #: GLU fused); `naive` = the per-expert loop (the CPU path). An unknown name
    #: is LOUD at config load. `triton` falls back COUNTED to `padded` when the
    #: kernel cannot run (no triton import, input not on CUDA) — never silently.
    moe_kernel: str = "padded"
    #: The expert GLU's activation, the 2026-10-16 activation tournament arm
    #: (`docs/research` scan; formula sources named in `experts.expert_glu`):
    #: `silu` = SwiGLU's gate, the control and every number on record;
    #: `situ` = Kimi K3's Sigmoid-Tanh-Unit GLU (arXiv 2607.24653 Eq. 12) —
    #: both factors soft-capped, ‖·‖∞ ≤ β1·β2, matches SwiGLU to first order at
    #: the origin; `memglu` = closed-tail gate (arXiv 2608.07323 Eq. —, c0
    #: RMS-calibrated by the authors). Unknown name is LOUD at config load; a
    #: non-`silu` arm keeps the padded torch path (the fused triton kernel has
    #: silu in registers and is skipped COUNTED, `moe_triton_fallback`).
    expert_act: str = "silu"
    #: SiTU's soft-cap scales, Kimi K3's own values (§2.3.2: β1 = 4 gate,
    #: β2 = 25 up). ‖SiTU-GLU(x)‖∞ ≤ β1·β2 = 100 at these.
    expert_beta1: float = 4.0
    expert_beta2: float = 25.0
    #: MemGLU's RMS-calibration constant — the authors' 9M-scale value
    #: (`2608.07323` appendix table: 0.6039096539628113).
    memglu_c0: float = 0.6039096539628113
    #: SL1 of `docs/research/2026-10-16-budget-savers.md` §5: park the routed
    #: expert pass's saved activations in HOST RAM instead of VRAM
    #: (`experts._expert_pass`: `checkpoint(use_reentrant=False)` with
    #: `saved_tensors_hooks` inside). Loss and gradients are EXACT — the values
    #: are copied, never approximated (`tests/test_offload_experts.py`) — and
    #: `moe_offloaded` on the counter line is what proves the arm ran. Off by
    #: default: this round ships the switch and its parity gate, and the b1
    #: recipe turns it ON when it lands as a default. The ROUTED pass only:
    #: with `moe_topk = 0` there is no capacity-padded bank to offload and the
    #: switch is COUNTED inert (`moe_offload_inert_dense`).
    moe_offload: bool = False
    #: Rank of the always-on shared expert (TSCT2 §14's `R_act` ≥ 192 zone).
    shared_rank: int = 192

    # --- aux objectives -------------------------------------------------
    #: The FUTURE-BYTE head (arXiv 2404.19737, `model/fb.py`): a REAL arm, not
    #: an `_unsupported` stub. It is the pre-registered competitor to JEPA
    #: (AB-PROTOCOL row 1b) — a dense untied `d_model -> vocab` readout over the
    #: state `lm_head` already reads, one plain CE on the label rolled by -1,
    #: 196 864 params at `small` (2.14%), no extra forward. 0.0 = off, and the
    #: default is off: a preset nobody edited must not carry an untrained head.
    #: `validate` refuses it together with `jepa_weight`/`dspark_weight` — the
    #: A/B attributes its verdict to ONE signal per run.
    aux_fb_weight: float = 0.0
    #: ---- the DSpark draft head (2607.05147; port wiring 2026-10-16) ----
    #: The DSpark term adds next-K-byte draft supervision + the acceptance
    #: head. `dspark_weight` = γ (D2: the loss is γ-free, the CALLER scales);
    #: `dspark_k` = the draft reach in bytes; `dspark_stride` = the anchor
    #: spacing in byte positions (0 refused). `dspark_markov_rank` sizes the
    #: D9 markov feature. Refused with use_byteflow (the channel has no aux).
    dspark_weight: float = 0.0
    dspark_k: int = 4
    dspark_stride: int = 16
    dspark_markov_rank: int = 256
    #: SIGReg / LeJEPA (arXiv:2511.08544, `model/sigreg.py`): a REGULARISER, not
    #: a head. `loss = CE + sigreg_weight · sigreg_loss(z)`, `z` the strided
    #: sample of the state `lm_head` reads — no teacher, no predictor, no mask,
    #: no second forward, no parameters, no detach. 0.0 = off. The research
    #: doc's ONE arm to run first (`docs/research/2026-10-16-jepa-into-dormouse.md`
    #: §3; the burn `l1 + 0.1·koleo` bundle cannot attribute its own 0.082 BPB).
    #: The suggested grid is 0.005 / 0.05 / 0.1 — publish which.
    #: COEXISTS with `aux_fb_weight` by design: a regulariser and a head are
    #: different classes, so the one-signal-per-run rule (which is about two
    #: HEADS in one run) does not apply to this pair.
    sigreg_weight: float = 0.0
    #: ---- the second attention arm (MSA re-entry, ADR-0014; owner 2026-10-16) ----
    #: 1M-context sparse attention BESIDE KDA (which stays the default arm).
    #: `msa_capacity_ratio` = the arm's share of the attention demand (owner:
    #: 1/4 of KDA's); `adaptive_msa` = the model decides WHERE that 1/4 goes
    #: (per-iteration learned share, mean = the ratio; the `msa_gate` blend).
    use_msa: bool = False
    msa_topk: int = 32
    msa_block: int = 4
    msa_capacity_ratio: float = 0.25
    adaptive_msa: bool = False
    jepa_weight: float = _unsupported(0.0, "JEPA + KoLeo aux heads")
    act_quant: str | None = _unsupported(None, "BitNet-style activation quantization (STE)")
    act_group: int = 0

    # --- hashed n-gram memory (Engram) ---------------------------------
    #: Rows PER ORDER, rounded UP to a power of two when the table is built
    #: (the in-VRAM read masks the raw key, so the rounding is what makes the
    #: index in-range by construction — burn's `engram_tables`).
    engram_rows: int = 25000
    engram_orders: list[int] = field(default_factory=lambda: [2, 3, 4])
    engram_dim: int = 32
    #: Hard floor on the memory's share of its branch: `lam = min(w_mem, lam_max)`.
    engram_lam_max: float = 0.5

    def resolved_quant(self) -> str:
        """The factor format the forward actually runs (`none` → bf16)."""
        return "bf16" if self.quant == "none" else self.quant


@dataclass
class TrainConfig:
    """The train half: optimizer, cadences, data. Progress keys are exempt
    from the resume drift check (ADR-0021) and nothing else is."""

    steps: int = 200
    batch: int = 8
    seq_len: int = 512
    lr: float = 3e-4
    wd: float = 0.01
    grad_clip: float = 1.0
    #: TF32 for the remaining fp32 matmuls (the Muon Newton-Schulz iterations,
    #: the KDA internals, the fp32 lm_head path): measured 2026-10-16 at b1
    #: b8x1024 as 1399 -> 1201 ms/step (+16%, mostly the optimizer's NS bmm).
    #: OFF by default until a loss-parity check names its cost; NS is already
    #: an approximate iteration, so the expected cost is small — that is a
    #: prediction, not a substitute for the check.
    tf32: bool = False
    #: bitsandbytes 8-bit state for the AdamW group. The group at the FULL b1
    #: preset includes the Engram tables (402M params => 3.2 GB of fp32 m+v),
    #: which is exactly what this saves; measured cost +~180 ms/step with the
    #: tables OFF (2026-10-16), so it is the VRAM-fit lever, not a speed knob.
    optim_8bit: bool = False
    log_every: int = 10
    ckpt_every: int = 100
    ckpt_name: str = "port"
    eval: bool = False
    eval_every: int = 50
    eval_batches: int = 20
    #: `mix` = Muon on 2-D params + AdamW elsewhere (the default); `adamw` =
    #: the fallback arm, which is the same code with the routing disabled.
    opt: str = "mix"
    seed: int = 1
    #: WSD warmup, in TOKENS (not steps): the schedule is a function of how
    #: much data the run has seen, so a batch-size change does not change it.
    warmup_tokens: int = 0
    #: Fraction of the token budget spent on the WSD decay.
    decay_frac: float = 0.1
    #: JSONL telemetry sink: one row per event (log line, eval, checkpoint),
    #: consumed by `dormouse plot`. Empty = off. NOT a progress key — it is
    #: where measurements go, never what the run computes; re-pointing it on
    #: a resume is legal and expected for a moved run.
    json_log: str = ""
    data: str = ""
    eval_data: str = ""
    out_dir: str = "checkpoints"
    timers: bool = False
    #: Host-RAM Engram tables — burn's `--engram-ram` (Qwen3.8-Flash-Next §2.3
    #: via `crates/dormouse-train/src/offload.rs`): the n-gram tables and their
    #: ONE momentum buffer live in plain CPU fp32 memory (`Engram.to_host`) and
    #: only the batch's unique rows cross the bus, so the memory's VRAM is
    #: `batch × unique rows` rather than `orders × slots × dim × 4`. In-VRAM is
    #: the DEFAULT (the day-one table; the optimizer policy routes it to AdamW),
    #: and this flag is what a run switches on to buy memory capacity.
    #:
    #: Three things about it are refusals rather than fallbacks: a CPU run
    #: (host tables ARE CPU memory — the in-VRAM path is all the playbook is for,
    #: and the run would pay the row transfers for no VRAM), the byteflow channel
    #: (`use_byteflow`: no byte positions to hash), and `use_engram = false`
    #: (there is no table to host). Flipping the flag on a resume is refused by
    #: the snapshot drift check — it changes the parameter set AND the optimizer.
    engram_ram: bool = False
    #: The ADAPTIVE DEPTH arm (owner 2026-10-16): run the loop `T` iterations
    #: per TRAIN step, `T` sampled in `1..=max_iter` as a PURE function of
    #: `(seed, step)` (`train.loop._sample_depth`), with the loss normalized by
    #: the depth ACTUALLY run so depths are scale-comparable. `false` is
    #: ADR-0013's fixed depth — the DEFAULT; this field is the arm. It is a
    #: function of `(seed, step)` and not a global draw because burn's RNG is
    #: never seeded: two runs of one config would draw different depth
    #: schedules and a resume would continue a different objective (ADR-0021,
    #: the defect that fixed DSpark's stride).
    rand_depth: bool = False
    #: Print the held-out loss at depths 1..=max_iter on every eval line — the
    #: depth-efficiency curve a depth claim is measured by (the plan's
    #: `--eval-depths`). MEASUREMENT only, it trains nothing. It is on whenever
    #: `rand_depth` is (the arm must show its own curve), and can be asked for
    #: explicitly on a fixed-depth run.
    eval_depths: bool = False
    stress: bool = _unsupported(False, "stress protocol (--stress)")


@dataclass
class RunConfig:
    """A fully resolved run: provenance + both halves of the config."""

    source: str
    model: ModelConfig
    train: TrainConfig

    def diff_keys(self, other: RunConfig) -> list[str]:
        """`model.*` / `train.*` keys where `self` and `other` differ.

        `source` is provenance (a name and a path can resolve to the same
        values) and is excluded. Progress keys are exempt on the train side —
        extending a finished run is a legitimate resume. Everything else is
        strict, including every knob that reaches a forward.
        """
        out: list[str] = []
        for section, a, b in (
            ("model", self.model, other.model),
            ("train", self.train, other.train),
        ):
            for f in dataclasses.fields(a):
                if section == "train" and f.name in PROGRESS_KEYS:
                    continue
                if getattr(a, f.name) != getattr(b, f.name):
                    out.append(f"{section}.{f.name}")
        return sorted(out)

    def snapshot_toml(self) -> str:
        """The `<ckpt_name>.config.toml` record of this run."""
        return _to_toml({"model": self.model, "train": self.train})

    @classmethod
    def from_snapshot(cls, text: str) -> RunConfig:
        """Inverse of [`snapshot_toml`], strict about unknown keys."""
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            loud(f"config snapshot does not parse: {e}")
        return cls(
            source=raw.get("source", ""),
            model=_from_table(ModelConfig, raw.get("model", {}), where="snapshot.model"),
            train=_from_table(TrainConfig, raw.get("train", {}), where="snapshot.train"),
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _refuse_unsupported(cls: type, table: dict[str, Any], where: str) -> None:
    """A field whose mechanism is missing refuses when it is set NON-DEFAULT."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for name, value in table.items():
        mechanism = fields[name].metadata.get(UNSUPPORTED)
        if mechanism is not None and value != fields[name].default:
            loud(
                f"{where}.{name}={value!r} is not supported by this port: the "
                f"{mechanism} mechanism is not ported yet "
                f"(docs/decisions/2026-10-15-1b-readiness-plan.md §7 ships the "
                f"minimum the ecosystem lacks: TSCT, Engram, the byte stream). "
                f"Escape: remove the key or port the mechanism."
            )


def _known(cls: type, table: dict[str, Any], where: str) -> dict[str, Any]:
    """Strict key check: a table is a config only if every key is a field."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [k for k in table if k not in fields]
    if unknown:
        loud(
            f"{where}: unknown key(s) {sorted(unknown)}. Known keys: "
            f"{len(fields)}. Escape: remove the key, or port the mechanism it "
            f"was meant to drive and add the field to {cls.__name__}."
        )
    _refuse_unsupported(cls, table, where)
    return dict(table)


def _from_table(cls: type, table: dict[str, Any], where: str) -> Any:
    return cls(**_known(cls, table, where))


def validate(model: ModelConfig) -> None:
    """Coherence checks the burn side does in `config::validate`. Loud, never
    clamped: a clamped width is exactly the silent collapse ADR-0011 bans."""
    if model.d_model <= 0 or model.vocab <= 0:
        loud(f"d_model and vocab must be > 0, got {model.d_model}/{model.vocab}")
    if model.n_heads * model.head_dim != model.d_model:
        loud(
            f"n_heads*head_dim={model.n_heads * model.head_dim} != d_model="
            f"{model.d_model}: the head split must tile the residual width"
        )
    if model.rank < 1 or model.shared_rank < 1:
        loud(f"rank and shared_rank must be >= 1, got {model.rank}/{model.shared_rank}")
    if not 0.0 < model.engram_lam_max <= 1.0:
        loud(f"engram_lam_max={model.engram_lam_max} outside (0, 1]: it is a share")
    if len(model.engram_orders) == 0:
        loud("engram_orders is empty: the memory has no table to read")
    if model.moe_topk > model.n_experts:
        loud(
            f"moe_topk={model.moe_topk} > n_experts={model.n_experts}: a top-k "
            f"above the bank is refused, not clamped (clamping is the silent collapse)"
        )
    if model.moe_capacity <= 0.0:
        loud(f"moe_capacity={model.moe_capacity} must be > 0")
    if model.moe_kernel not in MOE_KERNELS:
        loud(
            f"moe_kernel={model.moe_kernel!r} is not an arm. Known: "
            f"{list(MOE_KERNELS)} — `padded` the cap-padded bmm bank (reference "
            f"and fallback), `triton` the unpadded grouped GEMM "
            f"(model/moe_kernels.py), `naive` the per-expert loop. Escape: name "
            f"one of those, or add the arm before the config can lie about it."
        )
    if model.expert_act not in EXPERT_ACTS:
        loud(
            f"expert_act={model.expert_act!r} is not an arm. Known: "
            f"{list(EXPERT_ACTS)} — `silu` the SwiGLU control, `situ` Kimi K3's "
            f"capped GLU (arXiv 2607.24653 Eq. 12, β1={model.expert_beta1}, "
            f"β2={model.expert_beta2}), `memglu` the closed-tail gate "
            f"(arXiv 2608.07323). Escape: name one of those, or land the arm and "
            f"its gold test BEFORE the config can lie about it."
        )
    if model.expert_beta1 <= 0.0 or model.expert_beta2 <= 0.0:
        loud(
            f"SiTU βs must be > 0 (gate β1={model.expert_beta1}, up "
            f"β2={model.expert_beta2}) — they are soft-cap scales, and a "
            f"non-positive one divides by zero or flips the cap."
        )
    if model.memglu_c0 <= 0.0:
        loud(f"memglu_c0={model.memglu_c0} must be > 0 (an RMS scale, not a sign)")
    if model.use_byteflow:
        _validate_byteflow(model)
    if model.aux_fb_weight < 0.0:
        loud(
            f"aux_fb_weight={model.aux_fb_weight} is negative: it is the "
            f"multiplier on the future-byte CE term, not a sign to flip. "
            f"Escape: 0.0 (off) or a positive weight (AB-PROTOCOL row 1b: 0.1)."
        )
    if model.sigreg_weight < 0.0:
        loud(
            f"sigreg_weight={model.sigreg_weight} is negative: it is the "
            f"multiplier on the SIGReg term, not a sign to flip. Escape: 0.0 "
            f"(off) or a positive weight (the research doc's grid: 0.005, 0.05, "
            f"0.1 — docs/research/2026-10-16-jepa-into-dormouse.md §3)."
        )
    if model.aux_fb_weight > 0.0 and (model.jepa_weight > 0.0 or model.dspark_weight > 0.0):
        which = "jepa_weight" if model.jepa_weight > 0.0 else "dspark_weight"
        loud(
            f"aux_fb_weight={model.aux_fb_weight} with {which}>0: the two aux "
            f"signals cannot share a run. The A/B (AB-PROTOCOL row 1b) puts the "
            f"future-byte head against pure CE so its verdict is attributed to "
            f"ONE signal; a run carrying both moves two things at once and "
            f"attributes the result to neither. Escape: set {which} = 0.0, or "
            f"aux_fb_weight = 0.0."
        )

    if model.quant != "none":
        _refuse_ladder_rung(model.quant)


def _validate_byteflow(model: ModelConfig) -> None:
    """Coherence of the ByteFlow arm's own fields and of what it is wired to.

    The chunker's RoPE table covers `byteflow_max_bytes`, so a longer window
    must fail here with the field named, not at the first forward; `K ≤ T`
    because every boundary is a position inside the window; and the local /
    global widths must tile their head splits (`byteflow.py`, `ByteFlowNet`).
    """
    if model.max_seq_len > model.byteflow_max_bytes:
        loud(
            f"max_seq_len={model.max_seq_len} > byteflow_max_bytes="
            f"{model.byteflow_max_bytes}: the chunker's RoPE table covers "
            f"byteflow_max_bytes positions, so a longer sequence is a loud forward "
            f"assert. Escape: raise byteflow_max_bytes with the model."
        )
    if model.byteflow_k_tokens < 1:
        loud("byteflow_k_tokens must be >= 1 (at least the BOS boundary)")
    if model.byteflow_k_tokens > model.max_seq_len:
        loud(
            f"byteflow_k_tokens={model.byteflow_k_tokens} > max_seq_len="
            f"{model.max_seq_len}: the chunker's Top-K needs K <= T, every boundary a "
            f"position inside the window. Escape: raise max_seq_len or lower K."
        )
    if model.byteflow_eps2 <= 0.0:
        loud("byteflow_eps2 must be > 0 (it divides d inside the logdet)")
    if model.byteflow_bins < 1:
        loud("byteflow_bins must be >= 1")
    for d, h, side in (
        (model.byteflow_d_local, model.byteflow_heads_local, "local"),
        (model.byteflow_d_global, model.byteflow_heads_global, "global"),
    ):
        if h < 1 or d % h != 0:
            loud(
                f"byteflow_{side} width {d} does not divide by its head count {h}: "
                f"the head split must tile the width. Escape: fix the byteflow_* widths."
            )
    # fp32 only, and there is no second reading of that: the ported net has no
    # bf16 path (the crate refuses --bf16 for the same reason — a cast copy
    # dressed as a dtype).
    if model.bf16:
        loud(
            "use_byteflow with bf16=true: the ByteFlow net has no bf16 handling, so the "
            "flag would either be a cast copy around every projection or a silent fp32 "
            "run wearing a bf16 label. Escape: `bf16 = false` (the paper's recipe), or "
            "drop use_byteflow."
        )
    # The channel runs OUR loop on K patch latents, and the memory arm needs a
    # byte position to hash an n-gram from — it has none on that path. Refused
    # rather than silently inert (an arm that runs and trains nothing is the
    # defect class the counters exist for).
    if model.use_engram:
        loud(
            "use_byteflow with use_engram=true: the channel runs the loop over K patch "
            "latents, so the n-gram memory has no byte position to read and would be "
            "built, counted and trained by nothing. Escape: `use_engram = false` (the "
            "byteflow arm alone carries the compression), or drop use_byteflow."
        )
    # SIGReg reads the state `lm_head` reads, and this channel's readout is
    # `bf.out_proj(out_acc)` — not that state. A weight with nothing to attach
    # to is refused, never silently dropped (ADR-0019).
    if model.sigreg_weight > 0.0:
        loud(
            "use_byteflow with sigreg_weight>0: the channel's readout is "
            "`bf.out_proj(out_acc)`, not the `norm(out)` state SIGReg reads, so the "
            "term would be built and attached to nothing. Escape: sigreg_weight = 0.0, "
            "or drop use_byteflow."
        )
    if model.dspark_weight > 0.0:
        loud(
            "use_byteflow with dspark_weight>0: the channel's forward carries no "
            "byte ids for the draft window to read. Escape: dspark_weight = 0.0, "
            "or drop use_byteflow."
        )
    if model.dspark_weight < 0.0:
        loud(
            f"dspark_weight={model.dspark_weight} is negative: it is γ, the "
            f"multiplier on the γ-free DSpark loss (D2). Escape: 0.0 (off) or a "
            f"positive weight."
        )
    if model.dspark_k < 1 or model.dspark_stride < 1:
        loud(
            f"dspark_k={model.dspark_k} / dspark_stride={model.dspark_stride}: "
            f"both are positive counts (the draft reach and the anchor spacing, "
            f"in bytes). Escape: k >= 1, stride >= 1."
        )
    if model.dspark_weight > 0.0 and model.aux_fb_weight > 0.0:
        loud(
            "dspark_weight with aux_fb_weight>0: two heads on one run breaks the "
            "one-signal-per-run A/B rule. Escape: pick one."
        )


def _refuse_ladder_rung(rung: str) -> str:
    """The quant ladder: `fp8`/`nvfp4` are Storage rungs since 2026-10-15 (they
    compress the factor bank and the checkpoint, the fp32 masters stay the
    optimizer's object — see `tsct.py` module doc); everything else refuses.

    GROUPED, not a fallback. What remains refused (`fp16/lbit/ternary/act
    quant`) is a TRAINING-time change — LittleBit's gates G-A..G-D
    (`docs/reviews/littlebit-design-2026-10-15.md`) must close first.
    """
    storage = ("fp8", "nvfp4")
    if rung in storage:
        return rung
    known = {"fp16", "fp4", "int4", "int8", "lbit", "ternary"}
    hint = (
        " (a training-time rung: docs/reviews/littlebit-design-2026-10-15.md "
        "carries the gates that must close first)"
        if rung in known
        else ""
    )
    loud(
        f"quant={rung!r} is not supported{hint}: the implemented rungs are the "
        f"STORE levels fp8/nvfp4 (checkpoint/bank compression; compute stays "
        f"bf16 factors with fp32 masters — docs/architecture/bf16-plan.md). "
        f'Escape: `quant = "none"` (bf16), `quant = "fp8"`, `quant = "nvfp4"`, '
        f"or close the LittleBit gates."
    )


def _candidates(name: str) -> list[Path]:
    """Search order, burn's: cwd `configs/`, then the port's own `configs/`."""
    return [Path(f"configs/{name}.toml"), PRESET_DIR / f"{name}.toml"]


def load_preset(name_or_path: str) -> ModelConfig:
    """Load a preset by NAME (search) or by explicit PATH.

    A bare name that is found nowhere is LOUD with every path tried, not a
    default: a missing preset is a typo, and a silent default would train a
    different model than the one asked for.
    """
    if "/" in name_or_path or name_or_path.endswith(".toml"):
        p = Path(name_or_path)
        if not p.is_file():
            loud(f"preset path {p} does not exist")
        return parse_preset(p.read_text(), where=str(p))
    name = name_or_path.removesuffix(".toml")
    tried = _candidates(name)
    for p in tried:
        if p.is_file():
            return parse_preset(p.read_text(), where=str(p))
    loud(f"unknown preset {name_or_path!r}; tried {[str(p) for p in tried]}")


def parse_preset(text: str, where: str = "<preset>") -> ModelConfig:
    """Parse + check a flat preset table. The one entry point for strictness."""
    try:
        table = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        loud(f"{where}: toml parse: {e}")
    model = _from_table(ModelConfig, table, where=where)
    validate(model)
    return model


def load_run(
    preset: str = "small",
    *,
    train: TrainConfig | None = None,
    train_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Resolve a run: preset (name or path) + both config halves + loud overrides.

    Model overrides get the same two refusals the preset file itself gets
    (`_refuse_unsupported` for a missing mechanism, `validate` for a
    self-inconsistent set), because an override that bypasses them is the
    silent-collapse shape the preset layer exists to prevent.
    """
    model = load_preset(preset)
    tcfg = train if train is not None else TrainConfig()
    if train_overrides:
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        unknown = [k for k in train_overrides if k not in known]
        if unknown:
            loud(
                f"unknown train key(s) {sorted(unknown)}. Escape: remove the "
                f"key, or add the field to TrainConfig and wire it."
            )
        _refuse_unsupported(TrainConfig, train_overrides, "train")
        tcfg = dataclasses.replace(tcfg, **train_overrides)
    if tcfg.opt not in ("mix", "adamw"):
        loud(f"opt={tcfg.opt!r}: known optimizers are 'mix' (default) and 'adamw'")
    if model_overrides:
        _refuse_unsupported(ModelConfig, model_overrides, "model")
        import dataclasses as _dc

        model = _dc.replace(model, **model_overrides)
        validate(model)
    if tcfg.batch < 1 or tcfg.seq_len < 1 or tcfg.steps < 1:
        loud(f"batch={tcfg.batch} seq_len={tcfg.seq_len} steps={tcfg.steps}: all must be >= 1")
    if tcfg.seq_len > model.max_seq_len:
        loud(
            f"seq_len={tcfg.seq_len} exceeds max_seq_len={model.max_seq_len}: "
            f"the run would train on sequences the model refuses to serve"
        )
    if model.use_byteflow and model.byteflow_k_tokens > tcfg.seq_len:
        loud(
            f"byteflow_k_tokens={model.byteflow_k_tokens} > seq_len={tcfg.seq_len}: the "
            f"chunker promotes K boundaries inside the T-byte window, so K > T has no "
            f"positions to select. Escape: raise `seq_len`/`--batch`'s window or lower "
            f"byteflow_k_tokens."
        )
    if tcfg.engram_ram:
        # The host path is a MODEL-shape decision that spans both halves of the
        # config, so its refusals live here, where both are resolved. They are
        # refusals and not fallbacks: a run that asked for host tables and
        # silently got the in-VRAM ones would report the VRAM it did not save.
        if model.use_byteflow:
            loud(
                "engram_ram with use_byteflow=true: the channel runs the loop over K "
                "patch latents, so the n-gram memory has no byte position to hash and "
                "there are no rows to host. Escape: use_byteflow=false, or "
                "engram_ram=false."
            )
        if not model.use_engram:
            loud(
                "engram_ram with use_engram=false: the model builds no table, so "
                "`engram_ram` would host nothing while the optimizer silently keeps its "
                "in-VRAM shape. Escape: use_engram=true, or engram_ram=false."
            )
    return RunConfig(source=preset, model=model, train=tcfg)


# --------------------------------------------------------------------------
# TOML writing (a snapshot is two flat tables: no dependency needed)
# --------------------------------------------------------------------------


def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_scalar(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    raise LoudError(f"cannot serialize {v!r} into a config snapshot")


def _to_toml(sections: dict[str, Any]) -> str:
    """Two flat tables. `None` is OMITTED — a missing key takes the class
    default, which is the schema's own rule for an unset optional."""
    out: list[str] = []
    for name, obj in sections.items():
        out.append(f"[{name}]")
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is not None:
                out.append(f"{f.name} = {_scalar(v)}")
        out.append("")
    return "\n".join(out)
