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
#: what the model computes. The burn list (cfg.rs `PROGRESS_KEYS`) plus
#: `json_log` — the telemetry sink is WHERE the measurement goes and it reaches
#: no forward (audit MINOR 23, 2026-10-17: the field's own docstring promised
#: this exemption while `diff_keys` refused the flip).
PROGRESS_KEYS = ("steps", "log_every", "ckpt_every", "eval", "eval_every", "json_log")

#: The `moe_kernel` literals — the expert-pass arms `experts._dispatch` routes
#: between. One list, read by `validate`: a name that is not here is LOUD.
MOE_KERNELS = ("padded", "triton", "naive")
#: The `expert_act` literals — the 2026-10-16 activation tournament arms, each
#: with its formula source in `experts.expert_glu`:
#: `silu` (control, SwiGLU), `situ` (Kimi K3, arXiv 2607.24653 Eq. 12),
#: `memglu` (closed-tail gate, arXiv 2608.07323). A new arm lands WITH its
#: gold test (`tests/test_expert_act.py`) before this tuple grows.
EXPERT_ACTS = ("silu", "situ", "memglu")
#: The `factor_qat` literals — what the FORWARD does to the TSCT factor
#: masters (`U`, `V`; the diagonal `s` rides as the scale, never quantized).
#: `none` = bf16 masters (every number on record); `ternary` = b1.58 STE
#: (arXiv:2402.17764 Eq. 1, `ternary.ternarize`, per-TENSOR absmean — the
#: axis burn-bitnet uses) with the soft-to-hard annealing burn-spectral
#: measured ("converges better than hard STE from step one", lib.rs:75-77);
#: `binary` = LittleBit's sign factors (arXiv:2506.13771: `sign(W)·mean(|W|)`,
#: `ternary.sign_binary`) under the same annealing; `mxfp4` = the OCP MX e2m1
#: codec (MX spec §5.2 Table 1, `mxfp.fake_quant_mx` — per-32 e8m0 scales,
#: RNE) under the same annealing, with the last dim zero-padded per row to a
#: 32-multiple when it is not one; `ternary_s3t` = S3T stochastic rounding
#: (arXiv:2412.04787, `ternary.stochastic_ternary`) under the same annealing,
#: its Bernoulli mask drawn from a process-local generator seeded by
#: `factor_qat_seed` (reproducible within a run, NOT across a resume —
#: `tsct.factor_qat_project`'s docstring and the line `train/loop.py` prints).
#: The compressed-forward sprint's arms; a new rung lands WITH its gold test
#: (`tests/test_factor_qat.py`) before this tuple grows.
FACTOR_QATS = ("none", "ternary", "binary", "mxfp4", "ternary_s3t")
#: The `act_quant` literals — the a4.8 ACTIVATION formats (`model/act_quant.py`,
#: transcribed from `crates/dormouse-core/src/act_quant.rs`): `none` = no
#: activation quantization (every number on record); `int8`/`int4` = symmetric
#: integer levels `2^(bits-1)-1` (127/7); `fp4` = e2m1, the OCP MX FP4 grid
#: (8 magnitudes, `tsct._NVFP4_MAGNITUDES`) scaled onto the format's MAX, which
#: is what makes all 8 reachable (the old 3-level bug: AGENTS §3.7 act-quant).
#: The attention path is upgraded to >= 8 bits unconditionally
#: (`act_quant.attn_format`), so `fp4`/`int4` never run 4-bit attention. Two
#: imported families dodge that scale rule: `mxfp8` = the OCP MX e4m3 codec
#: (spec §5.2, `mxfp.fake_quant_mx`; last dim padded per row to a 32-multiple
#: when needed), `v2_int4`/`v2_int8` = BitNet v2's Hadamard-rotated integer arms
#: (arXiv:2504.18415, `hadamard.v2_quantize`; needs a 2^k last dim, LOUD
#: otherwise — the rotation of a padded row is a different basis). All three
#: refuse a non-zero `act_group`: their grid is not the caller's to choose.
#: A new format lands WITH its gold test (`tests/test_act_quant.py`) before
#: this tuple grows.
ACT_QUANTS = ("none", "int8", "int4", "fp4", "mxfp8", "v2_int4", "v2_int8")
#: The `ns_kind` literals — the Newton–Schulz arms the Muon group runs
#: (`optim.NS_KINDS`, which is where they are executed; the list is duplicated
#: here the way `MOE_KERNELS` is, so the config layer does not import the
#: trainer). `classic` = the per-iteration `X @ X.T` recurrence: the default,
#: and every number on record. `triton` = the SAME chain FUSED into
#: ONE kernel launch (`optim/ns_kernels.py`): the whole Jordan chain runs inside
#: one program per slice with the `[r, r]` Gram in registers, against ~52 aten
#: dispatches for a `[2048, 64]` factor (measured, `scripts/bench_ns_triton.py`).
#: Its claim is a launch count, not a number, so its verdict is the speed rule —
#: and it is COUNTED: `ns_triton` when it ran, `ns_triton_fallback` + the reason
#: when it could not (rank > 64, no triton, not CUDA, non-contiguous, or
#: `ns_dtype="bf16"`, which it subsumes rather than composes with). A name that
#: is not here is LOUD.
NS_KINDS = ("classic", "triton")
#: The `muon_colrow_3d` literals — the arms of `colrow_normalize`'s 3-D branch
#: (`optim.COLROW_3D`, which is where they execute; the list is duplicated here
#: the way `NS_KINDS` is, so the config layer does not import the trainer).
#: `frobenius` = the per-slice Frobenius normalisation the batched Muon step ran
#: until the 2026-10-17 flip (`9ee82c5`): it is NOT `NormDir::ColRow`, it is a
#: pre-existing defect found 2026-10-16 by the equivalence gate, and it makes the
#: batched update exactly `1/sqrt(d0)` of the per-param one (8x at `[64, 5461]`,
#: 45x at `[2048, 2048]`). Every number on record BEFORE the flip ran it; it is
#: kept as the A/B control. `colrow` = the 2-D definition applied per slice,
#: bit-identical to the per-param path — the fix, and the DEFAULT since that
#: A/B verdict (AGENTS §1.2: it moves the update scale for every shape group
#: with >= 2 members, and the verdict cleared the bar). A name that is not here
#: is LOUD.
MUON_COLROW_3D = ("frobenius", "colrow")
#: The `muon_state_dtype` / `ns_dtype` literals — the rungs of the two precision
#: knobs (`optim.OPT_DTYPES`, which is where they execute; the list is
#: duplicated here the way `NS_KINDS` is, so the config layer does not import
#: the trainer). `fp32` is the default and every number on record. `bf16` =
#: Muon's momentum STATE stored in bf16 with the accumulation in fp32
#: (arXiv:2509.23106's compute-high/store-low rung; stochastic rounding is the
#: NEXT rung and is not implemented) / the Newton–Schulz CHAIN run in bf16 with
#: explicit fp32 in/out casts (nanogpt's practice, row 4 of
#: `docs/research/2026-10-16-muon-orthogonalization-cost.md`). Both are opt-in
#: and approximate by design; a name that is not here is LOUD.
OPT_DTYPES = ("fp32", "bf16")
#: The `muon_variant` literals — the DIRECTION arms (`optim.MUON_VARIANTS`, which
#: is where they execute; the list is duplicated here the way `NS_KINDS` is, so
#: the config layer does not import the trainer). `muon` is the update every
#: number on record ran; `normuon` keeps the momentum and the NS polar and
#: REPLACES the ColRow tail with NorMuon's row-wise EMA second moment applied the
#: AUTHORS' repo way (`zichongli5/NorMuon` `normuon.py:48-56`: row-scale by
#: `1/√(v+ε)`, restore the pre-normalization Frobenius norm, then
#: `× max(1, m/n)^0.5`) — the repo Muon's magnitude with NorMuon's direction.
#: Composing ColRow instead of replacing it makes the arm INERT (NorMuon's row
#: scaling IS ColRow's row half; measured in `optim.normuon_repo_update_`), so
#: this arm is a REPLACE, and against our ColRow control its magnitude is not
#: preserved either. The SF paper's `eta_hat = 0.2·η·√(mn)/‖P̂‖_F` form is the
#: rejected alternative (it fixes the step norm regardless of the matrix); it is
#: what the `sf` arm uses. A name that is not here is LOUD.
MUON_VARIANTS = ("muon", "normuon")


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
    #: **`0` = DROPLESS** (2026-10-17): every assignment reaches its expert, so
    #: the ROUTING IS PER POSITION and does not depend on the window's
    #: composition. The default and the presets' `2.0` keep the capacity rule —
    #: do not change them here. The dropless mode is what makes `segment_len`
    #: legal with `moe_topk > 0` (a capacity'd window changes WHICH tokens get an
    #: expert when it is cut — `experts.py` module docstring, and
    #: `config._validate_segment` refuses the pair), and it runs the RAGGED arm:
    #: the padded bank has no cap to size itself with, so the grouped GEMM runs
    #: with the static row-block bound and the kernels' on-device empty-block
    #: exit (`moe_dropless` + `moe_triton_gemm`), and the reference loop is the
    #: COUNTED fallback (`moe_naive_dispatch` + the reason).
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
    #: What the FORWARD does to the TSCT factor masters — the compressed-model
    #: sprint's central knob (`FACTOR_QATS`, formulas in `tsct.factor_qat_project`):
    #: `none` = bf16 masters, every number on record; `ternary` = b1.58 STE with
    #: burn-spectral's soft-to-hard annealing; `binary` = LittleBit sign factors +
    #: the same annealing. Applies to EVERY TSCT pair the model owns: the loop
    #: block's projections and the MoE experts' dual-latent factors alike (the
    #: same `factor_qat_project` on both paths — one place, one test). `s` stays
    #: fp32 in all arms: it IS the scale the annealing needs.
    factor_qat: str = "none"
    #: Annealing strength, 0..=1: 0 = masters untouched (bit-identical `none`),
    #: 1 = pure quantized forward with STE. A schedule ramping this 0→1 is the
    #: burn-measured better-than-hard-STE recipe; until the schedule lands, a
    #: constant is the honest form (a 2k-step A/B can hold it fixed).
    factor_qat_alpha: float = 1.0
    #: The `ternary_s3t` Bernoulli mask's seed (`tsct.set_factor_qat_seed`,
    #: called once per model build). The mask is a process-local generator's
    #: stream, not a pure function of `(seed, step)` — the projector has no step
    #: — so this makes a run reproducible and a RESUME not (it restarts the
    #: stream): `tsct.factor_qat_project`'s docstring owns the statement and
    #: `train/loop.py` prints it at start. Inert (LOUD) without that arm.
    factor_qat_seed: int = 0
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
    #: BitNet a4.8 ACTIVATION quantization (`ACT_QUANTS`, formulas in
    #: `model/act_quant.py`, transcribed from `crates/dormouse-core/src/act_quant.rs`):
    #: the activations that feed the factor matmuls are fake-quantized with an STE
    #: while the fp32 masters — and the activation graph — stay fp32. `none` is
    #: every number on record. Wired at three seams: `TSCTLinear.forward` (the
    #: readout and `lm_head`; any TSCT factor projection ON the attention path
    #: takes the a4.8 upgrade via `attn_path=True`), `DualLatentGLU.forward` (the
    #: experts) and the expert bank in `SparseMoE`. The ROUTER input and the KDA
    #: internals are out of scope by design.
    #: Two of `ACT_QUANTS`' formats are imported rather than a4.8: `mxfp8`
    #: (`mxfp.fake_quant_mx`) and `v2_int4`/`v2_int8` (`hadamard.v2_quantize`,
    #: which needs a 2^k `d_model` — refused at load, not silently padded).
    act_quant: str = "none"
    #: Activation columns per scale: 0 = one scale per token (row), else one per
    #: group of `act_group` columns (the a4.8 FFN recipe). Must divide `d_model`,
    #: because the quantizer reshapes the activations to `[..., d/group, group]`
    #: and a non-divisor fails on the first forward, far from the flag (`validate`
    #: refuses it, mirroring `config/validation.rs:220-231`).
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
    #: Newton-Schulz iterations for the Muon polar retraction. 5 = Muon's own
    #: default (the Jordan triple's original setting). 3 = the arm the
    #: 2026-10-16 cost note prices at ~-130 ms/step at b1: the step-count
    #: quality literature (arXiv:2606.00371 §3.2, 3 seeds, exact-SVD control)
    #: is NON-MONOTONE with a <0.003 BPB spread -> a tie prediction, and this
    #: port's own 3-iteration NS sits 1.17e-15 from the manifold at a 10:1
    #: spectral spread (docs/reviews/..., 2026-10-15). A change of this
    #: default goes through the standing 3-seed A/B, not a sentence.
    ns_steps: int = 5
    #: torch.compile(dynamic=True) around the model's forward — the one large
    #: speed lever left (the spike measured 273.5 vs 357.3 ms/step at the b1
    #: shape, 1.20x; dynamic=True because eval/resume lengths vary). The first
    #: warmup costs minutes and lands in ~/.cache/torch, so it is paid once
    #: per machine, not per run. OFF by default until a b1-shape run confirms
    #: the win end to end; the wrapper goes on AFTER resume (an
    #: OptimizedModule's state_dict carries a `_orig_mod.` prefix).
    torch_compile: bool = False
    #: The ANNEALED factor_qat ramp (burn-spectral's measured soft-to-hard
    #: recipe, `ste_ternary_annealed`: "converges better than hard STE from
    #: step one"): the trainer sets `model.set_factor_qat_alpha(min(1, step/N))`
    #: BEFORE each forward. `0` (default) = the module's constant
    #: `factor_qat_alpha` — the hard-STE arm the compression tournament
    #: measured LOSING to bf16 (+0.05/+0.09 BPB at 9.2M, 2026-10-16); this
    #: ramp is the one untested variant of that recipe.
    factor_qat_anneal_steps: int = 0
    #: The Newton–Schulz arm the Muon group runs (`NS_KINDS` above,
    #: `optim.zeropower_via_ns(kind=...)`). `classic` = the per-iteration
    #: `X @ X.T` recurrence: the default, byte-identical to this port's history.
    #: Forwarded by the trainer: `train/loop.py` builds
    #: `MixOptimizer(..., ns_kind=cfg.ns_kind)` (the same call site now also
    #: forwards `muon_colrow_3d`), so this field is what the run executes — not
    #: only what the gates and `scripts/bench_ns_triton.py` build directly.
    #: `triton` (the fused chain, `optim/ns_kernels.py`) rides the same field: it
    #: is COUNTED, and its `ns_triton` / `ns_triton_fallback` counters reach the
    #: log line through `opt.counters` (`train/loop.py`).
    ns_kind: str = "classic"

    #: The 3-D `colrow_normalize` arm the batched Muon step runs
    #: (`MUON_COLROW_3D` above, `optim.COLROW_3D` where it executes).
    #: **`colrow` (the default since the flip landed — `9ee82c5`, 2026-10-17)
    #: = the 2-D definition applied per slice** — bit-identical to the per-param
    #: path. The A/B verdict
    #: (6 runs x 2k steps, small, 81 920 B window): control 3.0542 vs fix
    #: 3.0113 = **-0.0428 BPB, 8.7x the control's seed spread, zero seed
    #: overlap** — the largest measured A/B win in the port's history, so the
    #: default flipped (AGENTS 1.2). Carry-forward caveats: the arm's curve
    #: CROSSES (worse at step 1000, better at 2000), its own spread is 3.7x
    #: the control's, and it is one preset/one budget — re-confirm on `medium`
    #: and `b1`. `frobenius` = the per-slice Frobenius normalisation the
    #: batched step ran before that (dims `(1, 2)` twice, so the update was
    #: exactly `1/sqrt(d0)` of the definition — 8x weak at `[64, 5461]`,
    #: 45x at `[2048, 2048]`); kept as the historical control arm. The 2-D
    #: path (and a shape group with one member) is the same in both arms.
    muon_colrow_3d: str = "colrow"

    #: Muon's momentum STATE dtype (`OPT_DTYPES`, `optim.Muon(state_dtype=...)`).
    #: `fp32` (the default) is every number on record. `bf16` halves the
    #: optimizer state — at b1 2.46 GiB -> 1.23 GiB, measured on the instantiated
    #: model's Muon groups, of which `[64, 5461] x 1024` (the expert bank
    #: factors) is 1.333 GiB fp32 alone — while the ACCUMULATION stays fp32 and
    #: only the store rounds (arXiv:2509.23106; the rung's provenance and the
    #: **not implemented** stochastic-rounding follow-up are in `optim`'s module
    #: docs, and the gate in `tests/test_muon_fast.py` holds the update to 1e-2
    #: relative against the fp32 arm). Resume-safe by construction: a snapshot
    #: that lacks the key fills this default, and a flip on a resume is refused
    #: by the snapshot drift check (`diff_keys`), so a live run's semantics and
    #: its resume do not move.
    muon_state_dtype: str = "fp32"
    #: The Newton–Schulz CHAIN's dtype (`OPT_DTYPES`): `fp32` (the default) is
    #: every number on record; `bf16` runs the polynomial — BOTH arms, `classic`
    #: and `gram` — in bf16 with explicit fp32 in/out casts (nanogpt's practice;
    #: row 4 of `docs/research/2026-10-16-muon-orthogonalization-cost.md`,
    #: predicted −100…−200 ms/step at b1, NOT measured on this card yet). The
    #: orthogonality bound it must keep is gated in `tests/test_muon_fast.py`.
    ns_dtype: str = "fp32"

    #: The Muon group's DIRECTION arm (`MUON_VARIANTS`, `optim.Muon(variant=...)`).
    #: `muon` (the default) is the update every number on record ran: momentum →
    #: optional ColRow → NS polar, step `-lr·max(1, m/n)^0.5`. `normuon` keeps the
    #: momentum and the NS polar and **replaces the ColRow tail** with the
    #: AUTHORS' repo form (`zichongli5/NorMuon` `normuon.py:48-56`, transcribed in
    #: `optim.normuon_repo_update_`): the row second moment
    #: `v_t = β2·v_{t-1} + (1−β2)·mean_cols(O_t ⊙ O_t)` of the post-polar update, a
    #: row scale by `1/√(v_t + 1e-10)`, a rescale by `‖O‖_F/‖Ô‖_F` that restores
    #: the pre-normalization Frobenius norm, then the repo's own
    #: `max(1, m/n)^0.5`.
    #:
    #: **Two measured facts a reader must know before reading an A/B on it**
    #: (`tests/test_muon_variants.py` pins both): (1) COMPOSING ColRow instead of
    #: replacing it makes the arm inert — NorMuon's row scaling IS ColRow's row
    #: half, so the statistic is constant across rows and the arm reproduces the
    #: control to `cos = 1.000000`; (2) the repo form's magnitude-preserving
    #: invariant holds against the REPO's Muon (no ColRow) and NOT against our
    #: ColRow control, whose `‖ColRow(O)‖_F = √m` exceeds the polar's
    #: `≈0.7·√min(m,n)` — measured `‖ΔW_arm‖/‖ΔW_control‖` = 0.7066 at
    #: `[64, 5461]`, 0.8701 at `[64, 64]`, 0.1293 at `[2048, 64]`. So this arm is
    #: a REPLACE (NorMuon's row half instead of ColRow's), not a direction-only
    #: knob, and the tall-factor magnitude gap is reported with it.
    #: Memory: `+m` fp32 scalars per matrix (negligible against `m·n`).
    #: `scripts/ab_muon_variants.py` is the driver. A name that is not in
    #: `MUON_VARIANTS` is LOUD at load and at the constructor.
    muon_variant: str = "muon"

    #: The SCHEDULE-FREE wrapper (`optim.Muon(sf=...)`), `false` by default.
    #: SF-NorMuon (arXiv:2605.23061, JPMorgan) around the `muon_variant`
    #: direction: the params hold `Y_t = (1−β1)Z_t + β1 X_t` during the
    #: forward/backward, `X_t` is the returned/eval iterate, `Z_t` the fast one,
    #: `s_t = Σ η²` drives the averaging coefficient `c = η_t²/s_t` (β1 = 0.9,
    #: `optim.SF_BETA1`). The
    #: trainer's seams: `sf_switch("train")` after the optimizer is built and
    #: after every eval/checkpoint, `sf_switch("eval")` before them
    #: (`train/loop.py`), and a NaN-skipped step advances none of it.
    #:
    #: **Two deliberate deviations from the paper, both named in `optim.py`:**
    #: (1) the Muon group's LR is CONSTANT after warmup (no WSD decay) while the
    #: AdamW half keeps the WSD schedule — a HYBRID SF-Muon + WSD-AdamW, not the
    #: paper's full-model SF (full SF-AdamW is the named follow-up); (2) our
    #: ColRow normalization stays composed (the paper's `P̂` is NorMuon's row
    #: statistic alone). **Two numeric deviations from Appendix F, both checked
    #: and named rather than silently adopted:** the reference's own momentum
    #: default is `momentum=0.8` while this port reuses the Muon group's existing
    #: `momentum` knob (`0.95`) so there is ONE momentum, and the reference warms
    #: up INSIDE the optimizer (`warmup_steps=2000` steps, `lr*(k+1)/warmup`)
    #: while this port keeps the trainer's token-based WSD warmup. Refused loudly
    #: with `opt="adamw"` (no Muon group to wrap). **`engram_ram` is OUTSIDE the
    #: wrapper** and stays on the WSD schedule: on that path the n-gram tables
    #: are not parameters at all (`Engram.to_host` deletes them before the
    #: (`Engram.to_host` deletes them before the optimizer exists) and they are
    #: updated by their own Nesterov/Sinkhorn step, so a b1-style run with
    #: `muon_sf` is SF on the Muon group + WSD on AdamW + WSD on the host
    #: tables — a partial SF, said here rather than discovered from a curve.
    #: A 2k-step A/B is predicted to TIE — the published gains are stated
    #: at ≥1000 tokens/param, far past a 2 000-step arm, and a tie is the
    #: designed outcome (`docs/research/2026-10-16-sf-muon.md` §4), so the
    #: verdict must read the `X`↔`Y` gap and the LR-schedule-free claim, not the
    #: endpoint alone. `scripts/ab_muon_variants.py` is the driver.
    muon_sf: bool = False
    #: CUDA-graph capture of the optimizer STAGE (`train/graph.py`): the grad
    #: clip + the mixed step are recorded once after `GRAPH_WARMUP_STEPS` eager
    #: steps and replayed with ONE launch per step afterwards. This is the
    #: direct attack on the measured bottleneck — the optimizer is 44-53% of a
    #: b1 step and the step is HOST-DISPATCH-BOUND (the card is idle 87% of a
    #: warm step; 46 341 `aten::mm` launches per step at b8·s512) — and it is
    #: the P0 idea of `docs/research/2026-10-17-olmo-core-ideas.md` §2.1,
    #: reimplemented from `microsoft/dion` `dion/cuda_graph.py`.
    #:
    #: **`false` by default, and the default path is byte-identical to its
    #: history** (gated: sha256 of a fixed-seed 10-step run, `tests/test_opt_graph.py`).
    #: The arm changes three things a reader must know before reading a number
    #: off it, all named in `graph.py`: the LR becomes a DEVICE tensor the
    #: trainer fills in place (a host float would be baked at capture and
    #: silently freeze the WSD schedule), `zero_grad` runs
    #: `set_to_none=False` (a fresh `.grad` every step breaks the pinned
    #: buffers), and the AdamW half rides torch's `capturable=True` — which is
    #: what makes a device-tensor LR readable there, and which MOVES the AdamW
    #: arithmetic (the bias correction is folded in on device), so the arm is
    #: bit-identical to its own eager path and not to the pre-arm path.
    #:
    #: Refused combinations, all LOUD at capture and COUNTED (`opt_graph_fallback`
    #: + the reason) rather than fatal: `muon_sf` (its `_sf_ckp` recursion is host
    #: arithmetic), `muon_variant="normuon"` (`_foreach_add_`'s `alpha` cannot take
    #: a device tensor without a host sync), `optim_8bit` (bitsandbytes' AdamW8bit
    #: has no capturable path), any `requires_grad` param with `.grad is None`
    #: (the participating set freezes at capture), and a non-CUDA device. The
    #: trainer also refuses `opt_graph` + `torch_compile` (a recompile under an
    #: active capture aborts it). A resume with a different value is refused by
    #: the snapshot drift check — it changes what the optimizer computes.
    opt_graph: bool = False
    seed: int = 1
    #: WSD warmup, in TOKENS (not steps): the schedule is a function of how
    #: much data the run has seen, so a batch-size change does not change it.
    warmup_tokens: int = 0
    #: Fraction of the token budget spent on the WSD decay.
    decay_frac: float = 0.1
    #: JSONL telemetry sink: one row per event (log line, eval, checkpoint),
    #: consumed by `dormouse plot`. Empty = off. A PROGRESS key: it is where
    #: measurements go, never what the run computes, so re-pointing it on a
    #: resume is legal and expected for a moved run (audit MINOR 23, 2026-10-17).
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
    #: SEGMENTED TRAINING: cut the training window into segments of this many
    #: bytes and carry the KDA state across the cuts (`loopblock._forward_segmented`,
    #: the StateFlow pattern — arXiv:2608.06838 — instead of truncated BPTT), so
    #: a sequence longer than one forward fits in VRAM can be trained. It is a
    #: MEMORY decision and not a model change: the logits agree with the
    #: one-shot forward to fp32 round-off (`tests/test_longctx.py`), the returned
    #: logits keep the `[b*seq_len, vocab]` shape, and the loss is the same
    #: `(Σ CE_i + CE_final)/(T+1)`, so nothing downstream changes.
    #:
    #: **`0` = OFF, and that is the default**: the field must be set explicitly
    #: to enter the branch. The state carried is PER ITERATION SLOT (the loop
    #: calls the attention arm once per iteration), the gradient rides it (the
    #: fused op takes `dht` and returns `dh0` — `fla/ops/kda/chunk.py:127`, `:172`),
    #: and each segment runs inside `checkpoint(use_reentrant=False)` so its
    #: activations are recomputed in the backward rather than stored.
    #:
    #: Refused loudly (see `_validate_segment`): `>= seq_len` (the window is one
    #: segment — pointless), `< 0` (0 is the off value), a non-divisor of
    #: `seq_len` (a short last segment would weight its bytes more than the
    #: others, so the loss would stop being the one-shot loss), `use_byteflow`
    #: (the loop runs over patch latents, not bytes), `use_msa` (the block-sparse
    #: arm selects blocks INSIDE a window, so cutting the window would change the
    #: attention pattern itself), `rand_depth` (two unvalidated axes in one
    #: run: the depth arm is un-A/B'd, and attributing a result to either would
    #: be a guess), and `moe_topk > 0` AT `moe_capacity > 0` (the capacity-padded
    #: dispatch DROPS the rows past its per-expert capacity, so which tokens get
    #: an expert is a function of the other tokens in the same forward — measured
    #: 0.39 on the tiny readout at `moe_capacity=1.0`, 2026-10-17; `moe_capacity=0`
    #: is the DROPLESS mode and IS allowed, `small`'s dense blend is exact). The
    #: last three are the finding this feature came with: each one would have been
    #: a SILENT model change wearing a memory flag.
    segment_len: int = 0
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
    if model.moe_capacity < 0.0:
        loud(
            f"moe_capacity={model.moe_capacity} is negative: it is a multiple of the "
            f"average expert load (`ceil(capacity·tokens·k/E)` rows per expert), and 0 "
            f"is the DROPLESS mode (every assignment reaches its expert, the padded "
            f"bank refuses, the ragged arm runs COUNTED). Escape: 0, or a positive "
            f"capacity (2.0 is the presets')."
        )
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
    if model.factor_qat not in FACTOR_QATS:
        loud(
            f"factor_qat={model.factor_qat!r} is not an arm. Known: "
            f"{list(FACTOR_QATS)} — `none` bf16 masters, `ternary` b1.58 STE "
            f"(arXiv 2402.17764), `binary` LittleBit signs (arXiv 2506.13771), "
            f"`mxfp4` the OCP MX e2m1 codec (spec §5.2, mxfp.py), `ternary_s3t` "
            f"S3T stochastic rounding (arXiv 2412.04787). Escape: name one of "
            f"those, or land the arm and its gold test (tests/test_factor_qat.py) "
            f"BEFORE the config can lie about it."
        )
    if not 0.0 <= model.factor_qat_alpha <= 1.0:
        loud(
            f"factor_qat_alpha={model.factor_qat_alpha} is outside 0..=1: it "
            f"blends master (0) with pure quantized forward (1). Escape: clamp "
            f"your schedule to that range."
        )
    if model.factor_qat == "none" and model.factor_qat_alpha != 1.0:
        loud(
            f"factor_qat_alpha={model.factor_qat_alpha} with factor_qat='none' "
            f"is inert — the quantizer is off, so the blend has nothing to "
            f"blend toward. Escape: set factor_qat, or leave alpha at 1.0."
        )
    if model.factor_qat_seed != 0 and model.factor_qat != "ternary_s3t":
        loud(
            f"factor_qat_seed={model.factor_qat_seed} with "
            f"factor_qat={model.factor_qat!r} is inert — only `ternary_s3t` draws "
            f"from a generator (the other rungs are deterministic projections). "
            f"Escape: factor_qat='ternary_s3t', or leave the seed at 0."
        )
    if model.act_quant not in ACT_QUANTS:
        loud(
            f"act_quant={model.act_quant!r} is not an activation format. Known: "
            f"{list(ACT_QUANTS)} — `none` off, `int8`/`int4` symmetric integer "
            f"levels (arXiv 2504.12285 §2.2; 127/7, formulas in "
            f"model/act_quant.py), `fp4` e2m1 (the OCP MX FP4 grid), `mxfp8` the "
            f"OCP MX e4m3 codec (spec §5.2), `v2_int4`/`v2_int8` BitNet v2's "
            f"Hadamard-rotated integer arms (arXiv 2504.18415). Escape: name one "
            f"of those, or land the format and its gold test "
            f"(tests/test_act_quant.py) BEFORE the config can lie about it."
        )
    if model.act_group < 0:
        loud(
            f"act_group={model.act_group} is negative: it is a block width in "
            f"activation columns (0 = one scale per token). Escape: 0, or a "
            f"positive divisor of d_model={model.d_model}."
        )
    if model.act_group != 0 and model.act_quant in ("mxfp8", "v2_int4", "v2_int8"):
        loud(
            f"act_group={model.act_group} with act_quant={model.act_quant!r} is "
            f"inert — that format carries its own scale grid (mxfp8: 32-wide e8m0 "
            f"blocks; v2_*: per-token absmax after the FWHT), so the group would "
            f"be silently ignored. Escape: act_group=0, or int8/int4/fp4, whose "
            f"scale IS the group's block max."
        )
    if model.act_quant in ("v2_int4", "v2_int8"):
        d = model.d_model
        if d < 1 or (d & (d - 1)) != 0:
            loud(
                f"act_quant={model.act_quant} rotates the activation with the "
                f"FWHT (BitNet v2, arXiv 2504.18415), which needs a LAST dim of "
                f"2^k — d_model={d} is not one. Zero-padding to 2^k is NOT the "
                f"MX tail pad: it changes the rotation's basis, so the port "
                f"refuses instead of quantizing a row it did not rotate. The same "
                f"rule is enforced per projection at the first forward "
                f"(hadamard._check names the width); this is the early refusal "
                f"for the width every TSCT projection in the loop block reads. "
                f"Escape: a power-of-two d_model, or another activation format."
            )
    if model.act_quant != "none" and model.act_group != 0 and model.d_model % model.act_group != 0:
        loud(
            f"act_group={model.act_group} must divide d_model={model.d_model} (or "
            f"be 0 for one scale per token): the quantizer groups the activations "
            f"in blocks of act_group and reshapes to [..., d_model/act_group, "
            f"act_group], so a non-divisor fails the reshape on the first "
            f"forward, a long way from the flag that set it "
            f"(config/validation.rs:220-231). Escape: pick a divisor of "
            f"d_model, or 0."
        )
    if model.act_group != 0 and model.act_quant == "none":
        loud(
            f"act_group={model.act_group} with act_quant='none' is inert — the "
            f"quantizer is off, so there is no block to group. Escape: set "
            f"act_quant, or leave act_group at 0."
        )
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

    # The dspark arm's OWN coherence (audit MINOR 12, 2026-10-17): these used to
    # live inside `_validate_byteflow`, which only runs under `use_byteflow`, so
    # the default path accepted `dspark_weight=-1.0` / `dspark_k=0` /
    # `dspark_stride=0` — a field that lies about being refused (§1.7). The
    # byteflow-only interaction (`dspark_weight>0` with the channel's forward)
    # stays in `_validate_byteflow`; the two-head rule is the aux_fb guard above.
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


def _refuse_ladder_rung(rung: str) -> str:
    """The quant ladder: `fp8`/`nvfp4`/`ternary` are Storage rungs (they
    compress the factor bank and the checkpoint, the fp32 masters stay the
    optimizer's object — see `tsct.py` module doc); everything else refuses.

    GROUPED, not a fallback. What remains refused (`fp16/lbit`) is a
    TRAINING-time change — LittleBit's gates G-A..G-D
    (`docs/reviews/littlebit-design-2026-10-15.md`) must close first.
    """
    storage = ("fp8", "nvfp4", "ternary")
    if rung in storage:
        return rung
    known = {"fp16", "fp4", "int4", "int8", "lbit"}
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


def _validate_segment(model: ModelConfig, tcfg: TrainConfig) -> None:
    """The segmented-training flag against the window and the arms it cannot ride.

    All of these are refusals rather than fallbacks, for the reason the whole
    config layer exists: a `segment_len` that quietly did nothing (or that cut
    the window in a way that changed the objective or the model) would show the
    flag on while the run reported it, which is the class of defect ADR-0019 is
    about. `0` is the OFF value and the only value any number on record ran.
    """
    n = tcfg.segment_len
    if n < 0:
        loud(
            f"segment_len={n} is negative: it is a count of bytes per segment, and "
            f"0 is the OFF value (the one-shot forward). Escape: 0, or a positive "
            f"divisor of seq_len={tcfg.seq_len}."
        )
    if n == 0:
        return
    if n >= tcfg.seq_len:
        loud(
            f"segment_len={n} >= seq_len={tcfg.seq_len}: the window is one segment, so "
            f"the segmented path would run the same forward through a checkpoint "
            f"wrapper and buy nothing while the log line reported it on. Escape: a "
            f"segment_len < seq_len (the recipe's 1024 at seq 8192 is 8 segments), or "
            f"segment_len=0 for the one-shot path."
        )
    if tcfg.seq_len % n != 0:
        loud(
            f"segment_len={n} does not divide seq_len={tcfg.seq_len}: a shorter last "
            f"segment is REFUSED rather than allowed, because the per-iteration CE is "
            f"a MEAN per segment — an uneven cut would weight the short segment's "
            f"bytes more, so the loss would stop being the one-shot loss and the "
            f"segmented run would be a different objective wearing a memory flag. "
            f"Escape: a divisor of {tcfg.seq_len}."
        )
    if model.use_byteflow:
        loud(
            f"segment_len={n} with use_byteflow=true: the byteflow shapes run the loop "
            f"over K patch latents (or not at all), not over the byte window, so there "
            f"is no byte sequence to cut into segments. Escape: use_byteflow=false, or "
            f"segment_len=0."
        )
    if model.use_msa:
        loud(
            f"segment_len={n} with use_msa=true: the block-sparse arm selects its "
            f"blocks INSIDE one window, so cutting the window changes the attention "
            f"pattern itself (a block spanning a cut can no longer be selected) — that "
            f"is a model change, and this flag is allowed to change only the memory. "
            f"Escape: use_msa=false, or segment_len=0."
        )
    if tcfg.rand_depth:
        loud(
            f"segment_len={n} with rand_depth=true: the adaptive-depth arm is "
            f"un-A/B'd, and segmenting adds a second axis (the carried state at depth "
            f"T vs the one-shot forward at depth T) — a result from a run carrying both "
            f"would be attributable to neither. Escape: rand_depth=false, or "
            f"segment_len=0."
        )
    if model.moe_topk > 0 and model.moe_capacity > 0:
        loud(
            f"segment_len={n} with moe_topk={model.moe_topk} at moe_capacity="
            f"{model.moe_capacity}: the capacity-padded top-k DROPS the rows past "
            f"`ceil(moe_capacity · tokens · k / E)` per expert, so WHICH tokens get "
            f"an expert is a function of how many other tokens share their forward — "
            f"cutting the window changes that set and the segmented forward stops "
            f"being the one-shot forward (measured on the tiny shape at "
            f"moe_capacity=1.0: 0.39 on the readout, 2026-10-17; at 2.0 no drop "
            f"happened on that shape, which is not a promise a collapsing router "
            f"keeps). Escape: moe_capacity=0 — the DROPLESS mode, where every "
            f"assignment reaches its expert and the routing is per position (it runs "
            f"the RAGGED arm, COUNTED as `moe_dropless` + `moe_triton_gemm`: the "
            f"padded bank has no cap, so the grouped GEMM takes the static row-block "
            f"bound and exits on-device on an empty block — no sync, ADR-0018 rule 2) "
            f"— or moe_topk=0 (the dense blend `small` ships), or segment_len=0."
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
    if tcfg.ns_kind not in NS_KINDS:
        loud(
            f"ns_kind={tcfg.ns_kind!r} is not an arm. Known: {list(NS_KINDS)} — "
            f"`classic` forms `X @ X.T` at every Newton–Schulz iteration (the "
            f"default), `triton` is the same chain fused into ONE kernel launch "
            f"(`ns_kernels.py`, COUNTED fallback to `classic`). Escape: name one "
            f"of those, or land the arm in `optim.NS_KINDS` and its gate BEFORE "
            f"the config can lie about it."
        )
    if tcfg.muon_colrow_3d not in MUON_COLROW_3D:
        loud(
            f"muon_colrow_3d={tcfg.muon_colrow_3d!r} is not an arm. Known: "
            f"{list(MUON_COLROW_3D)} — `frobenius` is the per-slice Frobenius "
            f"normalisation the batched Muon step has always run (it is NOT "
            f"ColRow: the batched update comes out `1/sqrt(d0)` of the per-param "
            f"definition — a defect, kept as the A/B control), `colrow` is the "
            f"2-D definition applied per slice (the fix, bit-identical to the "
            f"per-param path). Escape: name one of those, or land the arm in "
            f"`optim.COLROW_3D` and its parity gate BEFORE the config can lie "
            f"about it."
        )
    for name in ("muon_state_dtype", "ns_dtype"):
        rung = getattr(tcfg, name)
        if rung not in OPT_DTYPES:
            loud(
                f"{name}={rung!r} is not a rung. Known: {list(OPT_DTYPES)} — "
                f"`fp32` is the default and every number on record, `bf16` is the "
                f"low-precision rung (Muon state: bf16 storage with fp32 "
                f"accumulation, arXiv:2509.23106; NS chain: bf16 polynomial with "
                f"explicit fp32 in/out casts, nanogpt's practice). Escape: name one "
                f"of those, or land the rung in `optim.OPT_DTYPES` and its gate "
                f"BEFORE the config can lie about it."
            )
    if tcfg.muon_variant not in MUON_VARIANTS:
        loud(
            f"muon_variant={tcfg.muon_variant!r} is not an arm. Known: "
            f"{list(MUON_VARIANTS)} — `muon` is the update every number on record "
            f"ran (momentum -> optional ColRow -> NS polar, step "
            f"`-lr*max(1, m/n)^0.5`), `normuon` keeps that pipeline and replaces the "
            f"tail with NorMuon's row-wise EMA second moment plus the "
            f"`eta_hat = 0.2*eta*sqrt(m*n)/|Phat|_F` step scale (arXiv:2510.05491 "
            f"Alg 1 lines 5-11, transcribed by arXiv:2605.23061v1 App F). Escape: "
            f"name one of those, or land the arm in `optim.MUON_VARIANTS` and its "
            f"gate BEFORE the config can lie about it."
        )
    if tcfg.muon_sf and tcfg.opt != "mix":
        loud(
            f"muon_sf=true with opt={tcfg.opt!r}: the schedule-free wrapper rides the "
            f"Muon group, and `opt={tcfg.opt!r}` routes every parameter to AdamW — "
            f"there is no Muon group to wrap, so the flag would be inert while the run "
            f"reported it on. Escape: opt='mix', or muon_sf=false. (Full schedule-free "
            f"AdamW is the named follow-up, not this arm.)"
        )
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
    # The segmented-training flag is a MODEL-shape decision too: it meets the
    # window AND the arms, so its refusals live here, where both halves are
    # resolved — the same shape as the `engram_ram` block above.
    _validate_segment(model, tcfg)
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
