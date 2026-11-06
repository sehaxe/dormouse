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

`RunConfig` is the snapshot: the resolved model + train config plus provenance,
written next to a checkpoint as `<ckpt_name>.config.toml`. A resume refuses a
snapshot whose non-progress keys differ (ADR-0021) — extending a run is a legal
resume, changing what the model computes is not.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .instrument import LoudError, loud

if TYPE_CHECKING:  # the RL config lives in the post lane; importing it here
    from .post.rl import RLConfig  # would be a cycle at module load

#: `port/configs/`, reached from this file (`port/src/dormouse/config.py`).
PRESET_DIR = Path(__file__).resolve().parents[2] / "configs"

#: Train keys a resume is allowed to change: they decide WHEN we look, never
#: what the model computes. The burn list (cfg.rs `PROGRESS_KEYS`) plus
#: `json_log` — the telemetry sink is WHERE the measurement goes and it reaches
#: no forward (audit MINOR 23, 2026-10-17: the field's own docstring promised
#: this exemption while `diff_keys` refused the flip).
#: Keys a resume may change. `seq_len` joined on 2026-11-04: the window is not a
#: parameter, and the KDA/CSA arms carry no positional table, so widening it is a
#: curriculum step in the same class as extending `steps`. `max_seq_len` is the
#: buffer bound beside it and may only GROW, and only when no positional table
#: exists (byteflow/full attention size one) — see `_diff_keys`.
PROGRESS_KEYS = (
    "seq_len",
    # `moe_kernel` is an IMPLEMENTATION key, not an experiment: the three arms
    # are proven equal to 1e-6 (tests/test_moe_batched.py gates a/b), the way two
    # cuBLAS thread counts are. Switching it on a resume is a kernel change on
    # the same math — and on 2026-11-04 it was worth 8.6x on the step.
    "moe_kernel",
    "steps",
    "log_every",
    "ckpt_every",
    "eval",
    "eval_every",
    "json_log",
    # The WSD tail: both keys move WHERE the remaining steps land on the LR
    # curve and neither is read by a forward, which is the same class as
    # `steps` (scaling `steps` alone already re-plans the decay window, so
    # allowing these two is not a new freedom — it is the same freedom spelled
    # directly, and it is what makes "anneal now, longer, to almost zero" a
    # resume instead of a re-launch). The scheduled LR is logged every
    # `log_every`, so a resume that changes them is visible in the log line.
    "decay_frac",
    "min_lr_frac",
    # `ema_decay` selects WHICH copy of the same trajectory an eval reads: the
    # checkpoint keeps the raw iterate under `model` (what resume continues)
    # and the shadow under `model_ema` (what eval/save serve), so changing the
    # decay on a resume changes a view, never what was trained. It is a
    # progress key for that reason and is documented on `TrainConfig`.
    "ema_decay",
    # `ns_steps` is the Newton-Schulz iteration count, i.e. WHERE the remaining
    # steps sit on the same polynomial approximation. The gradient it produces
    # is a function of the momentum, not of the step's history, so changing it
    # on a resume is an A/B on the remaining budget rather than a new run. The
    # optimizer's param groups are restored from the CHECKPOINT by torch's
    # loader, so `train/loop.py` re-applies this key after `load_state_dict`
    # (`MixOptimizer.set_ns_steps`) — without that line the progress key would
    # be silently inert on a resume.
    "ns_steps",
    # `muon_chunk_elems` changes how many slices share ONE lockstep pass, not
    # the per-slice arithmetic: the greedy split is over whole matrices and
    # the NS chain is per slice, so a different cap is bit-identical
    # (`tests/test_opt_region.py`). It is a kernel/memory selection, the same
    # class as `moe_kernel`, and a preset that raises it must not refuse the
    # resume of a run that used the schema value.
    "muon_chunk_elems",
)

#: TRAIN defaults that ride a PRESET NAME (a preset file is model-only: the
#: schema is the train default, not a second TOML file). Only the geometries
#: where the arm was measured live here; `load_run` applies this AFTER the
#: `TrainConfig()` defaults and BEFORE explicit `train_overrides`, so
#: `--set key=...` still wins and a caller that passes a `TrainConfig` is not
#: touched. The snapshot records the resolved value, so a checkpoint pins it.
#:
#: m400/medium: `muon_state_dtype=bf16` is the 2026-11-05 optimizer-region arm
#: (the 100-step CPU gate and the launch census are in
#: `docs/reviews/2026-11-05-opt-region.md`); `muon_chunk_elems=1<<26` is the
#: same geometry's lockstep cap — it halves the chunk count (18 -> 14 at m400)
#: and keeps the peak intermediate at half of the b1-cap OOM size. b1 and the
#: smaller presets keep the historical schema values.
PRESET_TRAIN_DEFAULTS: dict[str, dict[str, object]] = {
    "m400": {"muon_state_dtype": "bf16", "muon_chunk_elems": 1 << 26},
    "medium": {"muon_state_dtype": "bf16", "muon_chunk_elems": 1 << 26},
}

#: The commit that removed the stubs below (`port(cleanup): delete the six dead
#: config stubs and the `_unsupported` machinery`, 2026-10-17).
DEAD_SNAPSHOT_COMMIT = "0e7d25a"

#: Snapshot keys that were VALID when their checkpoint was written and are not
#: fields of this branch's schema: `key -> (commit, default)`, where `commit` is
#: the commit that removed the key from this branch's schema, or — for a LANE
#: key — the commit that carries it on the branch this checkout does not have.
#:
#: This is **NOT a relaxation of the strict schema**: `_known` still refuses
#: every unknown key LOUDLY, the preset path never consults this list, and a key
#: that IS a field of the class is left alone even if it is listed here (a dead
#: entry can never shadow a live field). It is a DATED COMPAT LIST — a snapshot
#: is a record of what a run *was*, and every `checkpoints/*.config.toml` on disk
#: predates these removals, so refusing them would make the snapshots unreadable
#: (resume, eval, decode, serve, `--init-from`) for keys that reach no code
#: here. The same idea as `tools/check_doc_refs.py`'s `KNOWN_DEAD`: a dated
#: record is not a failure.
#:
#: The drop is VALUE-CHECKED, and that is the mechanism. A key whose snapshot
#: value equals its recorded `default` is dropped COUNTED (one printed line per
#: table, naming each key and its commit). A NON-DEFAULT value is a LOUD refusal
#: naming the key, its value, the default and the escape: the field is gone from
#: this code, so resuming would silently run a different forward or optimizer
#: than the checkpoint was trained with — a different model wearing the same
#: weights. `sw1` is the standing case: a 300 000-step run trained at
#: `act_quant = "int4"`, so retiring that field must refuse its resume, never
#: make the run quietly become `none`.
DEAD_SNAPSHOT_KEYS = {
    # removed by `port(cleanup): delete the six dead config stubs` (0e7d25a):
    "use_gr": (DEAD_SNAPSHOT_COMMIT, False),
    # "use_mhc": removed from this list 2026-10-21 — it is a real field again,
    # and a listed key that is a field is now skipped, not dropped.
    "use_mor": (DEAD_SNAPSHOT_COMMIT, False),
    "use_situ": (DEAD_SNAPSHOT_COMMIT, False),
    # "jepa_weight": removed from this list 2026-10-21 — it is a real field
    # again (JEPA as ONE mechanism, `model/jepa.py`); a listed key that is a
    # field is skipped, not dropped.
    "stress": (DEAD_SNAPSHOT_COMMIT, False),
    # removed by `delete the aux_fb future-byte head — a TIE` (39d0485):
    "aux_fb_weight": ("39d0485", 0.0),
    # LANE keys: live on branches this checkout does not carry, under these SAME
    # names — so there is nothing to map them onto here. No field on this branch
    # means the same thing (`bf16` is a storage flag, not `dtype`'s per-device
    # policy; no `train` field means "one entry per rank"). Both snapshots that
    # carry them hold the lane's DEFAULT, so the value check drops them today; a
    # snapshot from a lane run that MOVED one refuses (below) instead of
    # resuming as a single-process run wearing a DDP snapshot.
    "dtype": ("315d3d0", "none"),  # the device-policy lane (`train/device.py`)
    "dist": ("6f75dfa", False),  # the multigpu lane (`train/distributed.py`)
    "dist_backend": ("6f75dfa", "auto"),
    "dist_shard": ("6f75dfa", "none"),
    "dist_reduce": ("6f75dfa", "auto"),
    "dist_bucket_mb": ("6f75dfa", 0),
    "dist_capacity": ("6f75dfa", ""),
}

#: The `moe_kernel` literals — the expert-pass arms `experts._dispatch` routes
#: between. One list, read by `validate`: a name that is not here is LOUD.
MOE_KERNELS = ("padded", "triton", "naive", "batched", "bucketed")
#: The expert-pass arms `moe_shared_compute` can reshape: the arms whose bank
#: reads a per-expert `[d, rank]` projection. `padded`/`bucketed` keep the old
#: shape as their control, and `naive` is the per-expert reference loop.
MOE_SHARED_COMPUTE_KERNELS = ("batched", "triton")
#: The `moe_lb_mode` literals — WHICH load-balance controller moves the router
#: (`experts.SparseMoE._lfb_step` and `switch_aux_loss`): `sign` is LFB's
#: `-coef·sign(L−mean)` (the default, every existing pin), `prop` its
#: proportional form `-gamma·(L−mean)/max(1,mean)`, `aux` no bias controller at
#: all plus the Switch/GShard loss term. A name that is not here is LOUD.
MOE_LB_MODES = ("sign", "prop", "aux")
#: The `moe_gather` literals — the gather-fused FORWARD (`model/moe_sonic.py`,
#: SonicMoE §5.1.1): `off` is today's pass, `kernel` reads the expert rows
#: THROUGH the permute's row map instead of a materialised `[R, d]` copy. It is
#: a second switch on the same pass, not a fourth `moe_kernel`: `moe_kernel` is
#: what runs when the gather arm does not, and `moe_gather` is refused (LOUD)
#: off the bf16 loop-stack route, with the adaptive width, and with `factor_qat`.
MOE_GATHERS = ("off", "kernel")
#: The `moe_permute` literals — the routing permute's algorithm
#: (`experts.SparseMoE._permute`): `argsort` is the stable sort every number on
#: record runs, `count` the counting sort (`model/route_fused.py`). Same tensor.
MOE_PERMUTES = ("argsort", "count")
#: The `moe_combine` literals — HOW the routed pass's scatter closes. `torch` is
#: `zeros + index_add_` (three fp32 passes over `[n·k, f]` + `[n, f]`), `fused`
#: is one Triton pass (`moe_kernels._combine_fwd_kernel`) that reads the k rows
#: of a token and writes its output, accumulating in fp32 registers.
MOE_COMBINES = ("torch", "fused")
#: The `moe_combine_dtype` literals — the dtype the fused combine STORES. `fp32`
#: is bit-identical to today's `index_add_` (two fp32 adds are commutative);
#: `bf16` writes the routed activation in the compute dtype, so the caller's
#: `weighted.to(dt)` disappears. Only meaningful with `moe_combine="fused"`.
MOE_COMBINE_DTYPES = ("fp32", "bf16")
#: The `share_u_mode` literals — how the routed experts' INPUT basis is shaped
#: (`experts.py`'s module docstring): `off` is one `U` per expert, `plane` is
#: one `U` for the bank, `dict` is one dictionary plus a per-expert column
#: subset. A name that is not here is LOUD.
SHARE_U_MODES = ("off", "plane", "dict")
#: The `ffn_form` literals — what the expert slot is:
#: `bank` (today's routed `SparseMoE`), `dense` (one plain GLU, no rank) and
#: `tsct` (all three matrices independently factorised). A name that is not
#: here is LOUD; `tests/test_ffn_form.py` pins the arithmetic of all three.
FFN_FORMS = ("bank", "dense", "tsct")
#: The `moe_topk_mode` literals — how many experts each TOKEN runs
#: (`model/adaptive_topk.py`): `off` is the constant `moe_topk` width every
#: number on record runs; `entropy`/`margin` are the per-token confidence
#: policies. A name that is not here is LOUD.
MOE_TOPK_MODES = ("off", "entropy", "margin")
#: The widest per-token `k` the instrument can name (`moe_k1`..`moe_k8`). Above
#: this is LOUD: the fused backward's top-k tile also stops at 8.
MOE_TOPK_SLOTS_MAX = 8
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
#: The `dense_qat` literals — the DENSE path's switch. It is `FACTOR_QATS`
#: exactly, and deliberately: `tsct.dense_qat_project` IS `factor_qat_project`
#: (one function object), so every rung the factor path implements is the same
#: projection on a plain `[out, in]` weight, with the same annealed STE and the
#: same per-tensor axis. `none` = the master untouched (every number on
#: record); the others quantize every `nn.Linear`/`DenseLinear` weight in the
#: forward (KDA's q/k/v/g/beta, the controller, `mem_dense`, the MoE router and
#: `down`, MSA's, the Engram's three, DSpark's three, and the dense readouts
#: `out_proj`/`lm_head` when `use_tsct = false`). Embeddings, norms, biases and
#: the Engram/TSCT tables stay unquantized (BitNet convention — see
#: `tsct.dense_qat_project`'s comment).
DENSE_QATS = FACTOR_QATS
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
#: The `mm_dtype` literals — the COMPUTE precision of the 2-D GEMMs
#: (`model/tsct.py`, which is where they execute; the list is duplicated here
#: the way `NS_KINDS` is, so the config layer does not import the model).
#: `bf16` is the default and every number on record: fp32 masters cast to the
#: compute dtype, cuBLAS bf16 tensor cores. `fp8` routes `mm_mixed`'s 2-D
#: products through `torch._scaled_mm` (e4m3 × e4m3 → bf16 out) with per-tensor
#: amax scales computed from the bf16 operand — the spike's measured 1.6-1.8×
#: on 2-D GEMMs (`docs/research/2026-10-15-torch-spike.md` phase 2; grouped fp8
#: is ABSENT on sm_120 and is not attempted). A COMPUTE rung: the fp32 masters,
#: the checkpoint and the optimizer are untouched. **On torch 2.13 the lane is
#: FORWARD-ONLY** — `_scaled_mm` registers no derivative (`RuntimeError:
#: derivative for aten::_scaled_mm is not implemented`, measured), so a training
#: forward falls back to bf16 COUNTED (`mm_fp8_fallback` on the log line, the
#: reason on `counters.why`) and the no-grad forwards (eval, decode, the bench)
#: run it. A name that is not here is LOUD.
MM_DTYPES = ("bf16", "fp8")
#: The `residual_dtype` literals — the STORAGE dtype of the loop's residual
#: stream (`streams`, `h`, `ctx`, `normed`, the three branch writes, `y` and
#: `step_out`). `fp32` is the default and every number on record. `bf16` stores
#: those tensors at half the bytes and widens each arm at its own seam; it does
#: NOT change the compute dtype (`bf16` does that) and it does not touch the
#: state-carrying paths, which stay fp32 (see the field's docstring for the
#: list). It is refused with the fp32-only Triton chains (the fused residual /
#: epilogue kernels and the KDA tail's combine), never silently dropped.
RESIDUAL_DTYPES = ("fp32", "bf16")
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


@dataclass
class ModelConfig:
    """The model half of a preset — flat, every key a field, defaults == small.

    A key the burn side has and this port does not is NOT a field: setting it is
    an unknown key, refused loudly with the key named and the escape. A field
    that reaches no code is the defect that refusal exists to prevent (the burn
    history's `use_mhc` reached a trainer that never read it — it is ported for
    real in `use_mhc` below, and `use_gr` is the still-dead example).
    """

    # --- geometry -------------------------------------------------------
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64
    d_ffn: int = 2048
    vocab: int = 256
    max_seq_len: int = 512
    max_iter: int = 4
    #: ---- UNIQUE BLOCKS INSIDE THE LOOP (`model/loopblock.py`) ----
    #: The number of DISTINCT loop blocks the loop cycles over:
    #:
    #:     h = prelude(...)
    #:     for i in range(max_iter):
    #:         h = blocks[i % loop_blocks](h, iteration=i)
    #:     h = coda(...)
    #:
    #: `1` (the default) is today's weight-shared loop, bit for bit: no extra
    #: module is built, no RNG is drawn and no state-dict key appears.
    #: `loop_blocks=2` at `max_iter=4` runs block 0 on iterations 0/2 and block 1
    #: on 1/3 — COMPUTE is unchanged, the transformation is not. Each block owns
    #: its FULL body (controller, KDA, Engram, MoE bank, readout), so the cost is
    #: ≈ `loop_blocks ×` one block's parameters, which is the depth the loop
    #: could not buy by iterating one block (`medium`: see
    #: `tests/test_loop_blocks.py`'s table).
    #:
    #: LOOP-LEVEL, deliberately NOT per block (indexed by ITERATION, as today):
    #: `iter_embed`, the `attnres` pseudo-query slots, and the per-iteration
    #: gates `msa_gate`/`csa_gate`/`loom_gate`. They describe which pass of the
    #: loop is running, not which transformation runs it, so block 1 sees the
    #: same `iter_embed[1]` it would have seen as the only block. mHC's mixing
    #: matrices ARE per block (each block owns its residual operator); only the
    #: K-lane STATE is shared across the loop, because it is the loop's state.
    #: Refused: `< 1`; `> max_iter` (a block that never runs is a config lie);
    #: `> 1` with `use_byteflow` (Option A builds no loop at all, and Option B
    #: runs it over K patch latents — an unmeasured state contract).
    loop_blocks: int = 1
    #: ---- HALT HEAD (`model/halt.py`): learned inference-time stopping ----
    #: λ on the halt BCE. `0.0` (the default) builds NO head and adds no
    #: parameter or compute; a positive weight builds
    #: `Linear(d, d//4) -> SiLU -> Linear(d//4, 1)` and adds the term to the
    #: training loss. The target is the model's OWN per-iteration CE: stop at
    #: iteration `i` iff `CE_i - CE_{i+1} < halt_eps` or `i` is the loop's last
    #: iteration — never a function of the label sequence (module docstring).
    halt_weight: float = 0.0
    #: The CE improvement (nats) below which one more iteration is called
    #: unhelpful. 0.005 nats = 0.0072 BPB; set by the research scan's reading
    #: of the depth curves, and a training knob (the A/B keeps it fixed).
    halt_eps: float = 0.005
    #: The earliest iteration the INFERENCE stop may fire at (1 = after the
    #: first). A floor, not a target: `halt_min_iter > 1` guarantees a minimum
    #: amount of thinking on every input.
    halt_min_iter: int = 1
    #: The INFERENCE cap on loop depth when `halt_mode` is on. LOUDLY refused
    #: above `max_iter` IFF `halt_mode` is on (the `4`/`8` default pair must
    #: stay a legal config: `max_iter=4`, `halt_mode=false` runs today's model).
    halt_max_iter: int = 8
    #: Inference-only switch: run the loop up to `halt_max_iter` and stop at
    #: the first iteration whose stop probability exceeds 0.5. `false` (the
    #: default) keeps fixed-depth inference — every existing snapshot's mode.
    #: Training never halts (the head's BCE needs the full scheduled depth);
    #: `halt_mode` without a halt head is LOUD.
    halt_mode: bool = False
    rank: int = 64
    #: WHICH FFN sits in the expert slot (`model/experts.py`, `model/ffn.py`):
    #:
    #: * `bank` (the default, every run and checkpoint on record) — today's
    #:   `SparseMoE`: routed `DualLatentGLU` experts at `rank`, ONE always-on
    #:   shared expert at `shared_rank`, and ONE shared `down [f, d]`. The
    #:   routed experts are a rank-`rank` bottleneck by construction; that is
    #:   this form, not a "dense" one.
    #: * `dense` — ONE plain GLU FFN in the expert slot: `gate [d, f]`,
    #:   `up [d, f]`, `down [f, d]`, `3·d·f + f + d` parameters. No rank
    #:   anywhere, no router, no shared expert, no bank. Only legal with
    #:   `n_experts = 1` and `share_u_mode = "off"` (a dense FFN has nothing to
    #:   route or share).
    #: * `tsct` — THE factorised form, all three matrices independently:
    #:   `gate = U_g[d, r_g]·diag(s_g)·V_g[r_g, f]`, same for `up`, and
    #:   `down = U_d[f, r_d]·diag(s_d)·V_d[r_d, d]`. `ffn_rank_gate` sets
    #:   `r_g` for BOTH gate and up (they are the same d→f map), `ffn_rank_down`
    #:   sets `r_d`; `0` means `rank`. At
    #:   `r_g = r_d = min(d, f)` this is an exact reparameterisation of the
    #:   dense GLU (gate c of `tests/test_ffn_form.py`), not an approximation.
    #:   With `n_experts = 1` it is one FFN; with `n_experts > 1` it is a bank
    #:   of `tsct` experts (dense blend at `moe_topk = 0`, top-k dispatch at
    #:   `moe_topk > 0` with `moe_capacity = 0` — the TSCT path is dropless).
    #:
    #: The default is `bank` ON PURPOSE: a new default would move every
    #: existing run and checkpoint, and this field's job is to make the choice
    #: visible, not to silently upgrade the model. `use_tsct` is a DIFFERENT
    #: axis (which linears elsewhere are spectral); it never selected the FFN.
    ffn_form: str = "bank"
    #: The `tsct` form's gate/up rank `r_g`. `0` = `rank`. The same rank is
    #: used for gate and up because they are the two halves of one d→f GLU.
    #: Refused on the other forms (a rank knob on a form with no factors is an
    #: inert key), and above `min(d_model, d_ffn)` (not a factorisation).
    ffn_rank_gate: int = 0
    #: The `tsct` form's down rank `r_d`. `0` = `rank`. Refused on the other
    #: forms and above `min(d_model, d_ffn)`.
    ffn_rank_down: int = 0
    norm_eps: float = 1e-3
    #: ---- LayerRoPE (`model/layerrope.py`, arXiv:2610.09179) ----
    #: Depth-wise magnitude + angular superposition of the norm gain. ONE shared
    #: `gamma_shared [d]` and per-depth `(amp_i, theta_i)` scalars replace the
    #: per-site RMSNorm gains:
    #:
    #:     gamma_i = amp_i · rotate_pairs(gamma_shared, theta_i)
    #:
    #: The depth index is the PRELUDE layer, the ACTUAL loop iteration (`P+i`,
    #: not a compacted run position), the CODA layer, or the terminal final
    #: norm (`P + max_iter + C`); all sites at one depth share `gamma_i`. With
    #: the identity init (`amp_init=1.0, theta_init=0.0`) the ON forward is the
    #: OFF forward bit for bit, but the state-dict keys differ (`layerrope.*`
    #: replace the per-site `.norm.weight`s), so a resume across the flag is
    #: refused by the snapshot drift gate. ALLOWED with `use_mhc`,
    #: `use_attnres`, `use_csa`, `attn_window`, `segment_len`, `rand_depth`,
    #: `loop_blocks`, `prelude_layers`/`coda_layers`, `tie_embeddings`,
    #: `ema_decay`, `grad_accum`, `muon_sf` (the gain table is 1-D → AdamW) and
    #: the QAT arms (their seam is dense weights, never norm gains). REFUSED
    #: with `use_byteflow` (the channel path builds no byte-level norm tree),
    #: an odd `d_model` (pair rotation), `full_graph`/`opt_graph`
    #: (`validate_model_train`), and any subfield set without this master
    #: switch.
    layerrope: bool = False
    #: `amp_i` at initialisation. `1.0` is the identity; a magnitude is
    #: positive, so `<= 0` is refused rather than silently zeroing the arm.
    layerrope_amp_init: float = 1.0
    #: `theta_i` at initialisation (radians). `0.0` is the identity rotation.
    layerrope_theta_init: float = 0.0
    #: One `(amp, theta)` per `(depth, site)` instead of per depth. `false` is
    #: the paper's single shared vector — this flag exists for the follow-up
    #: per-site A/B, not for the first measurement.
    layerrope_per_site: bool = False
    #: Learn `theta_i`. `false` makes the angle a fixed BUFFER (at
    #: `layerrope_theta_init`), so the angular part is a constant rotation.
    layerrope_learn_theta: bool = True
    #: Which LayerRoPE form runs. `"v1"` is the implementation this branch
    #: shipped — ONE shared `gamma_shared` + per-depth `(amp_i, theta_i)`, at
    #: the norm READ sites only, one angle for every channel pair. `"paper"` is
    #: the project page's reference snippet: `gamma[d]` + FOUR scalars per site
    #: (`a_mag`, `b_mag`, `a_rot`, `b_rot`), log-linear magnitude
    #: `r_l = a_mag + b_mag·log_l`, RoPE-spectrum per-PAIR rotation
    #: `ang_j = θ_l·base^(-2j/d)`, applied at BOTH the read and the write of the
    #: attention branch and of the FFN/MoE branch — four sites per block,
    #: `4·(d+4)` parameters (one site is `d + 4` + 2 buffers). `"v1"` is the
    #: default so no number on record moves; `"paper"` is a different parameter
    #: tree, so a resume across the form is refused by the snapshot gate.
    layerrope_form: str = "v1"
    #: The paper form's `beta_init`: `b_mag` and `b_rot` at initialisation.
    #: `-0.5` is the paper's value, and it is NOT an identity — the gain is
    #: depth-scaled from step 1 (`r_l = β·log_l`). Refused with `form="v1"`,
    #: which reads no beta.
    layerrope_beta_init: float = -0.5
    #: The paper form's `base` of the RoPE spectrum `base^(-2j/d)`. `100.0` is
    #: the paper's value; `> 0` only. Refused with `form="v1"`.
    layerrope_base: float = 100.0
    #: The depth index. `"iteration"` (default, v1's rule) is the ACTUAL loop
    #: iteration — a shorter run uses a prefix of the table. `"position"` is
    #: the body's position in the NETWORK, independent of the loop iteration:
    #: the paper's looped-model reading (Parcae), under which every iteration
    #: of one weight-shared body reads the same gain. Refused with `form="v1"`.
    layerrope_index: str = "iteration"
    #: The paper form's initialisation. `"paper"` is `beta_init` as given (a
    #: live depth scaling from step 1). `"identity"` sets `b_mag = b_rot = 0`
    #: AND turns the spectrum off, so every depth's gain IS `gamma` and the
    #: forward equals the `layerrope=false` forward bit for bit — the arm's
    #: bit-identity gate, not the paper's init. Refused with `form="v1"`.
    layerrope_init: str = "paper"

    # --- precision ------------------------------------------------------
    #: bf16 activations/weights with fp32 masters (`docs/architecture/bf16-plan.md`).
    #: The forward matmuls run in bf16; the softmax, the loss and the TSCT
    #: latents stay fp32. Default True: the port's init path is the bf16 plan,
    #: unlike burn's `small.toml`, which still ships fp32 (listed in the port
    #: report as a deliberate deviation).
    bf16: bool = True
    #: The STORAGE dtype of the loop's residual stream (`RESIDUAL_DTYPES`).
    #:
    #: `fp32` (the default, every number on record) keeps the whole loop in
    #: fp32. `bf16` stores the tensors every iteration carries at half the
    #: bytes: the mHC lanes and the body state (`h`, `ctx`), the body's normed
    #: reads, the three branch writes (`attn`/`branch`/`ffn`), the residual
    #: update `y` and the readout `step_out`. `scripts/precision_census.py`
    #: prices the table; `tests/test_residual_bf16.py` measures the drift.
    #:
    #: **What stays fp32, and why** (the census's `required` column): the mHC
    #: lane RMSNorm, the hyper-network sigmoids and the Sinkhorn-Knopp
    #: projection (log-domain, 40 `logsumexp`), the lane mix; AttnRes' key norm
    #: and softmax; the KDA projections' input (fp32 masters, and the fused tail
    #: is fp32-input only), the log-decay `g` and the gated-delta state `S`; the
    #: CSA chunk scores and ring `sum`; the Engram read (fp32 tables and the
    #: order reduction); the readout accumulator `out_acc`; the final norm, the
    #: head and the CE. Those are the reductions and state updates, not the
    #: stream.
    #:
    #: This field is INDEPENDENT of `bf16`: `bf16` chooses the GEMM compute
    #: dtype, this chooses what the loop stores. Both combinations are legal.
    #: REFUSED, each with `Escape:`: `fused_residual` and `fused_epilogue`
    #: (their Triton chains are fp32-only) and `fused_kda_tail` (its `combine`
    #: kernel is fp32-only); the projection tail alone is not separately
    #: switchable, so the escape is `fused_kda_tail=false`.
    residual_dtype: str = "fp32"
    #: TSCT factor format. `none` resolves to bf16 factors at init; every
    #: ladder rung (fp8 → NVFP4 → LittleBit) is a REFUSAL, not a fallback.
    quant: str = "none"

    # --- arms -----------------------------------------------------------
    #: The hard ATTENTION ON/OFF switch (every number on record: true). `false`
    #: is the degenerate zero branch the loop has always had (`LoopBlock._window`
    #: writes `torch.zeros`), NOT an attention arm — kept bit-for-bit because
    #: `tests/test_decode_recurrent.py` and `bench_bf_dtype.py` run it.
    use_kda: bool = True
    #: WHICH attention arm runs when `use_kda` is on:
    #:
    #: * `kda` (the default) — the compressed-state gated-delta arm
    #:   (`model/kda.py`): O(1) state, the arm every number on record ran. The
    #:   legacy spelling `use_kda=true` means this arm.
    #: * `full` — exact causal attention (`model/fullattn.py`,
    #:   `F.scaled_dot_product_attention(is_causal=True)`): query `p` reads keys
    #:   `0..p`, no compression. `4·t·d_model = 4.2M` MAC/token at t=1024,
    #:   d=1024 — the same order as the expert FFN, i.e. affordable at our
    #:   context length. `q,k,v,o` = `4·d²` parameters against KDA's
    #:   `4·d² + d·n_heads`: at `medium` the delta is **-16 384** (total
    #:   141 910 405 → 141 894 021, `dormouse info`).
    #:
    #: `use_kda=false` does NOT mean `full`: an off switch that changed the arm
    #: it switches on would move every zeros-arm run. `full` with `use_kda=false`
    #: is refused (`validate`). REFUSED with it too, each naming the key and the
    #: escape: `segment_len>0` (`_validate_segment`: no O(1) boundary state) and
    #: the recurrent decode/prefill (`decode.recurrent_refusal`, `LoopBlock._attn`:
    #: a one-byte step has no window to attend over). Allowed: `use_mhc` (residual
    #: lanes, independent of the attention arm — `tests/test_fullattn.py` runs the
    #: pair) and `use_byteflow` (the channel's loop runs over K patch latents, and
    #: causal attention over those latents is well-defined — the same sequence
    #: KDA already attends over, uncompressed).
    attn_kind: str = "kda"
    #: ---- an EXACT LOCAL WINDOW beside whatever arm `attn_kind` names ----
    #: `model/fullattn.py`'s `FullAttention` restricted to the last `attn_window`
    #: positions (`0` = OFF, no module, no parameter, every hash unchanged):
    #: position `p` reads keys `max(0, p-W+1)..p` — strictly causal (`p` itself
    #: included, unlike CSA's strictly-earlier chunks) and NOTHING older. It is
    #: an ADDITIVE third path, summed into the same `attn` branch as KDA/CSA
    #: (`LoopBlock._window`), so `kda + window` and `kda + csa + window` are
    #: legal — the point is exact recall of the local band that KDA's O(1) state
    #: compresses away. Cost on top of the rest: `W·d` MAC/byte for the score
    #: and `W·d` for the value (`fullattn.window_mac_per_byte`), i.e. FLAT in the
    #: context length — the reason a 32 KiB window at 4 MiB is affordable.
    #: `torch.nn.functional.scaled_dot_product_attention` in this environment
    #: (torch 2.13.0+cu129) has NO `window_size` argument (`TypeError:
    #: unexpected keyword argument 'window_size'`, measured), so the band is an
    #: explicit causal-band boolean mask — exact on every backend, and it is
    #: what the chunked path slices.
    #: REFUSED loudly: negative, `attn_kind='full'` (the window is a strict
    #: subset of the arm already there — a silent second attention), with
    #: `segment_len>0` (a segment cut would hide the bytes before the cut, so
    #: the segmentation would change the model), with `use_byteflow` (its loop
    #: runs over patch latents, so a "byte window" would be a latent window),
    #: and on the recurrent decode (`decode.recurrent_refusal`).
    attn_window: int = 0
    #: Query-block size for `attn_window` (`0` = one block, the whole window).
    #: A MEMORY-ONLY lever: the banded score tensor is `C × (C+W-1)` per block
    #: instead of `t × t`, so a 4 MiB window never materialises a 4M×4M band.
    #: The output equals the unchunked path to bf16/fp32 round-off (printed as
    #: max|Δ| by the gate) — the SDPA kernel sees the same masked band, sliced.
    #: REFUSED without `attn_window`, and refused at `>= max_seq_len` (one
    #: block: the flag would buy nothing while the log line reported it on).
    attn_window_query_chunk: int = 0
    use_tsct: bool = True
    #: TIED EMBEDDINGS: `lm_head` IS `embed.weight`, transposed, and has no
    #: parameter of its own (`model/model.py::TiedHead`). That is the frontier
    #: stack's input/output matrix sharing: `d_model·vocab` parameters saved
    #: against a dense head, and the head's gradient is ADDED to the embedding
    #: table's (one matrix, two paths — gated).
    #:
    #: The tied head is a PLAIN `[vocab, d]` matrix, so on a `use_tsct=true`
    #: body it REPLACES the low-rank TSCT head (`u·diag(s)·vᵀ`, `rank`
    #: parameters) — a second difference a reader of that arm must know. On
    #: `use_tsct=false` the head was exactly `[vocab, d]`, so tying is the
    #: clean comparison and saves exactly `d_model·vocab`. `factor_qat` no
    #: longer reaches the head when tied (there is no head factor pair to
    #: quantize); the embedding is untouched by it either way.
    #:
    #: REFUSED with `use_byteflow` (both shapes bypass `lm_head`; the flag
    #: would be inert while the snapshot reported it on). `false` is every
    #: number on record, and the off path is bit-identical.
    tie_embeddings: bool = False

    # --- cheap width (2026-10-23): stop `d` from setting the dense bill ----
    #: MLA-style SHARED LATENT for KDA's four `d×d` projections (`q,k,v,g`).
    #: `0` (off) keeps four `nn.Linear(d, d)`: `4·d²` MAC/token per iteration,
    #: every number on record. `r > 0` projects the block input ONCE
    #: (`latent_down`, `d·r`) and reads the four projections off that latent
    #: (`latent_up`, `r → out` each), so the four cost `d·r + r·Σ out` =
    #: `5·d·r` while the width stays `d`. `beta_proj` is untouched (`d·h`).
    #: The reshapes and the FLA call are unchanged.
    #:
    #: NOT refused with `factor_qat != "none"`: the four projections — and this
    #: latent — are plain `nn.Linear` DENSE sites, and `factor_qat` reaches only
    #: TSCT factor pairs (`u`/`v`). The switches are disjoint, not ambiguous.
    #: `dense_qat` DOES reach them, through the same `dense_qat_kind` the module
    #: already reads (`dense_qat_skip_attn` keeps attention bf16, latent
    #: included). Refused: `r < 0`; `use_kda = false` (no module to reach).
    attn_latent_rank: int = 0
    #: GQA on KDA's `k`/`v` projections. `0` (off) is MHA: both project to
    #: `n_heads·head_dim = d`. `h_kv > 0` projects them to `h_kv·head_dim` and
    #: EXPANDS with `repeat_interleave(n_heads // h_kv, dim=2)` immediately
    #: before the kernel, so `q,k,v,g` enter FLA at `[b,t,n_heads,head_dim]`
    #: exactly as before and only the PROJECTION cost falls
    #: (`2·d² → 2·d·h_kv·head_dim`). Composes with `attn_latent_rank`: the k/v
    #: ups then go `r → h_kv·head_dim`.
    #: Refused LOUDLY: `h_kv < 0`, `h_kv > n_heads`, `n_heads % h_kv != 0`
    #: (a ragged group has no valid expansion), `use_kda = false`.
    n_kv_heads: int = 0
    #: ONE latent projection for q/k/v/g (`kda.KDAAttention.projections`): the
    #: four weights are concatenated to `[d + 2·kv_width + d, in]` and multiplied
    #: with ONE GEMM, then split by a view. OFF is the four `nn.Linear` calls
    #: every number on record ran, bit for bit. ON is NOT bit-identical to those
    #: four GEMMs: a single GEMM may reorder the K reduction, so the gate is a
    #: 1e-6 fp32 / 1e-2 bf16 relative match, stated rather than papered over
    #: (`tests/test_fused_epilogue.py`). Works with `attn_latent_rank` (the four
    #: read the shared latent) and with the QAT arm (each master is projected
    #: first, then concatenated). Needs `use_kda = true`; there is no module to
    #: reach otherwise.
    fused_latent_proj: bool = False
    #: Low-rank `out_proj` (the loop readout): `d → r → d`, cost `2·d·r`
    #: instead of `d²`. `0` (off) is the readout exactly as before — TSCT at
    #: `rank` on the default rung, a dense `DenseLinear(d, d)` when
    #: `use_tsct = false`. `r > 0` sets THIS projection's rank and nothing
    #: else: on the TSCT rung it builds `TSCTLinear(d, d, r)` (the same
    #: `U diag(s) Vᵀ` form, one rank knob on one matrix); on the dense rung it
    #: builds two dense linears (`down [r,d]`, `up [d,r]`), `2·d·r` params.
    #: NOTE the TSCT default `rank` is already 64, so at `d = 1024` and
    #: `r = d/4 = 256` the TSCT readout GROWS; the flag is a win where the
    #: readout was dense or `r < rank`. Refused: `r < 0`, `r > d_model`.
    out_proj_rank: int = 0
    #: The expert bank's SHARED input basis (owner directive 2026-10-21) — the
    #: OLD SPELLING of `share_u_mode = "plane"` below (kept because every
    #: `--set` and every snapshot names it this way; the two disagreeing is
    #: LOUD). Every ROUTED expert stops owning its `gate_u`/`up_u [d, rank]` and
    #: reads ONE tensor per site that `SparseMoE` owns (`experts.py`'s module
    #: docstring). The per-expert specialization left is `s` and `V [rank, f]`,
    #: so the bank's capacity lives in a shared dictionary of input directions
    #: instead of E interchangeable copies — the mechanism that can plausibly
    #: make a rank-2 bank carry anything (rank 16 with a private `U` measured
    #: +0.136 BPB, and topk-4 over the same weak bank +0.127; `scripts/ab_rank.py`
    #: measures the shared ladder).
    #:
    #: OFF by default, and off is bit-identical: no parameter is created, no RNG
    #: is drawn and no counter is emitted, so every snapshot on disk and every
    #: parameter count on record is unchanged. The always-on shared expert is
    #: NOT bound (it runs at `shared_rank`, a different shape, and E = 1).
    #:
    #: REFUSED loudly with `use_tsct = false` (the shared basis is a TSCT factor,
    #: so there is no basis to share with the projections dense). With
    #: `factor_qat != "none"` the basis IS quantized: ONE projection per site
    #: with its own single α (`dim=None`) and the rung's anneal, cached once per
    #: optimizer step and read by every expert (plane whole, dict through the
    #: seeded gather) — `experts.SparseMoE._qat_cache` owns the one entry.
    share_u: bool = False
    #: How the shared basis is shaped — THE switch (`share_u` above is its
    #: `plane` spelling, kept so every snapshot and `--set share_u=...` still
    #: means what it meant; the two disagreeing is a LOUD error, not a merge).
    #:
    #: * `off` — every expert owns its `gate_u`/`up_u [d, rank]`. Every number
    #:   on record, and bit-identical to the tree before the flag existed.
    #: * `plane` — ONE `[d, rank]` per site: all experts project the input onto
    #:   the SAME `rank`-dimensional plane. At rank 2 the whole bank can only
    #:   see two directions of a `d`-dim state, which is exactly the weakness
    #:   this mode's control exists to price.
    #: * `dict` — ONE `[d, share_u_rank]` DICTIONARY per site, and each expert
    #:   reads `rank` of its columns (`idx[E, rank]`, fixed and seeded, never
    #:   learned — see `experts.py`). Two experts then see different
    #:   `rank`-dimensional subspaces of the same dictionary, which is the one
    #:   thing a shared plane cannot do at rank 2.
    #:
    #: `dict` needs `use_tsct = true` for the same reason `plane` does, plus
    #: `share_u_rank > rank` (a width at or past the dictionary leaves no subset
    #: to choose). `factor_qat` rides either mode (one quantized basis per site).
    share_u_mode: str = "off"
    #: The `dict` mode's dictionary width `R`. `d*R` parameters per site, shared
    #: by every routed expert; each expert spends `rank` of its columns. 64 at
    #: medium (d=1024) is 65 536 parameters against the control's 78.3M bank.
    share_u_rank: int = 64
    #: ---- TSCT-2 (`docs/architecture/2026-11-05-tsct-2.md`) ----
    #: Per-expert INPUT selector inside the shared plane (see
    #: `experts.SparseMoE._banks`). `0` off. When on, the routed shared plane
    #: (`gate_u`/`up_u`) is widened from `rank` to `shared_rank` columns and each
    #: expert owns `w [rank, shared_rank]`: it selects its own `rank`-dimensional
    #: subspace of that wider shared plane, instead of only re-weighting the one
    #: shared `rank`-dim coordinate system. It must equal `rank`: the selector's
    #: output IS the GLU latent width, and the expert's write is `v [rank, f]`.
    #: The selector is FOLDED into the effective per-expert basis at the bank
    #: rebuild (`gate_u @ w.T [d, rank]`), so the per-token forward MAC is
    #: unchanged and the A/B is at equal active MAC; the once-per-step fold costs
    #: `E·d·rank·shared_rank` MAC per site. Init is `[I | 0]`, so at step 0 the
    #: arm is output-equivalent to the planeless path (gate b of
    #: `tests/test_tsct2.py`). Refused unless `share_u_mode='plane'`, with
    #: `factor_qat`, or `moe_shared_compute`.
    tsct_expert_basis: int = 0
    #: Per-expert LR multiplier from the routed counts (see
    #: `experts.expert_lr_multipliers` and `train.optim.MixOptimizer`):
    #: `lr_j = lr · (1 + c · (mean(L)/max(L_j,1) − 1))`, clipped to `[0.2, 5]`,
    #: applied to that expert's `v` only. `0.0` off. Counts are the LFB
    #: integrator's device-side `lfb_counts`, so the arm reads no host scalar on
    #: the step path. Requires a live LFB controller (`moe_lb_mode` sign/prop
    #: with a non-zero step) and the Muon half (`opt='mix'`, `muon_variant='muon'`,
    #: `muon_sf=false`): the fused AdamW group and the SF/NorMuon tails have no
    #: row-scale seam, and the flag is refused rather than silently inert.
    tsct_expert_lr: float = 0.0
    #: Revive cadence in OPTIMIZER STEPS: every N steps the coldest expert copies
    #: the hottest expert's `v`/`gate_s`/`up_s` (and `w` when the basis arm is on)
    #: plus noise, and both experts' `lfb_counts`/`lfb_bias` entries are reset.
    #: `0` off. Requires the same live LFB counts as `tsct_expert_lr`.
    tsct_expert_revive: int = 0
    #: The revive noise scale: `σ = tsct_revive_noise · std(v_hot)`, drawn from a
    #: generator seeded `(run seed + step)`.
    tsct_revive_noise: float = 0.01
    use_engram: bool = True
    #: AttnRes (`2603.15031`): the loop-boundary write becomes a softmax
    #: mixture over the depth sources with one learned pseudo-query per
    #: iteration slot (`loopblock.depth_attend`; the paper's score form).
    #: COMPOSES with `use_mhc`: with both on, mHC owns the K-lane state
    #: (`H_pre` reads it, `H_post`/`H_res` write it) and AttnRes owns WHICH
    #: UPDATE the write carries (`loopblock`'s "THE COMPOSITION"). Alone, it
    #: replaces ReZero exactly as it always did.
    use_attnres: bool = False
    #: mHC (`2512.24880`, DeepSeek 2025): the residual becomes TWO streams
    #: mixed by a doubly-stochastic (Sinkhorn-projected) matrix, at the LOOP
    #: BOUNDARY (the paper's K=2 placement, `docs/reviews/mhc-2026-09-30.md`).
    #: The iso-depth scaling study (`2604.21106`) measures hyperconnections
    #: raising the recurrence-equivalence exponent φ 0.453 -> 0.646, which is
    #: exactly the "deeper loops stop paying" wall we hit at iteration 4.
    #: `false` (the default) leaves the model bit-identical. Composes with
    #: `use_attnres` (see the field above); `loom` and `use_byteflow` stay
    #: refused alongside it.
    use_mhc: bool = False
    #: Lanes. Only K=2 ships: it is the configuration the paper measured and
    #: the one the review wires; `validate` refuses anything else.
    mhc_streams: int = 2
    #: Eq. 9's Sinkhorn normaliser (`model/mhc.sinkhorn_knopp`). `"logsumexp"`
    #: is the shipped arithmetic, bit for bit (`m -= logsumexp(m)`).
    #: `"log_softmax"` is the SAME expression as one ATen op, so only fp32
    #: roundoff moves (≤2.4e-7 absolute on the probability, measured) while the
    #: 20-iteration Python loop's per-normalisation subtract leaves the graph:
    #: 881 -> 81 dispatches per call. The `mhc_write` region's kernel profile
    #: (2026-11-05) priced that subtract family at 1 395 launches / 10 steps.
    #: It does NOT touch the fused kernel (`fused_residual._write_kernel`),
    #: whose 20 iterations are already in registers.
    mhc_sinkhorn: str = "logsumexp"
    #: The residual chain as fused Triton kernels (`model/fused_residual.py`):
    #: `H_pre`'s mix, the AttnRes write, `H_res`/`H_post` and the lane readout
    #: become two forward launches and two backward launches per iteration
    #: instead of ~780 (`tests/test_fused_residual.py` pins the count and the
    #: parity). OFF by default and bit-identical when off: the eager lines run
    #: unchanged. Needs `use_mhc` (there is no lane state to fuse without it —
    #: `validate` refuses the pair), K=2, and fp32 lanes; a CPU host is a
    #: COUNTED fallback (`fused_residual_fallback`), not a silent one.
    fused_residual: bool = False
    #: The per-iteration NORM CHAIN as ONE Triton read kernel
    #: (`model/fused_residual.epilogue_read`): the lane RMSNorm, the eight
    #: hyper-network dots, `ctx`, and the body's `RMSNorm(ctx)·LayerRoPE gain`
    #: in a single pass over the lanes/row instead of 6+ elementwise passes.
    #: The write half (AttnRes score, Sinkhorn, `H_post`/`H_res`, readout) is
    #: the existing `fused_residual` kernel: the loop body sits between the
    #: two, so one kernel per side is the floor, not a missed fusion. OFF is the
    #: eager lines, bit for bit. Needs `use_mhc` and fp32 lanes (same refusals
    #: as `fused_residual`); a CPU host is a COUNTED `fused_epilogue_fallback`.
    fused_epilogue: bool = False
    #: The KDA-side elementwise chain as TWO Triton kernels
    #: (`model/fused_kda.py`; `scripts/pass_census.py` ranks every group):
    #:
    #: 1. `tail` — the four fp32 projection outputs are the kernel's inputs:
    #:    `q/k/v` cast to the compute dtype, `g = logsigmoid(g)` fp32 and
    #:    `beta = sigmoid(beta)` become ONE launch per iteration instead of the
    #:    six eager passes (`kda.projections` + the four casts in `_fla_path`).
    #:    The traffic is unchanged (a cast reads its source and writes its
    #:    destination either way); five launches leave.
    #: 2. `combine` — `y = attn·attn_write + branch·attn_write + ffn·mlp_write`
    #:    is ONE launch over the three branch outputs instead of five eager
    #:    passes (`loopblock._step_post`). On the m400b run this is the census's
    #:    #2 group by bytes: 604 MB -> 201 MB per iteration, ~212 us at 1.9 TB/s.
    #:
    #: OFF by default and bit-identical when off: the eager lines run unchanged.
    #: fp32 inputs only (the residual stream and the KDA projections are fp32
    #: even under `bf16=true`, so the training arm qualifies); a CPU host or a
    #: bf16-cast model is a COUNTED `fused_kda_tail_fallback`. The census's #1
    #: group by traffic is the Engram read (982 MB/iteration), NOT this one —
    #: the flag is named for the arm it fuses, not for the ranking.
    fused_kda_tail: bool = False
    #: LOOM (`2610.01153`, 2026-10-01; scanned in
    #: `docs/research/2026-10-16-new-tech-scan.md` §2.1, whose row names our
    #: own `max_iter 4` / operating depth 2 as exactly the regime that paper
    #: measures as stuck at two loops). A REAL arm, not a stub:
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
    use_byteflow: bool = False
    #: ---- the prelude/coda topology (`arXiv 2610.23036`, `model/prelude.py`) ----
    #: A few UNIQUE layers that run once BEFORE the shared loop and a few that
    #: run once AFTER it: `embed → prelude(0..N) → LoopBlock ×T → coda(0..M) →
    #: RMSNorm → lm_head`. The loop stays weight-shared; only the layers outside
    #: it are unique. Each is a pre-norm residual SwiGLU FFN at `d_ff = d/2`
    #: (1 573 888 params at `medium`'s d=1024), deliberately NOT a second KDA:
    #: a KDA copy is 4.2M there, which would put 1+1 outside the +2-6M envelope
    #: and add a second attention arm to the model.
    #: `0`/`0` (the default) builds nothing, draws no RNG and adds no state dict
    #: key, so every number on record is bit-identical (`c8a2ff9` golden).
    #: Legal range 0-4 per side; refused outside it, and refused with
    #: `use_byteflow` (the byteflow shapes run no byte-level loop to wrap).
    prelude_layers: int = 0
    #: See `prelude_layers`. The coda runs on the loop's mean readout, before
    #: the final norm and `lm_head` — those two never move.
    coda_layers: int = 0

    # --- ByteFlow (arXiv 2603.03583): the byte-compression architecture -----
    # A REAL arm, not a stub: `byteflow.py` ports the crate's
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
    #: L2 fast path (the default: pure tensor ops, no host sync). REFUSED with
    #: `bf16 = true`: bf16 Cholesky is unstable (`config._validate_byteflow`).
    byteflow_logdet: bool = False
    #: Byte-sequence length the RoPE tables are built for.
    byteflow_max_bytes: int = 8192
    #: Which codec turns the byte window into K patch latents. `"attention"` is
    #: the ported ByteFlow net and the default — the bit-identical path every
    #: number on record ran. `"conv"` is `model/convcodec.py`: strided-conv
    #: patchify + low-rank unpatchify, no attention, ~1.9 % of the loop's MACs
    #: per patch instead of the attention codec's >400 %. Refused for any other
    #: name, and refused with `use_byteflow = false` (nothing would build it).
    byteflow_codec: str = "attention"
    #: Bytes per patch of the conv codec's strided patchify (`byteflow_codec =
    #: "conv"` only; the default 8 is the training target, 4/8 are the bench's
    #: arms). The runtime window must be divisible by it and K must be <= the
    #: patch count. REFUSED at a non-default value under the attention codec:
    #: the field would be inert.
    byteflow_patch_bytes: int = 8

    # --- MoE ------------------------------------------------------------
    n_experts: int = 3
    #: Experts per position. `0` is the DENSE blend (burn's default), `k` is
    #: the sparse top-k dispatch. Above `n_experts` is refused by validate.
    #: With `moe_topk_mode` != `off` this field is the ARM SWITCH (`> 0` selects
    #: the routed pass); the per-token width comes from `moe_topk_max` there.
    moe_topk: int = 0
    #: How many experts each TOKEN runs (`model/adaptive_topk.py`):
    #: `off` (the default) is the constant `moe_topk` width — every existing run
    #: and snapshot is bit-identical on it; `entropy` grows the width with the
    #: router's normalized entropy; `margin` shrinks it with the top-1/top-2
    #: logit gap. INFERENCE-ONLY: no new parameters and no training path.
    moe_topk_mode: str = "off"
    #: The narrow/wide ends of the adaptive width, in experts. `k_min >= 1`,
    #: `k_min <= k_max <= min(8, n_experts)`; a violation is LOUD at load. Inert
    #: when `moe_topk_mode = "off"`.
    moe_topk_min: int = 1
    moe_topk_max: int = 4
    #: The confidence window the policy ramps across. `entropy`: normalized
    #: router entropy in `[0, 1]`; `margin`: the top-1 minus top-2 LOGIT gap in
    #: logit units. Read only when `moe_topk_mode != "off"`; `lo < hi` is LOUD.
    moe_entropy_lo: float = 0.2
    moe_entropy_hi: float = 0.6
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
    #: Tie-breaking noise on the router logits, Switch-Transformer style
    #: (`std` in logit units). Measured 2026-11-04: with a fresh bank of 512
    #: near-identical experts and NO jitter, one expert took 3 800 of 8 192
    #: tokens, so the dropless bank needed 5.56 GiB and the "512 experts" were
    #: doing the work of two. 0.01 is the Switch value. OFF, and it consumes RNG
    #: only when on, so every existing pin and RNG stream is untouched.
    moe_router_jitter: float = 0.0
    moe_capacity: float = 1.0
    #: Loss-Free Balancing step (arXiv:2408.15664): before the top-k decision
    #: the LOGITS read `logits + b_e`, and after the forward `b_e` moves by
    #: `-coef · sign(load_e − mean load)` — a bias, NOT an aux loss (no
    #: interference gradients; the field converged here away from Switch-style
    #: losses). 0.0 = off. The update is on-device per routed forward.
    #: INERT under `moe_lb_mode="aux"` (that arm has no bias controller; the
    #: `moe_aux` term is its whole mechanism).
    moe_lb_coef: float = 0.0
    #: WHICH load-balance controller runs (`MOE_LB_MODES`, `experts._lfb_step`):
    #: `sign` (the default — every existing preset, pin and checkpoint) moves the
    #: bias by `-moe_lb_coef · sign(L_e − mean L)` from the CUMULATIVE counts
    #: `L`; `prop` by `-moe_lb_gamma · (L_e − mean L) / max(1, mean L)` on THIS
    #: window's histogram, clamped to ±`moe_lb_clamp`; `aux` turns the bias
    #: controller OFF and adds the Switch/GShard term
    #: `moe_aux_coef · E · Σ_e f_e · P_e` to the training loss (differentiable
    #: through the router only).
    #: Why `prop` exists (measured 2026-11-04, `docs/reviews/2026-11-04-routing-
    #: collapse-CONFIRMED.md`): `m400` at step 149 999 carries a saved bias of
    #: `[-218.27, +72.67]` and its router keeps **9 of 512** experts alive
    #: (top-10 = 100% of all assignments). The bias-less router reads 80/512, so
    #: the controller is what collapsed the bank. `sign` is a cumulative
    #: bang-bang integrator with no equilibrium and no bound — it drifted there;
    #: `prop` is a bounded, scale-free P loop on the current window.
    moe_lb_mode: str = "sign"
    #: The `prop` controller's step, in units of the RELATIVE deviation: a dead
    #: expert is `-1`, so it gains `moe_lb_gamma` per routed forward until the
    #: clamp. `0.001` is the paper's γ (2408.15664) and the value the
    #: routing-collapse note asks for; `0.0` disables the controller under this
    #: mode (the escape `moe_lb_coef=0` is for `sign`). Inert under `sign`/`aux`.
    moe_lb_gamma: float = 0.001
    #: The `prop` controller's output bound, in LOGIT units. The measured
    #: divergence this field exists for: `m400`'s saved bias spans
    #: `[-218, +73]` (mean `+34`) and its routing froze at 9 of 512 live
    #: experts — in logit space a ±200 bias permanently overpowers a router
    #: whose own scores are ±5. A bounded controller cannot do that. `sign` is
    #: the legacy arm and is NOT clamped (its pins predate this field). Inert
    #: under `sign`/`aux`.
    moe_lb_clamp: float = 2.0
    #: The `aux` term's weight. The term is `E · Σ_e f_e · P_e` with `f` summing
    #: to 1 over the bank, so it reads exactly `1` at a uniform load and a
    #: uniform gate (the Switch value, at any top-k) and up to `E` when one
    #: expert takes every assignment — one weight means the same thing at any
    #: bank size. `0.01` is the Switch value and its gradient reaches the router
    #: and nothing else. Inert unless `moe_lb_mode="aux"`.
    moe_aux_coef: float = 0.01
    #: The routing-HEALTH counter set (`experts._route_health`): per routed
    #: forward, FIVE device counters, drained on the log cadence like every other
    #: device counter (no host read on the step path) —
    #: `moe_route_live` = experts with at least one assignment,
    #: `moe_route_max_x1000` = the busiest expert's share × 1000,
    #: `moe_route_top10_x1000` = the ten busiest experts' share × 1000,
    #: `moe_route_ent_x1000` = the histogram's entropy normalized by `log E`
    #: × 1000 (1000 = uniform, 0 = one expert), and `moe_route_n` = the number of
    #: forwards accumulated. Each is a SUM over the window, so the reader divides
    #: by `moe_route_n`: `moe_route_live/moe_route_n` is the mean live count the
    #: collapse is measured by. OFF by default (it costs one `bincount`, one
    #: `topk` and five scalars per forward); a run that wants the reading turns
    #: it on.
    moe_routing_health: bool = False
    #: Which expert-pass arm the routed forward runs, and the A/B switch
    #: (`docs/research/2026-10-15-tsct-super-kernel.md` step 2):
    #: `padded` = the cap-padded `[E, cap, ·]` bank of bmms — the reference and
    #: the fallback every other arm is compared against; `triton` = the unpadded
    #: grouped GEMM (`model/moe_kernels.py`, one program per ragged row-block,
    #: GLU fused); `naive` = the per-expert loop (the CPU path); `batched` = the
    #: SAME bank with the cap chosen for this batch (the capacity rule when
    #: `moe_capacity > 0`, the largest group when dropless) — the arm that turns
    #: the dropless Python loop into three `bmm`s; `bucketed` = the same bmms over
    #: a PLAN of `[n_b, cap_b, ·]` banks (experts classed by row count), so a
    #: collapsed router costs `2 · 2048` rows per bucket instead of
    #: `E · max_group`. An unknown name is LOUD at
    #: config load. `triton` falls back COUNTED to `padded` when the kernel
    #: cannot run (no triton import, input not on CUDA) — never silently.
    moe_kernel: str = "padded"
    #: WHICH SHAPE the routed expert pass computes in, under `share_u` — OFF by
    #: default, so every pin and checkpoint on disk is untouched.
    #:
    #: OFF: the dispatch materializes the shared basis E-fold and runs
    #: `bmm(xg[E, cap, d], gate_u[E, d, r])` (and the same for `up_u`) — the
    #: expensive `d -> r` projection is repeated for every expert, even though
    #: `share_u` has ONE basis. ON: the same parameters are read ONCE as
    #: `W = [gate_u | up_u] [d, 2r]`, `lat = x[n, d] @ W` is one GEMM over the
    #: TOKENS (not the `n·k` assignments: two assignments of a token share the
    #: latent), and the bank then gathers `[n·k, 2r]` and keeps today's
    #: `silu(g)·u`, `@ v`, weighted scatter. The arithmetic is the same; the
    #: projection stops depending on E and its MAC drops by the top-k factor.
    #:
    #: Only `moe_kernel` in `MOE_SHARED_COMPUTE_KERNELS` has this shape
    #: (`batched` = the `[E, cap, ·]` bank, `triton` = the grouped GEMM fed the
    #: latents through two constant selector matrices); `padded`/`bucketed`
    #: stay the control. Refused LOUDLY with `share_u_mode='off'` (no shared
    #: basis to read), `rank < 1`, `share_u_mode='dict'` (each expert reads a
    #: different column subset, so there is no one `W`), a heterogeneous
    #: per-expert rank, `moe_gather='kernel'` (that arm owns the forward), and
    #: a `moe_kernel` outside the tuple — an arm asked for and silently not run
    #: is the defect ADR-0019 bans.
    #:
    #: Counters: `moe_shared_compute_calls` (one per routed dispatch that took
    #: this shape) and `moe_projection_rows` (the device-side row count of the
    #: ONE projection, i.e. the token count) — a log line without them did not
    #: run the arm it is labelled with.
    moe_shared_compute: bool = False
    #: The gather-fused FORWARD (`model/moe_sonic.py`, SonicMoE arXiv:2512.14080
    #: §5.1.1): the grouped GEMM takes the routing's ROW MAP (`toks`/`offs` from
    #: `gather_abi`) instead of a materialised `[R, d]` copy, so the kernel's A
    #: load is the only read of `x`'s rows. `off` is every number on record.
    #: `kernel` is a SECOND switch on the same pass — `moe_kernel` names the arm
    #: that runs when the gather arm cannot (COUNTED with its reason, the same
    #: way the triton arm falls back). Refused LOUDLY off the route it is gated
    #: on: `bf16=false`, `use_byteflow=true`, `moe_topk_mode != "off"` (the
    #: adaptive width moves unused slots to a sentinel expert past the bank) and
    #: `factor_qat != "none"` (the kernel reads the dequantized `gate_u`/`up_u`/
    #: `v`). Its BACKWARD is the shipped grouped adjoint, recomputed on the same
    #: rows — `moe_sonic.gather_glu` says what that costs and why. No speed claim
    #: exists until `scripts/bench_gather.py` runs.
    moe_gather: str = "off"
    #: The fused ROUTE HEAD (`model/route_fused.py`): ONE kernel for the
    #: unbiased softmax, the LFB bias add, the top-k and the gathered gate,
    #: instead of 4 tensor ops over the same `[n, E]` (topk alone is 5-6 CUDA
    #: kernels on this stack). Same result: the selection reads the RAISED
    #: logits (`logits + lfb_bias`), the weight is the unbiased softmax at the
    #: chosen ids, and among equal raised scores the LOWEST expert id wins (the
    #: one rule `torch.topk` leaves unspecified — `route_fused`'s docstring).
    #: OFF, and the OFF path's bits are untouched. Refused LOUDLY with
    #: `moe_topk = 0` (the dense blend has no top-k), `moe_topk_mode != "off"`
    #: (the adaptive width zeroes padding slots and keeps `probs`) and
    #: `moe_lb_mode = "aux"` (its term reads `probs`); a call the kernel cannot
    #: serve is COUNTED (`moe_route_fused_fallback` + reason) and runs the old
    #: path. Counters: `moe_route_fused_calls`, `moe_route_fused_fallback`.
    moe_route_fused: bool = False
    #: The permute's algorithm: `argsort` is today's `torch.argsort(stable=True)`
    #: (a CUB radix sort, ~12 kernels at n·k = 16 384) and `count` is the stable
    #: counting sort in `model/route_fused.py` (bincount + cumsum + one program
    #: per expert). The PERMUTATION IS IDENTICAL — stable within an expert, which
    #: the bank rows and the drop set depend on — so this is a cost knob, not a
    #: semantics one. Counters: `moe_permute_count`, `moe_permute_fallback`.
    moe_permute: str = "argsort"
    #: HOW the routed scatter closes (`MOE_COMBINES`). `torch` is today's
    #: `zeros + y.to(f32)·w + index_add_` — 1 476 MB and 4 launches per pass at
    #: m400, 81 % of the routing cluster's traffic. `fused` is ONE Triton pass
    #: that reads each token's k rows and writes its output, accumulating in
    #: fp32 registers: 671 MB / 2 launches at `moe_combine_dtype="fp32"` —
    #: bit-identical, because with k=2 the two fp32 adds are commutative. OFF.
    moe_combine: str = "torch"
    #: The dtype the fused combine stores (`MOE_COMBINE_DTYPES`): `fp32` keeps
    #: today's boundary; `bf16` removes the caller's `weighted.to(dt)` cast too
    #: and rounds the routed activation ONCE (2^-9 relative — the measured
    #: logits/gradient deltas are the gate's, not a claim). Refused unless
    #: `moe_combine="fused"` and `bf16=true`.
    moe_combine_dtype: str = "fp32"
    #: The bank cache's KEY: `False` is today's `_version` tuple (a
    #: 1 536-parameter walk per forward, measured 2.09 ms/call at m400);
    #: `True` is one explicit `bank_epoch` int bumped by the trainer after
    #: `opt.step` (`invalidate_banks_all`), by any `load_state_dict`, and by a
    #: checkpoint resume. A stale epoch is never served (`_banks` rebuilds) and
    #: an entry without an epoch, or from the future, is LOUD. OFF.
    moe_bank_epoch: bool = False
    #: PACKED EXPERT BANK (`experts.SparseMoE`): ONE `[E, rank, d_ffn]` master
    #: for the experts' write `v` (and, when `share_u_mode="off"`, one
    #: `[E, d_model, rank]` per input site) instead of E separate
    #: `nn.Parameter`s. The bank rebuild (`_banks`, once per optimizer step)
    #: stops running one `.to(dt)` per expert plus a stack: it casts the packed
    #: buffer ONCE and returns slices, measured `2·E + 3 -> 3` launches at
    #: E=512 — the 519-launch/step rebuild the 2026-11-05 census named.
    #:
    #: The scales (`gate_s`/`up_s`) stay one `[E, 2, rank]` buffer; the bank
    #: returns the two `[E, rank]` views (no copies, no stacks).
    #:
    #: The optimizer sees ordinary parameters: `packed_v`/`packed_gate_u`/
    #: `packed_up_u` are 3-D BATCHES of 2-D matrices, and `train.optim`'s
    #: `is_muon_param`/`Muon._build_groups` treat the leading dim as the batch
    #: (the Newton-Schulz chain and ColRow already have a per-slice 3-D form).
    #: `packed_s` is a scale tensor and goes to AdamW like every other `s`.
    #:
    #: A checkpoint stays readable both ways: `_load_from_state_dict` expands a
    #: packed state dict into per-expert keys for an unpacked model, and gathers
    #: the per-expert keys into the packed buffers for a packed model; the
    #: round-trip gate pins both directions (`tests/test_tsct_packed.py`).
    #:
    #: REFUSED with a storage rung (`quant != "none"`): `is_factor_key` names
    #: the per-expert `.v` leaf, and a packed `packed_v` would be written
    #: UNCOMPRESSED beside factors the rung compresses — a checkpoint that is
    #: half-packed and does not say so. REFUSED with `use_byteflow`, which
    #: builds no expert bank (the switch would be inert while the snapshot
    #: reported it on). OFF by default and bit-identical when off: no parameter
    #: is replaced, no RNG is drawn and no state-dict key appears.
    #: COMPOSES WITH TSCT-2: with `tsct_expert_basis` on, the shared plane is
    #: still folded per expert (`gate_u @ w_j.T`) — the selector is not a packed
    #: site — while the packed `v`/`packed_s` supply the write and the scales.
    #: The combined bank is `torch.equal` to the unpacked arm's
    #: (`tests/test_tsct_merge.py`); packed + `tsct_expert_lr` scales each
    #: expert row of `packed_v` exactly as it scales an unpacked `v`.
    tsct_packed: bool = False
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
    #: `train/loop.py` prints it at start. Inert (LOUD) without that arm — on
    #: EITHER path (`factor_qat` or `dense_qat`).
    factor_qat_seed: int = 0
    #: What the forward does to every DENSE weight master — the plain
    #: `nn.Linear`/`DenseLinear` path the factor rung does not reach (owner
    #: decision, 2026-10-18: the next training run trains in the FINAL
    #: QUANTIZED FORM, so `factor_qat = "ternary"` and `dense_qat = "ternary"`
    #: TOGETHER describe a model whose whole weight set is ternary in the
    #: forward). `tsct.dense_qat_project` IS `factor_qat_project` — the same
    #: rungs, the same annealed STE (`w + alpha·(q(w) − w).detach()`), the same
    #: per-tensor axis — so the two switches cannot drift. `none` (the default)
    #: is the master untouched, every number on record. Covered sites and the
    #: two deliberate exclusions (embeddings/norms/tables/biases — the BitNet
    #: convention) are listed at `DENSE_QATS`; `use_byteflow` is refused with a
    #: non-`none` value (its net's projections are not on this seam).
    dense_qat: str = "none"
    #: Keep ATTENTION weights out of `dense_qat` — the BitNet b1.58 rule, and
    #: the default. `tsct.DENSE_QAT_ATTN_SITES` names the ten sites (KDA's
    #: q/k/v/g/beta and the MSA pair + its selector): b1.58 (arXiv:2402.17764)
    #: quantizes the FFN weight matrices, not attention, and our own
    #: `act_quant` already upgrades the attention path to >= 8 bits. Measured
    #: 2026-10-21 on medium b8 s512, 2000 steps, one 81 920 B eval window, 3
    #: seeds: with those ten sites projected, `factor_qat=ternary` +
    #: `dense_qat=ternary` scored BPB 3.2733 against the bf16 control's 3.0750
    #: (+0.198, +6.5%). `true` passes `kind="none"` at those sites; `false`
    #: restores that measured run's behaviour and is what the A/B arm uses.
    dense_qat_skip_attn: bool = True
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
    #: ---- z-loss (PaLM §5 / Gemma-2 §9: the softmax's partition function) ----
    #: `loss = CE + z_loss · mean_i (logsumexp(logits_i))²`. The squared
    #: log-partition keeps the pre-softmax logits from drifting tall, which is
    #: the stability term the PaLM/Gemma recipes carry; it is a REGULARISER, so
    #: `pure=True` (the held-out number) skips it exactly as it skips every
    #: aux term. `0.0` (the default) adds no op, no parameter and no counter,
    #: and the off path is bit-identical.
    #:
    #: The term reaches the FINAL `lm_head` logits only (the `norm`+head exit),
    #: not the loop's per-iteration deep-supervision terms: those are a
    #: training-only readout of the same state, and a second z-term there would
    #: be a different objective than the paper's. REFUSED with `use_byteflow`
    #: (both channel shapes return `_byte_ce` and never build this term, so the
    #: flag would be inert). Negative values are refused.
    z_loss: float = 0.0
    #: ---- Gemma-2 logit soft-cap (`logit_softcap`) ----
    #: `logits <- cap * tanh(logits / cap)` on the LM head output, applied at
    #: the two places the head trains against CE (the loop's per-iteration
    #: readout and the final `norm`+`lm_head` exit) and therefore to the eval
    #: objective, which runs those same two paths. Every logit is bounded to
    #: `(-cap, cap)`, so every margin (`losses.ce_margin`) is bounded by `2·cap`
    #: — 30 nats at `cap=15`, against the fp32 softmax-collapse cliff at 16.6.
    #: The claim is a BOUNDED logit range (unbounded drift is what it removes),
    #: not that 15 makes the cliff unreachable: the worst-case margin is still
    #: above it. `0.0` (the default) enters no branch and is bit-identical to
    #: the pre-flag model.
    #:
    #: COMPOSES with `ce_fixed` (the cap is applied to the logits BEFORE the
    #: fixed loss sees them, so its margin arithmetic is computed on capped
    #: values) and is REFUSED with `use_byteflow`: both byteflow shapes return
    #: `_byte_ce` on patch latents and never reach this head, so the flag
    #: would be silently inert. Also REFUSED with `mtp_weight > 0`: MTP calls
    #: the shared head itself and its CE would read uncapped logits beside the
    #: model's capped ones. Negative values are refused.
    logit_softcap: float = 0.0
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
    #: ---- Multi-Token Prediction (DeepSeek-V3 §3.2, `model/mtp.py`) ----
    #: `mtp_k` = K future-byte modules that share THIS model's embedding and
    #: `lm_head` (the only new parameters are the K `Linear(2*d, d)` matrices);
    #: `mtp_weight` = λ on their uniform-mean CE. `0` / `0.0` (the default)
    #: builds nothing, so every preset's parameter count is unchanged. Refused
    #: loudly: `mtp_weight > 0` with `mtp_k == 0`, `mtp_k` outside `0..=8`, and
    #: `use_byteflow` (the channel has no byte ids for the future-byte
    #: embedding, the same refusal DSpark carries). Prior evidence: the
    #: `aux_fb` future-byte head Tied at 3 seeds at 9.2M and was deleted in
    #: `39d0485`; the published wins are large-scale, so the A/B decides here.
    mtp_k: int = 0
    mtp_weight: float = 0.0
    #: ---- Soft-MTP (arXiv:2610.38331, Teoh 2026) ----
    #: Route the NTP CE's gradient at the head's ALIGNED position into each MTP
    #: module, so the module trains against the model's own next-byte
    #: distribution instead of the one-hot byte. Straight-through on the NTP
    #: hidden: `ntp_hidden[:, k:] += s_k · (x_k.detach() - x_k)`, a forward
    #: no-op whose backward adds `-s_k · g_CE_ntp` to the module's own CE
    #: gradient. `s_k` is the reference's `kl_recovery_scale` with THIS repo's
    #: total-loss factor (`1/(T+1)`) and per-head means made explicit
    #: (`model/mtp.py`). OFF by default and bit-identical to the hard path when
    #: off. REFUSED with `mtp_weight == 0` (there is no MTP hidden to route
    #: into) and with `logit_softcap > 0` (below).
    mtp_soft: bool = False
    #: ---- the reference's per-head seam (`mtp.py`) ----
    #: `mtp_head_norm=true` adds a per-head final `RMSNorm` (WITH a learnable
    #: gain, matching the reference's `RMSNorm`; `K·d` parameters) between the
    #: MTP projection and the shared head. Each head's normalized hidden is
    #: read ONLY by that head's CE/coupling; the next module still consumes the
    #: raw projection output. The seam is what makes the soft-MTP KL identity
    #: exact at EVERY head instead of only the last, so `mtp_soft=true`
    #: REQUIRES it (LOUD otherwise). OFF by default: no module, no parameter,
    #: and the hard path is bit-identical to the pre-seam model.
    mtp_head_norm: bool = False
    #: SIGReg / LeJEPA (arXiv:2511.08544, `model/sigreg.py`): a REGULARISER, not
    #: a head. `loss = CE + sigreg_weight · sigreg_loss(z)`, `z` the strided
    #: sample of the state `lm_head` reads — no teacher, no predictor, no mask,
    #: no second forward, no parameters, no detach. 0.0 = off. The research
    #: doc's ONE arm to run first (`docs/research/2026-10-16-jepa-into-dormouse.md`
    #: §3; the burn `l1 + 0.1·koleo` bundle cannot attribute its own 0.082 BPB).
    #: The suggested grid is 0.005 / 0.05 / 0.1 — publish which.
    sigreg_weight: float = 0.0
    #: ---- JEPA as ONE mechanism (`model/jepa.py`): a latent predictor + its
    #: OWN anti-collapse term, never a separate gadget. The arm switch, kept as
    #: a bool so every arm reads the same way; it MUST agree with the weight
    #: (`__post_init__`): `jepa` is true iff `jepa_weight > 0`.
    jepa: bool = False
    #: λ on the whole JEPA term: `CE + jepa_weight · (cosine + reg)`. `0.0`
    #: builds no head, adds no parameter and leaves every number bit-identical.
    jepa_weight: float = 0.0
    #: The causal horizon `h`: predict `h[p+h]` from `h[p]` in the SAME forward
    #: (`1` = the next position). Refused below 1 and above `seq_len // 8` (the
    #: horizon must fit the window with room for a sample; the load path owns
    #: the `seq_len` half because a preset does not know it).
    jepa_horizon: int = 1
    #: The anti-collapse weight INSIDE the JEPA term: SIGReg on the predicted
    #: latents. It has its own weight so the mechanism is self-contained —
    #: `jepa_weight` scales the sum, never the two terms differently. `0.1` is
    #: the value `tests/test_jepa.py`'s collapse gate justifies (the predicted
    #: latents stay spread at 0.1 and collapse at 0.0).
    jepa_reg_weight: float = 0.1
    #: ---- energy-based scoring (EBT, arXiv:2507.02092; `model/energy.py`) ----
    #: `energy_weight` = λ on the margin-InfoNCE term: the target byte's energy
    #: must beat `energy_negs` negatives sampled from the `lm_head` softmax by
    #: `energy.MARGIN`. `0.0` (the default) builds NO head, adds no parameter
    #: and adds no compute, so every preset and snapshot is bit-identical.
    #:
    #: A 256-byte vocabulary is ENUMERABLE, which is why this is a contrastive
    #: head and not EBT's MCMC: scoring all candidates is one matmul, and the
    #: same head is the INFERENCE reranker (`scripts/rerank.py`). A quality
    #: claim needs that table AND `scripts/ab_energy.py`; neither has run.
    #:
    #: REFUSED loudly with `use_byteflow` (the channel runs the loop over K
    #: patch latents and has no byte position for the sampler) and with
    #: `dspark_weight > 0` (two aux objectives reading the same hidden state is
    #: not measured here — pick one arm per run).
    energy_weight: float = 0.0
    #: Negatives drawn per position from the `lm_head` softmax (target removed).
    #: `64` covers a quarter of the 256-byte vocabulary per position.
    energy_negs: int = 64
    #: InfoNCE temperature τ: lower is sharper (the margin is divided by it too).
    energy_temp: float = 1.0
    #: The head's bottleneck `k` in `Linear(d, k) -> SiLU -> Linear(k, 1)`.
    energy_rank: int = 64
    #: ---- the compressed, selectable KV memory (CSA) beside KDA ----
    #: `model/csa.py`. Per `csa_chunk` byte positions ONE shared key and value (`csa_kv_dim`
    #: wide, MQA), and each query attends exactly over the top `csa_topk`
    #: entries of the STRICTLY EARLIER chunks — the per-token detail KDA's
    #: O(1) state cannot hold, at `2·csa_kv_dim·2 / csa_chunk` bytes per context
    #: byte. ADDITIVE beside KDA behind a per-iteration gate and off by default:
    #: `use_csa=false` builds no module and no parameter. Chunk ∈ {4,8,16,32},
    #: width ∈ {64,128}. REFUSED loudly with `use_msa` (two sparse branches is
    #: not measured), `segment_len>0` (the boundary rule for a half-full chunk
    #: is undecided), `use_byteflow` (its loop runs over patch latents, so a
    #: byte chunk would silently be a latent chunk), `use_kda=false` and
    #: `attn_kind='full'` (this arm is specified beside KDA).
    use_csa: bool = False
    csa_chunk: int = 8
    csa_kv_dim: int = 64
    csa_topk: int = 32
    #: ---- the COARSE, DENSE second level (HCA) of the same arm ----
    #: `csa_coarse_chunk=0` (OFF) keeps the one-level arm bit-identical. When
    #: on (64/128/256/512, a multiple of `csa_chunk`), `csa_coarse_kv_dim`
    #: (64/128) wide keys/values are mean-pooled per coarse chunk and EVERY
    #: query attends densely over all strictly earlier coarse entries. The two
    #: outputs are summed and pass through the ONE shared `csa_gate` per
    #: iteration slot. This is the level that makes 1M decode affordable: at
    #: 256 bytes an entry, 4096 dense entries per 1M bytes (0.52M MAC/byte at
    #: kv=64, 1.0 MiB of state) against the fine level's 131 072-entry scan
    #: (8.4M MAC/byte). REFUSED loudly with `use_msa`, `segment_len>0`, a
    #: non-multiple of `csa_chunk`, or without `use_csa`.
    csa_coarse_chunk: int = 0
    csa_coarse_kv_dim: int = 64
    #: The fine level's off switch. `false` runs the coarse level ALONE (the
    #: exact-attention parity gate's configuration); refused without a coarse
    #: level, because it would build a branch that computes nothing.
    csa_fine: bool = True
    #: ---- the FUSED fine-selection path (off by default) ----
    #: `false` is the reference chain, statement for statement. `true` collapses
    #: the fine selection's score → top-k → gather → softmax → weighted sum into
    #: fewer dispatches (`_select_fused`: top-k's own values replace the score
    #: gather and supply the mask as `values > finfo.min`, masking before the
    #: softmax removes the renormalisation, and one batched matmul replaces the
    #: multiply-plus-reduce), and hoists the per-chunk ring append in
    #: `forward_state` into one bulk write per level (`append_many`).
    #: Arithmetic is the same; the parity is `1e-6` (fp32) against the reference,
    #: NOT bit-identity — the op order in the weighted sum changes. Refused
    #: without `use_csa` and with `csa_fine=false` (nothing would run).
    csa_fused: bool = False
    #: ---- the THIRD level (the MIDDLE one in cost, the OUTERMOST by chunk) ----
    #: `csa_mid_chunk=0` (OFF) keeps the two-level arm bit-identical. When on it
    #: must be a MULTIPLE of `csa_coarse_chunk` (of `csa_chunk` when coarse is
    #: off) and at least `csa_chunk`: one mid entry is a whole number of
    #: contained entries, so the three levels pool the same byte boundaries and
    #: `csa_hierarchical` can narrow selection by containment. Same algebra as
    #: the coarse level (mean-pool, DENSE causal mask over strictly earlier
    #: chunks, its own `W_{q,k,v,o}` at `csa_kv_dim`), its own ring in the
    #: `(fine, coarse, mid)` state, summed into the same shared `csa_gate`.
    #: REFUSED loudly without `use_csa`, if it is not a multiple of the level it
    #: contains, or if it is smaller than `csa_chunk` (a "middle" level finer
    #: than the finest one selects nothing new).
    csa_mid_chunk: int = 0
    #: ---- hierarchical selection for the FINE level ----
    #: `false` (the default) is today's flat selection: every query scores ALL
    #: earlier fine chunks. `true` scores the OUTERMOST level first (the largest
    #: chunk — the mid level, then the coarse one), keeps its top `csa_topk`,
    #: restricts the next level's candidates to the survivors, and only then
    #: scores the fine chunks inside the surviving coarse entries — so the fine
    #: attention runs over `csa_topk` survivors instead of `t / csa_chunk` keys.
    #: Requires `csa_mid_chunk > 0` (without a third level the "hierarchy" is a
    #: rename of the flat scan) and `csa_fine=true` (it narrows the fine scan).
    #: `false` keeps every pinned state_dict/logits hash byte-identical; the
    #: counters name the survivors per level (`csa_sel_coarse`/`_mid`/`_fine`).
    csa_hierarchical: bool = False
    #: ---- the second attention arm (MSA re-entry, ADR-0014; owner 2026-10-16) ----
    #: 1M-context sparse attention BESIDE KDA (which stays the default arm).
    #: `msa_capacity_ratio` = the arm's share of the attention demand (owner:
    #: 1/4 of KDA's); `adaptive_msa` = the model decides WHERE that 1/4 goes
    #: (per-iteration learned share, mean = the ratio; the `msa_gate` blend).
    #: Force the memory-efficient SDPA kernel instead of whatever PyTorch picks.
    #: Measured on the RTX 5060 Ti (sm_120, torch 2.13.0+cu129), b8 h16 t1024
    #: hd64 bf16 causal: flash 1.91 ms, EFFICIENT 0.91 ms -- PyTorch's default
    #: (flash) is 2.1x SLOWER here. Default False so every number on record keeps
    #: the backend it ran with; a run that turns it on records `sdpa_mem_efficient`.
    sdpa_mem_efficient: bool = False
    use_msa: bool = False
    msa_topk: int = 32
    msa_block: int = 4
    msa_capacity_ratio: float = 0.25
    adaptive_msa: bool = False
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
    #: group of `act_group` columns — OUR convention (the a4.8 paper has no
    #: groups; see `model/act_quant.py`'s table and
    #: `docs/research/2026-10-18-act4.md` §3.1). Must divide `d_model`,
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

    def __post_init__(self) -> None:
        """Resolve the `share_u` alias — ONE switch, two spellings, never two.

        `share_u = true` IS `share_u_mode = "plane"`: it is the boolean the
        variable existed as for one day, every `--set share_u=...` and every
        snapshot on disk spells it that way, and a second mode field must not
        turn that into a config where two keys disagree about the model. So the
        alias wins over the default and only a CONFLICTING mode is refused —
        the resolution happens here, the one place every construction passes
        (`parse_preset`, `--set` through `dataclasses.replace`, tests).
        """
        if self.jepa != (self.jepa_weight > 0.0):
            loud(
                f"jepa={self.jepa} with jepa_weight={self.jepa_weight}: the switch "
                f"and the weight must agree — `jepa` is true iff `jepa_weight > 0`. "
                f"A switch with a zero weight builds no head (the arm would be "
                f"inert), and a weight with the switch off is a key that lies about "
                f"what the run computes (model/jepa.py). Escape: `jepa=true` + a "
                f"positive `jepa_weight`, or `jepa=false` + `jepa_weight=0.0`."
            )
        if self.share_u_mode not in SHARE_U_MODES:
            loud(
                f"share_u_mode={self.share_u_mode!r} is not an arm. Known: "
                f"{list(SHARE_U_MODES)} — `off` is a per-expert `U`, `plane` one "
                f"basis for the whole bank, `dict` one dictionary plus a "
                f"per-expert column subset (`experts.py`). Escape: name one of "
                f"those, or land the mode before the config can lie about it."
            )
        if self.share_u:
            if self.share_u_mode not in ("off", "plane"):
                loud(
                    f"share_u=true and share_u_mode={self.share_u_mode!r}: two "
                    f"switches, two models. `share_u` is the old spelling of "
                    f"`share_u_mode='plane'`, so it cannot also ask for the shared "
                    f"dictionary. Escape: drop `share_u` (leave it false) and keep "
                    f"`share_u_mode`, or set `share_u_mode='plane'`."
                )
            self.share_u_mode = "plane"
        if self.tsct_packed and self.quant != "none":
            loud(
                f"tsct_packed=true with quant={self.quant!r}: the storage rung packs "
                f"the per-expert `.v` factor (`is_factor_key`), and the packed "
                f"`packed_v` buffer is not a factor key — the checkpoint would carry "
                f"an uncompressed bank beside compressed factors and say so nowhere. "
                f"Escape: quant='none', or tsct_packed=false."
            )
        if self.tsct_packed and self.use_byteflow:
            loud(
                "tsct_packed=true with use_byteflow=true: the byteflow shapes build "
                "no expert bank (their loop runs over patch latents), so the packed "
                "buffers would not exist while the snapshot reported the flag on. "
                "Escape: use_byteflow=false, or tsct_packed=false."
            )
        for name in ("prelude_layers", "coda_layers"):
            n = getattr(self, name)
            if not 0 <= n <= 4:
                loud(
                    f"{name}={n} outside 0..=4: the prelude/coda topology ships 0-4 "
                    f"UNIQUE layers per side (arXiv 2610.23036, model/prelude.py), "
                    f"and a longer stack is a model nobody priced — each layer is "
                    f"another 3·d·d/2 parameters. Escape: 0..=4."
                )
        if self.z_loss < 0.0:
            loud(
                f"z_loss={self.z_loss}: the term is `+ coef · mean(logsumexp²)`, a "
                f"POSITIVE stability penalty on the logits. A negative value would "
                f"pay the model to make the partition function tall, which is the "
                f"opposite of the PaLM/Gemma recipe and has no gate. Escape: "
                f"z_loss >= 0, or 0.0 to leave the objective alone."
            )
        if self.tie_embeddings and self.use_byteflow:
            loud(
                "tie_embeddings=true with use_byteflow=true: both byteflow shapes "
                "leave this model's `lm_head` behind (Option A has its own byte head; "
                "Option B decodes with `decode_chunks`), so there is no head to tie "
                "and the flag would be inert while the snapshot reported it on. "
                "Escape: tie_embeddings=false, or use_byteflow=false."
            )
        if self.z_loss > 0.0 and self.use_byteflow:
            loud(
                f"z_loss={self.z_loss} with use_byteflow=true: both byteflow shapes "
                f"compute `_byte_ce` and never reach the `norm`+`lm_head` exit this "
                f"term is defined on, so the regulariser would be silently absent. "
                f"Escape: z_loss=0.0, or use_byteflow=false."
            )
        if self.logit_softcap < 0.0:
            loud(
                f"logit_softcap={self.logit_softcap}: the Gemma-2 cap is "
                f"`cap * tanh(logits / cap)`, so a negative cap reflects the logits "
                f"and a zero cap divides by zero. Escape: logit_softcap > 0, or 0.0 "
                f"to leave the head untouched."
            )
        if self.logit_softcap > 0.0 and self.use_byteflow:
            loud(
                f"logit_softcap={self.logit_softcap} with use_byteflow=true: both "
                f"byteflow shapes return `_byte_ce` from their own patch-latent head "
                f"and never reach `norm`+`lm_head`, so the cap would be silently "
                f"inert while the snapshot reported it on. Escape: "
                f"logit_softcap=0.0, or use_byteflow=false."
            )
        if self.mtp_soft and not self.mtp_head_norm:
            loud(
                "mtp_soft=true with mtp_head_norm=false: the KL gradient-recovery "
                "identity is exact only when each MTP head reads its OWN "
                "normalized hidden. Without the seam `x_j` is also module "
                "j+1's input, so a later head's coupling back-propagates through "
                "the sequential projection and the identity holds only at the "
                "last head (measured rel ~0.79 at K=3 head 0, fp64). Escape: "
                "mtp_head_norm=true (the reference's per-head final RMSNorm), or "
                "mtp_soft=false."
            )
        if self.mtp_soft and self.logit_softcap > 0.0:
            loud(
                f"mtp_soft=true with logit_softcap={self.logit_softcap}: the "
                f"recovery identity equates the NTP head's distribution with the "
                f"MTP head's, and the soft path's verifier would be the CAPPED "
                f"`lm_head` output while `model/mtp.py` reads the SAME head "
                f"UNCAPPED — two different distributions are not a self-"
                f"distillation target, and the routed gradient would be off by the "
                f"cap's Jacobian. Escape: logit_softcap=0.0, or mtp_soft=false."
            )
        if self.logit_softcap > 0.0 and self.mtp_weight > 0.0:
            loud(
                f"logit_softcap={self.logit_softcap} with mtp_weight={self.mtp_weight}: "
                f"MTP calls the shared `lm_head` itself (`model/mtp.py`) and its CE "
                f"would read UNCAPPED logits while the model's two CE sites read "
                f"capped ones — two logit scales in one objective, exactly the "
                f"silent inconsistency this flag must not create. Escape: "
                f"mtp_weight=0.0, or logit_softcap=0.0."
            )
        # The loop's DISTINCT-body count. Both bounds are structural, not
        # measured: the loop selects `i % loop_blocks`, so 1 is the bottom of
        # the index, and a count above `max_iter` allocates a block that no
        # iteration can reach (the config-lies defect — parameters with no
        # forward and no gradient). `max_iter` moves freely, so the pair is
        # checked here where both are known.
        if self.loop_blocks < 1:
            loud(
                f"loop_blocks={self.loop_blocks}: the loop cycles over "
                f"`blocks[i % loop_blocks]`, so the count is >= 1 (`1` is today's "
                f"weight-shared block). Escape: 1..=max_iter={self.max_iter}."
            )
        if self.loop_blocks > self.max_iter:
            loud(
                f"loop_blocks={self.loop_blocks} > max_iter={self.max_iter}: block "
                f"{self.max_iter} would never be selected (the loop index is "
                f"`i % loop_blocks` with i < max_iter), so it would be a parameter "
                f"tree with no forward and no gradient — a config that lies about "
                f"the model. Escape: loop_blocks <= max_iter, or raise max_iter."
            )


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
    #: ---- AGC: adaptive gradient clipping (arXiv:2102.06171 §2.2) ----
    #: Per-parameter scaling `scale = λ · ||p|| / max(||g||, 1e-3)` clamped to
    #: `<= 1.0`, applied to `.grad` BEFORE the global `grad_clip` and after
    #: every micro-batch of `grad_accum` (one optimizer stage, one AGC pass).
    #: `0.0` (the default) enters no branch and is bit-identical to the
    #: pre-flag trainer. The mechanism, its exact variant and the paper's
    #: `λ·||W||²` threshold are documented at `train/optim.py::agc_clip`.
    #: Negative values are refused. It is device-only and adds no host sync.
    agc_clip: float = 0.0
    #: ---- AdamW's denominator floor (`MixOptimizer(eps=...)`) ----
    #: The equilibrium logit scale is bounded by about `log(1/eps)`: 1e-8 (the
    #: default, torch's own) allows ~18.4 nats while the fp32 softmax-collapse
    #: cliff is 16.6 (`losses.CLIFF`), so raising eps to 1e-6 lowers that
    #: ceiling to ~13.8 and is the sourced mechanism this knob tests — not a
    #: tuning guess. It reaches `torch.optim.AdamW`'s AdamW half only; Muon has
    #: no denominator. Must be > 0. Non-default values are snapshot-checked.
    adam_eps: float = 1e-8
    #: THE FIXED CROSS-ENTROPY (arXiv:2610.31642 §2): the loss keeps the
    #: target logit out of the `1 + e^{-m}` denominator, so the target
    #: gradient does not round to exactly 0 past the dtype's precision cliff
    #: (16.6 nats in fp32, 5.5 in bf16) and `sum_c dL/dlogit_c == 0` at every
    #: margin. `false` (the default, every number on record) is the historical
    #: `log_softmax → NLL` path BIT FOR BIT — the flag is a branch, not a
    #: rewrite. Honoured by `train`, `sft_train` and `rl_train`; the log line
    #: carries `ce_cliff=N`, the DEVICE-side count of tokens whose margin
    #: passed the cliff, so a real run reports the prevalence the fix acts on.
    #: The loss's arithmetic is fp32 on every path (the head is fp32 by the
    #: burn rule), which is why the live cliff is 16.6 and not bf16's 5.5.
    #: No refusal against `mm_dtype=fp8`: that names the 2-D GEMM's compute
    #: mode, not the logits the loss reads.
    ce_fixed: bool = False
    #: GRADIENT ACCUMULATION: micro-batch count per optimizer step. `1` (the
    #: default and every number on record) runs one forward/backward per step.
    #: `k > 1` runs `k` micro-batches of `batch` rows, divides EACH micro-loss
    #: by `k` before backward (so the accumulated gradient is the MEAN over the
    #: micro-batches, not their sum), and calls the optimizer ONCE per group;
    #: `step` counts optimizer steps, so the LR schedule and every eval/checkpoint
    #: cadence tick once per group and never per micro-batch. The data order
    #: matches a `batch·k` run exactly: `ByteStream.next_batch` returns
    #: consecutive `batch·(seq_len+1)`-byte windows, so `k` reads concatenate to
    #: the one wide window, and the gate holds the two to 1e-5 (fp reduction
    #: order is the only difference; a same-forward `[b·k, t]` batch is
    #: independent per row, so the math is the same mean). A CAPACITY-PADDED
    #: MoE (`moe_capacity > 0`) is not row-separable — the dispatch drops rows
    #: as a function of the forward's OTHER rows — so the equivalence cannot
    #: exist there for either side; the gate runs dropless (`moe_capacity=0`,
    #: the A/B battery's own COMMON).
    #:
    #: Interactions, each decided here rather than discovered:
    #: * `opt_graph` is REFUSED with `k > 1` (LOUD in `load_run`): the graph
    #:   captures ONE optimizer stage, and the arm this field names is the
    #:   hosting of k backward passes around that stage — a capture of a stage
    #:   whose gradient buffers are filled by a variable number of standing
    #:   eager passes is not the stage the graph was verified for;
    #: * `halt` and `segment_len` are ALLOWED: each micro-batch is a full
    #:   scheduled-depth training forward, the loss is a mean over positions,
    #:   and the mean of k micro-means is the wide batch's mean — the halt target
    #:   is per-forward by construction and segmentation is memory slicing of
    #:   one window, so neither changes meaning under accumulation;
    #: * `rand_depth` is ALLOWED and samples ONCE per optimizer step (all
    #:   micro-batches share the step's depth, matching one wide forward);
    #: * `engram_ram` is ALLOWED: the host rows accumulate every pending leaf
    #:   across the group and the host update runs once, after the group, which
    #:   is exactly the old "once per step" contract (the cost is pending-row
    #:   memory for `k` forwards).
    #: The NaN firewall treats a group as skipped if ANY micro-batch is
    #: non-finite: its contribution is sanitized on arrival, the group's shared
    #: flag ANDs the per-micro flags, and one multiply at the end zeroes every
    #: accumulated gradient. `nan_skipped` counts GROUPS, not forwards.
    grad_accum: int = 1
    #: TF32 for the remaining fp32 matmuls (the Muon Newton-Schulz iterations,
    #: the KDA internals, the fp32 lm_head path): measured 2026-10-16 at b1
    #: b8x1024 as 1399 -> 1201 ms/step (+16%, mostly the optimizer's NS bmm).
    #: OFF by default until a loss-parity check names its cost; NS is already
    #: an approximate iteration, so the expected cost is small — that is a
    #: prediction, not a substitute for the check.
    tf32: bool = False
    #: FLA chunk_kda's `disable_recompute` (`model/kda.py`'s setter, called once
    #: by `train/loop.py` beside `tf32`/`mm_dtype`). `False` (the default and
    #: every number on record) = the kernel recomputes its backward
    #: intermediates; `True` SAVES them instead — FLA's docstring: "beneficial
    #: for training small models at the cost of increased memory usage". A
    #: speed/VRAM trade decided by a bench, never by hope.
    kda_disable_recompute: bool = False
    #: The 2-D GEMM COMPUTE precision (`MM_DTYPES`, `model/tsct.py` where it
    #: executes — `tsct.set_mm_dtype` is called once by `train/loop.py`, the same
    #: shape as the `tf32` knob above). `bf16` (the default) is every number on
    #: record: fp32 masters, bf16 tensor-core GEMM. `fp8` runs `mm_mixed`'s
    #: products through `torch._scaled_mm` (e4m3 × e4m3 → bf16 out, per-tensor
    #: amax scales off the bf16 operand): the spike's 1.6-1.8× on 2-D GEMMs,
    #: measured on this card (`bench_f_fp8.py`: [32768,2048]×[2048→64] 23.8 →
    #: 39.3 TFLOP/s, [32768,64]×[64→5461] 14.2 → 22.3). It is a COMPUTE rung:
    #: masters, checkpoint and optimizer state do not move.
    #:
    #: **Two facts a reader must know before reading a number off it**, both
    #: measured on torch 2.13.0+cu129 (2026-10-18, CPU + the spike's GPU runs):
    #: (1) the lane is FORWARD-ONLY — `_scaled_mm` has no autograd formula
    #: (`RuntimeError: derivative for aten::_scaled_mm is not implemented`), so
    #: every forward whose operands require grad falls back to bf16 COUNTED
    #: (`mm_fp8_fallback` on the log line, the reason on `counters.why`); the
    #: trainer prints the same fact at start. A TRAINING run is therefore bf16
    #: compute, while its own no-grad eval forwards are fp8 — the flag's meaning
    #: until a custom backward lands. The mode is PROCESS-wide, set by `train()`
    #: (the `tf32` shape), so a standalone `dormouse eval`/`generate` process
    #: starts at the default `bf16`: that default IS the precision this run's
    #: training step ran, so the standalone eval is the training-precision
    #: measurement rather than a mismatch.
    #: (2) `_scaled_mm` is a 2-D op and e4m3 needs K and N divisible by 16, so a
    #: shape outside that (the medium router's N=3, its 2730-wide down
    #: projection) falls back COUNTED with its reason. Grouped/batched fp8 does
    #: not exist on sm_120.
    #: In the snapshot: a flip is refused on a resume by `diff_keys` (it changes
    #: what the forward computes), and an older snapshot that lacks the key
    #: fills this default and stays a legal resume.
    mm_dtype: str = "bf16"
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
    #: THE EVAL WINDOW, in SCORED BYTES. 196 608 = 24 x 8 x 1024: the burn
    #: flagship's b24 window, and the width that moved the SAME checkpoint
    #: +0.045..+0.050 BPB against the old 65 536
    #: (docs/reviews/2026-11-06-bpb-accounting.md §3). A BPB is only comparable
    #: within one window, so this field is the only source of that width.
    #: REFUSED below 65 536: one 64 KB window has a 95% interval of about
    #: +-0.19 BPB on this corpus — a narrower number is not a measurement.
    eval_bytes: int = 196_608
    #: Legacy explicit window in BATCHES (`eval_batches x batch x seq_len`).
    #: `None` = derive the batch count from `eval_bytes` (the default). A
    #: checkpoint snapshot that carries a number keeps its recorded window:
    #: its curve was measured there and must stay readable — that is also the
    #: deliberate below-the-floor override the tiny CPU fixtures use.
    eval_batches: int | None = None
    #: WHERE the eval window starts, in bytes into the eval file (or into the
    #: eval tail when no separate `eval_data` is set). 0 = the head. It is
    #: printed as `off=` on the eval line: `rewind()` alone makes two evals at
    #: the SAME offset comparable, not two runs at different ones.
    eval_offset: int = 0
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
    #: default goes through the standing 3-seed A/B, not a sentence. It is a
    #: PROGRESS KEY (`PROGRESS_KEYS`): an A/B may change it on a resume, and
    #: `train/loop.py` re-applies it after the checkpoint's optimizer state is
    #: loaded. The measured-parity driver is `scripts/ab_opt_region.py ab`
    #: (`ns5` vs `ns3`), which prints both arms' held-out BPB.
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
    #: relative against the fp32 arm). The SCHEMA default stays `fp32`; the
    #: measured presets override it through `PRESET_TRAIN_DEFAULTS` (m400 and
    #: medium are `bf16` since 2026-11-05 — `docs/reviews/2026-11-05-opt-region.md`
    #: carries the 100-step CPU gate, which is a TOLERANCE gate, not bit
    #: equality). The snapshot records the resolved value, so a live run's
    #: semantics and its resume do not move; an older fp32 snapshot resumes
    #: with `--set muon_state_dtype=fp32` (the strict drift check refuses the
    #: flip loudly rather than silently changing the state).
    muon_state_dtype: str = "fp32"
    #: The lockstep Newton-Schulz batch cap, in fp32 ELEMENTS per stacked
    #: buffer (`optim.MUON_CHUNK_ELEMS` is the schema default, 1<<25 = 33.5M =
    #: 134 MiB per intermediate). It is a MEMORY/launch trade: the chain holds
    #: ~6 temporaries per element alive for the whole batch, so a cap is what
    #: makes b1 fit; at m400/medium the measured cap is 1<<26 (2 chunks per big
    #: group instead of 4, ~203 fewer optimizer launches per step at the m400
    #: shape, ~1.6 GiB peak). Lower it (never to 0) on a card that OOMs in the
    #: optimizer stage. It is a PROGRESS KEY: a chunk split moves no value (the
    #: NS chain is per slice; `tests/test_opt_region.py` gates the bit-equality
    #: across caps), so the only thing a resume changes is the launch count.
    muon_chunk_elems: int = 1 << 25
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

    #: The SF arm's weight decay at the **fast iterate `Z`**, i.e. the
    #: `Z_{t+1} = (1−ηλ)Z_t − η̂P̂_t` term of arXiv:2605.23061v1 Algorithm 1
    #: line 17 — NOT at `Y` and NOT the AdamW group's `wd`. `0.01` is the
    #: port's standing `wd`; the theorem needs only **λ > 0** (Lemma 3.1 is
    #: what lifts the bounded-diameter assumption the SF-Spectral bound
    #: otherwise has to make), so a non-positive value is REFUSED under
    #: `muon_sf=true` rather than run outside the precondition. Inert when
    #: `muon_sf=false`: the control's Muon group carries `wd=0.0` and the
    #: state dict does not move.
    muon_sf_wd: float = 0.01

    #: Whether the per-step TSCT polar retraction runs while `muon_sf` is on.
    #: **`true` is REFUSED**: the retraction pulls each iterate onto the
    #: Stiefel manifold, so the averaged iterate `X` is a convex combination
    #: of retracted iterates and is NOT on it; the SF-Spectral proof's
    #: projection/Lyapunov step (arXiv:2605.23061v1 §3.1, Thm 2.1) needs a
    #: CONVEX set, and the closest-point projection onto Stiefel is not one.
    #: This port also has no `retract_tsct` call site at all (the Rust loop's
    #: per-step retraction was not ported), so `true` could only be an inert
    #: knob here — the S3 gap in `docs/research/2026-10-16-sf-muon.md:82-93`.
    #: The legal setting is therefore `false`, and it is explicit in the run's
    #: own snapshot rather than a fact a reader has to recover from the loop.
    muon_sf_retract: bool = False
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
    #: CUDA-graph capture of the WHOLE training step (`train/graph.py`'s
    #: `FullGraph`): forward + backward + the device firewall + `zero_grad` +
    #: the optimizer stage are recorded once and replayed with ONE launch per
    #: step. It is the direct attack on the measured 13 886 kernel launches per
    #: b24 s1024 step (fwd 4 938, bwd 3 843, opt 2 483 + clip), where the CPU
    #: dispatch time exceeds the GPU time (MFU 2.5%).
    #:
    #: **`false` by default and the default path is byte-identical** (gated:
    #: sha256 of a 5-step `state_dict` + logits from a clean `main` worktree,
    #: `tests/test_full_graph.py`).
    #:
    #: WHAT IT CHANGES, all named here because a reader must know before reading
    #: a number off it:
    #:
    #: * the input batch lives in a STATIC device buffer the loop fills with
    #:   `copy_` (a rebind would leave the recording pointing at the old rows);
    #: * every `.grad` is pre-allocated by the eager warmup and zeroed IN PLACE
    #:   (`set_to_none=False`), so the recording keeps its buffers;
    #: * the LR is a device tensor filled in place (the `opt_graph` seam);
    #: * the counters the forward writes are replayed from the capture, so a
    #:   COUNTED arm does not go silent;
    #: * the whole step's transients live in a graph-private pool that is
    #:   PERMANENT for the run — this costs VRAM, and on a card that is full it
    #:   is refused COUNTED rather than allowed to fault (`graph.py`).
    #:
    #: `full_graph` IMPLIES `opt_graph` (the stage is inside it), so setting both
    #: is refused. Every arm the capture cannot carry is refused LOUDLY before
    #: the first step, with its reason and its `Escape:` — see
    #: `train.graph.full_graph_refusal`. The one non-refusal a reader should know
    #: about: **data-dependent routing is fine** (the MoE top-k is a kernel on
    #: buffer VALUES, so a replay reads whatever the buffer holds) — what is not
    #: fine is a PYTHON branch on a routing value, and there is none on the
    #: training path (`grep` for a host read in `model/` finds `act_quant.py:240`,
    #: `loopblock.py:576`, `loopblock.py:674` and `engram._forward_host`, each of
    #: which is a refused arm).
    full_graph: bool = False
    #: EXPONENTIAL MOVING AVERAGE of the parameters, used at eval/save time.
    #: `0.0` (the default) keeps no shadow at all: every checkpoint/eval/serve
    #: path is the pre-arm one, bit-identical. `0 < d < 1` keeps a shadow of the
    #: PARAMETER TREE and applies `shadow <- d·shadow + (1-d)·param` after each
    #: optimizer step (once per `grad_accum` group), gated by the NaN firewall's
    #: device flag so a skipped step leaves the shadow exactly where it was —
    #: the update is `lerp_` with a 0-dim device weight, so it adds no host
    #: read. `1.0` is refused (a shadow that never moves is not an average).
    #:
    #: WHAT IT COVERS: `model.parameters()` and nothing else — the same tree the
    #: checkpoint's model half carries. On `engram_ram` the host n-gram tables
    #: are NOT parameters (they are a sidecar with their own update rule), so
    #: the shadow is the EMA of the model half while eval reads the live host
    #: tables. That is stated, not hidden: the alternative is a second 12 GB
    #: table copy for a substrate that is already its own moving state.
    #:
    #: CHECKPOINT CONTRACT: when armed, `save_ckpt` writes the shadow under the
    #: clearly named `model_ema` key BESIDE the raw training copy under `model`
    #: (resume continues the raw iterate and its optimizer state, never the
    #: EMA: the moments belong to the raw weights). `load_model_half` — the
    #: shared reader of eval/decode/serve/chat — PREFERS `model_ema` when it is
    #: present, so the checkpoint a reader serves is the averaged model. A
    #: resume restores the shadow from that key; a checkpoint without it starts
    #: the shadow from the loaded weights, COUNTED in the log.
    #:
    #: RESUME: `ema_decay` is in `PROGRESS_KEYS`, so `--set ema_decay=...` on a
    #: resume is LEGAL even though it changes what is evaluated. The reason is
    #: in the paragraph above: the checkpoint carries BOTH copies of the same
    #: optimizer trajectory, and this knob selects (or re-decays) the shadow,
    #: never the weights the training step continues from. Refused only with
    #: `muon_sf` (the params hold the fast iterate Y during a step, so a shadow
    #: taken there would average a variable that is not the returned iterate X;
    #: materialising X per step is the named follow-up) and outside [0,1).
    #: `opt_graph` is ALLOWED: the EMA update runs EAGERLY after the replay,
    #: outside the captured region, reading the replayed parameters.
    ema_decay: float = 0.0
    seed: int = 1
    #: WSD warmup, in TOKENS (not steps): the schedule is a function of how
    #: much data the run has seen, so a batch-size change does not change it.
    warmup_tokens: int = 0
    #: Fraction of the token budget spent on the WSD decay. A PROGRESS key: it
    #: moves the decay window of the REMAINING steps and never reaches a
    #: forward, exactly like `steps` (which already moves the same window by
    #: scaling the total). Re-planning a run's tail — "start the decay now, make
    #: it longer" — is a legitimate resume, and refusing it forced a re-launch
    #: from scratch (measured 2026-10-30: 320k-step flagship plateaued at 1.74
    #: by step 100k; the tail had to be re-planned, and only `steps` was legal).
    decay_frac: float = 0.1
    #: The LR the WSD decay lands on, as a fraction of `lr` (`wsd_lr`'s
    #: `final_frac`): the last step trains at `lr * min_lr_frac`. 0.1 is the
    #: schedule's long-standing floor; a small value (0.01) is a full anneal to
    #: (almost) zero. A PROGRESS key for the same reason as `decay_frac`: it
    #: shapes the remaining LR path, never the forward.
    min_lr_frac: float = 0.1
    #: JSONL telemetry sink: one row per event (log line, eval, checkpoint),
    #: consumed by `dormouse plot`. Empty = off. A PROGRESS key: it is where
    #: measurements go, never what the run computes, so re-pointing it on a
    #: resume is legal and expected for a moved run (audit MINOR 23, 2026-10-17).
    json_log: str = ""
    data: str = ""
    eval_data: str = ""
    #: The staged-mixture plan (`train/mixture.py`, TOML): when set, the train
    #: side reads N sources interleaved by per-stage weights instead of the one
    #: `data` file (the eval side is unchanged). Empty = off = every number on
    #: record. Non-progress: flipping it changes what the run reads, so the
    #: ADR-0021 drift check refuses a flip on a resume.
    mix_plan: str = ""
    #: The repetition guard (`train/mixture.py`): a source whose planned bytes
    #: exceed `max_epochs × its file size` is a LOUD refusal naming it (the
    #: research onset: MoE degrades at 4× repetition, dense at 8×). 0 = off.
    max_epochs: float = 4.0
    #: DOCUMENT MASKING (`train/data.py::DocPackedStream`): one DOCUMENT per
    #: TRAIN row instead of a flat concatenation. Each row is a whole document,
    #: or a contiguous chunk of one document when it is longer than `seq_len`,
    #: padded at the tail; the padded/cross-document target positions are
    #: excluded from the loss (`DormouseModel.forward(loss_mask=...)`).
    #:
    #: **Why one document per row and not whole-document PACKING** (two
    #: documents glued into one row, boundary targets masked): the KDA arm is a
    #: recurrence carried through the fused `chunk_kda` over the whole window,
    #: and there is no state reset at a mid-window boundary. A packed row would
    #: therefore let document B's positions attend to document A's — exactly
    #: the cross-document dependency this flag removes. Rows that never hold
    #: two documents need no reset anywhere, fused kernel or not.
    #:
    #: The cost, named: a corpus of short documents pays tail padding (a
    #: 200-byte document leaves ~80 % of a 1024-byte row empty), so the flag is
    #: a quality lever for short-document corpora and a near-no-op for the
    #: 6 MB `web` shards. It needs a `<data>.docs.json` sidecar
    #: (`scripts/build_doc_sidecar.py`): a `.bin` alone cannot say where a
    #: document ends, and a missing sidecar is a LOUD refusal, never a flat
    #: fallback. Refused with `segment_len>0` (the carried state has no
    #: boundary notion), `use_byteflow=true` (patch boundaries are not document
    #: boundaries), and the target-bearing aux terms.
    #:
    #: Non-progress: flipping it changes what the run trains on, so the
    #: snapshot drift check refuses a flip on a resume.
    doc_masking: bool = False
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
    #: The host update's transient (`model/engram.py::HostTables`). `true` (the
    #: default) divides the Nesterov look-ahead by its total weight per ROW,
    #: `1 - HOST_MOMENTUM**(t+1)`, `t` = that row's touch count: a cold row's
    #: first touch is the raw gradient instead of `1-μ² = 0.0975` of it, and a
    #: row with `t ≥ 337` (measured) divides by exactly `1.0` (bit-unchanged). `false` is
    #: burn's `offload.rs` formula verbatim — the A/B's `ram_old_formula` arm.
    #: Host path only (the in-VRAM table is an `nn.Parameter` on AdamW and never
    #: reads this). In the snapshot, so a flip on a resume is refused by
    #: `diff_keys` like every other non-progress key.
    engram_bias_correction: bool = True
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
    #: PER-ITERATION CHECKPOINTING (`model/loopblock.py::_window`): each of the
    #: `max_iter` passes over the SAME weights runs inside
    #: `checkpoint(use_reentrant=False)`, so only that iteration's INPUT state
    #: survives the forward and the deepest activation set kept for the backward
    #: is one iteration's, not `max_iter` iterations'. It is a MEMORY decision,
    #: not a model change: the same ops run on the same inputs and the backward
    #: recomputes them exactly (the gate is `1e-6` relative on logits and
    #: gradients, and CPU measures bit-identity). It composes with
    #: `segment_len` (a checkpointed iteration inside a checkpointed segment),
    #: `rand_depth` (the count is the ACTUAL depth, sampled before the call),
    #: `use_csa` (the ring state is carried as an input, never checkpointed),
    #: `use_engram`, `loop_blocks` and `grad_accum` (each micro-batch's backward
    #: consumes its own recompute before the next forward).
    #:
    #: `false` is the default and the historical graph, bit for bit. Counter
    #: `iter_checkpoint_saved` is the number of iterations the backward will
    #: recompute, once per window — a zero on a run with the flag on means the
    #: arm never fired. REFUSED with `full_graph` / `opt_graph`: a capture cannot
    #: see a recompute, and replaying a graph that skipped it would feed the
    #: backward a tensor the capture never kept.
    iter_checkpoint: bool = False
    #: THE NAMED RECOMPUTE ARM (`model/recompute.py`): "none" | "full" |
    #: "selective". `full` is `iter_checkpoint`'s whole-iteration checkpoint;
    #: `selective` keeps the READ half (the q/k/v/g projection outputs, the
    #: KDA/CSA boundary state, the Engram read) and checkpoints only the WRITE
    #: half (the MoE pass, the residual write, `out_proj`), so the backward
    #: never repeats the expensive read-side matmuls while the large MoE
    #: activations still leave VRAM. `iter_checkpoint=true` with this field at
    #: "none" is the historical alias for "full"; naming two different modes is
    #: refused. The per-iteration head runs OUTSIDE every checkpoint in all
    #: three arms (`_iter_ce`), so its matmul is never recomputed.
    activation_recompute: str = "none"
    #: PINNED HOST OFFLOAD of the LARGEST saved activations (`model/act_mem.py`),
    #: in GiB moved to host PER FORWARD; `0.0` is OFF and installs no hooks.
    #: The point is the idle PCIe time under a ~780 ms step (~32 GB/s on PCIe
    #: 4.0 x16, so ~6 GiB can move without extending the step), not device
    #: capacity. The first training forward PROFILES the saves; at its end the
    #: sizes are sorted descending and the largest prefix that fits the budget
    #: sets the threshold. Later forwards park every save at or above it (up to
    #: the budget) in a pinned buffer, and the backward copies them back EXACTLY
    #: — the loss and gradients are identical to the off run (`1e-6` relative
    #: gate; exact on CPU). Buffers are pooled per size class and returned on
    #: the backward edge, so the pool is bounded by the number of save sizes and
    #: does not grow with the step count. `grad_accum` is safe: each micro-batch's
    #: saves are consumed by its own backward before the next forward.
    #:
    #: Counters `offload_tensors` and `offload_gib_x100` (the bytes actually
    #: parked, integer hundredths of a GiB) appear on the log line, and
    #: `offload_profile_forward` marks the one forward that only measured sizes.
    #: REFUSED with `full_graph` / `opt_graph` (a capture cannot see a Python
    #: hook or an H2D/D2H copy) and with `use_byteflow` (the chunked head builds
    #: a second patch-latent forward whose save set is not measured here).
    activation_offload: float = 0.0

    # --- the RL lane (`post/rl.py` + `post/rl_train.py`) ----------------
    # The reward/advantage knobs are `RLConfig`'s fields, spelled here so the CLI
    # and a snapshot carry them; the mapping back is [`rl_config`], the ONE
    # construction site (so `RLConfig.__post_init__`'s refusals are the only
    # rules and cannot drift into a second copy).
    #: `S_sol`'s proxy: `thrift` (bytes against the hidden reference) or `none`.
    rl_quality: str = "thrift"
    #: X — the group-relative length penalty's ceiling; 0.0 = OFF.
    #: The default is MEASURED, not tasted: `scripts/measure_rl_length.py`
    #: (2026-10-23, 2 500-task corpus, 200 math tasks) found the longest correct
    #: answer 1.364x the median correct length (15 B vs 11 B). At s = 0.5 the
    #: ramp is 0.727, so X = 0.1 costs the longest correct answer 7.3 % of a
    #: passing reward (the stated 10 % budget; X = 0.2 costs 14.5 %, over it).
    rl_len_penalty: float = 0.1
    #: B — the percentile of the SUCCESSFUL lengths that sets ℓ*_q (0.5-0.75).
    rl_len_quantile: float = 0.5
    #: A — the group pass rate the length term needs (0.3-0.5).
    rl_len_gate: float = 0.4
    #: δ — the free band above ℓ*_q (0.0-0.1); s must stay above it.
    rl_len_delta: float = 0.0
    #: s — where the ramp saturates, as a multiple of ℓ*_q.
    rl_len_s: float = 0.5
    #: γ — the ramp's exponent (1-2).
    rl_len_gamma: float = 1.0
    #: The advantage arm (`RL_ADVANTAGES`): `crr` = the fork-replay shape
    #: (`post/train_steps.crr_advantage`), `group` = GRPO's group-normalised
    #: rewards. `crr` is the default because it is the arm with a gate behind it.
    rl_adv: str = "crr"
    #: The loss normaliser: `prompt-mean` (this lane's rule) or `token-mean` (the
    #: notes' §4.2 control arm).
    rl_agg: str = "prompt-mean"
    #: The task class `build_tasks` generates (`code` | `math` | `noop`) when
    #: `rl_corpus` is empty. `noop` exists only to test the verifier's refusal.
    rl_task_cls: str = "code"
    #: How many tasks a seed produces (`build_tasks(cls, seed, n=...)`); a step
    #: draws ONE of them as a pure function of `(seed, step, attempt)`. The
    #: generated pool is the smoke path: 128 is `TASKS_PER_CLASS`, and the
    #: default 32-task holdout leaves 96 to train on. A real run points
    #: `rl_corpus` at the 2 500-task JSONL instead.
    rl_tasks: int = 128
    #: G — rollouts per prompt: the group the reward, the advantage and the
    #: length penalty are all relative to. GRPO's group of 4-8 is the sourced band.
    rl_group: int = 4
    #: The rollout's byte budget (`correct@budget(bytes)`): prompt + response +
    #: 2 B per generation stroke. 256 B is the `code` class's own budget.
    rl_budget: int = 256
    #: A `scripts/build_rl_tasks.py` JSONL corpus. Empty = the generated
    #: `build_tasks(cls, seed + SFT_TASK_OFFSET)` pool.
    rl_corpus: str = ""
    #: Tasks carved out of the corpus as the held-out eval half (never trained).
    rl_holdout: int = 32
    #: Held-out eval cadence in steps; 0 = only the before/after readings.
    rl_eval_every: int = 50
    #: Sampled attempts per held-out task for `pass@budget` (plus the greedy
    #: `pass@1` attempt, which is always taken).
    rl_eval_samples: int = 2
    #: The difficulty band on the group's pass rate, INCLUSIVE at both ends:
    #: below `lo` the group is too hard, above `hi` it is consensus.
    rl_band_lo: float = 0.2
    rl_band_hi: float = 0.8
    #: Candidate prompts a step may draw before the counted fallback trains the
    #: last one (`rl_band_exhausted`). 1 disables the retry, not the band.
    rl_band_attempts: int = 4


#: The `rl_adv` literals — the two advantage arms `rl.rl_advantages` implements.
#: The list is duplicated here the way `MOE_KERNELS` is, so the config layer does
#: not import the post lane. A name that is not here is LOUD at load.
RL_ADVANTAGES = ("crr", "group")


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
        # `max_seq_len` is a BUFFER bound, not a learned constant, and the
        # KDA/CSA arms never index it. Growing it beside a growing `seq_len` is
        # the same curriculum step; shrinking it would invalidate every decode
        # caller. Arms that DO build a positional table (byteflow's RoPE table,
        # full attention's) keep the strict refusal.
        # `max_seq_len` bounds buffers (decode's window, byteflow's RoPE table),
        # not what the model learns. On an arm that builds NO positional table it
        # is excluded from the resume identity entirely — direction-agnostic,
        # because the two callers pass (new, stored) and (stored, new) and a
        # direction-sensitive rule was wrong for one of them (measured
        # 2026-11-04: the 8 KB resume refused on a field it had just allowed).
        # `validate_train` still keeps `seq_len <= max_seq_len`.
        positional = self.model.use_byteflow or self.model.attn_kind == "full"
        if not positional and self.model.max_seq_len != other.model.max_seq_len:
            pass  # allowed: a buffer bound is not an experiment
        for section, a, b in (
            ("model", self.model, other.model),
            ("train", self.train, other.train),
        ):
            for f in dataclasses.fields(a):
                if section == "train" and f.name in PROGRESS_KEYS:
                    continue
                if section == "train" and f.name in ("batch", "grad_accum"):
                    # the resume may re-shard the SAME bytes per step across a
                    # different window/batch split (b8 x s1024 == b1 x s8192);
                    # a different token budget is still a different run, and
                    # that is caught below by comparing the product.
                    if (
                        self.train.batch * self.train.seq_len * self.train.grad_accum
                        == other.train.batch * other.train.seq_len * other.train.grad_accum
                    ):
                        continue
                if section == "model" and f.name == "max_seq_len" and not positional:
                    continue  # handled above: grow-only, and a grow is a resume
                if section == "model" and f.name in (
                    "moe_kernel",
                    "moe_permute",
                    "moe_bank_epoch",
                    # `moe_combine="fused"` in fp32 is `torch.equal` on logits and
                    # every gradient (tests/test_moe_combine_fused.py), so it is an
                    # implementation key exactly like `moe_kernel`.
                    "moe_combine",
                    # The fused residual/epilogue kernels are the SAME arithmetic in
                    # one launch: parity 8.5e-7 on logits and every gradient
                    # (tests/test_fused_epilogue.py, test_fused_kda.py), and the off
                    # path is bit-identical. `mhc_sinkhorn="log_softmax"` is the same
                    # normalisation as one ATen op (2.4e-7, test_mhc_write_launches).
                    "fused_residual",
                    "fused_epilogue",
                    "fused_kda_tail",
                    "fused_latent_proj",
                    "mhc_sinkhorn",
                ):
                    # implementation keys: `moe_kernel`'s arms are proven equal
                    # to 1e-6, and the balancing rule decides selection, not
                    # representation. `moe_permute` changes no VALUE at all —
                    # `tests/test_moe_route_fused.py` pins the counting
                    # permutation element-wise against the stable argsort — and
                    # `moe_bank_epoch` changes only how the SAME bank is cached
                    # (`tests/test_moe_bank_cache.py` pins build equality), so a
                    # resume may switch either. `moe_route_fused` and
                    # `moe_combine*` are NOT here: they change numerics.
                    continue
                if getattr(a, f.name) != getattr(b, f.name):
                    out.append(f"{section}.{f.name}")
        return sorted(out)

    def snapshot_toml(self) -> str:
        """The `<ckpt_name>.config.toml` record of this run."""
        return _to_toml({"model": self.model, "train": self.train})

    @classmethod
    def from_snapshot(cls, text: str) -> RunConfig:
        """Inverse of [`snapshot_toml`], strict about unknown keys.

        One COUNTED exception: a key in [`DEAD_SNAPSHOT_KEYS`] was valid when
        the snapshot was written and is not a field of this branch. It is
        dropped only when its value IS the recorded default (a drop that changes
        nothing the run computes); a non-default value is refused LOUDLY rather
        than dropped, because dropping it would resume a different model. A key
        that is not on that list still refuses (`_known`).
        """
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            loud(f"config snapshot does not parse: {e}")
        return cls(
            source=raw.get("source", ""),
            model=_from_table(
                ModelConfig,
                _drop_dead_snapshot_keys(ModelConfig, raw.get("model", {}), "snapshot.model"),
                where="snapshot.model",
            ),
            train=_from_table(
                TrainConfig,
                _drop_dead_snapshot_keys(TrainConfig, raw.get("train", {}), "snapshot.train"),
                where="snapshot.train",
            ),
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


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
    return dict(table)


def _from_table(cls: type, table: dict[str, Any], where: str) -> Any:
    return cls(**_known(cls, table, where))


def _same_snapshot_value(value: Any, default: Any) -> bool:
    """Is `value` the recorded default? Type-guarded: `False == 0` in Python,
    and a snapshot that wrote `0` where the field was a bool is NOT at default."""
    if isinstance(value, bool) or isinstance(default, bool):
        return isinstance(value, bool) and isinstance(default, bool) and value == default
    return value == default


def _drop_dead_snapshot_keys(cls: type, table: dict[str, Any], where: str) -> dict[str, Any]:
    """Drop dead snapshot keys that sit at their recorded default, COUNTED.

    A snapshot is a record of what a run WAS: a [`DEAD_SNAPSHOT_KEYS`] key was
    valid when it was written and is not a field of `cls` on this branch. The
    drop is VALUE-CHECKED — the default is dropped, because nothing the run
    computed depends on a key that held its default; a non-default is a LOUD
    refusal, because dropping it would resume a different model or optimizer
    than the checkpoint was trained with. A key that IS a field of `cls` is
    never dropped: this list is dated, the schema is live.

    This does NOT relax the strict schema: `_known` still refuses any key not on
    the list, LOUDLY, and `parse_preset` never calls this — a preset says what
    will RUN, so a stale key there is exactly the lie the strict check catches.
    """
    live = {f.name for f in dataclasses.fields(cls)}
    dead = sorted(k for k in table if k in DEAD_SNAPSHOT_KEYS and k not in live)
    if not dead:
        return dict(table)
    for key in dead:
        commit, default = DEAD_SNAPSHOT_KEYS[key]
        value = table[key]
        if not _same_snapshot_value(value, default):
            loud(
                f"{where}: {key} = {value!r} is a dead snapshot key whose value is "
                f"NOT its default ({default!r}); the field is not in this code "
                f"(commit {commit}). Refusing, not dropping: dropping it would run a "
                f"different model/optimizer than the checkpoint was trained with. "
                f"Escape: run this checkpoint on the commit that still reads `{key}` "
                f"(`git log -S'{key}'` names it), or rewrite the snapshot's `{key}` to "
                f"{default!r} and accept that the run then computes something else."
            )
    named = ", ".join(f"{k} ({DEAD_SNAPSHOT_KEYS[k][0]})" for k in dead)
    print(
        f"config: {where}: dropped {len(dead)} dead snapshot key(s) {named} — each "
        f"at its recorded default, so the drop changes nothing this run computes. "
        f"COUNTED, not silent: a key outside DEAD_SNAPSHOT_KEYS still refuses "
        f"loudly, and a dead key off its default refuses too.",
        file=sys.stderr,
    )
    return {k: v for k, v in table.items() if k not in dead}


def validate_ffn(model: ModelConfig) -> None:
    """The `ffn_form` combinations that cannot work, LOUD with an `Escape`.

    One function with two callers (`validate` and `SparseMoE.__init__`): a
    hand-built `ModelConfig` reaches the module without the loader, and an arm
    that is silently inert is the defect ADR-0019 bans.
    """
    if model.ffn_form not in FFN_FORMS:
        loud(
            f"ffn_form={model.ffn_form!r} is not an FFN form. Known: "
            f"{list(FFN_FORMS)} — `bank` the routed SparseMoE on record, `dense` "
            f"one plain GLU with no rank, `tsct` all three matrices factorised. "
            f"Escape: name one of those."
        )
    if model.ffn_rank_gate < 0 or model.ffn_rank_down < 0:
        loud(
            f"ffn_rank_gate={model.ffn_rank_gate} / ffn_rank_down="
            f"{model.ffn_rank_down}: a rank is a width and 0 is the `use rank` "
            f"sentinel. Escape: >= 0."
        )
    d, f = model.d_model, model.d_ffn
    if model.ffn_form != "tsct":
        if model.ffn_rank_gate or model.ffn_rank_down:
            loud(
                f"ffn_rank_gate={model.ffn_rank_gate} / ffn_rank_down="
                f"{model.ffn_rank_down} with ffn_form={model.ffn_form!r}: only "
                f"the `tsct` form has factors to rank. Escape: ffn_form='tsct', "
                f"or both ranks 0."
            )
    if getattr(model, "tsct_packed", False) and model.ffn_form != "bank":
        loud(
            f"tsct_packed is a BANK feature (it packs the routed bank's factors), "
            f"and ffn_form={model.ffn_form!r} has no bank. Escape: "
            f"ffn_form='bank', or tsct_packed=false."
        )
    if model.ffn_form == "dense":
        if model.n_experts != 1:
            loud(
                f"ffn_form='dense' with n_experts={model.n_experts}: a dense FFN "
                f"has no bank to route over. Escape: n_experts=1 (the form IS the "
                f"expert), or ffn_form='bank'/'tsct'."
            )
        if model.share_u_mode != "off":
            loud(
                f"ffn_form='dense' with share_u_mode={model.share_u_mode!r}: "
                f"`share_u` is the BANK's shared input basis, and a dense FFN has "
                f"no factor basis to share. Escape: share_u_mode='off' (or "
                f"share_u=false), or ffn_form='bank'."
            )
        if model.moe_topk != 0:
            loud(
                f"ffn_form='dense' with moe_topk={model.moe_topk}: there is no "
                f"router and one expert, so a top-k would be an inert dispatch. "
                f"Escape: moe_topk=0, or ffn_form='bank'."
            )
        if (
            model.moe_shared_compute
            or model.tsct_expert_basis
            or model.tsct_expert_lr
            or model.tsct_expert_revive
        ):
            loud(
                "ffn_form='dense' with a bank-only arm (moe_shared_compute / "
                "tsct_expert_basis / tsct_expert_lr / tsct_expert_revive): the "
                "arm would reach no code. Escape: those fields off, or "
                "ffn_form='bank'/'tsct'."
            )
        if model.moe_lb_mode == "aux" and model.moe_aux_coef != 0.0:
            loud(
                f"ffn_form='dense' with moe_lb_mode='aux' and moe_aux_coef="
                f"{model.moe_aux_coef}: the Switch/GShard term needs a router "
                f"and this form has none. Escape: moe_lb_mode='sign' with "
                f"moe_lb_coef=0, or ffn_form='bank'."
            )
        if model.moe_lb_coef != 0.0 or (
            model.moe_lb_mode == "prop" and model.moe_lb_gamma != 0.0
        ):
            loud(
                f"ffn_form='dense' with a live LFB controller "
                f"(moe_lb_mode={model.moe_lb_mode!r}, coef={model.moe_lb_coef}, "
                f"gamma={model.moe_lb_gamma}): there is no router to rebalance. "
                f"Escape: a zero controller step, or ffn_form='bank'."
            )
        return
    if model.ffn_form == "bank":
        return
    # ---- tsct -----------------------------------------------------------
    lo = min(d, f)
    if model.rank > lo:
        loud(
            f"ffn_form='tsct' with rank={model.rank} > min(d_model, d_ffn)="
            f"{lo}: a rank above either dimension is not a factorisation (the "
            f"same rule `TSCTLinear` enforces). Escape: rank <= {lo}, or "
            f"ffn_form='dense'/'bank'."
        )
    rg = model.ffn_rank_gate or model.rank
    rd = model.ffn_rank_down or model.rank
    if rg < 1 or rd < 1:
        loud(
            f"ffn_form='tsct' with ffn_rank_gate={rg} / ffn_rank_down={rd}: a "
            f"factor rank must be >= 1. Escape: positive ranks."
        )
    if rg > lo or rd > lo:
        loud(
            f"ffn_form='tsct' with ffn_rank_gate={rg} / ffn_rank_down={rd} above "
            f"min(d_model, d_ffn)={lo}: not a factorisation. Escape: both <= "
            f"{lo}."
        )
    if model.share_u_mode != "off":
        loud(
            f"ffn_form='tsct' with share_u_mode={model.share_u_mode!r}: the "
            f"shared input plane is a BANK-form arm (`SparseMoE._banks`), and the "
            f"tsct form's gate/up already own their `V` factors. Escape: "
            f"share_u_mode='off', or ffn_form='bank'."
        )
    if model.moe_shared_compute:
        loud(
            "ffn_form='tsct' with moe_shared_compute=true: that arm fuses the "
            "BANK's shared input basis, which this form does not build. Escape: "
            "moe_shared_compute=false, or ffn_form='bank'."
        )
    if model.tsct_expert_basis or model.tsct_expert_lr or model.tsct_expert_revive:
        loud(
            "ffn_form='tsct' with a TSCT-2 bank arm (tsct_expert_basis / "
            "tsct_expert_lr / tsct_expert_revive): those read the routed bank's "
            "`_banks` and LFB counts, and this form is its own dispatch. Escape: "
            "those fields off, or ffn_form='bank'."
        )
    if model.moe_topk != 0 and model.moe_capacity > 0:
        loud(
            f"ffn_form='tsct' with moe_topk={model.moe_topk} at "
            f"moe_capacity={model.moe_capacity}: this dispatch is DROPLESS (it "
            f"has no capacity bank), so a capacity rule would silently do "
            f"nothing. Escape: moe_capacity=0 (dropless), or moe_topk=0 (dense "
            f"blend)."
        )
    if model.moe_lb_mode == "aux" and model.moe_aux_coef != 0.0:
        loud(
            f"ffn_form='tsct' with moe_lb_mode='aux' and moe_aux_coef="
            f"{model.moe_aux_coef}: the TSCT dispatch has no Switch/GShard term. "
            f"Escape: moe_lb_mode='sign' with moe_lb_coef=0, or ffn_form='bank'."
        )
    if model.moe_lb_coef != 0.0 or (
        model.moe_lb_mode == "prop" and model.moe_lb_gamma != 0.0
    ):
        loud(
            f"ffn_form='tsct' with a live LFB controller "
            f"(moe_lb_mode={model.moe_lb_mode!r}, coef={model.moe_lb_coef}, "
            f"gamma={model.moe_lb_gamma}): the TSCT dispatch does not move a "
            f"routing bias. Escape: a zero controller step, or ffn_form='bank'."
        )


def validate(model: ModelConfig) -> None:
    """Coherence checks the burn side does in `config::validate`. Loud, never
    clamped: a clamped width is exactly the silent collapse ADR-0011 bans."""
    validate_ffn(model)
    if model.d_model <= 0 or model.vocab <= 0:
        loud(f"d_model and vocab must be > 0, got {model.d_model}/{model.vocab}")
    if model.n_heads * model.head_dim != model.d_model:
        loud(
            f"n_heads*head_dim={model.n_heads * model.head_dim} != d_model="
            f"{model.d_model}: the head split must tile the residual width"
        )
    if model.attn_kind not in ("kda", "full"):
        loud(
            f"attn_kind={model.attn_kind!r} is not an attention arm. Known: 'kda' "
            f"(the compressed-state gated-delta arm every number on record ran) and "
            f"'full' (exact causal attention, model/fullattn.py). Escape: name one "
            f"of those, or land the arm and its gate BEFORE the config can lie "
            f"about it."
        )
    if not model.use_kda and model.attn_kind != "kda":
        loud(
            f"attn_kind={model.attn_kind!r} with use_kda=false: use_kda=false is the "
            f"HARD OFF switch — the loop's attention output is exactly zeros "
            f"(loopblock._window), so the arm named here would never run. The two "
            f"flags are not aliases: use_kda says WHETHER attention runs, attn_kind "
            f"says WHICH. Escape: use_kda=true (the default) for attn_kind='full', "
            f"or attn_kind='kda' if the zeros arm is what you meant."
        )
    if model.attn_window < 0:
        loud(
            f"attn_window={model.attn_window} is negative: it is a count of byte "
            f"positions the local band reaches back over, and 0 is the OFF value. "
            f"Escape: 0, or a positive window."
        )
    if model.attn_window_query_chunk < 0:
        loud(
            f"attn_window_query_chunk={model.attn_window_query_chunk} is negative: it "
            f"is a query-block size, and 0 is the OFF value (one block). Escape: 0, "
            f"or a positive block size."
        )
    if model.attn_window_query_chunk and not model.attn_window:
        loud(
            f"attn_window_query_chunk={model.attn_window_query_chunk} without "
            f"attn_window: the query blocks exist only to cap the banded score "
            f"tensor of the local window, so alone the flag would change nothing "
            f"while the log line reported it on. Escape: attn_window>0, or "
            f"attn_window_query_chunk=0."
        )
    if model.attn_window and model.attn_kind == "full":
        loud(
            f"attn_window={model.attn_window} with attn_kind='full': the exact arm "
            f"already reads every earlier position, so the window would be a strict "
            f"subset computed a second time — `4·d²` dead parameters and a silent "
            f"double attention. Escape: attn_kind='kda' (the window beside the "
            f"compressed arm is the arm this flag exists for), or attn_window=0."
        )
    if model.attn_window and model.use_byteflow:
        loud(
            f"attn_window={model.attn_window} with use_byteflow=true: the channel's "
            f"loop runs over K patch latents, not over the byte window, so a 'window "
            f"of {model.attn_window} bytes' would silently be a window of latents (a "
            f"different unit and a different band). Refused rather than repurposed. "
            f"Escape: use_byteflow=false, or attn_window=0."
        )
    if model.attn_window_query_chunk >= model.max_seq_len > 0:
        loud(
            f"attn_window_query_chunk={model.attn_window_query_chunk} >= "
            f"max_seq_len={model.max_seq_len}: the forward is at most one query "
            f"block, so the chunking would buy nothing while the log line reported "
            f"it on. Escape: a chunk < max_seq_len, or 0 (one block)."
        )
    if model.attn_latent_rank < 0:
        loud(
            f"attn_latent_rank={model.attn_latent_rank} is negative: a shared "
            f"latent is a WIDTH in (0, d_model], and 0 is the OFF sentinel. "
            f"Escape: 0, or a positive rank."
        )
    if model.attn_latent_rank and not model.use_kda:
        loud(
            f"attn_latent_rank={model.attn_latent_rank} with use_kda=false: the "
            f"KDA module is not built, so the key would reach no code (the "
            f"config-lies defect). Escape: use_kda=true, or attn_latent_rank=0."
        )
    if model.n_kv_heads < 0:
        loud(
            f"n_kv_heads={model.n_kv_heads} is negative: 0 is the OFF sentinel "
            f"(MHA) and a positive value is a head count. Escape: 0, or >= 1."
        )
    if model.n_kv_heads and not model.use_kda:
        loud(
            f"n_kv_heads={model.n_kv_heads} with use_kda=false: the KDA module "
            f"is not built, so the key would reach no code. Escape: use_kda=true, "
            f"or n_kv_heads=0."
        )
    if model.n_kv_heads > model.n_heads:
        loud(
            f"n_kv_heads={model.n_kv_heads} > n_heads={model.n_heads}: GQA shares "
            f"key/value heads with the query heads, so there cannot be more KV "
            f"heads than queries. Escape: n_kv_heads <= n_heads."
        )
    if model.n_kv_heads and model.n_heads % model.n_kv_heads:
        loud(
            f"n_kv_heads={model.n_kv_heads} does not divide n_heads={model.n_heads}: "
            f"GQA expands each KV head into a whole group of query heads "
            f"(`repeat_interleave(n_heads // n_kv_heads)`), and a ragged group has "
            f"no valid expansion. Escape: a divisor of n_heads, or 0 (MHA)."
        )
    if model.out_proj_rank < 0 or model.out_proj_rank > model.d_model:
        loud(
            f"out_proj_rank={model.out_proj_rank} outside 0..=d_model="
            f"{model.d_model}: a rank above either dimension of the square "
            f"readout is not a factorization, and 0 is the OFF sentinel. "
            f"Escape: 0..=d_model."
        )
    if model.csa_coarse_chunk and not model.use_csa:
        loud(
            f"csa_coarse_chunk={model.csa_coarse_chunk} without use_csa: the "
            f"coarse level is the second level of the CSA arm, so alone it would "
            f"build no module and the flag would lie. Escape: use_csa=true, or "
            f"csa_coarse_chunk=0."
        )
    if model.csa_mid_chunk and not model.use_csa:
        loud(
            f"csa_mid_chunk={model.csa_mid_chunk} without use_csa: the mid level is "
            f"the third level of the CSA arm, so alone it would build no module and "
            f"the flag would lie. Escape: use_csa=true, or csa_mid_chunk=0."
        )
    if model.csa_fused and not model.use_csa:
        loud(
            "csa_fused=true without use_csa: the fused path is the CSA arm's fine "
            "selection, and the arm is not built — the flag would run nothing. "
            "Escape: use_csa=true, or csa_fused=false."
        )
    if model.csa_hierarchical:
        if not model.use_csa:
            loud(
                "csa_hierarchical=true without use_csa: hierarchical selection "
                "narrows the FINE level's scan inside the CSA arm, which is not "
                "built. Escape: use_csa=true, or csa_hierarchical=false."
            )
        if not model.csa_mid_chunk:
            loud(
                "csa_hierarchical=true without csa_mid_chunk: the hierarchy needs a "
                "level above the fine one to select THROUGH — with only coarse+fine "
                "it would score the same coarse entries the dense branch already "
                "scores and then still scan every fine chunk, i.e. a rename of the "
                "flat search wearing a flag. Escape: csa_mid_chunk=<multiple of the "
                "coarse chunk>, or csa_hierarchical=false."
            )
        if not model.csa_fine:
            loud(
                "csa_hierarchical=true with csa_fine=false: the hierarchy narrows "
                "the FINE level's candidate set, and the fine level is off — the "
                "flag would select for an attention that does not run. Escape: "
                "csa_fine=true, or csa_hierarchical=false."
            )
    if model.use_csa:
        if model.csa_chunk not in (4, 8, 16, 32):
            loud(
                f"csa_chunk={model.csa_chunk} is not a chunk size: the arm pools "
                f"{model.csa_chunk} byte positions into one KV entry, and the "
                f"measured set is (4, 8, 16, 32). Escape: one of those."
            )
        if model.csa_kv_dim not in (64, 128):
            loud(
                f"csa_kv_dim={model.csa_kv_dim} is not a CSA width: the arm ships "
                f"64 and 128 (the memory table is asserted at 64). Escape: 64 or 128."
            )
        if model.csa_topk < (1 if model.csa_fine else 0):
            loud(
                f"csa_topk={model.csa_topk}: a top-k of 0 selects no chunk, which is "
                f"the branch switched off with its parameters and gate still built; "
                f"and a negative top-k is not a count. csa_topk=0 is legal only when "
                f"csa_fine=false. Escape: csa_topk >= 1, or use_csa=false."
            )
        if not model.csa_fine and model.csa_coarse_chunk == 0:
            loud(
                "csa_fine=false without a coarse level: the arm would build no "
                "level at all while paying its gate. Escape: csa_fine=true, or "
                "csa_coarse_chunk>0, or use_csa=false."
            )
        if model.csa_fused and not model.csa_fine:
            loud(
                "csa_fused=true with csa_fine=false: the fused path is the FINE "
                "selection, and the fine level is off — the flag would select for "
                "an attention that does not run. Escape: csa_fine=true, or "
                "csa_fused=false."
            )
        if model.csa_mid_chunk:
            contained = model.csa_coarse_chunk or model.csa_chunk
            if model.csa_mid_chunk < model.csa_chunk:
                loud(
                    f"csa_mid_chunk={model.csa_mid_chunk} is smaller than "
                    f"csa_chunk={model.csa_chunk}: the mid level is COARSER than the "
                    f"fine one (its entries are what the fine chunks are selected "
                    f"inside), so a mid chunk below a fine chunk is not a third level. "
                    f"Escape: csa_mid_chunk >= csa_chunk."
                )
            if model.csa_mid_chunk % contained:
                loud(
                    f"csa_mid_chunk={model.csa_mid_chunk} is not a multiple of "
                    f"{'csa_coarse_chunk' if model.csa_coarse_chunk else 'csa_chunk'}"
                    f"={contained}: a mid entry must be a whole number of the entries "
                    f"it contains, or the hierarchy would select a set that is not "
                    f"the union of its children's chunks (two different byte "
                    f"boundaries, and `csa_hierarchical` would then compute something "
                    f"the flat scan never computes). Escape: a multiple of "
                    f"{contained}."
                )
        if model.csa_coarse_chunk:
            if model.csa_coarse_chunk % model.csa_chunk:
                loud(
                    f"csa_coarse_chunk={model.csa_coarse_chunk} is not a multiple of "
                    f"csa_chunk={model.csa_chunk}: a coarse entry must be a whole "
                    f"number of fine entries, or the two levels pool different byte "
                    f"boundaries. Escape: a multiple of csa_chunk."
                )
            if model.csa_coarse_chunk not in (64, 128, 256, 512):
                loud(
                    f"csa_coarse_chunk={model.csa_coarse_chunk} is not a coarse "
                    f"chunk: the level means {model.csa_coarse_chunk} bytes into one "
                    f"dense entry, and the measured set is (64, 128, 256, 512). "
                    f"Escape: one of those."
                )
            if model.csa_coarse_kv_dim not in (64, 128):
                loud(
                    f"csa_coarse_kv_dim={model.csa_coarse_kv_dim} is not a coarse "
                    f"width: the level ships 64 and 128. Escape: 64 or 128."
                )
            if model.use_msa:
                loud(
                    "use_csa + csa_coarse_chunk with use_msa: two sparse/dense "
                    "attention levels beside MSA have not been measured. Escape: "
                    "one extra branch per run, or use_msa=false."
                )
        if model.use_msa:
            loud(
                "use_csa + use_msa: two sparse attention branches have not been "
                "measured together, so the run could not attribute a result to "
                "either. Escape: one sparse branch per run."
            )
        if model.attn_kind != "kda":
            loud(
                f"attn_kind={model.attn_kind!r} with use_csa: CSA is specified as an "
                f"ADDITIVE branch beside KDA's compressed state, not beside exact "
                f"attention. Escape: attn_kind='kda', or use_csa=false."
            )
        if not model.use_kda:
            loud(
                "use_csa with use_kda=false: the hard off switch makes KDA's output "
                "zeros, and CSA is the branch ADDED to it — the pair is not the "
                "measured arm. Escape: use_kda=true, or use_csa=false."
            )
        if model.use_byteflow:
            loud(
                "use_csa with use_byteflow=true: the channel runs the loop over K "
                "patch latents, so a 'csa_chunk bytes' would silently be a chunk of "
                "latents (a different unit, and a different window). Refused rather "
                "than repurposed. Escape: use_byteflow=false, or use_csa=false."
            )
    if model.moe_shared_compute:
        # This arm is a shape of the SHARED basis' projection: with no basis it
        # would have nothing to read, and with the dictionary the per-expert
        # column subset makes one `[d, 2r]` matrix impossible. The `rank < 1`
        # check runs HERE (before the generic one below) so the refusal names
        # the arm and its escape instead of a bare shape complaint.
        if model.share_u_mode == "off":
            loud(
                "moe_shared_compute=true with share_u_mode='off': the arm reads "
                "the ONE shared basis (`gate_u`/`up_u [d, rank]`) and projects "
                "the tokens through it once; with no shared basis every expert "
                "owns its own `U` and there is nothing to share. Escape: "
                "share_u_mode='plane' (or share_u=true), or "
                "moe_shared_compute=false."
            )
        if model.share_u_mode == "dict":
            loud(
                "moe_shared_compute=true with share_u_mode='dict': each expert "
                "reads a different `rank`-column subset of the dictionary, so "
                "there is no single `[d, 2r]` matrix to project the tokens with "
                "once. Escape: share_u_mode='plane', or "
                "moe_shared_compute=false (the dictionary's per-expert gather "
                "stays the dispatch's shape)."
            )
        if model.rank < 1:
            loud(
                f"moe_shared_compute=true with rank={model.rank}: the projection "
                f"produces a `[n, 2·rank]` latent and rank 0 has no columns. "
                f"Escape: rank >= 1, or moe_shared_compute=false."
            )
        if model.moe_kernel not in MOE_SHARED_COMPUTE_KERNELS:
            loud(
                f"moe_shared_compute=true with moe_kernel={model.moe_kernel!r}: "
                f"the shared projection is wired into "
                f"{list(MOE_SHARED_COMPUTE_KERNELS)} only; `padded`/`bucketed` "
                f"are the old shape's control and `naive` is the reference loop. "
                f"Escape: moe_kernel='batched' (or 'triton'), or "
                f"moe_shared_compute=false."
            )
        if model.moe_gather == "kernel":
            loud(
                "moe_shared_compute=true with moe_gather='kernel': the gather arm "
                "owns the routed forward and reads `x` through the routing map, so "
                "the shared projection would never run while the flag claimed it "
                "did. Escape: moe_gather='off', or moe_shared_compute=false."
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
    if model.moe_topk_mode not in MOE_TOPK_MODES:
        loud(
            f"moe_topk_mode={model.moe_topk_mode!r} is not a width policy. Known: "
            f"{list(MOE_TOPK_MODES)} — `off` the constant `moe_topk` (every number on "
            f"record), `entropy` the normalized router entropy, `margin` the top-1/"
            f"top-2 logit gap (model/adaptive_topk.py). Escape: name one of those, or "
            f"land the policy and its gate BEFORE the config can lie about it."
        )
    if not 1 <= model.moe_topk_min <= model.moe_topk_max <= MOE_TOPK_SLOTS_MAX:
        loud(
            f"moe_topk_min={model.moe_topk_min} / moe_topk_max={model.moe_topk_max} "
            f"outside 1 <= min <= max <= {MOE_TOPK_SLOTS_MAX}: the histogram counters "
            f"and the fused backward's top-k tile stop there, and an inverted window "
            f"would clamp every token to one end. Escape: the defaults 1/4."
        )
    if model.moe_topk_mode != "off":
        if model.moe_topk <= 0:
            loud(
                f"moe_topk_mode={model.moe_topk_mode!r} with moe_topk=0: the adaptive "
                f"width lives on the ROUTED pass, and 0 is the dense blend (every "
                f"expert, no router top-k). Escape: moe_topk > 0 — its VALUE selects "
                f"the arm here, the width comes from moe_topk_max."
            )
        if model.moe_topk_max > model.n_experts:
            loud(
                f"moe_topk_max={model.moe_topk_max} > n_experts={model.n_experts}: a "
                f"width above the bank cannot be filled. Escape: k_max <= n_experts."
            )
        if not model.moe_entropy_lo < model.moe_entropy_hi:
            loud(
                f"moe_entropy_lo={model.moe_entropy_lo} >= moe_entropy_hi="
                f"{model.moe_entropy_hi} with mode={model.moe_topk_mode!r}: the window "
                f"would be a step, not a ramp. Escape: lo < hi (defaults 0.2 < 0.6)."
            )
    if model.moe_capacity < 0.0:
        loud(
            f"moe_capacity={model.moe_capacity} is negative: it is a multiple of the "
            f"average expert load (`ceil(capacity·tokens·k/E)` rows per expert), and 0 "
            f"is the DROPLESS mode (every assignment reaches its expert, the padded "
            f"bank refuses, the ragged arm runs COUNTED). Escape: 0, or a positive "
            f"capacity (2.0 is the presets')."
        )
    if model.moe_lb_mode not in MOE_LB_MODES:
        loud(
            f"moe_lb_mode={model.moe_lb_mode!r} is not a router controller. Known: "
            f"{list(MOE_LB_MODES)} — `sign` the LFB bias integrator "
            f"(`-moe_lb_coef·sign(L−mean L)`, every existing preset), `prop` its "
            f"proportional form (`-moe_lb_gamma·(L−mean L)/max(1,mean L)`), `aux` "
            f"the Switch/GShard loss term instead of a bias. Escape: name one of "
            f"those, or land the controller and its gate before the config can lie "
            f"about it."
        )
    if model.moe_lb_gamma < 0.0:
        loud(
            f"moe_lb_gamma={model.moe_lb_gamma}: the proportional controller's step "
            f"is subtracted from the bias, so a negative step REINFORCES the "
            f"imbalance it is meant to cancel (the busiest expert gains bias). "
            f"Escape: 0.0 (off), or a positive step (0.05 is the arm's value)."
        )
    if model.moe_lb_clamp <= 0.0:
        loud(
            f"moe_lb_clamp={model.moe_lb_clamp}: the prop controller's bound is a "
            f"LOGIT budget and must be positive (`clamp_(-c, c)` with c <= 0 is the "
            f"wrong band). Escape: a positive bound (2.0 is the arm's)."
        )
    if model.moe_aux_coef < 0.0:
        loud(
            f"moe_aux_coef={model.moe_aux_coef}: a negative load-balance weight "
            f"rewards exactly the imbalance the term penalizes (`E·Σ_e f_e·P_e` is "
            f"minimized by a uniform load). Escape: 0.0, or a positive weight "
            f"(0.01 is Switch's)."
        )
    if model.moe_lb_mode == "aux" and model.moe_topk <= 0:
        loud(
            f"moe_lb_mode='aux' with moe_topk={model.moe_topk}: the aux term is a "
            f"function of the DISPATCH (`f_e` is the fraction of the window's "
            f"top-k assignments expert e took), and topk=0 is the dense blend, "
            f"which dispatches nothing — the arm would be inert while its counter "
            f"claimed it ran. Escape: moe_topk > 0, or moe_lb_mode='sign'."
        )
    if model.moe_kernel not in MOE_KERNELS:
        loud(
            f"moe_kernel={model.moe_kernel!r} is not an arm. Known: "
            f"{list(MOE_KERNELS)} — `padded` the cap-padded bmm bank (reference "
            f"and fallback), `triton` the unpadded grouped GEMM "
            f"(model/moe_kernels.py), `naive` the per-expert loop, `batched` the "
            f"same [E, cap, ·] bank with cap = this batch's largest group (the "
            f"dropless arm), `bucketed` the same bank split into row-count classes "
            f"(`[n_b, cap_b, ·]` per bucket, so a collapsed router cannot ask for "
            f"an E-many-expert bank). Escape: name one of those, or add the arm "
            f"before the config can lie about it."
        )
    if model.moe_gather not in MOE_GATHERS:
        loud(
            f"moe_gather={model.moe_gather!r} is not an arm. Known: "
            f"{list(MOE_GATHERS)} — `off` is today's pass (the permute materializes "
            f"`xg [R, d]`), `kernel` reads those rows THROUGH the row map "
            f"(model/moe_sonic.py, SonicMoE §5.1.1). Escape: name one of those."
        )
    if model.moe_gather == "kernel":
        # The three facts the arm's evidence does not cover, each with the code
        # that makes it true. A combination this arm refuses is not a numeric
        # impossibility; it is a route nobody has measured, and ADR-0011 says a
        # route is named before it runs, not after a run is attributed to it.
        if not model.bf16:
            loud(
                "moe_gather='kernel' with bf16=false: the gather arm is wired on the "
                "bf16 route only. Its launcher uses the bf16 tile set (`moe_sonic."
                "TILES`) and passes no `num_stages`, where the shipped 4-byte arm "
                "halves the wide tiles (`moe_kernels.tiles_for`) and passes "
                "`num_stages=1` (`_launch_stages`) — the recipe it adopted after the "
                "default set measured 196608 B against this card's 101376 B CTA "
                "ceiling. An fp32 gather arm would be a tile/stage choice nobody has "
                "launched. Escape: bf16=true (the b1 and medium route), or "
                "moe_gather='off' (the fp32 arm keeps the grouped GEMM, which does "
                "halve)."
            )
        if model.use_byteflow:
            loud(
                "moe_gather='kernel' with use_byteflow=true: the gather ABI is the loop "
                "stack's permute over `[tokens, d]` activations, and the channel runs "
                "that loop over K patch latents (`byteflow.compat_armed`) — an input "
                "path the prototype's byte arithmetic and its parity gates do not "
                "cover. Escape: use_byteflow=false, or moe_gather='off'."
            )
        if model.moe_topk_mode != "off":
            loud(
                f"moe_gather='kernel' with moe_topk_mode={model.moe_topk_mode!r}: the "
                f"adaptive width masks the raised logits and moves the unused slots to "
                f"a sentinel expert PAST the bank (`experts._permute`), so a forward's "
                f"row set is not the constant-width one this ABI was proven on — "
                f"`gather_abi`'s `offs` is a cumsum of `bincount(kept)` and the kernel "
                f"grid is `(n_experts, ...)`, i.e. it never sees that sentinel group. "
                f"Escape: moe_topk_mode='off', or moe_gather='off'."
            )
        if model.factor_qat != "none":
            loud(
                f"moe_gather='kernel' with factor_qat={model.factor_qat!r}: the kernel "
                f"reads the DEQUANTIZED bank — `_banks` applies `factor_qat_project` to "
                f"`gate_u`, `up_u` and `v` before the dtype cast, and those three "
                f"tensors are what `_gather_glu_kernel` loads — so the arm would ride "
                f"the quantization rung, whose alpha anneals every step, and no "
                f"gather-parity gate covers that pair. Escape: factor_qat='none', or "
                f"moe_gather='off'."
            )
    if model.moe_combine not in MOE_COMBINES:
        loud(
            f"moe_combine={model.moe_combine!r} is not an arm. Known: "
            f"{list(MOE_COMBINES)} — `torch` today's zeros+index_add, `fused` the "
            f"one-pass Triton scatter (`moe_kernels._combine_fwd_kernel`). Escape: "
            f"name one."
        )
    if model.moe_combine_dtype not in MOE_COMBINE_DTYPES:
        loud(
            f"moe_combine_dtype={model.moe_combine_dtype!r} is not a dtype. Known: "
            f"{list(MOE_COMBINE_DTYPES)}. Escape: name one."
        )
    if model.moe_combine_dtype == "bf16" and model.moe_combine != "fused":
        loud(
            f"moe_combine_dtype='bf16' with moe_combine={model.moe_combine!r}: the "
            f"torch arm's `y.to(f32) * w` and `index_add_` are fp32 by construction, "
            f"and a bf16 `zeros` + `index_add_` is a DIFFERENT rounding chain (per-row "
            f"bf16 adds) that no gate covers. Escape: moe_combine='fused', or "
            f"moe_combine_dtype='fp32'."
        )
    if model.moe_combine == "fused":
        # Which routed passes reach `moe_kernels.combine` (the only caller of
        # `_combine_fwd_kernel`): the gather arm always, the grouped GEMM (the
        # `triton` arm and the dropless route's default), and nothing else — the
        # padded/batched/bucketed banks have their own `index_add_`, and the
        # reference loop has its own scatter. An arm asked for and not reachable
        # is the ADR-0019 defect this refusal exists for.
        reachable = (
            model.moe_gather == "kernel"
            or model.moe_kernel == "triton"
            or (
                model.moe_capacity == 0
                and model.moe_kernel not in ("batched", "bucketed", "naive")
            )
        )
        if not reachable:
            loud(
                f"moe_combine='fused' with moe_kernel={model.moe_kernel!r} and "
                f"moe_capacity={model.moe_capacity}: that routed pass scatters with its "
                f"own bank/index_add_ and never reaches `moe_kernels.combine`, so the "
                f"arm would be inert while `moe_combine_fused` never appears on the log. "
                f"Escape: moe_capacity=0 (the dropless grouped GEMM), moe_kernel='triton', "
                f"moe_gather='kernel', or moe_combine='torch'."
            )
    if model.moe_combine_dtype == "bf16" and not model.bf16:
        loud(
            "moe_combine_dtype='bf16' with bf16=false: the routed activation is fp32, "
            "so a bf16 store would round a value the model deliberately kept at 4 "
            "bytes and the `weighted.to(dt)` cast it removes is a no-op. Escape: "
            "bf16=true, or moe_combine_dtype='fp32'."
        )
    if model.moe_permute not in MOE_PERMUTES:
        loud(
            f"moe_permute={model.moe_permute!r} is not an algorithm. Known: "
            f"{list(MOE_PERMUTES)} — `argsort` today's stable sort, `count` the "
            f"stable counting sort (`model/route_fused.py`). Escape: name one."
        )
    if model.moe_route_fused:
        # The three shapes the fused head has no code for, each stated as what
        # it would otherwise cost a reader: a silent non-fusion, or a value the
        # kernel does not produce.
        if model.moe_topk <= 0:
            loud(
                f"moe_route_fused=true with moe_topk={model.moe_topk}: the dense "
                f"blend runs every expert and has no top-k to fuse — the flag would "
                f"be inert while its counter claimed it ran. Escape: moe_topk > 0, "
                f"or moe_route_fused=false."
            )
        if model.moe_topk_mode != "off":
            loud(
                f"moe_route_fused=true with moe_topk_mode={model.moe_topk_mode!r}: the "
                f"adaptive width masks the raised logits, moves unused slots to a "
                f"sentinel expert and ZEROES their gate — the fused kernel selects a "
                f"constant width from the raised row and would drop that mask. "
                f"Escape: moe_topk_mode='off', or moe_route_fused=false."
            )
        if model.moe_lb_mode == "aux":
            loud(
                "moe_route_fused=true with moe_lb_mode='aux': the Switch/GShard term "
                "reads the softmax `probs`, and the fused route deliberately does not "
                "materialize them (that omission is the traffic it saves). Escape: "
                "moe_lb_mode='prop' or 'sign', or moe_route_fused=false."
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
    if model.share_u_mode != "off" and not model.use_tsct:
        loud(
            f"share_u_mode={model.share_u_mode!r} with use_tsct=false: the shared "
            f"basis is a TSCT input factor (`u [d, rank]`), and with the "
            f"projections dense there is no factor basis to share — the flag would "
            f"name a mechanism the model does not have. Escape: use_tsct=true, or "
            f"share_u_mode='off'."
        )
    if model.share_u_mode != "off" and model.factor_qat not in FACTOR_QATS:
        # The shared basis is projected with ONE α over the whole tensor
        # (`dim=None`): every rung in FACTOR_QATS supports that, including the
        # dictionary gather. A future rung that needs a per-expert slice of a
        # shared tensor is the combination this seam refuses, named here rather
        # than silently degraded in the forward.
        loud(
            f"share_u with factor_qat={model.factor_qat!r} "
            f"(share_u_mode={model.share_u_mode!r}): the shared basis is projected "
            f"with ONE α over the whole tensor, and this rung is not in "
            f"{list(FACTOR_QATS)}. Escape: name a known rung, or factor_qat='none'."
        )
    if model.share_u_mode == "dict":
        if model.share_u_rank < 2:
            loud(
                f"share_u_mode='dict' with share_u_rank={model.share_u_rank}: the "
                f"dictionary needs at least TWO columns for an expert to read a "
                f"plane (one column is a single direction, and the two latents "
                f"would be a rescaling of it). Escape: share_u_rank >= 2, or "
                f"share_u_mode='plane'."
            )
        if model.rank >= model.share_u_rank:
            loud(
                f"share_u_mode='dict' with rank={model.rank} >= "
                f"share_u_rank={model.share_u_rank}: each expert reads `rank` "
                f"COLUMNS of the dictionary, so a width at (or past) the "
                f"dictionary leaves no subset to choose and the index could read "
                f"out of range — every expert would see the whole plane this mode "
                f"exists to split. Escape: share_u_rank > rank (64 at medium), or "
                f"share_u_mode='plane'."
            )
    # ---- TSCT-2 (docs/architecture/2026-11-05-tsct-2.md) ------------------
    if model.tsct_expert_basis < 0:
        loud(
            f"tsct_expert_basis={model.tsct_expert_basis} is negative: it is the "
            f"per-expert selector's output width (0 = off). Escape: 0, or rank "
            f"({model.rank})."
        )
    if model.tsct_expert_basis:
        if model.share_u_mode != "plane":
            loud(
                f"tsct_expert_basis={model.tsct_expert_basis} with "
                f"share_u_mode={model.share_u_mode!r}: the selector is applied "
                f"INSIDE one shared plane, and only 'plane' has that plane. "
                f"('off' already gives every expert its own [d, rank] basis, so "
                f"the selector would be a different mechanism; 'dict' already "
                f"gives each expert a fixed column subset.) Escape: "
                f"share_u_mode='plane', or tsct_expert_basis=0."
            )
        if model.tsct_expert_basis != model.rank:
            loud(
                f"tsct_expert_basis={model.tsct_expert_basis} != rank="
                f"{model.rank}: the selector's output IS the GLU latent width, "
                f"and the expert's write is `v [rank, f]`. A narrower selector "
                f"would change `v` and therefore the model, not add a subspace "
                f"selector. Escape: tsct_expert_basis={model.rank} (the arm's "
                f"value), or change rank and re-price the whole bank."
            )
        if model.shared_rank <= model.rank:
            loud(
                f"tsct_expert_basis with shared_rank={model.shared_rank} <= "
                f"rank={model.rank}: the wider plane the selector selects FROM is "
                f"`shared_rank` (192 in every real preset), and a plane no wider "
                f"than the latent leaves nothing to select. Escape: shared_rank > "
                f"rank, or tsct_expert_basis=0."
            )
        if model.factor_qat != "none":
            loud(
                f"tsct_expert_basis with factor_qat={model.factor_qat!r}: the "
                f"shared plane is quantized ONCE per site before the experts read "
                f"it, and a per-expert selector after that quantization is a new "
                f"α/STE interaction with no gate. Escape: factor_qat='none', or "
                f"tsct_expert_basis=0."
            )
        if model.moe_shared_compute:
            loud(
                "tsct_expert_basis with moe_shared_compute=true: that arm projects "
                "the tokens through ONE `[d, 2·rank]` pair and selects the halves "
                "with a constant `[E, 2·rank, rank]`; a per-expert `w` has no "
                "constant selector. Escape: moe_shared_compute=false, or "
                "tsct_expert_basis=0."
            )
    if model.tsct_expert_lr < 0.0:
        loud(
            f"tsct_expert_lr={model.tsct_expert_lr} is negative: it is a "
            f"multiplier on the routed counts (0 = off). Escape: >= 0."
        )
    if model.tsct_expert_revive < 0:
        loud(
            f"tsct_expert_revive={model.tsct_expert_revive} is negative: it is a "
            f"cadence in optimizer steps (0 = off). Escape: >= 0."
        )
    if model.tsct_revive_noise < 0.0:
        loud(
            f"tsct_revive_noise={model.tsct_revive_noise} is negative: sigma is "
            f"a fraction of the donor's std. Escape: >= 0."
        )
    if model.tsct_expert_lr or model.tsct_expert_revive:
        if model.n_experts < 2:
            loud(
                f"tsct_expert_lr={model.tsct_expert_lr} / "
                f"tsct_expert_revive={model.tsct_expert_revive} with "
                f"n_experts={model.n_experts}: both arms are per-expert and a "
                f"bank of one has no other expert to starve, revive, or reweight. "
                f"Escape: n_experts >= 2, or both flags 0."
            )
        if model.moe_lb_mode == "aux":
            loud(
                "tsct_expert_lr / tsct_expert_revive with moe_lb_mode='aux': the "
                "aux arm adds the Switch/GShard loss and `_lfb_step` returns "
                "before it accumulates `lfb_counts`, so there are no counts to "
                "read. Escape: moe_lb_mode='sign' or 'prop', or both flags 0."
            )
        lb_step = model.moe_lb_gamma if model.moe_lb_mode == "prop" else model.moe_lb_coef
        if lb_step == 0.0:
            loud(
                f"tsct_expert_lr / tsct_expert_revive with "
                f"moe_lb_mode={model.moe_lb_mode!r} and a zero controller step "
                f"({'moe_lb_gamma' if model.moe_lb_mode == 'prop' else 'moe_lb_coef'}"
                f"=0): `_lfb_step` returns before it accumulates `lfb_counts`, so "
                f"the counts the arms read would stay zero. Escape: set the "
                f"controller step positive (prop: moe_lb_gamma=0.002), or both "
                f"flags 0."
            )
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
    if model.dense_qat not in DENSE_QATS:
        loud(
            f"dense_qat={model.dense_qat!r} is not an arm. Known: "
            f"{list(DENSE_QATS)} — the SAME projector as `factor_qat` "
            f"(`tsct.dense_qat_project` IS `factor_qat_project`), applied to "
            f"every plain nn.Linear/DenseLinear weight in the forward: `none` "
            f"the master untouched, `ternary` b1.58 STE (arXiv 2402.17764), "
            f"`binary` LittleBit signs (arXiv 2506.13771), `mxfp4` the OCP MX "
            f"e2m1 codec (spec §5.2, mxfp.py), `ternary_s3t` S3T stochastic "
            f"rounding (arXiv 2412.04787). Escape: name one of those, or land "
            f"the arm and its gold test (tests/test_dense_qat.py) BEFORE the "
            f"config can lie about it."
        )
    if model.factor_qat == "none" and model.dense_qat == "none" and model.factor_qat_alpha != 1.0:
        loud(
            f"factor_qat_alpha={model.factor_qat_alpha} with factor_qat='none' "
            f"and dense_qat='none' is inert — both quantizers are off, so the "
            f"blend has nothing to blend toward. Escape: set factor_qat or "
            f"dense_qat, or leave alpha at 1.0."
        )
    if model.factor_qat_seed != 0 and "ternary_s3t" not in (
        model.factor_qat,
        model.dense_qat,
    ):
        loud(
            f"factor_qat_seed={model.factor_qat_seed} with "
            f"factor_qat={model.factor_qat!r} / dense_qat={model.dense_qat!r} "
            f"is inert — only `ternary_s3t` draws from a generator (the other "
            f"rungs are deterministic projections), on either path. Escape: "
            f"factor_qat='ternary_s3t' or dense_qat='ternary_s3t', or leave the "
            f"seed at 0."
        )
    if model.act_quant not in ACT_QUANTS:
        loud(
            f"act_quant={model.act_quant!r} is not an activation format. Known: "
            f"{list(ACT_QUANTS)} — `none` off, `int8`/`int4` symmetric integer "
            f"levels (int8 per-token absmax: 2B4T arXiv 2504.12285; int4 127/7 "
            f"and the group rule are our convention — formulas in "
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
    if model.mhc_sinkhorn not in ("logsumexp", "log_softmax"):
        loud(
            f"mhc_sinkhorn={model.mhc_sinkhorn!r} is not a Sinkhorn normaliser. "
            f"Known: 'logsumexp' (the shipped bits) and 'log_softmax' (the same "
            f"expression as one ATen op). Escape: name one of those, or leave it "
            f"at its default 'logsumexp'."
        )
    if model.mhc_sinkhorn != "logsumexp" and not model.use_mhc:
        loud(
            f"mhc_sinkhorn='{model.mhc_sinkhorn}' with use_mhc=false is inert — "
            f"there is no lane chain and no Sinkhorn to normalise, so the field "
            f"would name an arm that never ran. Escape: use_mhc=true, or leave "
            f"mhc_sinkhorn at its default 'logsumexp'."
        )
    if model.use_mhc:
        for name, why in (
            (
                "loom",
                "LOOM adds a second, un-A/B'd per-iteration gate to the write "
                "(2610.01153); an mHC result from that run would be attributable to "
                "neither mechanism",
            ),
            (
                "use_byteflow",
                "the channel runs the loop over K patch latents, so the lanes would "
                "be patch lanes and the placement a different experiment",
            ),
        ):
            if getattr(model, name):
                loud(
                    f"use_mhc with {name}=true: {why}. Escape: {name}=false, or "
                    f"use_mhc=false — mHC's A/B (scripts/ab_mhc.py) moves ONE write "
                    f"axis at a time, and AttnRes (`use_attnres`) is the only "
                    f"partner it composes with (scripts/ab_mhc_attnres.py)."
                )
        if model.mhc_streams != 2:
            loud(
                f"mhc_streams={model.mhc_streams}: only K=2 ships. It is the K the "
                f"φ evidence used (2604.21106 §5.2) and 2409.19606 Tab. 1's knee; "
                f"mHC's own App. A.1 default of 4 is the named follow-up row, not a "
                f"shipped rung. Escape: mhc_streams=2, or land the n=4 arm and its "
                f"gate BEFORE the config can lie about it."
            )
    elif model.mhc_streams != 2:
        loud(
            f"mhc_streams={model.mhc_streams} with use_mhc=false is inert — the "
            f"field would name a K no lane carries. Escape: use_mhc=true (K=2), or "
            f"leave mhc_streams at its default 2."
        )
    if model.fused_epilogue and not model.use_mhc:
        loud(
            "fused_epilogue=true with use_mhc=false: the fused read kernel IS the "
            "lane chain (RMSNorm over K lanes, H_pre's mix, and the body's normed "
            "read), so with no lanes there is nothing it can replace and the flag "
            "would report an arm that never ran. Escape: use_mhc=true, or "
            "fused_epilogue=false."
        )
    if model.fused_latent_proj and (not model.use_kda or model.attn_kind != "kda"):
        loud(
            "fused_latent_proj=true without the KDA arm: the four projections it "
            "concatenates (q/k/v/g) are KDAAttention's, and `attn_kind='full'` "
            "builds FullAttention instead (with `use_kda=false` there is no "
            "attention at all) — the flag would name a GEMM that never runs. "
            "Escape: use_kda=true and attn_kind='kda', or fused_latent_proj=false."
        )
    if model.fused_residual and not model.use_mhc:
        loud(
            "fused_residual=true with use_mhc=false: the fused kernels ARE the mHC "
            "residual chain (H_pre's lane mix, the Sinkhorn-projected H_res, the "
            "H_post write, the lane readout) — with no lanes there is nothing they "
            "can replace, so the flag would report an arm that never ran. That also "
            "covers the AttnRes-ONLY write: it mixes over a GROWING source list "
            "([h0, y_1, …, y_i], re-scored every iteration), so it is not one "
            "iteration's local chain either. Escape: use_mhc=true (the composition, "
            "where the write is local: u_0 = 0, u_1 = y), or fused_residual=false "
            "(the ReZero write is three elementwise ops and needs no kernel)."
        )
    if model.residual_dtype not in RESIDUAL_DTYPES:
        loud(
            f"residual_dtype={model.residual_dtype!r} is not a dtype. Known: "
            f"{', '.join(RESIDUAL_DTYPES)}. Escape: one of those; the value names "
            f"the STORAGE dtype of the loop's residual stream "
            f"(scripts/precision_census.py prices it)."
        )
    if model.residual_dtype == "bf16" and model.fused_residual:
        loud(
            "residual_dtype='bf16' with fused_residual=true: the fused residual "
            "chain (`model/fused_residual.py`: _Pre/_Write and their two VJPs) "
            "reproduces the fp32 arithmetic only — its kernel refuses a non-fp32 "
            "lane (`unsupported_reason`), and the lanes it would read are now "
            "bf16. Silently falling back would report an arm that never ran, so "
            "this is refused. Escape: fused_residual=false (the eager mHC chain "
            "widens the lanes itself), or residual_dtype='fp32'."
        )
    if model.residual_dtype == "bf16" and model.fused_epilogue:
        loud(
            "residual_dtype='bf16' with fused_epilogue=true: the fused epilogue "
            "read kernel (`model/fused_residual.py::epilogue_read`) is fp32-only "
            "and refuses a bf16 lane. Escape: fused_epilogue=false, or "
            "residual_dtype='fp32'."
        )
    if model.residual_dtype == "bf16" and model.fused_kda_tail:
        loud(
            "residual_dtype='bf16' with fused_kda_tail=true: the tail's `combine` "
            "kernel (`model/fused_kda.py::combine`, the y = attn + branch + ffn "
            "pass) is fp32-only, and the loop's three branch writes are now bf16. "
            "The projection tail alone is not separately switchable, so the pair "
            "is refused rather than run half its kernels. Escape: "
            "fused_kda_tail=false, or residual_dtype='fp32'."
        )
    # The codec selector is checked even when `use_byteflow = false`: an
    # unknown name must not fall back to attention, and `conv` with nothing to
    # build it is the config-field-that-lies defect (ADR-0019).
    if model.byteflow_codec not in ("attention", "conv"):
        loud(
            f"byteflow_codec={model.byteflow_codec!r} is not a codec. Known: "
            f"'attention' (the ported ByteFlow net, the default) and 'conv' "
            f"(`model/convcodec.py`). Escape: name one of those."
        )
    if model.byteflow_codec == "conv" and not model.use_byteflow:
        loud(
            "byteflow_codec='conv' with use_byteflow=false: the codec is only built "
            "by the byteflow shapes, so the field would name a codec that never runs. "
            "Escape: use_byteflow=true, or byteflow_codec='attention'."
        )
    if model.use_byteflow:
        if model.loop_blocks > 1:
            loud(
                f"loop_blocks={model.loop_blocks} with use_byteflow=true: Option A "
                f"(`compat_armed` false) returns from `DormouseModel` before the "
                f"model tree is built — there is no loop at all, so blocks 1+ would "
                f"never run — and Option B runs this loop over K PATCH LATENTS, a "
                f"different sequence shape whose depth has not been measured with "
                f"distinct bodies. Escape: use_byteflow=false, or loop_blocks=1."
            )
        if model.prelude_layers or model.coda_layers:
            loud(
                f"prelude_layers={model.prelude_layers}/coda_layers="
                f"{model.coda_layers} with use_byteflow=true: the byteflow shapes "
                f"run no byte-level loop to wrap. Option A is a standalone net "
                f"with no loop at all (`bf_net` returns before the model tree), "
                f"and Option B runs this loop over K patch LATENTS "
                f"(`compat_armed`), so the patcher changes the sequence length "
                f"mid-model and a prelude/coda would act on patch states, not the "
                f"bytes they are specified on. Escape: use_byteflow=false, or "
                f"prelude_layers=0 and coda_layers=0."
            )
        if model.layerrope:
            # Before `_validate_byteflow`, so the LayerRoPE refusal is the one a
            # reader sees rather than an unrelated chunker-width error.
            loud(
                "layerrope=true with use_byteflow=true: both byteflow shapes leave "
                "this model's byte-level loop — Option A builds no loop at all and "
                "Option B runs it over K patch latents, so the LayerRoPE norm tree "
                "would never be built while the flag reported it on. Escape: "
                "use_byteflow=false, or layerrope=false."
            )
        _validate_byteflow(model)
    if model.sigreg_weight < 0.0:
        loud(
            f"sigreg_weight={model.sigreg_weight} is negative: it is the "
            f"multiplier on the SIGReg term, not a sign to flip. Escape: 0.0 "
            f"(off) or a positive weight (the research doc's grid: 0.005, 0.05, "
            f"0.1 — docs/research/2026-10-16-jepa-into-dormouse.md §3)."
        )
    if model.jepa_weight < 0.0:
        loud(
            f"jepa_weight={model.jepa_weight} is negative: it scales the whole "
            f"JEPA term (`model/jepa.py`), so a negative value flips the sign of "
            f"both the prediction and its anti-collapse term. Escape: 0.0 (off) "
            f"or a positive weight."
        )
    if model.jepa_reg_weight < 0.0:
        loud(
            f"jepa_reg_weight={model.jepa_reg_weight} is negative: it is the "
            f"anti-collapse weight INSIDE the JEPA term, and a negative value "
            f"would push the predicted latents toward collapse (the opposite of "
            f"`sigreg_loss`'s gradient). Escape: 0.0 (no regulariser) or a "
            f"positive weight (the gate's 0.1)."
        )
    if model.jepa_horizon < 1:
        loud(
            f"jepa_horizon={model.jepa_horizon}: the horizon is a positive count "
            f"of positions to look ahead (1 = the next position). Escape: >= 1, "
            f"and <= seq_len // 8 (the load path checks the window)."
        )
    # The ENERGY head's own coherence (`model/energy.py`, EBT arXiv:2507.02092):
    # the weight gates the module (0 builds nothing), and every other field is
    # a count/scale the loss would silently misuse. The two INTERACTIONS
    # (byteflow, dspark) are refusals, never a merge: this lane has one aux
    # objective, and the arm it would have to share a hidden state with is not
    # measured alongside it.
    if model.energy_weight < 0.0:
        loud(
            f"energy_weight={model.energy_weight} is negative: it is λ, the "
            f"multiplier on the margin-InfoNCE term. Escape: 0.0 (off) or a "
            f"positive weight (the A/B's grid is 0.05 / 0.1)."
        )
    if model.energy_negs < 1:
        loud(
            f"energy_negs={model.energy_negs}: the contrastive term needs at least "
            f"one sampled negative per position. Escape: >= 1 (the default 64)."
        )
    if model.energy_temp <= 0.0:
        loud(
            f"energy_temp={model.energy_temp}: it is InfoNCE's temperature, a "
            f"positive scale (the margin is divided by it too). Escape: > 0."
        )
    if model.energy_rank < 1:
        loud(
            f"energy_rank={model.energy_rank}: it is the head's bottleneck width "
            f"in Linear(d, k) -> SiLU -> Linear(k, 1). Escape: >= 1."
        )
    if model.energy_weight > 0.0 and model.dspark_weight > 0.0:
        loud(
            f"energy_weight={model.energy_weight} with dspark_weight="
            f"{model.dspark_weight}: both are aux objectives reading the SAME "
            f"hidden state (`model.norm(out)`), and their interaction is not "
            f"measured — a run with both armed could not attribute a BPB change "
            f"to either. Escape: one weight per run, or land the joint A/B first."
        )
    # The dspark arm's OWN coherence (audit MINOR 12, 2026-10-17): these used to
    # live inside `_validate_byteflow`, which only runs under `use_byteflow`, so
    # the default path accepted `dspark_weight=-1.0` / `dspark_k=0` /
    # `dspark_stride=0` — a field that lies about being refused (§1.7). The
    # byteflow-only interaction (`dspark_weight>0` with the channel's forward)
    # stays in `_validate_byteflow`.
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
    # The MTP arm's own coherence (`model/mtp.py`): the weight with no depth
    # asks for a term that has no module to average, and a depth above 8 is
    # outside the constructor's contract. Both are refusals, never clamps.
    if model.mtp_k < 0 or model.mtp_k > 8:
        loud(
            f"mtp_k={model.mtp_k} outside 0..=8: it is the number of future-byte "
            f"modules (DeepSeek-V3 §3.2). Escape: 0 (off, nothing is built) or "
            f"1..=8 (the constructor's ceiling, `mtp.MAX_K`)."
        )
    if model.mtp_weight > 0.0 and model.mtp_k == 0:
        loud(
            f"mtp_weight={model.mtp_weight} with mtp_k=0: the weight asks for the "
            f"MTP term but there are no future-byte modules to average (the head "
            f"is built IFF the weight is on, and K >= 1). Escape: mtp_k >= 1, or "
            f"mtp_weight=0.0."
        )
    if model.mtp_soft and not model.mtp_head_norm:
        loud(
            "mtp_soft=true with mtp_head_norm=false: the soft-MTP identity needs "
            "each head's OWN normalized hidden (the reference's per-head final "
            "RMSNorm); without it a later head's coupling flows back through the "
            "sequential projection. Escape: mtp_head_norm=true, or "
            "mtp_soft=false."
        )
    if model.mtp_soft and model.mtp_weight <= 0.0:
        loud(
            f"mtp_soft=true with mtp_weight={model.mtp_weight}: soft-MTP is a "
            f"GRADIENT-ROUTING rule for the MTP term, so with no MTP head there "
            f"is no hidden state to route into (the head is built IFF the weight "
            f"is on). Escape: mtp_weight > 0 with mtp_k >= 1, or mtp_soft=false."
        )
    if model.mtp_weight < 0.0:
        loud(
            f"mtp_weight={model.mtp_weight} is negative: it is λ, the multiplier "
            f"on the MTP CE. Escape: 0.0 (off) or a positive weight."
        )

    # The HALT HEAD's own coherence (`model/halt.py`). The weight gates the
    # module (0 builds nothing), the mode gates INFERENCE, and the cap is only
    # meaningful on the arm that runs it — so the `halt_max_iter <= max_iter`
    # refusal applies IFF a halt is requested. The default `max_iter=4` /
    # `halt_max_iter=8` pair must stay a legal config (every snapshot loads it).
    if model.halt_weight < 0.0:
        loud(
            f"halt_weight={model.halt_weight} is negative: it is λ, the multiplier "
            f"on the halt BCE. Escape: 0.0 (off) or a positive weight."
        )
    if model.halt_eps < 0.0:
        loud(
            f"halt_eps={model.halt_eps} is negative: it is a CE improvement in "
            f"nats, and a negative threshold would ask the head to stop when "
            f"thinking HURTS by more than |eps|. Escape: 0.0 or a small positive "
            f"value (the default 0.005)."
        )
    if model.halt_min_iter < 1:
        loud(
            f"halt_min_iter={model.halt_min_iter}: the first iteration is 1, so a "
            f"floor below it names a decision the loop cannot make. Escape: >= 1."
        )
    if model.halt_mode:
        if model.halt_weight <= 0.0:
            loud(
                f"halt_mode=true with halt_weight={model.halt_weight}: the "
                f"checkpoint has no halt head (the head is built IFF the weight "
                f"is on), so there is nothing to decide the stop. Escape: a "
                f"checkpoint trained with halt_weight>0, or halt_mode=false."
            )
        if model.halt_max_iter > model.max_iter:
            loud(
                f"halt_max_iter={model.halt_max_iter} > max_iter={model.max_iter}: "
                f"the inference cap cannot exceed the depth the loop was BUILT "
                f"for (the `iter_embed`/gate rows stop at max_iter), so the flag "
                f"would name thinking that has no slot. Escape: max_iter >= "
                f"{model.halt_max_iter} (the depth-8 arm), or halt_max_iter <= "
                f"{model.max_iter}."
            )
        if model.halt_max_iter < model.halt_min_iter:
            loud(
                f"halt_max_iter={model.halt_max_iter} < halt_min_iter="
                f"{model.halt_min_iter}: the cap must admit the floor. Escape: "
                f"swap them."
            )

    if model.layerrope:
        if model.d_model % 2:
            loud(
                f"layerrope=true with d_model={model.d_model}: the rotation acts on "
                f"adjacent channel PAIRS, so the width must be even. Escape: an even "
                f"d_model, or layerrope=false."
            )
        if model.layerrope_form not in ("v1", "paper"):
            loud(
                f"layerrope_form={model.layerrope_form!r} is not a LayerRoPE form. "
                f"Known: 'v1' (shared gamma, per-depth amp/theta, read sites only) "
                f"and 'paper' (gamma + 4 scalars per site, read AND write, RoPE "
                f"spectrum). Escape: name one of those, or layerrope=false."
            )
        if model.layerrope_index not in ("iteration", "position"):
            loud(
                f"layerrope_index={model.layerrope_index!r} is not a depth index. "
                f"Known: 'iteration' (the ACTUAL loop iteration, v1's rule) and "
                f"'position' (the body's position in the network, independent of "
                f"the loop iteration). Escape: name one of those, or "
                f"layerrope_form='v1' (which reads no index)."
            )
        if model.layerrope_init not in ("paper", "identity"):
            loud(
                f"layerrope_init={model.layerrope_init!r} is not a LayerRoPE init. "
                f"Known: 'paper' (the paper's beta_init, depth-scaled from step 1) "
                f"and 'identity' (every depth's gain IS gamma). Escape: name one of "
                f"those, or layerrope_form='v1' (which reads no init)."
            )
        if model.layerrope_base <= 0.0:
            loud(
                f"layerrope_base={model.layerrope_base}: the RoPE spectrum is "
                f"base^(-2j/d), so the base must be positive. Escape: base > 0 "
                f"(100.0 is the paper's), or layerrope_form='v1'."
            )
        if model.layerrope_form == "v1":
            # The paper form is the only reader of these; a non-default value on
            # the v1 path would be a config naming an arm the forward never runs
            # (the same rule the `layerrope=false` block below applies).
            for key in (
                "layerrope_beta_init",
                "layerrope_base",
                "layerrope_index",
                "layerrope_init",
            ):
                value = getattr(model, key)
                default = ModelConfig.__dataclass_fields__[key].default
                if value != default:
                    loud(
                        f"{key}={value!r} with layerrope_form='v1': the v1 path "
                        f"reads no beta/base/index/init — the paper form is their "
                        f"only reader, so this would name an arm the forward never "
                        f"runs. Escape: layerrope_form='paper', or {key}={default!r}."
                    )
            if model.layerrope_amp_init <= 0.0:
                loud(
                    f"layerrope_amp_init={model.layerrope_amp_init}: `amp_i` is the "
                    f"MAGNITUDE of the gain, so a non-positive value would zero or flip "
                    f"every normalised state. Escape: amp_init > 0 (1.0 is the identity), "
                    f"or layerrope=false."
                )
        else:
            for key in (
                "layerrope_amp_init",
                "layerrope_theta_init",
                "layerrope_per_site",
                "layerrope_learn_theta",
            ):
                value = getattr(model, key)
                default = ModelConfig.__dataclass_fields__[key].default
                if value != default:
                    loud(
                        f"{key}={value!r} with layerrope_form='paper': the paper "
                        f"form has no `amp`/`theta` table and no per-site variant "
                        f"(its magnitude and rotation ARE the four scalars per "
                        f"site), so this field would reach no code. Escape: "
                        f"layerrope_form='v1', or {key}={default!r}."
                    )
    else:
        for key in (
            "layerrope_amp_init",
            "layerrope_theta_init",
            "layerrope_per_site",
            "layerrope_learn_theta",
            "layerrope_form",
            "layerrope_beta_init",
            "layerrope_base",
            "layerrope_index",
            "layerrope_init",
        ):
            value = getattr(model, key)
            default = ModelConfig.__dataclass_fields__[key].default
            if value != default:
                loud(
                    f"{key}={value!r} without layerrope=true: the subfield reaches no "
                    f"code, so the config would name an arm the forward never runs. "
                    f"Escape: layerrope=true, or {key}={default!r}."
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
    if model.halt_weight > 0.0:
        loud(
            "use_byteflow with halt_weight>0: the halt head reads the byte loop's "
            "per-iteration readout, and the channel runs the loop over K patch "
            "latents with no per-byte CE to build a target from. Escape: "
            "halt_weight = 0.0, or drop use_byteflow."
        )
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
    if model.byteflow_codec == "conv":
        s = model.byteflow_patch_bytes
        if s < 1:
            loud(
                f"byteflow_patch_bytes={s} with byteflow_codec='conv': it is a count "
                f"of bytes per patch, so it must be >= 1. Escape: 8 (the default), or "
                f"a positive divisor of the training window."
            )
        if model.byteflow_max_bytes % s:
            loud(
                f"byteflow_max_bytes={model.byteflow_max_bytes} is not a multiple of "
                f"byteflow_patch_bytes={s}: the conv patchify covers "
                f"floor(max_bytes/s) patches and the remainder would be silently "
                f"dropped. Escape: a patch_bytes that divides the window, or lower "
                f"byteflow_max_bytes."
            )
        if model.byteflow_k_tokens > model.byteflow_max_bytes // s:
            loud(
                f"byteflow_k_tokens={model.byteflow_k_tokens} > the conv codec's patch "
                f"count {model.byteflow_max_bytes // s} "
                f"(byteflow_max_bytes={model.byteflow_max_bytes} / "
                f"byteflow_patch_bytes={s}): one latent per selected patch, so K above "
                f"the patch count has nothing to select. Escape: lower K / "
                f"byteflow_patch_bytes, or raise the window."
            )
        if model.byteflow_logdet:
            loud(
                "byteflow_logdet=true with byteflow_codec='conv': the exact log-det "
                "rate scores the ATTENTION codec's byte positions; the conv codec "
                "selects patches with the L2 streaming gains, so the flag would be "
                "inert while the config reported the exact route. Escape: "
                "byteflow_logdet=false, or byteflow_codec='attention'."
            )
    elif model.byteflow_patch_bytes != 8:
        loud(
            f"byteflow_patch_bytes={model.byteflow_patch_bytes} with "
            f"byteflow_codec='attention': the field configures the conv codec's "
            f"patchify, and the attention net has no patch stride, so the value would "
            f"be inert. Escape: byteflow_codec='conv', or leave it at 8."
        )
    for d, h, side in (
        (model.byteflow_d_local, model.byteflow_heads_local, "local"),
        (model.byteflow_d_global, model.byteflow_heads_global, "global"),
    ):
        if h < 1 or d % h != 0:
            loud(
                f"byteflow_{side} width {d} does not divide by its head count {h}: "
                f"the head split must tile the width. Escape: fix the byteflow_* widths."
            )
    # bf16 is ALLOWED here: the net runs its matmuls in bf16 with fp32 masters
    # (`byteflow.py`'s precision rule), the same policy as the loop. The ONE
    # pair it cannot carry is the exact-rate chunker: a bf16 Cholesky is
    # unstable, and the default L2 path never enters it.
    if model.bf16 and model.byteflow_logdet:
        loud(
            "bf16 + byteflow_logdet: the exact-rate chunker factorizes "
            "`I + c·HᵀH` with `torch.linalg.cholesky`, and a bf16 Cholesky is "
            "unstable — the selection would change on a numerical factorization "
            "error with no gradient to correct it. Escape: `byteflow_logdet = "
            "false` (the L2 fast path is the default and is bf16-safe), or "
            "`bf16 = false`."
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
    # `dense_qat`'s seam is the MODEL's dense weights; the ByteFlow net is a
    # separate module tree (`byteflow.py`: qkv/out, the SwiGLU pair, the chunk
    # projections, and the channel's in_proj/out_proj) that does NOT route
    # through `dense_qat_project`. Refused rather than silently half-quantized:
    # a reader of the config would otherwise believe the forward ran ternary
    # weights everywhere while the net's own projections stayed in their dtype.
    if model.dense_qat != "none":
        loud(
            f"dense_qat={model.dense_qat!r} with use_byteflow=true: the ByteFlow net's "
            f"own dense projections (`byteflow.py`: qkv/out, the SwiGLU pair, the "
            f"chunker's proj, the channel's in_proj/out_proj) are not on the "
            f"dense_qat seam, so the flag would report a quantized forward while those "
            f"weights stay in their own dtype. Escape: use_byteflow=false, or "
            f"dense_qat='none'."
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
    # JEPA predicts the loop's byte-position readout `h[p+h]` from `h[p]`; the
    # channel runs the loop over K patch latents, so there are no byte positions
    # to shift. Refused, never silently dropped (ADR-0019).
    if model.jepa_weight > 0.0:
        loud(
            f"use_byteflow with jepa_weight={model.jepa_weight}: JEPA predicts the "
            f"loop's byte-position readout `h[p+h]` from `h[p]`, and the channel "
            f"runs the loop over K patch latents where there are no byte positions "
            f"to shift (model/jepa.py). Escape: jepa=false and jepa_weight=0.0, or "
            f"drop use_byteflow."
        )
    if model.energy_weight > 0.0:
        loud(
            "use_byteflow with energy_weight>0: the channel runs the loop over K "
            "patch latents, so there is no per-BYTE position for the contrastive "
            "head to score and no byte state the sampler could read. Escape: "
            "energy_weight = 0.0, or drop use_byteflow."
        )
    if model.dspark_weight > 0.0:
        loud(
            "use_byteflow with dspark_weight>0: the channel's forward carries no "
            "byte ids for the draft window to read. Escape: dspark_weight = 0.0, "
            "or drop use_byteflow."
        )
    if model.mtp_weight > 0.0:
        loud(
            "use_byteflow with mtp_weight>0: the MTP modules read the byte ids and "
            "the loop's byte-position hidden state, and the channel runs the loop "
            "over K patch latents with no byte ids in that forward. Escape: "
            "mtp_weight = 0.0, or drop use_byteflow."
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


def _validate_doc(model: ModelConfig, tcfg: TrainConfig) -> None:
    """`doc_masking` against the window, the byteflow shapes and the aux targets.

    Each is a refusal rather than a silent no-op, for the ADR-0019 reason the
    config layer exists: a flag that is on while the run computes the flat
    objective is a config that lies about what trained. Every message names the
    escape.
    """
    if not tcfg.doc_masking:
        return
    if tcfg.full_graph:
        loud(
            "doc_masking=true with full_graph=true: the capture fills STATIC (x, y) "
            "buffers and the loss mask is a third per-step input it has no slot for — "
            "a replay would train on the first step's mask forever. Escape: "
            "full_graph=false (the eager step), or doc_masking=false."
        )
    if tcfg.segment_len:
        loud(
            f"doc_masking=true with segment_len={tcfg.segment_len}: the segmented "
            f"forward CARRIES the KDA state across cuts, and a carried state has no "
            f"notion of a document boundary — a document shorter than a segment would "
            f"still see the previous document's state at the seam. Escape: "
            f"segment_len=0 (the one-shot window, where one row holds one document), "
            f"or doc_masking=false."
        )
    if model.use_byteflow:
        loud(
            "doc_masking=true with use_byteflow=true: the byteflow shapes patch the "
            "byte stream at LEARNED boundaries (or run no byte loop at all), and a "
            "patch boundary is not a document boundary — the loss mask addresses byte "
            "positions, not patch latents. Escape: use_byteflow=false, or "
            "doc_masking=false."
        )
    for name in ("dspark_weight", "mtp_weight", "jepa_weight", "energy_weight", "halt_weight"):
        weight = float(getattr(model, name))
        if weight > 0.0:
            loud(
                f"doc_masking=true with {name}={weight}: this auxiliary objective has "
                f"its own target positions over the whole window, and the CE loss mask "
                f"does not reach it — it would train across the padding and document "
                f"boundaries the flag exists to remove, while the log reported the arm "
                f"on. Escape: set {name}=0.0, or doc_masking=false."
            )
    if model.moe_topk > 0 and model.moe_capacity > 0:
        loud(
            f"doc_masking=true with moe_topk={model.moe_topk} and "
            f"moe_capacity={model.moe_capacity}: the capacity-padded dispatch decides "
            f"which rows an expert keeps from the WHOLE batch, so a padded row (and a "
            f"document's masked last byte) can take capacity from a real token in "
            f"another row — the mask would hold on the loss but not on the forward. "
            f"Escape: moe_capacity=0 (the dropless route, where every assignment "
            f"reaches its expert — the b1 and A/B default), or doc_masking=false."
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
    if model.csa_coarse_chunk:
        loud(
            f"segment_len={n} with csa_coarse_chunk={model.csa_coarse_chunk}: the "
            f"coarse level is a dense ring carried across BYTES, and what a segment "
            f"cut does to a half-full coarse chunk is undecided — carrying it is a "
            f"model change, dropping it silently forgets the segment. Escape: "
            f"csa_coarse_chunk=0 (fine level only), or segment_len=0."
        )
    if model.use_csa:
        loud(
            f"segment_len={n} with use_csa: the CSA state is a chunk ring carried "
            f"across BYTES, and what a segment cut should do to a half-full chunk is "
            f"undecided — carrying it at the cut is a model change, and dropping it "
            f"silently forgets the segment. Escape: use_csa=false, or segment_len=0."
        )
    if model.attn_kind == "full":
        loud(
            f"segment_len={n} with attn_kind='full': exact attention has no O(1) "
            f"boundary state, so a segment would have to carry every preceding "
            f"segment's K/V (O(window) per iteration slot) — the memory this flag "
            f"exists to save, and the reason the compressed arm is the one it was "
            f"built on. Escape: segment_len=0, or attn_kind='kda'."
        )
    if model.attn_window:
        loud(
            f"segment_len={n} with attn_window={model.attn_window}: the local band "
            f"reaches {model.attn_window} positions back, and a segment cut hides "
            f"everything before it — so the first positions of every segment would "
            f"read a NARROWER band than the one-shot forward's, i.e. the "
            f"segmentation would change the model, not only the memory (the same "
            f"reason attn_kind='full' is refused here). Carrying the K/V across the "
            f"cut is the O(window·d) state segments exist to avoid. Escape: "
            f"attn_window=0, or segment_len=0."
        )
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
    if model.halt_weight > 0.0 or model.halt_mode:
        loud(
            f"segment_len={n} with halt_weight={model.halt_weight}/halt_mode="
            f"{model.halt_mode}: the segmented forward runs the loop per segment, "
            f"so the halt head's per-iteration states would be concatenations of "
            f"segment windows and its target a per-segment mean — two unvalidated "
            f"axes in one run. Escape: halt_weight=0 and halt_mode=false, or "
            f"segment_len=0."
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


def _validate_jepa(model: ModelConfig, tcfg: TrainConfig) -> None:
    """The JEPA arm against the window and the segmented path — both halves.

    The horizon crosses SEGMENT boundaries: the segmented forward runs the loop
    per segment, so `h[p+h]` can come from the next segment's window (a different
    state) or from the carried-state prefix, which is a second unvalidated
    objective. The horizon also has to fit the window with room for a sample:
    `seq_len // 8` is the widest the research plan names, and a horizon at the
    window's scale leaves almost nothing to predict over. Both are refusals, not
    fallbacks (ADR-0019: a term with nothing to act on must not look alive).
    """
    if model.jepa_weight <= 0.0:
        return
    if tcfg.segment_len > 0:
        loud(
            f"jepa_weight={model.jepa_weight} with segment_len={tcfg.segment_len}: "
            f"JEPA predicts `h[p+h]` from `h[p]`, and the segmented forward cuts "
            f"the window, so the horizon would cross a segment boundary — the "
            f"target would be a state from a different window (or from the carried "
            f"prefix), i.e. a different objective from the one-shot forward. "
            f"Escape: segment_len=0, or jepa_weight=0.0."
        )
    cap = tcfg.seq_len // 8
    if model.jepa_horizon > cap:
        loud(
            f"jepa_horizon={model.jepa_horizon} > seq_len // 8 = {cap} "
            f"(seq_len={tcfg.seq_len}): the horizon must fit the training window "
            f"with room for a predicted sample, otherwise the term reads a handful "
            f"of positions (or none) and the A/B cannot tell it apart from off. "
            f"Escape: a horizon <= {cap}, or jepa_weight=0.0."
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


def rl_config(tcfg: TrainConfig) -> RLConfig:
    """The [`RLConfig`] the RL keys describe — the ONE construction site.

    `load_run` builds one for its refusals and `post.rl_train` builds one to run
    with, both through here: a second mapping would be a second set of rules, and
    `RLConfig.__post_init__` is the only place the X/B/A/δ/s/γ relations are
    checked (ADR-0011: one rule, one place). The import is function-local because
    the post lane imports this module.
    """
    from .post.rl import RLConfig

    return RLConfig(
        quality=tcfg.rl_quality,
        length_penalty=tcfg.rl_len_penalty,
        quantile=tcfg.rl_len_quantile,
        gate=tcfg.rl_len_gate,
        delta=tcfg.rl_len_delta,
        s=tcfg.rl_len_s,
        gamma=tcfg.rl_len_gamma,
        agg=tcfg.rl_agg,
    )


#: The floor of a defensible held-out eval window (`TrainConfig.eval_bytes`).
#: One 64 KB window has a 95 % interval of about ±0.19 BPB on this corpus
#: (docs/reviews/2026-11-06-bpb-accounting.md §3), so a narrower window is a
#: reading of one document, not a BPB. ONE definition: `train/data.py` imports
#: this for the loader and `validate_train` refuses below it.
MIN_EVAL_BYTES = 65_536


def validate_train(tcfg: TrainConfig) -> None:
    """The train-half refusals the CLI and a hand-built `train()` share.

    ONE definition, two callers (`load_run` and `train/loop.py::train`): a
    hand-built `TrainConfig` that skips `load_run` must not be able to run a
    zero-micro-batch step or an EMA outside [0,1) either.
    """
    if tcfg.grad_accum < 1:
        loud(
            f"grad_accum={tcfg.grad_accum}: the field is a COUNT of micro-batches "
            f"per optimizer step, so 1 is one forward/backward and 0 or a negative "
            f"value would mean a step with no gradient at all. Escape: "
            f"grad_accum >= 1 (the default 1)."
        )
    if (tcfg.eval or tcfg.eval_data) and tcfg.eval_batches is None:
        if tcfg.eval_bytes < MIN_EVAL_BYTES:
            loud(
                f"eval_bytes={tcfg.eval_bytes} is below the {MIN_EVAL_BYTES} B "
                f"floor. One 64 KB window has a 95 % interval of about ±0.19 BPB "
                f"on this corpus, so a narrower window is not a measurement. "
                f"Escape: eval_bytes >= {MIN_EVAL_BYTES} (196608 is the default "
                f"and the flagship's window)."
            )
    if tcfg.eval_batches is not None and tcfg.eval_batches < 1:
        loud(
            f"eval_batches={tcfg.eval_batches}: the legacy window override is a "
            f"positive number of batches. Escape: eval_batches >= 1, or leave it "
            f"unset and set eval_bytes."
        )
    if tcfg.eval_offset < 0:
        loud(
            f"eval_offset={tcfg.eval_offset}: the eval window starts at a BYTE "
            f"offset into the held-out file, so a negative value is a rewind "
            f"outside the file. Escape: eval_offset >= 0 (0 = the head)."
        )
    if not tcfg.agc_clip >= 0.0:
        loud(
            f"agc_clip={tcfg.agc_clip}: AGC's λ multiplies `||p|| / max(||g||, 1e-3)` "
            f"and is clamped to <= 1.0, so a negative threshold asks for a negative "
            f"gradient and NaN is not a threshold. Escape: agc_clip >= 0 "
            f"(0.0 is OFF, the default)."
        )
    if not tcfg.adam_eps > 0.0:
        loud(
            f"adam_eps={tcfg.adam_eps}: AdamW's update divides by "
            f"`sqrt(v) + eps`, so eps must be a positive finite floor — 0 divides by "
            f"zero on a coordinate whose second moment is 0 and NaN is not a floor. "
            f"Escape: adam_eps > 0 (torch's 1e-8 is the default)."
        )
    if not 0.0 <= tcfg.decay_frac <= 1.0:
        loud(
            f"decay_frac={tcfg.decay_frac}: it is a FRACTION of the whole run "
            f"(`decay_steps = round(decay_frac * steps)`), so [0, 1] is the only "
            f"meaningful range — 0 is a constant LR after warmup and 1 is a decay "
            f"from step 0. Escape: 0 <= decay_frac <= 1 (0.1 is the default)."
        )
    if not 0.0 < tcfg.min_lr_frac <= 1.0:
        loud(
            f"min_lr_frac={tcfg.min_lr_frac}: `wsd_lr`'s `final_frac` is the LR "
            f"the decay LANDS on, so it must be in (0, 1] — 0 asks for a division "
            f"by a zero LR basin and >1 would make the 'decay' an increase. "
            f"Escape: 0 < min_lr_frac <= 1 (0.1 is the default, 0.01 is a full "
            f"anneal to near zero)."
        )
    if tcfg.grad_accum > 1 and tcfg.opt_graph:
        loud(
            f"grad_accum={tcfg.grad_accum} with opt_graph=true: the CUDA graph "
            f"captures ONE optimizer stage, and this arm is k EAGER backward "
            f"passes standing around that stage — a stage whose gradient buffers "
            f"were filled by a variable number of eager passes is not the stage "
            f"the graph was verified on, and a capture-time frozen participating "
            f"set is the `dion/cuda_graph.py:231-255` defect with a second way to "
            f"starve it. Escape: opt_graph=false (the eager optimizer), or "
            f"grad_accum=1."
        )
    if tcfg.full_graph and tcfg.opt_graph:
        loud(
            "full_graph=true with opt_graph=true names TWO captures of one region. "
            "Escape: full_graph=true alone (it implies the optimizer capture), or "
            "opt_graph=true alone."
        )
    if tcfg.iter_checkpoint and tcfg.full_graph:
        loud(
            "iter_checkpoint=true with full_graph=true: a CUDA graph capture records "
            "ONE launch sequence and replays it, while per-iteration checkpointing is "
            "a RECOMPUTE the backward performs on the eager graph — the capture "
            "cannot see it, and replaying a graph whose saved set was built without "
            "the recompute would hand the backward tensors the capture never kept. "
            "Escape: full_graph=false, or iter_checkpoint=false."
        )
    if tcfg.iter_checkpoint and tcfg.opt_graph:
        loud(
            "iter_checkpoint=true with opt_graph=true: the optimizer stage is "
            "captured and replayed, but checkpointing changes the graph the backward "
            "runs BEFORE that stage — a recompute a capture cannot see. A replayed "
            "stage whose gradients came from a different saved set is not the stage "
            "the capture was verified on. Escape: opt_graph=false, or "
            "iter_checkpoint=false."
        )
    if tcfg.activation_recompute not in ("none", "full", "selective"):
        loud(
            f"activation_recompute={tcfg.activation_recompute!r} is not a recompute "
            f"mode. Known: 'none', 'full', 'selective'. Escape: name one of those."
        )
    if tcfg.iter_checkpoint and tcfg.activation_recompute not in ("none", "full"):
        loud(
            f"iter_checkpoint=true with activation_recompute="
            f"{tcfg.activation_recompute!r}: two names for one recompute arm, and "
            f"they do not name the same one. Escape: iter_checkpoint=false and the "
            f"enum, or activation_recompute='full'."
        )
    for capture, why in (
        ("full_graph", "a capture cannot see a recompute the backward performs"),
        ("opt_graph", "the captured stage's gradients would come from a different graph"),
    ):
        if tcfg.activation_recompute != "none" and getattr(tcfg, capture):
            loud(
                f"activation_recompute={tcfg.activation_recompute!r} with {capture}=true: "
                f"{why}. Escape: {capture}=false, or activation_recompute='none'."
            )
    if tcfg.activation_offload != 0.0:
        if not math.isfinite(tcfg.activation_offload) or tcfg.activation_offload < 0.0:
            loud(
                f"activation_offload={tcfg.activation_offload}: the budget is a "
                f"non-negative GiB count moved to host per forward; 0.0 is OFF, and a "
                f"negative or non-finite value would install hooks with no useful "
                f"bound. Escape: activation_offload >= 0 (e.g. 4.0), or 0.0."
            )
        if tcfg.full_graph:
            loud(
                f"activation_offload={tcfg.activation_offload} with full_graph=true: "
                f"a capture cannot see a Python `saved_tensors_hooks` pair or its "
                f"H2D/D2H copies, so a replayed step would read whatever the capture "
                f"happened to leave on the device instead of the parked tensor. "
                f"Escape: full_graph=false, or activation_offload=0.0."
            )
        if tcfg.opt_graph:
            loud(
                f"activation_offload={tcfg.activation_offload} with opt_graph=true: "
                f"the optimizer stage is captured, but the backward that fills its "
                f"gradient buffers runs eager with host copies a capture cannot see — "
                f"a replayed stage would be fed gradients from a different graph than "
                f"the one it was verified on. Escape: opt_graph=false, or "
                f"activation_offload=0.0."
            )
    if tcfg.full_graph and tcfg.grad_accum > 1:
        loud(
            f"full_graph=true with grad_accum={tcfg.grad_accum}: the accumulation loop "
            f"is PYTHON (k forwards, a per-micro firewall AND, one group gate), so a "
            f"capture would record ONE micro-batch and replay it as the group. "
            f"Escape: grad_accum=1, or full_graph=false."
        )
    if tcfg.full_graph and tcfg.torch_compile:
        loud(
            "full_graph=true with torch_compile=true: a recompile attempted while a "
            "CUDA graph capture is active aborts the capture, and `torch.compile` is "
            "neither needed nor wanted under a graph. Escape: torch_compile=false, or "
            "full_graph=false."
        )
    if not 0.0 <= tcfg.ema_decay < 1.0:
        loud(
            f"ema_decay={tcfg.ema_decay}: the EMA is "
            f"`shadow <- d·shadow + (1-d)·param`, so the legal band is [0, 1). 0.0 "
            f"is OFF; 1.0 freezes the shadow at its initial weights, which is not "
            f"an average and would make every eval read the init. Escape: "
            f"0.0 <= ema_decay < 1.0."
        )
    if tcfg.ema_decay > 0.0 and tcfg.muon_sf:
        loud(
            f"ema_decay={tcfg.ema_decay} with muon_sf=true: during a training step "
            f"the parameters hold Y = (1-b1)·Z + b1·X, while the iterate this run "
            f"evaluates and checkpoints is X (materialised by `sf('eval')`). An EMA "
            f"updated after `opt.step()` would average Y, a different variable, and "
            f"the checkpoint would pair an X-EMA with an optimizer built for the "
            f"raw params. Materialising X every step for the shadow is the named "
            f"follow-up. Escape: muon_sf=false, or ema_decay=0.0."
        )


def validate_model_train(model: ModelConfig, tcfg: TrainConfig) -> None:
    """Cross-half refusals: a MODEL arm that a TRAIN capture cannot carry.

    ONE definition with two callers (`load_run` and `train/loop.py::train`), the
    same shape as `validate_train` — a hand-built run that skips `load_run`
    must hear the same refusal. LayerRoPE's per-depth gain is a device-side
    gather + rotation (no host read). `full_graph` is allowed because the
    captured loop's depth indices are fixed (the arms that would move them are
    refused by `train.graph.full_graph_refusal`); `opt_graph` is still refused,
    and the reason names the arm.
    """
    if tcfg.activation_offload > 0.0 and model.use_byteflow:
        loud(
            f"activation_offload={tcfg.activation_offload} with use_byteflow=true: "
            f"the chunked head builds a second patch-latent forward whose save set is "
            f"not measured here, so a global saved-tensor hook would park a graph "
            f"nobody has measured (and the chunked head's own re-forward conflicts "
            f"with a second hook pair). Escape: use_byteflow=false, or "
            f"activation_offload=0.0."
        )
    if tcfg.activation_offload > 0.0 and not model.bf16:
        loud(
            f"activation_offload={tcfg.activation_offload} with bf16=false: the arm "
            f"parks fp32 activations, i.e. twice the pinned bytes and twice the PCIe "
            f"traffic of the measured bf16 arm; the budget would price a different "
            f"tensor set. Escape: bf16=true, or activation_offload=0.0."
        )
    if tcfg.activation_recompute != "none" and model.use_byteflow:
        loud(
            f"activation_recompute={tcfg.activation_recompute!r} with use_byteflow=true: "
            f"both byteflow shapes leave this model's byte-level loop (Option A builds "
            f"none, Option B runs it over K patch latents), so there is no iteration "
            f"to wrap and the arm would be inert. Escape: use_byteflow=false, or "
            f"activation_recompute='none'."
        )
    if not model.layerrope:
        return
    # `full_graph` is ALLOWED. The per-depth gain is selected by a Python depth
    # index, but every index the captured loop reaches is a pure function of the
    # FIXED structure (`prelude_layers`, the iteration number, `max_iter`,
    # `coda_layers`); the two arms that would make it move — `rand_depth` and
    # `segment_len` — are refused by `train.graph.full_graph_refusal`, and the
    # capture records one `select` per fixed iteration. The m400b recipe
    # (paper LayerRoPE, max_iter=3) is the gate:
    # `tests/test_full_graph_m400b.py` asserts its capture counter. `opt_graph`
    # stays refused: its contract over the rotated gains has not been measured.
    if tcfg.opt_graph:
        loud(
            "layerrope=true with opt_graph=true: the optimizer stage is captured "
            "with the participating parameter set frozen, and LayerRoPE's gain table "
            "is indexed by depth at forward time — the capture contract has not been "
            "verified for the rotated gains. Escape: opt_graph=false, or "
            "layerrope=false."
        )


def load_run(
    preset: str = "small",
    *,
    train: TrainConfig | None = None,
    train_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    """Resolve a run: preset (name or path) + both config halves + loud overrides.

    Model overrides get the same refusals the preset file itself gets
    (`validate` for a self-inconsistent set, and the unknown-key check the
    preset layer runs), because an override that bypasses them is the
    silent-collapse shape the preset layer exists to prevent.
    """
    model = load_preset(preset)
    tcfg = train if train is not None else TrainConfig()
    if train is None:
        # The measured preset recipe (module comment above): a preset NAME (or
        # a path whose stem is one) gets its TRAIN defaults here, before the
        # explicit overrides, so `--set` remains the last word. A caller that
        # passes a `TrainConfig` owns every value and is not touched.
        preset_keys = PRESET_TRAIN_DEFAULTS.get(Path(preset).stem, {})
        if preset_keys:
            tcfg = dataclasses.replace(tcfg, **preset_keys)
    if train_overrides:
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        unknown = [k for k in train_overrides if k not in known]
        if unknown:
            loud(
                f"unknown train key(s) {sorted(unknown)}. Escape: remove the "
                f"key, or add the field to TrainConfig and wire it."
            )
        tcfg = dataclasses.replace(tcfg, **train_overrides)
    if tcfg.opt not in ("mix", "adamw"):
        loud(f"opt={tcfg.opt!r}: known optimizers are 'mix' (default) and 'adamw'")
    if tcfg.mm_dtype not in MM_DTYPES:
        loud(
            f"mm_dtype={tcfg.mm_dtype!r} is not a compute mode. Known: "
            f"{list(MM_DTYPES)} — `bf16` casts the fp32 masters to the compute "
            f"dtype and runs the bf16 tensor-core GEMM (the default and every "
            f"number on record), `fp8` runs the 2-D products through "
            f"`torch._scaled_mm` (e4m3 × e4m3 → bf16 out, per-tensor amax scales "
            f"off the bf16 operand; the spike measured 1.6-1.8× on 2-D GEMMs, and "
            f"the lane is FORWARD-ONLY on torch 2.13 — `_scaled_mm` has no "
            f"backward, so a training forward falls back to bf16 COUNTED). "
            f"Escape: name one of those, or land the mode in `tsct.MM_DTYPES` and "
            f"its gate BEFORE the config can lie about it."
        )
    if tcfg.max_epochs < 0:
        loud(
            f"max_epochs={tcfg.max_epochs} is negative: it is the repetition cap "
            f"in `train/mixture.py` (0 = off). Escape: a non-negative number, or "
            f"leave the default 4.0 (the measured MoE onset)."
        )
    if tcfg.mix_plan and not Path(tcfg.mix_plan).is_file():
        loud(
            f"mix_plan={tcfg.mix_plan!r} is not a file. Escape: a path to an "
            f"existing plan TOML (the shape is documented in `train/mixture.py`), "
            f"or leave it empty to read the single `data` file."
        )
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
    if isinstance(tcfg.ns_steps, bool) or not isinstance(tcfg.ns_steps, int) or tcfg.ns_steps < 1:
        loud(
            f"ns_steps={tcfg.ns_steps!r}: the Newton-Schulz iteration count is a "
            f"whole number >= 1 (`5` is Muon's default, `3` is the measured-parity "
            f"arm). A float or a bool would silently run a different chain. "
            f"Escape: an integer, e.g. --set ns_steps=3."
        )
    if (
        isinstance(tcfg.muon_chunk_elems, bool)
        or not isinstance(tcfg.muon_chunk_elems, int)
        or tcfg.muon_chunk_elems < 1
    ):
        loud(
            f"muon_chunk_elems={tcfg.muon_chunk_elems!r}: the lockstep cap is a "
            f"positive element count (the schema default is 1<<25; m400/medium "
            f"use 1<<26). Escape: an integer >= 1."
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
    if tcfg.muon_sf and tcfg.muon_sf_wd <= 0:
        loud(
            f"muon_sf_wd={tcfg.muon_sf_wd}: the schedule-free arm needs weight decay "
            f"at the FAST iterate Z, and a non-positive value leaves the theorem's "
            f"precondition unmet — arXiv:2605.23061v1 Lemma 3.1 is what lifts the "
            f"bounded-diameter assumption of the SF-Spectral bound "
            f"(`docs/research/2026-10-16-sf-muon.md:77-81`, S2). The decay is "
            f"`Z <- (1 - lr*wd)*Z - eta_hat*P_hat` (Alg 1 line 17), so it is NOT the "
            f"AdamW group's `wd`. Escape: muon_sf_wd>0 (the arm's 0.01), or "
            f"muon_sf=false."
        )
    if tcfg.muon_sf_retract:
        loud(
            "muon_sf_retract=true: the per-step TSCT polar retraction cannot run "
            "under the schedule-free average. The retraction is a closest-point "
            "projection onto the (NON-CONVEX) Stiefel manifold, while the "
            "SF-Spectral proof's projection/Lyapunov step "
            "(arXiv:2605.23061v1 §3.1, Thm 2.1) needs a convex set, and "
            "X=(1-c)U1+c*U2 of retracted factors has column norms <= 1 with "
            "equality only when the two agree — so the averaged iterate is not on "
            "the manifold the factor contract assumes "
            "(`docs/research/2026-10-16-sf-muon.md:82-93`, S3). This port's loop "
            "has no `retract_tsct` call site at all, so with muon_sf=false the flag "
            "would be inert too. Escape: muon_sf_retract=false (the default, and "
            "the legal arm)."
        )
    validate_train(tcfg)
    if model_overrides:
        import dataclasses as _dc

        model = _dc.replace(model, **model_overrides)
        validate(model)
    # The MODEL arm × TRAIN capture refusals (`full_graph`, `opt_graph`), after
    # the overrides so an override cannot bypass them.
    validate_model_train(model, tcfg)
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
    _validate_doc(model, tcfg)
    _validate_jepa(model, tcfg)
    # The RL lane's keys. The advantage arm is a literal list (checked here, the
    # way `opt` and `mm_dtype` are); the reward and penalty knobs are checked by
    # BUILDING the `RLConfig` they describe, whose `__post_init__` is LOUD — one
    # construction, so its rules cannot drift into a second copy here.
    if tcfg.rl_adv not in RL_ADVANTAGES:
        loud(
            f"rl_adv={tcfg.rl_adv!r} is not an arm. Known: {list(RL_ADVANTAGES)} — "
            f"`crr` is the fork-replay advantage (`post/train_steps.crr_advantage`, "
            f"read off each group's Rollout) and `group` is GRPO's "
            f"group-normalised reward. Escape: name one of those, or land the arm "
            f"in `post/rl.rl_advantages` and its gate BEFORE the config can lie "
            f"about it."
        )
    if tcfg.rl_tasks < 1 or tcfg.rl_group < 1 or tcfg.rl_budget < 1:
        loud(
            f"rl_tasks={tcfg.rl_tasks} rl_group={tcfg.rl_group} "
            f"rl_budget={tcfg.rl_budget}: a step needs at least one task to draw, "
            f"one rollout in the group and one byte of budget. Escape: positive "
            f"counts (the defaults are 128 / 4 / 256)."
        )
    if tcfg.rl_holdout < 0 or tcfg.rl_eval_every < 0 or tcfg.rl_eval_samples < 1:
        loud(
            f"rl_holdout={tcfg.rl_holdout} rl_eval_every={tcfg.rl_eval_every} "
            f"rl_eval_samples={tcfg.rl_eval_samples}: the holdout is >= 0, the eval "
            f"cadence is >= 0 (0 = before/after only), and pass@budget needs at "
            f"least one sampled attempt. Escape: the defaults are 32 / 50 / 2."
        )
    if not 0.0 <= tcfg.rl_band_lo <= tcfg.rl_band_hi <= 1.0:
        loud(
            f"rl_band=[{tcfg.rl_band_lo}, {tcfg.rl_band_hi}]: a pass rate lies in "
            f"[0, 1] and lo <= hi. Escape: the default [0.2, 0.8]."
        )
    if tcfg.rl_band_attempts < 1:
        loud(
            f"rl_band_attempts={tcfg.rl_band_attempts}: a step draws at least one "
            f"candidate. Escape: 1 (no retry) or more (the default is 4)."
        )
    rl_config(tcfg)
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
