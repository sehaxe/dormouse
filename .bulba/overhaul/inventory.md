# P0 inventory — codebase reduction / overhaul

Date: 2026-11-08. Tree: `dedde02` (plus uncommitted work from other lanes — see
"Working tree" below). Read-only: no code changed to produce this file.
Counts are `wc -l` over tracked sources; `port/.venv` excluded.

## 1. Where the lines are

| area | files | lines |
|---|---:|---:|
| `port/src/dormouse` | 93 | 51,604 |
| `port/tests` | 185 | 61,971 |
| `port/scripts` | ~125 | 49,744 |
| **Python total** | | **163,319** |
| `crates/` (Rust, frozen) | | 34,039 |
| `vendor/dormouse-fused` (Rust, frozen) | | 114,772 |
| **Rust total** | | **148,811** |
| `docs/**/*.md` | 347 | — |

Python detail:
- `model/` 25,179 · `train/` 10,443 · `post/` 2,707 · `selfplay/` 1,420.
- Largest files: `config.py` 5,641 · `model/experts.py` 3,730 · `train/loop.py`
  2,470 · `train/optim.py` 2,431 · `cli.py` 1,926 · `model/moe_kernels.py` 1,802
  · `model/loopblock.py` 1,695 · `model/model.py` 1,531 · `train/sft.py` 1,445.
- Script harness: `ab_*.py` 37 files / **15,301** lines · `bench_*.py` 40 /
  **10,593** · `probe_*.py` 10 / **1,973** → **27,867 lines (17 % of Python)**.

## 2. What is NOT replaceable (already measured against the library)

These are recorded losses for the "just use the library" plan; re-proposing them
without a new measurement violates ADR-0002/0018.

| custom | library | result |
|---|---|---|
| MoE dispatch (`experts.py`, `moe_kernels.py`) | `torch._grouped_mm` | padded bmm **21.4 ms vs 44.2 ms** (`experts.py:75`) |
| fused pointwise/residual | `torch.compile` | no win; routing blocks fusion (`2026-11-08-compile-matrix.md`) |
| attention | SDPA backends | default is already flash; forcing EFFICIENT is **2.14× slower backward** |
| KDA | FLA 0.5.2 | already used (optional `gpu` extra); SDPA fallback is COUNTED |
| fp8 training | torch fp8 | needs M≈32k; 0.66–1.09× at M=4096 |
| optimizer state | stock bf16 | our bf16 state is **+18 %** step (232→197 ms) |
| Muon | stock `torch.optim.Muon` (exists in 2.13) | ours = factor-space + fused NS + bf16 masters; not a drop-in |

**Consequence:** replacing model math with libraries is the *lowest*-value lever
here. The high-value cuts are dead code, the duplicated script harness, and the
frozen Rust lane.

## 3. Reduction ledger (ranked by lines, then risk)

| # | slice | lines removed (est.) | risk | gate |
|---|---|---:|---|---|
| S1 | Frozen Rust lane: `crates/` + `vendor/` from `main`; keep an archive-branch pointer | **~148,800** | high (doc citations, provenance) | Python suite unchanged; `check_doc_refs` green after ref updates; Rust CI jobs removed |
| S2 | Script harness: one shared A/B+bench driver, declarative experiment specs; keep every documented result reproducible | **~15,000–18,000** | medium | re-run 2 known A/B tables (`ab_rank`, `ab_sigreg`) and match `docs/reviews` |
| S3 | Unwired Python: `selfplay/` 1,420 + test 706; `model/act_ckpt.py` 418 + test 455 + bench 206; 14 superseded benches 2,726; two dead config fields 45; unwired vendor crates 726 | **~6,700** | low–medium | grep no production importer; `dead_knob_audit` rc 0; full CPU suite |
| S4 | QAT export (`model/export.py` 542): **keep**, it is a first-class product line (owner 2026-10-18) but has no CLI command — wire `dormouse export` instead of deleting | 0 (adds a command) | low | `pytest tests/test_export.py` + CLI round-trip |
| S5 | Library spikes (only land on a measured win): Liger mHC/RMSNorm/SwiGLU/CE; stock Muon; torchao dense QAT | 1,000–3,000 if they win | medium | A/B vs the current arm at same config; counters prove the arm ran |
| S6 | Config (`config.py` 5.6k) schema-driven rewrite | ~2,000 | high | `test_config`, resume/refusal goldens |
| S7 | Tests (62k): table-drive repeated goldens | 5,000–10,000 | medium | coverage per subsystem must not fall; no test deleted to pass |

S1+S2+S3 = **~170k lines** (~54 % of the whole repo including Rust), with the
Python-only cut at **~23k (~14 %)**. S2 is the largest Python-only cut and the
only one that also reduces maintenance of the evidence path.

## 4. Library survey (what the ecosystem offers, and the verdict here)

| library | what it covers | verdict |
|---|---|---|
| **Liger Kernel** | RMSNorm, RoPE, SwiGLU, CE, fused linear CE, **mHC**, MoE kernels; Triton only | **spike**: mHC + fused CE + RMSNorm are the only possible deletions; must beat current kernels at b8×s1024 |
| **torchao** | QAT for dense `nn.Linear` (int4/int8/fp8/MX/NVFP4); MoE QAT is an open RFC, not shipped | **limited**: can cover dense QAT only; our experts are 3D params routed through custom ops → no MoE replacement |
| **torch.optim.Muon** | stock Muon since 2.9, NS coefficients | **spike**: parity/LR semantics differ; replacement may cost speed; test before deleting ours |
| **TRL / verl / OpenRLHF** | RL/SFT trainers, vLLM rollouts | **no**: HF-Transformer-centric, multi-GPU/Ray; our byte-level custom model + 1 GPU makes integration larger than the 2.7k lines it could replace |
| **vLLM / SGLang** | serving | **no**: custom arch (TSCT/Engram/byte-level) is not supported; `serve.py` is 655 lines and works |
| **Grain / WebDataset / Streaming** | data loading | **no**: our mmap path is already 0 ms on the wall clock (CPU audit) and carries resume/overlap gates |
| **Pydantic** | config schema | **later**: would replace boilerplate, not the bespoke refusals; adds a dependency to a 5.6k-line validator |
| **safetensors / DCP** | checkpoint format | **optional**: custom `.pt` + config snapshot works; no LOC win |

## 5. Working tree / process

- `git status`: uncommitted edits by another lane in `port/src/dormouse/config.py`,
  `port/tests/test_lr_tail.py`, `port/tests/test_train.py`, `site/media/*`; untracked
  reviews. The active training run holds the GPU (7.3 GiB, 99 %).
- Do the overhaul in a dedicated worktree (`tools/wt.sh`) per AGENTS §1.6; no
  full-suite baseline until the tree is clean and the card/CPU are quiet.
- No push, no destructive git without the owner.

## 6. Open owner decisions

1. **Rust:** delete `crates/`+`vendor/` from `main` (archive branch is the source
   of truth) or keep `dormouse-core` as the in-repo formula source?
2. **Unwired Python:** delete `selfplay/`, `act_ckpt.py`, superseded benches; keep
   and wire QAT `export.py`?
3. **New deps:** approve Liger + torchao in a scratch venv for measurement only
   (no lock change unless an arm wins)?
