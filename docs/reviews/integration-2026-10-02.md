# integration — fleet merges to main, 2026-10-02

Owner: integration lane ("исправь всё, добиваем оборванное"). Scope: inventory and
merge of the branch wave that grouped on origin/main after the docs-heavy
over-commit. Everything below is measured today on this box unless named
otherwise. GPU discipline (§1.5) held: CUDA gates ran in one slot, in sequence.

## Inventory, verdict, action

| branch | base | ahead | verdict | action |
|---|---|---|---|---|
| `wt/graphfix` | 2ccff38 | 2 commits | ГОТОВО | **merged** `f2e9ab3`; follow-up fix `054337d` (see below) |
| `wt/fidelity-fix` | 2db2aa6 | 2 commits | ГОТОВО | **merged** `e790242` |
| `wt/kdafix2` | f4f7a0b | 2 commits | ГОТОВО (недописанный док) | **merged** `d65ed16`; its own findings doc was left uncommitted in the branch worktree — taken to main at `9a46ac2` |
| `wt/byteflow2` | 2ccff38 | 3 commits | ГОТОВО | **merged** (burn-byteflow crate + paper + findings); A/B byte-vs-byteflow is a queue item, not a merge gate |
| `wt/fixverify` | — | 0 ahead | already in main | none |
| `wt/carryfix` | — | 0 ahead (ancestor of main) | already in main | none |
| `wt/cuda-gate` | — | 0 ahead (ancestor of main) | already in main | none |

Related: `wt/kdaci`'s CI-proof doc (`kda-oracle-ci-2026-10-02.md`) is cited by the
kdafix findings doc; taken to main at `07ceef0`. The kdaci `#[ignore]` workaround
itself stays superseded by the kdafix fix, per its own findings doc.

## Gates

Run on the MERGED tree, not the branches (the branch-side claims were verified
by re-running what is cheap, and the CUDA side on the live card):

- graphfix: `dormouse-train` lib **60/60** (the two new CPU half-tests fail the
  merge as-committed — see the follow-up fix below); CUDA target
  `graph_seam_cuda` with `--test-threads=1`: **6/7 green**, all five
  differential/mechanism gates included `the_noengram_key_contract_captures_
  and_replays`. `the_stale_pointer_trap_is_reproduced_without_the_pin` fails
  identically on pre-merge main (proved by running it at `08cd648`) — pre-
  existing red, the graph-trainer lane's subject, NOT introduced by this wave.
  The default parallel test run poisons one CUDA context and cascades — run
  this target single-threaded.
- fidelity-fix: `dormouse-core` lib **89/89**; `burn-mhc` lib **9/9** (the new
  `b_res_starts_at_zero_not_the_10i_stiffness` pin green); `burn-attnres` lib
  **11/11** (the gain fixture green); inverted init gates (the two
  `> 0.05` / `> 0.3` asserts) measure the FIXED init, so a drift back onto the
  10I stiffness re-reddens them.
- kdafix2: `burn-kda` lib **12/12**; `--test kda_oracle` **9/9** (the designed
  reds are green: `kimi_linear_softplus_decay_matches_fla_reference`,
  `read_scale_matches_fla_reference`); `burn-gdn2`
  `ops_batched_autodiff` (autodiff feature) **1/1**; CUDA `--test fused_cuda`
  **4/4** on hardware — and the gate did not even compile off the shelf (see
  the fix commit). `--test bench_cuda` 1 ignored (by design).
- byteflow2: `burn-byteflow` crate tests **14/14** + facade test + oracle
  targets green (CPU).
- doc-gate (`tools/check_doc_refs.py`): was RED on 9 citations of the missing
  `docs/reviews/kdafix-2026-10-02.md` + its cited evidence doc; green after
  `9a46ac2` / `07ceef0` (0 missing paths).
- 1-step smoke on the release binary of the fully merged tree (real corpus,
  small/batch 8/s512): **green** — step 0 ce=5.571 bpb=8.038, aux=0.0289,
  retr_arm batched:0/factor:1, checkpoint saved, no NaN. (Step-0 numbers are
  autotune-cold, not measurements; §3.1.)
- `preset_exec` queued under the build lock at the end of the day's queue
  (agents `stage-gate-b`, `maxopt2` held it) — see the follow-up note.

## The follow-up fixes (each its own commit)

1. **`054337d` fix(graph): the keys gates precede the pin copies.** The
   branch's own doc comment promised the key-less quiet path does not touch
   the copy arms, but the gate sat AFTER `copy_into_int(x)`/`copy_into_int(y)`
   — on any non-CUDA build the copies refuse loudly and both new CPU
   half-tests failed on the pin refusal instead of testing the keys gate.
   Hoisted the consumption gate and the None-keys refusal above the copies;
   CUDA ordering for a real keyed feed unchanged.
2. **`fe69e5f` fix(kda): fused_cuda gate compiles after the read-scale merge.**
   The kdafix2 branch read `q.shape()` at the `chunk_wy_forward` call site,
   but main's state of the test had already moved `q` bare into the fused
   call — E0382 under `cuda,autodiff` features ONLY, invisible to every CPU
   gate. The head dim is captured before the move.
3. **`9a46ac2`/`07ceef0` docs(kda):** the kdafix findings doc and its cited
   kdaci evidence doc — both existed only in other lanes' worktrees uncommitted,
   while 9 committed files + the doc-gate pointed at them.

## Pre-existing reds found on the way (NOT this wave's defects, carried to their owners)

- `the_stale_pointer_trap_is_reproduced_without_the_pin` (graph_seam_cuda) —
  fails on pre-merge main identically; same crash class that killed the
  graph-trainer lane's `gbench2_graph` run ~18:48 today
  (`server.rs:144`, illegal address). Owner: graph-trainer / cuda-graph lane.
- `gated_delta_chunk_path_runs_at_the_production_shape`
  (`vendor/dormouse-fused`, burn-fused `gpu_production_shape`, cuda+autodiff,
  release) — matmul inner-dim mismatch `[10,12,16,64] × [10,12,512,64]`,
  fails identically with burn-kda/burn-gdn2 checked out at the kdafix2 base
  `f4f7a0b`. Pre-existing red → `tools/gpu-gate.sh` is red on main for reasons
  outside this wave. Owner: burn-gdn2 / chunk-path lane.
- `tools/lib_gate.sh:95` — the rc-overwriting red-mask (`rc=$rc2`) reported by
  the wgpu and dispatch-guard lanes: still unfixed, still a shared tool.
- fuse-library CI on main: the burn-kda designed-reds are now GREEN on the
  merged tree (this wave), burn-spectral's known-red ×3 remains (its own lane).

## Notes on the merged content itself

- burn-mhc `block.rs:27` carries `pub(crate) const B_RES_INIT: f32 = 10.0; //
  FALSIFY` — but it multiplies `Tensor::<2>::zeros`, so the init is 0 at ANY
  value of the constant: the multiplier is dead arithmetic wearing a mutation
  marker (a falsify mutant reverting it to 10 cannot flip the init, and no
  gate can see that mutant). Functional fix is correct (b_res == 0 is pinned
  by a burn-mhc test); the dead multiplier should be dropped or actually
  wired the next time someone touches the init — flagged, not changed, it is
  the mhc lane's marker.
- The dirty files found in the main checkout (fusion-flip WIP: `Cargo.lock`,
  `core/Cargo.toml` burn-situ cuda feature, the
  `research/reviews/graph-trainer-2026-10-01.md` continuation, and the
  burn-fused facade `wgpu` feature) were preserved
  exactly (backup in `/tmp/opencode/dirty-backup/`, stash bookkeeping fully
  unwound). The facade `wgpu` feature travelled through the byteflow2 merge
  already (both entries present, builds clean).
- No push: 17 commits ahead of `origin/main` locally, awaiting the owner's OK
  (§1.6).

## Queued next (per the lane's order)

- (b) byte-vs-byteflow A/B — 3 seeds, 2k steps, pure CE, ONE batch size
  (§1.2/§2.6); costs are the A/B-protocol's unknown-per-arm problem, unchanged.
- (c) next wave of idle branches (the inventory shows ~40 more `wt/*`; the
  digest's priority order governs).
- CUDA queue: burn-attnres gain fixtures on CUDA are covered only through the
  smoke today; a targeted `burn-attnres` cuda gate rides with the attnres
  A/B wave.
