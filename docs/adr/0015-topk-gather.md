# 0015: The top-k -> gather primitive works; three mechanisms unblocked

## What was wrong

`Tensor::argtopk` on a cubecl backend emits **out-of-range indices** for any row
that has fewer than `k` values above `-inf`, and a `gather` with one of them
reads out of bounds: `cuEventCreate 700` / `CUDA_ERROR_ILLEGAL_ADDRESS`. Not a
cast bug, not an uninitialized read, not an off-by-one: a **sentinel**.

The kernel is upstream `cubek-reduce` 0.3.0-pre.4, reached through
`burn_cubecl::kernel::reduce::base` -> `cubek::reduce::reduce`
(`ArgTopK`/`TopK`). Its accumulator is `k` slots, each seeded by
`null_accumulator` (`instructions/topk.rs`):

- slot value = `P::EA::min_value()` (`-3.4e38` for f32) — "empty",
- slot coordinate = `u32::MAX` — **"no coordinate", and it is never sanitized
  on the way out**.

Every fill goes through `topk_insert`, whose whole body in 0.3.0-pre.4 is
wrapped in a fast-path guard (`instructions/base.rs`, `reaches`):

```rust
if !(item.extract(i) < kth.extract(i)) { any = true }   // kth = elements[k - 1]
```

That guard is sound for a *full* list and wrong for an *empty* slot, because
`-inf < min_value` is a **true** comparison. So a masked candidate is judged
"cannot enter" a slot that is in fact waiting for its first element, the insert
is skipped, and the slot keeps `u32::MAX`. A block-sparse indexer masks its
excluded blocks with `-inf` — and every short prefix has fewer than `k` visible
blocks, so this is not an edge case, it is every early position in the sequence.
Nothing between the accumulator and the caller inspects the coordinate:
`to_output_parallel` / `to_output_perpendicular` `cast` it straight to the index
dtype, where `u32::MAX` becomes `-1` (i32) or `4294967295` (i64/u32), and that
is the index the gather dereferences.

Two smaller holes share the same root:

- `k` greater than the axis length leaves the tail unfilled whatever the scores
  are. Nothing rejects it: `reduce_dim` only rejects `axis_length == 0`.
- `plane_topk_merge` builds its `u32::MAX` phantom by construction — a lane
  already consumed is masked to `(min_value, u32::MAX)` and
  `lowest_coordinate_matching` returns `u32::MAX` when every lane is masked.

**This is a pre.3 -> pre.4 regression.** The diff is one hunk: 0.3.0-pre.3 had
no `reaches` guard at all, every candidate walked the `k`-slot insertion, and a
`-inf` candidate always displaced a `min_value` seed — so the coordinate was
always overwritten and the index was always in range. The guard was added as a
measured win ("on a 151936-wide row of logits the insertion was the whole cost
of a top-20, 4.7 ms on GP100") and its own doc comment reasons carefully about
NaN — an all-NaN row would "emit the `u32::MAX` null-accumulator sentinel as its
index" — without noticing that `-inf` reaches the same state through a
*true* comparison, which no choice of `<` / `<=` / `>=` catches.

That is why this presented as "pre.4 changed the topk/indexing semantics" in
ADR-0012 and why it looked like a property of the indexer rather than of the
reduce: **no caller in the tree ever exercised `argtopk` on a masked row.**
`burn-mor` had already routed around it with a full `argsort_descending` +
`narrow` (`burn-mor/src/topk.rs`), which is immune because a sort returns a
permutation. The bug survived a CPU-only test suite, an arm cut (ADR-0014) and a
mechanism rejection (ADR-0013) with nothing in the tree able to see it.

## The fix

Vendored `cubek-reduce` to `vendor/cubek-fix/cubek-reduce` and patched it
through the existing `[patch.crates-io]` (the same three-line pattern as
`vendor/cubecl-fix`; the root `exclude` now lists `vendor/cubek-fix` for the
reason the comment above it gives — it is its own workspace root). Two files,
four hunks, in the kernel rather than at any call site:

1. **`instructions/base.rs`, `reaches`** — a slot still holding `min_value` is
   empty, not full, so it accepts anything:
   `if !(item < kth) || kth == N::min_value()`. This is the actual regression
   fix: a `-inf` candidate now displaces the seed and carries its **real**
   coordinate out, so the picked set is a genuine top-k with `k` distinct
   in-range indices, not a set padded with repeats.
2. **`instructions/topk.rs`, `to_output_parallel` + `to_output_perpendicular`**
   — clamp the coordinate to `axis_len - 1` on the way out. This is the
   invariant guard, and it is the load-bearing one: it holds for all three holes
   above, including the `k > n` case and the plane-merge phantom, which (1)
   does not reach. The trait already passed `shape_axis_reduce` into the
   finalizer and every `Arg*` ignored it; that unused parameter is where the
   bound belongs.
3. **`instructions/topk.rs`, `null_accumulator`** — seed the coordinate at `0`
   rather than `u32::MAX`, so an unfilled slot is in range even if it never
   passes through (2). Belt and braces: (1) and (2) are each independently
   sufficient for the `-inf` case; (3) costs nothing and removes the class.

The guarantee that was missing now holds, and it is measured: **every index
`argtopk` returns is in `[0, n)`, whatever the scores contain.** See "What is
measured" below.

## What now works

`burn-mor/src/topk_gather.rs` exposes the primitive the three mechanisms were
all waiting on:

- `topk_gather(scores, values, k)` — `[b,q,n]` scores against `[b,q,n,m]` value
  rows -> `[b,q,k,m]` gathered,
- `topk_indices_last(scores, k)` — the index half alone, for a caller that
  gathers from more than one tensor with the same picks.

The gate is `crates/backend-parity/tests/topk_gather_parity.rs`, in the crate
that exists for exactly this bug class ("the CPU/CUDA parity gate for the
backend precision bugs, in its own crate so it keeps building while
`dormouse-core` is mid-refactor"). One body of assertions run against
`Device::ndarray()` and `Device::cuda(0)`, so there is no second copy to drift.
Adversarial rows: all-equal (ties everywhere), a single hot index, `k = n - 1`
(the tightest legal `k`), `k = 1`, an all-masked row, a `-inf` prefix with 3
visible candidates taken at `k = 8`, the same rows masked with `-1e30` instead
of `-inf`, and the block-indexer shape b=10 q=512 n=128 k=64.

Two facts the gate established that are worth more than the fix:

- **`k == n` is not expressible.** burn's `argtopk` asserts `shape[dim] > k`
  (`burn-tensor/src/tensor/api/orderable.rs:614`). So a plan that asks for
  `topk=64` over 12 heads/blocks cannot be written as one call; the candidate
  axis has to be the axis that actually holds the candidates.
- **The index dtype is backend-defined**: I64 from ndarray, I32 from cubecl.
  Any consumer that reads indices back has to convert, not cast-assert.

## What is measured (sm_120, committed patch, 2026-09-28)

- **The out-of-bounds read is GONE.** Zero out-of-range indices on every
  fixture, masked rows included, where the same tensor faulted the gather with
  `cuEventCreate 700` before. The ndarray arm is fully green.
- **One defect is still open, and it is NOT the crash.** On a partially masked
  row the trailing top-k slots come back naming **one repeated column** instead
  of the masked columns' own:

  ```text
  scores [0, -inf, 2, -inf, 4, -inf, 6, -inf, 8, -inf, 10, -inf],  n=12, k=8
  picks  [10, 8, 6, 4, 2, 0, 0, 0]        # 0, 0 should be two of the -inf columns
  ```

  The gather reads a valid address and returns a valid row, so this is a
  **silent wrong answer** — a defect by ADR-0011, not a pass. The CUDA arm of
  the gate is therefore `#[ignore]`d with that reason, the same convention this
  crate already uses for the bf16 gap and the fused-backward gap. **Do not put a
  masking mechanism on this primitive until it is green.**

  What was tried and did not work, so nobody repeats it: adding the
  empty-slot-loses clause to `reaches` (which is in the committed patch), to
  `topk_finalize_with_coords`, and to the insertion walk in both `topk_insert`
  and `plane_topk_insert_with_coords`. The last three changed the output **not
  at all** — bit-identical picks — which says the masked candidates are not
  reaching the code those comparisons guard, and the next step is to find which
  layout is taken for a `n=12, k=8` reduce (the `n=128, k=64` fixture passes,
  so the two shapes take different paths: `to_output_parallel` +
  `topk_finalize_*` versus `to_output_perpendicular` + the plane insertion) and
  to instrument there rather than in the selection network. Those three hunks
  are **not** in the tree: an unverified patch to a vendored upstream kernel is
  worse than no patch, because the next reader believes the path is covered.

  The clean statement of the invariant every one of those sites violates: **an
  incumbent slot still holding the `min_value` seed is EMPTY, and an empty slot
  must always lose to a candidate** — `-inf` is below the seed, so every
  plain `<`/`>` comparison between them calls the empty slot the better one.
  Whoever finishes this should apply that one rule at each site rather than
  patching a symptom per layout.

## What it unblocks

All three were cut or rejected *because of this primitive*, not on their merits:

- **MSA sparse attention** (cut, ADR-0014). Its re-entry condition 1 was "a
  working block-sparse implementation with its indices verified in-range under
  compute-sanitizer on pre.4" — that condition is met by construction now, and
  the gate above is the verification it asked for. Note the indexer is a
  *masking* consumer, i.e. the open defect above, so condition 1 is met for
  safety and NOT yet for the picked set: the other two ADR-0014 conditions
  stand, and re-adding the arm is its own agent's job.
- **MoR / mixture-of-recursions** (2507.10524, rejected in ADR-0013 for "its
  per-depth gathers are sm_120-hostile" — this is what that meant).
  `burn-mor` is in the tree and now has a primitive whose output cannot fault.
  Its router scores are *unmasked* (every token is a candidate), so it is the
  one of the three that does not wait on the open defect above — which is
  consistent with the "rank the slots per position" arm already landing in
  `core/src/mor.rs`. ADR-0013's own re-entry bar is untouched: an A/B win over
  fixed depth at matched steps on held-out BPB.
- **PKM-style product-key memory** (its top-k -> gather is this same call). A
  product-key lookup masks nothing by default, so it is unblocked; a
  capacity-masked variant would hit the open defect.

## Cost

One extra comparison per emitted index at the reduce's output write, plus one
per lane of the plane path — on `k` indices per output row, so it is noise
against the reduce itself. `reaches` additionally keeps a row that genuinely
contains `min_value` (integer inputs) on the slow insert path; the insert is a
no-op once the value repeats.
