# Upstream issue, drafted and ready to file

**Target:** `tracel-ai/cubek` (the matmul crate's home) — file it with a token
that has `repo` scope; the MCP token in this setup is read-only, so it could
not be created from here (2026-10-09).

**Title:** Autotuner picks a numerically different matmul candidate per
process; a training run corrupts itself because of it

---

## Summary

The autotuner selects a **numerically different matmul candidate per
process** for the same shape, and a training run silently corrupts itself
because of it. In our stack (burn 0.22.0 + cubecl 0.11.0, RTX 5060 Ti,
sm_120) `fp32 batch=10 seq=512` training of a 9.2M-param model destroys its
own run in roughly 2 runs out of 3: the loss goes `5.57 → 1e8..1e16 → NaN`
at step 1 and never recovers. We bisected the weights, the kernels' math,
races and memory errors out; the only per-process variable left is which
candidate won the benchmark.

## Evidence

With a decisions sink on (`cubecl.toml`):

```toml
[autotune.decisions]
file = "/tmp/decisions.jsonl"
```

six runs of one program (same binary, same flags, same seed, same data) each
make 125 tuning decisions. **50 of the 125 keys get a different winning
candidate between two runs.** Two keys flip together and separate our
outcomes perfectly — 4 exploding runs vs 2 clean runs:

| key | shape | 4 exploding runs | 2 clean runs |
|---|---|---|---|
| `31944fb2` | m=512 n=1024 k=128, both operands contiguous | `matmul_simple_cyclic_mma_multi_rows` (idx 15) ×4 | `matmul_ordered_double_mma_partition_k2_row_count8_rows_per_plane2` (idx 19) ×2 |
| `c5c11115` | m=64 n=2048 k=64, rhs transposed | `matmul_ordered_double_mma_partition_k2_row_count8_rows_per_plane2` (idx 19) ×4 | `matmul_simple_cyclic_mma_multi_rows` (idx 15), `matmul_double_cyclic_mma` (idx 21) |

The two winners on `31944fb2` medians are 23.5 µs and 24.3 µs — **1.6 % apart**,
so the choice is a coin flip on benchmark noise between two kernels that
disagree numerically. Candidates carry a stable `idx` while the `results`
array arrives in completion order, so the decision is stable inside one
process and different across them.

## What we excluded, with the measurement

- **Not the weights / not a write.** At learning rate 0 (a provably no-op
  update) every parameter of the model is *bit-identical* between step 0 and
  step 1 (a per-parameter diff prints `0.000e0` for all 54 tensors) while the
  step-1 forward reads `ce = 1.76e9`, and the held-out eval on those same
  weights reads `NaN`.
- **Not the chunked-WY math.** Our production-shape gate compares the batched
  arm against the sequential loop arm at B=10 H=12 T=512 k=v=64 chunk 16:
  forward/state drift 3.3e-4, all seven parameter gradients ≤ 3.3e-4.
- **Not a race, not an out-of-bounds access.** `compute-sanitizer --tool
  racecheck` on the module forward: 0 hazards. `--tool memcheck`: 0 errors.
- **Not in-process nondeterminism.** Two forwards of one graph agree bitwise,
  and a forward after a backward equals the forward before it.

## Minimal repro

One test, no model code (ours lives in
`vendor/dormouse-fused/burn-fused/tests/matmul_uninit_probe.rs`):

```rust
#[test]
fn checksum_per_shape() {
    let d = Device::cuda(0);
    d.seed(42); // same inputs in every process
    for (m, k, n) in [(64usize, 64, 64), (512, 128, 1024), (10, 512, 64)] {
        let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &d);
        let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 1.0), &d);
        let c = a.matmul(b);
        let v: Vec<f32> = c.into_data().try_to_vec().unwrap();
        println!(
            "matmul {m}x{k}x{n}: checksum {:016x}",
            v.iter().fold(0u64, |h, x| h.wrapping_mul(31).wrapping_add(x.to_bits() as u64))
        );
    }
}
```

Run it repeatedly with the decisions sink on and correlate the printed
checksum with the winner the sink names for that shape: a different checksum
for the same inputs is a candidate computing a different product, and our
training runs show it. (Seed the device — an unseeded `Tensor::random` makes
the inputs differ per process, which is a trap worth knowing about.)

## Two related findings in the same area

1. **The persistent autotune cache never materialises.** `target/environment`
   is never created (three runs from one cwd, 125 fresh decisions each, zero
   files), although the persistence code is linked into the binary (2 692
   symbols match `persistence`/`Store`). That cache is the only mechanism
   that would pin a choice across processes, which is why this is per-process
   rather than once-per-machine.
2. **A narrow-output matmul reads never-written memory, harmlessly.** Under
   `--tool initcheck`, a matmul whose output width is 2 reads never-written
   global memory (8-byte reads in
   `matmul_entry_lhs_f32_lhs_size_4_rhs_f32_rhs_size_2`); at output width 64
   there are zero such reads. The value does not reach the result — the same
   product computed with the rhs materialised contiguous differs by `0.000e0`
   at out_features 2/3/4 — so it is masked, but it is still a read of
   unwritten memory.

## What we would find most useful

Whether a candidate's selection can be pinned per shape (a stable,
machine-readable answer to "which kernel for this key"), or whether the
winner should be required to be reproduced before it is trusted — anything
that makes the choice survive the benchmark noise. Our current workaround is
none: we treat burn 0.22.0 as not fit for training until this is answered.

Environment: burn 0.22.0, cubecl 0.11.0, cubek-matmul 0.3.0 (all stock, no
patches), RTX 5060 Ti (sm_120), CUDA 12.5, driver 615.71.09, Linux.
