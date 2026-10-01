# The fused RMSNorm CUDA kernel: it lowers now, and there is a second defect behind it

**Lane:** rmsnorm-kernel · **worktree:** `wt/rmsnorm-kernel` off `ff9d211`
· **date:** 2026-09-30 · **box:** RTX 5060 Ti, one GPU.

The question AGENTS.md §3.3 records was *"the fused RMSNorm kernel never
produced a number on ANY device in the life of this project"* and
*"`34c5631` fixed a real defect"*. The second half is **wrong**, and the first
half was right for a reason nobody had identified. Both are now settled by
measurement, and a **second, independent defect** was found on the way, which
is the more important of the two.

Everything below was measured on this box, on the GPU, on 2026-09-30. No number
in this file is transcribed from a report, including the reports written earlier
tonight: the diagnosis in the previous lane's
`docs/archive/rmsnorm-kernel-2026-09-30.md` §2 is **retracted by the bisect in §2
below**, and its probe had never compiled (7 errors), which is why its §3 was
empty.

---

## 1. The headline numbers

| what | before | after |
|---|---|---|
| `fused_kernel_gate` | **RED** — `the lowered module does not verify: Expected operand type llvm.ptr, but found builtin.integer` | **GREEN**, 3/3 |
| the fused arm's accuracy | never measured, on any device | **worst 1.168e-7** relative over 84 outputs, asked 6 / skipped 0, against a scalar **f64** definition, at mean-squares spanning 1e-4..1e2 (tolerance 1e-5) |
| `d = 2` (fixture case `d2`) | — | **5.245e-1** relative, a SILENT wrong answer → now **refused** |
| the same input on the tensor path | — | 7.29e-8 |

The 1.168e-7 is ~86x inside the gate's 1e-5. The f32 sum bound over these rows
is ~5e-7, so that is what a correct f32 implementation should read.

---

## 2. Why it did not lower: the halving tree loop, and nothing else

`34c5631` added `#[comptime]` to the launch signature and left
`let threads = 256usize;` in the body. `9ac0377` did it properly —
`#[comptime] threads`, one `pub const THREADS` for both `CubeDim` and
`Shared::new_slice` — and **reproduced the identical error**. So the shared-memory
width was never the cause, and the doc comment `fused.rs` carried after
`34c5631` is retracted.

`tests/lower_probe.rs` is the bisect, and it is a **permanent** test file rather
than a throwaway. Each rung is a separate `#[test]` and must be run in its own
process (`--exact <rung>`): a launch failure poisons the CUDA context.

| rung | what it adds | result |
|---|---|---|
| `p0_three_buffers` | 3 buffers, an index, a multiply | LOWERED, 0e0 |
| `p1_runtime_scalar` | a runtime `f32` through `F::cast_from` | LOWERED, 0e0 |
| `p2_shared_write_sync_read` | `Shared::new_slice` at a `#[comptime]` size, one write, ONE barrier, one read | LOWERED, 0e0 |
| `p3_cond_write_then_bcast` | a conditional shared write, a SECOND barrier, a read by all threads | LOWERED |
| `p4_barrier_in_while` | **no shared at all**: a `sync_cube()` inside a runtime-bounded `while` | LOWERED, 0e0 |
| `p5_tree_reduction` | the real kernel's whole reduction, halving `while`, barrier in it | **FAILS** |
| `p6_serial_reduction` | the SAME kernel, reduction serialised by thread 0 | LOWERED, **1.72e-7** |
| `p7_full_kernel` | `src/fused.rs`'s kernel, verbatim | **FAILS** |
| `p9_tree_no_barrier_in_loop` | the halving `while` with **no barrier anywhere** | **FAILS** |
| `p10_comptime_for_loop_tree` | p5 with `for k in 0..log_threads` over a `#[comptime]` trip count | LOWERED, **1.72e-7** |
| `p8_the_crates_own_kernel_lowers` | the crate's own `RMSNorm::forward`, i.e. its own launch | **FAILS** |

### What each exclusion rules out

* **Not the shared-memory width, and not `Shared::new_slice`.** p2 does exactly
  that at a `#[comptime]` size and lowers.
* **Not "a barrier in a loop".** p4 has a `sync_cube()` in a runtime-bounded
  `while` and lowers. The source tree agrees: `burn-gdn2/src/kernel/chunk_cube.rs:243`,
  `burn-mhc/src/sinkhorn_cuda.rs:98` and `burn-spectral/src/moe_fused.rs:212` all
  put a barrier inside a runtime-bounded `while` and all run. **This hypothesis
  was stated in the brief, checked against the tree, and refuted before I
  changed anything.**
* **Not the barrier at all.** p9 is p5 with every barrier removed and it still
  fails. So the barrier is a bystander.
* **Not the launch, and not the `BufferArg` move-vs-clone.** p8 goes through the
  crate's own `rmsnorm_cuda`, which passes its buffers **by move**
  (`from_raw_parts(x_c.handle, ..)`), and it fails exactly like the hand-launched
  rungs — which all pass `.handle.clone()`, the form
  `fused_attnres.rs:504,516` uses. Same failure, both forms, so the move is not
  the cause. **The brief flagged this difference; it is refuted, not proven.**
* **What is left** is the shared-memory read-modify-write at an index derived
  from a **loop-carried runtime value** (`partial[tid + s]`, `partial[tid] +=`,
  `s` halving), inside a loop whose trip count is a runtime value.

### The fix, and why it is the fix

p10 is the identical reduction — same eight steps, same addresses, same barrier
per step — with the trip count made `#[comptime]`, which is the form every
working reduction in this tree already had:

```rust
// burn-attnres/src/fused_attnres.rs:351-357, running on every training step
for k in 0..lg {                       // lg from a #[comptime] parameter
    let stride = threads >> (k + 1);
    if tid < stride { red[tid] = red[tid] + red[tid + stride]; }
    sync_cube();
}
```

cubecl unrolls a comptime-bounded `for`, so every barrier lands in straight-line
code. `fused.rs` now takes `#[comptime] log_threads: u32` and does the same.
**The change is one loop header plus one parameter.** p10 and p6 agree to
1.7162427704345946e-7 — bit-identical, which is expected: at d=8 lanes 8..255
contribute exact zeros, so the tree and the serial sum agree in f32.

### The error's shape was a real hint

`Expected operand type llvm.ptr, but found builtin.integer` is a **pointer where
an integer is expected** — the opposite of what a mis-sized shared allocation
predicts. It says a value that should have become a pointer reached LLVM as a
raw integer, which is why the ladder spends most of its rungs on shared-memory
accesses and barriers rather than on sizes. That reading is what pointed at the
loop, and it is recorded here because it is the part of the brief that paid.

---

## 3. A SECOND defect, found on the way: the trailing cubes do not execute

The moment the kernel lowered, the tier-(a) oracle
(`tests/rmsnorm_kernel_cuda.rs`, the torch-ATen and
flash-linear-attention fixture) went red on a **numerical** mismatch, not a
compile one:

```
the FUSED KERNEL on case d2 differs from the reference's `out_torch` column by
5.245499700358899e-1 relative (> 1e-5)
```

`case d2` is `dims 1 2 2` — **two rows of two elements** — a shape
`fused_kernel_gate.rs` cannot see, because it only tries d in {4, 8, 16, 32}.

A sentinel (`-999.0` prefilled instead of `empty`) showed the trailing elements
were **never written**: not wrong, never written. The `rows x d` sweep:

```
rows=1 d=1: all written     rows=2 d=2: cubes 0,1 only
rows=1 d=2: all written     rows=2 d=3: cubes 0,1 only
rows=1 d=3: all written     rows=2 d=4: ALL
rows=1 d=4: all written     rows=3 d=2: cubes 0,1 only
rows=3 d=3: cubes 0,1 only  rows=3 d=4: ALL
rows=3 d=4: ALL             rows=4 d=2: cubes 0,1,2 only
rows=4 d=3: cubes 0,1,2 only  rows=4 d=4: ALL
```

### `tests/d2_isolate.rs` is the minimal reproducer, and what it excludes

The four variants of the real kernel — tree, no-conditional-write, serial
reduction, serial output — **all fail identically**, and so does a kernel with
**no shared memory and no barrier at all** (thread 0 doing the whole row). The
kernel body is therefore exonerated, and the bisect then exonerated everything
else too:

| suspect | verdict | evidence |
|---|---|---|
| the kernel body | **exonerated** | a 4-line kernel with one store drops the same cubes |
| the grid size | **exonerated** | `CubeCount::Static(4,1,1)` with a literal count ran 4/4, 5/5, 8/8, 12/12, 17/17, 31/31, 32/32, 64/64 |
| a runtime grid count | **exonerated** | 1..64 cubes from a runtime arg, all complete |
| the block size | **exonerated** | `CubeDim` 32/64/128/256/512/1024 truncate identically |
| a readback race | **exonerated** | 4 consecutive `into_data()` reads agree; an explicit `client.sync()` before the last changes nothing |
| **the output tensor's SHAPE** | **the trigger** | see below |

Same 4-line kernel, same 4-cube grid, same 16-element length; only the output
tensor's shape varies:

```
[1,16] 4/4   [2,8] 4/4   [4,4] 4/4   [8,2] 2/4   [16,1] 4/4
[2,1]  2/4   [3,1] 3/4   [4,1] 4/4   [1,4]  4/4   [1,2]  2/4
[1,1]  1/4   [5,1] 4/4
```

**The number of cubes that execute is a function of the output tensor's shape,
not of the grid.** The rule lives inside cubecl's launch path and **is not
isolated here** — I could not derive it from 12 samples and I am not going to
guess at it. What is established is the safe envelope: every `d >= 4` measured
(d in {4, 8, 13, 16, 32}, rows 1..=12) writes all its cubes, and every `d < 4`
measured with `rows > 1` does not.

### What shipped, and why it is a refusal and not a fix

`rmsnorm_cuda` now declines `d < MIN_FUSED_D` (4). That is deliberate and it is
the ADR-0019 rule, not a workaround for my own bug: the tensor path below it is
CORRECT, and `SKIPPED` is incremented, so the fallback is **COUNTED** rather
than a silent wrong answer. The alternative — shipping a fused arm that returns a
partly-unwritten tensor while the seam counter reports "ran" — is the exact
failure class this repo treats as a defect (`8fa5d4c`, the held-out eval passing
`hashed_ids = None`).

I declined the whole uncertain class rather than enumerating the shapes that
happen to work, because I do not know the rule. `MIN_FUSED_D` is a **measured
envelope, not a derived one**, and both halves of it are gates:
`fused_kernel_gate.rs` pins that `d < 4` DECLINES and that the fallback's answer
is correct, and `lower_probe.rs::p11_the_claimed_envelope` fails if the arm ever
runs below the floor.

**This is a real gap and it is left open on purpose**: the fused arm is now
correct but only for `d >= 4`, and no dormouse model uses `d < 4`.

---

## 4. Two defects in the GATE itself, which is why it could not have been trusted

Both were found by the fix, and both are the reason a gate that had never run
was worth writing before the diagnosis.

1. **The counter assertion was unsatisfiable by a working kernel.** It read
   `assert_eq!(takes, skipped - skipped0)` — that demands one skip per ask,
   which is the shape of a **dead** kernel. Its own failure message said
   "the fused arm DECLINED on a bare CUDA device", i.e. the message and the
   assertion said opposite things (the ADR-0020 defect, in a test). It could
   only ever have passed while the kernel never ran, which is precisely how
   `34c5631` and `9ac0377` both landed with it "passing" for a reason unrelated
   to the kernel working. Measured after the fix: `asked=6, skipped=0`, which the
   old assertion rejected. It now asserts what it means — nothing declined, and
   every one of the ROWS asks took the fused path.
2. **The two tests shared process-global counters with no lock.** `fused::calls()`
   is process-global statics and cargo runs the tests of one binary on several
   threads, so the sibling's ask landed inside this test's window and it read
   `took 7 asked, 1 skipped` for a test that had done 6 forwards. Both tests now
   hold a `SEAM` mutex.

---

## 5. The gate can fail — demonstrated, both directions

A gate that has never been red is not evidence.

**Red, the pre-fix kernel.** `lower_probe.rs::p5_tree_reduction` is that kernel
**verbatim** — same `while s > 0` halving tree, same shared read-modify-write,
same barrier — committed as an `#[ignore]`d reproducer, so the red side needs no
source mutation at all:

```
$ cargo test -p burn-rmsnorm --features cuda --test lower_probe -- --ignored --exact p5_tree_reduction
the lowered module does not verify: Compilation error: verification failed.
Expected operand type llvm.ptr, but found builtin.integer
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 11 filtered out; finished in 0.24s
```

`p7_full_kernel` (the whole pre-fix kernel) and `p9_tree_no_barrier_in_loop` (the
same tree with every barrier deleted) fail identically.

**Red, by reverting the fix in place.** I also mutated `fused.rs` itself — one
loop header, `for k in 0..log_threads` → `let mut s = threads / 2; while s > 0`
— and the gate went red with the same error. That round trip was interrupted
before its script printed (other lanes took the machine and the cubecl
autotuner's retry loop on a failed compile is slow), so the red output quoted
above is p5's rather than the mutation's. **What the round trip did establish is
the byte-identity**, which is the part that matters for the tree:

```
md5 of fused.rs before the mutation  70dfd2af837a60f26c84f213f8b27701
md5 of fused.rs while mutated         86a4d6701edb73619be1334f7c105849
md5 of fused.rs after the restore     70dfd2af837a60f26c84f213f8b27701   == before
```

I am reporting that as what it is: the revert/restore round trip preserved the
bytes, and the red side is demonstrated by the committed reproducer. I am not
claiming a clean scripted red/green transcript I do not have.

**Green, after the restore**, on the restored bytes:

```
$ cargo test -p burn-rmsnorm --features cuda --test fused_kernel_gate -- --nocapture
d in 1..3: the fused arm declined every one (COUNTED, not silent) and the tensor path answered all of them correctly
the FUSED kernel: asked 6, skipped 0, vs a scalar f64 definition - worst
1.1683123887065826e-7 relative over 84 outputs, 6 rows at d in {4,8,16,32} and
mean-square spanning 1e-4..1e2, tolerance 1e-5
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Whole crate, `--features cuda`: **33 passed, 0 failed, 3 ignored** (the three
reproducers). Whole crate, ndarray (no `cuda`): **7 passed, 0 failed** — the
`#[cfg(feature = "cuda")]`-gated targets are skipped, which is the point of
`required-features`.

`tests/oracle/mutate_fused_kernel.sh` covers the arithmetic mutants; this covers
the lowering regression.

## 6. What this does NOT establish

* **No backward, and that is the routing constraint.** `rmsnorm_cuda` returns a
  fresh `Tensor::<2>::empty` with raw handles written in, so the result carries
  no graph. Were it ever reached from an autodiff input the arm would return a
  leaf, receive **no gradient**, and every loss curve would still look healthy —
  the `8fa5d4c` defect verbatim. The dispatch refusal at `try_into_primitive` is
  the only thing preventing it, and
  `the_fused_kernel_declines_on_an_autodiff_device` is the gate that keeps it
  true. **Wiring this into a training forward means writing the adjoint first,
  not relaxing a `?`.** Out of scope here and not attempted.
* **The shape rule behind §3 is not known.** `tests/d2_isolate.rs` is the
  reproducer and the exclusion table; the mechanism inside cubecl is not
  isolated, and `MIN_FUSED_D = 4` is an envelope, not a law.
* **The runtime is not the trainer's.** The CUDA tests compile in the
  `vendor/dormouse-fused` workspace, where the repo root's `[patch.crates-io]` (which
  maps five vendored cubecl crates) does not apply, so the kernel ran on the
  **registry `cubecl 0.11.0-pre.4`** while the trainer runs the vendored fork.
  The arithmetic under test is our own source and does not depend on the runtime
  version; the claim is bounded by that and by nothing else.
* **Speed is unmeasured.** Nothing here says the fused arm is faster than the
  tensor path. The kernel has never run before today, so there is no step-time
  number for it, and the launch is the launch.
* arXiv:1910.07467's authors ship no code, so the tier-(a) columns are "the two
  public reference implementations of the mechanism"
  (`torch.nn.functional.rms_norm` at torch 2.14.0+cpu, and
  `fla.modules.layernorm.rms_norm_ref` at flash-linear-attention 0.5.2, both
  EXECUTED 2026-09-29 and pinned in `tests/fixtures/rmsnorm_oracle.txt`), not
  the paper's own code. A shared misreading of Zhang & Sennrich survives it.

---

## 7. Reproduce

```bash
cd vendor/dormouse-fused
# the gate, the numbers, and the d<4 refusal
cargo test -p burn-rmsnorm --features cuda --test fused_kernel_gate -- --nocapture
# the tier-(a) oracle against the kernel's own output
cargo test -p burn-rmsnorm --features cuda --test rmsnorm_kernel_cuda -- --nocapture
# the lowering ladder, one rung per process (a launch failure poisons the context)
cargo test -p burn-rmsnorm --features cuda --test lower_probe -- --exact p10_comptime_for_loop_tree
# the §3 reproducer and its exclusion table
cargo test -p burn-rmsnorm --features cuda --test d2_isolate -- --nocapture
# the reproducers, which fail on purpose
cargo test -p burn-rmsnorm --features cuda --test lower_probe -- --ignored
# the CPU/tensor oracle, which needs no GPU
cargo test -p burn-rmsnorm --test rmsnorm_oracle
```

**References, named per AGENTS.md §1.4.**
`burn-attnres/src/fused_attnres.rs:112-127, 351-357, 504, 516` — this repository,
read directly. `burn-gdn2/src/kernel/chunk_cube.rs:243`,
`burn-mhc/src/sinkhorn_cuda.rs:98`, `burn-spectral/src/moe_fused.rs:212` — this
repository, read directly. `cubecl-frontend 0.11.0-pre.4
src/frontend/container/slice/launch.rs:41-47` (`BufferArg::from_raw_parts(handle,
length: usize)`) and `cubecl-llvm 0.11.0-pre.4 src/shared/base.rs:458-463` (the
`verify_operation` that emits the error) — read from the **registry** crates this
build resolved, not from a vendored fork. `torch==2.14.0+cpu`,
`flash-linear-attention==0.5.2` — the two upstream references, executed
2026-09-29, transcripts in `tests/oracle/upstream/`, fixture in
`tests/fixtures/rmsnorm_oracle.txt`. The words "verified" and "bit-for-bit" are
not used about the §3 shape rule, because no external reference for it was
found.
