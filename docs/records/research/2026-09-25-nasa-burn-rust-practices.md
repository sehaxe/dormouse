# NASA-grade Rust for a small CUDA trainer: JPL Power of Ten × burn × Rust perf idioms

Date: 2026-09-25. Scope: how to write fast, minimal, reliability-critical Rust for dormouse (byte-level LM, burn+cubecl, 16 GB box). Primary sources read in full: Holzmann's "The Power of Ten" (PDF, pdftotext), the burn repo (shallow clone, code sampled), Rust API Guidelines, Rust Performance Book, candle sources, Ferrocene/Kani docs.

## TL;DR

- **"Space software = no runtime checks" is factually wrong.** JPL Power of Ten **Rule 5 mandates an average of ≥ 2 assertions per function**: "The assertion density of the code should average to a minimum of two assertions per function... When an assertion fails, an explicit recovery action must be taken." NASA-style reliability = *more* checks, but cheap, side-effect-free, and with defined recovery — exactly Rust's `assert!`/`Result` discipline.
- burn's core trick for minimal+fast: **checks live in one place** (`TensorCheck` — panics on programming errors, single uniform error formatting), **rank is a const generic** (`Tensor<const D: usize, K = Float>`), **dtype/kind are sealed PhantomData marker types**, and **unsupported capabilities are uninhabited enums**, not error strings.
- candle proves **enum dispatch beats trait objects** for hot tensor paths: `pub enum Storage { Cpu, Cuda, Metal }`, flat `match`, zero vtables; `Clone` is deliberately not implemented because allocation can fail — `try_clone` returns `Result`.
- Adoptable today for a solo project: **clippy `-D warnings` (P10 Rule 10), one parametrized behavioral test suite run across fp32/bf16/CUDA (burn-backend-tests pattern), Kani proofs on the 3–4 gnarly pure functions (shuffle, ring buffer, hash indexing)**. Ferrocene/ISO 26262 qualification is aspirational, not practical for dormouse.
- Kernel style from burn-cubecl: bounds guard is the **first statement** of every kernel (`if !output.is_in_bounds(...) { terminate!(); }`), edge-case semantics (empty reductions) are decided explicitly in one function with cross-references to numpy/torch behavior.

---

## 1. NASA JPL Power of Ten — what it actually prescribes

Source read in full: G. Holzmann, "The Power of Ten — Rules for Developing Safety Critical Code", NASA/JPL Laboratory for Reliable Software (computer.org/10.1109/MC.2006.212, local copy `p10.pdf` → text). The rules exist to make code **mechanically checkable**: "the set of rules has to be small, and must be clear enough that it can easily be understood and remembered. The rules will have to be specific enough that they can be checked mechanically." They are deliberately "somewhat strict – one might even say Draconian."

The ten rules, verbatim rule text, with the load-bearing rationale:

1. **Simple control flow**: "Restrict all code to very simple control flow constructs – do not use goto statements, setjmp or longjmp constructs, and direct or indirect recursion." Rationale: "Without recursion, though, we are guaranteed to have an acyclic function call graph, which can be exploited by code analyzers, and can directly help to prove that all executions that should be bounded are in fact bounded." Note: the rule explicitly permits early returns — "an early error return is the simpler solution."
2. **Fixed loop bounds**: "All loops must have a fixed upper-bound. It must be trivially possible for a checking tool to prove statically that a preset upper-bound on the number of iterations of a loop cannot be exceeded." Suggested mechanism: "add an explicit upper-bound to all loops that have a variable number of iterations... When the upper-bound is exceeded an assertion failure is triggered, and the function containing the failing iteration returns an error."
3. **No dynamic allocation after init**: "Do not use dynamic memory allocation after initialization." Rationale: unpredictable allocator/GC behavior, use-after-free class of bugs, and — combined with Rule 1 — "an upper-bound on the use of stack memory can be derived statically."
4. **Function length**: "No function should be longer than what can be printed on a single sheet of paper in a standard reference format... Typically, this means no more than about 60 lines of code per function."
5. **Assertions — the rule the owner's mental model gets wrong**. Verbatim:
   > "The assertion density of the code should average to a minimum of two assertions per function. Assertions are used to check for anomalous conditions that should never happen in real-life executions. Assertions must always be side-effect free and should be defined as Boolean tests. When an assertion fails, an explicit recovery action must be taken, e.g., by returning an error condition to the caller of the function that executes the failing assertion. Any assertion for which a static checking tool can prove that it can never fail or never hold violates this rule."

   Rationale: "The odds of intercepting defects increase with assertion density." Assertions verify "pre- and post-conditions of functions, parameter values, return values of functions, and loop-invariants." And critically for a perf-bound trainer: "Because assertions are side-effect free, they can be selectively disabled after testing in performance-critical code." The paper's example is exactly a check-then-recover:
   ```c
   if (!c_assert(p >= 0) == true) { return ERROR; }
   ```
6. **Smallest scope**: "Data objects must be declared at the smallest possible level of scope."
7. **Check returns and validate parameters**: "The return value of non-void functions must be checked by each calling function, and the validity of parameters must be checked inside each function." Ignoring a return value must be explicit ("cast to (void)") or commented — never accidental.
8. **Preprocessor restraint**: macros only for "simple macro definitions"; no token pasting, no recursive macros, "All macros must expand into complete syntactic units"; conditional compilation "rarely... more than one or two" — with 10 conditionals there are 2¹⁰ versions to test.
9. **Pointer restraint**: max one level of dereferencing; no function pointers (they "can become impossible for a tool to prove absence of recursion").
10. **Zero warnings + static analysis, from day one**: "All code must be compiled, from the first day of development, with all compiler warnings enabled at the compiler's most pedantic setting. All code must compile with these setting without any warnings... checked daily with at least one... state-of-the-art static source code analyzer and should pass the analyses with zero warnings." Even on false positives: "the code causing the confusion should be rewritten so that it becomes more trivially valid."

**Verdict for dormouse**: JPL prescribes *aggressive runtime checking with disciplined recovery* (Rules 2, 5, 7), not absence of checks. The checks are cheap (Boolean, side-effect-free, disable-able in release), the recovery is explicit (return error / checkpoint-and-exit), and the point is analyzability: small functions, acyclic calls, bounded loops, fixed memory. Every one of these maps onto an existing Rust mechanism.

## 2. burn — concrete adoptable practices (code sampled)

Shallow clone of github.com/burn-rs/burn (main, 2026-09-25); crate sizes: burn-tensor ≈ 21k LOC, burn-cubecl ≈ 20k, burn-autodiff ≈ 11k, burn-core ≈ 9k. Source: `/tmp/opencode/burn`.

### 2.1 Centralized input validation, panic on programming errors

`crates/burn-tensor/src/tensor/api/check.rs` — all shape/dtype/device validation routes through one `pub(crate) enum TensorCheck { Ok, Failed(FailedTensorCheck) }`, and the design comment is a manifesto:

> "It's crucial that the checks are really fast, but it doesn't matter when a failed check is discovered since the program will panic. ... when there is no way to recover, panic should be used instead of a result. ... Almost all checks highlight programming errors, which means invalid programs that should be fixed. ... Maybe the Backend API should return a result for each operation ... The downside ... is that all backend implementations might re-implement the same checks, which may result in unnecessary code duplication."

Pattern: builders like `TensorCheck::binary_ops_ew(ops, &lhs, &rhs)` chain `.register(op_name, TensorError::new(...).details(format!(...)))` so every op's failure message carries the op name and the offending values. **This is P10 Rule 5+7 implemented as one reusable module instead of assertions sprinkled per-caller** — the lazier and more reliable shape of the same idea.

### 2.2 Type-level encoding so some checks aren't needed at all

- Rank is a const generic, kind is a PhantomData marker: `crates/burn-tensor/src/tensor/api/base.rs`:
  ```rust
  pub struct Tensor<const D: usize, K = Float> where K: Basic {
      pub(crate) primitive: BridgeTensor,
      _kind: PhantomData<K>,
  }
  ```
- The kind hierarchy is **sealed and layered** (`crates/burn-tensor/src/tensor/kind.rs`, file header: "Sealed traits, limited to Bool, Float and Int types"):
  ```rust
  pub trait Numeric: Basic + crate::ops::Numeric {}
  pub trait Ordered: Numeric + crate::ops::Ordered {}
  pub trait FloatMath: Numeric + crate::ops::FloatMathOps {}
  ```
  Each op tier requires the narrowest capability — impossible-to-satisfy calls are compile errors, which is P10's "trivially possible to check statically" taken to its best case.
- **Unsupported = uninhabited type**, not runtime error: `crates/burn-backend/src/backend/base.rs` defines `pub enum GraphUnsupported {}` — "Uninhabited: `graph_stop_capture` on such backends always errors, so a value of this type can never be constructed." Backends with no CUDA-graph support express that in their associated type (`type GraphPrimitive`), so the un-callable path is un-representable.

### 2.3 Error taxonomy: cheap backtraces, one enum

`crates/burn-std/src/device_settings.rs`:

```rust
#[derive(Error, Serialize, Deserialize, Clone)]
pub enum ExecutionError {
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    WithContext { reason: String },
    #[error("An error happened during execution\nCaused by:\n  {reason}")]
    Generic { reason: String, #[serde(skip)] backtrace: BackTrace },
}
```

with the doc note "Capturing is cheap — a `BackTrace` resolves its frames only when it is displayed — so this is the constructor to reach for by default." Two variants, one format, backtrace free until printed. A whole 20k-LOC GPU backend layer runs on this tiny taxonomy.

### 2.4 Kernel style (burn-cubecl / cubecl)

`crates/burn-cubecl/src/kernel/unary_numeric.rs` — representative shape:

```rust
#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn unary_numeric<T: Numeric, N: Size, O: NumericUnaryOpFamily>(
    input: LinearView<'_, Vector<T, N>>,
    mut output: LinearViewMut<'_, Vector<T, N>>,
    options: &O::Options,
    #[define(T)] _dtype: ElemType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    output.write(ABSOLUTE_POS, O::Unary::<T, N>::execute(input.read(ABSOLUTE_POS), options));
}
```

Adoptable specifics: (a) the bounds guard is the *first statement* of every kernel — the one runtime check that may never be skipped; (b) per-op logic is a small trait family (`NumericUnaryOpFamily` with `type Options: LaunchArg`) so the launch wrapper is written once and every new unary op is ~3 lines; (c) the unsafe fast path is gated: `if tensor.can_mut() && tensor.is_nonoverlapping()` before `launch_unchecked`, else the checked launch; (d) vectorization is computed, not configured: `let vector_size = max_vector_size(&tensor);`.

`crates/burn-cubecl/src/kernel/reduce/base.rs` — edge semantics decided in one place, with cross-references:

```rust
fn empty_reduce_identity(config: ReduceOperationConfig, dtype: DType) -> Option<f64> {
    match config {
        ReduceOperationConfig::Sum | ... => Some(0.0),
        // Only floats can carry `NaN`; an integer mean of nothing has no representable value.
        ReduceOperationConfig::Mean => dtype.is_float().then_some(f64::NAN),
        ... => None,
```

> "Reducing zero elements yields the folded operation's identity. The extrema have no identity in a bounded numeric type ... numpy raises `ValueError` and torch raises `IndexError` for all of them."

Every documented quirk in the codebase names the paper/ticket/tool that motivated it (e.g., `autotune key` doc: "Autotune key representative of sum versions" with `#[autotune(anchor)]`). Comments justify *why*, never *what*.

### 2.5 Autodiff structure: stateless ops, explicit state

`crates/burn-autodiff/src/ops/backward.rs`:

```rust
/// Concrete types implementing this trait should not have any state.
/// If a state is necessary during the backward pass,
/// they should be declared with the associated type 'State'.
pub trait Backward<B, const N: usize>: Send + core::fmt::Debug where Self: Sized + 'static, B: Backend {
    type State: Clone + Send + core::fmt::Debug + 'static;
    fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
}
```

Arity is a const generic (`N`), state is a named associated type instead of a captured closure — the backward pass stays inspectable and checkpointable.

### 2.6 Process rules (CONTRIBUTING.md + contributor-book)

- "PR authors must understand, justify, and explain every change they propose. ... Do not use 'AI generated' as a justification for low-quality code."
- "Keep dependencies minimal." / "Bug fixes should include a regression test."
- `cargo run-checks` = fmt + typos + dependency audit + clippy + no_std check + backend tests, one command — the Rule-10 discipline packaged for humans.
- One **shared behavioral test suite** (`crates/burn-backend-tests/tests/`: `tensor.rs`, `tensor_f16.rs`, `autodiff.rs`, ...) run against every backend/fp-format; a `#[might_panic(reason = "...")]` proc-macro documents *which* panic message is acceptable in a test.
- Module system (`contributor-book/src/project-architecture/module.md`): "`#[derive(Module)]` generates parameter traversal"; "A module does not force the declaration of the forward pass, leaving it up to the implementer" — the derive owns exactly the mechanical part (traversal/serialization), nothing else.

## 3. Rust performance idioms (API Guidelines, Perf Book, candle)

### 3.1 Dispatch: enum + match over dyn

candle (`candle-core/src/storage.rs`) is the counter-example to "backend = trait object":

```rust
pub enum Storage { Cpu(CpuStorage), Cuda(CudaStorage), Metal(MetalStorage) }
...
pub(crate) fn matmul(&self, rhs: &Self, ...) -> Result<Self> {
    self.same_device(rhs, "matmul")?;
    self.same_dtype(rhs, "matmul")?;
    match (self, rhs) {
        (Self::Cpu(lhs), Self::Cpu(rhs)) => { let s = lhs.matmul(rhs, ...)?; Ok(Self::Cpu(s)) }
        ...
        (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp { ... }.bt()),
    }
}
```

Flat enum dispatch: no vtable on the hot path, exhaustive match = the compiler enforces that a new backend arm is handled everywhere, device/dtype guards run once before dispatch, and the fall-through arm is a structured error (defensive `unreachable!()` only where a prior check makes it truly impossible). Reliability-relevant detail: **"`Clone` is deliberately not implemented on Storage as cloning may fail because of out of memory. Instead `try_clone` should be used."** — fallible operations are typed as fallible even when ergonomic cost is paid. Its trait `BackendStorage` (`backend.rs`) returns `Result` everywhere and marks `unsafe fn alloc_uninit` with a `# Safety` doc: "The caller should ensure that the data is properly initialized as early as possible."

Rule of thumb for dormouse: **generic (compile-time) dispatch for the default path; enum dispatch for the few genuinely runtime-chosen things (quant format, optimizer variant); `dyn` only at true plugin boundaries (none today).**

### 3.2 Perf book (Nethercote, primary)

- Build config (`build-configuration.html`): release builds alone are "10-100x" vs dev; for max speed `codegen-units = 1`, `lto = "thin"/"fat"`, `panic = "abort"`, alternative allocator, `-C target-cpu=native`; "Always use a faster linker if you are on a platform that supports it" (mold — dormouse already does); "Benchmark all changes, one at a time."
- Bounds checks (`bounds-checks.html`): remove checks by construction, not `unsafe`: "Replace direct element accesses in a loop by using iteration... make a slice of the `Vec` before the loop... Add assertions on the ranges of index variables." `get_unchecked` is "a last resort."
- Allocation: preallocate with capacity, prefer stack/`array`/`SmallVec`-style fixed buffers in hot loops (Heap Allocations / Standard Library Types chapters).

### 3.3 API Guidelines (checklist.html, primary)

The items that matter for a numeric trainer: **C-VALIDATE** "Functions validate their arguments"; **C-GOOD-ERR** error types meaningful and well-behaved; **C-CUSTOM-TYPE** "Arguments convey meaning through types, not `bool` or `Option`"; **C-NEWTYPE** static distinctions; **C-SEALED** sealed traits protect against downstream impls; **C-FAILURE** docs include error, panic, and safety considerations; **C-DEBUG** all public types implement `Debug`. Note C-VALIDATE is literally P10 Rule 7 and C-FAILURE is the doc-comment form of Rule 5.

## 4. Safety-critical Rust in 2026: practical vs aspirational

- **Ferrocene** (ferrous-systems): open-source qualified Rust toolchain — ISO 26262 **ASIL D**, IEC 61508 SIL 4, IEC 62304 Class C; the 26.02.0 release added ISO 26262 ASIL B for a certified subset of `core`. The *toolchain is free*, but qualification value comes from the quality-managed process around it (spec: the Ferrocene Language Specification is freely readable). For a solo ML project: **adopt the FLS as a language-behavior reference; certification itself is aspirational** — nobody audits dormouse, so paying Ferrocene's rigidity buys nothing beyond what stock rustc + clippy already give.
- **Kani** (model-checking/kani): bit-precise model checker, stable, ASE 2026 paper. Harness style:
  ```rust
  #[kani::proof]
  fn check_my_property() {
      let input: u8 = kani::any();
      assert!(meets_specification(input, function_under_test(input)));
  }
  ```
  Automatically checks panics, arithmetic overflow, and custom assertions over **all** inputs. **Practical for dormouse today** on the small pure host functions where a wrap-around or off-by-one is catastrophic and inputs are bounded: Fisher–Yates shuffle index math, the 64 MB ring-buffer wrap, FNV key packing for the Engram tables (a collision/aliasing proof), host-Adam row-index bounds. Kani on GPU code is not practical; keep the proof surface pure and host-side.
- **MISRA-style rule sets for Rust**: no dominant finished standard as of 2026; the Rust Safety-Critical Consortium is still consolidating practice. The pragmatic equivalent for one person: Power of Ten (10 rules) + Rust API Guidelines checklist + `clippy --all-targets -D warnings`. That combination is *more* mechanically checkable than most 100-rule MISROns.

## 5. Rules for dormouse

1. **Centralize shape/dtype/device validation in one check module; panic there, don't thread `Result` through every op.** One place to keep messages honest, zero duplication, and programming errors crash loudly at the call site. [1][2]
2. **Average ≥ 2 cheap assertions per non-trivial function** (`assert!`/`debug_assert!` on pre/post-conditions, loop bounds, invariants like "grad norm finite"); side-effect-free only; hot-loop ones under `debug_assert!` so release stays fast — this is the literal NASA rule, not its negation. [1]
3. **Every assertion failure has a named recovery at the top**: the train loop's NaN/panic guard saves the checkpoint and exits (already implemented via `--guard`) — that *is* Rule 5's "explicit recovery action". No check exists without a defined failure path. [1]
4. **No heap allocation inside the step loop; all buffers sized at init** (ring, batches, workspace, ckpt BufWriter buffer). Anything fallible during the loop (ckpt write, host-Adam sync) returns `Result` and is checked. [1][6]
5. **Functions ≤ 60 lines, acyclic call graph, no recursion in the training path.** The PonderNet loop is iteration, not recursion. Keeps every function reviewable as a unit. [1]
6. **Zero warnings is a gate**: `cargo clippy --all-targets -D warnings` (and `cargo run-checks`-equivalent) must pass before any commit; on a false positive, rewrite the code to be "more trivially valid," don't silence. [1]
7. **Const generics for rank, sealed PhantomData markers for dtype/kind tiers, uninhabited enums for unsupported paths** — every check the type system can do is a check that costs zero nanoseconds and never rots. [2]
8. **Enum dispatch (`match`) for runtime-switchable variants (quant format, optimizer, attention arms); no `dyn` in the hot path.** Exhaustive matches make adding an arm a compile-error-driven tour of every site. [6]
9. **Fallible ops return `Result` with a small thiserror enum + lazy `BackTrace::capture()`; ignoring a `Result` requires an explicit `let _ =` with a why-comment** — the Rust rendering of P10 Rule 7. No `unwrap` outside tests and provably-total code (`Option::Some` matches with exhaustive arms). [1][2][6]
10. **Macros only for mechanical repetition that the compiler could check by hand otherwise; never to hide control flow or dereference-like logic.** Derive-style macros only for traversal/serialization boilerplate. [1]
11. **One behavioral test suite, parametrized over fp32/bf16 × CPU/CUDA**, asserting the same op results within documented tolerance — format regressions (the Fp8 forward, bf16 casts) get caught by the same harness, not by watching loss curves. Bug fixes add one regression test. [2]
12. **Kani proof harnesses on the bounded pure host functions**: Fisher–Yates permutation bijectivity, ring-buffer wrap, FNV key packing/unpacking, row-index bounds in host tables. The rest of the GPU stack relies on rules 1–11 + integration tests; **Ferrocene/ISO 26262 is aspirational reading (FLS as language reference), not adopted.** [7]

## References

1. G. Holzmann, *The Power of Ten — Rules for Developing Safety Critical Code*, IEEE Computer, 2006 (NASA/JPL Lab for Reliable Software). PDF read in full; rules quoted verbatim. https://spinroot.com/gerard/pdf/P10.pdf
2. burn repository, main branch (shallow-cloned 2026-09-25): `crates/burn-tensor/src/tensor/api/check.rs`, `crates/burn-tensor/src/tensor/api/base.rs`, `crates/burn-tensor/src/tensor/kind.rs`, `crates/burn-backend/src/backend/base.rs`, `crates/burn-std/src/device_settings.rs`, `crates/burn-cubecl/src/kernel/unary_numeric.rs`, `crates/burn-cubecl/src/kernel/reduce/base.rs`, `crates/burn-autodiff/src/ops/backward.rs`, `crates/burn-backend-tests/`, `CONTRIBUTING.md`, `contributor-book/src/project-architecture/module.md`. https://github.com/burn-rs/burn
3. Rust API Guidelines, checklist. https://rust-lang.github.io/api-guidelines/checklist.html
4. N. Nethercote et al., *The Rust Performance Book* — chapters "Build Configuration", "Bounds Checks". https://nnethercote.github.io/perf-book/
5. Ferrocene toolchain and qualification status (ISO 26262 ASIL D; IEC 61508 SIL 4; IEC 62304; 26.02.0 adds ASIL B core subset). https://ferrocene.dev/ , https://ferrous-systems.com/blog/ferrocene-26-02-0/
6. candle repository: `candle-core/src/storage.rs`, `candle-core/src/backend.rs` (raw sources read 2026-09-25). https://github.com/huggingface/candle
7. Kani Rust Verifier, README + verification model (checks panics/overflows/assertions over all inputs; ASE 2026 paper). https://github.com/model-checking/kani
