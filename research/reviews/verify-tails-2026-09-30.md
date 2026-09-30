# Closing the four tails the verification audit found

**Lane:** `verify-tails`, worktree `wt/verify-tails` off `e31e335`.
**Date:** 2026-09-30. **Commits:** `3533ade`, `7555ac1`, `0c0a7e2`, `f9dea8a`.
**Audit read:** `research/reviews/fix-verification-2026-09-30.md` (895 lines, `wt/fixverify`).

Every cargo invocation went through `tools/build_lock.sh run`. A training control
was on the GPU throughout (`first_run_2000`, then `first_run_10000-`); all work
below is CPU/ndarray, so nothing here needed the card and nothing collided with
the run.

---

## 0. Verdict

| # | tail | what I did | gate |
|---|---|---|---|
| 1 | `dspark_loss` does not detach the confidence target | fixed, **plus** two more defects in the same expression, **plus** the fixture had no gradients at all | `dspark_loss_gradients_agree_with_the_official_loss`, red→green demonstrated at 3270x–279147x the bound |
| 2 | seven doc comments state the opposite of the code | adjudicated one at a time: 4 stale docs reworded, 1 false coverage claim made executable, 2 stale line references replaced | unchanged: 10+4+7 green |
| 3 | `mutate_kernel.sh` M6 is dead | re-anchored at the parity predicate, **and** a dead anchor now fails the sweep instead of passing it | 6 mutants, every one red, 0 dead anchors |
| 4 | the generator's documented command is incomplete | fixed, and verified by following it from a clean venv | regenerates byte-identical from scratch |

Nothing was refused. Two findings were larger than the brief expected — both in
item 1 — and one of them (the stop-gradient finding) changes what a legitimate
gate on this quantity can even be.

---

## 1. The DSpark confidence target

### 1.1 The fix, and the numbers

`loss.py:145` is `confidence_targets = accept_rate_3d.detach()`. The
analytical acceptance rate is computed **from `draft_logits`**, so leaving it
attached pulls `-cl * d(c_star)/d(draft_logits)` into the drafter's gradient —
a term upstream does not train, carrying the **largest** of the three weights
(`confidence_head_alpha` = 1.0, against 0.1 and 0.9).

Measured against the reference's own autograd, L2 norm of `d(total)/d(draft_logits)`:

| case | \|g\| no-detach (was ours) | \|g\| detached (DeepSeek) | ratio |
|---|---|---|---|
| `saturated_conf` | 1.227131 | 0.073771 | **16.6x too large** |
| `block7_exact` | 0.118244 | 0.087681 | 1.35x |
| `tiny_mask` | 0.131755 | 0.153709 | 1.17x |
| `main` | 0.056004 | 0.054848 | 1.02x |
| `aligned_identical` | 0.034983 | 0.034983 | 1.00x |
| `big_logits` | 0.048648 | 0.048648 | 1.00x |
| `all_masked_off` | 0.0 | 0.0 | — |
| `no_confidence_head` | 0.057351 | 0.057351 | 1.00x |

Four of eight move. The four that do not are the **negative controls** and they
are in the fixture on purpose: `all_masked_off` (the mask zeroes every
gradient), `aligned_identical` and `big_logits` (`accept_rate` clamps to 1 and
its gradient is 0), `no_confidence_head` (no head). An implementation that
built a numeric indicator from the target anyway moves them.

One line, `src/lib.rs`:

```rust
let c_star = accept_rate_target(draft_logits, target_logits).detach(); // [B, L]
```

### 1.2 A second defect, in the same expression, that the audit did not find

With the detach fixed the gradient gate still failed — on `grad_conf` for
`aligned_identical`, at **exactly 2.0x** on 10 of 14 coordinates:

```
aligned_identical  grad_conf  10/14 coordinates outside the bound;
                   worst at [0] ours -1.723984e-1  DeepSeek -8.619919e-2  9885.3x the bound
```

Root cause, read rather than guessed. The BCE was spelled the way it is usually
written out, `max(x,0) - x*c + log(1+exp(-|x|))`. That is correct as a **value**
and correct as a gradient everywhere except the two kinks it introduces itself.
At `x == 0` exactly, burn's `relu_backward` masks `output <= 0`
(`burn-backend/src/backend/ops/activation.rs:55`, so `relu'(0) = 0`) and `abs`'s
subgradient at 0 is 0, so the derivative collapses to `-c` where the true
derivative — and `binary_cross_entropy_with_logits`' — is `sigmoid(x) - c`.

The fixture's `aligned_identical` case has `confidence_pred` all **zeros**, so it
lands on the kink exactly, and the recorded gradient is `w_k/den` = 0.172398
against DeepSeek's half of it.

Fixed by using the algebra `binary_cross_entropy_with_logits` is itself written
in, which cannot have a kink and is one term shorter:

```rust
let one_minus_c = c_star.neg().add_scalar(1.0);
let bce_per = cl.clone().mul(one_minus_c).add(activation::softplus(cl.neg(), 1.0));
```

Three terms become two, `relu` and `abs` disappear, and the `|x| = 40` stability
the earlier fix was for is unchanged (`softplus` switches to the identity above
its threshold). The forward values did not move: `the_confidence_term_agrees_
with_the_official_loss` stayed green throughout.

This is the shape of thing the project has paid for three times today — a
correct-looking formula, right in value, wrong on a set the fixture only reaches
by construction. It was found on the **first run** of the new gate.

### 1.3 The thing that changes what a gate on this quantity can be

My first draft of the generator self-validated `grad_draft` by central-differencing
the official scalar. It refused to emit, on 20 of 24 probed coordinates. That
was not a tolerance problem and I did not loosen a tolerance:

| quantity (case `main`, d/d(draft[0])) | value |
|---|---|
| autograd through DeepSeek's own code | **-0.00598767** |
| ce + l1 terms only, target detached | **-0.00598767** |
| ce + l1 + a LIVE confidence target | **-0.00991132** |
| central difference of the forward scalar | **-0.00991344** |

The central difference is **stable** — I swept `h` from 1e-3 to 0.2 and it never
moved off -0.00991. It converges to the LIVE target's gradient, because the
`.detach()` makes the returned scalar **non-differentiable-consistent with the
graph that produced it**. A finite-difference gate would therefore report the
correct stop-gradient as broken, and by a stable, confident, large margin.

Two consequences, and the second is the important one:

1. A tier-(a) reference for this quantity exists **only** as autograd on the
   pinned source. There is no finite-difference route to it.
2. The detach is not a choice between two defensible gradients. **Ours was the
   live one** — we were training a term upstream does not train.

So the three checks the generator runs, each with one job:

| check | what | tolerance |
|---|---|---|
| invariance | `grad_draft` **bit-identical** with and without a head | exact, none offered |
| cross-check | recorded gradients vs this file's own differentiated transcription | 1e-3 rel, 1e-5 abs |
| finite difference | `grad_conf` only — valid, nothing on that path is detached | 5e-3 rel, 2e-5 abs |

All pass on all 8 cases; worst residuals are 0.0 (invariance and cross-check on
6 cases) and 4.04e-05 absolute on the conf FD. The cross-check is explicitly a
**tier (b) instrument under a tier (a) number** — it cannot prove the reference,
only that the recording is not mis-shaped. Recorded in
`docs/ORACLE-TIERS.tsv` and in the generator's docstring.

### 1.4 The gate, and its red→green demonstration

`dspark_loss_gradients_agree_with_the_official_loss`. Bound
`|ours − DeepSeek| ≤ 1e-7 + 1e-4·|DeepSeek|`, **read from the fixture header**
so the tolerance travels with the numbers it was measured against.

The bound is derived, not tuned: with the fix in place the worst element over
all 366 drafter and 111 head coordinates is **0** on `aligned_identical` /
`big_logits` and **≤ 3.7e-09 absolute** elsewhere — about two decades of f32
headroom. In the other direction:

```
main              448/448 coordinates out, worst 10748x the bound
tiny_mask          32/448,               worst  3270x
saturated_conf    320/448,               worst 279147x
block7_exact      224/224,               worst  7967x
```

**All three value tests stay green with the detach removed.** That is the whole
argument for having the gradient gate, and it is why I would not accept a value
oracle here as sufficient no matter how many cases it has.

Re-revert procedure: `cp` snapshot, `md5sum` recorded, single-line removal,
test, `cp` restore, `md5sum -c`. Restored byte-identical
(`353e0c09bd67479431ec27be2547a39e`). I used `cp` and not `git checkout -- .`
throughout — the audit's own process note records that one going the other way
and producing a false "the restore did not work".

**Suites:** `dspark_loss_oracle` 10/10 (was 8, +2 new), `burn-dspark --lib`
17/17, `dormouse-core --lib` 53/53, `e2m1_oracle` 4/4, `act_quant` 7/7,
`cargo check --workspace` clean.

**One device change with a reason:** `burn`'s `autodiff` feature is now a
dev-dependency of `burn-dspark` for the gradient test. That makes
`Device::default()` an autodiff device, so the three pre-existing value tests
now name `Device::ndarray()` explicitly. A golden comparison that silently
changes backend is a golden comparison nobody re-ran.

---

## 2. The seven doc comments, adjudicated one at a time

"Fix the doc" and "the doc is right and the code is the lie" are different jobs.
Each was checked against the code or the fixture before rewording, not from the
audit's word. Six were real; the seventh (`mutate_kernel.sh`) is item 3.

| # | file:line | verdict | what |
|---|---|---|---|
| 1 | `act_quant.rs:113` | **doc wrong** — reworded | said "ties away from zero" (= ties-UP). The rule has been ties-to-**even-code** since `dc5d667`, 40 lines below, with the seven-tie table beside it. I re-derived the tie behaviour from the ladder rather than trusting either comment: at all seven interior ties our level equals the even-code member. Reworded to say why there is no "towards" direction at all — even-code **alternates**, so 0.75 and 3.5 go up and the other five go down. |
| 2 | `act_quant.rs:217` | **doc wrong** — reworded | said "ties to the coarser level". Right about five of seven, wrong about 0.75 and 3.5. Reworded to "the member whose code is even, which alternates". |
| 3 | `burn-dspark/src/lib.rs:6-9` | **doc wrong** — reworded | said "NOT yet matched against the official DeepSpec implementation — the 7 tests here are hand-derived and **cannot detect a wrong loss term**". The crate has a tier-(a) oracle that ran DeepSeek's own loss, and it has now found and fixed **three** wrong loss terms. Reworded to name the oracle, what it caught, and — the sentence that was still true — that the rest of the crate is hand-derived from the paper with no reference of any kind. |
| 4 | `e2m1_oracle.rs:161` | **doc wrong** — reworded | "RED ON PURPOSE … Not fixed here … the fix is the owner's". Fixed in `dc5d667`. Reworded to record the defect, its measurement, the fixing commit, and that **re-reverting `dc5d667` turns the test red at exactly the four named ties**. A red-on-purpose marker left in a green test is worse than none: the next reader is told the code is broken when it is not. |
| 5 | `dspark_loss_oracle.rs:275` | **doc wrong** — reworded | same shape as 4, for `8c3bd2a`, including the stale claim that the fix "is NOT made here, because this lane's task is the evidence and the fix is the owner's". Both fixes are made; the re-revert numbers are recorded. |
| 6 | `dspark_loss_oracle.rs:56-62` | **doc wrong, and it was a coverage claim the fixture did not deliver** — reworded in two places, and made executable | see below. |
| 7 | `mutate_kernel.sh:118-121` | **script wrong** — item 3 | |

### 2.1 Item 6 in detail: the denominator claim

The `den + 1e-6` vs `den.clamp_min(1.0)` divergence is **real**, bounded, and
unreachable from the live call site (`crates/dormouse-core/src/aux.rs` passes an
all-ones mask, so `wm.sum() ≥ 2.86`). The comment claimed `all_masked_off` and
`tiny_mask` made the region reachable. Measured, they sit at `den` = 0.0 and
`den` = 1.0, where the two formulas differ by 1.3e-7 and 1.0e-6 relative against
a `TOL_REL` of 1e-5 — one to two orders of magnitude **below** the tolerance.
The audit's table and mine agree on every case.

So the doc is the defect and I did **not** make the region reachable: doing so
means shipping a fixture case the test then fails on for a reason that is not a
defect, which would trade a false claim for a red gate. Instead:

* both comments now state the measured boundary rather than a coverage claim;
* the Rust side gained `no_fixture_case_reaches_the_denominator_difference`,
  which computes the gap per case — **and skips cases whose numerators are
  exactly zero**, which is the part the first draft of it got wrong and the
  `all_masked_off` case caught — then fails if any case ever does reach it.

The claim is now executable, so it cannot rot back into a coverage claim. This
is the generalisable shape: a coverage claim is a test that does not exist.

---

## 3. `mutate_kernel.sh` M6 was dead, and the script could not tell me

M6 anchored on `mask_fill(a.clone().greater_equal_scalar(lo), *level)`, which
`dc5d667` **replaced** with an `if i % 2 == 0` parity predicate — because
ties-to-even-code alternates along the ladder and no single `>=` expresses it.
I confirmed the anchor count is 0. `patch()` exits non-zero on a non-unique
anchor, so M6 printed "PATCH FAILED", **ran no test**, and the sweep continued
reporting success. The single most valuable mutant in the file — the one its own
comment calls "a mutant that 'fixes' a red test" — was not running, and nothing
said so.

M6 is re-anchored at the parity predicate and **inverts** it, giving
ties-to-odd-code: still wrong, at a **different** set of ties than the current
code gets wrong. That is the property worth keeping — if it ever goes green, the
test is pinned to "not ties-up" rather than to the reference's rule.

A dead anchor is not a one-line shell defect; it is the same class as a gate
nothing runs (`34c5631`/`9ac0377`) and a fix nothing can see (`dc5d667`). So
the script now cannot have that failure quietly:

* `patch()` failure is counted; `perturb` says **"THIS MUTANT DID NOT RUN"** in
  the mutant's own slot rather than only in the trailer; the sweep **exits
  non-zero** if any anchor no longer matches.
* A mutant that leaves its gate **green** is recorded and the sweep **exits
  non-zero**. "Ran" and "killed something" are now different claims and both
  are checked.
* The per-mutant diff compared each mutated file against the **other crate's**
  snapshot, printing two unrelated files at each other — 60 lines of noise
  burying the one line that changed. Fixed to diff against its own.

**Sweep, under the build lock:**

```
baseline          10 + 4 green
M1 decay dropped   4 failed
M2 mask dropped    4 failed
M3 0.5 doubled     3 failed
M4 0.75 back       3 failed
M5 scale to 1      2 failed
M6 parity inverted 1 failed   <- was silently not running
after             10 + 4 green, both files md5-identical to the snapshot
verdict           6 mutants, every one red, 0 patch failures, 0 dead anchors
```

Baselines updated: both "N red ON PURPOSE" lines were stale, and the header
described the script as covering the direction a red test cannot. It now says
the opposite, which is the honest reason the mutants exist — a red test proves
it can fail only while it is red, and all three defects it named have since
been fixed.

---

## 4. The generator's documented command

`gen_dspark_loss_oracle.py` said `uv pip install --index-url
https://download.pytorch.org/whl/cpu torch numpy`. That does not work, and the
error does not say why: **`--index-url` replaces the index**, so the CPU wheel
host is the only place the resolver looks, and it carries torch and numpy and
nothing else.

```
error: No solution found when resolving dependencies
  cause: Because pyyaml was not found in the package registry and you
  require pyyaml, we can conclude that your requirements are unsatisfiable.
```

Worse than a resolution failure: `torch numpy` **does** install, and the
generator then dies at `import` with a bare `ModuleNotFoundError: No module
named 'yaml'`, pointing at DeepSeek's import graph rather than at the missing
dependency. `deepspec/modeling/dspark/__init__.py` → `common.py` reaches
`deepspec/utils/config.py` (`import yaml`) and
`deepspec/modeling/dspark/gemma4/modeling.py`
(`from transformers.cache_utils import Cache`).

The documented recipe is now **two commands** — torch from the wheel index,
everything else from PyPI — with the failure quoted, because that error is the
thing that will be searched for. `--extra-index-url` also resolves it, and I did
not use it: it silently relaxes dependency-confusion protection, which is not a
flag to put in a reproducibility recipe.

**Verified by following the written instructions from scratch**, on a clean venv
built only from the documented four commands:

```
uv venv --python 3.12 /tmp/opencode/fresh2
VIRTUAL_ENV=… uv pip install --index-url …/whl/cpu torch==2.14.0+cpu
VIRTUAL_ENV=… uv pip install numpy pyyaml transformers
git clone https://github.com/deepseek-ai/DeepSpec.git …
python gen_dspark_loss_oracle.py --deepspec … > ../fixtures/dspark_loss_oracle.txt
```

```
0d8bd297fba1194787232ab3e3291bd2  regenerated
0d8bd297fba1194787232ab3e3291bd2  committed fixture
BYTE-IDENTICAL
```

**torch is pinned, and the pin is worth it.** Unpinned, the resolver picks
2.14.1 today and newest-tomorrow; all 8 cases' values and all 477 gradient
coordinates came out bit-for-bit the same on both — itself worth recording —
but a golden that diffs by one line every few weeks trains everyone to ignore
fixture diffs, which is the expensive failure mode.

There is no venv setup script in the tree; the generator's docstring was the
only place the command lived.

---

## 5. Follow-ups, not mine

1. **`tools/oracle_gate.py` reports 1 violation and it is pre-existing.** The
   `burn-muon-plus/tests/oracle/muon_oracle.bin` row points at a file that is
   not in the tree. Verified pre-existing by stashing my change and re-running
   (same 1 violation). Other lane's row; I did not touch it.
2. **The `den + 1e-6` divergence is still in the code**, bounded and unreachable.
   Deciding whether to match upstream exactly is the owner's: it is a numerical
   change to a shipped objective for no benefit at the only live call site. The
   doc and the new test both name it rather than hide it.
3. **AGENTS.md §3.2 needs a row for this lane.** The AGENTS.md text at the time
   of writing still describes `burn-dspark`'s `dspark_loss` without the detach
   and the gradient gate, and its `--dspark-weight` bullet lists two fixed
   things and not the third. I did not edit AGENTS.md: another lane owns it and
   §1.6 says not to edit files an agent is in.

---

## 6. Reproduction

```bash
tools/wt.sh new verify-tails            # off e31e335
cd /home/sehaxe/dormouse-wt/verify-tails

# item 1 — the suites
tools/build_lock.sh run t -- bash -c 'cd vendor/burn-fused && \
  cargo test -p burn-dspark --features training --test dspark_loss_oracle'
tools/build_lock.sh run t -- bash -c 'cd vendor/burn-fused && \
  cargo test -p burn-dspark --features training --lib'

# item 1 — the RED demonstration (cp snapshot, md5 before and after)
SRC=vendor/burn-fused/crates/burn-dspark/src/lib.rs
cp "$SRC" /tmp/lib.rs.bak && md5sum "$SRC"
python3 - <<'PY'
p="vendor/burn-fused/crates/burn-dspark/src/lib.rs"
s=open(p).read()
a="accept_rate_target(draft_logits, target_logits).detach()"
assert s.count(a)==1
open(p,"w").write(s.replace(a,"accept_rate_target(draft_logits, target_logits)"))
PY
tools/build_lock.sh run t -- bash -c 'cd vendor/burn-fused && \
  cargo test -p burn-dspark --features training --test dspark_loss_oracle'
cp /tmp/lib.rs.bak "$SRC" && md5sum "$SRC"   # 353e0c09bd67479431ec27be2547a39e

# item 1 — regenerate the fixture from a clean venv, following the docstring
cd vendor/burn-fused/crates/burn-dspark/tests/oracle
/tmp/opencode/fresh2/bin/python gen_dspark_loss_oracle.py --deepspec /tmp/opencode/fresh2-ds \
  > /tmp/regen.txt && md5sum /tmp/regen.txt ../fixtures/dspark_loss_oracle.txt

# item 3 — the full sweep
tools/build_lock.sh run m -- bash vendor/burn-fused/crates/burn-dspark/tests/oracle/mutate_kernel.sh

# item 2 / 4 — gates
tools/build_lock.sh run t -- bash -c 'cd /home/sehaxe/dormouse-wt/verify-tails && \
  cargo test -p dormouse-core --test e2m1_oracle && \
  cargo test -p dormouse-core --lib act_quant && cargo test -p dormouse-core --lib'
tools/oracle_gate.py        # 115 registered, 184 scanned, 1 pre-existing violation

git log --oneline e31e335..HEAD
# 0c0a7e2 test(dspark): M6 ran no test at all
# f9dea8a docs(oracle): the dspark rows now say what the gradient gate covers
# 7555ac1 docs(dspark): adjudicate six doc comments against the code, one at a time
# 3533ade fix(dspark): the confidence target is a LABEL - detach it, and gate the gradient
```

### Citations, per AGENTS.md §1.4

* **`DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`** (2026-07-09),
  `deepspec/modeling/dspark/loss.py`, sha256
  `2e91efcaff780eec0748ef3f6f0a31374f119f609c664cc79289fdd922335328`. Fresh
  clone on this box 2026-09-30, still HEAD upstream on that date. Its
  `compute_dspark_loss` was **executed** on CPU, torch 2.14.0+cpu, one-rank
  gloo, and its **backward** is the gradient reference. `loss.py:145` (the
  detach), `:151-156` (the BCE), `:239-252` (the denominators) read from the
  clone.
* **`pytorch/ao@3972ed01`** (2026-09-25) — cited in the reworded
  `act_quant.rs` comment for the tie rule; I did **not** re-run torchao this
  lane, so the e2m1 claim here rests on the fixture already in the tree and on
  my own reading of the ladder. The audit's §2 re-derivation from raw nibbles is
  the evidence for that rule and it stands unchallenged.
* **torch 2.14.0+cpu** (wheel from `download.pytorch.org/whl/cpu`), numpy 2.5.2,
  transformers 5.17.0, pyyaml. Both venvs built from scratch on 2026-09-30;
  2.14.1+cpu also run, and produced identical values and gradients.
* **burn-tensor / burn-backend 0.22.0-pre.4** — `activation.rs:55` for
  `relu_backward`'s `<= 0` mask, which is what makes the kink. Read, not run:
  the kink's effect is measured (9885x the bound on a fixture coordinate), the
  mechanism behind it is read.
* The L2 gradient table in §1.1 is computed with **DeepSeek's own formulas** in
  torch on the fixture's inputs, faithful to both implementations and not a
  call into either — the same construction the audit used, re-derived
  independently and agreeing with it to six significant digits on every case.
* arXiv:2607.05147 (DSpark) ships no code that was run here; no claim is made
  about the paper's authors' implementation, only about DeepSeek's released
  one.
