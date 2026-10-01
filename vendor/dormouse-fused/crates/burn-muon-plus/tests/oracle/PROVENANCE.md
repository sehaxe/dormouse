# `burn-muon-plus` tier-(a) oracle — provenance

Written 2026-09-30 at `wt/muon-oracle`, off `c3314e9`. Every sha, path, line and
number below was read out of the environment on that date; the commands are in
`transcript.txt`.

## What this oracle is compared against

| | |
|---|---|
| **repo** | <https://github.com/K1seki221/MuonPlus> |
| **commit** | `8a9ace123afedaab8ba75ea0b19315594ae1da7c` (HEAD; last push 2026-02-26T17:05:37Z) |
| **why this is the authors' code** | arXiv:2602.21545 **v1** (2026-02-25) and **v2** (2026-02-26), abstract, last sentence: *"We provide our code here: https://github.com/K1seki221/MuonPlus."* The repo was created 2026-02-25T03:20:22Z, one day after v1, and last pushed 2026-02-26, the day of v2. Not a fork. |
| **files pinned** | `utils/optim/muon_plus.py`, `utils/optim/polar_express.py`, plus the two empty `__init__.py` files the package layout needs |
| **functions used** | `apply_post_polar_norm` (`muon_plus.py:87-120`) — the expected values; the literal `a, b, c = (3.4445, -4.7750, 2.0315)` at `muon_plus.py:56` — gated at zero tolerance; `coeffs_list` (`polar_express.py:14-23`) — gated against what `src/lib.rs:186-188` claims about App. D.3 |

**The pin is byte-identical and machine-checked.** `gen_oracle.py` asserts the
sha256 of all four files before importing anything and exits 2 with `PIN ROTED`
on a mismatch; `falsify.sh` step D edits one digit of the pinned file and shows
it firing. The two non-empty files:

```
665ea6cb1f6f631f7839c430f2f2c8cbd62dd7e6b8a24fb752f1211c57ab2cab  utils/optim/muon_plus.py
ac792c2da2eab161be56e06f2c9804adfc2918bfa954a551ef5b5e9b43f6a871  utils/optim/polar_express.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  utils/__init__.py  (empty)
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  utils/optim/__init__.py  (empty)
```

They are byte-identical to
`https://codeload.github.com/K1seki221/MuonPlus/tar.gz/8a9ace123afedaab8ba75ea0b19315594ae1da7c`.
**No line of upstream source was edited, including to add a provenance header**,
which is why `docs/protocols/ORACLE-TIERS.tsv` carries a written waiver on these four rows
instead of a `github.com` URL in the file: the URL is here, and the sha256 is
enforced at generation time.

## v3 dropped the code link from its abstract

`aba296f` moved this crate's citation from "the paper" to **v3**, correctly —
v3 is the only version with the App. D coefficient tables. But v3 is also the
one version whose abstract **no longer names the code**:

```
v1  25 Feb 2026:  "We provide our code here: https://github.com/K1seki221/MuonPlus."   (2 hits)
v2  26 Feb 2026:  same sentence                                                       (2 hits)
v3  14 May 2026:  no github.com string anywhere in the paper's text                    (0 hits)
```

So a reader who cites v3 and then looks for the code finds nothing. The tier-(a)
provenance lives in `gen_oracle.py` and in this file for exactly that reason.

## The paper's own version history, measured

`pdftotext -layout` on all three PDFs (`transcript.txt` §1, and the file sizes:
v1 3 887 294 B / 9 pages, v2 3 887 286 B / 9 pages, v3 9 843 907 B / 12 pages):

| | v1 | v2 | v3 |
|---|---|---|---|
| occurrences of `3.4445` | 0 | 0 | **1** |
| occurrences of `1.875` | 0 | 0 | **3** |
| appendices present | A, B, C | A, B, C | A–G |
| `polar operator` occurrences | 0 | 0 | 1 |

v1 and v2 differ only in typo fixes (`emprical`→`empirical`, `powrful`→`powerful`,
`founation`→`foundation`, `Quantity`→`Quantitative`) and table reflow. The
`1.875` claim in the brief is confirmed: **neither v1 nor v2 contains the string
`1.875` at all**, and neither has an App. D.

App. D.1 (v3, verbatim): *"In [15], the coefficients are set to
`(a, b, c) = (3.4445, −4.7750, 2.0315)`."* — matches `NS_COEFFS` exactly.
§3.1 (v3, verbatim): *"For the polar operator, we adopt the same configuration
as in [15]."* `[15]` is Jordan et al. 2024, i.e. KellerJordan/Muon.

## The honest tier of each claim

| claim | tier | because |
|---|---|---|
| `NS_COEFFS` is the paper's polar operator triple | **(a)** | compared to a decimal literal in the authors' own committed source, zero tolerance |
| App. D.3's PolarExpress schedule ends at `(1.875, -1.25, 0.375)` | **(a)** | compared to the same list in the authors' own committed source |
| `normalize` implements the authors' `apply_post_polar_norm`, all four directions | **(a)** | expected values are that function's actual return values |
| the Newton-Schulz **iteration** | **not gated** | see below |

## The Newton-Schulz iteration, and why there is no numeric gate for it

The authors run it in **bfloat16** (`X = G.bfloat16()`, `muon_plus.py:55`); we
run f32. Measured on this box (`transcript.txt` §3, reproducible with
`measure_ns_snr.py`):

| ns_steps | shape | \|out\|max | noise (their dtype) | signal (`a` off by 1e-4 rel) | ratio |
|---|---|---|---|---|---|
| 5 | 8x8 | 0.4770 | 2.032e-01 | 5.810e-04 | 0.003 |
| 5 | 16x8 | 0.6394 | 2.126e-01 | 3.206e-04 | 0.002 |
| 5 | 32x8 | 0.7110 | 2.340e-02 | 3.968e-04 | 0.017 |
| 5 | 8x32 | 0.5499 | 4.032e-02 | 2.515e-04 | 0.006 |
| 5 | 64x16 | 0.4050 | 4.506e-02 | 1.498e-04 | 0.003 |

Worst signal/noise over every shape and every step count from 1 to 5:
**0.0004**. The dtype difference the authors' own code carries is **2500x
larger** than the defect a numeric gate would be hunting. Any bar that passes
every row is above the signal, so it catches nothing — a green line over
nothing, which is the failure this lane exists to prevent. **The constant is
gated instead, at zero tolerance, where a single wrong digit is caught exactly.**

Two more measurements from the same probe, because they close off the other
obvious shapes:

- **"forget the tall-matrix transpose" moves the output by 4.1e-6** — smaller
  than the bf16 noise by 50 000x. The transpose is a FLOP optimization, not a
  semantic change: the quintic is equivariant, so it is algebraically inert.
  There is no bar that catches it, and no bar should.
- `X Xᵀ` built the other way is a shape error on the wide arm, not a numeric
  one, so it is a compile-time or O(1) failure rather than a tolerance
  question.

## Re-running it

```sh
# The venv goes OUTSIDE the crate, and that is not a style preference:
# tools/oracle_gate.py walks tests/ for *.py, so a venv under tests/oracle/ is
# 5,509 unregistered files and turns the gate into 5,516 violations. Measured.
mkdir -p /tmp/muon-oracle-venv
uv venv --python 3.12 /tmp/muon-oracle-venv
uv pip install --python /tmp/muon-oracle-venv/bin/python torch numpy \
  --index-url https://download.pytorch.org/whl/cpu
cd vendor/dormouse-fused/crates/burn-muon-plus
/tmp/muon-oracle-venv/bin/python tests/oracle/gen_oracle.py        # writes muon_oracle.bin
/tmp/muon-oracle-venv/bin/python tests/oracle/measure_ns_snr.py    # the table above
bash tests/oracle/falsify.sh                                       # proves the gate can fail
cd ../.. && cargo test -p burn-muon-plus --test muon_oracle -- --nocapture
```

torch is 2.14.0+cpu, Python 3.12.14. `TORCHDYNAMO_DISABLE=1` turns the authors'
`@torch.compile` into eager; the arithmetic is unchanged.

`gen_oracle.py` is deterministic — no RNG is used anywhere in it — so
re-running it on any machine must reproduce `muon_oracle.bin` byte for byte.
That is itself checkable and is worth doing once after any torch upgrade:
`gen_oracle.py` prints the byte count, and a changed count is a changed fixture.
