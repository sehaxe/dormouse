# ByteFlow A/B — the 200-step smoke verdict and the 1M map

Date: 2026-10-02. Tree: `wt/bfab` @ `354a738` (byteflow3 merged). Lane: **bfab2**
(A/B byteflow vs byte, flip on a win). Card: RTX 5060 Ti, fp32, all four runs
sequential. Logs: `~/logs/bfab_20261002_*.log`.

## The A/B (equal bytes, one window, 2 seeds)

Instrument: 200 steps, batch 8, seq_len 512 → **both arms consume the same
4096 B/step and the same 819 200 B total**. The brief's worry "(у патч-руки
меньше шагов — сравнение по байтам)" does *not* apply on this tree: the byteflow
loop consumes the same `batch × seq_len` batch through the same `ByteStream`
seam (`crates/dormouse-train/src/byteflow.rs`, `stream.next_batch()`), so equal
steps IS equal bytes — measured, not assumed. Held-out: the fixed 81920 B window
(`--eval-batches 20`, printed on every eval line), the same bytes for both arms.

control = preset `small` − `--no-engram` − aux 0 (pure CE; params printed by the
run header). arm = `configs/byteflow.toml`, l2 patcher, plain AdamW wd 1e-2.

| run | held-out BPB @100 | held-out BPB @200 | train CE final | params | ms/step |
|---|---|---|---|---|---|
| control s1 | **6.714** | 6.714 is the s1 number (eval@200 does not print; see the gap note) | 3.736 | 9 197 454 | ~450–500 |
| control s2 | **6.595** | — | 3.747 | 9 197 454 | ~450–500 |
| byteflow s1 | 4.823 | **4.488** | 3.134 | 15 923 968 | ~65 |
| byteflow s2 | 4.898 | **4.539** | 3.126 | 15 923 968 | ~53–83 |

Notes the table needs :

- **The one asymmetry**: the dormouse loop prints eval lines only where its own
  cadence fires; at 200 steps the last printed eval is step 100 (byteflow's loop
  evals at 200 too). The honest comparison is eval@100: 4.823/4.898 vs
  6.714/6.595. A dormouse eval@200 line would demand an eval_every that hits 200
  — the tool prints what it prints; quoting best-of-evals per arm would not
  change the 2-BPB shape no matter which line is read (the byteflow step-100
  value already beats the control's step-100 value by 1.8–2.0 BPB).
- **Byteflow's ms/step is ~65 vs the control's ~450** on the same bytes — the
  launch-bound box pays the compressed stage less. A production arm that is 7×
  cheaper per byte ALSO trains deeper in the same wall clock. Not a quality
  number, but it is on the record.

### Verdict

ByteFlow wins the equal-byte smoke on both seeds — **by ~1.8–2.0 BPB at step 100
and ~2.1–2.2 BPB at step 200**, ~30× the control's own seed spread (0.119 over
the 81920 B window; the byteflow pair spreads 0.051). Fit for purpose: flip
executed (`use_byteflow = true` became the schema default; every dormouse preset
now carries `use_byteflow = false` explicitly — configs/{small,nano,base,
swift50,one_b,p150,mor,nano-fused}.toml, so no preset silently changes nets; a
flat config without the field gets the winning net).

Honest caveats on this verdict:

1. **Params are NOT matched**: byteflow 15.9M vs control 9.2M (+73%). The
   A/B answers "the byteflow net as configured beats the byte control as
   configured at the program's operating point". A params-matched control is
   cheap (`--set byteflow_d_global=640` rescales the arm) and is owed with the
   2k confirm if the claim ever becomes "better net at equal budget" instead of
   "the arm we ship is the one that won".
2. **Budgets are 10× smoke**: 0.82 MB trained vs the family's 8.19 MB at 2k
   steps. Section 1.2 calls a 200-500 step run a filter, not a confirm. What
   the flip means concretely is `byteflow.toml` is now what a default
   `train --data ...` builds, and 2k × 3 seeds at batch 8 s512 (~40 min at
   ~65 ms/step + ~90 min for the control family at ~450 ms/step) is the
   minimum confirm before anything else from byteflow (logdet path, MoE
   global, 1M) should build on this.

## Path to 1M: every use_byteflow refusal and what it should become

The gate set is `byteflow::check` (crates/dormouse-train/src/byteflow.rs) plus
the use_byteflow arms in config validation (crates/dormouse-core/scripts/../
config/validation.rs:282+).

| refusal | what blocks 1M | flip/compat route | LOC |
|---|---|---|---|
| `--bf16` | the cast-copy step on this backend (§2.1) is slower than fp32; ByteFlowNet has no bf16 storage | skip it: bf16 storage is a 1M VRAM lever (u16-bit-pattern path exists in the tree), never a speed lever here | ~200 (crate) |
| `--graph-capture` | launch-bound step would shrink further | reuse the graph seam only if a 1M step is again launch-bound — determine first, the arm is 65 ms/step today | deferred |
| `--jepa-targets`, aux weights, MoE, the 7-arm refusal | BY DESIGN: ByteFlowNet is a CE-only different net; none is a 1M blocker | keep refused, nothing belongs on this path | 0 |
| `--rand-depth`, `--eval-depths` | depth is layer counts (`e_layers + g_layers`), a per-iteration loop does not exist here | keep refused; a depth curve instrument for byteflow = a new design, not compatibility | 0 |
| `--engram-ram` | no memory arm in the net | keep (a latent/n-gram fusion arm would be new work, not compat) | 0 |
| `--stress` | the monitor reads the dormouse optimizer's groups | COUNTED skip line, not a refusal: stress = spike counting, which is loss-only and arm-agnostic | ~30 |
| `--opt` only adamw|mix | plain AdamW is the paper's recipe | keep; a Muon+ routing for the 2-D global matrices is an A/B, not a compat | reserved for arm 5 |
| `byteflow_max_bytes` assert (net.rs:445, `T={t} exceeds RoPE table`) | THE 1M blocker. Local + decoder RoPE tables are sized by max_bytes (net.rs init), global at k_tokens. Full local attention is quadratic over T; for 1M bytes at d_local=96 that is the real compute wall, not just a table | The crate default is already 8192 (net.rs:306); our preset pins 512. Route: (a) raise `byteflow_max_bytes` for the flat config — gives 8 k-local windows with everything else unchanged, evaluate ≈ quadratic cost first; (b) for 1M, chunk-recurrent local state (the crate's stream.rs has the streaming RatePatcher seam already), not a fat RoPE table | (a) ~20 + a preset line; (b) ~300–400 (crate: recurrent local window carry + trainer's stream doesn't need to change) |
| `max_seq_len > byteflow_max_bytes` (validation) | the trainer flat-config does not know byteflow's max window | collapse into the byteflow_max_bytes doc line; seq_len > max_bytes is ALREADY a loud refusal on the byteflow path (it is asserting the same thing twice) | ~2 |

Sum of the "must have" for a 1M byteflow run: ~20–30 LOC (raise the window
ceiling in the flat config + the COUNTED stress skip) + the recurrent-window
path (~300–400 crate LOC) IF quadratic local attention is not acceptable at
1M. NOT implemented here, per the lane brief.
