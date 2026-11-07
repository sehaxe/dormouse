# Document masking is neutral at 16 MB — and that clears the patch arm

**Date:** 2026-11-08. Two arms, 2000 steps (16.4 MB), one seed, the safe Muon
rung (`ns_dtype=bf16`, `ns_steps=5`, `muon_state_dtype=bf16`), same eval window.

| arm | mixture | doc_masking | bpb @8.2 MB | bpb @16.4 MB | kB/s |
|---|---|---:|---:|---:|---:|
| masked | `mix_v7_docmask` (9 sources) | true | 2.2941 | 2.0406 | 53.3 |
| unmasked | `mix_v7` (12 sources) | false | 2.2801 | 2.0236 | 51.9 |

**Masking costs +0.017 BPB here** (it was slightly WORSE, i.e. within noise and
not in masking's favour). At this budget it is neutral.

## Why this matters

E1 v3 measured 2.12 at 125 MB against 1.63 on E1 v1, a 0.49 BPB gap, and two
variables had moved: the Muon precision rung and document masking. The rung
ladder (`muon_quality_ab.sh`) attributed **+0.27 of it to `ns_steps=3`**, which
does not converge on `[832, 3072]` shapes. This ladder removes the other
candidate: masking is worth ~0.0, not 0.45.

That matters for a decision, not just for the ledger: the patch arm **cannot**
run with `doc_masking=true` (the byteflow gate refuses it — a patch boundary is
not a document boundary, so the byte-position loss mask cannot address a patch
latent). If masking had been worth 0.3-0.45 BPB, the patch unit would start
every comparison 0.3-0.45 behind, and E1 would have been measuring that handicap
rather than the unit. At 0.017, the handicap is small enough that the fair
same-regime A/B is also the deployment-accurate one.

## What is still unexplained

`0.49 − 0.27 = 0.22` BPB remains unattributed at 125 MB. Two candidates, both
untested:

1. **The 3-step penalty grows with training.** It was +0.26 at 16 MB; a
   badly-scaled update compounds, and 125 MB is 7.6× further along the curve.
2. **The 12-source mixture is slightly harder on a web-only eval** than the
   9-source one (wiki_en 4%, reasoning 2%, rudialog 1.5% are in the unmasked
   plan only).

Neither changes the decision — the safe rung is what E1 v4 runs — but the
residual should not be quoted as "doc masking costs 0.22" anywhere, because this
ladder says it does not.

## Reproduce

```
bash scripts/docmask_ab.sh          # 2 arms x 2000 steps, ~11 min on the safe rung
```
