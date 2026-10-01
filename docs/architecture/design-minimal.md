# design-minimal: the floor of dormouse (2026-09-25)

Question: cut 90% of the code, keep all functionality and performance. Answer with
numbers, not adjectives. Baseline: 15,513 LOC across 4 crates (fused-grad-coverage
@ 9605896). Verdicts: docs/archive/audit-2026-09-25.md.

## 1. The honest math

| Scenario | LOC | Δ | Functionality |
|---|---|---|---|
| today | 15,513 | — | everything |
| A: fused/ dies at its gate (doctrine already measured it losing 1.25x) | ~7,650 | **−51%** | zero loss — fused is flag-gated OFF and lost its confirm A/B |
| B: A + knives the audit pre-justifies (gr.rs, KoLeo, dead config fields, scratch examples, dead pad) | ~7,300 | **−53%** | zero loss — each dies on a written verdict or missing-verdict rule |
| C: B + pending knives all die at their A/Bs (aux heads vs pure CE, act_quant with fused gone, MSA at s512) | ~6,000 | **−61%** | zero loss IF the A/Bs go that way; each has a scheduled run |
| D: "90%" = 1,550 LOC | 1,550 | −90% | **dies**: Muon+ (−476), Engram RAM-offload (−376, the core mission), PonderNet loop (core arch), presets, the drift check, the filter (−1,427), the test harnesses (−3,100) |

Scenario D is not a refactor, it is a different, weaker product. The floor of
"byte-level LM trainer, Muon+, RAM-offload, loud failures, verified" is
**~5.5–6k LOC**. That is the genius design: every remaining line held by a
verdict, every mechanism behind a deep module. 90% is a number that kills the
mission; ~60% is the number that fulfills it.

## 2. Target shape (after Scenario C)

```
dormouse-core (~2.6k)   model.rs, loop_block.rs, attention.rs, param.rs,
                        config/ (dead fields gone), lib.rs seam
                        [act_quant.rs only if fused/ survives its gate]
dormouse-data (~1.7k)   lib.rs (stream, loud failures, floor) + bin/filter.rs
                        (two-region DCLM; the tool that builds the corpus)
dormouse-train (~2.2k)  lib.rs split (P10 R4: snapshot/warmup/step/eval/ckpt,
                        584-line body → ≤5 files x ≤120 lines), optim.rs,
                        offload.rs (optimizer renamed or reverted to Adam),
                        stress.rs, cfg.rs
dormouse-cli (~0.5k)    train, generate, serve
tests                   model_seam.rs, ckpt_roundtrip.rs, parametrized
                        fp32/bf16 x CPU/CUDA suite — the only test surface
```

Deep-module discipline (codebase-design vocabulary): the interface each caller
must learn stays small — `ByteStream::new(root)`, `DormouseModel::forward_with_hidden`,
`train_loop(cfg)`, preset TOML → typed cfg. Complexity lives behind the seam,
testable through it. One adapter per seam (no speculative generics).

## 3. Sequencing — cuts happen at verdict moments, never before

1. **pre.4 merge** (debugger in flight) → fusion backend becomes testable.
2. **The fused gate** — the single highest-leverage act in the repo: wire the
   already-written direct adjoints (kernels.rs:1317/1376/1441, zero callers
   today), run the 50-step flagship A/B. Win → fused earns ~4.9k LOC, delete the
   generic adjoint path (−2.5k) and act_quant's burn-path twin. Lose → delete
   all 7.4k, pre-justified by the measurement that already exists.
3. **The bench gate goes live first** — `benches/history.tsv` is empty; the
   doctrine's instrument never fired. Wire bench.sh into the merge flow; every
   verdict from here on lands as a TSV row, not prose.
4. **Knife A/Bs in one session** (config-only arms on the official fp32 run):
   GR vs ReZero, aux vs pure-CE (+KoLeo third arm), then delete per outcome.
5. **Mechanical cuts now** (no A/B needed, verdicts already written): scratch
   examples (−261), dead config fields (−70), dead data pad (−5),
   max_iter default 8→4 (executing a recorded verdict), silent-skip → assert
   (lib.rs:875), sidecar-mismatch → hard error (offload.rs:187).

## 4. What "90%" would actually cost

Named functionality that dies on the road from 6k to 1.5k: Muon+ ColRow (the
optimizer ADR-0006 exists for), host-RAM n-gram offload (the Qwen-playbook core:
billions of params without VRAM), the PonderNet adaptive-depth loop (the
architecture), ADR-0005 checkpoint integrity (saved real data three times), the
corpus filter (46 GB → 19.7 GB), and all local verification. Each of those is a
"keep" with a verdict. The genius design is not fewer principles — it is zero
unprincipled lines.

## 5. Perf budget stays the judge

BPB on the held-out tail at fixed steps, ≤1.2x step time, memory capped. Every
cut above is justified by a verdict; every keep must keep winning. When the
official fp32 run lands on corpus v2, history.tsv starts filling and the loop
closes: verdict → cut → measure → verdict.
