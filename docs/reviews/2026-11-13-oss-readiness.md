# OSS readiness audit — 2026-11-13

**Scope:** `/home/sehaxe/dormouse`, read-only (no file touched, no GPU job started).
**Snapshot:** local `main` = `220b2f4` (2026-11-13 12:33), **65 commits ahead of `origin/main` = `354520fb` (2026-11-08)**.
**Question:** «стремимся к хорошему опенсорсу, а не к пет-проекту» — where does it actually stand?

Scoring key: 5 = solid hobby repo, 7 = better than most hobby repos, 9 = better than most funded OSS.

## Scores

| # | dimension | score | the one thing that decides it |
|---|---|---:|---|
| 1 | Architecture | **5** | Live lane (`port/src`, 88 modules, 55 759 lines) is cleanly layered — but two preset trees, two READMEs that disagree, and a 100 KB rulebook at the root. |
| 2 | Code quality | **6** | 82 % fully typed (1 101/1 340), 0 bare `except`, 2 dead config fields of 266. But 41 functions >120 lines (`config.validate` = 1 259). |
| 3 | Tests | **6** | 2 594 tests, behaviourally named, anti-rot gates. 82 of them fail on GitHub because they pin exact hashes. |
| 4 | Documentation | **6** | 440 `.md`, a real provenance ledger, retractions kept — but 132 of 362 docs still describe the frozen Rust lane. |
| 5 | CI/CD | **3** | `main` on GitHub: 7 of 13 jobs red; 19 m 22 s; a contributor's first PR is red before they touch anything. |
| 6 | Repo hygiene | **5** | Commit messages are excellent; the working tree is not — 57 branches, 61 worktrees, 44 GB, 231 MB `.git`, 37 MB model blob in history. |
| 7 | Reproducibility | **5** | Recipe, data, seed, config are all named. The logs that prove every number are untracked; no weights published. |
| 8 | Licensing & compliance | **4** | MIT `LICENSE` is clean. The code slice has **no declared licence** and the README admits 40 % of it is copyleft. |
| 9 | README as landing page | **6** | Leads with a product sentence and a banner, then immediately becomes a lab notebook — and headlines the *wrong* (older) number. |
| 10 | Public site | **5** | Beautiful, product-first, generated from logs — 4 days and 0.29 BPB out of date, with two contradictory `xz` figures. |

**Mean 5.1.** The engineering discipline is a 7–8; the *packaging* of that discipline is a 4–5. Nothing here needs more science before Monday; it needs the front door rebuilt and the checks made to pass on a machine that is not this one.

---

## Evidence, dimension by dimension

### 1. Architecture — 5

What works: `port/src/dormouse/` has honest seams — `config.py` (schema + refusals), `instrument.py` (LOUD/COUNTED), `model/`, `train/`, `cli.py`. Presets are data (`load_preset`), the config is strict flat TOML, and a resume refuses a drifted snapshot. `docs/guides/presets.md` exists, and `dead_knob_audit.py` reports **264 LIVE / 2 DEAD of 266 fields** — flags are not folklore.

What breaks it:

- **Two preset trees.** Root `configs/` (16 files, Rust-era) and `port/configs/` (44, live). `docs/guides/presets.md` documents the *dead* one and cites `crates/dormouse-core/src/config/schema.rs` as authority. A newcomer reading the guide opens the wrong directory.
- **Two READMEs that describe different projects.** `port/README.md:1-10` calls dormouse "the byte-level **1B** framework … sparse MoE with hashed n-gram memory, and a weight-shared loop … down to 1.28 bits/weight". None of that is the live model; the root README says 95M dense vanilla because the fancy arms lost. A reader who starts in `port/` gets a model that has not existed for weeks.
- **`main` is not the only branch.** Root README: "One branch now." Reality: 57 local branches, 62 tags, 61 worktrees (`git branch | wc -l`, `git worktree list | wc -l`). Remote is clean (2 refs) — the mess is local — but `CONTRIBUTING.md:38` still tells contributors to open PRs against `backup/main-local-2026-11-08`, a branch that no longer exists.
- **Scale of the god functions.** `config.py` is 6 301 lines with `validate()` at 1 259 lines (:3797); `train/loop.py:1413` `_train_impl` is 1 123. The config module has absorbed every experiment's validation.
- **41 functions over 120 lines** (AST scan of `port/src`), 88 modules for a 95M dense transformer — `jepa.py`, `specdecode.py`, `selfplay/`, `nvfp4.py`, `mxfp.py` are all live code paths for arms that did not make the recipe. That is a research repo's normal state; it is not what a 5-minute newcomer tour survives.

*Can a newcomer find the training path in 5 minutes?* Partly: root README → Quick start is correct and runnable. But the same screen then says "Two lanes", links to five audits, and the second README contradicts the first. **5.**

### 2. Code quality — 6

Measured, not vibes (AST over `port/src`, 88 files / 55 759 lines / 1 340 functions):

| metric | value |
|---|---|
| fully annotated functions | 1 101 / 1 340 = **82 %** |
| functions > 120 lines | 41 (worst: `config.validate` 1 259, `_train_impl` 1 123, `DormouseModel.__init__` 560) |
| bare `except:` | **0** |
| `except Exception` | 16 — all at process/system boundaries (`conftest.gpu_free`, subprocess wrappers) |
| `TODO/FIXME/XXX/HACK` in `port/src` | **2** |
| dead config fields | **2 / 266** |

The good part is the error philosophy and it is real: `instrument.py`'s LOUD/COUNTED/SILENT taxonomy, and gates that enforce it (`tests/test_step_audit.py` fails any hot loop with no counter or no written excuse — it caught `set_act_store` in the working tree during this audit). `model/export.py` opens with a 40-line contract stating what the format does *not* buy ("this port has no int8/LUT ternary kernel … the file is a distribution/at-rest artifact"). That is funded-project-quality comment discipline.

The deductions: the 41 long functions are where the bugs the project itself found lived (config-vs-optimizer defaults); 6 301 lines in one config module is a monolith with a schema attached; and stale prose inside live code (`port/README.md` is documentation, but `pyproject.toml:46` still says "RTX **5060** Ti" as the measured card while the root README says 5070 Ti — the machine facts drifted between two files). **6.**

### 3. Tests — 6

Counted, not estimated: `pytest --collect-only` → **2 594 tests collected**, 199 `test_*.py`, 67 909 lines of tests against 55 759 of source. Full CPU suite: **734 s** on this box *while a training run occupied the GPU and the CPU* (documented idle figure: ~9 min, `port/AGENTS.md:24`).

Naming is behaviour-first, not `test_foo_2`: `test_every_step_loop_is_counted_or_excused`, `test_the_arm_refuses_a_non_cuda_device_up_front`, `test_a_non_cuda_device_is_refused_before_any_card_call`, `test_roundtrip_is_exact_enough_to_call_it_the_network`. That is better than most funded repos.

Coverage *shape* is where it splits:

- **Genuinely gated (good):** silent no-ops (`test_ternary` pins that `ternarize` is not idempotent; the export `_kind()` bug is now refused), eval-window overlap (`test_eval_overlap`), docmask, checkpoint/resume invariants, param count from the module tree, dead knobs, doc refs, bf16 numerics against an fp64 reference, and the step census.
- **Not gated by construction:** anything that needs the card. 36 `skipif(not torch.cuda.is_available())` files — every fused kernel (KDA, MoE, CSA) is skipped on CI, so "CPU-green ≠ GPU-tested" is a *documented* hole (`port/AGENTS.md:40`), not a closed one.
- **The weakest area, and it is the reason CI is red: cross-machine determinism.** 68 test files use exact SHA-256 pins (`PIN_LOGITS_SHA`, `PIN_STATE_SHA`, `_sha(logits) == pin_logits`). On `ubuntu-latest` the suite produced **82 FAILED**, all of the shape `assert '1342b518…' == '56c32d87…'` — `act_quant=none moved the logits`, `the KDA logits moved`, `logits moved off the 550bc43 pin`. The project already knows why: `port/AGENTS.md:36` — "26/28 hash-pinned byte-exact tests fail at `OMP_NUM_THREADS=4`". A gate that is only green at one thread count on one machine is a calendar, not a test.

So: the suite catches the bug classes this project actually hit (silent no-ops, off-path drift, config/optimizer disagreement) — but its portability property is unenforced, and that property is the thing a contributor meets first. **6** (would be 7.5 the day the pins become tolerance-based or opt-in).

### 4. Documentation — 6

Volume and honesty are the strengths, and they are unusual:

- **440 tracked `.md`**; `docs/research/` 94, `docs/reviews/` 144, `docs/adr/` 26, `docs/guides/` 18.
- A **claims ledger** (`docs/reviews/2026-11-07-claims-ledger.md`, 208 KB): 1 627 claims machine-extracted from 109 documents, each with `file:line`, unit, kind and status. It reports **12 DERIVED, 1 606 UNVERIFIABLE, 9 RETRACTED, 0 mismatches**. Publishing "1 606 unverifiable" about yourself is rarer than any score here.
- `docs/guides/data-provenance.md` is the best document in the repo: per-dataset HuggingFace revision sha, shard URL and byte size, the dataset's own licence sentence, the `sha256` of every produced `.bin`, and what a commercial release would still lack.
- Retractions are kept in place with dates — the README keeps `1.4621` marked retracted-with-reason next to the live number, and `docs/reviews/2026-11-12-ternary-artifact.md:26` says "two of my claims were wrong" and names them.

**Is "every retraction kept" an asset or a liability for a stranger?** Asset — when it is scoped. `AGENTS.md` (root) opens with a *Mission* paragraph describing "byte-level LM trainer in Rust (burn 0.22.0-pre.4 + cubecl CUDA) … weight-shared loop block … PonderNet-style adaptive depth" — the frozen lane, presented as the mission. A stranger's first 200 words in the repo are about code that is frozen. That is the cost of keeping everything: no document says "these five documents are current, everything else is a dated note."

Deductions: **132 of 362 `docs/*.md` mention `crates/`**, 9 of 18 guides cite `crates/` as the authority; `docs/README.md` explains the *naming convention* but never gives a reading order; and there is no single "start here → this config → this log → this number" chain. **6.**

### 5. CI/CD — 3

`gh run list` on the pushed `main` (`354520fb`), run 42480254891, **19 m 22 s**:

| job | result |
|---|---|
| cpu (lib tests), cuda-compile, claims-ledger, unused deps, coverage | success |
| **python-test** | **failure — 82 tests** |
| **python-lint** | **failure — `Found 1 error.`** |
| **python-gates** | **failure** |
| **site** | **failure — cannot link** |
| **core :: rustfmt + clippy** | **failure** |
| **docs :: rustdoc** | **failure** |
| **ci :: actionlint** | **failure** |
| **supply chain** | **failure** |

The separate `fused-library` run (42480254932) fails 3 of 6 jobs.

Three distinct root causes, all cheap:

1. **`site` cannot link.** `site/gen` is not a workspace member, but it inherits `../.cargo/config.toml:53` → `rustflags = ["-C","link-arg=-fuse-ld=mold"]`; the `site` job never installs mold. Log: `collect2: fatal error: cannot find 'ld'`. The `cpu` job installs mold and passes — so the fix is one missing `apt-get install -y mold`.
2. **`python-test`'s 82 failures** are the hash pins above, not regressions.
3. **`python-lint`, `python-gates` and the docs gates are already fixed locally, unpushed.** `python-lint`'s `Found 1 error.` → `f78929b` ("ruff check is green on the tracked tree"); `python-gates`' `NAMED PATHS THAT DO NOT EXIST: 4` → `5f65c9f` ("doc-refs resolves every path its code cites", re-verified in this audit: `3644 path references resolved, 0 missing`). The 10 remaining ruff findings are all in untracked scratch scripts (`scripts/bench_activation_vram.py`, `opt_vram_probe.py`, `bench_loop_econ.py`), so the tracked tree is lint-clean.

**What a contributor gets on a PR:** an outstanding *contract* (`.github/PULL_REQUEST_TEMPLATE.md` demands the root cause, the red line from the un-fixed code, the artifact behind every number, and "no test deleted, skipped or loosened") wrapped around **7 red checks they did not cause**. That combination reads as a repo where the rules are for other people. The workflow file itself is thoughtful — `permissions: contents: read` on a job that runs repo code, a `claims-ledger` job that is `continue-on-error` *with the reason written above it*, actionlint installed with a comment explaining why an absent shellcheck is a silent pass. Craft is high; the outcome is red. **3.**

### 6. Repo hygiene — 5

**Commit messages: 20 of 20 sampled are better than most funded repos.** Conventional prefix + the specific claim: `docs(longctx): KDA vs CSA at 8k — the 2.2x was measured against the wrong baseline`; `docs(ternary): the artifact costs 2.94% for 2.31x — and two of my claims were wrong`; `perf(optim): one batched matmul per shape group — NS stacks the list, bmm's it`. A reader can reconstruct intent from `git log` alone.

Everything around them is loose:

| item | number |
|---|---|
| local branches / tags / worktrees | 57 / 62 / 61 |
| working directory | **44 GB** (`target/`, `wt/`, `docs-site/node_modules`, `port/.venv`, checkpoints) |
| `.git` | 231 MB |
| largest blobs ever committed | 37 MB `docs-site/public/demo/model-engram-off.dormouse`, 16 MB `model.dormouse`, 14 MB `graphify-out/graph.json`, 10.5 MB `ref_f64_broad.bin` **twice** (two vendor forks) |
| tracked PDFs | 42 (~40 MB) |
| tracked `.webm` / `.jpg` / `.png` | 4 / 5 / 12 (site media — defensible, they are the site) |
| tracked agent state | `.bulba/` — `plan.md`, `memory.md`, `sessions/`, `research/` |
| secrets | none in tracked files (only `.venv/site-packages` hits) |

`.gitignore` is thorough and *explains itself* (`a tracked file is unaffected by .gitignore, so the rule below only bites after git rm --cached`). The problem is the history is already dirty: two 37 MB/16 MB model files, the whole `graphify-out/` graph, and duplicated vendor fixtures. `git gc` will not fix that; the repo needs to stay as-is or be re-rooted, and either way it is 231 MB for a 55 k-line Python project.

`docs-site/` (481 MB, 348 committed content copies that `ingest.mjs` regenerates) is a second, hand-synced documentation tree inside the same repo — the drift vector is built in. **5.**

### 7. Reproducibility — 5

What a stranger *does* get:

- The recipe, byte for byte: `docs/reviews/2026-11-09-2gb-run.md:1-12` names config, params, `seq_len`, `batch`, `lr`, `opt=mix`, `muon_lr_mult=12`, `torch_compile=true`, `mix_plan`, `doc_masking=false`, **seed 1**, eval window and offset.
- The corpus, scripted: `port/scripts/refetch_pretrain.py`, `fetch_mix_sources.py`, `carve_eval_tails.py`, plus per-file `sha256` in `docs/guides/data-provenance.md`.
- The resolved config, saved with every checkpoint (`<ckpt>.config.toml`), and `dormouse info` recomputes the parameter count from the module tree.

What is missing, and it is the same thing every time:

1. **The logs are not in the repo.** `port/logs/runs/` holds **766 run directories / 6.7 MB** — untracked and unignored. Every published BPB is derived from one of them. The claims ledger's own number is the verdict: **1 606 of 1 627 claims UNVERIFIABLE**.
2. **No published weights.** `checkpoints/run2gb/port.pt` exists only here; `gh release list` is empty. The ternary artifact (82.4 MB) is a paragraph in a review, not a downloadable file.
3. **The eval window is not published.** `data_local/best/fwedu_eval.bin` is gitignored; `sha256 003ba82d…` is the only handle, and the README itself warns that the window that produced `1.4946` **died on 2026-11-08**.
4. **The landing-page numbers cannot be regenerated.** `site/README.md` invokes the generator with `--logs /home/sehaxe/logs`; root `logs/` does not exist in the checkout. There is no path from repo → page.

Verdict: **the recipe reproduces within ~0.01 BPB; the numbers do not reproduce as artifacts.** That is the honest gap between "documented" and "reproducible". **5.**

### 8. Licensing & compliance — 4

- `LICENSE` is stock MIT, © 2026 sehaxe. Fine.
- **The code dataset has no declared licence.** `docs/guides/data-provenance.md:29`: `codeparrot/codeparrot-clean`, licence column = **"none declared"**, weight 0.11. The root README already flags the consequence: "the code slice's licence is unresolved (40 % of it is copyleft)".
- No `NOTICE`, no `third-party.md`, no attribution for the vendored forks (`vendor/dormouse-fused/`, `vendor/burn-fused/`) — which are patched copies of other people's code, and `CODEOWNERS` even explains they need deliberate review. Their licences are not surfaced anywhere a reader would look.
- Dependency licences *are* gated (`deny.toml`, a `supply chain` job) — and that job is **red** on `main`.
- No `CITATION.cff`; for a project whose entire selling point is "cite the measurement", there is no way to cite the project.
- The dataset ledger is genuinely good lawyering-adjacent work (ODC-By, CommonCrawl ToU, PII form, "not legal advice", "the questions a lawyer must answer"). The gap is the one file with no licence at all, sitting in the published mixture.

**4** — the process is above average, the published artefact carries an unresolved-copyleft component and no attribution file. A university mentor will ask this question.

### 9. README as a landing page — 6

The opening is already the right *shape*:

> **A byte-level language model that trains on one workstation — with every claim measured, and every retraction kept.**

Then: banner, 5 badges, and inside the first screen a paragraph about lanes and a force-push. The Results table — the reason anyone is here — comes after "What is this", and it headlines:

> `flagship3` 1.5185 … **`e1v4` bytes arm 1.4946** @ 250.0 MB

while `docs/reviews/2026-11-12-ternary-artifact.md:7` records **1.1871 at 2.64 GB** (and 1.1940 at 2.0 GB before it). The README's "Status, honestly" repeats the stale claim: "The best clean number is **1.4946 BPB at 250 MB**". A stranger reads a 0.31 BPB weaker project than exists, *and* the repo's own docs contradict the repo's own front page.

What it does right: the retraction block is explicit ("its window overlapped training bytes"), it states the seed noise floor (`1σ ≈ 0.011 BPB`), it names what is *not* proven ("a 1M forward has not been run"), and it points at the audits. That honesty is the differentiator; it just is not *led* with.

Would a stranger star it? Only if they read past the second screen. The first screen sells a policy, not a result. **6.**

### 10. Public site (`site/index.html`) — 5

Craft: high. Self-contained 53 KB + 3 WebM, two themes, halftone hero, generated numbers from logs by `site/gen` ("If a log is missing, the row is missing — there is no default and no placeholder"). Tone is product-first and non-defensive — section 03 says of TSCT: "It is an experimental arm: at equal bytes a dense layer still wins, so the live model is dense and the bank stays in the queue." That "measured and rejected" content **helps** a stranger: it is the single most credible paragraph on the page.

Accuracy, three findings:

1. **The headline is 4 days and 0.29 BPB stale.** Section 08: "1.74 at the start, **1.48 after 120 MB**". The record is 1.1871. Because `site/README.md` says "No CI gate on freshness: the logs live outside the repository … Regenerate after a run and commit the HTML", the page silently rots and nothing fails.
2. **The `xz` baseline contradicts the review.** Site: "A strong general compressor needs **2.71** bits for the same bytes … on the same 196 KB." `docs/reviews/2026-11-09-2gb-run.md:44`: `xz -9` = **2.3134** on `fwedu_eval.bin`. Both are "196 KB, offset 0" — of two different windows, and neither labels which. This is exactly the failure mode `docs/protocols/AB-PROTOCOL.md` was written to kill ("a BPB is only comparable within one window"), reproduced on the marketing page.
3. **The generator is not reproducible from the repo.** `--logs /home/sehaxe/logs` is a path on one machine.

**5** — the best-looking artefact in the repo and the one with the weakest claim chain.

---

## The 10 things to fix first (ranked by impact)

| # | action | size |
|---|---|---|
| 1 | **Push the 65 commits.** `origin/main` is 5 days and every headline result behind local `main`; a mentor clicking GitHub on Monday sees the old project. Then confirm `ci` is green on the *pushed* HEAD. | **S** |
| 2 | **Fix the `site` job**: add `apt-get install -y mold` to it (the `cpu` job already does) — or scope `rustflags` to workspace members only. Currently `site/gen` inherits `.cargo/config.toml:53` and dies at link. | **S** |
| 3 | **Make the hash pins portable**: move `PIN_LOGITS_SHA`-class tests behind `-m pins` (opt-in, owner's box) or compare with an explicit tolerance; keep them as a nightly/release job. That single change takes `python-test` from 82 failures to green on any runner. | **M** |
| 4 | **Rewrite the README's Results block** around 1.1871 @ 2.64 GB, with 1.1940, 1.4946 and the retracted 1.4621 below it as dated stages. Delete "the best clean number is 1.4946". | **S** |
| 5 | **Regenerate `site/index.html`** from the 2.64 GB logs, label the window next to *every* number including the `xz` baseline, and fix the two contradictory `xz` values (2.71 vs 2.3134). | **S** |
| 6 | **One front door**: delete or rewrite `port/README.md` (it describes a 1B MoE loop model that does not exist) and fix `CONTRIBUTING.md:38`, which sends PRs to a deleted branch. Keep exactly one "how to run it" page. | **S** |
| 7 | **Publish the evidence**: commit a pruned `logs/` set (or the JSONL for the three record runs) and attach the 82.4 MB ternary file + the 190 MB bf16 checkpoint to a GitHub Release. This is the only step that moves the ledger off 1 606 UNVERIFIABLE. | **M** |
| 8 | **Retire the Rust-era surface**: move root `configs/` and the 9 guides that cite `crates/` into `docs/archive/`, and make `docs/README.md` a 5-document reading order instead of a naming table. | **M** |
| 9 | **Close the licence hole**: add `NOTICE`/`third-party.md` (vendored forks + dataset licences), either drop the `codeparrot` (no-declared-licence) slice from the published recipe or state the exposure in the model card, add `CITATION.cff`. | **M** |
| 10 | **Split `config.validate` (1 259 lines) and `_train_impl` (1 123)** into per-mechanism validators/step phases. Lowest urgency, highest structural debt — and the two places the project's own audits found bugs. | **L** |

## Three things already better than typical

1. **Claim discipline made mechanical, not cultural.** `scripts/claims_ledger.py` (1 627 claims, `file:line`, DERIVED/UNVERIFIABLE/RETRACTED), `dead_knob_audit.py` (2 dead of 266 fields), `tools/check_doc_refs.py` (3 644 references, 0 missing), and `stepaudit` (a test that fails on any hot loop with no counter or no written excuse). Most funded projects have a `CONTRIBUTING.md` asking for this; this one has executables that check it.
2. **`docs/guides/data-provenance.md`.** Per-dataset HF revision sha, resolved shard URL and byte size, the licence sentence quoted from the card, `sha256` of every produced file, licence gaps named, and "not legal advice" with the lawyer questions listed. I have not seen this in hobby repos and rarely in funded ones.
3. **The review contract.** `PULL_REQUEST_TEMPLATE.md` (root cause, the red line from un-fixed code, artifact per number, "no test deleted/skipped/loosened", "every new degradation is LOUD or COUNTED"), four issue templates, and `CODEOWNERS` that explains why vendored code still needs a human. Plus commit messages a reader can reconstruct intent from — 20 of 20 sampled were specific and honest.

## The single sentence a stranger should read first

> **dormouse is a byte-level language model trained from scratch on one 16 GB GPU — 95M parameters, no tokenizer, 1.19 bits per byte on held-out English — and every number in this repo names the run, config and log that produced it, including the numbers that turned out wrong.**

*(What changed: the number moved from 1.4946 to the actual record, the tokenizer-free claim stayed, the honesty claim stayed but got attached to an artefact instead of to a policy, and the lane/force-push paragraph moves to the bottom.)*
