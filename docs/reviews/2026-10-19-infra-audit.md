# Infrastructure and process audit — 2026-10-19

**Scope.** The tree, CI, repo hygiene, reproducibility and process discipline of
`main` (`7f0b656`, 2026-10-19 17:45 +0300) — not the model code (a separate
reviewer holds that). Canonical checkout `wt/pyrestructure`; the live training
checkout `port` (`4fa81fb`, `main` **behind 11 commits**) was read as evidence and
not modified. Every finding names the command or the line that shows it.

Method: diffed the two trees, ran the gates (`dormouse train --help`,
`pytest`, `npm run check`, `tools/check_doc_refs.py`), read the last 10 CI runs
with `gh run view --log-failed`, and probed the two claims that decide
reproducibility: the live run's recipe, and whether the suite can be green next
to a training job.

---

## CRITICAL

### C1. The live run's recipe is not in the repository

`port` runs `dormouse train small_wide …` (pid 990522, started 16:37); its only
non-`--set` input is a preset git does not track:

```
$ git -C port status --porcelain
?? port/configs/small_wide.toml
?? port/scripts/ab_qat.py
$ ls /home/sehaxe/dormouse/wt/pyrestructure/configs/
b1.toml  medium.toml  mix_big.toml  small.toml  tiny.toml      # no small_wide.toml
$ git log --all --oneline -- configs/small_wide.toml           # empty, every branch
```

The command line lives on tmpfs (`/tmp/opencode/launch_sw1.sh`) — the exact
failure mode AGENTS §2.4 item 5 records ("`/tmp` is a 32 GB tmpfs: it dies on
reboot"). The JSONL sink does **not** carry the resolved config; its first row is
the whole provenance: `{"event": "run", "source": "small_wide", "steps": 300000,
"batch": 32, "seq_len": 512, "lr": 0.001, "opt": "mix", "resumed": 0}` — no
`out_dir`, no `ckpt_name`, no `mix_plan`, no flag list. The console log
(`~/logs/sw1_long.log:1-4`) names the preset, the param count and the `mix_plan`;
the checkpoint's `.config.toml` snapshot lands only at step 10 000
(`--set ckpt_every=10000`; the run was at step 7 000 when read) in an ignored
directory. A running 300k-step job is therefore recoverable from nothing that
survives a reboot except an untracked TOML and a log directory.

**Smallest correct fix:** commit the TOML as `small_wide.toml` under the tracked
`configs/` directory and the launch script beside it under a tracked `scripts/`
subdirectory. Two files, ~3.5 KB.

### C2. The README's quick start does not parse

`README.md:57-60` and `README.ru.md:57-60` — the project's first command:

```
$ ./.venv/bin/dormouse train medium --data corpus.bin --eval held.bin --device cuda \
    --steps 60000 --batch 24 --set seq_len=1024 --set engram_ram=true \
    --set moe_capacity=0 --set moe_kernel=triton --ckpt-name mix1 --out-dir checkpoints/mix1
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ No such option: --ckpt-name                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
EXIT=2
```

`dormouse train --help` lists `--data --eval --batch --steps --lr --device
--resume --json-log --set` and nothing else (`src/dormouse/cli.py:223-244`); the
working spelling is `--set ckpt_name=… --set out_dir=…`, which is what the live
run uses and what **no document shows** (`grep -rn "set ckpt_name" docs/
README.md README.ru.md docs-site/` → no hits). `bench`, `sft` and `serve` do take
`--ckpt-name`/`--out-dir`, so four of the five quick-start lines read as correct
and the fifth cannot run.

### C3. Data and artifacts are outside both checkouts, so "clone → train → measure" is false

- The live run reads `/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin`
  (19.8 GB) and `…/eval_mix/eval_mix.bin` — the **parent** of both checkouts. The
  port checkout's `data_local/` holds only `bench/`; the canonical root has no
  such directory and no `checkpoints/` at all.
- The headline recipe commits the author's paths as a fixture —
  `tests/fixtures/snapshot_configs/mix1.config.toml:97-98`: `data =
  "/home/sehaxe/dormouse/data_local/corpus_bin_filtered.bin"`, `eval_data =
  "…/eval_mix/eval_mix.bin"`.
- The mixture points at a UUID drive: `configs/mix_big.toml:79-82` (`web|wiki|
  ruweb|math = "/mnt/e43497ab-…/aria_data/pretrain/mix/*"`).
- `eval_mix.bin` is buildable from nothing in the tree: `grep -rn eval_mix`
  returns that fixture line and no script, doc or download path.
- No checkpoint ships except the 16.3 MB demo blob, so `bench`/`sft`/`serve` have
  nothing to load; the bench step also needs `--download` first (`data_local/bench`
  is the default `--data-dir`) and the README never says so.

**Smallest set that makes the claim true:** (1) fix C2's flags; (2) ship a small
corpus + eval pair (or a fetch script that builds them) and point the quick start
at it; (3) commit the `small_wide.toml` preset; (4) document `bench --download`;
(5) say in the README that the headline numbers are *this box's*, mount included.

### C4. The front page asserts the depth verdict the repo retracted

`README.md:24`: "three passes through the same block per token (**the measured
optimum**; four was a tie, two was worse)"; `README.md:43` lists "depth-3
(−0.007)" among the mechanisms "admitted by a 3-seed A/B". The tree says the
opposite three times over:

```
$ grep -n max_iter configs/small.toml configs/medium.toml configs/b1.toml
configs/small.toml:49:max_iter = 4
configs/medium.toml:24:max_iter = 4
configs/b1.toml:14:max_iter = 4  # … the fin-metric re-run (2026-10-16) shows training
  at 4 gives the best readout at EVERY eval depth … the earlier '3 wins' was a metric
  artifact, corrected in the plan
$ sed -n '478p' docs/decisions/2026-10-15-1b-readiness-plan.md
  # "the DEPTH verdict, CORRECTED by the fin metric (2026-10-16) … training depth pays"
```

The live run's own eval line agrees with the configs, not the README:
`~/logs/sw1_long.log` step 2500 →
`depths=1:2.0222(fin),2:1.8616(fin),3:1.8314(fin),4:1.8233(fin)`. This is the one
place a reader is actively misled rather than merely unsupported.

---

## MAJOR

### M1. `benches/history.tsv` was deleted from `main` and is cited ~87 times

```
$ git show --stat ba3f8ea | grep benches
 benches/determinism.tsv  | 6 -
 benches/history.tsv      | 171 -
$ grep -rn "benches/history.tsv" docs/ AGENTS.md README.md | wc -l     # 87
$ git cat-file -e rust-legacy:benches/history.tsv && echo exists       # exists there
```

It backs rows that carry public claims: the readiness plan's ledger (lines 20,
24, the ByteFlow trio, the 5-gram anchors), `AGENTS.md` §3.1 (the bench gate row,
the step-0 retraction), `docs/guides/train-your-first.md`. The doc-refs gate
cannot catch it — `tools/check_doc_refs.py:60` skips `benches/` wholesale
(`LEGACY_PREFIXES`), defensible for a *dated* document and wrong for a live one
that says "the numbers below are the state of 2026-10-19" (`README.md:31`).

### M2. AGENTS.md §3 does not carry the current lane's retractions

`grep -c` in `AGENTS.md`: `b1night` 0, `medbig2` 0, `sft9` 0, `invalidate_banks`
0, "version freeze" 0. The CUDA-graph `_version` defect that froze the expert
bank for whole runs — the finding that invalidates three checkpoints — is
recorded only in `.bulba/memory.md` (agent scratch, published as a site page but
not the rulebook) and `docs/reviews/2026-10-17-port-night-report.md`. §3.2's list
is entirely Rust-lane, and the README's `mix1` row does not say whether that run
is pre- or post-fix; `.bulba/memory.md` says the re-run uses the fix, which is a
statement that belongs in §3.2 with its commit.

### M3. The README's A/B paragraph is not backed by anything in the tree

`README.md:43` claims "Every mechanism in the tree was admitted by a 3-seed A/B on
a fixed protocol" and lists rejected arms with numbers (NorMuon +0.072,
schedule-free Muon +0.043, SIGReg +0.148, ternary-from-scratch +0.04). Greps for
those numbers attached to those mechanisms return nothing in `docs/` or
`research/`. `AGENTS.md:740` says the opposite of the headline sentence: "**Every
A/B verdict in `docs/protocols/AB-PROTOCOL.md`.** Not one arm has been judged."
The plan has the ColRow figure (−0.0428, consistent) and calls the 2026-10-16 aux
verdict "the port's first completed 3-seed verdict". One sentence is doing the
work of a results section, and both of its halves are unsupported.

### M4. The local suite cannot be green while training runs

```
$ ./.venv/bin/python -c "import torch; p=torch.nn.Parameter(torch.zeros(4));
    torch.optim.AdamW([p], lr=1e-3).step()"
RAISED: AcceleratorError CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

Any test that steps an optimizer fails for a reason unrelated to the code under
test while the card is held (`.bulba/memory.md` 2026-10-18 records 104 such
failures here, and the `pgrep -x train` gate cannot see the Python trainer). So
"full tests green" is verifiable only on CI, and CI has no GPU: `ci.yml` runs
`CUDA_VISIBLE_DEVICES="" uv run pytest`, the runner reports **69 skipped**, and
four more self-skip under `CI` via `tests/conftest.py:25` (`local_pin`: float-hash
pins in `mm_fp8`, `muon_variants`, `ns_triton`, `opt_graph`). Every CUDA gate —
determinism harness, bank-invalidation probe, KDA gradient gate — runs only when
a human remembers, on a free card.

### M5. The committed demo artifact is unverified and its own README denies it exists

`docs-site/public/demo/README.md:1,10` says the artifact is "**GENERATED, not
committed**" and "**not in the repository**" — but it is committed
(`git ls-files -s docs-site/public/demo/model.dormouse` → blob `a4173227`,
16 265 176 B, commit `697f92a`), and `.gitignore:36` carries the ignore rule
commented out with the reason. No gate loads it: the parity test exports a fresh
artifact into `tmp_path` and injects it (`tests/test_demo_engine.py:76-100`,
`DM_PARITY_ARTIFACT`), and the only assertion touching the shipped blob is that
the README mentions it (`tests/test_demo_engine.py:169-172`). Drift between
`engine.js` and the 16 MB payload ships green and breaks the live page at
runtime — the deploy gate cannot see it.

### M6. The parity gate's node version is unpinned

The last two red runs were both `ci`, both this defect: the gate grepped one
reporter spelling, and the runner's node (v20, TAP) escapes the leading hash where
the box's node (v26, spec) does not —

```
$ gh run view 39266194267 --log-failed | tail -3
FAILED tests/test_demo_engine.py::test_engine_matches_python_on_the_tiny_fixture - StopIteration
FAILED tests/test_demo_engine.py::test_engine_matches_python_on_the_small_polygon - StopIteration
2 failed, 811 passed, 69 skipped in 89.79s
$ git show -s --format=%B 50ef56d   # "node's default test reporter is the spec one from
  v22 … TAP before that (the runner's node)"
```

`ci.yml` installs no node at all — the gate runs on whatever the runner ships —
while `docs-site.yml:41-43` pins `node-version: 22`. The fix normalises both
spellings and attaches the engine's output to the failure (good), but the engine
still runs on an unpinned V8 in one workflow and a pinned one in the other.

### M7. The plan and the contributor guide describe the lane that left

`docs/decisions/2026-10-15-1b-readiness-plan.md` is four days old and its
evidence column cites `benches/history.tsv`, `.rs` paths, cargo aliases and the
mount — none on `main`. `CONTRIBUTING.md`'s "which command for which question"
table is entirely cargo (`cargo check-train`, `cargo test-core`,
`cargo build-probe`) and `.cargo/config.toml` is absent, so the aliases resolve
nowhere; the Rust block is labelled history in prose and then used as the primary
table anyway. Nothing in the tree states the status of the *current* run (sw1,
~7 000 of 300 000 steps, loss ~1.93, on the fixed bank-invalidation path).

---

## MINOR

- **m1.** `ci.yml:11` says "the same 861 tests"; the runner reports 882 (811 + 69
  + 2). Commit `7e62ef0` updated the README only.
- **m2.** 8 dependabot PRs open, none merged (six action bumps incl.
  `checkout` 4→7, three docs-site deps incl. **mermaid 11→12**, a major). The
  scoping rationale is written into `.github/dependabot.yml`; the queue is not
  being worked.
- **m3.** No tags and no releases (`git tag` empty, `gh release list` empty);
  `dist/` holds a locally built `dormouse-0.0.1` wheel nobody can install from a
  release.
- **m4.** No `CODE_OF_CONDUCT`. Issue templates (2), PR template and `CODEOWNERS`
  are present and specific. `SECURITY.md:17` scopes "the vendored library under
  `vendor/`", which now lives on `rust-legacy`.
- **m5.** `.editorconfig:6` says "828 tracked files" (measured 2026-10-01);
  `git ls-files | wc -l` is 527.
- **m6.** `README.md:27` says `small` is 9.5M; the measured count is 9 197 390
  (9.2M in the plan and `AGENT_CONTEXT.md`).
- **m7.** The docs check validates internal links only —
  `docs-site/tools/check.mjs:69` skips `https?:|mailto:|data:|#` — so an external
  404 deploys green. The internal half is real and green: `npm run check` reports
  **230 html files, 53 876 internal links, 0 broken, mermaid 5 rendered / 0 left
  as code**, and it runs before `upload-pages-artifact`.
- **m8.** The public site publishes agent scratch verbatim (`.bulba/memory.md`,
  `plan.md`, `architecture.md`, `goal.md`; `docs-site/tools/manifest.mjs:243-246`)
  and `docs-site.yml` triggers on `.bulba/**`, so an unreviewed agent note change
  redeploys the public site.
- **m9.** "the 8 MB commit rule" is cited in `.gitignore:34` and defined nowhere.

---

## What is genuinely good

- **The retraction culture is real and specific.** `AGENTS.md` §3.2 names each
  dead number with the commit that killed it, separates "the arm is retracted"
  from "the history is retracted", and states the rule that makes it work: "a
  fallback is not excused by having a defensible meaning, it is excused by the
  caller being able to tell the reader which arm ran". 69 files in
  `docs/reviews/`, including audits that found the project's own bugs.
- **Every checkpoint's recipe is a test fixture.**
  `tests/fixtures/snapshot_configs/` keeps 21+ real run snapshots and
  `tests/test_config.py:87-105` walks them, so a schema change cannot orphan them.
  The headline is internally consistent through it: `mix1.config.toml` (steps
  260 000, batch 24, seq 1024, eval_every 5 000) gives 24×1024×60 000 = 1.47 GB at
  the step the README's 1.936 is dated to.
- **The doc-refs gate explains itself and admits its blind spots.**
  `tools/check_doc_refs.py` keeps a `KNOWN_DEAD` ledger where every dead name
  carries the reason it may stay, checks that a name does not *disappear* from the
  list, and reports the 260 known-dead references every run ("444 files scanned,
  1153 path references resolved, 0 named paths missing").
- **The deploy gate is wired in the right order** (`ingest → build → check →
  upload`), the check is verified green here, and the site is live with
  `build_type: workflow`.
- **CI failures were root-caused, not skipped.** `50ef56d` replaces an empty
  `StopIteration` with an assertion carrying the engine's own output; the tempting
  `skipif` on the reporter was not taken.
- **Tooling discipline where it costs nothing:** pre-commit hooks are local-only
  and run the same three commands as CI; `.gitignore` carries the reason for each
  rule, including the commented-out demo-artifact line that records a decision
  instead of hiding it.

---

## Verdict

The epistemics are first-class — retractions next to claims, evidence named, and
the gate that would have caught a lie is the one that is checked in. The plumbing
is not: the first command on the front page does not parse, the headline number's
data is unobtainable, the run that is live right now is recoverable only from an
untracked TOML and a tmpfs script, the file behind ~87 citations lives on another
branch, and no CI job ever touches a GPU. A reader can trust the writing and
cannot yet run the thing.

**Operational maturity: 5/10** — honesty 9, reproducibility 3, automation 5.

---

*Not done here (read-only on existing files): this document needs a line in
`docs-site/tools/manifest.mjs` per AGENTS §1.8 to appear on the site. It is
gate-clean against `tools/check_doc_refs.py`'s rules as written.*
