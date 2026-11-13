# The optimizer twin is already cut — what §3.3 actually describes, and what duplication is left

**Date:** 2026-10-01 · **Lane:** "вырезать двойника оптимизатора" · **Branch:** `wt/dedup-opt`

## The headline: the mandate's premise is false, and deleting `routing.rs` would break the trainer

The task was: `crates/dormouse-core/src/routing.rs` is dead, cut it.

It is not dead. It is **the live runtime path**, and the twin it was supposed to
replace was **deleted four days before this lane was written**, by `831e3a0`
(2026-09-28, `fix(train): the optimizer policy was declared twice and the two
copies disagreed`).

The trace, from the trainer entry point:

```
crates/dormouse-train/src/lib.rs:886   build_optim(&model, &cfg)
crates/dormouse-train/src/optim.rs:528   -> build_optim_mode(model, cfg, &cfg.opt)
crates/dormouse-train/src/optim.rs:480   -> optimizer_groups(model, cfg)
crates/dormouse-train/src/optim.rs:352   -> routing::routing(model, cfg.factors_fallback)   <== routing.rs RUNS
```

`optim.rs` no longer *decides* anything; it **assembles** what `routing.rs`
declares, into `burn::module::ParamGroup`s by `ParamId`, then verifies the
install against the live tree (`optim.rs:373 check_installed`). The string
policy is gone from the tree entirely — every symbol `831e3a0` deleted is
absent:

```
$ grep -rn "MUON_PATH_MARKERS\|is_muon_param\|is_qk_param\|is_engram_table_param\|QK_HEAD_MARKERS\|ENGRAM_TABLE_MARKER\|effective_muon_markers" --include=*.rs crates/
crates/dormouse-core/src/routing.rs:426:  // (is_muon_param said the weight was Muon+, ...
```

One hit, and it is a **comment remembering the divergence**, not code. The
markers are not in `optim.rs` any more; §3.3's "builds the optimizer from
path-string markers and is what runs" describes a file that no longer exists.

So the mandate's condition — «ЕСЛИ тесты routing.rs проверяют что-то, чего
optim.rs не проверяет — перенеси ЭТИ проверки, потом удаляй файл» — never
fires. The tests of `routing.rs` and the checks in `optim.rs` are **not two
views of one policy**; they are **two different questions about one policy**:

| | asks | fails when |
|---|---|---|
| `routing.rs` `Routing::check` (:205) | is the DECLARATION total? | a parameter is in no group, or in two |
| `optim.rs` `check_installed` (:373) | does the INSTALL reproduce the declaration? | the `ParamGroup`s the optimizer holds disagree with it |

Deleting `routing.rs` would delete the declaration and take
`optimizer_groups`, `build_optim`, the startup banner (`lib.rs:899`) and
`validate_routing` with it. **Not done. Not proposed.**

## What §3.3 got wrong, and the commit that made it wrong

The bullet is stale in a way that is worse than merely out of date: it names
the wrong file as authoritative, and it was written *after* the fix.

`routing.rs:265-268` (added `d81d920`, 2026-10-01, three days after `831e3a0`):

```rust
/// Note this trait is the ID-based DECLARATION. The policy that actually runs
/// is the path-marker implementation in `dormouse-train/src/optim.rs`, and
/// they agree today; until that is reconciled, say which one you mean
/// (AGENTS.md §3.3).
```

`git log -L` puts that text in `d81d920` (2026-10-01). `831e3a0` is
2026-09-28. **A docs commit reintroduced the twin into the narrative after the
code commit removed it** — §1.4's rule in its purest form: a claim that names
no evidence, contradicted by the code in the same repo.

The same false claim is live in three more places, all still asserting a twin:

- `docs/glossary.md:436-446` — a whole glossary section, "markers (path-string
  routing, **the live implementation**)", documenting `MUON_PATH_MARKERS`,
  `is_muon_param`, `is_qk_param` as the running policy with citations to
  `optim.rs:84,96,142,101,123,145`. **Every one of those line numbers is dead.**
- `docs/glossary.md:713` — disagreement **row 16**, "routing policy", OPEN.
- `docs/papers/muon-plus.md:186` — **D17**, "this is the id-based one,
  tests-only per `routing.rs`, with the path-marker copy deleted". Half right:
  the copy *is* deleted, and then it calls `routing.rs` tests-only.

This is why AGENTS.md §3.3 is worth more than the two sentences it spends: a
**glossary section teaches a reader a vocabulary that does not exist**, and
`docs/glossary.md` is the file §1.7 makes binding ("If a document and the code
disagree about a name or a default, fix one of them; never invent a third
name").

## What duplication IS left: two, both small and both real

### 1. `GroupCounts`, declared twice — the real twin, ~15 lines

Same four fields, same meaning, two crates:

- `routing.rs:65` — `#[derive(Default, Debug, PartialEq, Eq)]`, four documented `pub` fields.
- `optim.rs:440` — `#[derive(Default, Debug)]`, same four, undocumented.

`optim.rs`'s is `pub` and re-exported at `lib.rs:27`, so it is API. The
declaration is what the banner prints (`lib.rs:899`), so **routing.rs's** is
the one with the field docs that belong there.

**Cut:** delete `optim.rs`'s, re-export core's.

### 2. Two module visitors over the same tree — `IdCollector` vs `PathCollector`

Both walk the live `DormouseModel` with a `Vec<String>` path stack and collect
`(path, id)`. `optim.rs`'s also collects the rank `D`.

- `routing.rs:127 IdCollector` — `Vec<(String, ParamId)>`, used at `:161` (`rest_of`) and `:206` (`check`).
- `optim.rs:450 PathCollector` — `Vec<(String, ParamId, usize)>`, plus `param_paths` (:471), `pub` and used by tests in `lib.rs` and `tests/routing_policy.rs`.

Identical stack discipline, identical join, identical `visit_float`. The rank
is the *only* difference, and one impl costs one extra tuple slot.

**Cut:** `routing.rs`'s `IdCollector` is the wrong direction — it is in the
lower crate and does less. `param_paths` already carries the rank, so core
should own the one visitor and the train crate re-export it.

## What is NOT a twin (checked, so nobody re-litigates it)

- **`Routing::check` vs `check_installed`** — declaration vs install (§ table
  above). Both load-bearing; `optim.rs:355` calls `check`, then `:357` calls
  `check_installed`.
- **`Group::ALL`** (`routing.rs:54`) — no caller. Dead, but it is a
  compile-time tripwire on the enum, not a policy copy; deleting it is a
  separate call and not this lane's.
- **`role_of`/`Role`** — the *what is this* axis, used by `future_byte.rs:81`
  in prose and by `group_of`. Not an optimizer group.
- **Muon vs HeadWiseMuon** (`optim.rs:168`) — two `Optimizer` impls serving
  two different groups, both counted on the eval line. Not a policy copy.

## Line count

Net **−~30** (a `GroupCounts` struct + a `ModuleVisitor` impl + one
re-export), not the −200 the mandate projected: that projection assumed a
557-line file could be deleted, and it cannot be. The 557 lines of `routing.rs`
are the live policy, and the tests in it are the ones that make a new arm
failing to compile instead of silently training on AdamW.

## Follow-ups (not this lane)

1. **`Group::ALL` has no caller** — `routing.rs:54`. Either use it in
   `validate_routing` (it would then assert every group is *servable*) or
   delete it. 3 lines either way; needs an owner decision on whether the
   tripwire is wanted.
2. **`docs/adr/0017-dormouse-fused.md`** describes the id-based declaration as the mechanism.
   That is now correct and no ADR change is needed — but the ADR should say
   the trainer *calls* it, since the "two implementations" framing came from
   here.