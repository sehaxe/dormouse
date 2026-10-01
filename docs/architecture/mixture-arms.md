# Mixture arms: the anchors, acted on

`docs/research/2026-09-27-domain-anchors-renamed.md` measured the training mixture and left
it unbuilt. This is the build. Full record, with every byte count, every window
and every anchor:

    /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/mixture/MANIFEST.md

Three arms on disk as directory trees of symlinks (12–22 GB of mixture at zero
bytes copied). The trainer drains files one at a time under a per-epoch shuffle,
so bytes are the weight — `--data <arm dir>` is the whole interface. No config
change, no new flag.

| arm | total | what it is |
|---|---|---|
| `arm-a-prose` | 21.86 GB | control: prose only, bytes ∝ corpus size |
| `arm-b-anchored` | 17.91 GB | priced by 5-gram BPB × unseen rate, + code/math/genomics |
| `arm-c-heavy` | 11.99 GB | same price, code+math doubled, genomics 12.8% |

The price is one line: `share ∝ g × q`, the 5-gram BPB times the share of
held-out contexts unseen in train. `g` alone cannot separate prose (every text
domain is 2.25–2.76 BPB); `q` alone over-rewards small files. The product ranks
on "hard to predict **and** not locally memorizable".

## What re-deriving the anchors changed

Ten of ten inherited text rows reproduce to three decimals **at the inherited
2 MB window**. What that window hid:

- **`math` 2.446 is a JSON container, not math.** `mix/math` sorts its two
  `.jsonl` files (10.9 GB) before 13,708 `.txt` shards, so any 2 MB window is
  pure JSON. Same window, same size: jsonl 2.446, txt **2.757**. The mixture
  takes the txt only. Same failure mode that killed the "books is where the
  language lives" claim.
- **`books` has no stable bar.** 2.482 / 2.434 / 3.095 / 2.557 at 2 / 4 / 6.3 /
  10 MB — one 2.15 GB parquet, so the window walks documents.
- **`genomics` is bimodal, not a number.** 0.16–0.19 BPB on low-complexity
  windows, 2.16–2.19 on coding windows, 2.909 on held-out *species*. The
  inherited 2.010 prices the hard half only.
- **1.5 MB of train text is too small a sample.** Every sharded text domain is
  0.07–0.19 BPB pessimistic at 2 MB. The table is a valid *ranking* — which is
  all a mixture needs — and not an absolute bar.
- **`web` (103 GB) is the largest text domain on disk and was missing from the
  table.** Measured 2.458–2.537. The `real_filtered_v2` corpus actually being
  trained on reads **2.849** and its held-out tail **2.911**: DCLM filtering
  doubles the unseen-context rate, because the filter removes exactly the
  boilerplate a counter lives on.
- **`qa` reads 262 KB, not 2.6 GB.** The passage lives in `document.html`,
  inside a parquet *struct*; the reader extracts only top-level `StringArray`
  columns and drops it silently. The 3.245 row measures `id` + question stems.
  `qa` gets 0% of every arm until that is fixed (one `downcast` in
  `dormouse-data`, not done here).

## One number to not look at

| arm | blended, all domains | blended, text only |
|---|---|---|
| A | 2.505 | **2.505** |
| B | 2.405 | 2.612 |
| C | **2.317** | 2.628 |

C wins globally and is worst on text — the global number is counting the
genomics share, whose counter bar is 0.19. Every arm is judged on its own
per-domain ladder.

## Gate

`verify_mixture.py` — 8 checks, 53,889 assertions, exit 0. It resolves every
symlink, checks every extension against `collect_files` (an unaccepted one is
skipped *silently* and the arm trains on nothing while looking fine), proves no
link reaches an eval region, re-asserts `leaked_ids == []`, and reconciles every
byte total. All five fault injections — dangling link, wrong extension, eval
leak, emptied domain, extra file — were caught. `anchors` on all 26 arm/domain
directories returns the full 2 MB, so nothing is silently empty.

## Cost, and the order

5,000 steps = 25.6 M tokens = 5.67 h/arm at the canary's 4,081 ms/step;
17.0 h for all three. A run consumes 0.12–0.21% of an arm, so no share repeats.

**Run B, then A, then C.** B is the arm that tests the hypothesis and the only
one that can be falsified cheaply: if the pricing is right it beats A on code,
math, reasoning and genomics at once. A is only interpretable beside B. C is the
expensive bet and is the one that can be cut to zero without losing the answer.

Caveat: 4,081 ms/step was measured on a single-file corpus; these are
1,333–3,127-file trees. Time the first 30 steps (`--timers`) before committing.
