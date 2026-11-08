# Incident: the merge deleted 94 GB of artifacts (2026-11-08, 02:5x)

**Severity:** the pretraining corpus and the older checkpoints are gone. The
*record* is intact: every number in `logs/` (16 GB) and every document survived.
**Cause:** my `git merge origin/main` (the other lineage, see `.bulba/plan.md`),
not the disk cleanup.

## What happened

`origin/main` tracks two paths as **symlinks** (mode `120000`):

```
checkpoints -> /home/sehaxe/dormouse/port/checkpoints   (blob 26ade5c)
data_local  -> /home/sehaxe/dormouse/port/data_local    (blob 999f55a)
```

Locally both were **untracked, gitignored directories** with the real data. A
merge checkout must materialise the tracked symlink at that path; git treats an
ignored directory as expendable, so it removed the directory to make room, wrote
the symlink, and `git merge --abort` then removed the symlink. Net effect: the
directory is gone and nothing in git can restore it.

## What was lost

| path | size | what |
|---|---:|---|
| `dormouse/data_local/` | ~80 GB | `mix4/` (5.77 GB train + eval), `best/` (fwedu 3 GB, cosmo 3 GB, finemath 1.5 GB, ru, science, code, math), the parquet caches, `web_eval.bin` |
| `dormouse/checkpoints/` | 14 GB | `m400`, `m400b`, `flagship`, `flagship2`, `dense103`, `final_dense`, `trio_bf_s2/s3` |

## What survives

| path | size | what |
|---|---:|---|
| `dormouse/port/checkpoints/` | 25 GB | `b1run`, `b1night`, `medium`, `qat`, `newmodel2`, `cont`, `flagship`, `mix1`, … |
| `/tmp/opencode/ck/flagship3.pt` | 762 MB | the checkpoint behind every site number and the `1.4621` evaluation |
| `/home/sehaxe/logs/` | 16 GB | every run's JSONL + log: the numbers are re-derivable as *numbers*, not as weights |

`/tmp` is a tmpfs: **copy `flagship3.pt` somewhere persistent before reboot.**

## Recovery status (2026-11-08 04:20)

| step | status |
|---|---|
| `flagship3.pt` out of /tmp | **done** — `checkpoints/flagship3/` |
| code / math / RU slices (200 MB each) | **done** — `data_local/mix/*.bin`, 600 MB |
| fwedu / cosmo / finemath (1.2 GB each) | **BLOCKED**: Hugging Face answered HTTP 429 — "we had to rate limit your IP… pass a HF_TOKEN". The three small slices used the anonymous quota (500 requests / 300 s) and the parquet reader issues many range requests per shard. |
| eval window | gone; must be carved fresh from the refetched fwedu, and every number on it is new |

To resume: put a token in `HF_TOKEN` (or `~/.cache/huggingface/token`) and run
`uv run port/scripts/refetch_pretrain.py --cap-mb 1200`; without a token, wait
out the window and lower the cap.

## Recovery (the plan)

1. **Immediate (5 min):** copy `/tmp/opencode/ck/` to `dormouse/checkpoints/flagship3/`.
2. **Small slices (network, ~30 min):** `uv run port/scripts/fetch_mix_sources.py`
   fetches ~120 MB each of code/math/RU from the Hub — enough for a smoke, not
   for a training run.
3. **The primary source (network, hours):** FineWeb-Edu is a Hub dataset and can
   be sliced with DuckDB over HTTPS the same way `fetch_mix_sources.py` does for
   the other three; a 1-3 GB slice is the minimum for the recipe ladder. Cosmopedia,
   FineMath and peS2o follow the same pattern (`docs/guides/dataset.md` names the
   sources and the build).
4. **Full corpus (network + CPU, a day):** `build_mix4.py` rebuilds `mix4/` from
   the mount `mix/` — **that mount is also gone**, so the web/RU/wiki shares must
   be re-fetched first.

## The rule this costs

A tracked symlink over an ignored directory is a loaded gun: any checkout,
merge, or `git restore` can delete the target tree. The fix is one of:

- stop tracking `data_local`/`checkpoints` as symlinks (they are local artifacts,
  not source), or
- move the artifacts outside the repo (`/mnt/…`) and point the symlink there, so
  a checkout can only remove the link, never the data.

Until one of those lands, **do not merge or checkout with these paths present.**
