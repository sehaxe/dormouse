# Segmented CSA — the boundary contract

Branch `segcsa`, off clean `main` `d1b2c2e` (2026-11-07). Port only
(`port/src/dormouse/`); CPU gates, no GPU run. The 1M VRAM numbers in §5 that
are not computed from our own formulas are the premise this task started from
(the 300 KiB-per-byte one-shot activation figure) and are marked as such.

## 1. The contract, in one paragraph

A segment cut is a **memory** decision and may not move a chunk boundary. The
CSA state already carries the current chunk's `sum` and `count`, so a window
may start inside a chunk: the first `chunk - count` bytes COMPLETE the carried
chunk as `(sum + their sum) / chunk` — the same mean the one-shot `avg_pool1d`
computes over those `chunk` bytes — and a trailing partial chunk is carried the
same way. Nothing is dropped (which would silently forget the bytes before the
cut) and nothing is padded (which would average in bytes that do not exist).
The state's `n·chunk + count` is the number of bytes it has seen, so the byte
position of a window's first token is reconstructed **from the state alone**
(`pos = (count + arange(t)) // chunk + n`); no caller passes an offset, and the
causal mask (`key < pos`), the top-k set and the coarse/mid levels are the
one-shot ones. `segment_len` need not be a multiple of `csa_chunk` — that is the
hard case the carry exists for. On the training path the ring is **fp32**
(`CSAState.dtype`): a bf16 ring rounds every pooled key a second time and the
1e-5 logit gate cannot hold (bf16 ulp at 1.0 is 7.8e-3). The decode ring stays
bf16, which is what `state_bytes`/the memory table describe; a chunked prefill
now leaves an fp32 ring and therefore costs twice the table's bytes (follow-up
4.2).

## 2. Why this is the only contract that is the same model

Three alternatives were on the table when `_validate_segment` refused the pair:

| cut rule | logits at 1M | what it is |
|---|---|---|
| **drop** the partial chunk | wrong from the cut on | forgets bytes the one-shot pass scored |
| **pad** it to a whole chunk | wrong at both sides of the cut | averages in bytes that do not exist, and shifts every later chunk index |
| **restart** the position counter | wrong mask / wrong top-k | positions re-read chunk 0 of the new segment |
| **carry** (this commit) | equal to 1e-5 | same chunk membership, same mask, same selection |

The two directions that are easy to get wrong and are both gated: keeping the
chunk counter `n` while ignoring the phase `count` (the "rotation offset" at 0)
drops the chunk a cut completed from the query's allowed set; keeping the phase
while restarting `n` lets every segment re-read chunk 0. Both are in §7.

## 3. The two boundaries, byte by byte

`csa_chunk=8`, `segment_len=100`, window 200 bytes (cuts at 100 and 200):

| segment | bytes | fine chunks completed here | carried at the cut |
|---|---|---|---|
| 0 | 0..99 | 0..11 (0..95) | chunk 12 = bytes 96..99, `count=4`, `sum=Σx[96:100]` |
| 1 | 100..199 | 12 completed as `mean(x[96:104])`; 13..24 (104..199) | — (`count=0`, `n=25`) |

Query positions: `pos = (4 + j) // 8 + 12`, i.e. byte 100 → chunk 12 (its own,
never allowed), byte 104 → chunk 13 (chunk 12 allowed). The ring entry 12 is
asserted equal to `W_k(mean(x[:, 96:104]))` to 1e-6, and entries 11/13 to
`mean(x[88:96])`/`mean(x[104:112])` — the whole boundary membership, not only
the median value.

Coarse level, `csa_coarse_chunk=256`, window 600, same 100-byte segments: the
segment-2 bytes 200..255 complete coarse chunk 0 (`mean(x[0:256])`), bytes
256..299 are carried (`count=44`); byte 200 has `pos_coarse=0`, byte 256 has
`pos_coarse=1`. The coarse ring entries 0/1 are asserted against
`mean(x[0:256])`/`mean(x[256:512])`.

## 4. Parity numbers

Tiny CPU fixture: `d=64`, one loop block, `max_iter=2`, batch 2, fp32 logits,
`csa_chunk=8`; the reference is one call to `DormouseModel.forward`
(`segment_len=0`). `tests/test_segment_csa.py`, run on this branch:

| seq | segment_len | segments | coarse | max abs Δlogits | Δloss |
|---|---|---|---|---|---|
| 256 | 64 | 4 | off | 1.43e-6 | 0 |
| 256 | 128 | 2 | off | 1.91e-6 | 0 |
| 200 | **100** | 2 | off | 1.91e-6 | 0 |
| 512 | 64 | 8 | 256 | 1.25e-6 | 0 |
| 512 | 128 | 4 | 256 | 1.28e-6 | 0 |
| 600 | **100** | 6 | 256 | 1.55e-6 | 0 |

The gate is 1e-5; the measured error is 5-8× inside it and moves by ~0.5e-6
between runs because fp32 CPU reduction order moves with the thread pool (the
pin in gate 1 therefore fixes `torch.set_num_threads(1)`: the same fixture
measured four different shas at 1/2/4/10 threads).

State, 600 bytes / six 100-byte segments against one continuous
`forward_state` (gate 5, tolerance 1e-6): fine `k` 1.79e-7, `v` 2.38e-7,
coarse `k` 4.10e-8, `v` 4.38e-8, both `sum`s bit-identical, and
`n·chunk + count == bytes seen` asserted after every segment (75/0 fine,
2/88 coarse at the end).

Gradients (gate 9, one-shot vs two segments, all 48 parameter tensors):
worst relative difference **8.4e-7**. Gate 8 also proves the carry is a graph
edge at the module level: a loss on segment 2 puts |g|=244.9 on segment 1's
input bytes through the completed chunk.

## 5. Memory at 1M

The activation premise this task started from: **307 200 B of activation per
byte of context**, i.e. 286 GiB for a 1M-byte window in one forward (the
number is from the brief; the code change below is what removes the window
factor). With `segment_len` the stream activation is `segment_len · 307 200 B`,
**independent of total context**: 1.17 GiB at 4096, 2.34 GiB at 8192.

What segmentation does NOT remove is the CSA ring and the selection tensor:

| term (per layer, per iteration slot) | formula | at 1M |
|---|---|---|
| fine ring, fp32 (training) | `2·kv·4/chunk` = 64 B/byte | 64.0 MiB |
| coarse ring, fp32 | `2·kv·4/256` = 2 B/byte | 2.0 MiB |
| fine ring, bf16 (decode; the table's bytes) | 33 B/byte | 33.0 MiB |
| coarse ring, bf16 | 1 B/byte | 1.0 MiB |
| fine score tensor, per segment | `4·seg·(ctx/8)` | 2.0 GiB at seg 4096 |
| coarse score tensor, per segment | `4·seg·(ctx/256)` | 64 MiB at seg 4096 |

So for `flagship_base` (one block, `max_iter=3`): training ring
**66 MiB × 3 = 198 MiB**; decode ring 33 MiB × 3 = 99 MiB (the brief's "34 MB"
is one slot). The context-dependent *selection* tensor is why the 1M command
in §6 ships the coarse level alone (`csa_fine=false`): with the fine scan on,
a 4096-byte segment materialises a 2 GiB score matrix per level per slot, which
is the 286 GiB problem at 1/256 the price rather than a solved one. The
designed fix is `csa_hierarchical`, and it is refused here because its prefill
is already not the one-shot model (§8.1). Fine-on alternative that fits a 16 GB
card: `segment_len=1024` → 512 MiB of fine scores, at ~4× the per-window CSA
work of the coarse-only arm.

## 6. The exact GPU command for a 1M-curriculum run

`flagship_base` is the preset that already turns CSA on (fine + coarse, one
block, `max_iter=3`, dropless MoE at `moe_capacity=0`). `seq_len` and
`max_seq_len` are resume PROGRESS keys; `segment_len` is not, so the curriculum
must keep one value from the first stage to the last. 1M is divisible by 4096,
and 8192 (stage 1) / 4096 = 2 segments is legal.

```bash
cd /home/sehaxe/dormouse/port

# stage 1 — 8 KB window, 2 segments, the SAME segment_len the 1M stage keeps
systemd-run --user --scope -p MemoryMax=40G \
  uv run dormouse train flagship_base \
    --data /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_filtered/corpus.bin \
    --eval  /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin \
    --device cuda --batch 1 --steps 2000 \
    --set seq_len=8192 --set segment_len=4096 \
    --set csa_fine=false --set csa_coarse_chunk=256 --set csa_coarse_kv_dim=64 \
    --set out_dir=$HOME/logs/seg1m --set ckpt_name=seg1m

# stage 2 — extend the same checkpoint to 1M (seq_len/max_seq_len are progress keys)
systemd-run --user --scope -p MemoryMax=40G \
  uv run dormouse train flagship_base \
    --data /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_filtered/corpus.bin \
    --eval  /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/aria_data/pretrain/real_eval/eval_tail.bin \
    --device cuda --batch 1 --steps 4400 --resume \
    --set seq_len=1048576 --set max_seq_len=1048576 --set segment_len=4096 \
    --set csa_fine=false --set csa_coarse_chunk=256 --set csa_coarse_kv_dim=64 \
    --set out_dir=$HOME/logs/seg1m --set ckpt_name=seg1m \
    --set json_log=$HOME/logs/seg1m/seg1m.jsonl
```

Both commands resolve through `load_run` + `validate_model_train` on this
branch (checked: `seq=1048576`, `segment_len=4096`, `csa_fine=False`,
`csa_coarse_chunk=256`, `csa_hierarchical=False`). The eval window is
`eval_batches × batch × seq_len`, so it grows with the curriculum; the BPB is
only comparable inside one window.

## 7. Gates, and the scratch mutation that reddens each

`tests/test_segment_csa.py` — 10 functions / 13 cases, all CPU. Every
mutation below was applied to a copy of `model/csa.py` (or `config.py`),
run, and restored; "internal assert off" means the
`n·chunk + count == bytes seen` guard inside `_carried_state` was removed so
the gates could be shown to fail for their own reason rather than on the guard.

| gate | what it pins | RED on |
|---|---|---|
| 1 | `segment_len=0` is the pinned one-shot (`2bc9cf69…`, coarse `4eb661ad…`, captured from `d1b2c2e`), never enters `forward_state` | M8 (one-shot `forward` +1e-7) |
| 2 | 2-4 segments == one pass, `segment_len` a multiple of the chunk (64/128) | M3 (0.31 / 0.26) |
| 3 | the hard case `segment_len=100 % 8` | M1, M2, M3, M4 |
| 4 | the coarse ring carries too (512/64, 512/128, 600/100) | M1 0.114, M2 0.082, M3 0.36, M4 0.035 (600/100), M5 0.114 |
| 5 | carried state == continuous state, 1e-6; `n·chunk+count == bytes` after every segment | M1, M2, M3 (0.38), M4 (0.028), M5 (0.045), M6 |
| 6 | the chunk a cut splits is `mean(x[96:104])` (and coarse 0/1) | M1 (0.58), M2 (0.33), M5 (IndexError: empty ring) |
| 7 | `topk>=n` is hand-written dense strict-causal attention across the cut | M1 (0.030), M2 (0.021), M3 (0.383), M4 (0.030) |
| 8 | the carry is a gradient edge (segment 2's loss moves segment 1's bytes) | M1, M6 (internal assert off) |
| 9 | segmented gradients == one-shot gradients (this is the checkpoint replay gate) | M1 (0.318), M2 (0.284), M3 (1.073), M4 (0.080), **M6 (CheckpointError: different number of tensors saved)** |
| 10 | the refusals: `segment_len < csa_chunk`, `csa_hierarchical` | M7 (remove both checks → DID NOT RAISE) |

Mutations, as named in the brief:

* **M1 drop the partial chunk** — zero `sum`/`count` instead of carrying them
  (`internal assert off`). Gate 2 stays green: with every cut on a chunk
  boundary there is nothing to carry. That is exactly why gate 3 exists.
* **M2 pad it** — close the chunk at the cut with `sum / chunk` (missing bytes
  zero), append it, restart `count=0`. The next segment then pools from the
  wrong boundary.
* **M3 restart the position counter** — `pos = (count + j)//chunk` without
  `+ n`. Every fine query re-reads chunk 0.
* **M4 keep the rotation offset at 0** — `pos = j//chunk + n`, i.e. drop the
  phase `count`. Green on every 64/128 case *and on the 512-byte coarse cases*
  because those cuts are chunk-aligned; red on 100. The non-multiple case is
  load-bearing, not decorative.
* **M5 ignore the coarse ring** — create but never advance the coarse state.
  Forward logits diverge 0.114; the fine gates stay green.
* **M6 mutate the incoming state instead of copying it** — the natural
  implementation, and the one the logits cannot see: gates 2/3/4/6/7/8 all
  stay green and gate 9 dies with
  `CheckpointError: A different number of tensors was saved during the
  original forward and recomputation`. `_forward_segmented` wraps the CSA call
  in `checkpoint`, whose backward replays the body on the same state object;
  an in-place append appends every chunk twice in the backward. This is why
  `_carried_state` copies the ring into a fresh `CSAState`.
* **M7** remove the two new refusals; gate 10 does not raise.
* **M8** add `+1e-7` to the one-shot `forward`; gate 1's sha pin reddens.

## 8. Refusals kept / added, and the follow-ups

Kept loud: `use_msa`, `use_byteflow`, `use_kda=false`, `attn_kind='full'`,
`segment_len >= seq_len`, a non-divisor of `seq_len`, `rand_depth`, `halt`,
capacity-dropping MoE, `doc_masking`, JEPA/DSpark, and `csa_fine=false` without
a coarse level. Added:

* `segment_len < csa_chunk` — a segment shorter than one chunk completes no
  fine entry; the first segment's candidate ring is empty (and the
  hierarchical gather would index an empty tensor), a shape no gate covers.
* `segment_len` with `csa_hierarchical` — see 8.1.

### 8.1 Pre-existing: the hierarchical PREFILL is not the one-shot model

Measured on **clean `main` `d1b2c2e`**, no branch code involved:
`csa_hierarchical=true, csa_mid_chunk=512, csa_coarse_chunk=256,
csa_chunk=8, csa_topk=1000`, 600 bytes —
`csa.forward` (one-shot) vs `csa.forward_state(x, None)` differ by **1.7e-2**
while `forward`'s hierarchical and flat paths agree to 6e-8. The ring has no
entry for the coarsest level's last PARTIAL chunk, and `_hier_fine` derives its
fine candidate set from that level's survivors, so up to
`coarse_chunk/chunk - 1` fine chunks are missing from the candidates. With
`topk=32` the top-k masks it in practice, which is why no gate saw it. A
segmented forward would inherit the gap at every cut, so the pair is refused
until the prefill is repaired (add the carried partial parent's children,
masked by `cand < pos`, then remove the refusal and add a parity case).

### 8.2 Chunked prefill leaves an fp32 ring

`forward_state` builds fp32 whatever the caller (the training contract). The
decode path's `step_state` still builds bf16, which is what the memory table
describes; a `prefill="chunked"` run continuing into decode now carries an
fp32 ring and costs 2× the table's bytes. Fix if it matters: a `dtype`
parameter on `forward_state` for the decode caller (the training path keeps
the default).

### 8.3 Ring copy per segment

`_carried_state` copies `n·kv` entries per level per segment so the checkpoint
replay is idempotent (§7 M6). It is O(ring) against an attention read that is
O(segment·ring), so it is small — but it is there, and a copy-on-write ring
with an append-only chunk list would remove it if a profile ever notices.

## 9. Suite status

* `tests/test_segment_csa.py`: 13/13 green.
* The touched CSA/longctx/step-audit set (`test_segment_csa`, `test_csa`,
  `test_csa_coarse`, `test_csa_fused`, `test_longctx`, `test_step_audit`):
  **82 passed**. `test_step_audit` was red until the new ring-append loop was
  counted as `csa.ring_append` (and the now-stale `forward_state` excuse was
  removed).
* Full CPU suite in this worktree: **43 failed**. Clean `main` `d1b2c2e` on
  this box: **39 failed**; the diff is 4, all missing UNTRACKED data in a
  fresh worktree (`port/checkpoints/` for 3 `test_config` snapshot tests, and
  `port/sft_pairs/{aria_en,aria_reason,code.qa}` for `test_sft_v2`'s manifest).
  No code failure is new. The brief's known reds and the `test_muon_*`,
  `test_moe_kernel`, `test_ns_triton`, `test_opt_*`, `test_residual_bf16`
  families are red identically on main.
* `ruff check src tests scripts/ab_longctx.py` clean. `ruff format` is NOT run
  on the touched files: the tree is not format-clean today (reformatting
  `test_csa_coarse.py` alone produced 60 lines of unrelated churn), so only the
  edited hunks were kept.
* `tools/check_doc_refs.py` reports **0** broken paths from this document. Its
  remaining refs are the untracked `docs/research/2026-11-0*.md` notes absent
  from a fresh worktree (the same class as the checkpoint data above).

## 10. Files

* `port/src/dormouse/model/csa.py` — `CSAState.dtype` (fp32 training / bf16
  decode), `_carried_state` (the contract, per level), `forward_state`
  (phase-aware positions, no half-full-chunk refusal), module docstring.
* `port/src/dormouse/model/loopblock.py` — the `self.csa is None` refusal is
  gone; the hierarchical and MoE capacity seams stay; `_forward_segmented`'s
  docstring names the ring carry and the functional-copy reason.
* `port/src/dormouse/config.py` — the CSA+segment refusals removed, the
  `segment_len < csa_chunk` and hierarchical checks added, docstrings updated.
* `port/src/dormouse/train/loop.py` / `model/model.py` — the segment-loop
  comments now name both carried states and why no byte offset is passed.
* `port/src/dormouse/stepaudit.py` — the stale `forward_state` excuse removed;
  the new ring-append loop is counted as `csa.ring_append`.
* `port/scripts/ab_longctx.py` — the pre-registered driver's "use_csa refuses
  segment_len" claim corrected (its CSA arms also set `attn_window`, so their
  row is still 0, for the band reason).
* `port/tests/test_segment_csa.py` (new), `test_csa.py` / `test_csa_coarse.py`
  (their gate-6 refusal assertions updated to the new contract).
