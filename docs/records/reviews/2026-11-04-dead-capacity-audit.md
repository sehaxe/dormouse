# 2026-11-04 — dead-capacity audit: what else is doing nothing

Auditor: read-only pass over `port/`, at `34f1460` (tree otherwise clean apart
from two untracked research docs). Every command below was run with
`CUDA_VISIBLE_DEVICES=""`; the card was left to the sibling probe.

Ranked by **"could this have cost quality or speed for months"**.

The two known holes (432/512 experts dead, the dropless per-expert loop) are not
re-litigated. What follows is the next layer: parameters that get no gradient,
mechanisms that cannot change the output, counters whose zero is
indistinguishable from "off", and gates that pass without the arm running.

---

## 0. Headline

The routing collapse is **worse than reported, and the tool that reported it is
not measuring the arm.** On the real `m400` checkpoint (step 149 999) with real
bytes:

| selection | live experts | busiest expert | distinct expert pairs / 1024 tokens |
|---|---|---|---|
| **the arm** (`logits + lfb_bias`) | **8 / 512 (1.6 %)** | **1024 / 2048 assignments (50 %)** | **7** |
| the `route_hist.py` probe (`logits` only) | 67 / 512 (13.1 %) | 161 | 43 |
| `lfb_bias` zeroed / constant | 67 / 512 (13.1 %) | 161 | 43 |

The probe and the arm agree on **0.2 % of selected slots**. The reason is that
`lfb_bias` has range **[-218, +73]** (std 38.5) while the router's logits have
std **0.28** and range ±2.47 — the top-k is decided by the bias ranking, not by
the router, for essentially every token. The 80/512 on record is the *bias-free*
number, i.e. an understatement by ~10×.

Reproduce (loads the checkpoint on CPU, ~1 min, self-contained):

```bash
cd /home/sehaxe/dormouse/port && CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch; sys.path.insert(0, "src"); torch.set_num_threads(12)
from dormouse.config import RunConfig; from dormouse.model import DormouseModel
from dormouse.model import experts as E
base = "/home/sehaxe/dormouse/checkpoints/m400"
run  = RunConfig.from_snapshot(open(f"{base}/m400.config.toml").read())
st   = torch.load(f"{base}/m400.pt", map_location="cpu", mmap=True, weights_only=False)
m    = DormouseModel(run.model, device="cpu").eval(); m.load_state_dict(st["model"], strict=True)
moe  = m.loop.moe; moe.lfb_bias.copy_(st["moe"]["loop.moe"].float())
raw  = open("/home/sehaxe/dormouse/data_local/best/fwedu.bin", "rb").read(2 * 512 + 1)
ids  = torch.tensor(list(raw[:-1]), dtype=torch.long).view(2, 512)
tgt  = torch.tensor(list(raw[1:]),  dtype=torch.long).view(2, 512)
real, orig = [], E.SparseMoE._lfb_step
E.SparseMoE._lfb_step = lambda self, i, c: (real.append(i.detach().clone()), None)[1]
def hist(tag):
    real.clear()
    with torch.no_grad(): m(ids, tgt, depth=1)
    c = torch.bincount(real[0].reshape(-1), minlength=512).float()
    print(f"{tag:16s} live={(c>0).sum():3d}/512 max={int(c.max()):4d} pairs={len({tuple(sorted(r.tolist())) for r in real[0]})}")
hist("trained bias"); keep = moe.lfb_bias.clone()
moe.lfb_bias.zero_();            hist("bias zeroed")
moe.lfb_bias.copy_(keep); E.SparseMoE._lfb_step = orig
print("bias", float(keep.min()), float(keep.max()))
PY
```

Expected on `34f1460`: `live=  8/512 max=1024 pairs=   7` / `live= 67/512 ... pairs=  43` /
`bias -218.27 72.67`.

---

## 1. CRITICAL — the LFB controller is an unbounded ramp, and it has frozen the router

`src/dormouse/model/experts.py:1009-1025`

```python
if self.lfb_coef == 0.0 or not self.training:
    return
with torch.no_grad():
    counts = _bincount(index.reshape(-1), self.n_experts).float()
    self.lfb_counts.add_(counts)
    self.lfb_bias.sub_(self.lfb_coef * torch.sign(self.lfb_counts - self.lfb_counts.mean()))
```

`lfb_counts` accumulates **for the life of the run** (`experts.py:859-861`,
`persistent=True`) and is never decayed; the update is a *sign* integrator. A
sign integrator of a cumulative count whose sign per expert is essentially
constant is a **ramp**, not a controller — there is no clamp, no decay, and no
counter for the bias magnitude. At m400 (`moe_lb_coef = 0.01`, 3 iterations per
step) the bias has run to ±200.

Consequence, measured on the checkpoint:

* 8 experts alive, one expert in **every** token's top-2;
* 7 distinct expert pairs over 1024 tokens;
* the router's logits (std 0.28, range ±2.5) cannot change the selection: the
  eight winners sit at `b ≈ +72`, the median expert at `b ≈ +34` (a ~38-wide
  gap, ~15× the whole logit range), and four sit at `b = −218` — 786 k router
  parameters train the gate *weights* against a selection they cannot move;
* 504 of 512 experts receive no gradient on these windows, so the bank is dead
  capacity by construction.

The `_lfb_step` docstring says the cumulative mean "is what converges"; measured,
it diverges (bias) and freezes (routing). The A/B that chose it measured *drops*
on one short run, not load spread over 150k steps.

**Check (one line):** the script in §0. Or from the checkpoint alone:
`python -c "import torch;s=torch.load('checkpoints/m400/m400.pt',map_location='cpu',mmap=True,weights_only=False);print(s['moe']['loop.moe'].abs().max())"` → `218.27`.

**Fix direction (not applied):** per-batch (not cumulative) load, clamp
`|b| <= c`, and a counter `moe_lfb_bias_absmax` drained at log cadence so this
can never happen unreported again.

---

## 2. CRITICAL — `route_hist.py` does not measure the routing

`scripts/route_hist.py:70-77`

```python
counts_holder = getattr(moe, "_last_counts", None)
if counts_holder is None:
    print("the arm did not record counts; falling back to a direct router probe")
    ...
    logits = x @ w.t().to(torch.bfloat16)
    idx = logits.topk(moe.topk, dim=-1).indices
```

`moe._last_counts` **is never assigned anywhere in the tree** — grep returns this
one read and zero writes, so the "fallback" is the *only* path, on every run,
and its number is presented as the model's. The probe also:

* **omits `lfb_bias`** (`experts.py:961` `raised = logits + self.lfb_bias` is what
  the top-k actually reads) — measured error: 67 vs 8 live experts;
* uses **iteration 0 only** (`grabbed[0]` of a pre-hook that fires per
  iteration), while the real routing differs per iteration via `iter_embed[i]`;
* uses `topk(moe.topk)` instead of the adaptive-width path when
  `moe_topk_mode != "off"`;
* prints `live experts 0/512` for a **dense** MoE (`moe_topk = 0` → `topk(0)` is
  empty), i.e. a fully healthy bank reads as a total collapse;
* has no gate and no test (`grep -rn route_hist tests/` → nothing).

**Check:** `grep -rn "_last_counts" port/` → one line, all read.

**Fix direction:** add `self._last_counts` under `no_grad` in `_dispatch`
(one `bincount`), and make the probe refuse rather than guess when it is absent.

---

## 3. HIGH — the MSA "learned" block indexer never receives a gradient

`src/dormouse/model/sparse/sparse.py:110-119`

```python
proj  = torch.einsum("bihd,bjd->bijh", q, k_blk)
score = torch.relu(proj).sum(-1)
...
sel   = score.topk(k_eff, dim=-1).indices
keep  = torch.isfinite(score.gather(-1, sel))
```

`score` has no differentiable consumer: its only uses are `topk().indices`,
`masked_fill`, `gather`, `isfinite`. Measured: `loop.msa.selector.q_idx.weight`
and `.k_idx.weight` come back with `p.grad is None` on every arm (CPU SDPA and
the fused route both — the discrete selection is route-independent).

At m400 (`d=1536`, `MSA_Q_HEADS=4`, `head_dim=64`) that is
`1536*4*64 + 1536*64 = 491 520` parameters carried in the shipped preset, and —
worse than the parameters — the block-selection pattern is a **fixed random
projection drawn at init**. Nothing in the port trains it: no distill term, no
aux loss, no counter (`msa_blocks` counts blocks *used*, not the indexer), no
gradient gate. AGENTS §3.5.4 says the indexer must be trained in two stages;
that is not implemented here.

This invalidates the reading of the MSA verdict on record ("tie → MSA off in
`b1`", `tests/test_param_count.py:26`): the arm that was deleted was
*random-block* sparse attention, never the learned-indexer arm.

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'EOF'
import sys,torch; sys.path.insert(0,'src')
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
c=ModelConfig(d_model=64,n_heads=4,head_dim=16,d_ffn=128,max_iter=2,rank=8,
              n_experts=4,moe_topk=2,engram_rows=64,engram_dim=8,bf16=False,
              shared_rank=16,use_msa=True)
m=DormouseModel(c); ids=torch.randint(0,256,(2,64)); m(ids,ids)[1].backward()
print([n for n,p in m.named_parameters() if p.grad is None])
EOF
```

→ `['loop.attn.beta_proj.weight', 'loop.msa.selector.q_idx.weight',
'loop.msa.selector.k_idx.weight']` (on CPU; `beta_proj` is item 11).

---

## 4. HIGH — promotion rules that can pass without the mechanism

### 4a. `scripts/collect_ab.py:50-69` — a tie is a WIN when the control is tight

```python
c_spread = max(ctl) - min(ctl)
bar = c_mean - 2 * c_spread
...
elif mean <= bar: verdict = "WIN"
```

With **one** control log (the only guard is `if not ctl`), `c_spread = 0`, so
`bar == c_mean` and **an exact tie is promoted as a WIN**. The docstring promises
3 control seeds; nothing enforces it. The collector also never checks `scored`
(the eval window), `nan_skipped`, or the config — so arms measured on different
windows are ranked together, which AGENTS §2.6 says is not a number.

### 4b. Eleven drivers print a "3 seeds" void check they do not perform

`scripts/ab_mhc.py:351`, `ab_attn.py:441`, `ab_csa.py`, `ab_prelude.py`,
`ab_mhc_attnres.py`, `ab_byteflow.py`, `ab_multiblock.py`, `ab_docmask.py`,
`ab_engram_ram.py`, `ab_act_quant.py`, `ab_depth8.py` all accept `--seeds` and
then print, e.g. (`ab_mhc.py:333`):
`"void checks: ... {len(seeds)} distinct seeds per arm.\n"`.
`summarize()` (`ab_byteflow.py:186-188`) is `max-min` with no length guard, so
with `--seeds 1` the spread is 0 and *any* improvement wins. Only
`ab_cheapwidth:162`, `ab_energy:304`, `ab_jepa:206`, `ab_sf_muon:197` actually
refuse fewer than two seeds. `ab_multiblock.py:140` knows the gap ("the counter
line carries no per-body key") and closes it from the params row — the right
pattern, not copied.

**Check:** `grep -rn "len(control)\|len(seeds)" port/scripts/ab_*.py` — four
hits, in four files out of fifteen.

---

## 5. HIGH — config knobs that are accepted and do nothing

All four verified by **bit-identical logits** under a changed value:

| knob | where it is read | when it is inert | evidence |
|---|---|---|---|
| `msa_topk` | `sparse/attention.py:200` (`k = round(k_share*n_blocks)`) | whenever `adaptive_msa=true` (`k_override` bypasses `selector.topk`) | `msa_topk=2` vs `16`, adaptive on → `torch.equal` = True |
| `msa_capacity_ratio` | `loopblock.py:1091` | whenever `adaptive_msa=false` | `0.25` vs `1.0` → identical logits |
| `msa_gate` (parameter, `max_iter`) | `loopblock.py:1088` | whenever `adaptive_msa=false` — but it is **built anyway** (`loopblock.py:366-369`) and counted by `param_count` | gate `0` vs `[5,0]` → identical logits |
| `adaptive_msa` | `loopblock.py:1087` guarded by `self.max_iter > 1` | `max_iter=1`, accepted by `validate` | `adaptive` on vs off → identical logits |

`configs/medium.toml` and `configs/m400.toml` set `use_msa=true`,
`adaptive_msa=true`, **and** `msa_topk=32`. The arm runs
`k = round(1.0 · 0.25 · 256) = 64` blocks per query (t=1024, block=4), i.e. **2×
the declared width** — twice the gathered keys, in the presets whose MSA
overhead was already measured at 1.6× the arm-off run on `small`.
`msa_topk=32` is decorative in exactly the two presets that set it.

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
T = dict(d_model=64,n_heads=4,head_dim=16,d_ffn=128,vocab=256,max_seq_len=64,max_iter=2,
         rank=8,n_experts=4,moe_topk=2,engram_rows=64,engram_dim=8,bf16=False,shared_rank=16)
def build(**o): torch.manual_seed(0); return DormouseModel(ModelConfig(**{**T,**o}))
ids = torch.randint(0,256,(2,64)); tgt = ids.clone()
def out(m):
    with torch.no_grad(): return m(ids,tgt)[0]
for label, a, b, mut in (
    ("capacity_ratio .25 vs 1.0 (adaptive off)", dict(use_msa=True,msa_capacity_ratio=.25),
                                                dict(use_msa=True,msa_capacity_ratio=1.0), None),
    ("topk 2 vs 16 (adaptive on)", dict(use_msa=True,adaptive_msa=True,msa_topk=2),
                                   dict(use_msa=True,adaptive_msa=True,msa_topk=16), None),
    ("adaptive on vs off (max_iter=1)", dict(use_msa=True,adaptive_msa=True,max_iter=1),
                                        dict(use_msa=True,adaptive_msa=False,max_iter=1), None),
    ("msa_gate 0 vs [5,0] (adaptive off)", dict(use_msa=True,msa_topk=4),
                                           dict(use_msa=True,msa_topk=4), [5.,0.])):
    ma, mb = build(**a), build(**b)
    if mut is not None:
        with torch.no_grad(): mb.loop.msa_gate.data = torch.tensor(mut)
    print(f"{label}: identical={torch.equal(out(ma), out(mb))}")
PY
```

→ four `identical=True` lines.

---

## 6. MEDIUM — `residual_scale` is dead under **both** residual arms, not just mHC

`loopblock.py:430` builds it unconditionally; the only read is
`loopblock.py:1276` `h = ctx + y * blk.residual_scale`, inside the final `else`
(the ReZero arm). Under `use_attnres` (`:1253-1255`) and `use_mhc`
(`:1256-1273`) it is never read.

* `residual_scale = 0.25` under `use_mhc` → logits bit-identical.
* `residual_scale = 0.25` under `use_attnres` → logits bit-identical.
* No counter says "the ReZero scale is inert"; `rezero_write` only counts the
  branch that *does* read it.
* `param_count` counts it (`tests/test_param_count.py:56`) and the optimizer
  groups it.
* Only the mHC half is documented, in a test docstring
  (`tests/test_full_graph.py:29` "`loop.residual_scale` has no gradient under
  `use_mhc`"); `use_attnres` is not mentioned anywhere.

Cost: 1 parameter per body, but the A/B confound is real — "ReZero vs mHC" and
"ReZero vs AttnRes" each remove a *learned residual scale* as well as changing
the operator, and a user tuning the scalar sees no effect.

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
T = dict(d_model=64,n_heads=4,head_dim=16,d_ffn=128,vocab=256,max_seq_len=64,max_iter=2,
         rank=8,n_experts=4,moe_topk=2,engram_rows=64,engram_dim=8,bf16=False,shared_rank=16)
ids = torch.randint(0,256,(2,64)); tgt = ids.clone()
for arm in ({"use_mhc": True}, {"use_attnres": True}):
    torch.manual_seed(0); a = DormouseModel(ModelConfig(**{**T, **arm}))
    torch.manual_seed(0); b = DormouseModel(ModelConfig(**{**T, **arm}))
    b.loop.residual_scale.data.fill_(0.25)
    with torch.no_grad(): la, lb = a(ids,tgt)[0], b(ids,tgt)[0]
    print(f"{list(arm)[0]}: residual_scale ignored = {torch.equal(la, lb)}")
PY
```

---

## 7. MEDIUM — `counters.why` has writers and no readers

Nine modules attach a fallback reason to `counters.why`
(`experts.py:1689,1695,1717,1733,1739,1798,1807,1820`, `kda.py:407,487`,
`tsct.py:212`, `moe_kernels.py:583`, `sparse/attention.py:341`,
`train/ns_kernels.py:455`, `train/optim.py:1350`, `train/graph.py:516`,
`decode.py:560,880`). The trainer's log line is
`train/loop.py:1765` `window.line()`, and `Counters.line()` prints
`f"{k}={self[k]}"` over the dict — `why` is an *attribute*, not an entry. The
JSONL writes `counters=dict(window)`; `Counters.merge`/`absorb` copy keys and
device tensors only, so `why` is dropped at the first merge too.

So every docstring claiming "the reason on `counters.why`" is false at the only
consumer that matters. A training log shows `moe_triton_fallback=…` and
`moe_naive_dispatch=…` but never *why*.

**Check:** `grep -rn "\.why" port/src | grep -v "why ="` → no `print`/`line`
consumer.

---

## 8. MEDIUM — zero-vs-not-run still bites in five places

The instrument drops integer zeros by design (`instrument.py:147-150`: floats
are kept, ints are not). The remaining counters where that matters:

| counter | line | what a missing key means |
|---|---|---|
| `qat_reuse` | `model.py:701-703` (only counted `if reuse:`) | "cache reused 0 times" **or** "QAT arm off" — no `qat_miss` exists, so a cache that misses every call is invisible |
| `moe_offloaded` | `experts.py:1611-1613` | counts the *branch*: on an eval/`no_grad` forward `checkpoint` parks nothing yet `moe_offloaded=2` (verified) — the mechanism did nothing and the log says it ran |
| `moe_dropped_assignments` | `experts.py:1669` | "0 drops" and "no capacity arm" are the same absence |
| `moe_k{width}` | `experts.py:1036-1038` | a width no token chose vanishes from the histogram |
| `layerrope_forward`, `fused_*_launches` | `model.py:717`, `fused_residual.py:957,999,1025` | arm-off and arm-ran-zero are the same |

`activation_offload` gets this right (`model.py:672` gates on
`torch.is_grad_enabled()`); `moe_offload` does not.

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
T = dict(d_model=64,n_heads=4,head_dim=16,d_ffn=128,vocab=256,max_seq_len=64,max_iter=2,
         rank=8,n_experts=4,moe_topk=2,engram_rows=64,engram_dim=8,bf16=False,shared_rank=16,
         moe_offload=True)
m = DormouseModel(ModelConfig(**T)).eval()
ids = torch.randint(0,256,(2,64))
with torch.no_grad(): _,_,c = m(ids, ids, pure=True)
print("eval no_grad: moe_offloaded =", c.get("moe_offloaded"))   # 2, with nothing parked
PY
```

---

## 9. MEDIUM — LayerRoPE paper: two `for_iteration` calls per iteration, and the gate is red

`loopblock.py:936-946` computes `gains` and `read_gain`; `loopblock.py:981`
sets `gains = None`; `loopblock.py:982-991` computes `gains` **again**. Merged
commit `1583e74` is titled "resolve the gains once per iteration" — the merge
kept both copies.

Measured: `max_iter=3` → `for_iteration` called **6** times per forward
(`[(0,0),(0,0),(0,1),(0,1),(0,2),(0,2)]`), and
`layerrope_forward = 24` where the arm applies 12.

`tests/test_layerrope.py:847` asserts `== 4 * max_iter` and is **failing at
HEAD** (`assert 24 == 12`). Values are identical (both slices come from the same
prebuilt table), so this is a counter lie + a redundant per-iteration op, not a
numerical bug — but it is the arm's only "it ran" evidence, and it reads 2×.

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch; sys.path.insert(0, "src")
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
T = dict(d_model=64,n_heads=4,head_dim=16,d_ffn=128,vocab=256,max_seq_len=64,max_iter=3,
         rank=8,n_experts=4,moe_topk=0,engram_rows=64,engram_dim=8,bf16=False,shared_rank=16)
m = DormouseModel(ModelConfig(**{**T, "layerrope": True, "layerrope_form": "paper"}))
calls, orig = [], m.layerrope.for_iteration
m.layerrope.for_iteration = lambda b,i: (calls.append((b,i)), orig(b,i))[1]
ids = torch.randint(0,256,(2,64))
with torch.no_grad(): m(ids, ids)
print("calls", calls)   # 6 for max_iter=3; want 3
PY
```

---

## 10. MEDIUM — `PROGRESS_KEYS` carries a model-side key; a day-one gate is red and the entry is dead

`config.py:55` adds `"moe_kernel"` to `PROGRESS_KEYS`, which is consumed only in
`RunConfig.diff_keys` under `section == "train"` (`config.py:1981`). `moe_kernel`
is a `ModelConfig` field, so the entry does nothing; the real exemption is the
model-side branch at `config.py:1995-1997` (commit `0f62bfb`).

`tests/test_config.py:339` asserts `set(PROGRESS_KEYS) <=
set(TrainConfig().__dataclass_fields__)` and fails at HEAD. So the resume
exemption that `5323735` claimed is split across two places, one of which is
inert, and the suite went red.

**Check:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_config.py::test_snapshot_round_trip_and_progress_exemption`.

---

## 11. MEDIUM — the dead-parameter gate is configured away from every arm where parameters die

`tests/test_param_count.py:209-220` is the only whole-model
"no parameter without a gradient" gate:

```python
model = DormouseModel(tiny_cfg(use_engram=False, use_kda=False))
...
dead = [n for n, p in model.named_parameters() if p.grad is None]
assert dead == []
```

It runs with **no attention, no memory, no MSA, no mHC/AttnRes, no halt/MTP**.
Every parameter that actually dies (items 3, 6, and the `msa_gate`) is outside
the fixture, so the gate is green while the shipped presets carry inert
parameters. The docstring even excuses `beta_proj` by hand ("a different,
pre-existing arm") — accurately, but that is the gate declining to look.

**Check:** change the fixture to `tiny_cfg(use_msa=True)` and the gate fails on
`loop.msa.selector.{q_idx,k_idx}.weight`.

---

## 12. LOW — adaptive MSA makes one host read per iteration, uncounted

`loopblock.py:1091` `float(share[i].detach()) * self.msa_capacity_ratio` — a
D2H read once per iteration (`__float__` spy: 2 calls for `max_iter=2`, all from
`:1091`). It is the only host read left in a training forward, it is present in
`medium`/`m400` (`adaptive_msa=true`), and it is **not counted** by any counter.
It is also why `full_graph` refuses `use_msa`
(`train/graph.py:623-628`, which cites the now-stale `loopblock.py:576`).

**Check:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import sys, torch, traceback, collections; sys.path.insert(0, "src")
from dormouse.model import DormouseModel; from dormouse.config import ModelConfig
T = dict(d_model=64,n_heads=4,head_dim=16,d_ffn=128,vocab=256,max_seq_len=64,max_iter=2,
         rank=8,n_experts=4,moe_topk=2,engram_rows=64,engram_dim=8,bf16=False,shared_rank=16)
hits, orig = collections.Counter(), torch.Tensor.__float__
def spy(self):
    fr = traceback.extract_stack()[-2]; hits[f"{fr.filename.split('/port/')[-1]}:{fr.lineno}"] += 1
    return orig(self)
torch.Tensor.__float__ = spy
m = DormouseModel(ModelConfig(**{**T, "use_msa": True, "adaptive_msa": True, "msa_topk": 4}))
with torch.no_grad(): m(torch.randint(0,256,(2,64)), torch.randint(0,256,(2,64)))
torch.Tensor.__float__ = orig
print(dict(hits))   # {'src/dormouse/model/loopblock.py:1091': 2}
PY
```

---

## 13. LOW — the halt term has no counter, and `last_halt` can be stale

`model.py:1011-1028` adds `halt_weight * halt_loss(...)` with **no
`counters.count`**, while `z_loss` (`:1009`), `sigreg` (`:1044`), `jepa`
(`:1055`) and `energy` (`:1063`) all count "the term was added". `halt_loss`
returns `None` on a `rand_depth` step that ran one iteration (`halt.py:134-135`),
and `self.last_halt = term.detach()` is assigned **only when non-None** — so
`train/loop.py:121-131 halt_field` prints the *previous* step's value, and
returns `""` when there never was one. A head that never trained can show a
plausible `halt=` number forever.

**Check:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_halt.py` passes; then run two forwards, second with `depth=1` under `rand_depth` and diff `model.last_halt`.

---

## 14. LOW — counters that count the config, not the work

* `model.py:655-660`: `counters.count("dense_qat_arms")` fires once per forward
  whenever `dense_qat != "none"`, whether or not a single weight was projected
  (`dense_qat_skip_attn=True` skips every attention site). The name says "arms";
  the value is a flag echo. The real proof is the test.
* `experts.py:884`: `share_u_qat_forward` is the same shape (flag echo).
* `experts.py:1036-1038` `moe_k_mean_x100` is an integer division: a mean width
  below 0.01 would vanish.

**Check:** `grep -n "dense_qat_arms" port/src/dormouse/model/model.py`.

---

## 15. LOW — knobs read only under a different `expert_act`

`expert_beta1` (4.0), `expert_beta2` (25.0) and `memglu_c0` (0.604) reach
`expert_glu` only when `expert_act` names `situ`/`memglu`
(`experts.py:720-727`, `:304-308`). With the default `expert_act="silu"` they
are accepted and ignored, with no refusal and no counter — the same shape as
`msa_topk` (item 5), but at least these are named for their activation.

**Check:** set `expert_beta1=1.0` with `expert_act="silu"` and diff the logits.

---

## 16. Known, listed for completeness (not new)

* `loop.attn.beta_proj.weight` — no gradient on the SDPA fallback
  (`kda.py:600-640` `_sdpa_path_carry` computes `q,k,v,g` and never `beta`).
  Documented in `tests/test_param_count.py:212` with a **stale citation**
  (`kda.py:500` is inside `step_state`; the real code is `:600-640`). 9 216
  parameters at `small` (`d=768`, `h=12`); zero on the FLA/CUDA arm.
* `tests/test_dashboard.py:71` asserts the literal `"flagship"`, which
  `frag_hero` only prints when **no run is found** (`scripts/dashboard.py:51`
  default). On a box with live logs it fails; in CI it passes. It is the third
  red test at HEAD and it measures ambient machine state, not the dashboard.

---

## 17. Suite state at HEAD (acceptance check)

`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q` (CPU, several minutes):

```
exit 1 — 3 failed (the run's counts line is lost to a teardown exception)
FAILED tests/test_config.py::test_snapshot_round_trip_and_progress_exemption   (item 10)
FAILED tests/test_dashboard.py::test_fragments_render                          (item 16)
FAILED tests/test_layerrope.py::test_paper_j_counter_and_refusals              (item 9)
```

So "full tests green" is **not** met at `34f1460`; two of the three failures are
real code/counter defects introduced today (`5323735`, `1583e74`), one is a
state-dependent test.

---

## 18. What was checked and is clean

So that this memo does not read as "everything is broken":

* every parameter of `mHC` (`phi_pre/post/res`, `alpha_*`, `b_*`), `Engram`,
  `CSA` (fine / coarse / mid / hierarchical), `share_u` (plane and dict),
  `LOOM`, `halt`, `MTP` (hard and soft), `JEPA`, `DSpark`, `Energy`,
  `attn_window`, `prelude`/`coda`, `loop_blocks=2` receives a non-zero gradient
  on the tiny fixture;
* `engram_rows`/`engram_arms`, `moe_shared_forwards`, `iter`, `depth`,
  `forward`, `csa_forward`, `msa_blocks` are host counters that cannot silently
  read zero for a running arm;
* the dropless route is genuinely gated (`tests/test_moe_batched.py:273-310`
  pins bit-identity *and* which counter names the arm);
* `_banks`' `_version`+grad-state cache key, and the optimizer's
  `p.grad is not None` filter, are correct (the dead params above are never
  weight-decayed).
