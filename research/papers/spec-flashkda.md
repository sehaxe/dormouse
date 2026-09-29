# FlashKDA — literal transcription of `tests/torch_ref.py`

Single source: `github.com/MoonshotAI/FlashKDA`, branch **`master`** (there is no `main`;
`.../main/...` 404s). Everything below was fetched read-only on **2026-09-29**. No GPU,
no cargo, no other repo file touched.

## 1. Provenance

| file | URL | branch | HTTP | lines (incl. trailing newline split) |
|---|---|---|---|---|
| `tests/torch_ref.py` (**the source of this document**) | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/tests/torch_ref.py` | `master` | 200 | **248** |
| `flash_kda/__init__.py` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/flash_kda/__init__.py` | `master` | 200 | 42 |
| `csrc/fwd.h` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/csrc/fwd.h` | `master` | 200 | 28 |
| `csrc/flash_kda.cpp` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/csrc/flash_kda.cpp` | `master` | 200 | 233 |
| `csrc/smxx/fwd_kernel1.cuh` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/csrc/smxx/fwd_kernel1.cuh` | `master` | 200 | 587 |
| `csrc/smxx/fwd_kernel2.cuh` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/csrc/smxx/fwd_kernel2.cuh` | `master` | 200 | 840 |
| `csrc/smxx/fwd_launch.cu` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/csrc/smxx/fwd_launch.cu` | `master` | 200 | 239 |
| `tests/test_fwd.py` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/tests/test_fwd.py` | `master` | 200 | 432 |
| `tests/test_fwd_full.py` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/tests/test_fwd_full.py` | `master` | 200 | 275 |
| `docs/20260420-flashkda-v1-deep-dive.md` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/docs/20260420-flashkda-v1-deep-dive.md` | `master` | 200 | 88 |
| `README.md` | `https://raw.githubusercontent.com/MoonshotAI/FlashKDA/master/README.md` | `master` | 200 | 132 |
| tree listing | `https://api.github.com/repos/MoonshotAI/FlashKDA/git/trees/master?recursive=1` | `master` | 200 | 34 entries |

**No 404s.** Everything requested resolved. There are **no Triton files** in this repo —
the kernels are CUTLASS/CuTe `.cuh` under `csrc/smxx/`. GitHub code search for `g_min`
was not usable (HTTP 401, unauthenticated), so the `g_min` claim below rests on the
complete recursive tree listing plus the per-file greps, not on code search.

Repo tree (paths only): `.clangd.template`, `.gitignore`, `.gitmodules`,
`BENCHMARK_GB200.md`, `BENCHMARK_H20.md`, `LICENSE`, `README.md`, `benchmarks/`,
`config.yaml`, `csrc/`, `docs/`, `flash_kda/`, `setup.py`, `setup_clangd.sh`, `tests/`.

---

## 2. The recurrence, transcribed literally from `torch_ref.py`

Verbatim, in order, source variable names, source line numbers. Nothing paraphrased,
nothing corrected. Line 248 is the artifact of the trailing newline.

```python
  1| import torch
  2| from torch.utils.cpp_extension import load_inline
  3| 
  4| # ============================================================
  5| # sigmoid via tanh.approx.f32: tanh(x*0.5)*0.5+0.5
  6| # ============================================================
  7| _sigmoid_cuda_src = r"""
  8| #include <torch/extension.h>
  9| #include <cuda_runtime.h>
 10| 
 11| __global__ void sigmoid_tanh_fp32_kernel(const float* __restrict__ input,
 12|                                          float* __restrict__ output, int n) {
 13|     int idx = blockIdx.x * blockDim.x + threadIdx.x;
 14|     if (idx < n) {
 15|         float xh = input[idx] * 0.5f;
 16|         float th;
 17|         asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(xh));
 18|         output[idx] = th * 0.5f + 0.5f;
 19|     }
 20| }
 21| 
 22| torch::Tensor sigmoid_tanh_fp32(torch::Tensor input) {
 23|     auto output = torch::empty_like(input);
 24|     int n = input.numel();
 25|     sigmoid_tanh_fp32_kernel<<<(n + 255) / 256, 256>>>(
 26|         input.data_ptr<float>(), output.data_ptr<float>(), n);
 27|     return output;
 28| }
 29| """
 30| 
 31| sigmoid_ext = load_inline(
 32|     name='sigmoid_ext',
 33|     cpp_sources='torch::Tensor sigmoid_tanh_fp32(torch::Tensor input);',
 34|     cuda_sources=_sigmoid_cuda_src,
 35|     functions=['sigmoid_tanh_fp32'],
 36|     extra_cuda_cflags=['-O2'],
 37|     verbose=False,
 38| )
 39| 
 40| # ============================================================
 41| # Numeric helpers
 42| # ============================================================
 43| 
 44| LOG2E = 1.4426950408889634
 45| 
 46| 
 47| def fp32_ex2_ftz(x):
 48|     if x.dtype == torch.float16:
 49|         x = x.to(torch.float32)
 50|     ret = torch.special.exp2(x)
 51|     ret = torch.where(ret.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(ret), ret)
 52|     return ret
 53| 
 54| 
 55| def fp32_fma(c, a, b):
 56|     assert c.dtype == torch.float32
 57|     assert a.dtype == torch.float32
 58|     assert b.dtype == torch.float32
 59|     return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)
 60| 
 61| 
 62| def l2_normalize_kernel_match(x):
 63|     """L2 normalize matching kernel's warp-shuffle tree reduction with FMA.
 64|     x: [..., D] bf16, D must be 128.
 65|     """
 66|     x_f32 = x.float()
 67|     groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)
 68| 
 69|     partials = torch.zeros(*x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device)
 70|     for i in range(8):
 71|         partials = fp32_fma(partials, groups[..., i], groups[..., i])
 72| 
 73|     for offset in [8, 4, 2, 1]:
 74|         indices = torch.arange(16, device=x.device) ^ offset
 75|         partials = partials + partials[..., indices]
 76| 
 77|     inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
 78|     return (x_f32 * inv_norm).to(x.dtype)
 79| 
 80| 
 81| # ============================================================
 82| # (I + L)^-1: 8x8 fp32 forward substitution + 16x16 bf16 block merge.
 83| # Mirrors the kernel's inv_fwd_subst_fused_1warp bit-for-bit (replaces the fp16
 84| # Neumann series, which loses accuracy when |L| -> 1 near-collinear keys).
 85| # ============================================================
 86| 
 87| def inv_fwd_subst_16(L):
 88|     """(I + L)^-1 for strictly-lower fp32 L [16, 16] or [N, 16, 16], bf16 out."""
 89|     squeeze = L.dim() == 2
 90|     if squeeze:
 91|         L = L.unsqueeze(0)
 92|     n = L.shape[0]
 93|     device = L.device
 94|     seed = L  # strictly-lower fp32; X = I + L = I + seed
 95| 
 96|     # Diagonal 8x8 blocks inverted by fp32 forward substitution
 97|     # (kernel-exact FMA order: rank-1 updates below the pivot row,
 98|     # pivot row broadcast from its finalized values at step s).
 99|     inv8 = torch.cat([seed[..., :8, :8], seed[..., 8:, 8:]], dim=0)
100|     idx8 = torch.arange(8, device=device)
101|     inv8[:, idx8, idx8] = 1.0
102|     for s in range(7):
103|         row_scale = -inv8[:, :, s]  # [2N, 8]
104|         for p in range(s):
105|             inv8[:, s + 1:, p] = fp32_fma(
106|                 inv8[:, s + 1:, p], row_scale[:, s + 1:], inv8[:, s, p:p + 1])
107|         inv8[:, s + 1:, s] = row_scale[:, s + 1:]
108| 
109|     # Merge: P = diag(A^-1, B^-1) bf16, M = [0 0; C 0] bf16 (the kernel
110|     # quantizes fp32 -> bf16 only at the HMMA inputs), dc = P @ M (fp32 acc),
111|     # o = bf16(-dc) @ P (fp32 acc), INV = P + bf16(o) (disjoint blocks).
112|     P = torch.zeros(n, 16, 16, dtype=torch.bfloat16, device=device)
113|     P[:, :8, :8] = inv8[:n].to(torch.bfloat16)
114|     P[:, 8:, 8:] = inv8[n:].to(torch.bfloat16)
115|     M = torch.zeros_like(P)
116|     M[:, 8:, :8] = seed[..., 8:, :8].to(torch.bfloat16)
117|     INV = torch.empty_like(P)
118|     for b in range(n):
119|         dc = torch.mm(P[b], M[b], out_dtype=torch.float32)
120|         o = torch.mm((-dc).to(torch.bfloat16), P[b], out_dtype=torch.float32)
121|         INV[b] = P[b] + o.to(torch.bfloat16)
122|     return INV.squeeze(0) if squeeze else INV
123| 
124| 
125| # ============================================================
126| # Torch reference implementation
127| # ============================================================
128| 
129| def torch_ref(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None):
130|     """Torch reference, supports both fixed-length and variable-length sequences.
131| 
132|     Input: [B, T, H, D] (4D). B must be 1 when cu_seqlens is provided.
133| 
134|     initial_state/final_state can be:
135|       - None: no state (zero-init / skip store)
136|       - bf16 tensor: [N, H, D, D]
137|       - fp32 tensor: [N, H, D, D] (converted to bf16 for compute, back to fp32 for output)
138|     """
139|     assert q.dim() == 4, f"Expected 4D input [B, T, H, D], got {q.dim()}D"
140|     B = q.shape[0]
141|     if cu_seqlens is not None:
142|         assert B == 1, f"B must be 1 when cu_seqlens is provided, got B={B}"
143|     # Reshape to [B*T, H, D] for internal processing
144|     q = q.reshape(-1, *q.shape[2:])
145|     k = k.reshape(-1, *q.shape[2:])
146|     v = v.reshape(-1, *q.shape[2:])
147|     g = g.reshape(-1, *q.shape[2:])
148|     beta = beta.reshape(-1, *q.shape[2:])
149|     out = out.reshape(-1, *q.shape[2:])
150|     if B > 1:
151|         T_seq = q.shape[0] // B
152|         cu_seqlens = torch.arange(0, B * T_seq + 1, T_seq, dtype=torch.long, device=q.device)
153|     _, H, D = q.shape
154|     CHUNK = 16
155|     device = q.device
156|     scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)
157| 
158|     q = l2_normalize_kernel_match(q)
159|     k = l2_normalize_kernel_match(k)
160| 
161|     if A_log is not None:
162|         assert dt_bias is not None
163|         assert A_log.dtype == torch.float32
164|         assert g.dtype == torch.bfloat16
165|         assert dt_bias.dtype == torch.float32
166|         g = g.to(torch.float32) + dt_bias.unsqueeze(0)
167|         a_log_exp = fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
168|         scale = lower_bound * LOG2E
169|         g = scale * sigmoid_ext.sigmoid_tanh_fp32(a_log_exp * g)
170| 
171|     state_fp32 = (initial_state is not None and initial_state.dtype == torch.float32) or \
172|                  (final_state is not None and final_state.dtype == torch.float32)
173| 
174|     if cu_seqlens is None:
175|         T = q.shape[0]
176|         cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)
177| 
178|     N = len(cu_seqlens) - 1
179| 
180|     if initial_state is not None:
181|         work_state = initial_state.to(torch.bfloat16).clone()
182|     else:
183|         work_state = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=device)
184| 
185|     for seq_idx in range(N):
186|         bos = cu_seqlens[seq_idx].item()
187|         eos = cu_seqlens[seq_idx + 1].item()
188|         seq_len = eos - bos
189|         n_chunks = (seq_len + CHUNK - 1) // CHUNK
190| 
191|         for chunk_idx in range(n_chunks):
192|             t0 = bos + chunk_idx * CHUNK
193|             actual_len = min(CHUNK, eos - t0)
194| 
195|             for h in range(H):
196|                 g_chunk = torch.zeros(CHUNK, D, dtype=g.dtype, device=device)
197|                 q_chunk = torch.zeros(CHUNK, D, dtype=q.dtype, device=device)
198|                 k_chunk = torch.zeros(CHUNK, D, dtype=k.dtype, device=device)
199|                 v_chunk = torch.zeros(CHUNK, D, dtype=v.dtype, device=device)
200|                 beta_chunk = torch.zeros(CHUNK, dtype=beta.dtype, device=device)
201| 
202|                 g_chunk[:actual_len] = g[t0:t0 + actual_len, h, :]
203|                 q_chunk[:actual_len] = q[t0:t0 + actual_len, h, :]
204|                 k_chunk[:actual_len] = k[t0:t0 + actual_len, h, :]
205|                 v_chunk[:actual_len] = v[t0:t0 + actual_len, h, :]
206|                 beta_chunk[:actual_len] = beta[t0:t0 + actual_len, h]
207| 
208|                 g_cumsum = g_chunk.cumsum(dim=0)
209|                 g_total = g_cumsum[-1:]
210|                 k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
211|                 q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
212|                 neg_g_cumsum_bf16 = fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
213|                 k_inv = k_chunk * neg_g_cumsum_bf16
214|                 g_total_exp_bf16 = fp32_ex2_ftz(g_total).to(torch.bfloat16)
215|                 k_restored = k_inv * g_total_exp_bf16
216|                 L = torch.mm(k_decayed, k_inv.t(), out_dtype=torch.float32)
217|                 Mqk = torch.matmul(q_decayed, k_inv.t())
218| 
219|                 # Fuse sigmoid via tanh.approx: beta is bf16 logits
220|                 beta_activated = sigmoid_ext.sigmoid_tanh_fp32(beta_chunk.to(torch.float32))
221|                 beta_val_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
222|                 L = torch.tril(L, diagonal=-1) * beta_activated.unsqueeze(-1)
223|                 Mqk = torch.tril(Mqk)
224| 
225|                 INV = inv_fwd_subst_16(L)
226| 
227|                 state_slice = work_state[seq_idx, h]
228|                 v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())
229|                 v_chunk = v_chunk * beta_val_bf16
230| 
231|                 U = torch.matmul(INV, v_chunk)
232|                 _out = torch.matmul(q_decayed, state_slice.t())
233|                 _out = _out + torch.matmul(Mqk, U)
234| 
235|                 delta_s = torch.mm(k_restored.t(), U, out_dtype=torch.float32)
236| 
237|                 g_total_exp = fp32_ex2_ftz(g_total)
238|                 g_total_exp = g_total_exp.squeeze(0).unsqueeze(-1)
239|                 work_state[seq_idx, h] = fp32_fma(delta_s, state_slice.to(torch.float32).t(), g_total_exp).to(torch.bfloat16).t()
240| 
241|                 out[t0:t0 + actual_len, h] = _out[:actual_len]
242| 
243|     if final_state is not None:
244|         if state_fp32:
245|             final_state.copy_(work_state.to(torch.float32))
246|         else:
247|             final_state.copy_(work_state)
248|
```

### Reading notes (facts about the lines above, not rewrites of them)

- **Line 168 rebinds the name `scale`.** The `scale` argument (softmax-style `D^-0.5`,
  line 156) is *not* the one used at line 211 — line 211 uses `scale_bf16`, captured
  from the argument before line 168. Line 168's `scale` is only the gate's
  `lower_bound * LOG2E` multiplier at line 169. Both exist; they are not the same value.
- `g` is pre-activation logits. Line 166 adds `dt_bias`; line 169 maps to
  `lower_bound * LOG2E * sigmoid(exp(A_log*LOG2E) * (g + dt_bias))`, computed in
  base-2 so the inner `exp` is a single `exp2` (line 167).
- `beta` is bf16 **logits** (line 219 comment); sigmoid is applied at line 220 via
  `tanh.approx.f32`. `beta_val_bf16` is unsqueezed to `[CHUNK, 1]` (line 221) and
  broadcasts over `D` at line 229.
- `k_decayed` / `k_inv` / `k_restored` (lines 210, 213, 215) implement
  `k_i·2^{g_≤i}` and `k_i·2^{-g_≤i}`; the decay is **per-key-channel** (`[CHUNK, D]`
  cumsum at line 208), not scalar per token. Base-2 throughout, so the cumsum is
  already the log2 exponent.
- The state is stored **transposed**: `work_state[seq_idx, h]` is `[D, D]`, updated at
  line 239 with a trailing `.t()`.
- State compute is bf16 throughout; only `fp32_fma` at line 239 and the `.t()` in
  `state_slice` at line 232 are fp32. This is the numeric floor to match, not a
  convenience.

---

## 3. Every constant the file sets

| # | line | name | exact value | note |
|---|---|---|---|---|
| 1 | **154** | `CHUNK` | `16` | also line 189, 192, 193, 196–200 |
| 2 | **44** | `LOG2E` | `1.4426950408889634` | module-level |
| 3 | **77** | L2-norm epsilon | `1e-6` | inside `rsqrt(partials[..., 0:1] + 1e-6)` |
| 4 | **67** | L2 groups reshape | `16, 8` | `[D]` → `[16, 8]`, i.e. **D = 128** |
| 5 | **64** | `D` (docstring requirement) | `128` | "D must be 128"; `D` itself is a runtime input (line 153) |
| 6 | **70** | L2 partial-count loop | `range(8)` | 8 FMA partials |
| 7 | **73** | warp tree reduction offsets | `[8, 4, 2, 1]` | XOR shuffle, 16 lanes |
| 8 | **74** | lane index | `torch.arange(16)` | 16 lanes |
| 9 | **101** | `inv8` diagonal seed | `1.0` | identity on the diagonal |
| 10 | **99, 112, 114, 116** | inverse block split | `8` | 8×8 fp32 blocks, 16×16 total |
| 11 | **102** | substitution steps | `range(7)` | |
| 12 | **51** | `exp2` flush-to-zero | `torch.finfo(torch.float32).tiny` | sub-normals → 0 |
| 13 | **15, 18** | sigmoid PTX | `tanh.approx.f32` on `x*0.5`, then `*0.5 + 0.5` | inline asm, fp32 |
| 14 | **25** | sigmoid launch | `<<<(n + 255) / 256, 256>>>` | 256 threads/block |
| 15 | **129** | `lower_bound` | **argument, not a constant** | no default in this file |
| 16 | **156** | `scale` | **argument, not a constant** | captured to bf16 before line 168 |
| 17 | — | `H` | **argument, not a constant** | line 153 |
| 18 | — | decay init (`a_log`/`b_alpha`) | **NOT IN `torch_ref.py`** | `A_log` is an *input* (line 129) |

**`g_min` is NOT in `torch_ref.py`.** No occurrence anywhere in the 248 lines. The
nearest thing in this file is the `lower_bound` argument at line 129, used at line 168.
Note that the name in the repo is `lower_bound`, not `g_min`.

Constants that live **outside** `torch_ref.py` but pin the same model shape — do not
attribute them to this file:

- `csrc/flash_kda.cpp:10-11` — `constexpr int CHUNK = 16; constexpr int D = 128;`
- `csrc/flash_kda.cpp:128` — `float gate_scale = float(lower_bound * 1.4426950408889634);`
- `csrc/flash_kda.cpp:138` — `constexpr int CHUNK = 16;` (again, in the varlen entry)
- `csrc/smxx/fwd_launch.cu:31` — `constexpr int CHUNK = 16;`
- `csrc/smxx/fwd_kernel1.cuh:295-296` — `rsqrtf(q_sq + 1e-6f)` / `rsqrtf(k_sq + 1e-6f)`
- `csrc/smxx/fwd_kernel1.cuh:258` — `float a_log_exp = expf(A_log_ptr[head_idx]);` (natural
  exp, not the base-2 form of the reference — the log2 conversion is folded into
  `gate_scale`)
- `flash_kda/__init__.py:34` — "Currently requires `K = V = 128`."
- `README.md:101` / `flash_kda/__init__.py:24` — `lower_bound` "range from -5.0 to 0"
- `tests/test_fwd.py:225,268,323,374` and `tests/test_fwd_full.py:23` —
  `LOWER_BOUND = -5.0`; `tests/test_fwd.py:240` / `test_fwd_full.py:35` —
  `scale = 1.0 / math.sqrt(D)`; `tests/test_fwd.py:224` — `B, T, H, D = 1, 8192, 96, 128`.
  These are *test inputs*, not reference constants; `A_log` there is
  `torch.rand(H, ...)` (`test_fwd.py:236`) or `torch.full((H,), 0.0, ...)`
  (`test_fwd.py:332`) — **no `a_log = -3` init appears in this repo's tests**, and
  `b_alpha` does not appear at all.

---

## 4. What the file does NOT contain — the ceiling

Read this before implementing against it.

1. **No backward.** `torch_ref.py` is forward only. The whole repo is forward only:
   the tree has `fwd_kernel1.cuh`, `fwd_kernel2.cuh`, `fwd_launch.cu`,
   `csrc/flash_kda.cpp`, `csrc/fwd.h` — no `bwd` file, and `flash_kda/__init__.py`
   exports only `fwd`. **Any gradient claim about FlashKDA in this repo is unsupported
   by this repo.** For dormouse this is the load-bearing gap.
2. **`g_min` is absent.** See §3. If a design doc or a neighbouring library uses
   `g_min` and attributes it to FlashKDA, that attribution is not supported by this
   file. The parameter here is `lower_bound`.
3. **No decay initialization.** No `a_log = -3`, no `b_alpha`, no softplus, no
   `-exp(A)` form. `A_log` and `dt_bias` are *inputs* (line 129, asserted fp32 at
   lines 163/165). If you need an init, it is yours to choose and must be labelled
   as such.
4. **No `lower_bound` default or recommended value.** It is a required argument
   (line 129). The `-5.0` figure is test/README-level, not a reference constant.
5. **No RoPE, no Q/K head-wise splitting, no normalization placement.** The reference
   L2-normalizes `q` and `k` (lines 158-159) with a *fixed* 16×8 reshape and gives no
   hook for a rotary or any other positional encoding. Nothing about RoPE is here.
6. **No head count, no layer stacking, no FFN, no controller.** One head at a time
   (line 195, `for h in range(H)`), one operator, no model.
7. **No batch/sequence dimension beyond `cu_seqlens`.** `B>1` is handled by synthesizing
   a uniform `cu_seqlens` (lines 150-152); ragged batching is only via explicit
   `cu_seqlens`. No packing/padding logic beyond zero-padding the tail chunk
   (lines 196-206).
8. **No output gate, no per-token normalization, no `use_short_conv`.** None of the
   GDN-family peripherals are present. The only "activation" is
   `sigmoid(a_log_exp * (g + dt_bias))` scaled by `lower_bound` (lines 166-169).
9. **No tolerance, no reference-output fixture.** The file computes; it asserts
   nothing about correctness. Tolerances live in the test files, which I read but did
   not transcribe (`tests/test_fwd.py`, `tests/test_fwd_full.py`).
10. **No `D` other than 128 is claimed to work.** `l2_normalize_kernel_match` hardcodes
    the 16×8 reshape (line 67) and its docstring says `D` must be 128 (line 64).
    It is not written to be generic.
11. **Not a performance reference.** It is a numerics reference and is a Python triple
    loop over `seq_idx` × `chunk_idx` × `h` (lines 185/191/195). No timing, no
    TFLOP/s, no comparison to FLA. The only performance artifacts in the repo are
    `BENCHMARK_H20.md`, `BENCHMARK_GB200.md`, `benchmarks/`, which I did not read.
