# GatedDeltaNet-2 (NVlabs) — official spec transcription

Read-only transcription of the NVIDIA reference implementation. Everything below is
copied or mechanically derived from the files listed in §1. No equation is invented;
gaps are marked **NOT STATED**.

---

## 1. Provenance

| item | value |
|---|---|
| repo | `github.com/NVlabs/GatedDeltaNet-2` |
| tree API | `https://api.github.com/repos/NVlabs/GatedDeltaNet-2/git/trees/main?recursive=1` → HTTP 200, 31 entries, `truncated: false` |
| branch | `main` |
| HEAD sha | `a5552fe3c67e0ebc7ef1220df68ae8896ec62d56` |
| HEAD date | 2026-08-29T17:54:49Z, commit "add GDN-2 poster" (Ali Hatamizadeh, 1 file changed +4/-0) |
| raw prefix | `https://raw.githubusercontent.com/NVlabs/GatedDeltaNet-2/main/<path>` — all fetches HTTP 200 |
| fetched on | 2026-09-29 |

Files read, with line counts (`wc -l`):

| path | lines | bytes |
|---|---|---|
| `lit_gpt/gdn2.py` | 396 | 17622 |
| `lit_gpt/gdn2_ops/chunk_gdn2.py` | 2193 | 82284 |
| `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py` | 498 | 18475 |
| `lit_gpt/config.py` | 153 | 6184 |
| `lit_gpt/__init__.py` | (read via grep) | 1127 |
| `lit_gpt/model.py` | (imports only, via grep) | 16765 |
| `requirements.txt` | 39 | 1394 |
| `pyproject.toml` | 54 | 1473 |
| `Dockerfile` | 55 | 1791 |
| `README.md` | 196 | 7537 |
| `scripts/tsz1024x4k_100B_swa_gdn2.sh` | 61 | 1846 |

**Path correction:** the task brief named `gdn2_ops/chunk_gdn2.py`. It is actually at
`lit_gpt/gdn2_ops/chunk_gdn2.py`; there is no top-level `gdn2_ops/`.

Not read: `paper/GDN2_paper.pdf` (486638 bytes), `assets/`, `pretrain.py`, `cache.py`,
`data.py`, `lit_gpt/{utils,tokenizer,packed_dataset,rmsnorm,rotary,speed_monitor,
fused_cross_entropy,fused_rotary_embedding}.py`.

---

## 2. The chunked-WY algebra, transcribed

All line numbers are `lit_gpt/gdn2_ops/chunk_gdn2.py` unless stated.
Pipeline order is the one documented at lines 31-39 and implemented by
`chunk_gdn2_fwd` (953-1052).

### 2.0 Pre-step — `ChunkGDN2Function.forward` (2021-2117)

```
2055:  chunk_size = 64
2059:  if use_qk_l2norm_in_kernel:
2060:      q, q_rstd = l2norm_fwd(q)          # from fla.modules.l2norm  (import line 64)
2061:      k, k_rstd = l2norm_fwd(k)
```

`l2norm_fwd` is external; its eps is **NOT STATED** in this repo. (The recurrent
kernel's own l2norm uses `+1e-6`, line `fused_recurrent_gdn2.py:199-200`.)

### 2.1 Gate cumsum — `chunk_gdn2_fwd` (982-1000)

```
982:   if use_gate_in_kernel:
983-992:  g = kda_gate_chunk_cumsum(g, A_log, dt_bias, scale=RCP_LN2,
                                     chunk_size=64, cu_seqlens, chunk_indices, lower_bound)
993:   else:
994-1000:  g = chunk_local_cumsum(g, scale=RCP_LN2, chunk_size=64, ...)
```

`RCP_LN2` is `1/ln2` (imported line 67 from `fla.ops.utils.constant`; its value is
**NOT STATED** here). Consequence, visible everywhere below: the cumsum lives in
base-2 and every decay is `exp2(...)`, never `exp`. `g` enters as log-space decay and
leaves as an in-chunk cumulative sum.

### 2.2 Buffers — `chunk_gdn2_fwd_intra` (795-896)

```
815:  BT = chunk_size          # 64
816:  BC = 16
821:  Aqk  = torch.empty(B, T, H, BT, dtype=k.dtype)
822:  # Akk must be zero-initialized - kernel only writes lower triangular.
823:  Akk  = torch.zeros(B, T, H, BT, dtype=k.dtype)
825:  Akkd = torch.empty(B, T, H, BC, dtype=torch.float32)   # diagonal BC x BC blocks
```

### 2.3 Step 1 — intra-chunk score matrices

Two mutually exclusive paths, selected by `safe_gate`.

**(a) default path (`safe_gate=False`, lines 849-861)** →
`chunk_gdn2_fwd_intra_token_parallel` (205-242) → kernel
`chunk_gdn2_fwd_kernel_intra_token_parallel` (113-202). One program per token.
Index arithmetic: `i_c = i_t // BT`, `i_s = (i_t % BT) // BC`, `i_tc = i_c*BT`,
`i_ts = i_tc + i_s*BC` (155-158).

```
178-181: b_q = q[i_t];  b_k = k[i_t];  b_g = g[i_t];  b_b = b[i_t]      (all → f32)
183:     b_k = b_k * b_b                              # erase gate folded into key ONCE
185:     for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
189-190:     b_kj = k[j];  b_gj = g[j]
192:         b_kgj = b_kj * exp2(b_g - b_gj)          # decay weight t <- j
194:         b_kgj = where(m_k[None,:], b_kgj, 0.0)
196:         b_Aqk = sum(b_q   * b_kgj, axis=1) * scale         # [BH]  -> Aqk[i_t, j % BT]
197:         b_Akk = sum(b_k   * b_kgj, axis=1) * where(j < i_t, 1.0, 0.0)   # -> Akkd[i_t, j - i_ts]
199-202:  store into Aqk (col j % BT) and Akkd (col j - i_ts)
```

This produces the **diagonal `BC x BC` blocks only** (the `Akkd` buffer) and the
diagonal `BC`-column blocks of `Aqk`. The `j < i_t` factor (197) is what makes `Akk`
strictly lower triangular.

**(b) `safe_gate=True` (828-848)** → kernel `chunk_gdn2_fwd_kernel_intra_sub_chunk`
(266-359). Blocked, per `BC=16` sub-chunk, with `b_gn` the decay at the sub-chunk's
midpoint:

```
320:     b_gn = g[i_ti + min(BC//2, T - i_ti - 1)]      # or gather(..., axis=0)
326:     b_gm = b_g - b_gn
327:     b_gq = where(m_c[:,None], exp2( b_gm), 0.)      # [BC, BK]
328:     b_gk = where(m_c[:,None], exp2(-b_gm), 0.)      # [BC, BK]
330:     b_kgt = trans(b_k * b_gk)                       # [BK, BC]
332:     b_bk  = (b_b * b_k)                              # erase gate on the query side
334:     b_Aqk = dot(b_q  * b_gq, b_kgt) * scale          # [BC, BC], includes diagonal
335:     b_Akk = dot(b_bk * b_gq, b_kgt)                  # [BC, BC]
338-340: m_Aqk = o_i[:,None] >= o_i[None,:]              # inclusive
        m_Akk = o_i[:,None] >  o_i[None,:]              # strict
        m_I   = o_i[:,None] == o_i[None,:]
342-348: mask and store into Aqk and Akkd
```

followed by a 16-step serial inverse of the diagonal block (352-359):

```
352:     b_Ai = -b_Akk
353:     for i in range(2, min(BC, T - i_ti)):
354:         b_a  = -Akkd[i_ti + i]          # row i of the lower triangle
355:         b_a  = where(o_i < i, b_a, 0.)
356:         b_a += sum(b_a[:,None] * b_Ai, 0)          # rank-1 column update
357:         b_Ai = where((o_i == i)[:,None], b_a, b_Ai)
358:     b_Ai += m_I                                   # + I  =>  (I + T)^-1
359:     store b_Ai
```

With `safe_gate=True` this inverted block is the **final** diagonal; step 2 skips its
own substitution (see the `if not USE_SAFE_GATE` at 560).

### 2.4 Step 2 — inter-sub-chunk blocks + the triangular solve

Kernel `chunk_gdn2_fwd_kernel_inter_solve_fused` (388-657). The 64-wide chunk is cut
into four 16-wide sub-chunks (421-424: `i_tc0..i_tc3`).

Off-diagonal blocks, accumulated over K-blocks (454-533). Notation:
`gnN = g` at row `i_tcN` for the current K-column, `gqnN = exp2(gN - gnN)`.

```
460-461: b_k0, b_g0  loaded at i_tc0
473:     b_gn1 = g[i_tc1, k]                                     # [BK]
475:     b_gqn = where(m_tc1[:,None], exp2(b_g1 - b_gn1[None,:]), 0)   # [BC, BK]
477:     b_kgt = trans(b_k0 * exp2(b_gn1[None,:] - b_g0))         # [BK, BC]
479:     b_bk1 = b_b1 * b_k1
480:     b_Aqk10 += dot(b_q1  * b_gqn, b_kgt)
481:     b_Akk10 += dot(b_bk1 * b_gqn, b_kgt)
-- same shape for the other five off-diagonal pairs --
495:     b_gqn2 = where(m_tc2[:,None], exp2(b_g2 - b_gn2[None,:]), 0)
496:     b_qg2  = b_q2 * b_gqn2 ;  497: b_bkg2 = (b_b2 * b_k2) * b_gqn2
499:     b_kgt  = trans(b_k0 * exp2(b_gn2 - b_g0))
500:     b_Aqk20 += dot(b_qg2, b_kgt)      501: b_Akk20 += dot(b_bkg2, b_kgt)
503:     b_kgt  = trans(b_k1 * exp2(b_gn2 - b_g1))
504:     b_Aqk21 += dot(b_qg2, b_kgt)      505: b_Akk21 += dot(b_bkg2, b_kgt)
-- third sub-chunk --
519:     b_gqn3 = where(m_tc3[:,None], exp2(b_g3 - b_gn3[None,:]), 0)
520-521: b_qg3 = b_q3 * b_gqn3 ; b_bkg3 = (b_b3 * b_k3) * b_gqn3
523:     b_kgt = trans(b_k0 * exp2(b_gn3 - b_g0))   ->  524-525  A(3,0)
527:     b_kgt = trans(b_k1 * exp2(b_gn3 - b_g1))   ->  528-529  A(3,1)
531:     b_kgt = trans(b_k2 * exp2(b_gn3 - b_g2))   ->  532-533  A(3,2)
536-549: store the six Aqk off-diagonal blocks, each multiplied by `scale`
```

Note the asymmetry, and it is the GDN-2 point: the `Aqk` row operand is raw `q`
("for Aqk we use q (no absorption)", line 478), while the `Akk` row operand is
`b ⊙ k` (479, 497, 521). `b` never enters `Aqk`.

Diagonal blocks are read back from `Akkd` (551-558) and, unless `USE_SAFE_GATE`,
inverted in place by the same serial substitution as §2.3(b) (560-593):

```
560:  if not USE_SAFE_GATE:
561-562:   m_A = o_i[:,None] > o_i[None,:] ;  m_I = o_i[:,None] == o_i[None,:]
564-567:   b_Ai00..33 = -where(m_A, b_Ai00..33, 0)
569-573:   for i in range(2, min(BC, T - i_tc0)):            # sub-chunk 0
               b_a00 = -Akkd[i_tc0 + i]; b_a00 = where(o_i < i, b_a00, 0.)
               b_a00 += sum(b_a00[:,None] * b_Ai00, 0)
               b_Ai00 = where((o_i == i)[:,None], b_a00, b_Ai00)
574-578:   same, offsets i in [BC+2, 2*BC),      masked o_i < i - BC
579-583:   same, offsets i in [2*BC+2, 3*BC),    masked o_i < i - 2*BC
584-588:   same, offsets i in [3*BC+2, 4*BC),    masked o_i < i - 3*BC
590-593:   b_Ai00..33 += m_I                       # + I
```

Then the 4x4 block inverse merge (595-632), all dots at
`input_precision=SOLVE_TRIL_DOT_PRECISION`:

```
598:  b_Ai10 = -dot(dot(b_Ai11, b_Akk10), b_Ai00)
603:  b_Ai21 = -dot(dot(b_Ai22, b_Akk21), b_Ai11)
608:  b_Ai32 = -dot(dot(b_Ai33, b_Akk32), b_Ai22)

614:  b_Ai20 = -dot(b_Ai22, dot(b_Akk20, b_Ai00) + dot(b_Akk21, b_Ai10))
620:  b_Ai31 = -dot(b_Ai33, dot(b_Akk31, b_Ai11) + dot(b_Akk32, b_Ai21))
626:  b_Ai30 = -dot(b_Ai33, dot(b_Akk30, b_Ai00)
                             + dot(b_Akk31, b_Ai10)
                             + dot(b_Akk32, b_Ai20))
```

Result: the full `BT x BT` `Akk_inv`, stored at 637-657 as the 16 blocks
`Akk00,10,11,20,21,22,30,31,32,33` into the zero-initialized `Akk` buffer.
This matrix is `A` in the rest of the pipeline.

### 2.5 Step 3 — WY auxiliaries `w` and `u`

`recompute_w_u_fwd_gdn2_kernel` (694-786), launched by `recompute_w_u_fwd_gdn2`
(899-950, `BK = 64`, `BV = 64` at 917-918, `BT = A.shape[-1]` at 916).

```
728-730: b_A = A[i_t*BT : (i_t+1)*BT, 0:BT]              # [BT, BT]

# ---- u : H-term, value axis, BV block of V
732-738: b_v, b_wg  = v[.., i_v*BV:..], wg[.., i_v*BV:..]
741:     b_vb = (b_v * b_wg)
742:     b_u  = dot(b_A, b_vb)                            # [BT, BV]
743:     store u

# ---- w : P-term, key axis, BK block of K
752-754: b_k, b_b = k[.., i_k*BK:..], b[.., i_k*BK:..]
754:     b_kb = b_k * b_b
758-759: b_gk = gk[.., i_k*BK:..] ;  b_kb *= exp2(b_gk)     # b ⊙ k ⊙ exp2(gk)
785:     b_w  = dot(b_A, b_kb)                             # [BT, BK]
786:     store w
```

Matching the comment at 884 and the GDN-2-vs-KDA block at 669-677:

```
KDA:  u = A @ (beta * v)            w = A @ (beta * exp(gk) * k)
GDN2: u = A @ (wg * v)              w = A @ (b * exp(gk) * k)          # 674-675
```

Two side products, both gated on the caller passing the tensor:

```
761-768:  STORE_QG ->  qg = q * exp2(b_gk)                 # b_q * exp2(b_gk)
770-783:  STORE_KG ->  gn = gk at last row of the chunk (771: min(i_t*BT+BT, T) - 1)
                         b_kg = b_k * where(t < T, exp2(b_gn[None,:] - b_gk), 0)
```

In the production path `q` is passed as `q if disable_recompute else None` (891), so
`STORE_QG` is normally **false** and `qg` is `None` (926, 1045-1047). `STORE_KG` is
true whenever `gk` is given, i.e. always (927).

### 2.6 Steps 4 and 5 — inter-chunk scan and output: **NOT STATED**

```
1017-1029:  h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
                k=kg, w=w, u=u, gk=g, initial_state=..., output_final_state=...,
                cu_seqlens=..., cu_seqlens_cpu=..., chunk_indices=...,
                use_exp2=True, transpose_state_layout=...)
1031-1043:  o = chunk_gla_fwd_o_gk(
                q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale,
                cu_seqlens=..., chunk_size=64, chunk_indices=...,
                use_exp2=True, transpose_state_layout=...)
```

Both are imports from `flash-linear-attention` (line 83:
`from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h`;
line 65: `from fla.ops.gla.chunk import chunk_gla_fwd_o_gk`). **The chunk-level state
recurrence and the output kernel are not in this repository.** Their algebra is
**NOT STATED** here and must be read from `fla` at whatever commit the build picks
(unpinned git, `Dockerfile:37-39`).

The same holds for the backward halves the file delegates: `chunk_gated_delta_rule_bwd_dhu`
(83), `chunk_kda_bwd_dAv` (84), `kda_gate_bwd` (85).

### 2.7 Backward — the WY vector-Jacobian product

`chunk_gdn2_bwd_kernel_wy_dqkg_fused` (1244-1417). The file states the GDN-2 rule in
prose at 1221-1222:

```
  dA += dU @ (w * V)^T              # write gate, value axis
  dA += dW @ (b * exp(gk) * K)^T    # erase gate, key axis
```

and explains at 1214-1220 why a scalar post-scale cannot be used: with scalar `beta`
the contribution factors as `dU @ (beta*V)^T = beta * (dU @ V^T)`, but `b` and `w` are
row diagonals on two different axes, so they are baked in directly.

Code, in order:

```
1315-1316:  p_A = block_ptr(A, (BT,T), (1, H*BT), (0, i_t*BT), (BT,BT), order=(0,1))
            b_A = load(p_A)          # NOTE: b_A is A TRANSPOSED, [col, row]
1331-1332:  b_gn = g at (min(T, i_t*BT+BT) - 1)                      # [BK]
1353-1356:  b_h = h block;  b_dh = dh block;  b_v_new = v_new;  b_do = do;  b_dv = dv
1358:       b_dgk      += sum(b_h * b_dh, axis=0)                   # [BK]
1359:       b_dq       += dot(b_do, b_h)
1360:       b_dk       += dot(b_v_new, b_dh)
1361:       b_dw_flow  += dot(b_dv, b_h)
1364:       if i_k == 0:                                            # once per V-block
1371-1372:     b_wg = wg ;  b_dA += dot(b_dv, trans(b_v * b_wg))     # WRITE gate into dA
1374:         b_dvb      = dot(b_A, b_dv)                            # A^T @ dV
1375-1376:     b_dv2 = b_dvb * b_wg ;  b_dw_gate = b_dvb * b_v
1381-1382:  b_gk_exp = exp2(b_g) ;  b_gb = b_gk_exp * b_b
1383:       b_dgk    *= exp2(b_gn)
1384:       b_dq     *= b_gk_exp * scale
1385:       b_dk     *= where(m_t[:,None], exp2(b_gn[None,:] - b_g), 0)
1387:       b_kg      = b_k * b_gk_exp
1389:       b_dw_flow = -b_dw_flow.to(b_A.dtype)                      # negated here
1390:       b_dA     += dot(b_dw_flow, trans((b_kg * b_b)))          # ERASE gate into dA
1392:       b_dkgb    = dot(b_A, b_dw_flow)                           # A^T @ (-dW)
1394:       b_db      = b_dkgb * b_kg
1399-1400:  b_kdk = b_k * b_dk ;  b_dgk += sum(b_kdk, axis=0)
1401:       b_dg = b_q * b_dq - b_kdk + m_last[:,None] * b_dgk + b_kg * b_dkgb * b_b
1402:       b_dk = b_dk + b_dkgb * b_gb
```

Then the symmetrised dA, which is what makes the solve chain differentiable
(1411-1417):

```
1411:  m_A = (o_t[:,None] > o_t[None,:]) & (m_t[:,None] & m_t)
1412:  b_dA = where(m_A, b_dA, 0)
1413:  b_dA = dot(b_dA, b_A)          # dA @ A^T
1414:  b_dA = dot(b_A, b_dA)          # A^T @ dA
1415:  b_dA = where(m_A, -b_dA, 0)
1417:  store dA
```

Intra-chunk backward for `dq, dk, db, dg` is `chunk_gdn2_bwd_kernel_intra`
(1441-1708); it consumes `dAqk, dAkk` and reduces the decay gradient by a reverse
cumulative sum across the chunk (1423-1426). Its elementwise body was not
transcribed here — it is not part of the forward `chunk_wy_forward` algebra.

---

## 3. State layout and scan

### 3.1 The per-token recurrence (the ground truth, quoted)

`lit_gpt/gdn2_ops/fused_recurrent_gdn2.py:18-23`:

```
    S <- Diag(alpha) * S                  # channel-wise decay
    v_new = (w * v) - (b * k)^T S        # gated write minus gated read
    S <- S + k (v_new)^T                 # rank-one write
    o = S^T q                             # output read
```

The same recurrence in closed form appears twice —
`chunk_gdn2.py:13-15` and `lit_gpt/gdn2.py:57-58`:

```
    S_t = (I - k_t (b_t * k_t)^T) Diag(alpha_t) S_{t-1} + k_t (w_t * v_t)^T
```

`b_t ∈ [0,1]^{d_k}` is the channel-wise erase gate on the key axis, `w_t ∈ [0,1]^{d_v}`
is the channel-wise write gate on the value axis, `alpha_t` is the channel-wise decay
(`chunk_gdn2.py:17-19`). "Setting `b_t` and `w_t` to a shared scalar broadcast recovers
KDA; further collapsing `alpha_t` to a scalar recovers Gated DeltaNet" (20-22).
`gdn2.py:62`: "Setting `b_t = beta_t · 1` and `w_t = beta_t · 1` recovers KDA exactly."

**Factual observation, not a fix:** the two printed forms order the decay differently
relative to the erase term — the closed form applies `Diag(alpha_t)` to
`(I - k b^T) S_{t-1}`, the operational form applies it to `S` first and then erases.
The repo contains no comment reconciling them. Both are quoted verbatim; neither is
asserted here to be the intended semantics.

### 3.2 State indexing (per-token kernel)

```
 55-56:  Each program owns one (sequence, value-head, K-block, V-block) tile
 60-62:  State layout selectable via TRANSPOSE_STATE: default [K, V]; transposed [V, K]
125-134:  pid -> (i_k, i_v, i_n, i_hv);  i_h = i_hv // (HV // H)     # GVA: v-head -> k-head
160-168:  TRANSPOSE_STATE ? b_h = zeros([BV, BK]) : zeros([BK, BV])   # f32, in registers
169-191:  b_h += load(h0 + (i_n*HV + i_hv)*K*V + o_k[:,None]*V + o_v[None,:])   # [N,HV,K,V]
193:      for i_t in tl.range(0, T, num_stages=num_stages):          # token-serial carry
341-343:  final_state = new_empty(N, HV, V, K) if transpose else (N, HV, K, V), f32
```

Per-token body, `fused_recurrent_gdn2.py:193-244`:

```
194-202:  load q,k,v ; 199-200 l2norm with +1e-6 ; 201 b_q = b_q * scale
204-216:  gate-in-kernel:
             b_A = A_log[i_h]                                        # per head
             if HAS_DT_BIAS: b_g = b_g + dt_bias[i_h*K + o_k]        # per channel
             USE_LOWER_BOUND:  b_gk = lower_bound * sigmoid(exp(b_A) * b_g)
             else:              b_gk = -exp(b_A) * softplus(b_g)
         else: b_gk = b_g                                           # pre-computed
219-222:  b_h *= exp(b_gk[:, None])                                 # DECAY
224-225:  b_bk = b_b * b_k                                           # b ⊙ k
230-232:  erase_d = sum(b_h * b_bk[:, None], 0)                       # [BV]  (or [None,:], 1)
234-235:  b_v_new = b_w * b_v - erase_d                              # (w ⊙ v) - (b ⊙ k)^T S
242-243:  b_h += b_k[:, None] * b_v_new[None, :]                      # RANK-ONE WRITE
243:      b_o = sum(b_h * b_q[:, None], 0)
```

### 3.3 How the chunked form differs

What is visible in this repo:

1. The scan is **chunk-level, not token-level**. Chunks of `BT = 64` are independent
   units; the per-token loop is replaced by the WY auxiliaries `w = A (b ⊙ k ⊙ exp2(gk))`
   and `u = A (w ⊙ v)` (§2.5), which are `64 x K` and `64 x V` dense matmuls.
2. The channel-wise gate `b` is **absorbed into the key tile before the dot products**
   — stated as the defining change at 96-97: "The erase gate b is folded into the key
   tile before the dot product, which is the only GDN-2-specific change relative to the
   gated delta rule." Concretely `b_k = b_k * b_b` at 183 and `b_bk = b_b * b_k` at 332.
3. The decay is applied by **pre-multiplying tiles by `exp2` of in-chunk cumsum
   differences**, never as an explicit state scaling. Inside a chunk all pairwise
   weights `exp2(g_t - g_j)` (192, 475-533) are materialised as tile factors.
4. The tail decay `kg = k ⊙ exp2(g_n - gk)` (770-783) is the per-token decay's
   contribution at the chunk boundary, handed to the shared chunk kernel.
5. The state is not carried in registers across tokens; it is carried as the
   `initial_state` / `final_state` pair at chunk granularity, `fp32`, shape `[N, H, K, V]`
   (1100-1101, 1150).

What is **NOT STATED** here: the chunk-to-chunk state update itself, the meaning of
the `w` vs `u` split inside `chunk_gated_delta_rule_fwd_h`, and the intra-chunk output
formula inside `chunk_gla_fwd_o_gk`. Those are in `fla` (§2.6). Do not infer them from
this file.

---

## 4. Every constant the code sets

### 4.1 Tiling and schedule

| constant | value | line(s) |
|---|---|---|
| `chunk_size` / `BT` | **64** | `chunk_gdn2.py` 2055 (authoritative, set in the autograd forward), 214, 804, 815, 966, 1725, 1799, 1871; docstring 25-26 |
| sub-chunk `BC` | **16** | 215 (`sub_chunk_size: int = 16`), 816 (`BC = 16`) |
| `BT` in `recompute_w_u` | `A.shape[-1]` (= 64) | 916 |
| `BK` in `recompute_w_u` | **64** | 917 |
| `BV` in `recompute_w_u` | **64** | 918 |
| `BK` autotune (solve) | {32, 64} | 381 |
| `BK` autotune (bwd WY) | {32, 64} | 1234 |
| `BV` autotune (bwd WY) | {32, 64} | 1235 |
| `BH` autotune | {1, 2, 4, 8} | 106 |
| `num_stages` | {2, 3, 4} | 260, 688, 1237 |
| `num_warps` (solve) | {1, 2, 4} | 382 |
| `NUM_WARPS_WY` | [2, 4] on Hopper, else [2, 4, 8] | 79 |
| `NUM_WARPS_INTRA` / `_GENERIC` | [1, 2, 4] on Hopper, else [1, 2, 4, 8] | 80-81 |
| Hopper filter (WGMMA) | drop `BK == 32 and num_warps == 4` | 1238, noted 1225-1226 |
| `BK` (safe-gate path) | `next_power_of_2(K)` | 830 |
| `BK`,`BV` (recurrent kernel) | `BK = next_power_of_2(K)`, **`BV = 32`** | `fused_recurrent_gdn2.py` 327-328 |
| varlen binary search | `for _ in range(20)` | 136 |

### 4.2 Precision and eps

| item | value | line(s) |
|---|---|---|
| `SOLVE_TRIL_DOT_PRECISION` | `'tf32' if check_shared_mem() else 'ieee'` | 362 |
| `scale` (softmax/QK) | `k.shape[-1] ** -0.5` | 1179-1180; `fused_recurrent_gdn2.py` 321-322 |
| l2norm eps (recurrent kernel only) | `+1e-6`, inside the sqrt | `fused_recurrent_gdn2.py` 199-200 |
| l2norm eps (chunk path) | **NOT STATED** — `l2norm_fwd` is from `fla` (64) | 2060-2061 |
| `norm_eps` → `FusedRMSNormSwishGate` | **1e-5** | `lit_gpt/gdn2.py` 112, 96, 212 |
| `rel_tol` on the two expand_v integer checks | 1e-5 | `gdn2.py` 137, 147 |
| fp32 assertion on `initial_state` | `assert initial_state.dtype == torch.float32` | 1150 |
| key headdim ceiling | `assert k.shape[-1] <= 256` | 1166 |

**Factual observation.** The prose at 370-372 says the solve matmuls run at "fp32
(ieee) when shared memory allows it, tf32 otherwise", and the file's own comment at
598 calls the block merge "UNCHANGED vs KDA". Line 362 evaluates the condition the
other way round: `tf32` **if** `check_shared_mem()` **else** `ieee`. Quoted, not
reconciled.

### 4.3 Decay / gate bounds

| item | value | line(s) |
|---|---|---|
| `lower_bound` admissible range | `-5 <= lower_bound < 0` (hard raise) | 1157-1163 |
| `safe_gate` gate-value requirement | "requires gate values in [-5, 0)" | 1115-1116 |
| `b` typical range | `[0, 2]` (docstring) | 1093-1094 |
| `w` typical range | `[0, 1]` (docstring) | 1096-1097 |
| `b` lift to `[0, 2]` | `if allow_neg_eigval: b = b * 2.0` (opt-in, default `False`) | `gdn2.py` 118, 340-341; docstring 84-88 |
| cumsum base switch | `scale=RCP_LN2`, all decay via `exp2` | 67, 987, 996, 1027, 1041 |
| `RCP_LN2` value | **NOT STATED** (imported from `fla.ops.utils.constant`) | 67 |

### 4.4 Layer init (`lit_gpt/gdn2.py`)

| item | value | line(s) |
|---|---|---|
| `A_log` | `log(empty(H, f32).uniform_(1, 16))` — per head, so `exp(A_log) ∈ [1, 16]` | 198 |
| `A_log` weight decay | disabled (`_no_weight_decay = True`) | 199 |
| `dt` | `exp(rand(K) * (log 0.1 - log 0.001) + log 0.001)` = logU(1e-3, 1e-1) | 200-202 |
| `dt` clamp | `.clamp(min=1e-4)` | 202 |
| `dt_bias` | `dt + log(-expm1(-dt))` — softplus inverse | 203 |
| `dt_bias` weight decay | disabled | 205 |
| linear init | `xavier_uniform_(weight, gain=2 ** -2.5)`, bias zeroed | 225-227 |
| channel-wise log-decay | `g = -A_log.float().exp().repeat_interleave(head_k_dim) * softplus(f_proj(x).float() + dt_bias)` | 311-314 |
| erase gate | `b = b_proj(x).sigmoid()` | 319 |
| write gate | `w = w_proj(x).sigmoid()` | 320 |
| `use_gate_in_kernel` in the layer | **`False`** — g is built in fp32 in the layer, not in the kernel | 357, 373 |
| short conv | `kernel_size = conv_size = 4`, `bias = False`, `activation = "silu"`, on q/k/v | 108, 162-180 |

### 4.5 Layer defaults (`lit_gpt/gdn2.py:101-113`)

`hidden_size = 2048`, `expand_v = 1`, `head_dim = 128`, `num_heads = 16`,
`num_v_heads = None` (→ `num_heads`), `mode = "chunk"`, `use_short_conv = True`,
`allow_neg_eigval = False`, `conv_size = 4`, `conv_bias = False`, `layer_idx = None`,
`norm_eps = 1e-5`.

- `head_k_dim = head_dim`; `head_v_dim = int(head_dim * expand_v)`; `key_dim = H*K`;
  `value_dim = HV*head_v_dim` (130-133).
- Short non-training sequences auto-switch to the recurrent kernel at
  `q_len <= 64` (270); `assert mode == "chunk"` during training (272).
- GVA: `q, k, g, b` are `repeat`ed across the value-head group (331-335).
- Shipped config `gdn2_1.3B`: `n_layer 18`, `n_embd 2304`, `n_head 18`,
  `block_size 4096`, `vocab_size 32000`, `gdn2_per_layer 1`, `norm_eps 1e-5`,
  `1_302_638_112` params; `swa_gdn2_1.3B` is the same with `gdn2_per_layer 2`,
  `1_300_314_384` params (`lit_gpt/config.py:112-131`, 132-141+).

---

## 5. Runnability verdict

**Verdict: NOT RUNNABLE in this environment as a self-contained artifact. It is
reference-reading-only here, and it is not even fully self-contained on any machine —
the core scan lives in an external package.**

What I checked, and what each check says:

1. **The repo is not self-contained.** The inter-chunk state recurrence and the output
   kernel are imports: `chunk_gated_delta_rule_fwd_h` from `fla.ops.common.chunk_delta_h`
   (`chunk_gdn2.py:83`) and `chunk_gla_fwd_o_gk` from `fla.ops.gla.chunk` (65).
   `fla` = `flash-linear-attention`, installed from an **unpinned git HEAD**:
   `pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
   --no-build-isolation` (`Dockerfile:37-39`, `requirements.txt:29-31`). Any
   reproduction attempt therefore depends on a third-party commit this repo does not
   record. This is the blocking finding, independent of hardware.

2. **Target architecture excludes this GPU.** `Dockerfile:3`
   `ENV TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"` — sm_80 (A100) and sm_90 (H100). The
   `+PTX` on 9.0 would in principle JIT forward to sm_120, but that is not a supported
   configuration for this image and would not cover the sm_80 SASS. `README.md:97`
   states the scaling result was measured "on a single H100".

3. **There is nothing to run.** No `tests/` directory in the tree (31 entries, listed
   in full). No checkpoint, no weights, no dataset, no pretrained artifact. The only
   entry point shipped is `scripts/tsz1024x4k_100B_swa_gdn2.sh`, an SLURM script for
   **4 nodes x 8 GPUs** (`#SBATCH --ntasks-per-node=8`, `--nodes=4`,
   `--gres=gpu:8`) driving `pretrain.py` from a private container image
   `IMAGE="/myroot/myimage.sqsh"` (line 25) on a private FineWeb-Edu path
   (`TRAIN_DATA=/data/fineweb-edu/data`, 28).

4. **Build cost is large even on a capable box.** `Dockerfile` installs, beyond the
   base image `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel` (1): `apache-tvm-ffi`,
   `torch-c-dlpack-ext`, `cloudpickle`, `ml-dtypes`, `psutil`, `z3-solver`,
   `nvidia-cutlass-dsl` (17-24), `tilelang==0.1.8` (26),
   `quack-kernels==0.3.4` (27), `causal-conv1d==1.6.1` (29), `mamba` from git with
   `MAMBA_FORCE_BUILD=TRUE` (31-33), `flash-attn==2.8.3` (35),
   `flash-linear-attention` from git (37-39), lightning/tensorboard/pyarrow/lm-eval
   (41-56). `torch==2.9.0` is pinned (`requirements.txt:8`, `pyproject.toml:38`),
   `requires-python >= 3.10` (`pyproject.toml:10`).

5. **The minimal import path is wider than it looks.** `gdn2.py` itself only needs
   torch, triton, einops and `fla` (33-42). But it is a module inside the `lit_gpt`
   package, and `lit_gpt/__init__.py:13` does `from lit_gpt.model import GPT`, while
   `lit_gpt/model.py:15-16` does `from flash_attn import flash_attn_func,
   flash_attn_varlen_func` at module top level. So even
   `from lit_gpt.gdn2 import GatedDeltaNet2` executes `__init__.py` and therefore
   requires a compiled `flash-attn`. Working around that means loading the file
   outside the package, at which point you are re-implementing the layer anyway.

6. **The kernels are Triton, and they are readable as reference.** They are pure
   `tl.dot` / `exp2` / `where` over `[BT=64] x [BT=64]` and `[BT] x [K]`/`[BT] x [V]`
   tiles with an autotune wrapper. The algebra in §2 and §3 is complete for the
   intra-chunk half; the inter-chunk half is not, and the gap is a package boundary,
   not an omission in my reading.

**Bottom line for the plan.** Cite §2.2-§2.5 and §3.1 as the spec for
`chunk_wy_forward` — those are the equations our crate has to match, and they are
transcribed above with line numbers. Do not plan on running the artifact, and do not
plan on reading the chunk-level scan here. If the chunk-level scan is needed, the
follow-up target is `sustcsonglin/flash-linear-attention`, pinned to a recorded commit,
and that pinning is the change that makes the oracle reproducible at all.

---

## 6. Open questions / could not read

- `paper/GDN2_paper.pdf` (486638 bytes) — not read. Everything in §2-§4 is from code,
  not from the paper; the paper may state the recurrence or the loss spikes with
  different wording than the docstrings.
- The chunk-level state recurrence and output formula — **NOT STATED** in this repo
  (external, see §2.6, §3.3, §5.1).
- `RCP_LN2`'s numeric value, `l2norm_fwd`'s eps, `check_shared_mem()`'s threshold,
  `kda_gate_chunk_cumsum`'s bounded-gate formula — all external, **NOT STATED** here.
- `chunk_gdn2_bwd_kernel_intra` (1441-1708) elementwise body — read only in
  outline; it is not part of the forward WY algebra.
- The `Akk`/`Akkd` naming: `Akk` holds the **inverse** `(I + T)^{-1}` (named `A`
  from §2.5 onward), `Akkd` holds the raw strictly-lower blocks. The code uses
  `Akk` for the inverse at 637-657 and as `A` at 728-730; there is no comment stating
  this rename. Worth confirming against the paper before mirroring the names.
- No `pip`/`nvidia-smi`/import was executed, per instructions. The runnability
  verdict is derived from the files listed above and nothing else.
