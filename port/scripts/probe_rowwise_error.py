"""Corrected rowwise `_scaled_mm`: error vs the fp32 reference + timing, live shape."""

import sys

sys.path.insert(0, "scripts")
import probe_fp8_kernel as P
import torch

x = P.make_pattern("row_outlier")
torch.manual_seed(0)
w = (torch.randn(P.N, P.K, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
ref = P.fp32_reference(x, w)
qx, sx = P.q_row(x)
qw, sw = P.q_row(w)
swT = sw.reshape(1, P.N).contiguous()
y = torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=swT, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
print("rowwise error vs fp32 ref:", P.err_stats(y, ref, row=0))
t = P.bench(lambda: torch._scaled_mm(qx, qw.t(), scale_a=sx, scale_b=swT, out_dtype=torch.bfloat16))
print("rowwise ms:", t)
print("bf16 ms:", P.bench(lambda: x @ w.t()))
