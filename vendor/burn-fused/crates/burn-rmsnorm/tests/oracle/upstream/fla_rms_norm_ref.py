# VERBATIM source of fla.modules.layernorm.rms_norm_ref, extracted with
# inspect.getsource() from the file that was EXECUTED to build
# ../fixtures/rmsnorm_oracle.txt.  sha256 of that file:
#   e78b729bcba29b30d8d6ddb6ce17d465261a6462be1363a04a6b3dc51c6d5c6f
# Upstream: https://github.com/fla-org/flash-linear-attention
#   a PyPI wheel with NO git revision, so the sha256 above is the only handle
#   for the exact bytes these came from.
# Do not edit; re-extract with tests/oracle/gen_rmsnorm_oracle.py --dump.

def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False,
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)
