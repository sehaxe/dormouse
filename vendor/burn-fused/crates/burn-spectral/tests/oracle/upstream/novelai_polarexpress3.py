# this is polar express with fp16
from itertools import repeat
import torch

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    m, n = X.size(-2), X.size(-1)
    safety_factor = 16.0
    target_std = 16 / safety_factor / (n * (n + m)) ** 0.25  # experimentally found, maybe there's better
    X = X * (target_std / X.std(correction=0))
    X = X.to(torch.float16)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    a, b, c = hs[0]
    A = X @ X.mT
    AA = A @ A
    eighth_power_singular_values = torch.sum(AA**2, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    # inverse singular value upper bound
    isvub = 1.01 * torch.pow(eighth_power_singular_values + 1e-35, -0.125)  # 1e-56 causes torch compile error
    # print("X size post norm, P2", X.std() * isvub)
    # a <- a * isvub
    # b <- b * isvub^3
    # c <- c * isvub^5
    # it's possible a fp32-bf16 pointwise multiplication would make sense. But I don't know how
    B = (b * isvub**3).to(A.dtype) * A + (c * isvub**5).to(A.dtype) * AA
    X = (a * isvub).to(A.dtype) * X + B @ X  # X <- aX + bX^3 + cX^5

    for a, b, c in hs[1:]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
