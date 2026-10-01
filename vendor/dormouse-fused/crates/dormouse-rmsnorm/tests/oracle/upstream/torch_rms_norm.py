# VERBATIM source of the two PyTorch entry points that were EXECUTED to
# build ../fixtures/rmsnorm_oracle.txt, extracted with inspect.getsource()
# from torch 2.14.0+cpu (https://github.com/pytorch/pytorch) at tag v2.14.0.  The arithmetic they dispatch to
# is COMPILED -- aten/src/ATen/native/layer_norm.cpp::rms_norm, shipped in
# the same wheel -- and is not quotable here, which is why the row records
# the version and the wheel source rather than a C++ excerpt.

# ---- torch.nn.functional.rms_norm --------------------------------
def rms_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    r"""Apply Root Mean Square Layer Normalization.

    See :class:`~torch.nn.RMSNorm` for details.
    """
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            rms_norm, (input, weight), input, normalized_shape, weight=weight, eps=eps
        )
    return torch.rms_norm(input, normalized_shape, weight, eps)

# ---- torch.nn.RMSNorm.forward -----------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
