r"""Quantized saved-tensor checkpointing: the backward reads 4/8-bit codes.

THE SEAM. `torch.autograd.graph.saved_tensors_hooks` is where autograd decides
what it KEEPS for the backward. [`QuantSavedHooks`] installs a pair of hooks
around a forward region that replaces every saved float tensor above
`min_numel` elements with a PACKED CODEC payload — MXFP4 (e2m1 elements, one
e8m0 power-of-two scale per 32, `mxfp.py`) or MXFP8 (e4m3 elements, same scale)
— and DEQUANTIZES it back at the backward edge. Memory win on everything
autograd saves: **3.76x** (mxfp4) or **1.94x** (mxfp8) against bf16, counted
exactly.

WHY THE EXISTING CODECS AND NOT A NEW ONE. `mxfp.quantize_mxfp4` /
`dequantize_mxfp4` / `quantize_mxfp8` / `dequantize_mxfp8` already pack to bytes
and already carry the OCP spec's rules, the RNE grid and the e8m0 conventions
(`mxfp.py`'s docstring names every one), plus `bytes_per_elements_*` for the
accounting. `act_quant.py`'s `int4`/`int8` are FAKE-QUANT — dequantized values
with no packing — so a second 4-bit packer here would be a second grid with the
3-level-defect failure mode this repo already paid for (AGENTS §3.7 act-quant).
This module therefore adds ZERO element codecs: it adds the padding rule, the
payload, the hooks and the accounting.

WHAT IS EXACT AND WHAT IS LOSSY, in one place (the whole point of the module):

* **The forward is untouched.** The hooks fire when autograd SAVES, never when
  it computes; no forward value depends on them. So a loss computed under the
  hooks is bit-identical to the same loss computed without them — measured, and
  pinned by test (`torch.equal`), not a bound.
* **The backward sees the dequantized value**, i.e. the saved tensor rounded
  onto the format's grid. Every gradient downstream of a packed save is
  therefore an estimate: finite, bounded (table below), not exact. This is the
  deliberate trade — bytes on device for a re-quantization of the backward's
  inputs. Nothing is recomputed, nothing is skipped, and no NaN can be
  introduced by the codec (`mxfp.py` refuses non-finite input LOUDLY, so a
  non-finite saved tensor stops the run instead of silently poisoning a scale).
* **Only the forward's values are exact.** A caller that needs exact gradients
  must not enable the arm; a caller that needs smaller saves accepts a
  re-quantized backward.

RELATIONSHIP TO SL1 (`experts.py`'s `moe_offload`, budget-savers §5). Both
install `saved_tensors_hooks` around the same kind of region; they trade
different things:

| | what it does to a saved tensor | cost | quality |
|---|---|---|---|
| SL1 host offload | COPIES it to host RAM, back at the backward edge | PCIe per save | exact |
| this module | REPLACES it with 4/8-bit codes, same device | codec compute | lossy |

They are orthogonal and composable, and the composition is a chained pack hook,
NOT nesting: torch keeps ONE hook pair per save (the innermost), so
`saved_tensors_hooks` nested inside `saved_tensors_hooks` means the outer pair
never sees the inner region's saves (measured on 2.13.0; `experts.py:102-111`
is the same caveat from the SL1 side, and `tests/test_act_ckpt.py` pins it).
The two cheap compositions, both exact on the SL1 side: (1) offload the CODES —
the packed payload is 3.76x smaller, so the same VRAM win costs 3.76x less PCIe
traffic; (2) quantize on the device where the backward runs and offload
everything else. Neither is built here (YAGNI until a lane wires one).

**THE CHECKPOINT INTERACTION, measured here on 2.13.0 and NOT obvious** (a
3-op body: `sin`, `exp`, `mul`): with the hooks wrapped AROUND a
`checkpoint(use_reentrant=False)` call, the pack hook fires for the checkpoint
node's INPUTS only (1 tensor of 4) and the body still runs twice (the recompute
happens) — the region's internals went to checkpoint's own innermost pair. With
the hooks INSIDE the checkpointed function, all 4 saves are packed AND the body
runs ONCE: because checkpoint's holder list stays empty, it has nothing to
recompute, so the region's activations live as codes instead of being
regenerated. The two mechanisms are therefore mutually exclusive WITHIN one
region (recompute or pack, whoever is innermost), and a `QuantSavedHooks`
wrapped around a region without reading the counter is exactly the SILENTLY
INERT shape `experts.py:102-111` records — `act_ckpt_packed` is the proof, and
`0` means the arm did not run. The trainer's `segment_len` path
(`loopblock.py:566`) is the call site this lands on.

THE ERROR BOUND (measured, this box, torch 2.13.0+cu129 CPU, fp32, seed 0,
`uv run python scripts/bench_act_ckpt.py` reproduces every number here; no
external oracle exists for these, so they are measurements, not a certificate.
The analytic shape is `|x − x_q| ≤ block_scale · (half the grid step at that
magnitude)`, `block_scale = 2^ceil(log2(amax/6)) ≤ amax/3`). `zeros` is the
fraction of elements the 4-bit grid flushes to exactly 0 — the heavy-tailed
cost the max-error column HIDES, because one huge block amax shrinks the ratio
while the small elements inside that block fall under the grid's first step:

| fixture, `[64, 128]` fp32, seed 0 | kind | max abs err / absmax | rms err / rms x | zeros |
|---|---|---|---|---|
| `randn` | mxfp4 | 1.21e-1 | 1.16e-1 | 11 % |
| `randn` | mxfp8 | 3.04e-2 | 2.68e-2 | 0 % |
| `randn · e^{2·randn}` | mxfp4 | 5.45e-2 | 9.99e-2 | **75 %** |
| `randn · e^{2·randn}` | mxfp8 | 2.94e-2 | 2.70e-2 | 0.2 % |
| tiny preset, batch 2, `min_numel=4096`, worst grad | mxfp4 | **2.40e-1** | — | — |
| same | mxfp8 | **6.28e-2** | — | — |

The tiny-preset row is the one that prices the trade on a real graph: the LOSS
is bit-identical (the forward is untouched, `torch.equal`), and the worst
relative parameter-gradient move is ~2.4e-1 at 4 bits (`loop.controller.weight`
on this preset).

**Weight saves are packed too**, and on a real model they are the big tensors:
autograd saves a linear's `weight.t()` like any other operand, so those nodes'
backwards read re-quantized weights. Measured on the tiny preset: 92 of the 383
saves the hook sees are parameter storages, 12 of which clear the
`min_numel=4096` gate (a tiny model's factors are 512 elements; the save COUNT
is batch-independent, only the sizes move). Measured, skipping every
parameter-storage save does not lower the worst gradient move (2.42e-1 vs
2.40e-1) — the worst case is activation-driven here — and the escape for a
caller that wants weights exact is to scope the hooks to the activation region
or set `min_numel` above the parameter sizes. The share is counted by
`tests/test_act_ckpt.py::test_o_parameter_storage_saves_ride_the_hook` and
printed by the bench.

THE COST, honestly: on this CPU the 4-bit arm's forward+backward measures
2.5-3x the hooks-off run and the 8-bit arm 1.0-1.8x (the bench prints both;
the spread is this box's load, not the codec's variance) — the codec is
Python-side torch ops over ~116 small tensors, and the hot spot inside it is
`mxfp.quantize_mxfp4`'s `torch.searchsorted` (measured 15 ms per 32 k elements
on this box). **No GPU number exists** — the card belongs to a training run —
so the device-side cost is unmeasured, and the trainer wiring should measure it
before trusting the ratio. One thing that measurement will find, and it is in
the CODEC, not in the hooks: `mxfp.py`'s guards are host reads
(`_check_finite` `mxfp.py:185`, `_e8m0_encode`'s range check `:141`,
`e8m0_decode`'s NaN check `:165`, and the two `assert bool(...all())`s at
`:222`/`:275`) — 2-3 device syncs per packed tensor, i.e. hundreds per forward
at this save count, which is the shape §1.3 of the port's rules forbids. The
fix belongs in `mxfp.py` (a device-side flag drained on the log cadence) and is
NOT taken here: this module reuses that codec rather than editing it.

V2, NOT BUILT HERE (the honest note the contract needs): for a QUANTIZED seam
the same idea is exact. When the forward already ran the quantizer — an
`act_quant.fake_quant_act` activation, a `tsct.factor_qat_project` weight — the
value autograd saves is already ON the format's grid, so packing it costs no
further error and the backward sees exactly what the forward saw. Wiring that
is a follow-up: it needs the seam to declare its format, which is a trainer
change, not a hook change.

WHAT THIS DOES NOT DO (each named so a reader does not have to guess):
* this module never recomputes anything itself (the checkpoint section above is
  about the OTHER mechanism's recompute), compresses no CHECKPOINT FILE (saves
  only), and moves nothing to host;
* **weight saves are packed too** — see the table's last row;
* fp16 saves are outside the exactness claim: e2m1/e4m3 values and e8m0 scales
  are exactly representable in bf16 and fp32, while fp16's range/subnormals can
  round or flush them. The port's activations are bf16 or fp32; an fp16 save is
  asserted finite on the way out rather than silently carried;
* no claim about CUDA-graph capture: the pack/unpack pair is Python, and a
  capture window's rules are a measurement nobody has taken here.
"""

from __future__ import annotations

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..instrument import Counters, loud
from .mxfp import (
    BLOCK,
    bytes_per_elements_mxfp4,
    bytes_per_elements_mxfp8,
    dequantize_mxfp4,
    dequantize_mxfp8,
    quantize_mxfp4,
    quantize_mxfp8,
)

#: The formats a saved tensor can be packed into. `none` installs the hooks and
#: does nothing (so the call site is unconditional); the two codecs are the OCP
#: MX rows of `mxfp.py` §5.2 Table 1, 4 and 8 bits per element.
KINDS = ("none", "mxfp4", "mxfp8")

#: The counter names. A zero is a statement of its own: `act_ckpt_packed=0` on
#: a run whose flag is on means the arm never fired (the placement trap).
_PACKED = "act_ckpt_packed"
_PASSTHROUGH = "act_ckpt_passthrough"
_ORIG_BYTES = "act_ckpt_orig_bytes"
_BF16_BYTES = "act_ckpt_bf16_bytes"
_PACKED_BYTES = "act_ckpt_packed_bytes"


def _check_kind(kind: str) -> None:
    """LOUD on anything that is not a [`KINDS`] literal, naming the escape."""
    if kind not in KINDS:
        loud(
            f"act_ckpt: kind={kind!r} is not a saved-tensor format. Known: "
            f"{list(KINDS)} — `mxfp4` (e2m1 + per-32 e8m0 scale, 3.76x off bf16) "
            f"and `mxfp8` (e4m3, same scale, 1.94x) are the two packed codecs in "
            f"`mxfp.py`; `none` installs the hooks and packs nothing. `int4`/"
            f"`int8` are NOT here: `act_quant`'s are fake-quant (no packing), and "
            f"a second 4-bit packer is a second grid. Escape: name one of "
            f"{list(KINDS)}."
        )


class Packed:
    """One saved tensor, packed: codes + scales + what the unpack needs.

    A CLASS, not a tuple, for the same reason `experts._OnHost` is one: the
    unpack hook must tell its own cargo from the values the pack hook passes
    through untouched, and the payload has to carry the shape, dtype and device
    the hook contract requires back (`saved_tensors_hooks`' documented
    contract: equal in value, size, dtype AND device).

    `codes`/`scales` are the codec's own outputs (`mxfp.py`), `numel` is the
    ORIGINAL element count — the padded tail (up to 31 zeros, one MX block) is
    storage, not data, and [`unpack_tensor`] slices it off.
    """

    __slots__ = ("kind", "codes", "scales", "shape", "dtype", "device", "numel")

    def __init__(
        self,
        kind: str,
        codes: torch.Tensor,
        scales: torch.Tensor,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        numel: int,
    ) -> None:
        self.kind = kind
        self.codes = codes
        self.scales = scales
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.numel = numel

    def nbytes(self) -> int:
        """The stored size: code bytes + scale bytes (uint8, so numel == bytes)."""
        return self.codes.nbytes + self.scales.nbytes

    def __repr__(self) -> str:  # pragma: no cover - a debugging convenience
        return f"Packed({self.kind}, {tuple(self.shape)}, {self.dtype}, {self.nbytes()} B)"


def _pad_to_block(n: int) -> int:
    """`n` rounded up to a multiple of the MX block width (0 for n == 0)."""
    return n + (-n) % BLOCK


def pack_tensor(t: torch.Tensor, kind: str) -> Packed | torch.Tensor:
    """Quantize+pack `t` — the pure codec, no hooks, no counters.

    The tensor is FLATTENED and the last dim is zero-padded up to a multiple of
    [`mxfp.BLOCK`] (32) before the codec, then the codes/scales are kept; the
    pad never enters the scales' arithmetic (a zero cannot raise a block amax,
    and the pad is storage the unpack slices off), which is the same padding
    argument `mxfp.fake_quant_mx` carries for the aligned case. Block
    membership is therefore over the flat sequence — a declared layout, like
    the nibble order, not a spec claim (§5.1 leaves the layout to the
    implementation).

    `kind="none"` returns `t` itself (identity, exact): there is nothing to
    pack, and the hook's passthrough path is the same object.

    Non-finite input is refused by the codec (LOUD, `mxfp._check_finite`): a
    NaN would poison the block amax and therefore the whole block's scale,
    which is a silent corruption with an obvious cause — the NaN firewall
    belongs upstream (ADR-0019).
    """
    _check_kind(kind)
    if kind == "none":
        return t
    assert t.is_floating_point(), f"pack_tensor packs float tensors, got {t.dtype}"
    n = t.numel()
    flat = t.detach().reshape(-1).to(torch.float32)
    pad = _pad_to_block(n) - n
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    if kind == "mxfp4":
        codes, scales = quantize_mxfp4(flat)
        expected = bytes_per_elements_mxfp4(_pad_to_block(n))
    else:
        codes, scales = quantize_mxfp8(flat)
        expected = bytes_per_elements_mxfp8(_pad_to_block(n))
    assert codes.nbytes + scales.nbytes == expected, (
        "the packed size must be the codec's own accounting"
    )
    return Packed(kind, codes, scales, t.shape, t.dtype, t.device, n)


def unpack_tensor(p: Packed | torch.Tensor) -> torch.Tensor:
    """Dequantize a [`Packed`] back to the original shape/dtype/device.

    A value the pack hook passed through (anything that is not a [`Packed`]) is
    returned AS IS — the identity, which is what makes the passthrough arm
    bit-identical to the no-hook run.

    The device leg is not optional (`saved_tensors_hooks`' contract, and the
    lesson SL1 records at `experts.py:112-115`): the codes live wherever the
    pack put them, and the backward kernel needs the tensor on the device the
    forward saved it from.
    """
    if not isinstance(p, Packed):
        return p
    n = p.numel
    if p.kind == "mxfp4":
        vals = dequantize_mxfp4(p.codes, p.scales, (_pad_to_block(n),))
    else:
        assert p.kind == "mxfp8", f"unknown packed kind {p.kind!r}"
        vals = dequantize_mxfp8(p.codes, p.scales, (_pad_to_block(n),))
    out = vals[:n].reshape(p.shape).to(p.dtype)
    if out.device != p.device:
        out = out.to(p.device)
    if p.dtype == torch.float16:
        # Outside the exactness claim (see the module docstring): fp16's range
        # can overflow to inf on a grid value that bf16/fp32 carry exactly.
        assert bool(torch.isfinite(out).all()), (
            "act_ckpt: the unpacked fp16 save is not finite — the block scale "
            "exceeded fp16's range. Escape: save bf16/fp32 (the port's "
            "activation dtypes), or keep the tensor out of the hooks."
        )
    return out


class QuantSavedHooks:
    """Context manager: pack every saved tensor above `min_numel` (COUNTED).

        with QuantSavedHooks("mxfp4", min_numel=4096, counters=counters):
            _, loss, _ = model(ids, targets)
        loss.backward()   # the unpack fires here, on the stored hooks pair

    `kind="none"` installs the same hooks and packs nothing, so a call site can
    be unconditional and the arm's cost is the flag's alone. The pack hook
    fires on EVERY save in either case — which is the point: the placement
    (inside a checkpointed region, per the module docstring) is proven by the
    counter, not by the flag.

    `counters` is an optional [`Counters`]; one is created if omitted and is
    always reachable as `self.counters`, so `memory_report(h.counters)` works
    either way.

    Safe to nest and to reuse: each `__enter__` installs a fresh hooks pair, and
    `__exit__` restores the previous one from a stack (so the same object can be
    entered twice; the innermost pair wins for a given save — see the module
    docstring's SL1 composition note).
    """

    def __init__(
        self,
        kind: str,
        *,
        min_numel: int = 4096,
        counters: Counters | None = None,
    ) -> None:
        _check_kind(kind)
        if min_numel < 0:
            loud(
                f"act_ckpt: min_numel={min_numel} is negative — it is an element "
                f"count (a tensor with FEWER elements than this is left exact). "
                f"Escape: 0 packs every tensor the hook sees."
            )
        self.kind = kind
        self.min_numel = int(min_numel)
        self.counters = Counters() if counters is None else counters
        self._ctxs: list[object] = []

    def _pack(self, t: object) -> object:
        """The pack hook: `Packed` for a packable save, the tensor itself else.

        Anything that is not a non-empty CPU/CUDA float tensor above `min_numel`
        is passed through untouched (and counted): a non-tensor, an index
        tensor, an off-device tensor, a small tensor — and an EMPTY one, which
        the MX codec refuses with a message about padding a last dim of 0.
        """
        if self.kind == "none" or not isinstance(t, torch.Tensor):
            return t
        if (
            not t.is_floating_point()
            or t.device.type not in ("cpu", "cuda")
            or t.numel() < self.min_numel
            or t.numel() == 0
        ):
            self.counters.count(_PASSTHROUGH)
            return t
        p = pack_tensor(t, self.kind)
        assert isinstance(p, Packed), "a non-`none` kind must pack"
        self.counters.count(_PACKED)
        self.counters.count(_ORIG_BYTES, t.numel() * t.element_size())
        self.counters.count(_BF16_BYTES, t.numel() * 2)
        self.counters.count(_PACKED_BYTES, p.nbytes())
        return p

    def _unpack(self, p: object) -> object:
        """The unpack hook: dequantize onto the saved device, passthrough as-is."""
        return unpack_tensor(p)  # type: ignore[arg-type]

    def __enter__(self) -> QuantSavedHooks:
        ctx = saved_tensors_hooks(self._pack, self._unpack)
        ctx.__enter__()
        self._ctxs.append(ctx)
        return self

    def __exit__(self, *exc: object) -> bool | None:
        assert self._ctxs, "exit without enter"
        return self._ctxs.pop().__exit__(*exc)


def memory_report(counters: Counters) -> dict[str, float]:
    """Packed vs original bytes and the ratios — the accounting, one place.

    `orig_bytes` is what the packed tensors ACTUALLY occupied
    (`numel · element_size`), `bf16_bytes` the same count at 2 B/element (the
    number the 3.76x/1.94x claims are against), `packed_bytes` the codec's own
    output. On a CPU run the originals are fp32, so the honest ratio against
    them is ~7.5x, not 3.76x — which is why both ratios are returned and the
    bench prints both.

    Both ratios are NaN when nothing was packed: a 0-byte payload has no ratio,
    and printing one would be the kind of number this repo deletes.
    """
    packed = counters.get(_PACKED_BYTES, 0)
    orig = counters.get(_ORIG_BYTES, 0)
    bf16 = counters.get(_BF16_BYTES, 0)
    return {
        "packed_tensors": counters.get(_PACKED, 0),
        "packed_bytes": packed,
        "orig_bytes": orig,
        "bf16_bytes": bf16,
        "ratio_orig": orig / packed if packed else float("nan"),
        "ratio_bf16": bf16 / packed if packed else float("nan"),
    }
