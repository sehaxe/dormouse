r"""The packed QAT export: the ternary release as ONE file.

WHAT IT IS. After QAT (the factor rung on, `factor_qat_project`'s projection
baked into the factors), the model is exported to a single file whose TSCT
factor matrices are b1.58 ternary — 2-bit codes, four per byte, one fp16
absmean scale per factor — and whose `s`/`gate_s`/`up_s` diagonals are fp16.
Everything else (KDA projections, embeddings, norms, router, controller,
Engram tables) stays bf16. The loaded model's forward reads the values
`factor_qat_project(t, "ternary", alpha=1.0)` read on the original masters, so
the export is a FORMAT, not a second quantization: `load_qat` constructs with
`factor_qat="none"` because the projection is already IN the numbers, and
projecting them again would re-scale by the fraction of non-zero codes
(`ternarize` is not idempotent — `tests/test_ternary.py` pins that).

WHAT IT BUYS, AND WHAT IT DOES NOT. Disk and RAM at rest: a ternary factor is
`n/4 + 2` bytes against bf16's `2n`, i.e. ~8x on the packed matrices. It does
NOT buy GPU speed: this port has no int8/LUT ternary kernel, so the forward
dequantizes at LOAD time and every matmul runs exactly as before (`mm_mixed`'s
bf16 lane). The file is a distribution/at-rest artifact; a fast ternary
forward needs the kernel lane that does not exist yet.

THE CONTAINER (`torch.save`, `weights_only`-loadable, one dict):

    {
      "format":  "dormouse-qat-export",   # magic; a foreign file refuses
      "version": 1,
      "mode":    "ternary" | "mxfp4",     # the rung the factors were packed at
      "tensors": {key: record | tensor},  # state-dict keys, `_orig_mod.` stripped
    }

`record` (a packed factor matrix):
    ternary: {"rung": "ternary", "codes": uint8 [n/4], "scale": fp16 0-dim,
              "shape": (d0, d1), "n": int}
    mxfp4:   {"rung": "mxfp4", "codes": uint8 [padded_n/2],
              "scales": uint8 e8m0 [padded_n/32], "shape": (d0, d1),
              "pad": int, "n": int}
tensor: fp16 for the diagonals, bf16 for every other float, unchanged for
non-float buffers. A `record` never appears at a non-factor key, and this
format is NOT the resume format: `load_ckpt` wants fp32 masters and the
optimizer state, which this file does not carry (it is the release artifact).

THE ROUNDINGS, both bounded, both stated where a caller can check them: the
scale is stored fp16, so `±fp16(α)` costs ≤ 2^-11 relative against `α`; the
factors are loaded as bf16, costing ≤ 2^-9 relative. `verify_roundtrip`
measures the LOGITS' max abs difference between the reference forward (the
projection, fp32 masters) and the loaded model — the number that says whether
the file is the same network.

LOUD refusals: a non-float master, an empty tensor, a numel not divisible by 4
(pack_ternary packs 4 codes per byte and does not silently pad), a non-finite
value (`ternarize`'s own guard), a foreign/truncated file, and a key-set
mismatch — including the host-table case, where `engram.tables` is a `.ngram`
sidecar rather than a parameter and the error names that.
"""

from __future__ import annotations

import dataclasses
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import loud
from .mxfp import BLOCK, bytes_per_elements_mxfp4, dequantize_mxfp4, quantize_mxfp4
from .ternary import _alpha
from .tsct import _ternary_dequant, _ternary_quant, is_factor_key

#: The container's magic and version — `_read_container` refuses anything else.
FORMAT = "dormouse-qat-export"
VERSION = 1

#: The factor rungs this module packs. `ternary` is the release; `mxfp4` is the
#: same report with `mxfp.quantize_mxfp4` as the codec (the `factor_qat=mxfp4`
#: rung's storage).
MODES = ("ternary", "mxfp4")

#: The diag scales — fp16, never packed: they ARE the factorization's diagonal
#: (`tsct.py`: `s` is the scale, and a quantized scale would multiply the whole
#: row).
_SCALE_LEAVES = ("s", "gate_s", "up_s")

#: The report's groups, in print order. `factors` is the packed matrices (plus
#: their fp16 scales), `scales` the fp16 diagonals, `bf16` every other float,
#: `other` the non-float buffers.
_GROUPS = ("factors", "scales", "bf16", "other")


def pack_ternary(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    """`ternarize(t)` → (uint8 codes 4/byte, absmean scale) — the file's codec.

    The codes are `ternary.ternarize`'s output signs (that function is reused,
    not re-derived): 0 for the dead zone, 1 for `+α`, 2 for `−α`, 3 unused.
    The scale is the same fp32 `mean(|W|)` `ternarize` divides by, so
    [`unpack_ternary`] is BIT-EXACT to `ternarize(t)` for the same `t`.

    LOUD on what it cannot pack: a non-float master, an empty tensor (the
    absmean of nothing is NaN, and `ternarize` would return NaN codes), and a
    numel not divisible by 4 (the tail would need silent padding; the escape is
    to pad the last dim before calling).
    """
    t = t.detach()
    if not t.is_floating_point():
        loud(
            f"pack_ternary: masters are float tensors, got {t.dtype}. A ternary "
            "codebook is a projection of a real-valued master; an integer tensor "
            "has no absmean to divide by. Escape: cast the master to float."
        )
    if t.numel() == 0:
        loud(
            "pack_ternary: an empty tensor has no absmean (mean of nothing is NaN, "
            "so every code would be NaN). Escape: do not export an empty factor."
        )
    if t.numel() % 4:
        loud(
            f"pack_ternary: {t.numel()} elements is not a multiple of 4 (shape "
            f"{tuple(t.shape)}). The codec packs exactly 4 codes per byte and does "
            "not silently pad a tail. Escape: pad the last dim to a multiple of 4."
        )
    codes, _ = _ternary_quant(t)  # the codec; its own alpha is not this one
    return codes, float(_alpha(t))


def unpack_ternary(
    codes: torch.Tensor,
    scale: float,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of [`pack_ternary`]: `{0, ±scale}` in `dtype`, `shape` applied.

    Bit-exact to `ternarize(t)` when `scale == float(ternary._alpha(t))` and
    `dtype == t.dtype` (the hand fixture in `tests/test_export.py` pins it).
    """
    assert codes.dtype == torch.uint8, f"codes are uint8, got {codes.dtype}"
    n = math.prod(shape)
    assert codes.numel() * 4 == n, (
        f"codes hold {codes.numel() * 4} slots but shape {tuple(shape)} has {n}: "
        "packing is exactly 4 codes per byte and unpack does not guess a tail"
    )
    # A trust-boundary check: the file is on disk, and a corrupted code 3 would
    # silently decode to 0 (it is not a level of the codebook) — refuse instead.
    fields = torch.stack([codes & 3, (codes >> 2) & 3, (codes >> 4) & 3, (codes >> 6) & 3])
    assert int(fields.max()) <= 2, (
        "a 2-bit code is 3, which is not a ternary level {0, +α, −α}: the codes "
        "are corrupt (or from another codec). Escape: re-export the file"
    )
    vals = _ternary_dequant(codes, float(scale), n)
    return vals.reshape(shape).to(dtype)


def pack_mxfp4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`factor_qat_project(t, "mxfp4", 1.0)`'s values as the MX codec's storage.

    The rung's projector pads the LAST dim to a multiple of `mxfp.BLOCK` (32)
    and slices back (`fake_quant_mx`), so this does the same and returns the pad
    alongside the `(packed nibbles, e8m0 scales)`. The pad is per-row zeros, so
    it never moves a real element's block scale — the padded round trip is
    exactly the projector's values (pinned in `tests/test_export.py`).
    """
    t = t.detach()
    if not t.is_floating_point():
        loud(f"pack_mxfp4: masters are float tensors, got {t.dtype} (see pack_ternary)")
    if t.dim() == 0 or t.shape[-1] == 0:
        loud(
            f"pack_mxfp4: MX blocks run along the LAST dim and this tensor is "
            f"{tuple(t.shape)}. Escape: pass a factor matrix with a non-empty last dim."
        )
    pad = (-t.shape[-1]) % BLOCK
    tp = F.pad(t, (0, pad)) if pad else t
    packed, scales = quantize_mxfp4(tp)  # non-finite input refuses loudly there
    return packed, scales, pad


def unpack_mxfp4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
    pad: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of [`pack_mxfp4`]: the e2m1 grid × e8m0 block scales, pad cut."""
    padded = (*shape[:-1], shape[-1] + int(pad))
    vals = dequantize_mxfp4(codes, scales, padded)
    if pad:
        vals = vals[..., : shape[-1]]
    return vals.reshape(shape).to(dtype)


def _kind(key: str) -> str:
    """The storage class of a state-dict key: factor | scale | plain.

    A factor is `tsct.is_factor_key`'s set (`u`, `v`, `gate_u`, `up_u`); a scale
    is the diag suffix. Everything else — including a float buffer like
    `lfb_counts` — is `plain` and rides at bf16.
    """
    if is_factor_key(key):
        return "factor"
    if key.rpartition(".")[2] in _SCALE_LEAVES:
        return "scale"
    return "plain"


def _packed_bytes(kind: str, t: torch.Tensor, mode: str) -> int:
    """The bytes this key occupies in the file — the report's per-key arithmetic.

    Factors: ternary is `n/4` code bytes + a 2-byte fp16 scale; mxfp4 is the
    codec's own `17 B/32` over the PADDED width. Scales are fp16, other floats
    bf16, non-floats as-is.
    """
    if kind == "factor":
        if mode == "ternary":
            return t.numel() // 4 + 2
        rows = math.prod(t.shape[:-1]) if t.dim() > 1 else 1
        return bytes_per_elements_mxfp4(t.numel() + rows * ((-t.shape[-1]) % BLOCK))
    if kind == "scale":
        return t.numel() * 2
    if t.is_floating_point():
        return t.numel() * 2
    return t.nbytes


@dataclass
class ExportReport:
    """What an export weighs: per-group bytes, totals, and the two baselines.

    `total_bytes` is the PAYLOAD (the tensors as stored, which is what the
    compression ratios are computed from); `file_bytes` is the container on
    disk (`torch.save` adds its zip metadata — the difference is named, never
    folded into the ratio).
    """

    mode: str
    groups: dict[str, int]  # group -> stored bytes
    counts: dict[str, int]  # group -> numel
    keys: dict[str, int]  # group -> tensor count
    bf16_groups: dict[str, int]  # group -> the same group at bf16
    fp32_groups: dict[str, int]  # group -> the same group at fp32
    total_bytes: int
    bf16_bytes: int
    fp32_bytes: int
    file_bytes: int = 0

    @property
    def ratio_vs_bf16(self) -> float:
        return self.bf16_bytes / self.total_bytes

    @property
    def ratio_vs_fp32(self) -> float:
        return self.fp32_bytes / self.total_bytes

    def table(self) -> str:
        """The size table a demo (or a report) prints."""
        lines = [
            f"QAT export, mode={self.mode}: {sum(self.keys.values())} tensors, "
            f"{sum(self.counts.values()):,} params",
            f"{'group':>8}  {'keys':>5}  {'params':>14}  {'bytes':>14}  "
            f"{'vs bf16':>8}  {'vs fp32':>8}",
        ]
        for g in _GROUPS:
            if not self.keys.get(g):
                continue
            b = self.groups[g]
            lines.append(
                f"{g:>8}  {self.keys[g]:5d}  {self.counts[g]:14,}  {b:14,}  "
                f"{self.bf16_groups[g] / b:7.2f}x  {self.fp32_groups[g] / b:7.2f}x"
            )
        lines.append(
            f"{'total':>8}  {sum(self.keys.values()):5d}  {sum(self.counts.values()):14,}  "
            f"{self.total_bytes:14,}  {self.ratio_vs_bf16:7.2f}x  "
            f"{self.ratio_vs_fp32:7.2f}x"
        )
        lines.append(f"on disk (torch.save container): {self.file_bytes:,} B")
        return "\n".join(lines)


def _clean_sd(model: nn.Module) -> dict[str, torch.Tensor]:
    """`state_dict()` with the `torch.compile` wrapper prefix stripped.

    The same normalization `save_ckpt` does: the `_orig_mod.` prefix is a
    wrapper artifact, never part of the file format, and a prefixed export
    would not load into the unwrapped model `load_qat` builds.
    """
    return {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}


def export_plan(model: nn.Module, mode: str = "ternary") -> ExportReport:
    """The size arithmetic of an export, WITHOUT writing anything (meta-safe).

    This is what the demo's size table and the medium/b1 estimates are made
    with: it reads shapes and dtypes only, so a `device="meta"` model plans at
    a b1 scale with no storage to pay for.
    """
    if mode not in MODES:
        loud(f"export mode={mode!r} is not a factor rung. Known: {list(MODES)}")
    groups = dict.fromkeys(_GROUPS, 0)
    counts = dict.fromkeys(_GROUPS, 0)
    keys = dict.fromkeys(_GROUPS, 0)
    bf16_groups = dict.fromkeys(_GROUPS, 0)
    fp32_groups = dict.fromkeys(_GROUPS, 0)
    bf16 = fp32 = 0
    for key, t in _clean_sd(model).items():
        kind = _kind(key)
        g = {"factor": "factors", "scale": "scales", "plain": "bf16"}[kind]
        if kind == "plain" and not t.is_floating_point():
            g = "other"
        groups[g] += _packed_bytes(kind, t, mode)
        counts[g] += t.numel()
        keys[g] += 1
        if t.is_floating_point():
            bf16_groups[g] += t.numel() * 2
            fp32_groups[g] += t.numel() * 4
            bf16 += t.numel() * 2
            fp32 += t.numel() * 4
        else:
            bf16_groups[g] += t.nbytes
            fp32_groups[g] += t.nbytes
            bf16 += t.nbytes
            fp32 += t.nbytes
    return ExportReport(
        mode=mode,
        groups=groups,
        counts=counts,
        keys=keys,
        bf16_groups=bf16_groups,
        fp32_groups=fp32_groups,
        total_bytes=sum(groups.values()),
        bf16_bytes=bf16,
        fp32_bytes=fp32,
    )


def _record(t: torch.Tensor, mode: str) -> dict:
    """One factor matrix → its container record (see the module docstring)."""
    if mode == "ternary":
        codes, scale = pack_ternary(t)
        return {
            "rung": "ternary",
            "codes": codes,
            "scale": torch.tensor(scale, dtype=torch.float16),
            "shape": tuple(t.shape),
            "n": t.numel(),
        }
    packed, scales, pad = pack_mxfp4(t)
    return {
        "rung": "mxfp4",
        "codes": packed,
        "scales": scales,
        "shape": tuple(t.shape),
        "pad": pad,
        "n": t.numel(),
    }


def _unpack_record(rec: dict, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """One container record → a tensor (bf16 by default: the load's precision)."""
    rung = rec.get("rung")
    shape = tuple(rec["shape"])
    if rung == "ternary":
        return unpack_ternary(rec["codes"], float(rec["scale"]), shape, dtype)
    if rung == "mxfp4":
        return unpack_mxfp4(rec["codes"], rec["scales"], shape, int(rec["pad"]), dtype)
    loud(
        f"export record carries rung={rung!r}, which is not one of {list(MODES)}. "
        "Cause: the file was written by a different codec. Escape: re-export it "
        "with this module."
    )


def export_qat(model: nn.Module, path: str | Path, *, mode: str = "ternary") -> ExportReport:
    """Pack `model` into `path` (one file) and return its size report.

    The walk is the state dict's: factors packed at `mode`, diagonals fp16,
    every other float bf16, non-float buffers as-is. The write is atomic
    (`<path>.tmp` then replace), like `save_ckpt`.
    """
    report = export_plan(model, mode)
    with torch.no_grad():
        tensors: dict[str, object] = {}
        for key, t in _clean_sd(model).items():
            kind = _kind(key)
            if kind == "factor":
                tensors[key] = _record(t, mode)
            elif kind == "scale":
                tensors[key] = t.detach().to(torch.float16)
            elif t.is_floating_point():
                tensors[key] = t.detach().to(torch.bfloat16)
            else:
                tensors[key] = t.detach()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": FORMAT, "version": VERSION, "mode": mode, "tensors": tensors}, tmp)
    tmp.replace(path)
    report.file_bytes = path.stat().st_size
    return report


def _read_container(path: str | Path) -> dict:
    """`path` → the container dict, LOUD on anything that is not one."""
    path = Path(path)
    if not path.is_file():
        loud(
            f"export file {path} does not exist. Escape: point at a file written "
            "by export_qat, or create one first."
        )
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 — the reason is in the message
        loud(
            f"{path} cannot be read as a dormouse QAT export: {e}. Escape: point "
            "at a file written by export_qat."
        )
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        got = ck.get("format") if isinstance(ck, dict) else type(ck).__name__
        loud(
            f"{path} is not a dormouse QAT export (format={got!r}, expected "
            f"{FORMAT!r}). Escape: point at a file written by export_qat (a training "
            "checkpoint is a different format — load it with load_ckpt)."
        )
    if ck.get("version") != VERSION:
        loud(f"{path} has version {ck.get('version')!r}, this module writes {VERSION}")
    if ck.get("mode") not in MODES:
        loud(f"{path} names mode {ck.get('mode')!r}, not one of {list(MODES)}")
    return ck


@contextmanager
def factor_qat_mode(model: nn.Module, mode: str, alpha: float = 1.0):
    """Force every QAT consumer in `model` to `(mode, alpha)` for the block.

    The walk is `DormouseModel.set_factor_qat_alpha`'s (every module carrying
    the attribute), plus the mode — `verify_roundtrip` uses it to build the
    reference forward that the export's numbers must match, and restores the
    model's own setting on exit (including on an exception).
    """
    if mode not in ("none", *MODES):
        loud(f"factor_qat_mode: {mode!r} is not a mode. Known: ['none', *{list(MODES)}]")
    assert 0.0 <= alpha <= 1.0, f"anneal alpha {alpha} outside 0..=1"
    prev: list[tuple[nn.Module, str, float]] = []
    for m in model.modules():
        if hasattr(m, "factor_qat"):
            prev.append((m, m.factor_qat, getattr(m, "factor_qat_alpha", 1.0)))
            m.factor_qat = mode
            m.factor_qat_alpha = alpha
    try:
        yield
    finally:
        for m, fq, a in prev:
            m.factor_qat = fq
            m.factor_qat_alpha = a


def load_qat(path: str | Path, cfg, *, device: torch.device | str = "cpu") -> nn.Module:
    """Rebuild the model from `cfg` and fill it from the export at `path`.

    Factor records unpack to bf16 values (the file's stated precision) and the
    rest loads as stored; the model is constructed with `factor_qat="none"`
    because the projection is already baked in — see the module docstring for
    why re-projecting is not idempotent.
    """
    from .model import DormouseModel

    ck = _read_container(path)
    # Only `factor_qat` is turned off: the projection is baked into the factor
    # records. `factor_qat_alpha` is left as the cfg has it — with the projector
    # off it is inert for the factors, and it is ALSO the dense path's alpha
    # (`tsct.qat_linear`), whose weights this export does NOT bake (they ride at
    # bf16), so overriding it here would change the loaded model's dense forward.
    model = DormouseModel(dataclasses.replace(cfg, factor_qat="none"), device=device)
    slots = model.state_dict()
    file_keys = set(ck["tensors"])
    missing = sorted(set(slots) - file_keys)
    extra = sorted(file_keys - set(slots))
    if missing or extra:
        hint = ""
        if any(k.endswith("engram.tables") for k in missing):
            hint = (
                " The missing `engram.tables` says the file came from a host-table "
                "(`train.engram_ram`) run: those tables live in a `<ckpt>.ngram` "
                "sidecar, not in the parameters, so this export cannot restore them "
                "— load the training checkpoint with load_ckpt instead."
            )
        loud(
            f"export/cfg mismatch: {len(missing)} model key(s) absent from the file "
            f"{missing[:3]}, {len(extra)} file key(s) not in the model {extra[:3]}. "
            "Cause: the file was written from a different model config. Escape: "
            f"load it with the cfg that produced it.{hint}"
        )
    sd: dict[str, torch.Tensor] = {}
    for key, v in ck["tensors"].items():
        if isinstance(v, dict):
            if not is_factor_key(key):
                loud(
                    f"export file has a packed record at non-factor key {key!r}: "
                    "the container is malformed. Escape: re-export it."
                )
            sd[key] = _unpack_record(v)
        else:
            sd[key] = v
    model.load_state_dict(sd, strict=True)
    return model


def verify_roundtrip(
    model: nn.Module,
    path: str | Path,
    batch,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Forward `batch` through the reference (projection on) and the loaded
    file; return the max abs logits difference.

    `batch` is `input_ids [b, t]` or a `(input_ids, targets)` pair (the targets
    are not read — only the logits are compared). The reference forward is the
    ORIGINAL model with every QAT consumer forced to the file's mode at
    `alpha=1.0` (the projection the export baked), and the model's own setting
    is restored on the way out; the loaded side is `load_qat`'s model, whose
    factors already ARE the projection.
    """
    ck = _read_container(path)
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    dev = next(model.parameters()).device if device is None else torch.device(device)
    x = x.to(dev)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), factor_qat_mode(model, ck["mode"], 1.0):
            ref, _, _ = model(x)
        loaded = load_qat(path, model.cfg, device=dev)
        loaded.eval()
        with torch.no_grad():
            got, _, _ = loaded(x)
    finally:
        model.train(was_training)
    assert ref.shape == got.shape, f"logits shapes differ: {ref.shape} vs {got.shape}"
    return float((ref - got).abs().max())
