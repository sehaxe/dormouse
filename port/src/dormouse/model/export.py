r"""The packed QAT export: the ternary release as ONE file.

WHAT IT IS. After QAT (the projections on, `factor_qat_project`'s / the dense
path's projection baked in), the model is exported to a single file whose
quantized weight matrices are b1.58 ternary — 2-bit codes, four per byte, one
fp16 absmean scale per tensor — and whose `s`/`gate_s`/`up_s` diagonals are
fp16. Everything the model did NOT train quantized (embeddings, norms, the
attention weights a `dense_qat_skip_attn` run kept on their masters) stays
bf16. The loaded model's forward reads the values the training-time projection
read on the original masters, so the export is a FORMAT, not a second
quantization: `load_qat` constructs with `factor_qat="none"` and
`dense_qat="none"`, because the projection is already IN the numbers, and
projecting them again would re-scale by the fraction of non-zero codes
(`ternarize` is not idempotent — `tests/test_ternary.py` pins that).

WHICH TENSORS. The set comes from the MODEL, never from a name list: every
TSCT factor key (`tsct.is_factor_key`), plus every 2-D `nn.Linear`/`DenseLinear`
weight whose declared kind is on — its own `_qat_kind` where the site splits
its children's kinds (`transformer.TransformerBlock`), else the nearest
owner's `dense_qat` (`GatedFFN`, `DenseLinear`, `KDAAttention`, `SparseMoE`,
`Engram`, `DSPark`, ...). A site that joins the quantized set is packed with no
edit here, and a site that leaves it is not; [`_weight_kinds`] is the whole
rule.

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
      "mode":    "ternary",            # the rung the factors were packed at
      "tensors": {key: record | tensor},  # state-dict keys, `_orig_mod.` stripped
    }

`record` (a packed factor matrix):
    ternary: {"rung": "ternary", "codes": uint8 [n/4], "scale": fp16 0-dim,
              "shape": (d0, d1), "n": int}
tensor: fp16 for the diagonals, bf16 for every other float, unchanged for
non-float buffers. A `record` sits at a key the model declared quantized; the
load checks its shape against the slot. This format is NOT the resume format:
`load_ckpt` wants fp32 masters and the optimizer state, which this file does
not carry (it is the release artifact).

THE ROUNDINGS, both bounded, both stated where a caller can check them: the
scale is stored fp16, so `±fp16(α)` costs ≤ 2^-11 relative against `α`; the
values are loaded as bf16, costing ≤ 2^-9 relative. [`roundtrip_deltas`]
measures the LOGITS both ways: the file's own distance from the reference
forward (the projection, fp32 masters) and the quantized forward's distance
from the bf16 masters — the error training already accepted. The first belongs
BELOW the second by a wide margin; if it does not, that is a bug, not a
tolerance.

LOUD refusals: a non-float master, an empty tensor, a numel not divisible by 4
(pack_ternary packs 4 codes per byte and does not silently pad), a non-finite
value (`ternarize`'s own guard), a foreign/truncated file, a key-set
mismatch — including the host-table case, where `engram.tables` is a `.ngram`
sidecar rather than a parameter and the error names that — a selected weight
that is not 2-D, a selected weight whose rung this file has no codec for, and
`mode="ternary"` on a model with nothing packable (the silent no-op: a bf16
file labelled ternary).
"""

from __future__ import annotations

import dataclasses
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from ..instrument import loud
from .ternary import _alpha
from .tsct import DenseLinear, _ternary_dequant, _ternary_quant, is_factor_key

#: The container's magic and version — `_read_container` refuses anything else.
FORMAT = "dormouse-qat-export"
VERSION = 1

#: The factor rungs this module packs: ternary only (the release).
MODES = ("ternary",)

#: The diag scales — fp16, never packed: they ARE the factorization's diagonal
#: (`tsct.py`: `s` is the scale, and a quantized scale would multiply the whole
#: row).
_SCALE_LEAVES = ("s", "gate_s", "up_s")

#: The report's groups, in print order. `factors` is the packed TSCT factor
#: matrices (plus their fp16 scales), `dense` the packed plain linear weights,
#: `scales` the fp16 diagonals, `bf16` every other float, `other` the non-float
#: buffers.
_GROUPS = ("factors", "dense", "scales", "bf16", "other")

#: The attributes a forward reads its dense quantizer from: `dense_qat` on every
#: owner that projects its own children (`GatedFFN`, `DenseLinear`, KDA, the
#: loop block, ...), and `_qat_attn`/`_qat_ffn` on the block that splits them.
#: [`_dense_qat_mode`] flips all three, so a baseline is read off the model's
#: own wiring instead of a second transcription of it.
_DENSE_KIND_ATTRS = ("dense_qat", "_qat_attn", "_qat_ffn")

#: Storage class -> report group.
_GROUP_OF = {"factor": "factors", "dense": "dense", "scale": "scales"}


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


def _declared_kind(model: nn.Module, name: str) -> str | None:
    """The rung module `name`'s weight is projected at, or None if undeclared.

    Nearest-first: the linear's own tag (`_qat_kind`, written where the kind is
    decided — `TransformerBlock`'s two site groups), then its owner, then the
    owner's owner, because a `DenseLinear`'s `dense_qat` is its own while a plain
    `nn.Linear`'s comes from the module that projects it. `None` means no module
    on the path declares anything — the site is not quantized, but it is REPORTED
    (see [`_weight_kinds`]) rather than assumed bf16.
    """
    parts = name.split(".")
    # Nearest first, the model itself last: a site can declare its kind on the
    # root (`site.dense_qat`) as easily as on its owner.
    chain = [model.get_submodule(".".join(parts[:i])) for i in range(len(parts), 0, -1)]
    for m in (*chain, model):
        kind = getattr(m, "_qat_kind", None)
        if kind is None:
            kind = getattr(m, "dense_qat", None)
        if isinstance(kind, str):
            return kind
    return None


def _weight_kinds(model: nn.Module, mode: str) -> tuple[dict[str, str], list[str]]:
    """{state-dict key: rung} for every DENSE weight the model trained quantized.

    The set is the model's own (see [`_declared_kind`]); there is no name list,
    so a future site cannot be silently missed. `undeclared` is what the walk
    could not classify either way — the report prints those keys, so a site
    whose owner names no kind shows up instead of riding bf16 unremarked.

    LOUD on a selected weight that is not 2-D (the codec and its one per-tensor
    absmean are a MATRIX's) and on a selected rung that is not this file's
    (`mode`): a `mxfp4`/`binary`/`ternary_s3t` master is a different
    quantization, and storing it as 2-bit ternary codes would be a lie about the
    artifact.
    """
    kinds: dict[str, str] = {}
    undeclared: list[str] = []
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Linear, DenseLinear)):
            continue
        key = f"{name}.weight"
        kind = _declared_kind(model, name)
        if kind is None:
            undeclared.append(key)
            continue
        if kind == "none":
            continue
        if m.weight.ndim != 2:
            loud(
                f"{key} is selected for packing at {kind!r} but its weight has "
                f"shape {tuple(m.weight.shape)}: the ternary codec stores a "
                "MATRIX with one per-tensor absmean scale. Escape: keep the site "
                "2-D, or declare it 'none' and ship it bf16."
            )
        if kind != mode:
            loud(
                f"{key} trained at dense_qat={kind!r} but this file writes "
                f"mode={mode!r} (the only codec: 2-bit codes + one fp16 scale per "
                "tensor). Storing a different rung's master in it would be a "
                "different quantization, not a format. Escape: export the model "
                "that trained at this rung, or declare this site 'none' and ship "
                "it bf16."
            )
        kinds[key] = kind
    return kinds, undeclared


def _storage_kind(key: str, dense: dict[str, str]) -> str:
    """The storage class of a state-dict key: factor | dense | scale | plain.

    A factor is `tsct.is_factor_key`'s set (`u`, `v`, `gate_u`, `up_u`); a dense
    key is one [`_weight_kinds`] selected; a scale is the diag suffix.
    Everything else — including a float buffer like `lfb_counts` — is `plain`
    and rides at bf16.
    """
    if is_factor_key(key):
        return "factor"
    if key in dense:
        return "dense"
    if key.rpartition(".")[2] in _SCALE_LEAVES:
        return "scale"
    return "plain"


def _packed_bytes(kind: str, t: torch.Tensor, mode: str) -> int:
    """The bytes this key occupies in the file — the report's per-key arithmetic.

    A packed matrix (factor or dense): `n/4` code bytes + a 2-byte fp16 scale.
    Scales are fp16, other floats bf16, non-floats as-is.
    """
    if kind in ("factor", "dense"):
        return t.numel() // 4 + 2
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
    #: 2-D linear weights no module declares a kind for — not packed, and named
    #: here so a future site is visible instead of silently bf16.
    undeclared: list[str] = field(default_factory=list)

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
        if self.undeclared:
            lines.append(
                f"not quantized by any declaration ({len(self.undeclared)}): "
                + ", ".join(self.undeclared[:3])
                + (", ..." if len(self.undeclared) > 3 else "")
            )
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
    with: it reads shapes, dtypes and the modules' declared kinds only, so a
    `device="meta"` model plans at a b1 scale with no storage to pay for.
    """
    if mode not in MODES:
        loud(f"export mode={mode!r} is not a factor rung. Known: {list(MODES)}")
    dense, undeclared = _weight_kinds(model, mode)
    sd = _clean_sd(model)
    factors = [k for k in sd if is_factor_key(k)]
    if not dense and not factors:
        loud(
            f"export mode={mode!r} but nothing in this model is packable: no "
            "TSCT factor keys and no 2-D linear the model declares quantized "
            f"(dense_qat={getattr(getattr(model, 'cfg', None), 'dense_qat', 'none')!r}, "
            f"factor_qat={getattr(getattr(model, 'cfg', None), 'factor_qat', 'none')!r}). "
            "A file written here would be the bf16 model with a 'ternary' label — "
            "the silent no-op this refusal exists to remove. Escape: export a "
            "model that TRAINED quantized, or ship the bf16 checkpoint as itself."
        )
    groups = dict.fromkeys(_GROUPS, 0)
    counts = dict.fromkeys(_GROUPS, 0)
    keys = dict.fromkeys(_GROUPS, 0)
    bf16_groups = dict.fromkeys(_GROUPS, 0)
    fp32_groups = dict.fromkeys(_GROUPS, 0)
    bf16 = fp32 = 0
    for key, t in sd.items():
        kind = _storage_kind(key, dense)
        g = _GROUP_OF.get(kind, "bf16")
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
        undeclared=undeclared,
    )


def _record(t: torch.Tensor, mode: str) -> dict:
    """One quantized matrix (factor or dense) → its container record."""
    assert mode == "ternary", f"export mode {mode!r} has no factor record"
    codes, scale = pack_ternary(t)
    return {
        "rung": "ternary",
        "codes": codes,
        "scale": torch.tensor(scale, dtype=torch.float16),
        "shape": tuple(t.shape),
        "n": t.numel(),
    }


def _unpack_record(rec: dict, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """One container record → a tensor (bf16 by default: the load's precision)."""
    rung = rec.get("rung")
    shape = tuple(rec["shape"])
    if rung == "ternary":
        return unpack_ternary(rec["codes"], float(rec["scale"]), shape, dtype)
    loud(
        f"export record carries rung={rung!r}, which is not one of {list(MODES)}. "
        "Cause: the file was written by a different codec. Escape: re-export it "
        "with this module."
    )


def export_qat(model: nn.Module, path: str | Path, *, mode: str = "ternary") -> ExportReport:
    """Pack `model` into `path` (one file) and return its size report.

    The walk is the state dict's: every key the model declared quantized (the
    factor set plus [`_weight_kinds`]) packed at `mode`, diagonals fp16, every
    other float bf16, non-float buffers as-is. The write is atomic
    (`<path>.tmp` then replace), like `save_ckpt`.
    """
    report = export_plan(model, mode)
    dense, _ = _weight_kinds(model, mode)
    with torch.no_grad():
        tensors: dict[str, object] = {}
        for key, t in _clean_sd(model).items():
            kind = _storage_kind(key, dense)
            if kind in ("factor", "dense"):
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
    model's own setting on exit (including on an exception). The DENSE path's
    mode is NOT forced (`dense_qat` decides which weights the export packs, so
    forcing it would change the model under the export); its `alpha` is, and it
    is the same ramp.
    """
    if mode not in ("none", *MODES):
        loud(f"factor_qat_mode: {mode!r} is not a mode. Known: ['none', *{list(MODES)}]")
    assert 0.0 <= alpha <= 1.0, f"anneal alpha {alpha} outside 0..=1"
    prev: list[tuple[nn.Module, str, float]] = []
    dense_alpha: list[tuple[nn.Module, float]] = []
    for m in model.modules():
        if hasattr(m, "factor_qat"):
            prev.append((m, m.factor_qat, getattr(m, "factor_qat_alpha", 1.0)))
            m.factor_qat = mode
            m.factor_qat_alpha = alpha
        if hasattr(m, "dense_qat_alpha"):
            dense_alpha.append((m, m.dense_qat_alpha))
            m.dense_qat_alpha = alpha
    try:
        yield
    finally:
        for m, fq, a in prev:
            m.factor_qat = fq
            m.factor_qat_alpha = a
        for m, a in dense_alpha:
            m.dense_qat_alpha = a


@contextmanager
def _dense_qat_mode(model: nn.Module, kind: str):
    """Force every dense quantizer the FORWARD reads to `kind` for the block.

    The three attribute names are the forward's seam, not a weight list:
    `dense_qat` on each owner that projects its own children, and the transformer
    block's two site groups. `_qat_kind` is NOT in that set — it is an export
    annotation, read by [`_weight_kinds`], never by a forward.
    """
    assert kind in ("none", *MODES), f"dense kind {kind!r} is not a mode"
    prev: list[tuple[nn.Module, str, str]] = []
    for m in model.modules():
        for attr in _DENSE_KIND_ATTRS:
            old = getattr(m, attr, None)
            if isinstance(old, str):
                prev.append((m, attr, old))
                setattr(m, attr, kind)
    try:
        yield
    finally:
        for m, attr, old in prev:
            setattr(m, attr, old)


def load_qat(path: str | Path, cfg, *, device: torch.device | str = "cpu") -> nn.Module:
    """Rebuild the model from `cfg` and fill it from the export at `path`.

    Packed records unpack to bf16 values (the file's stated precision) and the
    rest loads as stored; the model is constructed with `factor_qat="none"` and
    `dense_qat="none"` because the projection is already baked in — see the
    module docstring for why re-projecting is not idempotent.
    """
    from .model import DormouseModel

    ck = _read_container(path)
    # Both projectors are off: the packed records hold the projected values.
    # `dense_qat="none"` matters as much as `factor_qat="none"` — a model built
    # with it on would project the stored codes a SECOND time (and `ternarize`
    # is not idempotent), and would read its bf16 neighbours as if they had been
    # packed. The alphas are inert with both off, so they ride as the cfg has
    # them.
    model = DormouseModel(
        dataclasses.replace(cfg, factor_qat="none", dense_qat="none"), device=device
    )
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
            slot = slots[key]
            if tuple(v["shape"]) != tuple(slot.shape) or v["n"] != slot.numel():
                loud(
                    f"export record {key!r} holds a {tuple(v['shape'])} matrix "
                    f"({v['n']} elements) but the model's slot is "
                    f"{tuple(slot.shape)} ({slot.numel()}): the file was written "
                    "from a different model config. Escape: load it with the cfg "
                    "that produced it, or re-export."
                )
            sd[key] = _unpack_record(v)
        else:
            sd[key] = v
    model.load_state_dict(sd, strict=True)
    return model


def roundtrip_deltas(
    model: nn.Module,
    path: str | Path,
    batch,
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """The file's fidelity vs the quantization the model accepted, in logits.

    Three forwards, all on `batch` (`input_ids [b, t]` or an `(input_ids,
    targets)` pair — the targets are not read):

    - `ref`: the ORIGINAL model with every projection forced on at `alpha=1.0`
      (the values the export baked), restored on the way out;
    - `master`: the same weights with every projection OFF — the bf16 network,
      which is what the training-time quantized forward is measured against;
    - `got`: `load_qat`'s model, whose packed records already ARE the projection.

    Returns `export_abs`/`export_rel` (|ref − got| — the STORAGE: fp16 scale,
    bf16 values, and nothing else) and `quant_abs`/`quant_rel` (|ref − master|
    — the ternary quantizer's own error, which training paid on every step).
    The export is correct iff the first is far below the second; equal-ish means
    the pack or the load is wrong, not that the tolerance is tight. `_rel` is
    divided by max|ref|, so it is a fraction of the logits' own scale.
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
            with _dense_qat_mode(model, "none"):
                master, _, _ = model(x)
        loaded = load_qat(path, model.cfg, device=dev)
        loaded.eval()
        with torch.no_grad():
            got, _, _ = loaded(x)
    finally:
        model.train(was_training)
    assert ref.shape == got.shape, f"logits shapes differ: {ref.shape} vs {got.shape}"
    scale = max(float(ref.abs().max()), 1e-30)
    export_abs = float((ref - got).abs().max())
    quant_abs = float((ref - master).abs().max())
    return {
        "export_abs": export_abs,
        "export_rel": export_abs / scale,
        "quant_abs": quant_abs,
        "quant_rel": quant_abs / scale,
    }


def verify_roundtrip(
    model: nn.Module,
    path: str | Path,
    batch,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Forward `batch` through the reference (projection on) and the loaded
    file; return the max abs logits difference.

    The number that says whether the file is the same network — see
    [`roundtrip_deltas`] for it beside the quantizer's own error, which is the
    scale it has to be read against.
    """
    return roundtrip_deltas(model, path, batch, device=device)["export_abs"]
