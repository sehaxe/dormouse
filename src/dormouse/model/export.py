r"""The `.dormouse` artifact: one self-describing file for every reader.

WHAT IT IS. A model, its weights and its config in ONE file that any reader can
parse without torch: an 8-byte magic, a u32 header length, a JSON header, and
the tensor blobs 8-byte aligned after it. The browser demo
(`docs-site/public/demo/engine.js`) and `dormouse decode`/`serve` read the SAME
bytes — that is the point of the format, and why there is one writer and one
reader here rather than a web exporter beside a release exporter.

LAYOUT (all offsets absolute, from byte 0, so a reader can map the file and
slice without walking the records):

    b"DMSEXP\x01\x00"   magic (8 B; the last two bytes are the format version)
    u32 header_len      the JSON header's length in bytes
    u32 flags           reserved, 0
    header              UTF-8 JSON, space-padded to 8 B
    blobs               each record's bytes, 8-byte aligned

HEADER. `format`, `version`, `preset`, `mode` (the factor rung), `architecture`
(`small-byte-lm` when `engine_refusal` is None), `engine_refusal` (the named
field mismatches, or None), `created`, `commit`, `step`, `seed`, `model` (the
ModelConfig as a dict — the reader rebuilds the model from it), `dtypes`, and
`tensors`: one record per state-dict key:

    {name, dtype, shape, offset, length}                dense
    {name, dtype: "ternary", shape, offset, length, scale}
    {name, dtype: "mxfp4", shape, offset, length, scales_offset, scales_length, pad}

DTYPES, and why these: dense tensors are **fp16** (2 B/param). The checkpoint's
masters are fp32 and the model trains in bf16, whose mantissa is 8 bits against
fp16's 11 — so fp16 storage loses LESS than the precision the run was trained
in, at half the size of fp32; every reader widens to fp32 before computing.
`--dense-dtype fp32` exists for a reader that wants the masters verbatim. The
Engram table has its own rung (`--engram-dtype`, fp32 default): it is 33% of
`small`'s parameters and the one tensor a browser wants small, so it is a
choice, not a constant. TSCT factors are the ternary codec's 2-bit codes plus
their scale — `scale` is a JSON number, i.e. an exact fp64 round trip, which is
why a factor unpacks BIT-EXACT to `ternarize(master)` (the codec's own
contract) rather than to a 2^-11-rounded copy of it.

LOUD refusals, all of them named: an unknown state-dict key (a key this file
format has no slot for), a snapshot whose model half disagrees with the named
preset, a missing Engram table (an `engram_ram` run whose `.ngram` sidecar is
gone), a non-finite weight, a non-float factor, a numel not divisible by 4, a
rung that is not one of `MODES`, a truncated or foreign file, an offset that
runs past the end, and a dtype name this module did not write.

WHAT IT IS NOT: a resume format. The optimizer state is not here — the
training checkpoint (`train.loop.save_ckpt`) stays the resume artifact. And it
is a STORAGE format: this port has no int8/LUT ternary kernel, so a loaded
model's forward is the same fp32/bf16 matmul path it always was.

THE ROUNDINGS are stated where a caller can check them: dense fp16 (2^-11
relative), the engram table at its own rung, the factors at their codec's grid
(±mean|W|). `verify_export` measures the LOGITS' max abs difference between the
reference forward (the projection on the fp32 masters) and the model rebuilt
from the file — the number that says whether the file is the same network.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..instrument import loud
from .mxfp import BLOCK, bytes_per_elements_mxfp4, dequantize_mxfp4, quantize_mxfp4
from .ternary import _alpha
from .tsct import _ternary_dequant, _ternary_quant, is_factor_key

#: The container's magic. The trailing `\x01\x00` is the format version, so a
#: reader can refuse a future file before it parses the header.
MAGIC = b"DMSEXP\x01\x00"
FORMAT = "dormouse"
VERSION = 1

#: The factor rungs this module packs. `ternary` is the release; `mxfp4` is the
#: same report with `mxfp.quantize_mxfp4` as the codec.
MODES = ("ternary", "mxfp4")

#: Blob alignment. 8 covers every dtype here (fp32 needs 4), and it is what
#: makes an mmap/`fetch` + typed-array view legal without a copy.
ALIGN = 8

#: The diag scales — fp16, never packed: they ARE the factorization's diagonal
#: (`tsct.py`: `s` is the scale, and a quantized scale would multiply the whole
#: row).
_SCALE_LEAVES = ("s", "gate_s", "up_s")

#: The report's groups, in print order. `factors` is the packed matrices,
#: `scales` the diagonals, `dense` every other float, `engram` the n-gram
#: table(s) at their own rung, `other` the non-float buffers.
_GROUPS = ("factors", "scales", "dense", "engram", "other")

#: The header's dtype names ↔ torch. Every dtype the writer can emit; an
#: unknown torch dtype is a LOUD refusal rather than a guess.
_TORCH_OF = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "u8": torch.uint8,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}
_BYTES_OF = {k: torch.tensor([], dtype=v).element_size() for k, v in _TORCH_OF.items()}
_NAME_OF = {v: k for k, v in _TORCH_OF.items()}

#: The ONE architecture `docs-site/public/demo/engine.js` implements: the
#: `small` preset's shape as `configs/small.toml` ships it, field by field.
#: A mismatch is not a warning — the engine refuses the file and names the
#: field, because an engine that silently runs a different polygon is a demo
#: of something else. `engram_lam_max`/`norm_eps`/`bf16`/`seed` are NOT here:
#: the engine READS them from the header, so any value is a working value.
ENGINE_SHAPE: dict[str, object] = {
    "d_model": 768,
    "n_heads": 12,
    "head_dim": 64,
    "d_ffn": 2048,
    "vocab": 256,
    "max_seq_len": 512,
    "max_iter": 4,
    "rank": 64,
    "n_experts": 3,
    "moe_topk": 0,
    "shared_rank": 192,
    "engram_rows": 25000,
    "engram_orders": [2, 3, 4],
    "engram_dim": 32,
    "use_tsct": True,
    "use_kda": True,
    "use_engram": True,
    "use_attnres": False,
    "use_byteflow": False,
    "use_msa": False,
    "loom": False,
    "quant": "none",
    "factor_qat": "none",
    "dense_qat": "none",
    "expert_act": "silu",
    "dspark_weight": 0.0,
}

#: Fields the engine ignores, each with the reason — the refusal above is only
#: honest if the ignored set is named too (ADR-0019's "which arm ran").
ENGINE_IGNORED = {
    "bf16": "a compute-dtype setting: the engine computes fp32 (its reference path)",
    "moe_capacity": "inert at moe_topk=0: the dense blend has no capacity rule",
    "moe_lb_coef": "the LFB bias steers the top-k selection, which moe_topk=0 never runs",
    "moe_kernel": "a dispatch choice for the routed pass, absent at moe_topk=0",
    "moe_offload": "an activation-offload switch for the routed pass (COUNTED inert at topk=0)",
    "factor_qat_alpha": "inert with factor_qat=none (the engine reads the packed values)",
    "sigreg_weight": "a training regulariser: no parameters, no forward term",
    "seed": "the engine's sampler is seeded by the page, not by the run",
}


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
    """The storage class of a state-dict key: factor | scale | engram | plain.

    A factor is `tsct.is_factor_key`'s set (`u`, `v`, `gate_u`, `up_u`); a scale
    is the diag suffix; the n-gram table has its own rung (`--engram-dtype`);
    everything else — including a float buffer like `lfb_counts` — is `plain`.
    """
    if is_factor_key(key):
        return "factor"
    leaf = key.rpartition(".")[2]
    if leaf in _SCALE_LEAVES:
        return "scale"
    if key.endswith("engram.tables"):
        return "engram"
    return "plain"


def _group(kind: str, t: torch.Tensor) -> str:
    if kind == "engram":
        return "engram"
    if kind == "factor":
        return "factors"
    if kind == "scale":
        return "scales"
    return "dense" if t.is_floating_point() else "other"


def _stored_dtype(key: str, t: torch.Tensor, dense_dtype: str, engram_dtype: str) -> str:
    """The dtype NAME this key rides at — one decision, both the plan and the write."""
    if not t.is_floating_point():
        name = _NAME_OF.get(t.dtype)
        if name is None:
            loud(
                f"{key}: {t.dtype} is not a dtype this container writes. Known: "
                f"{sorted(_TORCH_OF)}. Escape: add it to `_TORCH_OF` and its reader."
            )
        return name
    return engram_dtype if _kind(key) == "engram" else dense_dtype


def _packed_bytes(key: str, t: torch.Tensor, mode: str, dense_dtype: str, engram_dtype: str) -> int:
    """The bytes this key occupies in the file — the report's per-key arithmetic."""
    kind = _kind(key)
    if kind == "factor":
        if mode == "ternary":
            return t.numel() // 4
        rows = math.prod(t.shape[:-1]) if t.dim() > 1 else 1
        return bytes_per_elements_mxfp4(t.numel() + rows * ((-t.shape[-1]) % BLOCK))
    return t.numel() * _BYTES_OF[_stored_dtype(key, t, dense_dtype, engram_dtype)]


@dataclass
class ExportReport:
    """What an export weighs: per-group bytes, totals, and the two baselines.

    `total_bytes` is the PAYLOAD (the blobs as stored, which is what the
    compression ratios are computed from); `file_bytes` is the container on
    disk — the difference is the header and the alignment padding, and it is
    named rather than folded into the ratio.
    """

    mode: str
    groups: dict[str, int]  # group -> stored bytes
    counts: dict[str, int]  # group -> numel
    keys: dict[str, int]  # group -> tensor count
    fp16_groups: dict[str, int]  # group -> the same group at 2 B/param
    fp32_groups: dict[str, int]  # group -> the same group at 4 B/param
    total_bytes: int
    fp16_bytes: int
    fp32_bytes: int
    file_bytes: int = 0
    header_bytes: int = 0
    pad_bytes: int = 0

    @property
    def ratio_vs_fp16(self) -> float:
        return self.fp16_bytes / self.total_bytes

    @property
    def ratio_vs_fp32(self) -> float:
        return self.fp32_bytes / self.total_bytes

    def table(self) -> str:
        """The size table a demo (or a report) prints."""
        lines = [
            f"export, mode={self.mode}: {sum(self.keys.values())} tensors, "
            f"{sum(self.counts.values()):,} params",
            f"{'group':>8}  {'keys':>5}  {'params':>14}  {'bytes':>14}  "
            f"{'vs fp16':>8}  {'vs fp32':>8}",
        ]
        for g in _GROUPS:
            if not self.keys.get(g):
                continue
            b = self.groups[g]
            lines.append(
                f"{g:>8}  {self.keys[g]:5d}  {self.counts[g]:14,}  {b:14,}  "
                f"{self.fp16_groups[g] / b:7.2f}x  {self.fp32_groups[g] / b:7.2f}x"
            )
        lines.append(
            f"{'total':>8}  {sum(self.keys.values()):5d}  {sum(self.counts.values()):14,}  "
            f"{self.total_bytes:14,}  {self.ratio_vs_fp16:7.2f}x  {self.ratio_vs_fp32:7.2f}x"
        )
        lines.append(
            f"on disk: {self.file_bytes:,} B (header {self.header_bytes:,} B + "
            f"{self.total_bytes:,} B payload + {self.pad_bytes:,} B alignment)"
        )
        return "\n".join(lines)


def _clean_sd(model: nn.Module) -> dict[str, torch.Tensor]:
    """`state_dict()` with the `torch.compile` wrapper prefix stripped.

    The same normalization `save_ckpt` does: the `_orig_mod.` prefix is a
    wrapper artifact, never part of the file format, and a prefixed export
    would not load into the unwrapped model `load_export` builds.
    """
    return {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}


def export_plan(
    model: nn.Module,
    mode: str = "ternary",
    *,
    dense_dtype: str = "fp16",
    engram_dtype: str = "fp32",
) -> ExportReport:
    """The size arithmetic of an export, WITHOUT writing anything (meta-safe).

    This is what the demo's size table and the medium/b1 estimates are made
    with: it reads shapes and dtypes only, so a `device="meta"` model plans at
    a b1 scale with no storage to pay for.
    """
    if mode not in MODES:
        loud(f"export mode={mode!r} is not a factor rung. Known: {list(MODES)}")
    for name, d in (("dense_dtype", dense_dtype), ("engram_dtype", engram_dtype)):
        if d not in ("fp32", "fp16"):
            loud(f"{name}={d!r}: this container stores dense tensors as fp32 or fp16")
    groups = dict.fromkeys(_GROUPS, 0)
    counts = dict.fromkeys(_GROUPS, 0)
    keys = dict.fromkeys(_GROUPS, 0)
    fp16_groups = dict.fromkeys(_GROUPS, 0)
    fp32_groups = dict.fromkeys(_GROUPS, 0)
    fp16 = fp32 = 0
    for key, t in _clean_sd(model).items():
        kind = _kind(key)
        g = _group(kind, t)
        groups[g] += _packed_bytes(key, t, mode, dense_dtype, engram_dtype)
        counts[g] += t.numel()
        keys[g] += 1
        if t.is_floating_point():
            fp16_groups[g] += t.numel() * 2
            fp32_groups[g] += t.numel() * 4
            fp16 += t.numel() * 2
            fp32 += t.numel() * 4
        else:
            fp16_groups[g] += t.nbytes
            fp32_groups[g] += t.nbytes
            fp16 += t.nbytes
            fp32 += t.nbytes
    return ExportReport(
        mode=mode,
        groups=groups,
        counts=counts,
        keys=keys,
        fp16_groups=fp16_groups,
        fp32_groups=fp32_groups,
        total_bytes=sum(groups.values()),
        fp16_bytes=fp16,
        fp32_bytes=fp32,
    )


def _blob(t: torch.Tensor, dtype: str) -> bytes:
    """One tensor → its bytes, at the header's dtype name."""
    tt = t.detach().to(_TORCH_OF[dtype]).contiguous()
    return tt.numpy().tobytes()


def _records(
    sd: dict[str, torch.Tensor], mode: str, dense_dtype: str, engram_dtype: str
) -> list[dict]:
    """Every key → `(record, blobs)` with offsets left at 0 (assigned later).

    The blobs are the file's payload in record order: dense at its rung, a
    ternary factor as one codes blob (its scale rides in the header), an mxfp4
    factor as its codes plus its e8m0 scale row.
    """
    out: list[dict] = []
    for key, t in sd.items():
        if t.is_floating_point() and not torch.isfinite(t).all():
            loud(
                f"{key} holds a non-finite value (inf/NaN): a file that encodes it "
                "would decode to a network that is not the one trained. Cause: a "
                "diverged run, or a corrupt checkpoint. Escape: fix the weights "
                "(the NaN firewall is the trainer's), or export another step."
            )
        kind = _kind(key)
        if kind == "factor":
            if mode == "ternary":
                codes, scale = pack_ternary(t)
                out.append(
                    {
                        "name": key,
                        "dtype": "ternary",
                        "shape": list(t.shape),
                        "offset": 0,
                        "length": codes.numel(),
                        "scale": scale,
                        "_blobs": [codes.numpy().tobytes()],
                    }
                )
            else:
                packed, scales, pad = pack_mxfp4(t)
                out.append(
                    {
                        "name": key,
                        "dtype": "mxfp4",
                        "shape": list(t.shape),
                        "offset": 0,
                        "length": packed.numel(),
                        "scales_offset": 0,
                        "scales_length": scales.numel(),
                        "pad": int(pad),
                        "_blobs": [packed.numpy().tobytes(), scales.numpy().tobytes()],
                    }
                )
            continue
        dtype = _stored_dtype(key, t, dense_dtype, engram_dtype)
        blob = _blob(t, dtype)
        out.append(
            {
                "name": key,
                "dtype": dtype,
                "shape": list(t.shape),
                "offset": 0,
                "length": len(blob),
                "_blobs": [blob],
            }
        )
    return out


def _host_tables(model: nn.Module) -> dict[str, torch.Tensor]:
    """`{key: rows}` for every Engram on host tables — the memory's state.

    An `engram_ram` run's tables are NOT parameters (`Engram.to_host` deletes
    them), so a state dict alone would export a memory-less model under the
    same name. The caller restores the tables from the `.ngram` sibling before
    this runs (`train.loop.load_model_state`), so what is read here is the
    memory the checkpoint's step count belongs to.
    """
    from .engram import Engram

    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, Engram) and mod.host is not None:
            out[f"{name}.tables"] = mod.host.rows
    return out


def write_export(
    model: nn.Module,
    path: str | Path,
    *,
    mode: str = "ternary",
    preset: str = "",
    seed: int | None = None,
    step: int | None = None,
    commit: str | None = None,
    dense_dtype: str = "fp16",
    engram_dtype: str = "fp32",
) -> ExportReport:
    """Pack `model` into `path` (one `.dormouse` file) and return its size report.

    The walk is the state dict's: factors packed at `mode`, the Engram table at
    `engram_dtype`, every other float at `dense_dtype`, non-float buffers as-is.
    A host-table Engram is folded in as its `engram.tables` key (see
    [`_host_tables`]) — a LOUD refusal if it is missing while the config says
    `use_engram`. The write is atomic (`<path>.tmp` then replace), like
    `save_ckpt`.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        loud(
            "write_export: `model` carries no `.cfg` — the header IS the config, so "
            "the writer needs a DormouseModel (a bare nn.Module or a state dict has "
            "no architecture to describe). Escape: pass the model the trainer built."
        )
    report = export_plan(model, mode, dense_dtype=dense_dtype, engram_dtype=engram_dtype)
    sd = _clean_sd(model)
    host = _host_tables(model)
    if host:
        for key, rows in host.items():
            if key in sd:
                loud(
                    f"export: {key} is both a parameter and a host table — the model "
                    "is inconsistent"
                )
            sd[key] = rows
    if cfg.use_engram and not any(k.endswith("engram.tables") for k in sd):
        loud(
            "export: use_engram=true but the state dict carries no `engram.tables`. "
            "Cause: a host-table run (`train.engram_ram`) whose `<ckpt>.ngram` "
            "sidecar was not loaded, so this export would write a memory-less model "
            "under the run's name. Escape: load the checkpoint through "
            "`train.loop.load_model_state` (it restores the sidecar) before exporting."
        )
    records = _records(sd, mode, dense_dtype, engram_dtype)
    blobs = [rec.pop("_blobs") for rec in records]  # the payload, not the header
    header = {
        "format": FORMAT,
        "version": VERSION,
        "preset": preset,
        "mode": mode,
        "dtypes": {"dense": dense_dtype, "engram": engram_dtype},
        "architecture": None,
        "engine_refusal": None,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "step": step,
        "seed": seed,
        "model": dataclasses.asdict(cfg),
        "tensors": records,
    }
    why = engine_refusal(cfg)
    header["engine_refusal"] = why
    header["architecture"] = "byte-lm" if why else "small-byte-lm"
    # The offsets are absolute and the header's own length depends on them
    # (a bigger base can add a digit to every offset), so the layout is a
    # fixpoint: assign, serialize, recompute the base, repeat until stable.
    # Monotone in `base`, so it converges in one or two passes.
    base = 16
    rel = 0
    for _ in range(8):
        rel = 0
        for rec, parts in zip(records, blobs, strict=True):
            for i, blob in enumerate(parts):
                rel += (-rel) % ALIGN
                if i == 0:
                    rec["offset"] = base + rel
                else:
                    rec["scales_offset"] = base + rel
                rel += len(blob)
        head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        new_base = 16 + len(head)
        new_base += (-new_base) % ALIGN
        if new_base == base:
            break
        base = new_base
    else:  # pragma: no cover — the loop is a layout fixpoint, not a heuristic
        loud("export: the header/offset layout did not converge in 8 passes")
    pad = (-(16 + len(head))) % ALIGN
    assert 16 + len(head) + pad == base, "the layout fixpoint disagrees with the write"
    # `rel` counts the inter-blob alignment too, so the file's size is
    # header + payload + that padding, exactly — the identity the report and
    # `test_exported_file_size_is_what_the_report_says` both rest on.
    pads = rel - report.total_bytes
    assert pads >= 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(head), 0))
        f.write(head)
        f.write(b" " * pad)
        for parts in blobs:
            for blob in parts:
                f.write(blob)
                f.write(b"\x00" * ((-f.tell()) % ALIGN))
    tmp.replace(path)
    report.file_bytes = path.stat().st_size
    report.header_bytes = 16 + len(head) + pad
    report.pad_bytes = pads
    return report


@dataclass
class ExportFile:
    """A parsed `.dormouse` file: the header, the records and the blob bytes.

    `buf` is the file in memory (one read). `tensor(name)` returns the tensor a
    record names, at the container's own dtype widened to fp32 — the values a
    reader computes with, never a copy of the file's layout.
    """

    path: Path
    header: dict
    buf: bytes

    @property
    def records(self) -> list[dict]:
        return self.header["tensors"]

    @property
    def cfg(self):
        """The ModelConfig the writer recorded — the reader's construction input."""
        from ..config import ModelConfig

        raw = self.header["model"]
        try:
            return ModelConfig(**raw)
        except TypeError as e:
            loud(
                f"{self.path}: the header's `model` does not fit this build's "
                f"ModelConfig ({e}). Cause: the file was written by another "
                "version of the tree. Escape: re-export with this build."
            )

    def record(self, name: str) -> dict:
        for rec in self.records:
            if rec["name"] == name:
                return rec
        loud(
            f"{self.path}: no tensor named {name!r}. Cause: a key-set mismatch "
            "(the file was written from a different config). Escape: re-export."
        )

    def names(self) -> set[str]:
        return {rec["name"] for rec in self.records}

    def _slice(self, offset: int, length: int, dtype: str) -> torch.Tensor:
        if offset < 0 or length < 0 or offset + length > len(self.buf):
            loud(
                f"{self.path}: a record points outside the file (offset {offset}, "
                f"length {length}, file {len(self.buf)} B). Cause: a truncated or "
                "corrupt file. Escape: re-export it."
            )
        t = _TORCH_OF[dtype]
        count = length // _BYTES_OF[dtype]
        return torch.frombuffer(bytearray(self.buf[offset : offset + length]), dtype=t, count=count)

    def tensor(self, name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The named tensor, decoded and widened to `dtype` (fp32 by default)."""
        rec = self.record(name)
        shape = tuple(rec["shape"])
        kind = rec["dtype"]
        if kind == "ternary":
            codes = self._slice(rec["offset"], rec["length"], "u8")
            return unpack_ternary(codes, rec["scale"], shape, dtype)
        if kind == "mxfp4":
            codes = self._slice(rec["offset"], rec["length"], "u8")
            scales = self._slice(rec["scales_offset"], rec["scales_length"], "u8")
            return unpack_mxfp4(codes, scales, shape, int(rec.get("pad", 0)), dtype)
        return self._slice(rec["offset"], rec["length"], kind).reshape(shape).to(dtype)

    def tensors(self, *, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        return {rec["name"]: self.tensor(rec["name"], dtype=dtype) for rec in self.records}

    def engine_refusal(self) -> str | None:
        """Why `docs-site/public/demo/engine.js` cannot run this file, or None."""
        return engine_refusal(self.cfg)


def _check_layout(path: Path, header: dict, buf: bytes) -> None:
    """Every structural claim the reader depends on, checked once, LOUD."""
    if header.get("format") != FORMAT:
        loud(
            f"{path} is not a dormouse artifact (format={header.get('format')!r}, "
            f"expected {FORMAT!r}). Escape: point at a file written by "
            "`dormouse export` (a training checkpoint is a different format — load "
            "it with load_ckpt)."
        )
    if header.get("version") != VERSION:
        loud(f"{path} has version {header.get('version')!r}, this build writes {VERSION}")
    if header.get("mode") not in MODES:
        loud(f"{path} names mode {header.get('mode')!r}, not one of {list(MODES)}")
    seen: set[str] = set()
    for rec in header.get("tensors", []):
        for field in ("name", "dtype", "shape", "offset", "length"):
            if field not in rec:
                loud(f"{path}: a tensor record is missing {field!r} — the file is malformed")
        if rec["name"] in seen:
            loud(f"{path}: tensor {rec['name']!r} appears twice — the file is malformed")
        seen.add(rec["name"])
        dtype = rec["dtype"]
        if dtype not in ("ternary", "mxfp4", *_TORCH_OF):
            loud(
                f"{path}: tensor {rec['name']!r} has dtype {dtype!r}, which this "
                f"reader does not know. Known: fp32, fp16, bf16, u8, i32, i64, bool, "
                "ternary, mxfp4. Escape: re-export with this build."
            )
        if rec["offset"] % ALIGN:
            loud(
                f"{path}: tensor {rec['name']!r} starts at offset {rec['offset']}, "
                f"not {ALIGN}-byte aligned — a typed-array view would be illegal"
            )
        if rec["offset"] + rec["length"] > len(buf):
            loud(
                f"{path}: tensor {rec['name']!r} runs past the end of the file "
                f"({rec['offset']} + {rec['length']} > {len(buf)}). Cause: a truncated "
                "file. Escape: re-export it."
            )


def read_export(path: str | Path) -> ExportFile:
    """`path` → the parsed container, LOUD on anything that is not one."""
    path = Path(path)
    if not path.is_file():
        loud(
            f"export file {path} does not exist. Escape: point at a file written by "
            "`dormouse export`, or create one first."
        )
    buf = path.read_bytes()
    if len(buf) < 16 or buf[:8] != MAGIC:
        got = buf[:8]
        loud(
            f"{path} is not a dormouse artifact (magic {got!r}, expected {MAGIC!r}). "
            "Escape: point at a file written by `dormouse export`."
        )
    header_len, _flags = struct.unpack_from("<II", buf, 8)
    if header_len <= 0 or 16 + header_len > len(buf):
        loud(
            f"{path}: header length {header_len} does not fit the file ({len(buf)} B). "
            "Cause: a truncated or corrupt file. Escape: re-export it."
        )
    try:
        header = json.loads(buf[16 : 16 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        loud(f"{path}: the header is not readable JSON ({e}). Escape: re-export it.")
    if not isinstance(header, dict):
        loud(f"{path}: the header is {type(header).__name__}, not an object")
    _check_layout(path, header, buf)
    return ExportFile(path=path, header=header, buf=buf)


def load_export(
    path: str | Path | ExportFile, *, device: torch.device | str = "cpu", cfg=None
) -> nn.Module:
    """Rebuild the model from the artifact at `path` (or a parsed [`ExportFile`]).

    Dense tensors load at fp32 (the file's fp16 widened), factor records unpack
    to their codec's values, and the Engram table loads from its own blob — so
    an `engram_ram` run's memory is IN this file, unlike the old container where
    it was a sidecar the load could only complain about.

    The model is constructed with `factor_qat="none"` because the projection is
    already baked in — re-projecting is not idempotent (`ternarize` re-scales by
    the non-zero fraction, `tests/test_ternary.py` pins that). `cfg`, if given,
    must equal the header's; the reader's authority is the FILE.
    """
    from .model import DormouseModel

    f = path if isinstance(path, ExportFile) else read_export(path)
    file_cfg = f.cfg
    if cfg is not None and cfg != file_cfg:
        loud(
            f"{f.path}: the header's model config differs from the one given "
            f"({[k for k in vars(cfg) if getattr(cfg, k, None) != getattr(file_cfg, k, None)]}). "
            "Escape: drop the argument (the file IS what it is)."
        )
    model = DormouseModel(dataclasses.replace(file_cfg, factor_qat="none"), device=device)
    slots = set(model.state_dict())
    file_keys = f.names()
    missing = sorted(slots - file_keys)
    extra = sorted(file_keys - slots)
    if missing or extra:
        loud(
            f"export/cfg mismatch: {len(missing)} model key(s) absent from the file "
            f"{missing[:3]}, {len(extra)} file key(s) not in the model {extra[:3]}. "
            "Cause: the file was written from a different model config. Escape: "
            "read it with the header's own config (`read_export(p).cfg`)."
        )
    model.load_state_dict(f.tensors(), strict=True)
    return model


@contextmanager
def factor_qat_mode(model: nn.Module, mode: str, alpha: float = 1.0):
    """Force every QAT consumer in `model` to `(mode, alpha)` for the block.

    The walk is `DormouseModel.set_factor_qat_alpha`'s (every module carrying
    the attribute), plus the mode — `verify_export` uses it to build the
    reference forward that the file's numbers must match, and restores the
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


def verify_export(
    model: nn.Module,
    path: str | Path,
    batch,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Forward `batch` through the reference (projection on) and the loaded file;
    return the max abs logits difference.

    `batch` is `input_ids [b, t]` or an `(input_ids, targets)` pair (the targets
    are not read — only the logits are compared). The reference forward is the
    ORIGINAL model with every QAT consumer forced to the file's mode at
    `alpha=1.0` (the projection the export baked), and the model's own setting
    is restored on the way out; the loaded side is [`load_export`]'s model,
    whose factors already ARE the projection. A dense-dtype round trip shows up
    here too, which is the point: the number covers the whole file, not just the
    codec.
    """
    f = read_export(path)
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    dev = next(model.parameters()).device if device is None else torch.device(device)
    x = x.to(dev)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), factor_qat_mode(model, f.header["mode"], 1.0):
            ref, _, _ = model(x)
        loaded = load_export(f, device=dev)
        loaded.eval()
        with torch.no_grad():
            got, _, _ = loaded(x)
    finally:
        model.train(was_training)
    assert ref.shape == got.shape, f"logits shapes differ: {ref.shape} vs {got.shape}"
    return float((ref - got).abs().max())


def engine_refusal(cfg) -> str | None:
    """Why the browser engine cannot run `cfg`, or None — the fields, by name.

    The engine implements exactly one architecture (`ENGINE_SHAPE`, the `small`
    polygon); anything else is refused with the mismatching fields listed, not
    a warning. This is what the export command prints and what the page shows
    when it loads a file the engine cannot run.
    """
    bad = [
        f"{k}={getattr(cfg, k, None)!r} (engine implements {want!r})"
        for k, want in ENGINE_SHAPE.items()
        if getattr(cfg, k, None) != want
    ]
    return None if not bad else "; ".join(bad)
