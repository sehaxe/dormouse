"""Tensor-level reader for a DMCK checkpoint.

The brief's numbers came from comparing the checkpoint FILE. That instrument is
contaminated: the file carries NaN slots and |v|>=1e30 slots whose count and
positions move between runs. This parses the burnpack record into NAMED
TENSORS with their byte ranges, so a comparison is a comparison of parameter
tensors, and NaN/huge contamination is reported separately instead of being
silently counted as a difference.

Layout (measured, not assumed):
  0..8    magic b"DMCK\0\2\0\0"
  8..48   u64 step, u64 model_len, u64 optim_len, u64 teacher_len, u64 flags
  then three burnpack records: model, optim, teacher
  record: b"NRUB" | u16 ver | u32 namelen | names-blob | metadata | data
  every metadata entry carries data_offsets: (u32,u32) = BYTE range, absolute
  from the start of the record, plus a shape.
"""
import struct
import numpy as np

MAGIC = b"DMCK\x00\x02\x00\x00"
BIG = 1e30  # the brief's "huge" threshold
DO = b"ldata_offsets"


def read_sections(path):
    b = open(path, "rb").read()
    assert b[:8] == MAGIC, f"not a DMCK v2 checkpoint: {b[:8]!r}"
    step, mlen, olen, tlen, flags = struct.unpack("<QQQQQ", b[8:48])
    off = 48
    return {
        "step": step, "flags": flags, "size": len(b),
        "model": b[off:off + mlen],
        "optim": b[off + mlen:off + mlen + olen],
        "teacher": b[off + mlen + olen:off + mlen + olen + tlen],
    }


def _cbor(buf, p):
    """Minimal CBOR reader -> (value, next_pos). Enough for the record metadata."""
    ib = buf[p]
    mt, ai = ib >> 5, ib & 0x1F
    p += 1
    if ai < 24:
        val = ai
    elif ai == 24:
        val = buf[p]; p += 1
    elif ai == 25:
        val = int.from_bytes(buf[p:p + 2], "big"); p += 2
    elif ai == 26:
        val = int.from_bytes(buf[p:p + 4], "big"); p += 4
    elif ai == 27:
        val = int.from_bytes(buf[p:p + 8], "big"); p += 8
    else:
        raise ValueError(f"indefinite/unsupported ai={ai} at {p-1}")
    if mt == 0:
        return val, p
    if mt == 3:
        return buf[p:p + val].decode(), p + val
    if mt == 4:
        out = []
        for _ in range(val):
            v, p = _cbor(buf, p)
            out.append(v)
        return out, p
    if mt == 5:
        out = {}
        for _ in range(val):
            k, p = _cbor(buf, p)
            v, p = _cbor(buf, p)
            out[k] = v
        return out, p
    if mt == 7:
        return {20: False, 21: True, 22: None}.get(ai, f"simple({ai})"), p
    raise ValueError(f"major type {mt} ai={ai}")


def tensors(rec):
    """-> [(name, shape, start_byte, end_byte)] with ABSOLUTE byte offsets.

    The record is: b"NRUB" | u16 ver | u32 size, then a CBOR map
    {"tensors": {name -> {dtype, shape, data_offsets, param_id}}} in [0, 5632),
    then the f32 data, then nothing -- the data section ENDS at the record end.

    `data_offsets` are RELATIVE to the data section, which starts at

        base = len(record) - (sum(sizes) + sum(internal gaps))

    where the internal gaps are the 3 padding runs visible between consecutive
    declared ranges (208 + 252 + 252 bytes; without subtracting them the solve
    lands 712 B high and every tensor reads misaligned float data).

    THIS IS SELF-VALIDATING, because a wrong base does not raise -- it silently
    returns a window shifted by a few thousand bytes, which is still dense
    plausible f32 data for 53 of 54 tensors. That is not a hypothetical: base=0
    read every tensor 1408 floats early, produced only 398 absurd values (all in
    the first tensor, where the window lands on the record's own header instead
    of on data), and produced a table of numbers that were wrong. The
    plausibility assertion below is what catches it. Measured: base=5632 gives
    0 values >=1e28 and 0 NaN across all 9,197,390 slots of all six runs; the
    nearest wrong base (5631) gives 961,989 absurd values.
    """
    assert rec[:4] == b"NRUB", rec[:4]
    body, p = _cbor(rec, 10)
    body = body["tensors"]
    raw = []
    for name, meta in body.items():
        assert meta["dtype"] == "F32", (name, meta["dtype"])
        s, e = meta["data_offsets"]
        shape = list(meta["shape"])
        n, exp = (e - s) // 4, int(np.prod(shape)) if shape else 1
        assert n == exp, (name, s, e, shape, n, exp)
        raw.append((name, shape, s, e))
    assert raw, "empty tensor map"
    srt = sorted((s, e) for _, _, s, e in raw)
    gaps = sum(max(0, srt[i][0] - srt[i - 1][1]) for i in range(1, len(srt)))
    base = len(rec) - (sum(e - s for _, _, s, e in raw) + gaps)
    assert base >= p, (base, p)          # data may not overlap the metadata
    out = [(n, sh, s + base, e + base) for n, sh, s, e in raw]
    # SELF-CHECK: at the right base every slot is a plausible init value.
    for n, sh, s, e in out:
        a = np.frombuffer(rec, dtype="<f4", count=(e - s) // 4, offset=s)
        f = a[np.isfinite(a)]
        assert f.size and np.abs(f).max() < 1e6, (
            f"base {base} is wrong: {n} has max|v|={np.abs(f).max():.3e}")
    return out


def load(path, section="model"):
    """-> {tensor name: np.float32 array}"""
    rec = read_sections(path)[section]
    out = {}
    for name, shape, s, e in tensors(rec):
        a = np.frombuffer(rec, dtype="<f4", count=(e - s) // 4, offset=s)
        out[name] = a.reshape(shape) if shape else a
    return out


def clean_mask(a):
    b = a.astype(np.float32)
    return np.isfinite(b) & (np.abs(b) < BIG)


def compare(ta, tb, verbose=False):
    """Per-tensor value comparison over CLEAN slots only (finite, |v|<1e30)."""
    assert set(ta) == set(tb), set(ta) ^ set(tb)
    rows, tot = [], dict(n=0, diff=0, nan=0, big=0, clean=0, nbig=0)
    for k in sorted(ta):
        a, b = ta[k].astype(np.float32).ravel(), tb[k].astype(np.float32).ravel()
        assert a.shape == b.shape, (k, a.shape, b.shape)
        ma, mb = clean_mask(a), clean_mask(b)
        m = ma & mb
        d = a[m] != b[m]
        ca, cb = classify(a), classify(b)
        rows.append((k, a.size, int(m.sum()), int(d.sum()), ca, cb))
        tot["n"] += a.size
        tot["diff"] += int(d.sum())
        tot["nan"] += ca["nan"]
        tot["big"] += ca["big"]
        tot["nbig"] += cb["nan"] + cb["big"]
        tot["clean"] += int(m.sum())
    if verbose:
        for k, n, m, d, ca, cb in rows:
            if d or ca["nan"] or ca["big"] or cb["nan"] or cb["big"]:
                print(f"  {k:36s} n={n:8d} clean={m:8d} diff={d:7d} "
                      f"nan={ca['nan']}/{cb['nan']} big={ca['big']}/{cb['big']}")
    return rows, tot


def classify(a):
    f = a.astype(np.float32)
    fin = np.isfinite(f)
    return {"n": f.size, "nan": int(np.isnan(f).sum()),
            "inf": int(np.isinf(f).sum()),
            "big": int((fin & (np.abs(f) >= BIG)).sum())}
