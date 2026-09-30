"""Forward-mode AD in numpy, f64, over `gen_bwd_f64.forward`.

WHY FORWARD MODE AND NOT ONLY FINITE DIFFERENCES.  The adjoint under test is a
REVERSE-mode construction.  A forward-mode derivative is a different *method*,
not a second transcription of the same one, and it is exact to f64 round-off
(~1e-15) instead of carrying an O(h^4) truncation term.  So:

  * the forward-mode gradient is the ORACLE the Rust test compares against, and
  * the finite differences are the CHECK on the oracle.  If they disagree by
    more than the FD band, one of them is wrong and the generator says which.

Both are in the fixture's provenance because both are load-bearing.

`D` is a dual number: a value and its tangent.  Every operation the forward
uses is implemented once, so an operation the forward needs and this file does
not have is a loud `AttributeError`, not a silent zero.

`tools/gen_bwd_f64.py` owns the forward itself, the fixture, the faults and the
FD; this file owns the differentiation.
"""
import numpy as np


class D:
    __slots__ = ("v", "t")

    def __init__(self, v, t=None):
        self.v = v
        self.t = np.zeros_like(v) if t is None else t

    def __add__(self, o):
        o = _d(o)
        return D(self.v + o.v, self.t + o.t)

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, -self.t)

    def __sub__(self, o):
        o = _d(o)
        return D(self.v - o.v, self.t - o.t)

    def __rsub__(self, o):
        return _d(o) - self

    def __mul__(self, o):
        o = _d(o)
        return D(self.v * o.v, self.t * o.v + self.v * o.t)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _d(o)
        return D(self.v / o.v, (self.t * o.v - self.v * o.t) / (o.v * o.v))

    def __rtruediv__(self, o):
        return _d(o) / self

    def __getitem__(self, k):
        return D(self.v[k], self.t[k])

    def reshape(self, *s):
        return D(self.v.reshape(*s), self.t.reshape(*s))

    def __len__(self):
        return len(self.v)


def _d(x):
    return x if isinstance(x, D) else D(np.asarray(x, dtype=np.float64))


def matmul(a, b):
    a, b = _d(a), _d(b)
    return D(a.v @ b.v, a.t @ b.v + a.v @ b.t)


def swap(a, i, j):
    a = _d(a)
    return D(np.swapaxes(a.v, i, j), np.swapaxes(a.t, i, j))


def cat(parts, axis):
    """Concatenate duals.  The inclusive cumsum is built from this: `G[t]` is a
    DIFFERENT exponential per row, so the running sum cannot live in one slot."""
    return D(
        np.concatenate([p.v for p in parts], axis=axis),
        np.concatenate([p.t for p in parts], axis=axis),
    )


def exp(a):
    a = _d(a)
    e = np.exp(a.v)
    return D(e, a.t * e)


def inv_unit_lower(L, c):
    """`(I+L)^-1` for a STRICTLY lower triangular `L`, by forward substitution.

    Same recurrence and same algorithm as `forward.rs:603-608`: row `i` of the
    inverse is ASSIGNED from `-L[i, :i] @ inv[:i, :i]`, so the seed must be the
    IDENTITY and not `I + L`.  Seeding `I + L` and then assigning leaves
    `L[i,:i]` doubled in the untouched columns, which is silent: the routine
    returns a matrix, just the wrong one.  (It did.)
    """
    shape = L.v.shape
    v = np.zeros(shape)
    idx = np.arange(c)
    v[..., idx, idx] = 1.0
    t = np.zeros(shape)
    for i in range(1, c):
        prev_v, prev_t = v[:, :, :, :i, :i], t[:, :, :, :i, :i]
        arow_v, arow_t = L.v[:, :, :, i:i + 1, :i], L.t[:, :, :, i:i + 1, :i]
        v[:, :, :, i:i + 1, :i] = -(arow_v @ prev_v)
        t[:, :, :, i:i + 1, :i] = -(arow_t @ prev_v + arow_v @ prev_t)
    return D(v, t)


def forward_dual(inp, scale, c, nt, causal, strict):
    """`gen_bwd_f64.forward` with every intermediate a dual number.

    Returns `(outs, s_traj)`: the per-chunk outputs `[B,H,nt,c,V]` and the
    state trajectory.  `inp` values may be plain arrays (tangent 0) or `D`, which
    is how a seed is injected — `_d` passes a `D` through untouched.
    """
    B, H = _d(inp["q"]).v.shape[:2]
    fold = lambda t: _d(t).reshape(B, H, nt, c, _d(t).v.shape[3])
    q5, k5, v5, g5, b5, w5 = (fold(inp[n]) for n in ("q", "k", "v", "g", "b", "w"))

    parts, run = [], None
    for i in range(c):                       # inclusive cumsum along time
        run = g5[:, :, :, i:i + 1, :] if run is None else run + g5[:, :, :, i:i + 1, :]
        parts.append(run)
    G = cat(parts, 3)
    E = exp(G)
    kog, qg = k5 / E, q5 * E
    aqk = matmul(qg, swap(kog, 3, 4)) * (scale * causal)
    bk = b5 * k5
    akk = matmul(bk * E, swap(kog, 3, 4)) * strict
    mi = inv_unit_lower(akk, c)
    W, U = matmul(mi, bk * E), matmul(mi, w5 * v5)
    # LOG domain, both operands. `E` is exp(G); using it here would exponentiate
    # the cumsum twice, and the error is invisible in f64 arithmetic while
    # changing the function by O(1). FLA does the same thing
    # (`fla/ops/common/chunk_delta_h.py:236-240`, commit 9f38d249): it loads
    # the raw `g` for `b_g_last` and exponentiates only the difference.
    g_last_log = G[:, :, :, c - 1:c, :]
    g_last = exp(g_last_log)
    k_dec = k5 * exp(g_last_log - G)

    state = _d(inp["state"])
    traj, outs = [state], []
    for i in range(nt):
        v_new = U[:, :, i] - matmul(W[:, :, i], state)
        outs.append(matmul(aqk[:, :, i], v_new) + matmul(qg[:, :, i], state) * scale)
        state = state * swap(g_last[:, :, i], 2, 3) + matmul(swap(k_dec[:, :, i], 2, 3), v_new)
        traj.append(state)
    return cat(outs, 2), traj


def grad(inp, d_out, scale, chunk=16, names=None, progress=None):
    """Exact f64 forward-mode gradient of `<out, d_out>` w.r.t. the 7 inputs.

    One forward per COORDINATE, not per input: a one-hot seed at `x[i]` yields
    `dL/dx[i]` and nothing else, so the FULL tensor costs `sum(numel)` forwards.
    That is the point: a sampled probe (the 6-coordinate style in
    `tests/ops_batched_autodiff.rs`) passes an adjoint that is wrong at 99% of
    its coordinates, and this compares all 3200 of them.

    The seed must be injected as a DUAL (`D(value, tangent)`): handing the
    forward a plain array seeds the tangent at zero, and the whole gradient
    comes out identically zero — which looks like "the function is constant".
    """
    B, H, T, K = inp["q"].shape
    V, c = inp["v"].shape[3], chunk
    nt = T // c
    causal, strict = np.tril(np.ones((c, c))), np.tril(np.ones((c, c)), -1)
    for n in names or ("q", "k", "v", "g", "b", "w", "state"):
        out = np.empty(inp[n].shape)
        flat = out.reshape(-1)
        seed = np.zeros(inp[n].shape)
        for i in range(inp[n].size):
            seed.reshape(-1)[i] = 1.0
            src = dict(inp)
            src[n] = D(inp[n].copy(), seed.copy())
            o, _traj = forward_dual(src, scale, c, nt, causal, strict)
            flat[i] = (o.t.reshape(B, H, T, V) * d_out).sum()
            seed.reshape(-1)[i] = 0.0
        yield n, out
