"""ByteFlow parity: the burn crates' three oracle suites, ported to pytest.

The goldens are the VALUES carried by
`vendor/dormouse-fused/crates/burn-byteflow/tests/` — formula fixtures, not a
spec of that crate's call graph:

* **field** (`byteflow_field_oracle.rs` + `fixtures/byteflow_field_oracle.txt`) —
  Canon eq. (10) against a torch-cpu f64 computation (2e-7 absolute).
* **rate** (`byteflow_rate_oracle.rs` + `fixtures/byteflow_oracle.txt`) —
  `R_ε` / `ΔR` eq. (11)-(12) against a numpy f64 EIGEN route (2e-7 relative),
  plus the qualitative rows the fixture carries.
* **patcher** (`dynamic_patcher_oracle.rs`, its values computed by the test
  itself) — the streaming patcher against the paper's LITERAL T×T log-det by
  LU (1e-9 relative, f64 on both sides).

The two fixture files are byte-identical copies of the burn ones (`fixtures/
byteflow_oracle.txt`, `fixtures/byteflow_field_oracle.txt`); the fixture's own
comments name its generator. The patcher suite's inputs are a deterministic LCG
replicated from the burn test, and its expected values come from a DIFFERENT
decomposition of the same equation (LU on the T×T form vs our Cholesky on the
folded d×d form), so a wrong folding, a wrong `d/ε²` or a dropped ½ cannot
agree with it by construction.

Tolerances, named where they are used:

* `BAR_F32 = 2e-7` — absolute, the f32 round-trip floor of the field fixture
  (values are dumped at `%.9g`; `|out| ≲ 1`, and this is the bar the burn
  oracle itself uses).
* `BAR_F64 = 2e-7` — relative (`denom = max(|want|, 1)`), the burn rate oracle's
  own bar: our route is f64 but the crate's `marginal_gains_exact` returns an
  f32 tensor, and the fixture's dr values are compared against that.
* `BAR_EXACT = 1e-9` — relative, f64-vs-f64 on the patcher suite (LU vs
  Cholesky on a well-conditioned SPD matrix; measured agreement ~1e-13).

Everything runs on CPU in fp32/fp64 as the oracles do; no GPU is needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from dormouse.model.byteflow import (
    ByteFlowConfig,
    ByteFlowNet,
    CanonLayer,
    RateMode,
    RatePatcher,
    coding_rate_exact,
    effective_bins,
    marginal_gains_exact,
    marginal_gains_l2,
    select_positions,
)

FIXTURES = Path(__file__).parent / "fixtures"
#: The number of fixture text lines is not pinned; the VALUES are.
FIELD_TXT = (FIXTURES / "byteflow_field_oracle.txt").read_text()
RATE_TXT = (FIXTURES / "byteflow_oracle.txt").read_text()

BAR_F32 = 2e-7
BAR_F64 = 2e-7
BAR_EXACT = 1e-9


# ---------------------------------------------------------------------------
# the two fixture parsers (the format is the contract between generator and
# test — one reader per file, as on the burn side)
# ---------------------------------------------------------------------------


def _rel(got: float, want: float, denom_floor: float = 1.0) -> float:
    return abs(got - want) / max(abs(want), denom_floor)


def parse_rate_cases(text: str) -> dict[str, tuple[list[list[float]], dict[str, float]]]:
    """`case <name>` / `row <f32...>` / `<name> <f64>` — `byteflow_rate_oracle.rs:72-111`."""
    cases: dict[str, tuple[list[list[float]], dict[str, float]]] = {}
    name: str | None = None
    rows: list[list[float]] = []
    scalars: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        head = parts[0]
        if head == "case":
            if name is not None:
                cases[name] = (rows, scalars)
            name = " ".join(parts[1:])
            rows, scalars = [], {}
        elif head == "row":
            rows.append([float(v) for v in parts[1:]])
        else:
            # any other non-comment line is `<name> <f64>`; a malformed one is loud
            assert len(parts) == 2, f"malformed fixture line `{line}`"
            scalars[head] = float(parts[1])
    if name is not None:
        cases[name] = (rows, scalars)
    return cases


def h_from(rows: list[list[float]]) -> torch.Tensor:
    """Fixture rows (batch 1) → `[1, t, d]` f32, exactly as dumped."""
    assert rows, "fixture case has no `row` block"
    return torch.tensor(rows, dtype=torch.float32).reshape(1, len(rows), len(rows[0]))


def oracle_marginals(scalars: dict[str, float]) -> list[float]:
    """`dr1, dr2, …` in position order — the fixture's own key convention."""
    out: list[float] = []
    i = 1
    while f"dr{i}" in scalars:
        out.append(scalars[f"dr{i}"])
        i += 1
    return out


RATE_CASES = parse_rate_cases(RATE_TXT)


def _case(name: str) -> tuple[list[list[float]], dict[str, float]]:
    assert name in RATE_CASES, f"fixture case `{name}` missing"
    return RATE_CASES[name]


def _cmp_marginals(name: str, got: list[float], want: list[float]) -> None:
    assert len(got) == len(want), f"{name}: marginal length"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert _rel(g, w) < BAR_F64, f"{name} dr{i + 1}: {g:.12e} vs oracle {w:.12e}"


def _field_matrix(section: str) -> list[list[float]]:
    """The float rows immediately following the named section header."""
    out: list[list[float]] = []
    started = False
    for raw in FIELD_TXT.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if started:
                break
            continue
        if not started:
            started = line == section
            continue
        try:
            out.append([float(v) for v in line.split()])
        except ValueError:
            break
    assert out, f"fixture section `{section}` empty"
    return out


def _field_scalar(key: str) -> float:
    for raw in FIELD_TXT.splitlines():
        parts = raw.strip().split()
        if parts and parts[0] == key:
            assert len(parts) == 2, f"fixture scalar `{key}` malformed"
            return float(parts[1])
    raise AssertionError(f"fixture scalar `{key}` missing")


def _flat(rows: list[list[float]]) -> list[float]:
    return [v for row in rows for v in row]


# ---------------------------------------------------------------------------
# suite 1 — the field oracle: Canon, eq. (10)
# ---------------------------------------------------------------------------


def test_canon_layer_matches_the_torch_field_oracle():
    """`byteflow_field_oracle.rs:73-109`: our f32 Canon vs the torch-cpu f64 one.

    The fixture carries no RNG (every gate and input cell is a deterministic
    formula), so a disagreement names the transcription — the pad rule, the tap
    order or the gate layout — and not a seed.
    """
    d, t = int(_field_scalar("d")), int(_field_scalar("t"))
    assert (d, t) == (8, 12), "pinned by the generator's header"
    gates, inp, want = (
        _field_matrix("gates_f32"),
        _field_matrix("input_f32"),
        _field_matrix("output_f32"),
    )
    assert len(gates) == 4, "four taps, eq. (10)"
    assert all(len(r) == d for r in gates)

    layer = CanonLayer(d)
    with torch.no_grad():
        layer.gates.copy_(torch.tensor(gates, dtype=torch.float32))
    got = layer(torch.tensor(inp, dtype=torch.float32).reshape(1, t, d))
    got_flat = got.reshape(-1).tolist()
    assert len(got_flat) == t * d and len(want) == t
    for i, (g, w) in enumerate(zip(got_flat, _flat(want), strict=True)):
        assert abs(g - w) < BAR_F32, f"canon[{i}]: {g:.10} vs torch {w:.10}"


def test_canon_identity_init_and_tap_formula():
    """The crate's own in-crate values (`net.rs:743-808`), recomputed by hand.

    Identity init (`w0 = 1`, rest 0) must be an exact no-op, and with `w1 = 2`
    the output must be `h_t + 2·h_{t−1}` with `h_0` untouched — the two facts a
    shifted tap or a transposed gate layout breaks.
    """
    layer = CanonLayer(2)
    assert layer.gates.reshape(-1).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    h = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]], dtype=torch.float32)
    assert torch.equal(layer(h), h), "identity init must be a no-op"

    with torch.no_grad():
        layer.gates.copy_(
            torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        )
    got = layer(h).reshape(-1).tolist()
    for t in range(4):
        for c in range(2):
            expect = h[0, t, c].item() + (2.0 * h[0, t - 1, c].item() if t >= 1 else 0.0)
            assert abs(got[2 * t + c] - expect) < 1e-5, f"[{t},{c}]"


# ---------------------------------------------------------------------------
# suite 2 — the rate oracle: eq. (11)-(12)
# ---------------------------------------------------------------------------


def test_orthonormal_marginals_match_the_paper_constant():
    """k orthonormal rows: `R = k/2·ln(1 + d/ε²)` and every `ΔR_t` is that constant."""
    rows, s = _case("orthonormal_marginals")
    got = marginal_gains_exact(h_from(rows), s["eps2"]).reshape(-1).tolist()
    _cmp_marginals("orthonormal_marginals", got, oracle_marginals(s))
    closed = 0.5 * math.log(1.0 + 4.0 / s["eps2"])  # eq. (11) read on the eigenvalues
    assert abs(got[0] - closed) < 1e-6, f"{got[0]} vs closed form {closed}"


def test_rank_collapse_drops_the_marginal_and_stays_finite():
    """A repeated row lowers the marginal; the rate stays finite (the crate's
    `exact_rate_finite_on_rank_deficient`, `chunk.rs:189-196`)."""
    rows, s = _case("rank_collapse")
    got = marginal_gains_exact(h_from(rows), s["eps2"]).reshape(-1).tolist()
    _cmp_marginals("rank_collapse", got, oracle_marginals(s))
    assert s["rate"] < s["orthonormal_same_shape"], "a duplicate row must compress BETTER"
    assert got[2] < got[1] + BAR_F64, "ΔR at the repeated row < ΔR at the new direction"
    rate = coding_rate_exact(h_from(rows), s["eps2"]).item()
    assert math.isfinite(rate) and rate > 0.0


def test_outlier_is_the_marginal_argmax():
    """The Top-K criterion itself: the most novel position is the argmax of ΔR."""
    rows, s = _case("outlier")
    got = marginal_gains_exact(h_from(rows), s["eps2"]).reshape(-1).tolist()
    _cmp_marginals("outlier", got, oracle_marginals(s))
    assert max(range(len(got)), key=got.__getitem__) == int(s["argmax_dr"])


def test_marginals_telescope_to_the_full_rate():
    """`Σ ΔR_t = R(h_1:T)` — eq. (11) ↔ eq. (12) (`chunk.rs:271-278`)."""
    rows, s = _case("telescope")
    got = marginal_gains_exact(h_from(rows), s["eps2"]).reshape(-1).tolist()
    total = sum(got)
    assert _rel(total, s["rate"]) < BAR_F64 * 10.0, f"telescope {total} vs {s['rate']}"
    _cmp_marginals("telescope", got, oracle_marginals(s))


def test_prefix_rates_are_nonneg_and_the_cholesky_route_matches_eigen():
    """Every prefix rate ≥ 0, and our Cholesky value equals the fixture's
    EIGEN-route value (`byteflow_rate_oracle.rs:246-267`) — two decompositions
    of one formula agreeing is the only evidence a misread eq. (11) cannot fake."""
    rows, s = _case("signs")
    got = marginal_gains_exact(h_from(rows), s["eps2"]).reshape(-1).tolist()
    _cmp_marginals("signs", got, oracle_marginals(s))
    assert all(math.isfinite(v) and v >= 0.0 for v in got), "SPD prefix log-det ≥ 0"

    chol = coding_rate_exact(h_from(rows), s["eps2"]).item()
    eig = s["eigen_rate_last"]
    # The fixture's own two routes agree at f64 (what the burn oracle asserts)…
    assert abs(s["cholesky_rate_last"] - eig) / max(abs(eig), 1.0) < 1e-12, (
        f"fixture Cholesky {s['cholesky_rate_last']} vs eigen {eig}"
    )
    # …and OUR route reproduces it through an f32 return, so the bar here is the
    # f32 round-trip of a value of size 21 (5.4e-9 relative, measured).
    assert abs(chol - eig) / max(abs(eig), 1.0) < BAR_F64, f"Cholesky {chol} vs eigen {eig}"


def test_l2_fast_path_keeps_the_telescope_and_agrees_on_the_outlier():
    """Appendix B's L2 path against the exact rate on the same block.

    What is pinned is what an L2 misreading WOULD break: the telescope identity
    (a cumsum/shift error kills it) and the argmax on the one case the two
    criteria cannot disagree about. The rank agreement is NOT pinned — at init
    the criteria legitimately pick different top-K (the paper's Table 4 prices
    the difference at ~0.01 BPB of loss, not as rank equality).
    """
    rows, s = _case("signs")
    l2 = marginal_gains_l2(h_from(rows)).reshape(-1).tolist()
    total = sum(l2)
    assert math.isfinite(total) and total > 0.0
    assert abs(total - math.sqrt(sum(v * v for v in _flat(rows)))) < 1e-3, "L2 telescope"

    rows_o, s_o = _case("outlier")
    exact = marginal_gains_exact(h_from(rows_o), s_o["eps2"]).reshape(-1).tolist()
    l2_o = marginal_gains_l2(h_from(rows_o)).reshape(-1).tolist()

    def argmax(v: list[float]) -> int:
        return max(range(len(v)), key=v.__getitem__)

    assert argmax(exact) == argmax(l2_o) == int(s_o["argmax_dr"])


def test_exact_rate_matches_the_closed_form_orthonormal():
    """`chunk.rs:167-187`: three unit basis vectors in d=4 ⇒ `R = ½·3·ln(1 + d/ε²)`."""
    h = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    eps2 = 0.5
    expected = 0.5 * 3.0 * math.log(1.0 + 4.0 / eps2)
    got = coding_rate_exact(h.reshape(1, 3, 4), eps2).item()
    assert abs(got - expected) < 1e-5, f"exact rate {got} vs closed form {expected}"


def test_l2_gains_telescope_to_the_total_rate():
    """`chunk.rs:198-222`: ΣΔR = √S_T exactly, for any input."""
    torch.manual_seed(0)
    h = torch.randn(3, 17, 5, dtype=torch.float32)
    gains = marginal_gains_l2(h).detach()
    total = h.pow(2).sum(-1).sum(1).sqrt()
    summed = gains.sum(1)
    assert torch.allclose(total, summed, atol=1e-4), f"{total.tolist()} vs {summed.tolist()}"


def test_selection_is_bos_first_chronological_and_rank_driven():
    """`chunk.rs:224-268`: constant stream picks early positions, an outlier is
    always selected, and every row is `[0, ...]` strictly ascending."""
    flat = torch.ones(1, 16, 4, dtype=torch.float32)
    assert select_positions(marginal_gains_l2(flat), 4).reshape(-1).tolist() == [0, 1, 2, 3]

    mixed = flat.clone()
    mixed[0, 10] = 100.0
    sel = select_positions(marginal_gains_l2(mixed), 4).reshape(-1).tolist()
    assert 10 in sel, f"the outlier must be selected, got {sel}"

    torch.manual_seed(1)
    gains = torch.rand(4, 33)
    sel = select_positions(gains, 7)
    assert sel.shape == (4, 7)
    for row in sel.tolist():
        assert row[0] == 0, "position 0 (BOS) forced"
        assert all(a < b for a, b in zip(row, row[1:], strict=False)), f"must ascend: {row}"


def test_rate_oracle_falsifies_the_three_mutants():
    """A gate that cannot fail measures nothing: the missing-½, the ε²→ε⁴
    misreading and the future-right (post-increment) rate must each go red
    against the same fixture rows the real implementation is green under."""
    rows, s = _case("rank_collapse")
    h = h_from(rows)
    want = oracle_marginals(s)
    base = marginal_gains_exact(h, s["eps2"]).reshape(-1).tolist()

    mutants = {
        "no-half": [v * 2.0 for v in base],
        "eps4": marginal_gains_exact(h, s["eps2"] * s["eps2"]).reshape(-1).tolist(),
        "shift": base[-1:] + base[:-1],  # the future-right reading of eq. (12)
    }
    for name, got in mutants.items():
        assert any(_rel(g, w) > BAR_F64 for g, w in zip(got, want, strict=True)), (
            f"mutant `{name}` stayed within the oracle bar"
        )


# ---------------------------------------------------------------------------
# suite 3 — the dynamic patcher oracle (`dynamic_patcher_oracle.rs`)
# ---------------------------------------------------------------------------


class Lcg:
    """The burn tests' deterministic LCG, bit for bit (`stream.rs:183-193`).

    u64 wrapping arithmetic; the f32 division is by `u32::MAX as f32`, which is
    exactly 2^32 in f32, so the draw is `(x >> 33) / 2^32 · 2 − 1`.
    """

    MUL = 6364136223846793005
    ADD = 1442695040888963407
    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_f32(self) -> float:
        self.state = (self.state * self.MUL + self.ADD) & self.MASK
        n = self.state >> 33
        return float(
            torch.tensor(float(n), dtype=torch.float32)
            / torch.tensor(4294967295.0, dtype=torch.float32)
            * 2.0
            - 1.0
        )

    def row(self, d: int, scale: float = 1.0) -> list[float]:
        return [self.next_f32() * scale for _ in range(d)]


def _direct_rate_t_by_t(rows: torch.Tensor, t: int, d: int, c: float) -> float:
    """The paper's LITERAL eq. (11): `I_T + (d/ε²)·H·Hᵀ`, log-det by LU."""
    h = rows.reshape(t, d).to(torch.float64)
    m = c * (h @ h.T)
    m.diagonal().add_(1.0)
    return 0.5 * float(torch.linalg.slogdet(m).logabsdet)


def _direct_rate_d_by_d(rows: torch.Tensor, t: int, d: int, c: float) -> float:
    """The d×d reading of the same equation — the Sylvester folding's cross-check."""
    h = rows.reshape(t, d).to(torch.float64)
    m = c * (h.T @ h)
    m.diagonal().add_(1.0)
    return 0.5 * float(torch.linalg.slogdet(m).logabsdet)


def test_streaming_patcher_matches_the_direct_logdet():
    """The patcher's rate — eq. (11) folded through Cholesky + Sylvester — against
    a third decomposition of the same equation (LU on the unfolded T×T form)."""
    rng = Lcg(0xBEEF)
    t, d, eps2 = 24, 8, 0.5
    c = d / eps2
    rows = [rng.row(d) for _ in range(t)]
    flat = torch.tensor(rows, dtype=torch.float32)

    p = RatePatcher(d, eps2, RateMode.LOGDET)
    gains = [p.push(r) for r in rows]
    streamed = p.rate_()

    direct = _direct_rate_t_by_t(flat, t, d, c)
    assert _rel(streamed, direct, denom_floor=0.0) < BAR_EXACT, (
        f"streamed {streamed:.14e} vs direct T×T {direct:.14e}"
    )
    direct_d = _direct_rate_d_by_d(flat, t, d, c)
    assert _rel(direct, direct_d, denom_floor=0.0) < BAR_EXACT, "Sylvester folding"

    # Every ΔR against the direct prefix differences, computed from scratch.
    prev = 0.0
    for i, g in enumerate(gains):
        want = _direct_rate_t_by_t(flat[: i + 1], i + 1, d, c) - prev
        assert _rel(g, want, denom_floor=1e-12) < BAR_EXACT, f"ΔR_{i + 1}"
        prev += want
    total = sum(gains)
    assert _rel(total, streamed, denom_floor=0.0) < BAR_EXACT, "telescope"

    # Falsify: the two eq. (11) reading errors must be red against the same route.
    assert _rel(2.0 * streamed, direct, denom_floor=0.0) > 1e-3, "no-half mutant"
    p4 = RatePatcher(d, eps2 * eps2, RateMode.LOGDET)
    for r in rows:
        p4.push(r)
    assert _rel(p4.rate_(), direct, denom_floor=0.0) > 1e-3, "ε²→ε⁴ mutant"


def test_delta_r_topk_borders_land_on_the_planted_bursts():
    """A stream whose information profile is known by construction: the Top-K
    borders land on the three 100x-norm bursts, keep the BOS first, stay
    chronological, and equal the batch path's own selection within each mode."""
    t, d, eps2, k = 64, 8, 0.5, 5
    planted = [10, 30, 50]
    rng = Lcg(42)
    rows = [rng.row(d) for _ in range(t)]
    for pos in planted:
        rows[pos] = [v * 100.0 for v in rows[pos]]
    flat = torch.tensor(rows, dtype=torch.float32).reshape(1, t, d)

    for mode in (RateMode.LOGDET, RateMode.L2):
        p = RatePatcher(d, eps2, mode)
        gains = [p.push(r) for r in rows]
        borders = p.borders(k)
        assert len(borders) == k, f"{mode}: exactly K borders"
        assert borders[0] == 0, f"{mode}: BOS forced first"
        assert all(a < b for a, b in zip(borders, borders[1:], strict=False)), (
            f"{mode}: chronological, got {borders}"
        )
        for pos in planted:
            assert pos in borders, f"{mode}: burst at {pos} missing from {borders}"
        ranked = sorted(range(1, t), key=lambda a: (-gains[a], a))[: k - 1]
        assert sorted([0, *ranked]) == borders, f"{mode}: top-(K−1) + BOS"

    # Degenerate Ks (`stream.rs:150-175`).
    p = RatePatcher(d, eps2, RateMode.LOGDET)
    for r in rows:
        p.push(r)
    assert p.borders(1) == [0]
    assert p.borders(t) == list(range(t))

    # Streaming == batch, within each mode; and the streamed LogDet gains are the
    # batch values themselves (the streamed path is f64, the batch one returns
    # f32, so the bar is that f32 round-trip).
    batch_exact = marginal_gains_exact(flat, eps2).reshape(-1).tolist()
    batch_l2 = marginal_gains_l2(flat).reshape(-1).tolist()
    for mode, batch in ((RateMode.LOGDET, batch_exact), (RateMode.L2, batch_l2)):
        p = RatePatcher(d, eps2, mode)
        stream_gains = [p.push(r) for r in rows]
        assert p.borders(k) == select_positions(torch.tensor([batch]), k).reshape(-1).tolist(), (
            f"{mode}: streaming borders != batch borders"
        )
        # Gain VALUES: only the LogDet pair shares the f64 route, and only its
        # f32 return separates it (2e-7, the crate's own bar). The L2 batch path
        # accumulates in f32 through a cumsum the planted burst inflates to ~1e3,
        # so its late ΔR values carry ~1e-4 ABSOLUTE noise — that pair is
        # compared absolutely, which still catches an off-by-one (the burst moves
        # its neighbours' ΔR by ~1e-2).
        for i, (g, w) in enumerate(zip(stream_gains, batch, strict=True)):
            if mode is RateMode.LOGDET:
                assert _rel(g, w, denom_floor=1e-12) < 2e-7, f"{mode} ΔR_{i}"
            else:
                assert abs(g - w) < 1e-3, f"{mode} ΔR_{i}: {g} vs {w}"

    # Falsify: the fixed-stride chunker these borders replace cannot land on the
    # planted bursts at once — if it could, the gate would measure nothing.
    stride = [i * t // k for i in range(k)]
    assert any(pos not in stride for pos in planted)


def test_patcher_reset_starts_a_fresh_sequence():
    """`stream.rs:219-232`: the state is a sum, not a carry — the restarted
    window reproduces the first window's rate exactly."""
    p = RatePatcher(3, 0.5, RateMode.L2)
    p.push([1.0, 2.0, 3.0])
    r1 = p.rate_()
    p.push([4.0, 5.0, 6.0])
    assert p.rate_() > r1
    p.reset()
    assert p.rate_() == 0.0 and p.is_empty()
    p.push([1.0, 2.0, 3.0])
    assert p.rate_() == r1

    # And the window's gains are a telescoping decomposition of its own rate.
    gains = p.gains()
    assert abs(sum(gains) - p.rate_()) < 1e-12


def test_l2_push_matches_the_batch_cumsum():
    """`stream.rs:195-217`: the streaming L2 state IS the batch path's running norm."""
    rng = Lcg(7)
    t, d = 17, 5
    rows = [rng.row(d) for _ in range(t)]
    p = RatePatcher(d, 0.5, RateMode.L2)
    got = [p.push(r) for r in rows]
    total = 0.0
    prev = 0.0
    for i, r in enumerate(rows):
        total += sum(float(v) * float(v) for v in r)
        rate = math.sqrt(total)
        assert abs(rate - prev - got[i]) < 1e-12, f"gain {i}"
        prev = rate
    assert len(p) == t
    assert abs(sum(got) - p.rate_()) < 1e-12


def test_effective_bins_window_arithmetic():
    """`net.rs:700-711`'s four values: the upsample's bin count is the largest
    divisor of T not exceeding B (1 always divides)."""
    assert effective_bins(16, 512) == 16, "divisible: full B"
    assert effective_bins(16, 24) == 12, "largest divisor of 24 ≤ 16"
    assert effective_bins(8, 11) == 1, "prime above bins: single bin"
    assert effective_bins(8, 7) == 7, "t below bins: per-position bins"


# ---------------------------------------------------------------------------
# the ported net's own properties (`net.rs`'s in-crate tests)
# ---------------------------------------------------------------------------


def tiny_net_config(**over) -> ByteFlowConfig:
    """The crate's `tiny_config` (`net.rs:540-558`) — same shape, same numbers."""
    base = dict(
        d_local=16,
        d_global=32,
        k_tokens=6,
        e_layers=2,
        g_layers=2,
        n_heads_local=2,
        n_heads_global=2,
        w_local=8,
        d_ff_local=32,
        d_ff_global=48,
        bins=8,
        eps2=0.5,
        max_bytes=64,
    )
    base.update(over)
    return ByteFlowConfig(**base)


def test_output_shape_is_static_across_input_lengths():
    """Any T divisible by `bins` works with the SAME model (`net.rs:560-570`)."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    for t in (16, 32, 64):
        x = torch.zeros(2, t, dtype=torch.long)
        assert tuple(net(x).shape) == (2, t, 256), f"T={t}"


def test_forward_is_deterministic():
    """`net.rs:686-694`: two identical forwards are equal, bit for bit."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    bytes_ = torch.zeros(1, 24, dtype=torch.long)
    assert torch.equal(net(bytes_), net(bytes_))


def test_encode_decode_split_matches_the_monolithic_forward():
    """The channel's contract (`net.rs:716-726`): front → our loop → back is
    bit-identical to the net's own forward when the "loop" IS its global blocks."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    bytes_ = torch.zeros(2, 16, dtype=torch.long)
    mono = net(bytes_)
    h, z, sel = net.encode_chunks(bytes_, net.k_tokens)
    for blk in net.global_blocks:
        z = blk(z)
    assert torch.equal(mono, net.decode_chunks(z, h, sel))


def test_sliding_window_attention_respects_the_window():
    """`net.rs:572-600`: a byte beyond `w_local` + the Canon kernel's reach must
    not change earlier outputs — a future/past leak is an O(1) difference."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    net.eval()
    x = torch.randn(1, 16, 16)
    base = net.encoder[0](x)
    perturbed = x.clone()
    perturbed[0, 15] = 5.0
    changed = net.encoder[0](perturbed)
    delta = (base[0, :8] - changed[0, :8]).abs().max().item()
    assert delta < 1e-5, f"future leaked into past: {delta}"


def test_arbitrary_lengths_upsample_with_fallback_bins():
    """T not dividing `bins` still produces logits (`net.rs:699-711`)."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())  # bins=8
    assert tuple(net(torch.zeros(2, 12, dtype=torch.long)).shape) == (2, 12, 256)


def test_rate_mode_logdet_runs_finite_end_to_end():
    """`net.rs:728-740`: the exact-rate chunker is finite through a whole forward."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config(rate_mode=RateMode.LOGDET))
    logits = net(torch.zeros(1, 16, dtype=torch.long))
    assert torch.isfinite(logits).all()


def test_backward_reaches_every_parameter_finitely():
    """`net.rs:638-684`: every parameter — including the Canon gates, `proj` and
    the stacked upsample weights — receives a finite gradient. A parameter with
    no gradient would be an arm that runs and trains nothing."""
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    bytes_ = torch.tensor([[7, 3, 200, 41, 90, 5, 12, 88, 60, 1, 250, 17, 33, 76, 99, 4]])
    net(bytes_).pow(2).sum(-1).mean().backward()
    names = {n for n, p in net.named_parameters() if p.grad is not None}
    all_names = {n for n, _ in net.named_parameters()}
    assert all_names - names == set(), f"no gradient: {sorted(all_names - names)}"
    assert any("canon" in n for n in all_names)
    assert "proj.weight" in all_names and "upsample_w" in all_names
    for n, p in net.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {n}"


def test_rates_vs_fields_ordering():
    """The two stages' ORDER is the architecture: a burst's `ΔR` peaks, the
    chunker promotes it, and the promoted rows are what the global stage sees.

    Asserted as an ordering, not a value: the promoted patch at the burst must
    carry the burst's own row — which is the property a wrong gather or a
    wrong chunk index breaks (both would still produce `[b,k,d]` tensors).
    """
    torch.manual_seed(0)
    net = ByteFlowNet(tiny_net_config())
    x = torch.randint(0, 256, (1, 32), dtype=torch.long)
    h = net.embedding(x)
    for blk in net.encoder:
        h = blk(h)
    gains = marginal_gains_l2(h).detach()
    _, z, sel = net.encode_chunks(x, 6)
    assert sel.shape == (1, 6) and sel[0, 0].item() == 0
    assert sel.reshape(-1).tolist() == sorted(sel.reshape(-1).tolist()), "chronological"
    # The promoted positions ARE the top-(K−1) of the coding rate plus BOS: the
    # ordering ΔR → selection → patches is the architecture, and a wrong gain
    # reading shows up as a different promotion set, not as a shape error.
    ranked = sorted(range(1, 32), key=lambda a: (-float(gains[0, a]), a))[:5]
    assert sel.reshape(-1).tolist() == sorted([0, *ranked])
    # And the K rows the global stage sees ARE the selected rows of h.
    assert torch.equal(z, net.proj(h.gather(1, sel[..., None].expand(-1, -1, 16))))


@pytest.mark.parametrize("t", [8, 16, 32])
def test_upsample_is_a_per_position_bin_linear_map(t):
    """Stage 4's arithmetic: `s_t = h_t + g_chunk(t)·W_bin(t)` — checked against a
    hand-built per-position reference (the batched-GEMM reshape must be that)."""
    torch.manual_seed(0)
    cfg = tiny_net_config()
    net = ByteFlowNet(cfg)
    k, b = 4, 1
    g = torch.randn(b, k, cfg.d_global)
    sel = torch.tensor([[0, 1, t // 2, t - 1]])
    bins = effective_bins(cfg.bins, t)
    got = net.upsample(g, sel, t, b, bins)
    per = t // bins
    for pos in range(t):
        chunk = int((sel[0] <= pos).sum()) - 1
        b_i = min(pos // per, bins - 1)  # ⌊t·B/T⌋ == pos // per for T = B·per
        want = g[0, chunk] @ net.upsample_w[b_i]
        assert torch.allclose(got[0, pos], want, atol=1e-5), f"pos {pos}"
