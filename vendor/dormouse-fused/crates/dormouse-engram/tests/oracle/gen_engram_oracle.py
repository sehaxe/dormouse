#!/usr/bin/env python3
"""Golden-vector generator for the Engram oracle.

Runs the OFFICIAL reference implementation
(``deepseek-ai/Engram@main:engram_demo_v1.py``) and writes the tensors a
dormouse-engram test compares against.

    curl -sSLO https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py
    python3 gen_engram_oracle.py engram_demo_v1.py ../fixtures/engram_oracle.txt

Every number in the fixture comes out of the reference's own forward pass. This
file transcribes no formula: where a column looks like a restatement, it is the
reference's `Engram.forward` / `ShortConv.forward` code with ONE of its tokens
changed (marked `# VAR:`), so a Rust test can tell which side of a distinction
our implementation is on instead of merely checking that it is "close".

The two substitutions, and why neither hides anything:

1. ``CompressedTokenizer`` -> an identity map over 256 ids. The reference
   compresses DeepSeek-V3 token ids into a de-duplicated string-keyed vocab;
   we hash BYTES (vocab 256). This is the only step a byte-level model cannot
   inherit, and nothing numeric in this fixture depends on it: the hash VALUES
   are pinned structurally (the prime ladder) and by property, never as
   numbers, because the multipliers come from numpy PCG64 on one side and
   splitmix64 on the other. Everything downstream of the lookup is the
   reference's own code.
2. Reference RMSNorm gains set to 1 and Linear biases set to 0 before the
   numeric run, because dormouse-engram has neither (see the structural columns).
   Gains==1 and bias==0 are legal parameter settings, so the comparison stays
   exact; the MISSING parameters are recorded separately and asserted by the
   Rust test rather than papered over here.

`# VAR: eps` is the important one. The reference's `nn.RMSNorm(hidden_size)`
defaults to `eps=None`, which torch resolves to `torch.finfo(f32).eps` =
1.19e-7; dormouse-engram hardcodes `1e-5`. That is a real divergence, so the
fixture carries BOTH: `*_shipped` (the reference exactly as published) and
`*_eps1e5` (the reference with eps set to 1e-5, i.e. what dormouse-engram computes).
Matching the second tightly and missing the first by a measured amount is what
turns "there is a difference" into "the difference is exactly this, and nothing
else".
"""
from __future__ import annotations

import importlib.util

import math
import os
import sys
import types

import numpy as np
import torch

REFERENCE_URL = (
    "https://raw.githubusercontent.com/deepseek-ai/Engram/main/engram_demo_v1.py"
)

# Shapes. Deliberately small: this is a differential test, not a benchmark.
GATE_D = 64
GATE_L = 19  # len(GATE_S)
MOD_B, MOD_L, MOD_HC, MOD_D = 2, 6, 3, 64
MOD_TABLES, MOD_EMBED = 8, 24  # per-head embed dim; total_embed = 192
CONV_K, CONV_DIL = 4, 3  # reference: kernel_size=4, dilation=max_ngram_size=3

# The RMS eps dormouse-engram hardcodes (lib.rs:219, 225, 204).
BURN_RMS_EPS = 1e-5

# Gate fixtures: the |s| values that decide whether this test can see anything.
#
# gate = sigmoid(sqrt(max(|s|, 1e-6)) * sign(s)), s = <k, q>/sqrt(D).
# `clamp_min(1e-6)` and `add(1e-6)` are IDENTICAL at s = 0 (both give
# sqrt(1e-6)) and converge as |s| grows; they separate only just ABOVE the
# threshold, where `add` doubles the radicand. So the discriminating band is
# 1e-6 < |s| and it is widest at the threshold. Below 1e-6 they agree again
# (both are ~sqrt(1e-6)); rows 0-3 are controls that prove the fixture's
# precision claim does not extend there.
GATE_S = [
    0.0,  # exact zero: identical under both formulas. Control.
    1e-9,  # far below: agree to ~1e-12. Control.
    9.9e-7,  # just below: agree to ~0.4%. Control.
    1.0e-6,  # AT the threshold: add gives sqrt(2e-6), clamp sqrt(1e-6).
    1.001e-6,
    1.05e-6,
    1.2e-6,
    1.5e-6,
    2.0e-6,
    2.5e-6,
    5.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    0.05,
    0.125,
    0.5,
    2.0,  # |s| <= sqrt(D) = 8 is the ceiling: a normalised pair has L2 norm
    3.0,  # sqrt(D), so s = sqrt(D)*<u,v> and |<u,v>| <= 1.
    -3.0,  # negative s: exercises the sign() arm.
    -1.05e-6,
]
# Rows whose |s| lands in the discriminating band: just above 1e-6, where
# `add` doubles the radicand and the gate moves by >= 8e-5, while the
# reference's own f32 error there is still <= 8e-6. The upper edge is 3e-6
# because the separation shrinks as 1/sqrt(|s|) while the f32 error grows.
# The Rust test asserts the fixture actually reaches this band; if it ever
# stops reaching it, the test goes red instead of silently becoming blind.
GATE_BAND = (1e-6, 3e-6)

# (vocab_size, max_ngram, n_head, layer_ids) for the prime ladder.
# 256 is our byte vocabulary; 25_000 is the shipped `small` engram_rows; the
# rest are shapes where a power-of-two table would still look plausible.
LADDERS = [
    (256, 3, 4, [1]),
    (256, 4, 8, [1]),
    (25_000, 3, 8, [1]),
    (25_000, 3, 4, [1, 15]),
    (1_000_000, 3, 8, [1]),
    (646_400, 4, 8, [1]),
]


def install_stubs() -> None:
    """Stub the two imports the reference only needs for its tokenizer.

    `transformers` / `tokenizers` become empty modules so the file imports with
    no HuggingFace round trip. Nothing in the Engram math touches them;
    `CompressedTokenizer` is swapped out separately.
    """
    for name in ("transformers", "tokenizers"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        if name == "transformers":
            mod.AutoTokenizer = object  # type: ignore[attr-defined]
        else:
            sub = types.ModuleType("tokenizers.normalizers")
            sub.NFKC = sub.NFD = sub.StripAccents = object  # type: ignore[attr-defined]
            sub.Lowercase = sub.Strip = object  # type: ignore[attr-defined]
            sub.Sequence = sub.Replace = object  # type: ignore[attr-defined]
            mod.normalizers = sub  # type: ignore[attr-defined]
            mod.Regex = object  # type: ignore[attr-defined]
        sys.modules[name] = mod


class IdentityTokenizer:
    """Stand-in for `CompressedTokenizer`: byte values in, byte values out.

    The real one maps DeepSeek token ids into a de-duplicated string-keyed
    vocabulary. Over 256 ids that is the identity, and it is the one step a
    byte-level model cannot inherit.
    """

    def __init__(self, tokenizer_name_or_path: str = "", vocab_size: int = 256) -> None:
        self.lookup_table = np.arange(vocab_size, dtype=np.int64)
        self.num_new_token = vocab_size

    def __len__(self) -> int:
        return self.num_new_token

    def __call__(self, input_ids):
        return np.asarray(input_ids, dtype=np.int64)


def load_reference(path: str):
    install_stubs()
    spec = importlib.util.spec_from_file_location("engram_demo_v1", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CompressedTokenizer = IdentityTokenizer
    return mod


def f32_list(t) -> list[float]:
    """Shortest round-tripping decimal for every f32, in row-major order.

    Rust's f32 parser is correctly rounded, so these parse back bit-identical.
    """
    a = t.detach().to(torch.float32).contiguous().numpy()
    return [float(repr(x)) for x in a.reshape(-1).tolist()]


def rms_norm_like(n: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a reference `nn.RMSNorm` (used with weight == 1, i.e. plain)."""
    return n(x)


def gate_from_normed(normed_key: torch.Tensor, normed_query: torch.Tensor,
                     d: int, add_variant: bool, reverse_dot: bool = False) -> torch.Tensor:
    """`Engram.forward` lines 371-373, with one token swappable.

    shipped: `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    VAR:     `gate.abs().add(1e-6).sqrt() * gate.sign()`

    `reverse_dot` sums the 64 products back to front. The arithmetic is
    identical and the mathematics is unchanged, so the difference between the
    two orders is a pure measurement of how much this formula's answer MOVES
    under a perturbation that no one would call changing the model.
    """
    prod = normed_key * normed_query
    gate = (prod.flip(-1) if reverse_dot else prod).sum(dim=-1) / math.sqrt(d)
    gate = gate.abs().add(1e-6).sqrt() * gate.sign() if add_variant else \
        gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    return gate.sigmoid()


def solve_pair(d: int, target_s: float, seed: int):
    """key/query whose reference-normalised dot gives `target_s`.

    RMSNorm outputs RMS == 1, so a normalised d-vector has L2 norm sqrt(d), NOT
    1. With u, v unit and k = sqrt(d)*u, q = sqrt(d)*v the reference computes
    s = <k, q>/sqrt(d) = sqrt(d) * <u, v>, hence `rho = target_s / sqrt(d)`.
    RMSNorm is scale invariant, so only the DIRECTIONS matter; the magnitudes
    are lopsided on purpose so the fixture also exercises that invariance.
    Built in float64, emitted as float32.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    w = rng.standard_normal(d)
    w -= (w @ u) * u  # orthogonal complement
    w /= np.linalg.norm(w)
    rho = target_s / math.sqrt(d)  # <u, v_hat> wanted
    if abs(rho) > 1.0:
        raise ValueError(f"|s|/sqrt(d) = {rho} is unreachable for unit vectors")
    # v = rho*u + sqrt(1-rho^2)*w has unit norm and <u, v> = rho exactly, for
    # either sign. Tilting u by +-c*w only ever reaches +|rho|.
    v = rho * u + math.sqrt(max(0.0, 1.0 - rho * rho)) * w
    v /= np.linalg.norm(v)
    key = u * (0.25 + 3.5 * rng.random())  # arbitrary scales
    query = v * (40.0 + 60.0 * rng.random())
    return key.astype(np.float32), query.astype(np.float32)


def measured_s(key: np.ndarray, query: np.ndarray, norms, d: int) -> float:
    """The s the REFERENCE actually produced, recomputed for the record.

    Burn and torch disagree in the last ulp of an RMS norm, so the fixture
    prints the ACHIEVED |s| and the test asserts the band was hit rather than
    trusting the target.
    """
    kt = torch.from_numpy(key)[None, None, :]
    qt = torch.from_numpy(query)[None, None, :]
    with torch.no_grad():
        k = rms_norm_like(norms[0], kt)
        q = rms_norm_like(norms[1], qt)
        return float((k * q).sum(-1).item() / math.sqrt(d))


def build_gate(mod, rms_eps_shipped: float) -> dict:
    """One row per target |s|, reference columns per row.

    The columns differ only in (RMS eps) x (clamp_min vs add) x (f32 vs f64).
    dormouse-engram is eps=1e-5 + add (lib.rs:219/225/234), so `gate_eps1e5_add` is
    the column it is WRONG to match and `gate_eps1e5` is the one it is right to
    match.

    The `*_f64` columns are the SAME formula evaluated in float64, and their gap
    from the f32 column is the reference's OWN f32 error at that |s| - the
    irreducible floor of any comparison against it. It is not small: at
    |s| ~ 1e-8 the f32 dot product of 64 unit terms cancels to noise, `sign()`
    reads the noise, and the gate lands on either side of 0.5. A test that
    demanded agreement there would be demanding agreement on rounding.
    """
    d = GATE_D
    norms = {
        "shipped": torch.nn.RMSNorm(d, eps=rms_eps_shipped),
        "eps1e5": torch.nn.RMSNorm(d, eps=BURN_RMS_EPS),
        "eps1e5_f64": torch.nn.RMSNorm(d, eps=BURN_RMS_EPS).to(torch.float64),
    }
    keys, queries, rows = [], [], []
    for i, target_s in enumerate(GATE_S):
        key, query = solve_pair(d, target_s, seed=1000 + i)
        keys.append(key)
        queries.append(query)
        kt = torch.from_numpy(key)[None, None, :]
        qt = torch.from_numpy(query)[None, None, :]
        with torch.no_grad():
            out = {}
            for tag, n in norms.items():
                # The f64 column must actually run in f64: an f32 input to an
                # f64-weight RMSNorm is dispatched as f32 (torch warns), which
                # would make the "f32 error" column identically zero and the
                # whole noise-floor claim vacuous.
                cast = (lambda t: t.to(torch.float64)) if tag.endswith("_f64") else (lambda t: t)
                nk, nq = rms_norm_like(n, cast(kt)), rms_norm_like(n, cast(qt))
                for add in (False, True):
                    out[f"gate{'_add' if add else ''}_{tag}"] = float(
                        gate_from_normed(nk, nq, d, add).reshape(-1)[0]
                    )
                    out[f"gate{'_add' if add else ''}_{tag}_rev"] = float(
                        gate_from_normed(nk, nq, d, add, reverse_dot=True).reshape(-1)[0]
                    )
            s = float((rms_norm_like(norms["eps1e5"], kt)
                       * rms_norm_like(norms["eps1e5"], qt)).sum(-1).item() / math.sqrt(d))
        # How far the reference's OWN f32 answer moves under a perturbation
        # that changes no mathematics: f32-vs-f64, and the dot product summed
        # back to front. The larger of the two is the floor of any comparison
        # against this implementation. At |s| = 1.5e-8 it is 5.0e-4, because
        # `sign()` reads f32 cancellation noise there.
        rows.append({
            "target_s": float(target_s),
            "s": s,
            "gate": out["gate_shipped"],
            "gate_add": out["gate_add_shipped"],
            "gate_eps1e5": out["gate_eps1e5"],
            "gate_eps1e5_add": out["gate_add_eps1e5"],
            "f32_error": max(
                abs(out["gate_eps1e5"] - out["gate_eps1e5_f64"]),
                abs(out["gate_eps1e5"] - out["gate_eps1e5_rev"]),
                abs(out["gate_add_eps1e5"] - out["gate_add_eps1e5_rev"]),
            ),
            "f32_error_clamp": abs(out["gate_eps1e5"] - out["gate_eps1e5_rev"]),
        })
    k = np.stack(keys)[None]
    q = np.stack(queries)[None]
    # Batched, so the fixture also exercises the [B, L, D] path the module uses.
    with torch.no_grad():
        kt, qt = torch.from_numpy(k), torch.from_numpy(q)
        cols = {}
        for tag, n in norms.items():
            cast = (lambda t: t.to(torch.float64)) if tag.endswith("_f64") else (lambda t: t)
            nk, nq = rms_norm_like(n, cast(kt)), rms_norm_like(n, cast(qt))
            for add in (False, True):
                base = f"gate{'_add' if add else ''}_{tag}"
                cols[base] = f32_list(gate_from_normed(nk, nq, d, add))
                if not tag.endswith("_f64"):
                    cols[f"{base}_rev"] = f32_list(
                        gate_from_normed(nk, nq, d, add, reverse_dot=True)
                    )
    return {
        "d": d,
        "rows": rows,
        "key": f32_list(torch.from_numpy(k)),
        "query": f32_list(torch.from_numpy(q)),
        **cols,
    }


def flatten_ladder(mapping, layer_id: int, max_ngram: int, n_head: int) -> list[int]:
    """`vocab_size_across_layers[layer]` flattened to slot order.

    Slot order is ngram-major then head-minor, which is the order our
    `NgramHasher` numbers its slots in.
    """
    table = mapping.vocab_size_across_layers[layer_id]
    return [int(table[j][i]) for j in range(max_ngram - 1) for i in range(n_head)]


def build_ladders(mod) -> list[dict]:
    out = []
    for vocab_size, max_ngram, n_head, layer_ids in LADDERS:
        mod.engram_cfg = mod.EngramConfig(
            engram_vocab_size=[vocab_size] * (max_ngram - 1),
            max_ngram_size=max_ngram,
            n_head_per_ngram=n_head,
            layer_ids=layer_ids,
        )
        mapping = mod.NgramHashMapping(
            engram_vocab_size=mod.engram_cfg.engram_vocab_size,
            max_ngram_size=max_ngram,
            n_embed_per_ngram=1,
            n_head_per_ngram=n_head,
            layer_ids=layer_ids,
            tokenizer_name_or_path="",
            pad_id=0,
            seed=0,
        )
        out.append({
            "vocab_size": vocab_size,
            "max_ngram": max_ngram,
            "n_head": n_head,
            "layer_ids": layer_ids,
            # single layer: the reference's `seen_primes` is shared across
            # layers, ours is a fresh list per hasher, so only layer 0 of a
            # multi-layer config is comparable (recorded, not silently dropped)
            "primes": flatten_ladder(mapping, layer_ids[0], max_ngram, n_head),
            "layer1_primes_multi": flatten_ladder(mapping, layer_ids[-1], max_ngram, n_head),
            "multipliers": [int(m) for m in mapping.layer_multipliers[layer_ids[0]]],
            "half_bound": int(
                np.iinfo(np.int64).max // mapping.tokenizer_vocab_size // 2
            ),
        })
    return out


def _set_norms(engram, gains: float, eps: float | None) -> None:
    """Apply the two settings that make dormouse-engram numerically comparable."""
    with torch.no_grad():
        for group in (list(engram.norm1), list(engram.norm2),
                      list(engram.short_conv.norms)):
            for n in group:
                n.weight.fill_(gains)
                if eps is not None:
                    n.eps = eps  # VAR: eps


def build_module(mod, rms_eps_shipped: float) -> dict:
    """Run the reference's full `Engram.forward` on fixed weights and inputs."""
    hidden_size, hc_mult = MOD_D, MOD_HC
    max_ngram = 3
    n_head = MOD_TABLES // (max_ngram - 1)
    mod.backbone_config = mod.BackBoneConfig(
        hidden_size=hidden_size, hc_mult=hc_mult, vocab_size=256, num_layers=2
    )
    mod.engram_cfg = mod.EngramConfig(
        engram_vocab_size=[256] * (max_ngram - 1),
        max_ngram_size=max_ngram,
        n_embed_per_ngram=MOD_EMBED * n_head,
        n_head_per_ngram=n_head,
        layer_ids=[1],
        pad_id=2,
        seed=0,
        kernel_size=CONV_K,
    )
    torch.manual_seed(20260929)
    engram = mod.Engram(layer_id=1).to(torch.float32)

    # Structural record: the reference's learnable-parameter inventory, for the
    # dimensions the reference SHIPS (hc_mult=4, hidden=1024, l.355-356 +
    # ShortConv's norms l.148-151, and l.351-354 for the Linear biases).
    ship_params, ship_norms, ship_bias = 0, 0, 0
    for group in (list(engram.norm1), list(engram.norm2),
                  list(engram.short_conv.norms)):
        for n in group:
            if n.weight is not None:
                ship_params += n.weight.numel()
                ship_norms += 1
    for lin in [engram.value_proj] + list(engram.key_projs):
        if lin.bias is not None:
            ship_bias += lin.bias.numel()
    SHIP_HC, SHIP_D = 4, 1024
    # norm1 (l.355) + norm2 (l.356) + ShortConv.norms (l.148-151) = 3 per head.
    norms_per_head = ship_norms // hc_mult
    shipped_norm_params = norms_per_head * SHIP_HC * SHIP_D
    # value_proj + key_projs[hc_mult] (l.351-354), one bias each.
    shipped_bias_params = (1 + SHIP_HC) * SHIP_D

    _set_norms(engram, 1.0, None)  # gains -> 1 (dormouse-engram has no gains)
    with torch.no_grad():
        for lin in [engram.value_proj] + list(engram.key_projs):
            lin.bias.zero_()  # bias -> 0 (dormouse-engram's Linears have no bias)

    # The reference's ShortConv has hc_mult * D independent channel kernels;
    # `depthwise_conv_1d` broadcasts one per channel across every group, so tile
    # the reference's kernels here to make the full-forward comparison exact.
    # The group difference is pinned structurally (the parameter count) by
    # `reference_has_learnable_norm_gains_and_burn_has_none`.
    #
    # The kernel is NOT flipped. `depthwise_conv_1d` (lib.rs:96-101) pairs
    # w[:, i] with x[t - i*dilation] - tap 0 is the CURRENT sample - while the
    # reference's tap 0 is the MOST delayed, so the two conventions are already
    # time-reverses of each other and the SAME array makes them agree. That
    # reversal is pinned by `depthwise_conv_is_the_time_reverse_...`.
    with torch.no_grad():
        w3 = engram.short_conv.conv.weight.reshape(hc_mult, hidden_size, CONV_K)
        engram.short_conv.conv.weight.copy_(
            w3[:1].repeat(hc_mult, 1, 1).reshape(hc_mult * hidden_size, 1, CONV_K)
        )

    # Params in burn's `EngramModule` traversal order (declaration order):
    #   memory.embedding.weight, key_projs[hc].weight, value_proj.weight,
    #   conv_weight.  No biases on either side. torch's Linear is [out, in];
    # burn's is [in, out], so the Linears are dumped TRANSPOSED here.
    params = []
    emb_w = engram.multi_head_embedding.embedding.weight
    params.append(("memory.weight", emb_w, list(emb_w.shape)))
    for i, lin in enumerate(engram.key_projs):
        t = lin.weight.t()  # [in, out] == burn's layout
        params.append((f"key_projs[{i}].weight", t, list(t.shape)))
    tv = engram.value_proj.weight.t()
    params.append(("value_proj.weight", tv, list(tv.shape)))
    # The faithful reference shape, [hc*D, k] (l.138-146 with groups=total), and
    # the [D, k] slice dormouse-engram's `with_short_conv` stores (lib.rs:160). Same
    # numbers, 1/hc_mult of the parameters - which is the structural claim, in
    # numeric form.
    cw = engram.short_conv.conv.weight.reshape(engram.short_conv.conv.in_channels, -1)
    params.append(("conv_weight", cw, list(cw.shape)))
    params.append(("conv_weight_group0", cw[:hidden_size], list(cw[:hidden_size].shape)))
    # The SAME kernel in dormouse-engram's tap order. `depthwise_conv_1d`
    # (lib.rs:96-101) pairs w[:, i] with x[t - i*dilation] - tap 0 is the CURRENT
    # sample - while the reference's tap 0 is the MOST delayed, so the two
    # conventions are time-reverses and exactly one of them must be flipped.
    # The reference's own forward above used the un-flipped kernel (it is the
    # reference's own module, unmodified), so the flip belongs HERE, on the
    # array the Rust side loads.
    params.append(("conv_weight_burn", cw[:hidden_size].flip(-1).contiguous(),
                   list(cw[:hidden_size].flip(-1).shape)))

    rng = np.random.default_rng(7)
    ids = rng.integers(0, 256, size=(MOD_B, MOD_L), dtype=np.int64)
    hidden = torch.from_numpy(
        rng.standard_normal((MOD_B, MOD_L, hc_mult, hidden_size)) * 0.7
    ).to(torch.float32)

    def run():
        with torch.no_grad():
            hashes = engram.hash_mapping.hash(ids)[1]
            embs = engram.multi_head_embedding(torch.from_numpy(hashes)).flatten(start_dim=-2)
            value = engram.value_proj(embs).unsqueeze(2)
            gates = []
            for i in range(hc_mult):
                key = engram.key_projs[i](embs)
                gates.append(
                    gate_from_normed(
                        rms_norm_like(engram.norm1[i], key),
                        rms_norm_like(engram.norm2[i], hidden[:, :, i, :]),
                        hidden_size, False,
                    ).unsqueeze(-1)
                )
            gates = torch.stack(gates, dim=2)
            gated = gates * value
            return embs, value, gates, gated, gated + engram.short_conv(gated)

    embs, value, gates, gated, out_shipped = run()
    _set_norms(engram, 1.0, BURN_RMS_EPS)  # VAR: eps
    _, _, gates_e, gated_e, out_burn = run()

    return {
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "kernel": CONV_K,
        "dilation": CONV_DIL,
        "embed_dim": MOD_EMBED,
        "num_tables": MOD_TABLES,
        "table_sizes": flatten_ladder(engram.hash_mapping, 1, max_ngram, n_head),
        "ids": [[int(v) for v in row] for row in ids],
        "hidden": f32_list(hidden),
        "embeds": f32_list(embs),
        "value": f32_list(value),
        "gates_shipped": f32_list(gates),
        "gates_eps1e5": f32_list(gates_e),
        "gated_shipped": f32_list(gated),
        "out_shipped": f32_list(out_shipped),
        "out_eps1e5": f32_list(out_burn),
        "conv_weight_shape": list(cw.shape),
        "params": [{"name": n, "shape": s, "data": f32_list(t)} for n, t, s in params],
        "ref_norm_count": int(ship_norms),
        "ref_norm_gain_params_fixture": int(ship_params),
        "ref_bias_params_fixture": int(ship_bias),
        "ship_hc_mult": SHIP_HC,
        "ship_hidden_size": SHIP_D,
        "ship_norm_gain_params": int(shipped_norm_params),
        "ship_bias_params": int(shipped_bias_params),
    }


def build_conv(mod, rms_eps_shipped: float) -> dict:
    """`ShortConv.conv` alone, i.e. our `depthwise_conv_1d` on the same input.

    The reference normalises first; we feed it an already-normed tensor so the
    comparison isolates the depthwise convolution (kernel, dilation, causality,
    left-zero-padding).

    TWO shapes, because the implementations differ structurally and the
    difference is invisible at hc_mult = 1:
    * `y_hc1` — hc_mult=1, where the reference's `hc_mult * D` channel kernels
      and dormouse-engram's `D` kernels are the same operator.
    * `shared` / `per_group` — hc_mult=3, SAME input, weights tiled
      across groups vs independent per group. `depthwise_conv_1d` (lib.rs:96-100)
      broadcasts one `w[d, i]` across every group, so it can only equal the
      first.
    """
    torch.manual_seed(31337)
    rng = np.random.default_rng(99)

    sc = mod.ShortConv(hidden_size=MOD_D, kernel_size=CONV_K, dilation=CONV_DIL,
                       norm_eps=rms_eps_shipped, hc_mult=1, activation=True).to(torch.float32)
    with torch.no_grad():
        for n in sc.norms:
            n.weight.fill_(1.0)
    x1 = torch.from_numpy(rng.standard_normal((MOD_B, MOD_L, 1, MOD_D))).to(torch.float32)
    w1 = sc.conv.weight.reshape(MOD_D, CONV_K)
    with torch.no_grad():
        # `[B, L, 1, D]` -> the conv's `[B, D, L]`, then back (reference
        # `ShortConv.forward` lines 170-177 with G == 1).
        y1 = sc.conv(x1.squeeze(2).transpose(1, 2))[..., :MOD_L].transpose(1, 2)

    sc3 = mod.ShortConv(hidden_size=MOD_D, kernel_size=CONV_K, dilation=CONV_DIL,
                        norm_eps=rms_eps_shipped, hc_mult=MOD_HC, activation=True).to(torch.float32)
    with torch.no_grad():
        for n in sc3.norms:
            n.weight.fill_(1.0)
    x3 = torch.from_numpy(rng.standard_normal((MOD_B, MOD_L, MOD_HC, MOD_D))).to(torch.float32)
    w_shared = sc3.conv.weight.reshape(MOD_HC, MOD_D, CONV_K)[:1].repeat(MOD_HC, 1, 1)
    w_per_group = sc3.conv.weight.reshape(MOD_HC, MOD_D, CONV_K).clone()
    w_per_group[1:] = torch.from_numpy(
        rng.standard_normal((MOD_HC - 1, MOD_D, CONV_K))
    ).to(torch.float32)
    out = {}
    x3t = x3.transpose(1, 2).reshape(MOD_B, MOD_HC * MOD_D, MOD_L)
    # `reversed=True` because dormouse-engram's conv is the time-reverse of the
    # reference's (see y_hc1_reversed); the hc1 test pins that, these let the
    # hc3 structural test be about the GROUPING alone.
    for name, w, rev in (("shared", w_shared, True), ("per_group", w_per_group, True),
                         ("shared_fwd", w_shared, False), ("per_group_fwd", w_per_group, False)):
        with torch.no_grad():
            flat = (w.flip(-1) if rev else w).reshape(MOD_HC * MOD_D, 1, CONV_K)
            y = torch.nn.functional.conv1d(
                x3t, flat,
                padding=CONV_DIL * (CONV_K - 1),
                dilation=CONV_DIL,
                groups=MOD_HC * MOD_D,
            )[..., :MOD_L].transpose(1, 2).reshape(MOD_B, MOD_L, MOD_HC, MOD_D)
        out[name] = f32_list(y)
    return {
        "x_hc1": f32_list(x1),
        "w_hc1": f32_list(w1),
        "y_hc1": f32_list(y1),
        # The tap REVERSAL. `depthwise_conv_1d` (lib.rs:96-101) pairs w[:, i]
        # with x[t - i*dilation], so tap 0 is the CURRENT sample; the reference
        # pairs w[:, i] with x[t - (k-1-i)*dilation], so tap 0 is the MOST
        # delayed. Both are causal; they are time-reverses of one another. The
        # weight is learnable, so this is a reparameterisation rather than a
        # correctness bug - but any weight transferred from the reference lands
        # reversed, and the recorded init means something different.
        "y_hc1_reversed": f32_list(
            torch.nn.functional.conv1d(
                x1.squeeze(2).transpose(1, 2), w1.flip(-1).reshape(MOD_D, 1, CONV_K),
                padding=CONV_DIL * (CONV_K - 1), dilation=CONV_DIL, groups=MOD_D,
            )[..., :MOD_L].transpose(1, 2)
        ),
        "x_hc3": f32_list(x3),
        # [D, k]: the group-tiled kernel, which is the shape dormouse-engram stores.
        "w_hc3": f32_list(w_shared[0]),
        "reference_weight_elems_hc3": int(MOD_HC * MOD_D * CONV_K),
        "burn_weight_elems_hc3": int(MOD_D * CONV_K),
        "bias": None,  # reference ShortConv.conv is bias=False (line 143)
        **out,
    }


FORMAT = """flat fixture: every data line is `key [chunk]: v v v ...`. No strings, no
nesting, so the Rust side parses it in fifteen lines and needs no serde. Long
tensors wrap at CHUNK values per line; a key's values are its chunks
concatenated in file order. `#` lines are prose.
"""


def header(doc: dict) -> list[str]:
    return [
        "# engram golden-vector fixture - GENERATED, do not hand-edit.",
        f"# source: {REFERENCE_URL}",
        "# regenerate: python3 gen_engram_oracle.py engram_demo_v1.py ../fixtures/engram_oracle.txt",
        f"# torch {torch.__version__}, numpy {np.__version__}",
        f"# the reference's own nn.RMSNorm resolves eps = {doc['rms_eps_shipped']:.6e}"
        f" (measured from its output; torch.finfo(f32).eps ="
        f" {float(torch.finfo(torch.float32).eps):.6e});"
        f" dormouse-engram hardcodes {doc['rms_eps_burn']:.6e}",
        FORMAT.strip(),
    ]


def emit(doc: dict, path: str, head: list[str], chunk: int = 64) -> None:
    lines: list[str] = [f"# {line}" for line in "\n".join(head).splitlines()]

    # F32 needs NINE significant decimal digits to round-trip. The default
    # `format(v, "g")` is SIX, which silently truncated every golden constant
    # to 0.49975 / 0.50025 and made the tests fail on correct code. Every
    # tensor column is f32, so `.9g` is the floor; f64 scalars pass `.17g`.
    def put(key: str, values, fmt: str = ".9g") -> None:
        vals = list(values)
        if not vals:
            lines.append(f"{key}:")
            return
        for i in range(0, len(vals), chunk):
            body = " ".join(
                format(v, fmt) if isinstance(v, float) else str(v)
                for v in vals[i:i + chunk]
            )
            lines.append(f"{key}{'' if i == 0 else ' ' + str(i // chunk)}: {body}")

    g, m, c = doc["gate"], doc["module"], doc["conv"]
    put("meta.rms_eps_shipped", [doc["rms_eps_shipped"]], ".17g")
    put("meta.rms_eps_burn", [doc["rms_eps_burn"]], ".17g")
    put("meta.gate_d", [g["d"]])
    put("meta.gate_band", list(doc["gate_band"]), ".17g")
    put("meta.module_shape", [m[k] for k in
                              ("hidden_size", "hc_mult", "embed_dim", "num_tables")])
    put("meta.conv_kernel", [m["kernel"], m["dilation"]])
    put("meta.ref_norm_count", [m["ref_norm_count"]])
    put("meta.ref_norm_gain_params_shipped", [m["ship_norm_gain_params"]])
    put("meta.ref_bias_params_shipped", [m["ship_bias_params"]])
    put("meta.ref_norm_gain_params_fixture", [m["ref_norm_gain_params_fixture"]])
    put("meta.ref_bias_params_fixture", [m["ref_bias_params_fixture"]])
    put("meta.conv_weight_elems_reference_hc3", [c["reference_weight_elems_hc3"]])
    put("meta.conv_weight_elems_burn_hc3", [c["burn_weight_elems_hc3"]])

    for col in ("target_s", "s", "f32_error", "f32_error_clamp", "gate", "gate_add",
                "gate_eps1e5", "gate_eps1e5_add"):
        put(f"gate.row.{col}", [r[col] for r in g["rows"]], ".17g")
    put("gate.key", g["key"])
    put("gate.query", g["query"])
    for col in ("gate_shipped", "gate_add_shipped", "gate_eps1e5",
                "gate_add_eps1e5", "gate_eps1e5_f64", "gate_eps1e5_rev",
                "gate_add_eps1e5_rev"):
        put(f"gate.out.{col}", g[col])

    for i, L in enumerate(doc["ladders"]):
        put(f"ladder.{i}.vocab_size", [L["vocab_size"]])
        put(f"ladder.{i}.max_ngram", [L["max_ngram"]])
        put(f"ladder.{i}.n_head", [L["n_head"]])
        put(f"ladder.{i}.primes", L["primes"])
        put(f"ladder.{i}.layer_last_primes", L["layer1_primes_multi"])
        put(f"ladder.{i}.multipliers", L["multipliers"])
        put(f"ladder.{i}.half_bound", [L["half_bound"]])

    put("module.table_sizes", m["table_sizes"])
    put("module.ids", [v for row in m["ids"] for v in row])
    for key in ("hidden", "embeds", "value", "gated_shipped", "out_shipped",
                "out_eps1e5"):
        put(f"module.{key}", m[key])
    put("module.gates.shipped", m["gates_shipped"])
    put("module.gates.eps1e5", m["gates_eps1e5"])
    for p in m["params"]:
        put(f"module.shape.{p['name']}", p["shape"])
        put(f"module.param.{p['name']}", p["data"], ".17g")

    put("conv.x_hc1", c["x_hc1"])
    put("conv.w_hc1", c["w_hc1"])
    put("conv.y_hc1", c["y_hc1"])
    put("conv.y_hc1_reversed", c["y_hc1_reversed"])
    put("conv.x_hc3", c["x_hc3"])
    put("conv.w_hc3", c["w_hc3"])
    put("conv.y_hc3.shared", c["shared"])
    put("conv.y_hc3.per_group", c["per_group"])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "engram_demo_v1.py"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "engram_oracle.txt"
    )
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    mod = load_reference(ref)

    # The eps the REFERENCE's own nn.RMSNorm resolves, MEASURED rather than
    # assumed: `eps=None` is resolved per call and torch 2.14 keeps the
    # attribute None. Feeding x = a*ones gives out = a*rsqrt(a^2 + eps), so
    # eps = a^2 * (1/out^2 - 1). a must be small enough that eps survives f32
    # rounding of the result: at a = 1 the recovered eps is exactly 0.
    probe = torch.nn.RMSNorm(64).to(torch.float32)
    a = 1e-3
    with torch.no_grad():
        out = float(probe(torch.full((64,), a))[0].item())
    rms_eps_shipped = a * a * (1.0 / (out * out) - 1.0)

    doc = {
        "rms_eps_shipped": rms_eps_shipped,
        "rms_eps_burn": BURN_RMS_EPS,
        "gate_band": list(GATE_BAND),
        "ladders": build_ladders(mod),
        "gate": build_gate(mod, rms_eps_shipped),
        "module": build_module(mod, rms_eps_shipped),
        "conv": build_conv(mod, rms_eps_shipped),
    }
    emit(doc, out_path, header(doc))
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
