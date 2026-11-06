"""The lecture's experiments, run against OUR OWN trained weights.

Everything here is measured on a real checkpoint
(`checkpoints/flagship/flagship.pt`, 102,678,943 params, byte-level, 4.2 GB of
text) and nothing is synthetic except the control that validates the theorem.

Outputs (into ../figures/):
  spectrum.pdf        singular values of four trained matrices, log scale
  mass.pdf            Frobenius mass captured by the best rank-r approximation
  eckart_young.pdf    theorem vs measurement (random + trained matrix)
  rank_ladder.pdf     measured BPB cost of truncating the attention rank

and ../numbers.json with every number the slides quote, so a reader can check
the deck against the code instead of trusting it.

Run:  CUDA_VISIBLE_DEVICES= .venv/bin/python src/lecture_svd.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT = Path("/home/sehaxe/dormouse/checkpoints/flagship/flagship.pt")
OUT = Path(__file__).resolve().parent.parent / "figures"
NUM = Path(__file__).resolve().parent.parent / "numbers.json"

# One visual language for the whole lecture: ink on paper, one accent.
INK = "#201D1D"
ACCENT = "#007AFF"          # the single accent: OpenCode blue
MUTED = "#646262"
GRID = "#E7E3E3"

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Noto Sans Mono", "DejaVu Sans Mono"],
    "mathtext.fontset": "dejavusans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_weights() -> dict[str, torch.Tensor]:
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    return blob["model"] if "model" in blob else blob


def spectrum(w: torch.Tensor) -> np.ndarray:
    return np.linalg.svd(w.double().numpy(), compute_uv=False)


def mass_curve(s: np.ndarray) -> np.ndarray:
    """Fraction of squared Frobenius norm kept by the best rank-r truncation."""
    e = np.cumsum(s**2)
    return e / e[-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W = load_weights()

    # The four matrices the lecture walks through. Chosen because they are the
    # four different SHAPES our model actually contains.
    picks = {
        "moe.down $1024{\\times}2730$": "loop.moe.down.weight",
        "q_proj $1024{\\times}256$": "loop.attn.q_proj.weight",
        "mem_dense $1024{\\times}1024$": "loop.mem_dense.weight",
        "expert $v$ $64{\\times}2730$": "loop.moe.experts.0.v",
    }
    spectra, numbers = {}, {}
    for label, key in picks.items():
        m = W[key].float()
        s = spectrum(m)
        spectra[label] = s
        k = min(m.shape)
        numbers[key] = {
            "shape": list(m.shape),
            "rank": int(torch.linalg.matrix_rank(m.double())),
            "sigma_max": float(s[0]),
            "sigma_min": float(s[-1]),
            "cond": float(s[0] / s[-1]) if s[-1] > 0 else None,
            "rank_for_90pct_mass": int(np.searchsorted(mass_curve(s), 0.90) + 1),
            "rank_for_99pct_mass": int(np.searchsorted(mass_curve(s), 0.99) + 1),
            "rank_for_999pct_mass": int(np.searchsorted(mass_curve(s), 0.999) + 1),
            "mass_at_rank": {
                str(r): float(mass_curve(s)[min(r, k) - 1]) for r in (8, 16, 32, 64, 128, 256)
            },
            "frobenius": float(np.linalg.norm(m.double().numpy())),
            "nuclear": float(s.sum()),
            "minshape": int(k),
        }

    # ---- Figure 1: the spectra themselves, on a log axis -------------------
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for (label, s), style in zip(spectra.items(), (("-", ACCENT), ("--", INK), ("-.", MUTED), (":", "#4a6b8a"))):
        ax.semilogy(np.arange(1, len(s) + 1), np.maximum(s, 1e-12), style[0], color=style[1], lw=1.5, label=label)
    ax.set_xlabel("index $i$")
    ax.set_ylabel("$\\sigma_i$")
    ax.grid(True, which="major", axis="y", alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "spectrum.png", dpi=300, transparent=True)
    plt.close(fig)

    # ---- Figure 2: how much mass a rank-r truncation keeps -----------------
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for (label, s), style in zip(spectra.items(), (("-", ACCENT), ("--", INK), ("-.", MUTED), (":", "#4a6b8a"))):
        c = mass_curve(s)
        ax.plot(np.arange(1, len(c) + 1), c, style[0], color=style[1], lw=1.5, label=label)
    ax.axhline(0.99, color=GRID, lw=1)
    ax.axvline(64, color=GRID, lw=1)
    ax.text(72, 0.42, "rank 64", color=MUTED, fontsize=13)
    ax.set_xlabel("rank $r$")
    ax.set_ylabel("$\\|W_r\\|_F^2 / \\|W\\|_F^2$")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "mass.png", dpi=300, transparent=True)
    plt.close(fig)

    # the dark card's twin: cream ink on #201D1D, one accent kept
    DARK, DARKMUTE, DARKGRID = "#201D1D", "#9A9898", "#302C2C"
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    for (label, s_), style in zip(spectra.items(), (("-", ACCENT), ("--", "#FDFCFC"), ("-.", DARKMUTE), (":", "#30D158"))):
        c = mass_curve(s_)
        ax.plot(np.arange(1, len(c) + 1), c, style[0], color=style[1], lw=1.5, label=label)
    ax.axhline(0.99, color=DARKGRID, lw=1)
    ax.axvline(64, color=DARKGRID, lw=1)
    ax.text(72, 0.40, "rank 64", color=DARKMUTE, fontsize=13)
    ax.set_xscale("log")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARKMUTE)
    ax.tick_params(colors=DARKMUTE)
    ax.set_xlabel("rank $r$", color=DARKMUTE)
    ax.set_ylabel("mass kept", color=DARKMUTE)
    ax.grid(True, alpha=0.25, color=DARKGRID)
    lg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in lg.get_texts():
        t.set_color("#FDFCFC")
    fig.savefig(OUT / "mass_dark.pdf", facecolor=DARK)
    plt.close(fig)

    # ---- Figure 3: Eckart-Young, theorem vs measurement -------------------
    rng = np.random.default_rng(0)
    # A deliberately ill-conditioned 256x512 matrix: the theorem must hold for
    # a matrix whose spectrum decays like 10^0..10^-3, not only for nice ones.
    A = rng.standard_normal((256, 256)) @ np.diag(np.logspace(0, -3, 256)) @ rng.standard_normal((256, 512))
    sA = np.linalg.svd(A, compute_uv=False)
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    theory, measured = [], []
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    for r in ranks:
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        measured.append(float(np.linalg.norm(A - approx, "fro")))
        theory.append(float(np.sqrt(np.sum(sA[r:] ** 2))))
    rel_err = float(max(abs(m - t) for m, t in zip(measured, theory)) / theory[0])
    numbers["eckart_young"] = {"ranks": ranks, "theory": theory, "measured": measured,
                               "max_rel_disagreement": rel_err}

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.loglog(ranks, theory, "-", color=ACCENT, lw=2, label="theory $\\sqrt{\\sum_{i>r}\\sigma_i^2}$")
    ax.loglog(ranks, measured, "o", color=INK, ms=5, mfc="none", label="measured $\\|A-A_r\\|_F$")
    ax.set_xlabel("rank $r$")
    ax.set_ylabel("error")
    ax.grid(True, alpha=0.6, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "eckart_young.png", dpi=300, transparent=True)
    plt.close(fig)

    # ---- Figure 4: the price we measured for truncating OUR model ---------
    # From the repo's own A/B record (docs/reviews, 3 seeds each, equal steps).
    ladder = {"rank": [16, 32, 64], "d_bpb": [0.136, 0.027, 0.0]}
    numbers["rank_ladder"] = ladder
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    xs = list(range(len(ladder["rank"])))
    ax.plot(xs, ladder["d_bpb"], "o-", color=ACCENT, lw=1.8, ms=6)
    for x, d in zip(xs, ladder["d_bpb"]):
        ax.annotate(f"{d:+.3f}", (x, d), textcoords="offset points", xytext=(0, 13),
                    fontsize=15, color=INK, ha="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r) for r in ladder["rank"]])
    ax.set_xlim(-0.45, len(xs) - 0.55)
    ax.set_ylim(-0.012, 0.155)
    ax.set_xlabel("attention rank $r$")
    ax.set_ylabel("$\\Delta$ bits per byte")
    ax.grid(True, axis="y", alpha=0.55)
    fig.savefig(OUT / "rank_ladder.png", dpi=300, transparent=True)
    plt.close(fig)

    # ---- the stored low-rank factors: rank is a fact, not a claim ---------
    uv = {}
    for name, (u, v) in {"out_proj": ("loop.out_proj.u", "loop.out_proj.v"),
                         "lm_head": ("lm_head.u", "lm_head.v")}.items():
        P = W[u].float() @ W[v].float()
        sv = np.linalg.svd(P.double().numpy(), compute_uv=False)
        nz = int((sv > sv[0] * 1e-12).sum())
        uv[name] = {"stored_as": [list(W[u].shape), list(W[v].shape)],
                    "product_shape": list(P.shape),
                    "numerical_rank": nz,
                    "nonzero_singular_values": nz,
                    "sigma_tail_ratio": float(sv[63] / sv[0]) if len(sv) > 63 else None}
    numbers["stored_factors"] = uv

    NUM.write_text(json.dumps(numbers, indent=2))
    print(json.dumps({k: v for k, v in numbers.items() if k != "stored_factors"}, indent=2)[:2000])
    print("\nfigures ->", sorted(p.name for p in OUT.glob("*.pdf")))
    print("numbers ->", NUM)


if __name__ == "__main__":
    main()
