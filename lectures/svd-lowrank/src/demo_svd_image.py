"""Live demo: compress one photograph with the SVD, in front of the audience.

Run it during the lecture (takes two seconds) and put the number on the board:

    python src/demo_svd_image.py --image ~/Pictures/Wallpapers/wallhaven-3q6kmd.jpg

It prints, for each rank r:  how many numbers you must store, the relative
Frobenius error by the theorem, and the error actually measured — which are the
same number, because Eckart--Young is not an approximation.

It also writes ../figures/image_demo.png: the original and four reconstructions,
for the slide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
INK = "#201D1D"
MUTED = "#646262"
ACCENT = "#007AFF"

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Noto Sans Mono", "DejaVu Sans Mono"],
    "font.size": 13,
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/home/sehaxe/Pictures/Wallpapers/wallhaven-3q6kmd.jpg")
    ap.add_argument("--size", type=int, default=480)
    a = ap.parse_args()

    im = Image.open(a.image).convert("L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((a.size, a.size), Image.LANCZOS)
    A = np.asarray(im, dtype=np.float64)
    m, n = A.shape

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    total = float(np.linalg.norm(A, "fro"))

    print(f"image {m}x{n} · stored as {m*n:,} numbers · Frobenius norm {total:,.1f}\n")
    print(f"{'rank':>5} {'numbers':>12} {'ratio':>8} {'theorem err':>12} {'measured':>10}")
    rows = []
    for r in (1, 5, 20, 50, 120, 480):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        err_theory = float(np.sqrt(np.sum(S[r:] ** 2))) / total
        err_measured = float(np.linalg.norm(A - approx, "fro")) / total
        stored = r * (m + n)
        print(f"{r:>5} {stored:>12,} {m*n/stored:>7.1f}x {err_theory:>11.2%} {err_measured:>9.2%}")
        rows.append((r, stored, m * n / stored, err_theory, np.clip(approx, 0, 255)))

    # the slide: original plus four truncations
    picks = [1, 5, 20, 50]
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 3.05))
    axes[0].imshow(A, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"original\n{m*n:,}\n0% error", fontsize=9.5, color=INK, linespacing=1.5)
    for ax, r in zip(axes[1:], picks):
        stored = r * (m + n)
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        err = float(np.linalg.norm(A - approx, "fro")) / total
        ax.imshow(np.clip(approx, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"rank {r}\n{stored:,}\n{err:.1%} error", fontsize=9.5, color=INK, linespacing=1.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#E7E3E3")
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "image_demo.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("\nfigure ->", OUT / "image_demo.png")
    print("\none line for the board:  r x (m + n) < m x n  ⇔  compression exists")


if __name__ == "__main__":
    main()
