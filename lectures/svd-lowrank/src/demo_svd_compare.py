"""Two pictures, same ranks, different price: what "энергия в хвосте" means.

The illustration is sharp line art with flat colour; the photo is smooth and busy.
Counter-intuitively the photo compresses BETTER, because its energy sits in a few
low-frequency components. That is the point of the slide.

    python src/demo_svd_compare.py --a art.jpg --b pool.jpg
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
INK = "#111111"

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Noto Sans Mono", "DejaVu Sans Mono"],
    "font.size": 12,
})

RANKS = [5, 20]


def square(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.float64)


def channels(A: np.ndarray) -> list[np.ndarray]:
    return [A[:, :, c] for c in range(3)]


def recon(ch: np.ndarray, r: int) -> tuple[np.ndarray, float]:
    u, s, vt = np.linalg.svd(ch, full_matrices=False)
    rec = (u[:, :r] * s[:r]) @ vt[:r, :]
    err = float(np.linalg.norm(ch - rec, "fro") / np.linalg.norm(ch, "fro"))
    return np.clip(rec, 0, 255), err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="sharp illustration")
    ap.add_argument("--b", required=True, help="smooth photograph")
    ap.add_argument("--size", type=int, default=340)
    ap.add_argument("--out", default="compare_demo.png")
    args = ap.parse_args()

    rows = []
    for path, label in ((Path(args.a).expanduser(), "рисунок"), (Path(args.b).expanduser(), "фото")):
        A = square(path, args.size)
        chans = channels(A)
        m, n = chans[0].shape
        cells = [(A.astype(np.uint8), 0.0, "оригинал")]
        for r in RANKS:
            recs, errs = zip(*(recon(c, r) for c in chans))
            cells.append((np.stack(recs, axis=-1).astype(np.uint8), float(np.mean(errs)), f"ранг {r}"))
        stored = {r: 3 * r * (m + n) for r in RANKS}
        rows.append((label, cells, m * n * 3, stored))

    fig, axes = plt.subplots(2, 1 + len(RANKS), figsize=(2.3 * (1 + len(RANKS)), 4.6))
    for i, (label, cells, total, stored) in enumerate(rows):
        for j, (arr, err, name) in enumerate(cells):
            ax = axes[i][j]
            ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#B8B8B8")
                sp.set_linewidth(0.6)
            if i == 0:
                ax.set_title(name, fontsize=12, color=INK, pad=7)
            if j == 0:
                ax.set_ylabel(label, fontsize=12, color=INK, labelpad=8)
            else:
                count = stored[list(stored)[j - 1]]
                ax.text(0.5, -0.045, f"{count:,} чисел · {err * 100:.1f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK)
    fig.tight_layout(pad=0.5, h_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / args.out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    for label, cells, total, stored in rows:
        print(label, "→", ", ".join(f"{n}: {e * 100:.1f}%" for _, e, n in cells[1:]),
              f"(всего чисел {total:,})")
    print("figure ->", OUT / args.out)


if __name__ == "__main__":
    main()
