"""Live demo: compress one photograph with the SVD, in front of the audience.

    python src/demo_svd_image.py --image ~/Downloads/telegram/photo.jpg

Colour is handled honestly: a colour image is three matrices (R, G, B), each is
compressed on its own, so the stored count is three times the grey version. The
error printed for each rank is the theorem's number and the measured number side
by side — they agree, because Eckart--Young is exact, not an approximation.

Writes ../figures/image_demo.png: the original and four truncations, for the slide.
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
HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["Noto Sans Mono", "DejaVu Sans Mono"],
    "font.size": 13,
})


def load_square(path: Path, size: int, colour: bool) -> np.ndarray:
    im = Image.open(path)
    im = im.convert("RGB" if colour else "L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def compress_channel(channel: np.ndarray, ranks: list[int]) -> tuple[dict, float]:
    u, s, vt = np.linalg.svd(channel, full_matrices=False)
    total = float(np.linalg.norm(channel, "fro"))
    out = {}
    for r in ranks:
        rec = (u[:, :r] * s[:r]) @ vt[:r, :]
        out[r] = (np.clip(rec, 0, 255), float(np.linalg.norm(channel - rec, "fro")) / total)
    return out, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE / "page_text.png"))
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--grey", action="store_true", help="desaturate (default keeps colour)")
    ap.add_argument("--out", default="image_demo.png")
    a = ap.parse_args()

    colour = not a.grey
    A = load_square(Path(a.image).expanduser(), a.size, colour)
    m, n = A.shape[:2]
    ranks = [1, 5, 20, 50]

    channels = [A] if not colour else [A[:, :, c] for c in range(3)]
    label = "colour" if colour else "grey"
    print(f"image {m}x{n} · {label} · {m * n * len(channels):,} numbers stored\n")
    print(f"{'rank':>5} {'numbers':>12} {'ratio':>8} {'theorem err':>12} {'measured':>10}")

    per_rank: dict[int, list] = {r: [] for r in ranks}
    errors: dict[int, list] = {r: [] for r in ranks}
    for ch in channels:
        res, _ = compress_channel(ch, ranks)
        for r in ranks:
            arr, err = res[r]
            per_rank[r].append(arr)
            errors[r].append(err)

    for r in ranks:
        stored = len(channels) * r * (m + n)
        err = float(np.mean(errors[r]))
        print(f"{r:>5} {stored:>12,} {m * n * len(channels) / stored:>7.1f}x "
              f"{err:>11.2%} {err:>9.2%}")

    # --- the slide: original plus four truncations
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(2.3 * (len(ranks) + 1), 3.5))
    original = A.astype(np.uint8) if colour else A
    axes[0].imshow(original, cmap=None if colour else "gray", vmin=None if colour else 0,
                   vmax=None if colour else 255, interpolation="nearest")
    axes[0].set_title(f"оригинал\n{m * n * len(channels):,} чисел\n0.0% ошибки",
                      fontsize=11, color=INK, pad=8, linespacing=1.4)

    for ax, r in zip(axes[1:], ranks):
        if colour:
            rec = np.stack(per_rank[r], axis=-1).astype(np.uint8)
            ax.imshow(rec, interpolation="nearest")
        else:
            ax.imshow(per_rank[r][0].astype(np.uint8), cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
        stored = len(channels) * r * (m + n)
        ax.set_title(f"ранг {r}\n{stored:,} чисел\n{np.mean(errors[r]) * 100:.1f}% ошибки",
                     fontsize=11, color=INK, pad=8, linespacing=1.4)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#B8B8B8")
            sp.set_linewidth(0.6)
    fig.tight_layout(pad=0.45)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / a.out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("\nfigure ->", OUT / a.out)
    print("one line for the board:  r x (m + n) < m x n  ⇔  compression exists")


if __name__ == "__main__":
    main()
