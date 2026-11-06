"""The demo that always works in a classroom: a page of text, compressed by rank.

Why text and not a photo: a screenshot of prose is something every student in the
room has seen compressed badly, it needs no image licensing, and the failure mode
(letters melting into grey bars) is the point — at rank 1 you cannot read a single
word, at rank 20 you can read almost everything, at rank 50 you cannot tell.

The source text is this repo's own README paragraph, so nothing is borrowed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]      # dormouse/
OUT = Path(__file__).resolve().parent.parent / "figures"

INK = "#111111"
MUTE = "#3A3A3A"
ACCENT = "#0055CC"
PAPER = "#FFFFFF"

FALLBACK = (
    "Линейная алгебра описывает, как пространство меняется под действием отображения. "
    "Матрица — это не таблица чисел, а форма: она поворачивает, растягивает и снова "
    "поворачивает. Сингулярные числа говорят, во сколько раз растянута каждая ось, "
    "и если какая-то ось растянута в ноль, значит это направление матрица не видит. "
)


def clean(md: str) -> str:
    """Markdown out, prose in: a textbook page has no asterisks or pipes."""
    keep = []
    for raw in md.splitlines():
        ln = raw.strip()
        if not ln or ln[0] in "#|>+*" or ln[:2] in ("1.", "2.", "3.", "4.", "5."):
            continue
        if ln.startswith("```") or "][" in ln or "http" in ln:
            continue
        ln = ln.replace("**", "").replace("*", "").replace("`", "")
        if any("а" <= c.lower() <= "я" for c in ln) and len(ln) > 40:
            keep.append(ln)
    return " ".join(keep)


def source_text(chars: int = 2600) -> str:
    """A paragraph of Russian prose from this repo, or a hand-written fallback."""
    here = Path(__file__).resolve().parent.parent
    page = Path(__file__).resolve().parent / "page_text.txt"
    if page.exists():
        # A textbook-style page written for this lecture: prose, no markdown, no meta.
        return " ".join(page.read_text(encoding="utf-8").split())[:chars]
    candidates = [here / "notes-ru.md", here / "README.md", ROOT / "README.md"]
    text = " ".join(
        clean(p.read_text(encoding="utf-8", errors="ignore")) for p in candidates if p.exists()
    )
    text = " ".join(text.split())
    # A real page of prose, never the same paragraph repeated: if the repo somehow
    # has less Russian than this, the hand-written fallback is the honest filler.
    return text[:chars] if len(text) >= 400 else FALLBACK


def render_page(text: str, w: int = 620, h: int = 780) -> Image.Image:
    """A page of prose: the same shape as a textbook scan, no external asset."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    font = None
    for cand in (
        "/usr/share/fonts/TTF/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    ):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 17)
            break
    if font is None:                                    # last resort: PIL's bitmap font
        font = ImageFont.load_default()

    margin, leading = 44, 25
    y = margin
    line: list[str] = []

    def flush(line_text: str, yy: int) -> int:
        d.text((margin, yy), line_text, font=font, fill=INK)
        return yy + leading

    for word in text.split():
        trial = (" ".join(line + [word])).strip()
        if d.textlength(trial, font=font) > w - 2 * margin:
            y = flush(" ".join(line), y)
            line = [word]
            if y > h - margin - leading:
                break
        else:
            line.append(word)
    if line and y <= h - margin - leading:
        flush(" ".join(line), y)
    return img


def compress(img: Image.Image, ranks: list[int]) -> dict:
    a = np.asarray(img, dtype=np.float64)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    fro = float(np.sqrt((s**2).sum()))
    out = {}
    for r in ranks:
        rec = (u[:, :r] * s[:r]) @ vt[:r]
        err = float(np.sqrt(((a - rec) ** 2).sum()) / fro)
        out[r] = (np.clip(rec, 0, 255).astype(np.uint8), err,
                  r * (a.shape[0] + a.shape[1]), a.size)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", default=None, help="override the source page text")
    a = ap.parse_args()

    text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else source_text()
    page = render_page(text)
    ranks = [1, 5, 20, 50]
    res = compress(page, ranks)

    plt.rcParams.update({"font.family": "monospace",
                         "font.monospace": ["Noto Sans Mono", "DejaVu Sans Mono"],
                         "font.size": 13})
    n = 1 + len(ranks)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 3.55))
    panels = [(np.asarray(page, dtype=np.uint8), 0.0, page.size[0] * page.size[1], page.size[0] * page.size[1], "оригинал")]

    for r in ranks:
        rec, err, kept, total = res[r]
        panels.append((rec, err, kept, total, f"ранг {r}"))

    for ax, (arr, err, kept, total, title) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#B8B8B8")
            sp.set_linewidth(0.6)
        ax.set_title(f"{title}\n{kept:,} чисел\n{err * 100:.1f}% ошибки",
                     fontsize=11.5, color=INK, pad=8, linespacing=1.4)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "text_demo.png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print("ranks:", ", ".join(f"r={r}: {res[r][1]*100:.1f}% err, "
                              f"{res[r][2]:,}/{res[r][3]:,} чисел" for r in ranks))
    print("figure ->", OUT / "text_demo.png")


if __name__ == "__main__":
    main()
