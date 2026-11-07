"""The pixel sky — a dithered background, generated, not stock art.

Reference: a pastel pixel-art sky (blue -> pink cumulus -> cream haze) with an
ordered dither, the look modern editors use behind a frosted panel.

    port/.venv/bin/python site/sky.py

Writes site/media/sky.png (1920x1080) and site/media/sky.webp. The PNG is a
16-colour palette image: a 480x270 field upscaled 4x with nearest-neighbour, so
every pixel is a real pixel and the file stays small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "media"
W, H = 960, 540          # the pixel field; the page upscales it 2x
SCALE = 2

# The palette, sampled from the reference: sky blues, cumulus pinks, cream,
# and one violet for the cloud shadows.
PALETTE = np.array([
    # sky, deep -> horizon
    (0x3F, 0x6F, 0xC4), (0x4E, 0x80, 0xD2), (0x5D, 0x8E, 0xD8), (0x6E, 0xA2, 0xE6),
    (0x82, 0xB4, 0xEE), (0x9A, 0xC8, 0xF4), (0xB6, 0xD8, 0xF8), (0xD2, 0xE8, 0xFA),
    # cloud shadow, violet -> mauve
    (0x6C, 0x64, 0x9C), (0x82, 0x76, 0xAC), (0x9A, 0x8A, 0xBC), (0xB4, 0xA2, 0xC8),
    # cloud body, pink -> blush
    (0xD8, 0xA8, 0xC0), (0xE8, 0xB8, 0xC8), (0xF2, 0xC6, 0xD2), (0xF8, 0xD4, 0xDA),
    # highlights, cream -> white
    (0xFA, 0xE0, 0xDE), (0xFC, 0xEA, 0xE4), (0xFD, 0xF2, 0xEA), (0xFE, 0xF8, 0xF0),
    (0xFF, 0xFC, 0xF8), (0xFF, 0xFE, 0xFC),
], dtype=np.float32)


def value_noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise from a coarse random grid, bilinearly upsampled."""
    g = rng.random((cells + 1, cells * 2 + 1)).astype(np.float32)
    xs = np.linspace(0, g.shape[1] - 1, w, dtype=np.float32)
    ys = np.linspace(0, g.shape[0] - 1, h, dtype=np.float32)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, g.shape[1] - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, g.shape[0] - 1)
    fx = (xs - x0)[None, :]; fy = (ys - y0)[:, None]
    top = g[np.ix_(y0, x0)] * (1 - fx) + g[np.ix_(y0, x1)] * fx
    bot = g[np.ix_(y1, x0)] * (1 - fx) + g[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


def fbm(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float32)
    amp, cells, norm = 1.0, 3, 0.0
    for _ in range(octaves):
        out += amp * value_noise(h, w, cells, rng)
        norm += amp
        amp *= 0.5
        cells *= 2
    out /= norm
    return out


def bayer8() -> np.ndarray:
    m = np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                  [48, 16, 56, 24, 50, 18, 58, 26],
                  [12, 44, 4, 36, 14, 46, 6, 38],
                  [60, 28, 52, 20, 62, 30, 54, 22],
                  [3, 35, 11, 43, 1, 33, 9, 41],
                  [51, 19, 59, 27, 49, 17, 57, 25],
                  [15, 47, 7, 39, 13, 45, 5, 37],
                  [63, 31, 55, 23, 61, 29, 53, 21]], dtype=np.float32)
    return m / 64.0 - 0.5


def main() -> None:
    rng = np.random.default_rng(20261108)
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)

    # sky: a vertical ramp, plus a warm glow near the horizon
    t = yy ** 1.15
    sky = np.stack([
        0.42 + 0.50 * t,      # R
        0.62 + 0.33 * t,      # G
        0.86 + 0.12 * t,      # B
    ], axis=-1)
    glow = np.exp(-((yy - 0.62) ** 2) / 0.02) * 0.18
    sky[..., 0] += glow
    sky[..., 1] += glow * 0.9
    sky[..., 2] += glow * 0.5

    # cumulus: warp the domain, then threshold a second field — the warp is what
    # turns blobs into the cauliflower edge the reference has.
    base = fbm(H, W, rng, 5)
    warp = (fbm(H, W, rng, 4) - 0.5) * 0.30
    detail = fbm(H, W, rng, 6)
    dens = np.clip((base + warp + detail * 0.35 - 0.52) * 3.6, 0.0, 1.0)
    band = np.clip(1.0 - np.abs(yy - 0.46) * 1.9, 0.0, 1.0)
    dens *= band ** 0.7

    # a fake normal from the density gradient, lit from the upper left: that is
    # where the pinks on top and the violets underneath come from.
    gy, gx = np.gradient(dens)
    lit = np.clip(0.5 - gx * 26.0 - gy * 20.0, 0.0, 1.0)
    shadow = np.array([0.42, 0.39, 0.61]); mauve = np.array([0.66, 0.58, 0.76])
    pink = np.array([0.94, 0.76, 0.80]); cream = np.array([1.0, 0.94, 0.90])
    cloud = (shadow * (1 - lit)[..., None] + pink * lit[..., None])
    cloud = cloud * (1 - dens[..., None] * 0.0)          # keep the ramp
    cloud = cloud * (1 - 0.35 * (dens < 0.15)[..., None]) + mauve * 0.35 * (dens < 0.15)[..., None]
    top = np.clip((dens - 0.72) / 0.28, 0.0, 1.0)[..., None]
    cloud = cloud * (1 - top) + cream * top

    img = sky * (1 - dens[..., None]) + cloud * dens[..., None]

    # horizon: a cream haze fading everything toward the bottom
    haze = np.clip((yy - 0.58) / 0.42, 0.0, 1.0) ** 1.7
    img = img * (1 - haze[..., None]) + np.array([0.97, 0.96, 0.92]) * haze[..., None]

    # ordered dither to the palette: the pixel-art step
    field = np.tile(bayer8(), (H // 8 + 1, W // 8 + 1))[:H, :W][..., None]
    step = 1.0 / 12.0
    d = np.clip(img + field * step, 0.0, 1.0) * 255.0     # the palette is 0..255
    flat = d.reshape(-1, 3).astype(np.float32)
    idx = ((flat[:, None, :] - PALETTE[None, :, :]) ** 2).sum(-1).argmin(1)
    out = PALETTE[idx].reshape(H, W, 3).astype(np.uint8)

    big = Image.fromarray(out, "RGB").resize((W * SCALE, H * SCALE), Image.NEAREST)
    OUT.mkdir(parents=True, exist_ok=True)
    small = big.quantize(colors=16, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    small.save(OUT / "sky.png", optimize=True)
    big.save(OUT / "sky.webp", quality=82, method=6)
    print(f"sky.png {small.size} {OUT.joinpath('sky.png').stat().st_size // 1024} KB")
    print(f"sky.webp {OUT.joinpath('sky.webp').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
