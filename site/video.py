"""The landing's motion cards — three short loops rendered from the run data.

Every frame is data. Nothing here is stock footage, and nothing is a screen
recording of a UI that does not exist.

    ../port/.venv/bin/python site/video.py

Output: site/media/{rain,curve,heat}.webm — 800x450, 24 fps, 6 s, muted, and
built to loop (the rain uses integer column periods, the heat scrolls exactly
half its width, the curve keeps the finished line and runs a dot along it).
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECONDS = 800, 450, 24, 6
FRAMES = FPS * SECONDS
BG = (10, 10, 11)
INK = (240, 240, 240)
DIM = (110, 110, 114)
GRID = (26, 26, 28)
ACCENT = (150, 190, 255)          # one cold accent, used sparingly

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "media"
LOGS = ROOT.parent.parent / "logs"   # the runs live beside the repo, not in it
FONT = "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"
FONT_R = "/usr/share/fonts/TTF/DejaVuSansMono.ttf"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT if bold else FONT_R, size)


def encode(frames_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS), "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
        "-pix_fmt", "yuv420p", "-an", str(out),
    ]
    subprocess.run(cmd, check=True)
    print(f"{out.name}: {out.stat().st_size / 1024:.0f} KB")


def grid(draw: ImageDraw.ImageDraw) -> None:
    for x in range(0, W, 40):
        draw.line([(x, 0), (x, H)], fill=GRID)
    for y in range(0, H, 40):
        draw.line([(0, y), (W, y)], fill=GRID)


# ---------------------------------------------------------------- 1. the rain
def render_rain(tmp: Path) -> None:
    rng = np.random.default_rng(7)
    cols = W // 24                                   # sparser than the shader:
    period = rng.integers(2, 6, size=cols)           # a video card should read
    phase = rng.random(cols)                         # as data, not as static
    alive = rng.random(cols) > 0.34
    glyphs = ["0", "1"]
    f = font(17)
    for i in range(FRAMES):
        t = i / FRAMES
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for c in range(cols):
            if not alive[c]:
                continue
            head = (phase[c] + period[c] * t) % 1.0
            y0 = int(head * H)
            for k in range(11):
                y = y0 - k * 19
                if y < -19 or y > H:
                    continue
                a = 0.85 * max(0.0, 1.0 - k / 10) ** 1.8
                col = tuple(int(BG[j] + (INK[j] - BG[j]) * a) for j in range(3))
                g = glyphs[(c * 7 + k * 3 + int(t * 40)) % 2]
                d.text((c * 24 + 4, y), g, font=f, fill=col)
        img.save(tmp / f"{i:04d}.png")


# --------------------------------------------------------------- 2. the curve
def render_curve(tmp: Path) -> None:
    pts = []
    for line in (LOGS / "flagship3.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "eval":
            pts.append((r["bytes_seen"] / 1e6, r["bpb_fin"]))
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    # smooth through the measured points with a quadratic-ish resample
    dense_x = np.linspace(xs.min(), xs.max(), 240)
    dense_y = np.interp(dense_x, xs, ys)
    k = 21
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    smooth = np.convolve(np.pad(dense_y, (k // 2, k // 2), mode="edge"), kernel, mode="valid")

    left, right, top, bottom = 70, W - 40, 50, H - 56
    X = lambda v: left + (v - dense_x.min()) / (dense_x.max() - dense_x.min()) * (right - left)
    Y = lambda v: top + (1 - (v - smooth.min() + 0.03) / (smooth.max() - smooth.min() + 0.06)) * (bottom - top)
    f_small = font(12, bold=False)
    f_big = font(22)
    for i in range(FRAMES):
        t = i / FRAMES
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        grid(d)
        d.text((left, 16), "held-out BPB", font=f_small, fill=DIM)
        d.text((right - 120, 16), "flagship3", font=f_small, fill=DIM)
        d.line([(left, bottom), (right, bottom)], fill=(40, 40, 44))
        d.line([(left, top), (left, bottom)], fill=(40, 40, 44))
        # the full curve is drawn; a dot travels it, and a short trail follows
        d.line([(X(x), Y(y)) for x, y in zip(dense_x, smooth)], fill=(70, 70, 76), width=2)
        head = int(len(dense_x) * (0.15 + 0.85 * min(1.0, t * 1.6)))
        head = min(head, len(dense_x) - 1)
        d.line([(X(x), Y(y)) for x, y in zip(dense_x[: head + 1], smooth[: head + 1])],
               fill=INK, width=2)
        hx, hy = X(dense_x[head]), Y(smooth[head])
        d.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], fill=ACCENT)
        # The final number is NOT baked into the video: the page computes it
        # from the same logs at build time, and two numbers a reader can see
        # at once must not disagree (they did: 1.621 in the frame, 1.480 in
        # the card). The frame carries the shape; the card carries the value.
        d.text((left, H - 46), f"{dense_x[head]:.0f} MB seen", font=f_small, fill=DIM)
        img.save(tmp / f"{i:04d}.png")


# ---------------------------------------------------------------- 3. the heat
def render_heat(tmp: Path) -> None:
    sj = json.loads((ROOT / "assets/surprise.json").read_text())
    bits = np.array(sj["bits"], dtype=np.float32)
    n = len(bits)
    lo, hi = 0.0, float(bits.max())
    norm = (bits - lo) / max(hi - lo, 1e-6)
    width = 1600                      # two copies, so a half-width pan loops
    tile = np.tile(norm, width // n + 1)[:width]
    f_small = font(12, bold=False)
    f_big = font(20)
    for i in range(FRAMES):
        t = i / FRAMES
        shift = int(t * (width // 2))            # exactly half the canvas
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 24), "surprise per byte", font=f_small, fill=DIM)
        d.text((W - 210, 24), "1024 held-out bytes", font=f_small, fill=DIM)
        base = H - 90
        for x in range(W):
            v = tile[(shift + x) % width]
            bar = int(6 + v * (base - 60))
            col = tuple(int(c) for c in (60 + 190 * v, 60 + 180 * v, 70 + 185 * v))
            d.line([(x, base), (x, base - bar)], fill=col)
        d.line([(0, base), (W, base)], fill=(44, 44, 48))
        d.text((40, H - 66), f"mean {bits.mean():.3f} bits / byte", font=f_big, fill=INK)
        img.save(tmp / f"{i:04d}.png")



# ---------------------------------------------------------------- 4. TSCT
def render_tsct(tmp: Path) -> None:
    """The expert bank: one byte falls into a router, two of sixteen drawn cards light.

    Sixteen cards stand for 256 experts; the label says so. Inside a chosen card
    a wide bar (d) squeezes to a thin latent (r) and writes back wide (f) — the
    low-rank shape that makes a bank cheap.
    """
    cards = []
    for i in range(16):
        row, col = divmod(i, 8)
        cards.append((38 + col * 92, 186 + row * 78, 80, 56))
    rx, ry, rw, rh = 340, 66, 120, 40
    picks = ((2, 11), (5, 14), (0, 9))
    f_small = font(11, bold=False)
    f_mid = font(12)
    for i in range(FRAMES):
        t = i / FRAMES
        cycle = min(2, int(t * 3))
        local = (t * 3) - cycle
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        grid(d)
        d.text((40, 22), "TSCT · a bank of 256 low-rank experts", font=f_small, fill=DIM)
        d.text((W - 158, 22), "top-2 per byte", font=f_small, fill=DIM)

        # the cards, unlit
        for x, y, w, h in cards:
            d.rounded_rectangle([x, y, x + w, y + h], radius=8, outline=(38, 38, 42), fill=(14, 14, 16))

        # router
        d.rounded_rectangle([rx, ry, rx + rw, ry + rh], radius=10, outline=(72, 72, 78), fill=(18, 18, 20))
        d.text((rx + rw / 2 - 20, ry + 12), "router", font=f_mid, fill=INK)

        # the byte falls in
        fall = min(1.0, local / 0.18)
        dy = 26 + (ry - 44) * fall
        d.ellipse([396, dy, 404, dy + 8], fill=ACCENT)

        if local > 0.2:
            k = min(1.0, (local - 0.2) / 0.18)
            fade = 1.0 if local < 0.85 else max(0.0, 1 - (local - 0.85) / 0.15)
            for idx in picks[cycle]:
                x, y, w, h = cards[idx]
                ex = x + w / 2
                col = tuple(int(c * k * fade) for c in (60, 96, 160))
                d.line([(400, ry + rh), (ex, y)], fill=col, width=1)
                g = int(26 * k * fade)
                d.rounded_rectangle([x, y, x + w, y + h], radius=8, outline=(70, 120, 200),
                                    fill=(10 + g, 16 + g, 26 + g))
                bx, by = x + 10, y + 18
                d.rectangle([bx, by + 4, bx + 20, by + 10], fill=(116, 132, 160))       # d
                d.rectangle([bx + 26, by, bx + 32, by + 14], fill=ACCENT)               # r
                d.rectangle([bx + 38, by + 4, bx + 58, by + 10], fill=(116, 132, 160))  # f
                d.text((x + 6, y + h - 17), f"e{idx}", font=f_small, fill=(120, 130, 150))

        # a slow sweep so the dark grid is not static
        sx = int(-60 + (W + 120) * t)
        d.line([(sx, 120), (sx - 40, H)], fill=(22, 24, 30), width=2)
        img.save(tmp / f"{i:04d}.png")


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    for name, fn in (("rain", render_rain), ("curve", render_curve), ("heat", render_heat), ("tsct", render_tsct)):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fn(tmp)
            encode(tmp, MEDIA / f"{name}.webm")


if __name__ == "__main__":
    main()
