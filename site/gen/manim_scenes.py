"""The landing page's film strip, rendered with Manim.

Six loops, one idea each, drawn in the page's palette: warm paper, ink, one
blue accent. Every scene is a closed loop (fade in, fade out) so it can run
unattended in a `<video autoplay loop>`.

Layout contract, so nothing collides: the safe box is x ∈ [-6.3, 6.3],
y ∈ [-3.2, 3.4]; the caption line owns y = -3.45 and nothing else enters it.

    manim -qm --format=mp4 site/gen/manim_scenes.py Bytes
    bash site/gen/render_clips.sh          # all six + posters
"""

import os

from manim import *  # noqa: F403

# Two palettes, the same two the page ships. `CLIP_THEME=dark` renders the set
# for the dark theme; the renderer writes `_dark` suffixes and the page swaps
# the `src` when the reader flips the theme.
DARK = os.environ.get("CLIP_THEME", "paper").lower() == "dark"

if DARK:
    PAPER = "#0A0B0D"
    INK = "#E8E6E1"
    DIM = "#9AA0A9"
    DIMMER = "#7C818B"
    ACCENT = "#7AA2E8"
    HOT = "#FF6B7A"
    GOOD = "#5FBF7F"
else:
    PAPER = "#F6F3EC"
    INK = "#1B1A17"
    DIM = "#6D675D"
    DIMMER = "#9A9386"
    ACCENT = "#2F6FD0"
    HOT = "#C0392B"
    GOOD = "#2E7D4F"

MONO = "DejaVu Sans Mono"

CAP_Y = -3.45


def mono(t, size=24, color=INK, weight=NORMAL):
    return Text(t, font=MONO, font_size=size, color=color, weight=weight)


class Clip(Scene):
    caption = ""

    def setup(self):
        self.camera.background_color = PAPER
        self.cap = None
        if self.caption:
            self.cap = mono(self.caption, size=17, color=DIMMER).move_to([0, CAP_Y, 0])

    def say(self, run=0.5):
        return FadeIn(self.cap, run_time=run)

    def clear(self, run=0.8):
        """Fade everything out, the caption last, so the loop closes cleanly."""
        objs = [m for m in self.mobjects if m is not self.cap]
        anims = [FadeOut(o) for o in objs]
        if self.cap is not None and self.cap in self.mobjects:
            anims.append(FadeOut(self.cap))
        self.play(*anims, run_time=run)


def cell_grid(rows, cols, w, h, color=INK, lo=0.06, hi=0.75, seed=1):
    def shade(i, j):
        x = (i * 7919 + j * 104729 + seed * 15486) % 997
        return lo + (hi - lo) * (x / 997.0) ** 1.5

    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            side = min(w / cols, h / rows) * 0.88
            g.add(
                Square(side_length=side, stroke_width=0, fill_color=color, fill_opacity=shade(r, c)).move_to(
                    [(c - (cols - 1) / 2) * (w / cols), ((rows - 1) / 2 - r) * (h / rows), 0]
                )
            )
    return g


class Bytes(Clip):
    """256 symbols: no vocabulary, the file itself."""

    caption = "256 symbols · the file itself · no tokenizer"

    def construct(self):
        word = mono("byte", size=120, color=INK, weight=BOLD).move_to([0, 0.6, 0])
        self.play(FadeIn(word, scale=0.94), run_time=0.9)
        self.play(word.animate.scale(0.42).move_to([0, 2.75, 0]), run_time=0.7)

        grid = cell_grid(16, 16, 4.6, 4.6, seed=4).move_to([0, -0.1, 0])
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in grid], lag_ratio=0.004), run_time=1.8)

        # one cell is a byte
        target = grid[2 * 16 + 1]
        self.play(target.animate.set_fill(ACCENT, opacity=1).scale(1.5), run_time=0.6)
        bits = VGroup(*[mono(b, size=44, color=ACCENT) for b in "01100010"]).arrange(RIGHT, buff=0.14)
        bits.move_to([0, -2.55, 0])
        self.play(FadeIn(bits, shift=UP * 0.12), run_time=0.9)
        self.play(FadeIn(mono("one byte = one character", size=20, color=DIM).next_to(bits, DOWN, buff=0.3), run_time=0.4))

        self.play(self.say(), run_time=0.4)
        self.play(Indicate(bits, color=ACCENT, scale_factor=1.03), run_time=0.9)
        self.wait(0.4)
        self.clear()


class Ternary(Clip):
    """Continuous weights snap to three levels; the file follows the levels."""

    caption = "three levels per weight · 82 MB instead of 190"

    def construct(self):
        axis_x = -3.4
        for i, (y, lab) in enumerate([(2.1, "+α"), (0.0, "0"), (-2.1, "−α")]):
            self.play(
                Create(Line([axis_x - 1.7, y, 0], [axis_x + 1.7, y, 0], stroke_width=1.5, stroke_color=INK, stroke_opacity=0.4)),
                FadeIn(mono(lab, size=22, color=DIM).move_to([axis_x - 2.1, y, 0])),
                run_time=0.28,
            )

        import random

        random.seed(7)
        dots = VGroup()
        for _ in range(180):
            x = axis_x + random.uniform(-1.55, 1.55)
            y = max(-2.6, min(2.6, random.gauss(0, 0.95)))
            dots.add(Dot([x, y, 0], radius=0.032, color=INK, fill_opacity=0.7))
        self.play(LaggedStart(*[FadeIn(d, scale=0.6) for d in dots], lag_ratio=0.006), run_time=1.6)

        levels = [2.1, 0.0, -2.1]
        snapped = VGroup()
        for d in dots:
            x, y = d.get_center()[0], d.get_center()[1]
            near = min(levels, key=lambda L: abs(L - y))
            snapped.add(Dot([x, near, 0], radius=0.032, color=ACCENT, fill_opacity=0.9))
        self.play(Transform(dots, snapped), run_time=2.0)

        counts = VGroup(
            mono("−α  40%", size=18, color=DIM),
            mono("0   20%", size=18, color=DIM),
            mono("+α  40%", size=18, color=DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([axis_x, -2.9, 0]).shift(DOWN * 0.0)
        counts.move_to([axis_x - 0.4, -3.0, 0])
        self.play(FadeIn(counts), run_time=0.4)

        # the file, in two steps
        track = Rectangle(width=4.6, height=0.4, stroke_width=1, stroke_color=INK, stroke_opacity=0.3, fill_opacity=0)
        track.move_to([3.5, 1.35, 0])
        full = Rectangle(width=4.6, height=0.4, stroke_width=0, fill_color=DIMMER, fill_opacity=0.5)
        full.align_to(track, LEFT).set_y(track.get_center()[1])
        lab = mono("bf16", size=21, color=DIM).next_to(track, UP, buff=0.28)
        self.play(FadeIn(track), FadeIn(full), FadeIn(lab), run_time=0.5)

        ter = Rectangle(width=0.58, height=0.4, stroke_width=0, fill_color=ACCENT, fill_opacity=0.9)
        ter.align_to(track, LEFT).set_y(track.get_center()[1])
        self.play(Transform(full, ter), Transform(lab, mono("ternary", size=21, color=ACCENT).next_to(track, UP, buff=0.28)), run_time=1.6)

        self.play(self.say(), run_time=0.4)
        self.play(FadeIn(mono("8× fewer bytes · 2 bits per weight", size=17, color=DIMMER).next_to(track, DOWN, buff=0.3)), run_time=0.5)
        self.wait(0.6)
        self.clear()


class Spectral(Clip):
    """TSCT is spectral compression, not routing: cut the spectrum, keep the head."""

    caption = "spectral cut · rank 6 of 14 · then three levels"

    def construct(self):
        W = cell_grid(8, 12, 2.6, 2.0, seed=7, lo=0.05, hi=0.6).move_to([-4.5, 1.15, 0])
        W_lab = mono("W 832 × 2048", size=18, color=DIM).next_to(W, UP, buff=0.22)
        self.play(FadeIn(W), FadeIn(W_lab), run_time=0.9)

        # the spectrum, wide and low, under everything
        vals = [2.0, 1.35, 0.92, 0.6, 0.38, 0.24, 0.14, 0.09, 0.055, 0.03, 0.018, 0.011, 0.007, 0.004]
        base = -1.35
        x0, dx = -5.6, 0.52
        bars = VGroup()
        for i, v in enumerate(vals):
            bars.add(
                Rectangle(width=0.3, height=v * 1.05, stroke_width=0, fill_color=INK, fill_opacity=0.55)
                .move_to([x0 + i * dx, base + v * 1.05 / 2, 0])
            )
        axis = Line([x0 - 0.3, base, 0], [x0 + len(vals) * dx, base, 0], stroke_width=1.1, stroke_color=DIM)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05), Create(axis), run_time=1.6)
        self.play(FadeIn(mono("σᵢ", size=18, color=DIMMER).move_to([x0 - 0.55, base + 0.35, 0])), run_time=0.3)

        cut_x = x0 + 5.5 * dx
        cut = DashedLine([cut_x, base - 0.15, 0], [cut_x, base + 2.55, 0], stroke_color=ACCENT, stroke_width=2, dash_length=0.09)
        self.play(Create(cut), run_time=0.4)
        self.play(VGroup(*bars[6:]).animate.set_opacity(0.15), run_time=0.9)
        self.play(FadeIn(mono("rank 6 keeps 87% of the energy", size=18, color=ACCENT).move_to([x0 + 3.1, base - 0.55, 0])), run_time=0.5)

        # W becomes two thin factors
        U = cell_grid(8, 6, 1.25, 2.0, color=ACCENT, lo=0.25, hi=0.9, seed=31).move_to([1.3, 1.55, 0])
        S = VGroup(*[Square(side_length=0.13, stroke_width=0, fill_color=HOT, fill_opacity=0.9) for _ in range(6)])
        S.arrange(DOWN, buff=0.24).move_to([2.35, 1.55, 0])
        V = cell_grid(6, 12, 2.1, 1.05, color=ACCENT, lo=0.25, hi=0.9, seed=57).move_to([4.2, 1.55, 0])

        self.play(
            FadeOut(W),
            FadeOut(W_lab),
            LaggedStart(FadeIn(U, shift=RIGHT * 0.15), FadeIn(S, shift=RIGHT * 0.15), FadeIn(V, shift=RIGHT * 0.15), lag_ratio=0.3),
            run_time=1.4,
        )
        self.play(
            FadeIn(mono("W ≈ U Σ Vᵀ", size=19, color=DIM).move_to([2.75, 3.0, 0])),
            FadeIn(mono("U", size=19, color=ACCENT).move_to([1.3, 0.25, 0])),
            FadeIn(mono("Σ", size=19, color=HOT).move_to([2.35, 0.25, 0])),
            FadeIn(mono("Vᵀ", size=19, color=ACCENT).move_to([4.2, 0.25, 0])),
            run_time=0.5,
        )

        # the factors are stored in three levels
        lev = [INK, ACCENT, PAPER]
        for k, cell in enumerate(U):
            cell.set_fill(lev[(k * 7 + 3) % 3], opacity=0.95)
        for k, cell in enumerate(V):
            cell.set_fill(lev[(k * 5 + 1) % 3], opacity=0.95)
        self.play(LaggedStart(*[c.animate.set_opacity(1) for c in U], lag_ratio=0.008), run_time=1.0)
        self.play(FadeIn(mono("−α / 0 / +α", size=18, color=DIM).move_to([4.2, -0.3, 0])), run_time=0.4)

        self.play(self.say(), run_time=0.4)
        self.wait(0.5)
        self.clear()


class Muon(Clip):
    """Newton–Schulz flattens the spectrum: every direction gets the same strength."""

    caption = "muon · five steps · every direction the same strength"

    def construct(self):
        vals = [2.35, 1.85, 1.4, 1.02, 0.7, 0.47, 0.31, 0.2, 0.13, 0.085]
        n = len(vals)
        axes = Axes(
            x_range=[0, n + 0.7, 1],
            y_range=[0, 2.6, 0.5],
            x_length=9.0,
            y_length=3.6,
            axis_config={"stroke_color": DIM, "stroke_width": 1.2, "include_tip": False},
        ).move_to([0, 0.55, 0])
        base = axes.c2p(0, 0)[1]

        def bars_for(values, color, opacity):
            g = VGroup()
            for i, v in enumerate(values):
                h = v * axes.y_length / 2.6
                g.add(
                    Rectangle(width=0.46, height=h, stroke_width=0, fill_color=color, fill_opacity=opacity)
                    .move_to([axes.c2p(i + 1, 0)[0], base + h / 2, 0])
                )
            return g

        bars = bars_for(vals, INK, 0.6)
        self.play(Create(axes), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.06), run_time=1.2)
        lab = mono("σᵢ · skewed", size=20, color=DIMMER).move_to([0, 3.05, 0])
        self.play(FadeIn(lab), run_time=0.3)

        # three iterations of flattening
        for step, target in enumerate([1.55, 1.2, 1.0], start=1):
            flat = bars_for([target] * n, ACCENT, 0.35 + 0.2 * step)
            self.play(
                Transform(bars, flat),
                Transform(lab, mono(f"σᵢ · step {step}", size=20, color=ACCENT).move_to([0, 3.05, 0])),
                run_time=1.2,
            )

        self.play(self.say(), run_time=0.4)
        self.play(Indicate(bars, color=ACCENT, scale_factor=1.02), run_time=0.9)
        self.wait(0.5)
        self.clear()


class Curve(Clip):
    """1.19 bits per byte against the bytes seen, with the compressors for scale."""

    caption = "held-out bits per byte · the gap is the result, not the height"

    def construct(self):
        axes = Axes(
            x_range=[0, 2.75, 0.5],
            y_range=[1.0, 2.7, 0.5],
            x_length=9.6,
            y_length=3.9,
            axis_config={"stroke_color": DIM, "stroke_width": 1.2, "include_tip": False},
            x_axis_config={"numbers_to_include": [0, 1, 2], "font_size": 17},
            y_axis_config={"numbers_to_include": [1.0, 1.5, 2.0, 2.5], "font_size": 17},
        ).move_to([0.1, 0.4, 0])

        xz = DashedLine(axes.c2p(0, 2.31), axes.c2p(2.72, 2.31), stroke_color=DIMMER, stroke_width=1.5, dash_length=0.1)
        xz_lab = mono("xz 2.31", size=17, color=DIM).move_to([axes.c2p(2.2, 2.44)[0], axes.c2p(2.2, 2.44)[1], 0])
        five = DashedLine(axes.c2p(0, 2.57), axes.c2p(2.72, 2.57), stroke_color=DIMMER, stroke_width=1.1, dash_length=0.06)
        five_lab = mono("5-gram 2.57", size=16, color=DIMMER).move_to([axes.c2p(2.2, 2.66)[0], axes.c2p(2.2, 2.66)[1], 0])

        self.play(Create(axes), run_time=0.7)
        self.play(FadeIn(xz), FadeIn(xz_lab), FadeIn(five), FadeIn(five_lab), run_time=0.6)

        pts = [(0.03, 1.56, "30 MB"), (0.12, 1.49, "120 MB"), (0.30, 1.40, None),
               (1.10, 1.29, "1.1 GB"), (2.00, 1.19, "2.0 GB"), (2.64, 1.19, None)]
        curve = VMobject(stroke_color=ACCENT, stroke_width=3.4)
        curve.set_points_smoothly([axes.c2p(x, y) for x, y, _ in pts])
        self.play(Create(curve), run_time=2.4)

        dots = VGroup()
        for x, y, label in pts:
            dots.add(Dot(axes.c2p(x, y), radius=0.055, color=ACCENT))
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.12), run_time=1.0)

        m1 = mono("1.56", size=19, color=DIM).move_to([axes.c2p(0.03, 1.56)[0] + 0.55, axes.c2p(0.03, 1.56)[1] + 0.22, 0])
        m2 = mono("1.49", size=19, color=DIM).move_to([axes.c2p(0.30, 1.40)[0], axes.c2p(0.30, 1.40)[1] + 0.3, 0])
        self.play(FadeIn(m1), FadeIn(m2), run_time=0.5)

        end = Dot(axes.c2p(2.64, 1.19), radius=0.09, color=ACCENT)
        self.play(FadeIn(end, scale=1.8), run_time=0.4)
        self.play(self.say(), run_time=0.4)
        self.play(Indicate(end, scale_factor=1.3), run_time=0.9)
        self.play(FadeIn(mono("this run ends here — the next architecture moves it", size=18, color=DIM).move_to([axes.c2p(1.55, 1.05)[0], axes.c2p(1.55, 1.05)[1], 0]), run_time=0.5))
        self.wait(0.6)
        self.clear()


class Surprise(Clip):
    """Each byte is a bet; the tall marks are where the model hesitated."""

    caption = "−log₂ p for every byte · taller means less certain"

    def construct(self):
        import random

        random.seed(5)
        text = "the model reads the file itself"
        values = []
        for ch in text:
            base = 0.3 if ch == " " else 0.6
            if ch in "fqzkx":
                base += 1.5
            values.append(base + random.uniform(0, 0.8))

        word = mono(text, size=28, color=INK).move_to([0, -1.5, 0])
        left = word.get_left()[0]
        step = word.width / len(text)
        top = word.get_top()[1] + 0.4

        bars = VGroup()
        for i, v in enumerate(values):
            h = 0.3 + v * 0.75
            bars.add(
                Rectangle(
                    width=step * 0.6,
                    height=h,
                    stroke_width=0,
                    fill_color=HOT if v > 1.7 else INK,
                    fill_opacity=0.85 if v > 1.7 else 0.5,
                ).move_to([left + step * (i + 0.5), top + h / 2, 0])
            )

        self.play(AddTextLetterByLetter(word, run_time=2.2))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.018), run_time=2.2)

        axis = Line([left - 0.15, top, 0], [word.get_right()[0] + 0.15, top, 0], stroke_width=1, stroke_color=DIM)
        self.play(Create(axis), run_time=0.3)

        tall = max(range(len(values)), key=lambda i: values[i])
        self.play(Indicate(bars[tall], color=HOT, scale_factor=1.35), run_time=0.9)
        self.play(FadeIn(mono("a guess here", size=17, color=HOT).next_to(bars[tall], UP, buff=0.18), run_time=0.4))

        self.play(self.say(), run_time=0.4)
        self.play(FadeIn(mono("mean 1.19 bits per byte", size=19, color=DIM).move_to([0, 2.85, 0]), run_time=0.5))
        self.wait(0.6)
        self.clear()
