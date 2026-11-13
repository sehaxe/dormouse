"""The promo reel: how the model works, drawn in matrices.

One 50-second film, no narration, in the landing page's palette. It walks the
actual pipeline — bytes → blocks → attention → the training step → the curve →
the compression — with real shapes and real numbers, not a metaphor.

    manim -qh --format=mp4 site/gen/manim_promo.py Promo
"""

import os

from manim import *  # noqa: F403

DARK = os.environ.get("CLIP_THEME", "paper").lower() == "dark"

if DARK:
    PAPER, INK = "#0A0B0D", "#E8E6E1"
    DIM, DIMMER = "#9AA0A9", "#7C818B"
    ACCENT, HOT, GOOD = "#7AA2E8", "#FF6B7A", "#5FBF7F"
else:
    PAPER, INK = "#F6F3EC", "#1B1A17"
    DIM, DIMMER = "#6D675D", "#9A9386"
    ACCENT, HOT, GOOD = "#2F6FD0", "#C0392B", "#2E7D4F"

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"


def mono(t, size=24, color=INK, weight=NORMAL):
    return Text(t, font=MONO, font_size=size, color=color, weight=weight)


def caption(t, color=DIMMER):
    return mono(t, size=17, color=color).to_edge(DOWN, buff=0.42)


def heat(rows, cols, w, h, seed=1, color=INK, lo=0.06, hi=0.9):
    """A matrix drawn as a grid of cells whose shade is its magnitude."""

    def f(i, j):
        x = (i * 7919 + j * 104729 + seed * 15486) % 997
        return lo + (hi - lo) * (x / 997.0) ** 1.6

    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            side = min(w / cols, h / rows) * 0.9
            g.add(
                Square(side_length=side, stroke_width=0, fill_color=color, fill_opacity=f(r, c)).move_to(
                    [(c - (cols - 1) / 2) * (w / cols), ((rows - 1) / 2 - r) * (h / rows), 0]
                )
            )
    return g


class Promo(Scene):
    def setup(self):
        self.camera.background_color = PAPER

    def section(self, title, sub=None, run=1.2):
        kick = mono(title, size=19, color=ACCENT)
        line = mono(sub, size=30, color=INK) if sub else None
        group = VGroup(kick, line) if line else VGroup(kick)
        if line:
            group.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        group.to_edge(UP, buff=0.75).to_edge(LEFT, buff=0.9)
        self.play(FadeIn(kick, shift=RIGHT * 0.15), run_time=0.35)
        if line:
            self.play(FadeIn(line, shift=RIGHT * 0.15), run_time=0.4)
        return group

    # ---------------------------------------------------------------- 1. bytes
    def part_bytes(self):
        self.play(FadeIn(mono("dormouse", size=96, color=INK, weight=BOLD).move_to(UP * 0.6)), run_time=1.0)
        self.play(FadeIn(mono("a byte-level model · one GPU", size=26, color=DIM).move_to(DOWN * 0.4)), run_time=0.6)
        self.wait(0.6)

        cells = heat(16, 16, 5.2, 5.2, seed=4, lo=0.02, hi=0.55)
        self.play(
            FadeOut(self.mobjects[0]),
            FadeOut(self.mobjects[1] if len(self.mobjects) > 1 else VGroup()),
            FadeIn(cells, scale=0.94),
            run_time=1.2,
        )
        bits = mono("01100100 01101111 01110010 01101101", size=30, color=ACCENT).next_to(cells, DOWN, buff=0.5)
        self.play(FadeIn(bits, shift=UP * 0.12), run_time=0.7)
        self.play(Indicate(bits, color=ACCENT, scale_factor=1.02), run_time=0.8)
        self.play(FadeOut(cells), FadeOut(bits), run_time=0.6)

    # --------------------------------------------------------------- 2. blocks
    def part_blocks(self):
        head = self.section("01 · the model", "twelve blocks, one direction")
        blocks = VGroup()
        for i in range(12):
            r = Rectangle(width=0.62, height=2.5, stroke_color=INK, stroke_width=1.1, fill_color=INK, fill_opacity=0.05)
            r.move_to([(i - 5.5) * 0.86, 0, 0])
            blocks.add(r)
        vec = VGroup(*[Square(side_length=0.1, stroke_width=0, fill_color=ACCENT, fill_opacity=0.9) for _ in range(12)])
        vec.arrange(RIGHT, buff=0.05).next_to(blocks, LEFT, buff=0.5)

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.1) for b in blocks], lag_ratio=0.06), run_time=1.0)
        self.play(FadeIn(vec), run_time=0.3)

        for i in range(12):
            self.play(
                vec.animate.next_to(blocks[i], UP, buff=0.25).scale(1.0),
                blocks[i].animate.set_fill(ACCENT, opacity=0.16).set_stroke(ACCENT, 2),
                run_time=0.11 if i else 0.25,
            )
            self.play(blocks[i].animate.set_fill(INK, opacity=0.05).set_stroke(INK, 1.1), run_time=0.06)

        lab = mono("12 × 832 · gated FFN · 95,035,200 parameters", size=20, color=DIM).next_to(blocks, DOWN, buff=0.55)
        self.play(FadeIn(lab), run_time=0.4)
        self.wait(0.4)
        self.play(FadeOut(VGroup(blocks, vec, lab, head)), run_time=0.6)

    # ------------------------------------------------------------ 3. attention
    def part_attention(self):
        head = self.section("02 · attention", "every position looks back")
        n = 12
        cellw = 0.36
        mask = VGroup()
        for r in range(n):
            for c in range(n):
                on = c <= r
                mask.add(
                    Square(
                        side_length=cellw * 0.86,
                        stroke_width=0,
                        fill_color=ACCENT if on else INK,
                        fill_opacity=(0.15 + 0.55 * ((r + c) % 5) / 5) if on else 0.05,
                    ).move_to([(c - (n - 1) / 2) * cellw, ((n - 1) / 2 - r) * cellw, 0])
                )
        mask.move_to(LEFT * 2.4 + UP * 0.1)
        lab = mono("causal mask", size=19, color=DIMMER).next_to(mask, UP, buff=0.3)

        q = heat(6, 6, 1.5, 1.5, seed=11, color=HOT, lo=0.15, hi=0.85).next_to(mask, RIGHT, buff=1.5).shift(UP * 1.2)
        k = heat(6, 6, 1.5, 1.5, seed=23, color=GOOD, lo=0.15, hi=0.85).next_to(q, DOWN, buff=0.35)

        self.play(FadeIn(mask, scale=0.97), FadeIn(lab), run_time=0.9)
        self.play(FadeIn(q, shift=LEFT * 0.2), FadeIn(k, shift=LEFT * 0.2), run_time=0.6)
        self.play(
            FadeIn(mono("Q", size=20, color=HOT).next_to(q, LEFT, buff=0.25)),
            FadeIn(mono("Kᵀ", size=20, color=GOOD).next_to(k, LEFT, buff=0.25)),
            FadeIn(mono("softmax(QKᵀ/√d)", size=19, color=DIM).next_to(mask, DOWN, buff=0.4)),
            run_time=0.5,
        )
        self.play(
            LaggedStart(*[Indicate(m, color=ACCENT, scale_factor=1.05) for m in mask[::13]], lag_ratio=0.05),
            run_time=1.6,
        )
        self.wait(0.3)
        self.play(FadeOut(VGroup(mask, lab, q, k, head)), *[FadeOut(m) for m in self.mobjects if isinstance(m, Text) and m not in (head,)], run_time=0.6)

    # --------------------------------------------------------------- 4. train
    def part_train(self):
        head = self.section("03 · one step", "forward, backward, orthogonalize")
        W = heat(10, 10, 2.6, 2.6, seed=7, color=INK, lo=0.05, hi=0.8).move_to(LEFT * 4.0 + UP * 0.1)
        x = VGroup(*[Square(side_length=0.16, stroke_width=0, fill_color=ACCENT, fill_opacity=0.85) for _ in range(10)])
        x.arrange(DOWN, buff=0.07).next_to(W, LEFT, buff=0.45)
        y = VGroup(*[Square(side_length=0.16, stroke_width=0, fill_color=ACCENT, fill_opacity=0.35) for _ in range(10)])
        y.arrange(DOWN, buff=0.07).next_to(W, RIGHT, buff=0.45)

        self.play(FadeIn(x), FadeIn(W), FadeIn(y), run_time=0.7)
        self.play(
            Indicate(x, color=ACCENT, scale_factor=1.1),
            W.animate.set_fill(ACCENT, opacity=0.22),
            run_time=0.8,
        )
        self.play(W.animate.set_fill(INK, opacity=0.72).set_stroke(INK, 0), run_time=0.5)

        loss = mono("loss 2.61", size=30, color=INK).move_to(RIGHT * 3.2 + UP * 1.5)
        bpb = mono("1.19 bits / byte", size=34, color=ACCENT).move_to(RIGHT * 3.2 + UP * 0.7)
        self.play(FadeIn(loss), run_time=0.3)
        self.play(Transform(loss, mono("loss 1.19 · held out", size=30, color=INK).move_to(RIGHT * 3.2 + UP * 1.5)), run_time=1.0)
        self.play(FadeIn(bpb, shift=UP * 0.1), run_time=0.5)

        # Muon: the spectrum flattens
        vals = [1.9, 1.35, 0.95, 0.62, 0.4, 0.25, 0.15]
        bars = VGroup()
        for i, v in enumerate(vals):
            bars.add(
                Rectangle(width=0.26, height=v, stroke_width=0, fill_color=INK, fill_opacity=0.6)
                .move_to([1.9 + i * 0.42, -1.35 + v / 2, 0])
            )
        axis = Line([1.65, -1.35, 0], [4.75, -1.35, 0], stroke_width=1.1, stroke_color=DIM)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.06), Create(axis), run_time=0.9)
        flat = VGroup()
        for i in range(len(vals)):
            flat.add(
                Rectangle(width=0.26, height=1.2, stroke_width=0, fill_color=ACCENT, fill_opacity=0.8)
                .move_to([1.9 + i * 0.42, -1.35 + 0.6, 0])
            )
        self.play(Transform(bars, flat), run_time=1.4)
        self.play(FadeIn(mono("muon · all directions equal", size=18, color=DIM).next_to(axis, DOWN, buff=0.25)), run_time=0.3)
        self.wait(0.4)
        self.play(FadeOut(VGroup(W, x, y, loss, bpb, bars, axis)), *[FadeOut(m) for m in self.mobjects if isinstance(m, Text)], run_time=0.6)

    # --------------------------------------------------------------- 5. curve
    def part_curve(self):
        head = self.section("04 · the result", "held-out bits per byte")
        axes = Axes(
            x_range=[0, 2.7, 0.5],
            y_range=[1.0, 2.7, 0.5],
            x_length=10.0,
            y_length=3.5,
            axis_config={"stroke_color": DIM, "stroke_width": 1.2, "include_tip": False},
            x_axis_config={"numbers_to_include": [0, 1, 2], "font_size": 18},
            y_axis_config={"numbers_to_include": [1.0, 1.5, 2.0, 2.5], "font_size": 18},
        ).move_to([0, 0.15, 0])
        pts = [(0.03, 1.56), (0.12, 1.49), (0.30, 1.40), (1.10, 1.29), (2.00, 1.19), (2.64, 1.19)]
        curve = VMobject(stroke_color=ACCENT, stroke_width=3.6)
        curve.set_points_smoothly([axes.c2p(x, y) for x, y in pts])
        xz = DashedLine(axes.c2p(0, 2.31), axes.c2p(2.66, 2.31), stroke_color=DIMMER, stroke_width=1.6, dash_length=0.1)

        self.play(Create(axes), FadeIn(xz), FadeIn(mono("xz 2.31", size=18, color=DIM).next_to(xz, RIGHT, buff=0.12)), run_time=0.7)
        self.play(Create(curve), run_time=2.0)
        dot = Dot(axes.c2p(2.64, 1.19), radius=0.075, color=ACCENT)
        lab = mono("1.19", size=38, color=ACCENT).next_to(dot, RIGHT, buff=0.2).align_to(dot, UP)
        self.play(FadeIn(dot, scale=1.7), FadeIn(lab, shift=RIGHT * 0.1), run_time=0.6)
        self.play(Indicate(lab, scale_factor=1.08), run_time=0.9)
        self.wait(0.5)
        self.play(FadeOut(VGroup(axes, curve, xz, dot, lab)), *[FadeOut(m) for m in self.mobjects if isinstance(m, Text)], run_time=0.6)

    # ----------------------------------------------------------- 6. compress
    def part_compress(self):
        head = self.section("05 · the file", "spectral cut, then three levels")
        W = heat(10, 14, 3.0, 2.2, seed=9, color=INK, lo=0.05, hi=0.7).move_to(LEFT * 4.2 + UP * 0.2)
        U = heat(10, 5, 1.05, 2.2, seed=31, color=ACCENT, lo=0.25, hi=0.95).move_to(UP * 0.2)
        V = heat(5, 14, 2.2, 1.05, seed=57, color=ACCENT, lo=0.25, hi=0.95).next_to(U, RIGHT, buff=0.5).align_to(U, UP)

        self.play(FadeIn(W), run_time=0.6)
        self.play(FadeOut(W), FadeIn(U), FadeIn(V),
                  FadeIn(mono("≈", size=32, color=DIM).move_to(LEFT * 2.2 + UP * 0.2)),
                  FadeIn(mono("U", size=20, color=ACCENT).next_to(U, DOWN, buff=0.2)),
                  FadeIn(mono("Vᵀ", size=20, color=ACCENT).next_to(V, DOWN, buff=0.2)),
                  run_time=0.9)

        track = Rectangle(width=5.4, height=0.34, stroke_width=1, stroke_color=DIM, fill_opacity=0).move_to(RIGHT * 3.4 + DOWN * 1.6)
        full = Rectangle(width=5.4, height=0.34, stroke_width=0, fill_color=DIMMER, fill_opacity=0.5)
        full.align_to(track, LEFT).set_y(track.get_center()[1])
        size = mono("190 MB", size=22, color=DIM).next_to(track, DOWN, buff=0.2)

        self.play(FadeIn(track), FadeIn(full), FadeIn(size), run_time=0.5)
        short = Rectangle(width=2.33, height=0.34, stroke_width=0, fill_color=ACCENT, fill_opacity=0.9)
        short.align_to(track, LEFT).set_y(track.get_center()[1])
        self.play(Transform(full, short), Transform(size, mono("82 MB", size=22, color=ACCENT).next_to(track, DOWN, buff=0.2)), run_time=1.2)
        self.wait(0.4)
        self.play(FadeOut(VGroup(U, V, track, full, size)), *[FadeOut(m) for m in self.mobjects if isinstance(m, Text)], run_time=0.6)

    # --------------------------------------------------------------- 7. close
    def part_close(self):
        big = mono("1.19", size=170, color=INK, weight=BOLD).move_to(UP * 0.75)
        unit = mono("bits per byte", size=34, color=DIM).next_to(big, DOWN, buff=0.2)
        strip = VGroup(
            mono("255 MB", size=30, color=ACCENT),
            mono("·", size=30, color=DIMMER),
            mono("12 000 tok/s", size=30, color=ACCENT),
            mono("·", size=30, color=DIMMER),
            mono("one GPU", size=30, color=ACCENT),
        ).arrange(RIGHT, buff=0.45).next_to(unit, DOWN, buff=0.75)

        self.play(FadeIn(big, scale=0.94), run_time=0.9)
        self.play(FadeIn(unit, shift=UP * 0.1), run_time=0.5)
        self.play(FadeIn(strip, shift=UP * 0.1), run_time=0.6)
        self.play(FadeIn(mono("dormouse", size=24, color=DIMMER).to_edge(DOWN, buff=0.6)), run_time=0.5)

    def construct(self):
        self.part_bytes()
        self.part_blocks()
        self.part_attention()
        self.part_train()
        self.part_curve()
        self.part_compress()
        self.part_close()
