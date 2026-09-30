#!/usr/bin/env python3
"""Parser tests for tools/trainboard.py — stdlib unittest, no fixtures beyond inline text.

Run: python3 tools/test_trainboard.py     (exit 1 on failure)

The gate that matters here is ADR-0019's: a malformed log line must be COUNTED and
skipped, never a crash and never a silent zero. So there is a test that a garbage
`step` line raises `skipped` and still returns every good line around it, and a
test that a field which is ABSENT is None (a gap) rather than 0.0.
"""
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trainboard as tb  # noqa: E402

# Verbatim from ~/logs/first_run_500_0930_1843.log (2026-09-30).
REAL500 = '''quant format: Fp8 (8 bits)
optimizer: Muon+ ColRow ns=8 + head-wise Muon q/k + Adam wd0 (tables) + AdamW (rest) [muon=15 qk=2 tables=36]
dormouse pretrain small params=9197454 steps=500 data="/mnt/x/real" lr=0.0001 backend=Device<Autodiff { device: Cube(Cuda(Cuda(0))) }>
warmup done
timer step 0: total=22354ms data=0.1ms fwd=6746ms bwd=7429ms (incl. loss sync + host-adam D2H) opt=7701ms retr=464.9ms ema=5.3ms gpu_step=22354ms
step      0 ce=5.571 bpb=8.038 best=5.571 lr=0.00e0 aux=0.0290 retr_arm=batched:0/factor:1
timer step 100: total=509ms data=0.1ms fwd=205ms bwd=211ms (incl. loss sync + host-adam D2H) opt=58ms retr=26.1ms ema=2.9ms gpu_step=509ms
step    100 ce=3.863 bpb=5.573 best=3.863 lr=1.00e-4 aux=0.0267 retr_arm=batched:0/factor:101
timer step 200: total=2081ms data=1.2ms fwd=922ms bwd=646ms (incl. loss sync + host-adam D2H) opt=350ms retr=139.2ms ema=15.8ms gpu_step=2080ms
step    200 ce=3.344 bpb=4.825 best=3.344 lr=1.00e-4 aux=0.0224 retr_arm=batched:0/factor:201
timer step 300: total=494ms data=0.2ms fwd=195ms bwd=214ms (incl. loss sync + host-adam D2H) opt=54ms retr=23.9ms ema=2.9ms gpu_step=494ms
step    300 ce=3.188 bpb=4.600 best=3.188 lr=1.00e-4 aux=0.0224 retr_arm=batched:0/factor:301
done steps=500 best ce=3.145 | no held-out eval ran, so no best checkpoint
'''

REAL_EVAL = 'step    500 EVAL ce=4.449 bpb=6.418 BEST over 81920 B (fixed window) fused kda=2012/0 asked=4096 bwd=0 declined=10276 ops=4096 node_bwd=0 norm=0/4619 muon_skipped=0/0 engram=0/0\n'

REAL_EVENTS = '''  firewall: 3 non-finite loss read(s) since the last log step (gradients zeroed on device; no optimizer-visible NaN)
guard: restarting in 30s (resume from last checkpoint, restart 1 of 3)
thread 'main' (44904) panicked at crates/dormouse-data/src/lib.rs:216:5:
train failed: config drift: 5 key(s) differ from the stored snapshot checkpoints/latest.config.toml
'''


def parse(text):
    with TemporaryDirectory() as d:
        p = Path(d) / "t.log"
        p.write_text(text)
        return tb.parse_log(p)


class ParseReal(unittest.TestCase):
    def setUp(self):
        self.r = parse(REAL500)

    def test_header(self):
        self.assertEqual(self.r.preset, "small")
        self.assertEqual(self.r.params, 9197454)
        self.assertEqual(self.r.header["steps"], "500")
        self.assertEqual(self.r.header["data"], "/mnt/x/real")
        self.assertEqual(self.r.quant, "Fp8 (8 bits)")
        self.assertEqual(self.r.headers, 1)

    def test_step_values_match_grep(self):
        # `grep 'step *300 ce=' REAL500` says 3.188
        s = self.r.steps[-1]
        self.assertEqual((s.step, s.ce, s.bpb, s.best, s.aux), (300, 3.188, 4.600, 3.188, 0.0224))
        self.assertEqual([p.step for p in self.r.steps], [0, 100, 200, 300])

    def test_timer_values_match_grep(self):
        t = self.r.timers[1]  # step 100
        self.assertEqual(t.step, 100)
        self.assertAlmostEqual(t.get("total"), 509.0, places=3)
        self.assertAlmostEqual(t.get("fwd"), 205.0, places=3)
        self.assertAlmostEqual(t.get("bwd"), 211.0, places=3)
        self.assertAlmostEqual(t.get("opt"), 58.0, places=3)
        self.assertAlmostEqual(t.get("retr"), 26.1, places=3)
        self.assertAlmostEqual(t.get("ema"), 2.9, places=3)
        self.assertAlmostEqual(t.get("gpu_step"), 509.0, places=3)
        self.assertAlmostEqual(self.r.timers[0].get("total"), 22354.0, places=0)

    def test_step0_timer_is_the_autotune_artifact(self):
        self.assertEqual(self.r.timers[0].step, 0)
        self.assertGreater(self.r.timers[0].get("total"), 10000)

    def test_done_line(self):
        self.assertIn("no held-out eval ran", self.r.done)
        self.assertEqual(self.r.evals, [])  # no EVAL line -> a gap, not a fake point

    def test_nothing_skipped(self):
        self.assertEqual(self.r.skipped, 0)


class ParseEval(unittest.TestCase):
    def setUp(self):
        self.r = parse(REAL500 + REAL_EVAL)

    def test_eval_window_is_captured(self):
        self.assertEqual(len(self.r.evals), 1)
        e = self.r.evals[0]
        self.assertEqual((e.step, e.bpb, e.window, e.is_best), (500, 6.418, 81920, True))
        self.assertEqual(e.ce, 4.449)

    def test_eval_is_not_a_train_point(self):
        self.assertEqual(len(self.r.steps), 4)  # unchanged: EVAL does not double-count
        self.assertEqual([e.step for e in self.r.evals], [500])

    def test_seam_tail_kept(self):
        seam = self.r.evals[0].seam
        self.assertIn("kda=2012/0", seam)
        self.assertIn("node_bwd=0", seam)     # \b stops bwd= eating node_bwd=
        self.assertIn("engram=0/0", seam)
        self.assertNotIn("EVAL", seam)
        self.assertNotIn("over", seam)
        self.assertNotIn("ce=", seam)


class Events(unittest.TestCase):
    def setUp(self):
        self.r = parse(REAL500 + REAL_EVENTS)

    def test_kinds(self):
        self.assertEqual([e.kind for e in self.r.events], ["nan", "restart", "panic", "fail"])
        self.assertEqual([e.step for e in self.r.events], [300, 300, 300, 300])

    def test_nan_firewall_text(self):
        self.assertIn("3 non-finite", self.r.events[0].text)

    def test_non_finite_ce_is_an_event_not_a_gap(self):
        # one good number keeps the point (with a gap); a wholly non-finite line
        # is the firewall firing: an EVENT, no point, and counted as skipped.
        r = parse("step 10 ce=NaN bpb=8.0 best=8.0 lr=1e-4\n")
        self.assertIsNone(r.steps[0].ce)            # unplottable
        self.assertEqual(r.steps[0].bpb, 8.0)      # its one good number survives
        self.assertEqual([e.kind for e in r.events], ["nan"])
        self.assertIn("ce=NaN", r.events[0].text)
        self.assertEqual(r.skipped, 0)             # not corruption: it was recorded

        r2 = parse("step 20 ce=NaN bpb=inf best=nan lr=nan\n")
        self.assertEqual(r2.steps, [])
        self.assertEqual([e.kind for e in r2.events], ["nan"])
        self.assertEqual(r2.skipped, 1)             # dropped from the plot, and it says so
        self.assertIn("bpb=inf", r2.events[0].text)


class CorruptIsCountedNotFatal(unittest.TestCase):
    TEXT = (REAL500
            + "step    400 ce=3.1\n"                 # truncated: one good field, rest absent
            + "step    450 ce=abc bpb=4.0\n"         # present-but-unparsable value
            + "timer step 450: total=abc\n"          # unparsable timer
            + "step    500 ce=2.9 bpb=4.1 best=2.9 lr=1e-4 aux=0.02\n")

    def setUp(self):
        self.r = parse(self.TEXT)

    def test_skipped_counted(self):
        self.assertEqual(self.r.skipped, 2)
        self.assertIn("step    450 ce=abc", self.r.skip_examples[0])
        self.assertIn("timer step 450", self.r.skip_examples[1])

    def test_truncated_line_keeps_its_one_number(self):
        # 400 has a real ce and nothing else: kept, with gaps. Dropping it would be
        # losing data; plotting bpb=0 would be a lie.
        s = [p for p in self.r.steps if p.step == 400][0]
        self.assertEqual(s.ce, 3.1)
        self.assertIsNone(s.bpb)
        self.assertIsNone(s.aux)

    def test_good_lines_around_the_damage_survive(self):
        self.assertEqual([p.step for p in self.r.steps], [0, 100, 200, 300, 400, 500])
        self.assertEqual(self.r.steps[-1].ce, 2.9)
        self.assertEqual([t.step for t in self.r.timers], [0, 100, 200, 300])


class GapsAreNotZeros(unittest.TestCase):
    def test_absent_aux_is_none(self):
        r = parse("step 10 ce=3.0 bpb=4.0 best=3.0 lr=1e-4\n")  # old-format line, no aux
        self.assertIsNone(r.steps[0].aux)
        xs, ys = r.col(r.steps, "aux")
        self.assertEqual((xs, ys), ([], []))

    def test_a_gap_in_one_field_does_not_shift_another(self):
        """col() drops gaps, so CE and BPB get different lengths on a partial line.

        Plotting one field's ys against another's xs either crashes (different
        lengths) or silently mis-times the curve (same length, wrong x). Each
        field must be plotted against its own xs.
        """
        # REAL500 has 4 step lines (0/100/200/300); the truncated one makes 5.
        r = parse(REAL500 + "step    500 ce=2.9\n")   # truncated: bpb/best/aux absent
        xc, ce = r.col(r.steps, "ce")
        xb, bpb = r.col(r.steps, "bpb")
        self.assertEqual(len(xc), 5)
        self.assertEqual(len(xb), 4)
        self.assertEqual(xc[-1], 500)      # ce kept its point
        self.assertEqual(xb[-1], 300)      # bpb stopped at the last full line
        self.assertNotEqual(xc, xb)

    def test_board_renders_with_a_gap(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            run = parse(REAL500 + "step    500 ce=2.9\n")
            for f in (tb.chart_loss(run, out), tb.chart_eval(run, out),
                      tb.chart_steptime(run, out)):
                self.assertTrue(f.stat().st_size > 5000, f)

    def test_eval_without_window_keeps_the_number(self):
        r = parse("step 10 EVAL ce=4.0 bpb=6.0 BEST\n")
        self.assertIsNone(r.evals[0].window)
        self.assertEqual(r.evals[0].bpb, 6.0)


class RestartedLogHasSessions(unittest.TestCase):
    """official_v5e.log holds 2 run headers; official_v5g.log holds 17.

    `step` resets on a guard restart, so the points are NOT one monotonic series —
    joining them draws a ramp from step 1500 through step 1000. Sessions must split.
    """

    TEXT = ('''dormouse pretrain small params=7526223 steps=100000 data="/mnt/x" lr=0.0003 backend=cuda(autodiff)
timer step 1000: total=70074ms data=0.1ms fwd=28001ms bwd=38975ms opt=2627ms retr=464.5ms ema=6.3ms gpu_step=70074ms
step   1000 ce=3.637 bpb=5.247 best=3.637 lr=1.50e-4 aux=0.2114
step   1500 ce=3.100 bpb=4.600 best=3.100 lr=1.50e-4 aux=0.2010
guard: restarting in 30s (resume from last checkpoint)
dormouse pretrain small params=7526223 steps=100000 data="/mnt/x" lr=0.0003 backend=cuda(autodiff)
timer step 1000: total=1676ms data=0.2ms fwd=679ms bwd=861ms opt=59ms retr=72.4ms ema=3.2ms gpu_step=1676ms
step   1000 ce=4.900 bpb=6.100 best=4.900 lr=1.50e-4 aux=0.3000
step   1500 ce=4.400 bpb=5.600 best=4.400 lr=1.50e-4 aux=0.2900
done steps=2000 best ce=3.100
''')

    def setUp(self):
        self.r = parse(self.TEXT)

    def test_two_sessions(self):
        self.assertEqual(self.r.headers, 2)
        self.assertEqual([p.sess for p in self.r.steps], [0, 0, 1, 1])
        self.assertEqual([t.sess for t in self.r.timers], [0, 1])

    def test_sessions_split_the_series(self):
        self.assertEqual([(s, [p.step for p in ps])
                          for s, ps in tb._sessions(self.r.steps)],
                         [(0, [1000, 1500]), (1, [1000, 1500])])

    def test_restart_is_an_event_on_both_charts(self):
        self.assertEqual([e.kind for e in self.r.events], ["restart"])
        self.assertEqual(self.r.events[0].step, 1500)

    def test_best_is_over_all_sessions_not_the_last(self):
        ces = [p.ce for p in self.r.steps]
        self.assertEqual(min(ces), 3.1)          # session 1, not session 2's 4.4
        self.assertEqual(ces.index(3.1), 1)

    def test_board_renders_a_restarted_log(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            for f in (tb.chart_loss(self.r, out), tb.chart_eval(self.r, out),
                      tb.chart_steptime(self.r, out),
                      tb.chart_throughput(self.r, out, 10, 512)):
                self.assertTrue(f.stat().st_size > 5000, f)


class OlderShape(unittest.TestCase):
    """Pre-2026-09-30 shape: no `aux=`, no seam counters, 20480 B window, no timers."""

    TEXT = '''dormouse pretrain small params=9195854 steps=20000 data="/mnt/x/real_sharded" lr=0.0001 backend=cuda(autodiff)
step      0 ce=5.547 bpb=8.003 best=5.547 lr=0.00e0
step    500 EVAL ce=4.517 bpb=6.517 over 20480 B (fixed window) fused kda=0/0 norm=0/1569 muon_skipped=0/0
guard: restarting in 30s (resume from last checkpoint)
guard: giving up (resumable=false, restarts=0) - a persistent failure needs a human
done steps=20000 best ce=2.619
'''

    def setUp(self):
        self.r = parse(self.TEXT)

    def test_parses(self):
        self.assertEqual(self.r.params, 9195854)
        self.assertEqual(len(self.r.steps), 1)
        self.assertEqual(self.r.evals[0].window, 20480)
        self.assertFalse(self.r.evals[0].is_best)
        self.assertEqual([e.kind for e in self.r.events], ["restart", "fail"])
        self.assertEqual(self.r.done, "steps=20000 best ce=2.619")
        self.assertEqual(self.r.skipped, 0)


class DegenerateLogIsNamed(unittest.TestCase):
    """pretrain_v2.log is 57 lines: two sessions, one step line each, 3 guard
    restarts. It must render, say it never trained, and not stack 3 labels on
    the same x pixel."""

    TEXT = ('dormouse pretrain small params=7526991 steps=200000 data="/x" lr=1e-4 backend=cuda(autodiff)\n'
            'timer step 0: total=10583ms data=0.2ms fwd=3027ms bwd=3618ms opt=3034ms retr=786.0ms ema=106.8ms gpu_step=10583ms\n'
            'step      0 ce=5.360 bpb=7.732 best=5.360 lr=0.00e0 aux=0.1637\n'
            'guard: restarting in 30s (resume from last checkpoint)\n'
            'guard: restarting in 30s (resume from last checkpoint)\n'
            'guard: restarting in 30s (resume from last checkpoint)\n'
            'dormouse pretrain small params=7526991 steps=200000 data="/x" lr=1e-4 backend=cuda(autodiff)\n'
            'step      0 ce=5.472 bpb=7.895 best=5.472 lr=0.00e0 aux=0.1606\n'
            'guard: no resumable checkpoint, giving up\n')

    def setUp(self):
        self.r = parse(self.TEXT)

    def test_parse(self):
        self.assertEqual(len(self.r.steps), 2)
        self.assertEqual({s.step for s in self.r.steps}, {0})
        self.assertEqual([e.kind for e in self.r.events], ["restart", "restart", "restart", "fail"])
        self.assertEqual(self.r.skipped, 0)

    def test_events_at_one_step_collapse_to_one_mark(self):
        """3 guard + 1 give-up all at step 0: one vline, one label, count in it."""
        with TemporaryDirectory() as d:
            out = Path(d)
            plt = tb._mpl()
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 2])
            n_before = len(ax.lines)
            tb._events(ax, self.r)
            self.assertEqual(len(ax.lines) - n_before, 1)   # one vline, not four
            self.assertEqual(ax.texts[0].get_text(), " restart+fail ×4@0")
            plt.close(fig)

    def test_board_renders(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            for f in (tb.chart_loss(self.r, out), tb.chart_eval(self.r, out),
                      tb.chart_steptime(self.r, out),
                      tb.chart_throughput(self.r, out)):
                self.assertTrue(f.stat().st_size > 5000, f)
            s = tb.write_index([self.r], out, {}, 10, 512).read_text()
            self.assertIn("restart", s)


class ChartsRender(unittest.TestCase):
    """The whole board on a real-shaped log, plus the empty case. No exceptions."""

    def test_full_board(self):
        plt = tb._mpl()
        with TemporaryDirectory() as d:
            out = Path(d)
            run = parse(REAL500 + REAL_EVAL + REAL_EVENTS)
            tsv = out / "w.tsv"
            tsv.write_text("wall_s\twall_iso\tpid\tevery_s\tgpu_util_pct\tvram_mib\tram_used_gib\tram_avail_gib\terrors\n"
                           "0.0\tx\t1\t5\t13\t900\t20.5\t43.5\t0\n"
                           "5.0\tx\t1\t5\t4\t1500\t21.0\t43.0\t0\n"
                           "10.0\tx\t1\t5\tnan\tnan\t21.2\t42.8\t1\n"
                           "15.0\tx\t1\t5\t88\t2000\t22.0\t42.0\t1\n")
            for f in (tb.chart_loss(run, out), tb.chart_eval(run, out),
                      tb.chart_steptime(run, out), tb.chart_throughput(run, out, 10, 512),
                      tb.chart_throughput(run, out), tb.chart_resources(tsv, out)):
                self.assertTrue(f.exists() and f.stat().st_size > 5000, f)
        _ = plt

    def test_empty_log_does_not_crash(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            run = parse("warmup done\nnothing here\n")
            for f in (tb.chart_loss(run, out), tb.chart_eval(run, out),
                      tb.chart_steptime(run, out), tb.chart_throughput(run, out, 10, 512)):
                self.assertTrue(f.exists())

    def test_index_html_carries_the_window(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            runs = [parse(REAL500 + REAL_EVAL)]
            p = tb.write_index(runs, out, {runs[0].path.stem: []}, 10, 512)
            s = p.read_text()
            self.assertIn("81,920 B", s)   # the window is IN the page
            self.assertIn("4.449", s)
            self.assertIn("3.188", s)
            self.assertIn("9,197,454", s)
            self.assertIn("kda=2012/0", s)  # seam counters survive to the page
            self.assertIn("<th>run</th>", s)  # session column, see RestartedLogHasSessions

    def test_index_tables_are_keyed_by_session_too(self):
        """A restarted log has two `step 1000` rows with different numbers.

        A table keyed on step alone is ambiguous exactly where it matters, so
        every point row carries its session and tools/spot_check.py keys on it.
        """
        restarted = parse(REAL500
                       + 'dormouse pretrain small params=9197454 steps=500 data="/mnt/x/real" lr=0.0001 backend=cuda\n'
                       + "step   1500 ce=2.1 bpb=3.0 best=2.1 lr=1.00e-4 aux=0.02\n")
        self.assertEqual(restarted.headers, 2)
        with TemporaryDirectory() as d:
            out = Path(d)
            s = tb.write_index([restarted], out, {}, 10, 512).read_text()
        body = s.split("<h3>step lines")[1].split("</table>")[0]
        rows = [[re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td>(.*?)</td>", tr, re.S)]
                for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S)]
        rows = [r for r in rows if r]           # the <th> header row has no <td>
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "1")          # first session
        self.assertEqual(rows[-1][0], "2")         # after the restart
        self.assertEqual(rows[-1][1], "1500")
        self.assertEqual(rows[0][1], "0")


class WatchFailsSoft(unittest.TestCase):
    def test_nvidia_hiccup_does_not_kill_the_sampler(self):
        """Every tick's nvidia-smi raises. The sampler must still write rows."""
        orig, calls = tb._nvidia, []

        def boom(gpu):
            calls.append(gpu)
            raise RuntimeError("simulated nvidia-smi hiccup")

        tb._nvidia = boom
        try:
            with TemporaryDirectory() as d:
                log = Path(d) / "run.log"
                log.write_text(REAL500)
                tsv = tb.sample_watch(os_getpid(), log, every=0.05, seconds=0.4, quiet=True)
                lines = tsv.read_text().strip().splitlines()
        finally:
            tb._nvidia = orig
        self.assertGreaterEqual(len(lines), 3)          # header + samples
        self.assertEqual(len(calls), len(lines) - 1)   # it retried, once per tick
        for row in lines[1:]:
            f = row.split("\t")
            self.assertEqual(f[4], "nan")              # util NaN, not a fake 0
            self.assertGreater(float(f[8]), 0)          # errors counted, kept counting

    def test_meminfo_is_read_not_guessed(self):
        used, avail = tb._meminfo()
        self.assertIsNotNone(avail)
        self.assertGreater(avail, 0.0)
        self.assertLess(used, 200.0)

    def test_pid_liveness(self):
        self.assertTrue(tb._pid_alive(os_getpid()))
        self.assertFalse(tb._pid_alive(999999999))


def os_getpid():
    import os
    return os.getpid()


if __name__ == "__main__":
    unittest.main(verbosity=2)
