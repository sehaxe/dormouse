#!/usr/bin/env python3
"""Spot-check: do the numbers trainboard plotted match `grep` on the same log?

This is the verification bar from the task, not a unit test — it reads the real
logs in ~/logs, so on a box without them it says so and exits 0. It compares
ROW BY ROW: parse the eval / step / timer tables out of index.html (rendered from
the same run model the charts plot), then grep the raw log for that step's line
and compare the field. A substring check against the whole page proves nothing —
"5.450" is in the page whether it is attached to step 19500 or step 500.

Run: python3 tools/spot_check.py [--out DIR] [--logs DIR]   (exit 1 on mismatch)
"""
import argparse
import re
import sys
from pathlib import Path


def _rows(section, ncols, after):
    """Table rows of width `ncols` that follow the `after` heading."""
    body = section.split(after, 1)[-1].split("</table>")[0]
    out = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        cells = re.findall(r"<td>(.*?)</td>", tr, re.S)
        if len(cells) == ncols:
            out.append([re.sub(r"\s+", " ", c).strip() for c in cells])
    return out


def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/opencode/final"))
    ap.add_argument("--logs", type=Path, default=Path.home() / "logs")
    ap.add_argument("--max-per", type=int, default=3, help="rows to check per run")
    a = ap.parse_args()
    index = a.out / "index.html"
    if not index.exists():
        print(f"spot_check: no {index} — generate the board first", file=sys.stderr)
        return 2
    txt = index.read_text()

    ok = bad = 0

    def _num(s):
        """'20,480 B' / '509.0' / '5.571' -> float, or None. The board formats for
        humans (thousands separators, fixed decimals); grep does not."""
        m = re.search(r"-?[\d,]*\.?\d+", s or "")
        return float(m.group(0).replace(",", "")) if m else None

    def check(run, topic, label, page, grep, occ=-1):
        """occ indexes which match of `grep` in the log to compare against.

        A restarted log holds the same `step N` twice (two sessions), so the
        caller passes the session's ordinal. Comparing against `occ=-1` (the last
        match) when the board row is session 1 is how this checker produced 44
        false MISSes on correct boards.
        """
        nonlocal ok, bad
        log = a.logs / run
        if not log.exists():
            return None
        m = re.findall(grep, log.read_text(errors="replace"))
        if not m:
            print(f"MISS {run} /{label}/ — grep found nothing", file=sys.stderr)
            bad += 1
            return None
        if occ >= len(m):
            occ = 0
        a_, b_ = _num(page), _num(m[occ])
        hit = a_ is not None and b_ is not None and abs(a_ - b_) < 5e-4
        bad += not hit
        ok += hit
        print(f"{'OK  ' if hit else 'MISS'} {topic:5} {run[:32]:34} {label:<24} "
              f"grep={m[occ]:<12} board={page:<14}")

    for key in re.findall(r"<h2>([^<]+\.log)</h2>", txt):
        sec = txt.split(f"<h2>{key}</h2>", 1)[1].split("</section>")[0]
        # ---- eval table: run | step | ce | bpb | window | BEST | seam
        ev = _rows(sec, 7, "<h3>eval points")
        for row in ev[:a.max_per] + ev[-1:]:
            sess, step, bpb, win = int(row[0]), row[1], row[3], row[4]
            check(key, "eval", f"EVAL r{sess} {step} bpb", bpb,
                  rf"step\s+{step}\s+EVAL\s+ce=\S+\s+bpb=(\S+)", occ=sess - 1)
            # An old-format EVAL line carries no `over N B` at all
            # (ab8m_ab8m_iter4.log). Board "—" and log-absent is a MATCH: the
            # window is a gap, not a fabricated number (AGENTS.md §2.6).
            if win == "—":
                hit = not re.search(rf"step\s+{step}\s+EVAL.*?over\s+[\d,]+\s*B",
                                    (a.logs / key).read_text(errors="replace"))
                bad += not hit
                ok += hit
                print(f"{'OK  ' if hit else 'MISS'} eval  {key[:32]:34} "
                      f"{f'EVAL r{sess} {step} window':<24} grep=<absent>     board=—")
            else:
                check(key, "eval", f"EVAL r{sess} {step} window", win,
                      rf"step\s+{step}\s+EVAL.*?over\s+([\d,]+)\s*B", occ=sess - 1)
        # ---- step table: run | step | ce | bpb | best | lr | aux
        st = _rows(sec, 7, "<h3>step lines")
        for row in st[:a.max_per] + st[-1:]:
            sess, step = int(row[0]), row[1]
            check(key, "loss", f"step r{sess} {step} ce", row[2],
                  rf"step\s+{step}\s+ce=(\S+)", occ=sess - 1)
            check(key, "loss", f"step r{sess} {step} bpb", row[3],
                  rf"step\s+{step}\s+ce=\S+\s+bpb=(\S+)", occ=sess - 1)
        # ---- timer table: run | step | total | fwd | bwd | opt | retr | ema | gpu_step
        tm = _rows(sec, 9, "timer lines of")
        for row in tm[:1] + tm[-1:]:
            sess, step = int(row[0]), row[1]
            check(key, "time", f"timer r{sess} {step} total", row[2],
                  rf"timer step {step}: total=(\S+)ms", occ=sess - 1)
            check(key, "time", f"timer r{sess} {step} fwd", row[3],
                  rf"timer step {step}:.*?fwd=(\S+)ms", occ=sess - 1)
            check(key, "time", f"timer r{sess} {step} bwd", row[4],
                  rf"timer step {step}:.*?bwd=(\S+)ms", occ=sess - 1)
    print(f"\nspot_check: {ok} matched, {bad} missed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
