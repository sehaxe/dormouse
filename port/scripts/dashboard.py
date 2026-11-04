"""Live read-only dashboard for the dormouse training run.

Zero dependencies: stdlib HTTP server + SSE, and htmx for the swap. It only reads
(files + `nvidia-smi` + `git log`); nothing in the training run is touched.

    python scripts/dashboard.py --port 3789 --host 0.0.0.0

htmx is served from `dashboard/static/htmx.min.js`; if that file is missing the
page still works (server-rendered, refreshed by `<meta http-equiv=refresh>`),
because a dashboard that dies with one asset is a bad dashboard.

Routes
    /                 the page
    /events           SSE: `metrics`, `hero`, `chart`, `panels`
    /frag/<name>      one fragment (first paint, and a curl-friendly probe)
    /api/snapshot     the whole snapshot as JSON
    /healthz          200 + a one-line status
"""

# The page and the fragments are HTML inside f-strings; wrapping them to 100 columns
# makes the markup harder to read than a marked exception.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # port/
REPO = ROOT.parent                                   # dormouse/
LOGS = Path(os.environ.get("DASH_LOGS", Path.home() / "logs"))


def _active_run() -> str:
    """The run to show: `DASH_RUN`, else the newest `<name>.jsonl` that has rows.

    Hard-coding `flagship` meant the board went stale the moment the next stair
    started (2026-11-04: the 200M run was live and the board still read 1.72).
    Ties break on mtime, so a resumed run wins over its predecessor's file.
    """
    explicit = os.environ.get("DASH_RUN")
    if explicit:
        return explicit
    best, best_mtime = "flagship", -1.0
    for f in LOGS.glob("*.jsonl"):
        try:
            m = f.stat().st_mtime
        except OSError:
            continue
        if m > best_mtime:
            best, best_mtime = f.stem, m
    return best


RUN = _active_run()
JSONL = LOGS / f"{RUN}.jsonl"
TRAIN_LOG = LOGS / f"{RUN}.log"
STATUS_LOG = LOGS / f"{RUN}_status.log"
CKPT = REPO / "checkpoints" / RUN
SFT_MANIFEST = ROOT / "sft_pairs" / "v2" / "full" / "manifest.json"
STATIC = ROOT / "dashboard" / "static"
# The LAST routing histogram (`scripts/route_hist.py --json PATH`). The panel
# prefers the live counters and falls back to this file, so a dead run still
# shows what its bank looked like without a second model load.
ROUTING_JSON = Path(os.environ.get("DASH_ROUTE_JSON", LOGS / f"{RUN}_route.json"))

REFRESH_S = 2.0
CACHE_S = 1.4


# ------------------------------------------------------------------ helpers


def _tail(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [ln for ln in data.splitlines() if ln.strip()][-n:]


def _jsonl(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _human_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def _human_secs(s: float) -> str:
    if s != s or s <= 0:
        return "—"
    if s > 86400 * 30:
        return f"{s / 86400:.0f} d"
    h, rem = divmod(int(s), 3600)
    m = rem // 60
    return f"{h} h {m:02d} m" if h else f"{m} m"


def _gpu() -> dict:
    q = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,"
        "clocks.sm,temperature.gpu,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(q, capture_output=True, text=True, timeout=2.5).stdout.strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        util, used, total, pw, plim, clk, temp, name = parts[:8]
        return {
            "ok": True,
            "util": float(util),
            "vram_used": float(used),
            "vram_total": float(total),
            "power": float(pw),
            "power_limit": float(plim),
            "clock": float(clk),
            "temp": float(temp),
            "name": name,
        }
    except Exception:
        return {"ok": False}


def _git() -> list[dict]:
    try:
        out = subprocess.run(
            ["git", "log", "-6", "--pretty=%h%x1f%cr%x1f%s"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=2.5,
        ).stdout.strip()
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        bits = line.split("\x1f")
        if len(bits) == 3:
            rows.append({"sha": bits[0], "when": bits[1], "subject": bits[2]})
    return rows


def _number(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_number(d: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d:
            v = _number(d[k])
            if v is not None:
                return v
    return None


def _share(d: dict, keys: tuple[str, ...]) -> float | None:
    """A share as 0..1, from an `_x100`/`_pct` counter or a fraction."""
    for k in keys:
        if k not in d:
            continue
        v = _number(d[k])
        if v is None:
            continue
        if k.endswith("_x100") or k.endswith("_pct"):
            return v / 100.0
        return v / 100.0 if v > 1.0 else v
    return None


def routing_data(s: dict) -> dict:
    """Routing health from the run's counters, else the LAST route_hist JSON.

    Counter names are the sibling MoE lane's (`moe_live_experts`,
    `moe_top10_share_x100`, `moe_max_share_x100`); the file carries the same
    fields from `scripts/route_hist.py --json`. Missing data renders an
    instruction, never an exception.
    """
    counters = s.get("counters") or {}
    run = s.get("run") or {}
    live = _first_number(
        counters, ("moe_live_experts", "live_experts", "moe_experts_live")
    )
    top10 = _share(
        counters,
        ("moe_top10_share_x100", "moe_top10_share_pct", "moe_top10_share",
         "top10_share_x100", "top10_share"),
    )
    max_share = _share(
        counters,
        ("moe_max_share_x100", "moe_max_share_pct", "moe_max_share",
         "max_share_x100", "max_share"),
    )
    experts = _first_number(counters, ("moe_n_experts", "n_experts", "moe_experts"))
    source = "counters" if live is not None else ""
    if live is None:
        try:
            hist = json.loads(ROUTING_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            hist = {}
        if hist:
            source = str(ROUTING_JSON)
            live = _first_number(hist, ("live_experts", "moe_live_experts"))
            top10 = top10 if top10 is not None else _share(
                hist, ("top10_share_x100", "top10_share")
            )
            max_share = max_share if max_share is not None else _share(
                hist, ("max_share_x100", "max_share")
            )
            experts = _first_number(hist, ("n_experts", "moe_n_experts")) or experts
            counts = hist.get("counts")
            if live is None and counts:
                n = int(experts or len(counts))
                live = float(sum(1 for v in counts[:n] if v))
    if experts is None:
        experts = _number(run.get("experts"))
    e = int(experts) if experts else None
    live_i = int(live) if live is not None else None
    frac = live_i / e if (live_i is not None and e) else None
    if frac is None:
        state = "missing"
    elif frac < 0.5:
        state = "bad"
    elif frac < 0.8:
        state = "warn"
    else:
        state = "ok"
    return {
        "live": live_i,
        "experts": e,
        "frac": frac,
        "top10": top10,
        "max_share": max_share,
        "state": state,
        "source": source,
    }


# ------------------------------------------------------------------ snapshot


class Cache:  # not a dataclass: the module is also loaded via importlib in tests
    at = 0.0
    value = None
    git_at = 0.0
    git = None


CACHE = Cache()


def snapshot() -> dict:
    now = time.time()
    if CACHE.value is not None and now - CACHE.at < CACHE_S:
        return CACHE.value

    rows = _jsonl(JSONL)
    logs = [r for r in rows if r.get("event") == "log"]
    evals = [r for r in rows if r.get("event") == "eval"]
    last = logs[-1] if logs else {}
    # Every number about the run comes from the run's OWN config snapshot: a
    # hard-coded 320k / 24 576 bytes-per-step reported the flagship's shape for
    # whatever was actually training (caught when the 200M stair started).
    run_cfg = _run_config()
    step = int(last.get("step") or (evals[-1]["step"] if evals else 0))
    total_steps = int(run_cfg.get("steps", 320_000))
    bpb = evals[-1].get("bpb") if evals else None
    prev_bpb = evals[-2].get("bpb") if len(evals) > 1 else None
    tps = float(last.get("tokens_per_s") or 0.0)
    bytes_per_step = (
        int(run_cfg.get("batch", 8)) * int(run_cfg.get("seq_len", 1024))
        * int(run_cfg.get("grad_accum", 1))
    )
    seen = step * bytes_per_step
    budget = total_steps * bytes_per_step
    eta = (budget - seen) / tps if tps > 0 else float("nan")

    # staleness: the trainer writes a log line every log_every steps
    mtime = JSONL.stat().st_mtime if JSONL.exists() else 0
    age = now - mtime if mtime else float("inf")
    state = "live" if age < 180 else ("stale" if age < 1800 else "dead")
    exited = ""
    try:
        exited = STATUS_LOG.read_text(encoding="utf-8").strip().splitlines()[-1]
    except (OSError, IndexError):
        pass

    ckpts = []
    if CKPT.exists():
        for p in sorted(CKPT.glob("*.pt"), key=lambda q: q.stat().st_mtime, reverse=True)[:5]:
            st = p.stat()
            ckpts.append(
                {"name": p.name, "mb": st.st_size / 1e6, "age": now - st.st_mtime}
            )

    sft = {}
    try:
        man = json.loads(SFT_MANIFEST.read_text(encoding="utf-8"))
        sft = {
            "rows": man["totals"]["train_rows"],
            "holdout": man["totals"]["holdout_rows"],
            "clean": man["totals"]["clean_rows"],
            "bytes": man["totals"]["bytes"],
            "mix": man.get("mix", {}),
            "sources": {
                k: {"rows": v["rows"], "risk": v.get("license_risk"), "lic": v.get("license")}
                for k, v in man["sources"].items()
            },
        }
    except (OSError, KeyError, json.JSONDecodeError):
        sft = {}

    if CACHE.git is None or now - CACHE.git_at > 20:
        CACHE.git = _git()
        CACHE.git_at = now

    counters = last.get("counters") or {}
    value = {
        "now": now,
        "state": state,
        "age": age,
        "exited": exited,
        "run": {
            "name": RUN,
            "shape": (
                f"d{run_cfg.get('d_model','?')} · b{run_cfg.get('batch','?')}"
                f"×s{run_cfg.get('seq_len','?')}×accum{run_cfg.get('grad_accum',1)}"
                f" · it{run_cfg.get('max_iter','?')}"
            ),
            "params": run_cfg.get("params"),
            "experts": run_cfg.get("n_experts"),
            "step": step,
            "total_steps": total_steps,
            "decay_at": int(total_steps * (1.0 - float(run_cfg.get("decay_frac", 0.15)))),
            "bpb": bpb,
            "bpb_prev": prev_bpb,
            "loss": last.get("loss"),
            "lr": last.get("lr"),
            "tps": tps,
            "seen": seen,
            "budget": budget,
            "eta": eta,
            "nan": last.get("nan_skipped", 0),
        },
        "stair": _stair(evals, run_cfg),
        "agent": _agent_eval(),
        "history": [
            {"step": e["step"], "bpb": e.get("bpb"), "loss": e.get("loss")}
            for e in evals[-120:]
            if e.get("bpb") is not None
        ],
        "projection": _projection(evals, total_steps),
        "counters": counters,
        "gpu": _gpu(),
        "ckpts": ckpts,
        "sft": sft,
        "git": CACHE.git or [],
        "tail": _tail(TRAIN_LOG, 9),
    }
    value["routing"] = routing_data(value)
    CACHE.value, CACHE.at = value, now
    return value


def _projection(evals: list[dict], total_steps: int) -> dict:
    """Linear fit of the last 6 evals in log-step space, projected to the end.

    Cheap and honest: it is the current slope, not a promise. The decay phase is
    not modelled, so the number is a floor-ish estimate and is labelled as such.
    """
    pts = [(e["step"], e["bpb"]) for e in evals[-6:] if e.get("bpb") is not None]
    if len(pts) < 3:
        return {}
    import math

    xs = [math.log(max(1, p[0])) for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-12:
        return {}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
    end = my + slope * (math.log(total_steps) - mx)
    return {"bpb": end, "slope": slope}


# ------------------------------------------------------------------ fragments


def _fmt(v, spec="{:.4f}", dash="—"):
    return dash if v is None else spec.format(v)


def frag_metrics(s: dict) -> str:
    r = s["run"]
    d = None if (r["bpb"] is None or r["bpb_prev"] is None) else r["bpb"] - r["bpb_prev"]
    arrow = "—" if d is None else f"{d:+.4f}"
    pct = 100.0 * r["step"] / max(1, r["total_steps"])
    pr = s.get("projection") or {}
    proj = (
        f"projection to {r['total_steps'] // 1000}k (current slope): {pr['bpb']:.3f}"
        if pr
        else "projection: needs 3 eval points"
    )
    return f"""<div class="nums">
  <div class="cell hero"><span class="k">held-out bpb</span>
    <div class="v">{_fmt(r['bpb'])}</div>
    <span class="sub">{arrow} vs previous eval · ×{8 / (r['bpb'] or 8):.1f} compression</span>
    <span class="sub">{proj}</span></div>
  <div class="cell"><span class="k">loss</span><div class="v">{_fmt(r['loss'], '{:.4f}')}</div>
    <span class="sub">lr {_fmt(r['lr'], '{:.1e}')}</span></div>
  <div class="cell"><span class="k">bytes / s</span><div class="v">{r['tps'] / 1000:.1f}<span style="font-size:19px">k</span></div>
    <span class="sub">{_human_bytes(r['seen'])} seen</span></div>
  <div class="cell"><span class="k">eta</span><div class="v" style="font-size:26px">{_human_secs(r['eta'])}</div>
    <span class="sub">{pct:.1f}% of {r['total_steps']:,}</span></div>
</div>"""


def frag_hero(s: dict) -> str:
    run = s.get("run", RUN)
    lines = s["tail"] or ["(no log yet)"]
    out = []
    for i, ln in enumerate(lines):
        cls = "ln key" if i >= len(lines) - 2 else "ln"
        if len(ln) > 150:
            ln = ln[:147] + "…"
        safe = html.escape(ln)
        for token, cls2 in ((" bpb=", "ok"), (" loss=", ""), ("nan_skipped=0", "ok")):
            if token in safe and cls2:
                safe = safe.replace(token, f'<span class="{cls2}">{token}</span>', 1)
        if i == len(lines) - 1:
            safe += '<span class="caret"></span>'
        out.append(f'<span class="{cls}">{safe}</span>')
    age = s["age"]
    return f"""<div class="term">
  <div class="bar-top"><span>{run} · train</span>
    <span>{'log ' + _human_secs(age) + ' old' if age < 1e9 else 'no log'}</span></div>
  <pre>{''.join(out)}</pre>
</div>"""


def frag_chart(s: dict) -> str:
    h = s["history"]
    if len(h) < 2:
        return '<div class="void">[–] waiting for the second eval point</div>'
    W, H, PADL, PADB, PADT = 1000.0, 150.0, 44.0, 22.0, 12.0
    xs = [p["step"] for p in h]
    ys = [p["bpb"] for p in h]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = max(0.004, (y1 - y0) * 0.18)
    y0, y1 = y0 - pad, y1 + pad
    def sx(x: float) -> float:
        return PADL + (x - x0) / max(1, (x1 - x0)) * (W - PADL - 8)

    def sy(y: float) -> float:
        return PADT + (1 - (y - y0) / max(1e-9, (y1 - y0))) * (H - PADT - PADB)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=True))
    area = f"{PADL:.1f},{H - PADB:.1f} {pts} {sx(x1):.1f},{H - PADB:.1f}"
    grid, labels = [], []
    for i in range(4):
        y = y0 + (y1 - y0) * i / 3
        py = sy(y)
        grid.append(f'<line x1="{PADL}" y1="{py:.1f}" x2="{W - 8}" y2="{py:.1f}"/>')
        labels.append(
            f'<text class="axis" x="{PADL - 8}" y="{py + 3.5:.1f}" text-anchor="end">{y:.3f}</text>'
        )
    for i in range(4):
        x = x0 + (x1 - x0) * i / 3
        labels.append(
            f'<text class="axis" x="{sx(x):.1f}" y="{H - 5}" text-anchor="middle">{x / 1000:.0f}k</text>'
        )
    dots = "".join(
        f'<circle class="pt" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4"/>' for x, y in zip(xs, ys, strict=True)
    )
    # the 105M reference on the same token axis (GB), when it exists
    st = s.get("stair") or {}
    ref_pts = ""
    if st.get("ref") and st.get("cur"):
        allx = [p[0] for p in st["ref"] + st["cur"]]
        x0, x1 = min(allx), max(allx)
        ry = [p[1] for p in st["ref"]]
        ys_all = ys + ry
        y0, y1 = min(ys_all), max(ys_all)
        pad = max(0.004, (y1 - y0) * 0.18)
        y0, y1 = y0 - pad, y1 + pad

        def sx2(x: float) -> float:
            return PADL + (x - x0) / max(1e-9, (x1 - x0)) * (W - PADL - 8)

        def sy2(y: float) -> float:
            return PADT + (1 - (y - y0) / max(1e-9, (y1 - y0))) * (H - PADT - PADB)

        ref_pts = " ".join(f"{sx2(x):.1f},{sy2(y):.1f}" for x, y in st["ref"])

    last = h[-1]
    return f"""<div class="chart">
<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img"
     aria-label="held-out BPB over training steps">
  <g class="grid">{''.join(grid)}</g>
  <polygon class="area" points="{area}"/>
  <polyline class="line" points="{pts}"/>
  {f'<polyline class="line ref" points="{ref_pts}"/>' if ref_pts else ""}
  {dots}
  <circle class="pt last" cx="{sx(last['step']):.1f}" cy="{sy(last['bpb']):.1f}" r="3.6"/>
  <text class="lbl" x="{min(W - 12, sx(last['step']) + 9):.1f}"
        y="{sy(last['bpb']) - 8:.1f}">{last['bpb']:.4f}</text>
  {''.join(labels)}
</svg></div>"""


def frag_routing(s: dict) -> str:
    r = s.get("routing") or routing_data(s)
    if r["live"] is None or not r["experts"]:
        cmd = f"scripts/route_hist.py --json {ROUTING_JSON}"
        return f'<div class="void">[–] no routing health yet — <code>{html.escape(cmd)}</code></div>'
    pct = 100.0 * r["frac"]
    label = {"ok": "healthy", "warn": "degraded", "bad": "collapsed"}[r["state"]]
    top10 = "—" if r["top10"] is None else f"{100 * r['top10']:.1f}%"
    mx = "—" if r["max_share"] is None else f"{100 * r['max_share']:.1f}%"
    source = (
        f' · source {html.escape(r["source"])}' if r.get("source") else ""
    )
    return f"""<div class="routing route-{r['state']}">
  <div class="gauge">
    <div class="row"><span class="lab">live</span>
      <div class="track"><div class="mk {r['state']}" style="width:{pct:.1f}%"></div></div>
      <span class="val">{r['live']} / {r['experts']}</span></div>
  </div>
  <div class="kv">
    <div class="k">live experts</div><div class="v">{r['live']} / {r['experts']} · {pct:.1f}%</div>
    <div class="k">top-10 share</div><div class="v">{top10}</div>
    <div class="k">max share</div><div class="v">{mx}</div>
  </div>
  <div class="route-state {r['state']}"><b>{label}</b> · live &lt; 50% of E is the red line{source}</div>
</div>"""


def frag_panels(s: dict) -> str:
    c = s["counters"]
    g = s["gpu"]

    def mech(name, key, note=""):
        v = c.get(key)
        val = f"{v:,}" if isinstance(v, int) else "—"
        cls = "v" if isinstance(v, int) and v > 0 else "v off"
        return f'<div class="k"><b>[+]</b> {name}{note}</div><div class="{cls}">{val}</div>'

    mechanisms = (
        mech("csa fine", "csa_forward")
        + mech("csa coarse", "csa_coarse_forward")
        + mech("mhc × attnres", "mhc")
        + mech("layerrope", "layerrope_forward")
        + mech("share_u", "share_u_forward")
        + mech("engram rows", "engram_rows")
        + mech("moe triton gemm", "moe_triton_gemm")
        + mech("kda (fla)", "kda_fla")
    )

    if g.get("ok"):
        util = g["util"]
        vram = 100.0 * g["vram_used"] / max(1.0, g["vram_total"])
        pw = 100.0 * g["power"] / max(1.0, g["power_limit"])
        gpu_html = f"""<div class="gauge">
  <div class="row"><span class="lab">util</span>
    <div class="track"><div class="mk{' hot' if util > 90 else ''}" style="width:{util:.0f}%"></div></div>
    <span class="val">{util:.0f}%</span></div>
  <div class="row"><span class="lab">vram</span>
    <div class="track"><div class="mk" style="width:{vram:.0f}%"></div></div>
    <span class="val">{g['vram_used'] / 1024:.1f}/{g['vram_total'] / 1024:.0f} GiB</span></div>
  <div class="row"><span class="lab">power</span>
    <div class="track"><div class="mk" style="width:{pw:.0f}%"></div></div>
    <span class="val">{g['power']:.0f}/{g['power_limit']:.0f} W</span></div>
  <div class="row"><span class="lab">clock</span>
    <div class="track"><div class="mk" style="width:{min(100, g['clock'] / 31.8):.0f}%"></div></div>
    <span class="val">{g['clock']:.0f} MHz</span></div>
  <div class="row"><span class="lab">temp</span>
    <div class="track"><div class="mk{' hot' if g['temp'] > 78 else ''}" style="width:{min(100, g['temp']):.0f}%"></div></div>
    <span class="val">{g['temp']:.0f} °C</span></div>
</div>"""
    else:
        gpu_html = '<div class="void">[–] nvidia-smi unavailable</div>'

    if s["ckpts"]:
        ck = "".join(
            f"<li><span class='m'>·</span><b>{html.escape(x['name'])}</b>"
            f"<span class='r'>{x['mb']:.0f} MB · {_human_secs(x['age'])} ago</span></li>"
            for x in s["ckpts"]
        )
        ck_html = f"<ul class='rows'>{ck}</ul>"
    else:
        ck_html = '<div class="void">[–] no checkpoint yet</div>'

    ag = s.get("agent") or {}
    if ag:
        def pct(v):
            return f"{100 * v:.0f}%" if isinstance(v, (int, float)) else "—"

        ag_html = f"""<div class="kv">
  <div class="k">format (valid call)</div><div class="v">{pct(ag.get('format'))}</div>
  <div class="k">prose, no call</div><div class="v">{ag.get('no_call', '—')}/{ag.get('prompts', '—')}</div>
  <div class="k">executable</div><div class="v">{pct(ag.get('exec'))}</div>
  <div class="k">task success</div><div class="v">{ag.get('passed')}/{ag.get('tasks')}</div>
  <div class="k">sandbox refusals</div><div class="v">{ag.get('audit', '—')}</div>
</div><div class="void" style="margin-top:10px">{html.escape(str(ag.get('run')))} · {html.escape(str(ag.get('state')))}</div>"""
    else:
        ag_html = '<div class="void">[–] no agent eval yet</div>'

    if s["sft"]:
        sf = s["sft"]
        mix = " · ".join(
            f"{k.replace('_', ' ')} {v:.0f}%"
            for k, v in sorted(sf["mix"].items(), key=lambda kv: -kv[1])[:5]
        )
        risky = [k for k, v in sf["sources"].items() if v.get("risk") == "unclear"]
        sft_html = f"""<div class="kv">
  <div class="k">train rows</div><div class="v">{sf['rows']:,}</div>
  <div class="k">holdout</div><div class="v">{sf['holdout']}</div>
  <div class="k">clean subset</div><div class="v">{sf['clean']:,}</div>
  <div class="k">bytes</div><div class="v">{_human_bytes(sf['bytes'])}</div>
  <div class="k">quarantined</div><div class="v">{len(risky)} sources</div>
</div><div class="void" style="margin-top:10px">{html.escape(mix)}</div>"""
    else:
        sft_html = '<div class="void">[–] corpus not built</div>'

    commits = "".join(
        f"<li><span class='m'>›</span><b>{html.escape(r['subject'][:58])}</b>"
        f"<span class='r'>{r['sha']} · {html.escape(r['when'])}</span></li>"
        for r in s["git"][:5]
    ) or "<li><span class='m'>·</span>no git history<span class='r'></span></li>"

    return f"""<div class="grid">
  <div class="panel"><h3>mechanisms · live counters</h3><div class="kv">{mechanisms}</div></div>
  <div class="panel"><h3>gpu</h3>{gpu_html}</div>
  <div class="panel"><h3>checkpoints</h3>{ck_html}</div>
  <div class="panel"><h3>sft corpus v2</h3>{sft_html}</div>
  <div class="panel"><h3>capability · agent eval</h3>{ag_html}</div>
  <div class="panel"><h3>eval history</h3>
    <div id="evals" sse-swap="evals">{frag_evals(s)}</div></div>
  <div class="panel"><h3>recent commits</h3><ul class="rows">{commits}</ul></div>
</div>"""


def frag_evals(s: dict) -> str:
    h = s["history"][-8:]
    if not h:
        return '<div class="void">[–] no eval yet</div>'
    rows, prev = [], None
    for p in h:
        d = "—" if prev is None else f"{p['bpb'] - prev:+.4f}"
        cls = "steady" if prev is None or abs(p["bpb"] - prev) < 0.0005 else (
            "down" if p["bpb"] < prev else "up"
        )
        rows.append(
            f"<li><span class='m'>·</span><b>{p['step']:,}</b>"
            f"<span class='r'>{p['bpb']:.4f} <span class='{cls}'>{d}</span></span></li>"
        )
        prev = p["bpb"]
    return "<ul class='rows'>" + "".join(reversed(rows)) + "</ul>"


FRAGS = {
    "metrics": frag_metrics,
    "hero": frag_hero,
    "chart": frag_chart,
    "panels": frag_panels,
  "routing": frag_routing,
    "evals": frag_evals,
}


# ------------------------------------------------------------------ page


def page(s: dict, query: str = "") -> str:
    r = s["run"]
    pct = 100.0 * r["step"] / max(1, r["total_steps"])
    decay = 100.0 * r["decay_at"] / max(1, r["total_steps"])
    state = s["state"]
    label = {"live": "live", "stale": "stalled", "dead": "offline"}[state]
    host = socket.gethostname().split(".")[0]
    dark = "theme=dark" in query
    htmx = '<script src="/static/htmx.min.js" defer></script>' if (STATIC / "htmx.min.js").exists() else ""
    fallback = "" if htmx else '<meta http-equiv="refresh" content="5">'
    cfg = _jsonl_snapshot_cfg()
    return f"""<!doctype html>
<html lang="en"{" data-theme=dark" if dark else ""}><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dormouse · live</title>
{fallback}
{htmx}
<link rel="stylesheet" href="/static/dash.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' fill='%23201d1d'/><path d='M4 3h5a3 3 0 0 1 0 6H7v4H4z' fill='%23fdfcfc'/></svg>">
</head>
<body>
<div class="wrap" hx-ext="sse" sse-connect="/events">

  <div class="top">
    <a class="wordmark" href="/">
      <span class="bracket">[d]</span> dormouse <span class="slash">/ live</span>
    </a>
    <div class="meta">
      <span class="live {state}"><span class="dot"></span>{label}</span>
      <span><b>{host}</b></span>
      <span>{cfg}</span>
      <span id="clock">—</span>
    </div>
  </div>

  <div class="block" style="--i:0">
    <h2>run <span class="hint">{html.escape(r['shape'])} · {_params(r['params'])}</span></h2>
    <div class="run">
      <span class="id">{r['name']}</span>
      <span>step <b>{r['step']:,}</b> / {r['total_steps']:,}</span>
      <span>{pct:.2f}%</span>
      <span>nan_skipped <b>{r['nan']}</b></span>
      <span>{_human_bytes(r['seen'])} of {_human_bytes(r['budget'])}</span>
    </div>
    <div class="bar">
      <div class="fill" style="width:{pct:.3f}%"></div>
      <div class="decay" style="left:{decay:.3f}%"></div>
    </div>
    <div class="scale"><span>0</span><span>{r['total_steps'] // 2:,}</span>
      <span>decay {r['decay_at']:,}</span><span>{r['total_steps']:,}</span></div>
  </div>

  <div class="block" style="--i:1" id="hero" sse-swap="hero">{frag_hero(s)}</div>
  <div class="block" style="--i:2" id="metrics" sse-swap="metrics">{frag_metrics(s)}</div>
<div class="block" style="--i:3"><h2>held-out bpb <span class="hint">eval every 2k steps ·
      64 KiB window</span></h2>
    <div id="chart" sse-swap="chart">{frag_chart(s)}</div></div>
  <div class="block" style="--i:4" id="routing" sse-swap="routing">
    <h2>routing health <span class="hint">live experts / E · below 50% is a dead bank</span></h2>
    {frag_routing(s)}</div>
  <div class="block" style="--i:5;padding:0;border:0" id="panels" sse-swap="panels">{frag_panels(s)}</div>

  <div class="foot">
    <span>read-only · SSE every {REFRESH_S:.0f}s · no torch, no GPU lock</span>
    <span>{s['exited'] or 'training in progress'}</span>
    <span><a href="/api/snapshot">/api/snapshot</a></span>
  </div>
</div>
<script>
const c = document.getElementById('clock');
const tick = () => c.textContent = new Date().toLocaleTimeString('en-GB', {{hour12:false}});
tick(); setInterval(tick, 1000);
</script>
</body></html>"""




def _params(n) -> str:
    """`param_count` lives behind the model build; the snapshot has no count.

    Showing "—" beats showing the flagship's 102 678 943 on a run that is not it
    (that is the bug this whole block exists to kill).
    """
    return f"{int(n):,} params" if n else "params n/a"



def _stair(evals: list[dict], run_cfg: dict) -> dict:
    """The active run against the 105M flagship, on TOKENS SEEN, not steps.

    The whole "why 200M" question is a token-vs-capacity trade, and comparing
    curves by step answers it wrongly (a bigger model is behind at equal steps
    by construction). Missing reference -> {} and the panel hides itself.
    """
    try:
        bps = (
            int(run_cfg.get("batch", 8)) * int(run_cfg.get("seq_len", 1024))
            * int(run_cfg.get("grad_accum", 1))
        )
        ref_rows = _jsonl(LOGS / "flagship_resume.jsonl") + _jsonl(LOGS / "flagship.jsonl")
        ref = sorted({(r["step"], r["bpb"]) for r in ref_rows if r.get("event") == "eval"})
        ref_bps = 8 * 1024 * 3
        return {
            "name": RUN,
            "ref_name": "flagship 105M",
            "cur": [[round(st * bps / 1e9, 3), round(b, 4)] for st, b in
                    ((e["step"], e["bpb"]) for e in evals)],
            "ref": [[round(st * ref_bps / 1e9, 3), round(b, 4)] for st, b in ref],
        }
    except Exception:
        return {}



def _agent_eval() -> dict:
    """The newest `logs/agent_eval_*.json`: the capability numbers, not BPB.

    Absent file -> {} and the panel says so. These four are reported apart on
    purpose (a blended score hides which one is broken).
    """
    try:
        files = sorted(LOGS.glob("agent_eval_*.json"), key=lambda f: f.stat().st_mtime)
        if not files:
            return {}
        d = json.loads(files[-1].read_text(encoding="utf-8"))
        return {
            "run": files[-1].stem.replace("agent_eval_", ""),
            "state": d.get("state"),
            "format": (d.get("format") or {}).get("rate"),
            "prompts": (d.get("format") or {}).get("prompts"),
            "no_call": (d.get("format") or {}).get("no_call"),
            "exec": (d.get("execution") or {}).get("rate"),
            "passed": sum(v.get("passed", 0) for v in (d.get("task_success") or {}).values()),
            "tasks": sum(v.get("tasks", 0) for v in (d.get("task_success") or {}).values()),
            "audit": (d.get("escape_audit") or {}).get("attempts"),
        }
    except Exception:
        return {}


def _run_config() -> dict:
    """The active run's own snapshot: `<out_dir>/<name>.config.toml` [train]+[model].

    Flat, because every consumer here wants one key. Missing file -> {} (the
    board then shows the fallbacks rather than refusing to render).
    """
    try:
        import tomllib

        p = CKPT / f"{RUN}.config.toml"
        cfg = tomllib.loads(p.read_text())
        t, m = cfg.get("train", {}), cfg.get("model", {})
        out = {**m, **t}
        out["params"] = cfg.get("params") or m.get("params")
        return out
    except Exception:
        return {}


def _jsonl_snapshot_cfg() -> str:
    try:
        import tomllib

        p = CKPT / f"{RUN}.config.toml"
        cfg = tomllib.loads(p.read_text())
        m, t = cfg.get("model", {}), cfg.get("train", {})
        return (
            f"lr {t.get('lr', 0):.1e} · it{m.get('max_iter', '?')} · "
            f"E{m.get('n_experts', '?')} · rank {m.get('rank', '?')}"
        )
    except Exception:
        return "config n/a"


# ------------------------------------------------------------------ server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dormouse-dashboard/1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/healthz":
            s = snapshot()
            return self._send(
                200,
                f"ok step={s['run']['step']} state={s['state']}\n".encode(),
                "text/plain; charset=utf-8",
            )
        if path == "/api/snapshot":
            return self._send(
                200, json.dumps(snapshot(), default=str).encode(), "application/json"
            )
        if path == "/":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            return self._send(
                200, page(snapshot(), query).encode(), "text/html; charset=utf-8"
            )
        if path.startswith("/frag/"):
            name = path.split("/")[-1]
            fn = FRAGS.get(name)
            if fn is None:
                return self._send(404, b"unknown fragment\n", "text/plain")
            return self._send(200, fn(snapshot()).encode(), "text/html; charset=utf-8")
        if path == "/events":
            return self._sse()
        if path.startswith("/static/"):
            name = path.rsplit("/", 1)[-1]
            p = STATIC / name
            if not p.is_file():
                return self._send(404, b"not found\n", "text/plain")
            ctype = {
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml",
            }.get(p.suffix, "application/octet-stream")
            return self._send(200, p.read_bytes(), ctype)
        return self._send(404, b"not found\n", "text/plain")

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                s = snapshot()
                for name, fn in FRAGS.items():
                    body = "".join(fn(s).splitlines())
                    payload = f"event: {name}\ndata: {body}\n\n"
                    self.wfile.write(payload.encode())
                self.wfile.flush()
                time.sleep(REFRESH_S)
        except (BrokenPipeError, ConnectionResetError):
            return


def serve(host: str = "0.0.0.0", port: int = 3789) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def _urls(port: int) -> None:
    print(f"  local      http://127.0.0.1:{port}/")
    try:
        ip = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2
        ).stdout.strip().splitlines()
        if ip:
            print(f"  tailscale  http://{ip[0]}:{port}/")
    except Exception:
        pass
    print(f"  https      tailscale serve --bg --https=443 http://127.0.0.1:{port}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=3789)
    args = ap.parse_args(argv)
    httpd = serve(args.host, args.port)
    print(f"dormouse dashboard on {args.host}:{args.port}")
    _urls(args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
