"""Build the v2 SFT corpus from OPEN, frontier-distilled datasets.

The owner's rule: newest teachers, and a license that is either unambiguous or
explicitly quarantined. Every source is one file under `sft_pairs/v2/full/`, and
the low-risk subset is mirrored byte-for-byte under `sft_pairs/v2/clean/`, so a
risky source can be dropped without rebuilding.

Shape of a row (what `train/sft.py::_chat_pairs` consumes):

    {"prompt": ..., "response": ..., "source": ..., "domain": ...,
     "license": ..., "license_risk": "low" | "unclear"}

Filters, applied in this order, each counted at the first rule that fires:
control bytes -> prompt/response length -> language -> degenerate -> exact dup ->
near dup (MinHash bottom-k, J >= 0.9, within a source) -> eval leak (>= 50 bytes
shared with `web_eval.bin`) -> per-source quota (smallest hash wins, so the cap
is not the head of the file).

Determinism: no timestamps, sorted iteration, fixed JSON separators. Two runs on
the same inputs produce byte-identical files and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

PROMPT_MAX = 400
RESPONSE_MAX = 500
WINDOW = 1024  # the model's window; render(prompt, response) must fit it
MINHASH_K = 128
NEAR_DUP_J = 0.90
LEAK_BYTES = 50
HOLDOUT_ROWS = 300
META_MINHASH_SEED = 0x5EED_1234

ASCII_RE = re.compile(r"[A-Za-z]")
CYR_RE = re.compile(r"[\u0400-\u04FF]")
ALNUM_RE = re.compile(r"[0-9A-Za-z\u0400-\u04FF]")

#: (name, domain, license, risk, quota). Quotas sum to ~25k; long-trace sources
#: are capped near 10% of the budget, the two `unclear` ones are quarantined.
SOURCES = [
    ("tool_ratchet", "tool", "apache-2.0", "low", 6000),
    ("code_glm", "code", "mit", "low", 6000),
    ("code_fable", "code", "cc-by-4.0", "low", 1500),
    ("code_kimi", "code", "cc-by-4.0", "low", 800),
    ("reason_qwen38", "reasoning", "other:alibaba-terms", "unclear", 4000),
    ("reason_gpt61", "reasoning", "gpl-2.0", "unclear", 2500),
    ("math_r1", "math", "apache-2.0", "low", 4000),
]
QUOTA = {n: q for n, _d, _l, _r, q in SOURCES}
RISK = {n: r for n, _d, _l, r, _q in SOURCES}
LICENSE = {n: l for n, _d, l, _r, _q in SOURCES}
DOMAIN = {n: d for n, d, _l, _r, _q in SOURCES}

DROP_KEYS = (
    "control",
    "prompt_long",
    "response_long",
    "language",
    "degenerate",
    "exact_dup",
    "near_dup",
    "leak",
    "quota",
    "render",
)


# ---------------------------------------------------------------- utilities


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, vectorised. Wraps modulo 2**64."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def sketch(text: bytes, k: int = MINHASH_K) -> np.ndarray:
    """Bottom-k 5-byte-shingle sketch. A 5-byte shingle is a 40-bit value, so the
    packed integer is unique per shingle: no hashing collisions before `_mix`,
    which is itself a bijection, so identical sketches mean identical samples."""
    b = np.frombuffer(text, np.uint8).astype(np.uint64)
    if b.size < 5:
        return np.empty(0, np.uint64)
    v = (
        (b[:-4] << np.uint64(32))
        | (b[1:-3] << np.uint64(24))
        | (b[2:-2] << np.uint64(16))
        | (b[3:-1] << np.uint64(8))
        | b[4:]
    )
    h = _mix(v ^ np.uint64(META_MINHASH_SEED))
    return np.unique(h)[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(a.size + b.size - inter)


def buckets(sig: np.ndarray, n: int = 4, bits: int = 20) -> list[int]:
    return [int(sig[i] & np.uint64((1 << bits) - 1)) for i in range(min(n, sig.size))]


def lang_of(text: str) -> str | None:
    latin = len(ASCII_RE.findall(text))
    cyr = len(CYR_RE.findall(text))
    tot = latin + cyr
    if tot < 2:
        return None
    if latin / tot >= 0.85:
        return "en"
    if cyr / tot >= 0.85:
        return "ru"
    return None


def degenerate(prompt: str, response: str) -> str | None:
    if not response.strip():
        return "empty"
    if not ALNUM_RE.search(response):
        return "no-alnum"
    if prompt.isascii() and not response.isascii():
        non = sum(1 for ch in response if ord(ch) > 127)
        if non / max(1, len(response)) > 0.40:
            return "non-ascii"
    return None


def has_control(s: str) -> bool:
    return any(ord(ch) < 32 and ch not in "\t\n\r" for ch in s)


def key_of(name: str, body: bytes) -> bytes:
    return hashlib.blake2b(f"{name}|".encode() + body, digest_size=8).digest()


# ---------------------------------------------------------------- leak guard


class LeakGuard:
    """50-byte-window membership against the eval bytes, exact on a hit."""

    W = LEAK_BYTES
    P = np.uint64(0x100000001B3)

    def __init__(self, eval_path: Path) -> None:
        self.raw = eval_path.read_bytes()
        self.needles: list[bytes] = []
        self.sorted_hash: np.ndarray | None = None
        if len(self.raw) < self.W:
            return
        h = self._windows(np.frombuffer(self.raw, np.uint8))
        order = np.argsort(h)
        self.sorted_hash = h[order]
        self.needles = [self.raw[int(i) : int(i) + self.W] for i in order[:4096]]
        self._pows = self._pow_table(len(self.raw) + 1)

    @staticmethod
    def _pow_table(n: int) -> np.ndarray:
        p = np.full(n, np.uint64(0x100000001B3), dtype=np.uint64)
        p[0] = np.uint64(1)
        return np.cumprod(p)

    def _windows(self, b: np.ndarray) -> np.ndarray:
        n = b.size
        if n < self.W:
            return np.empty(0, np.uint64)
        pows = self._pow_table(n + 1)
        inv = np.cumprod(np.full(n, np.uint64(pow(int(self.P), -1, 1 << 64)), np.uint64))
        h = np.cumsum(b.astype(np.uint64) * inv) * pows[:n]
        a = h[self.W - 1 :]
        bprev = np.concatenate(([np.uint64(0)], h[: n - self.W]))
        return a - bprev * pows[self.W]

    def leaks(self, body: bytes) -> bool:
        if self.sorted_hash is None or len(body) < self.W:
            return False
        b = np.frombuffer(body, np.uint8)
        wh = self._windows(b)
        pos = np.searchsorted(self.sorted_hash, wh)
        pos = np.clip(pos, 0, self.sorted_hash.size - 1)
        hits = wh[wh == self.sorted_hash[pos]]
        if hits.size == 0:
            return False
        for start in np.nonzero(np.isin(wh, hits))[0][:8]:
            s = int(start)
            if body[s : s + self.W] in self.raw:
                return True
        return False


# ---------------------------------------------------------------- per-source readers


def _chat_pair(messages: list[dict]) -> tuple[str, str] | None:
    """First user message -> last assistant message (system prompts dropped)."""
    user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    asst = [m.get("content") for m in messages if m.get("role") == "assistant"]
    if not user or not asst or not isinstance(asst[-1], str):
        return None
    return user, asst[-1]


def _ratchet_tools(user_msg: str) -> tuple[str, str]:
    """Split `Tools:\\n{json}\\n...\\n<request>` into (query, compact tool list)."""
    lines = user_msg.splitlines()
    tools, rest = [], []
    for line in lines:
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                rest.append(line)
                continue
            if "name" in obj:
                tools.append(obj)
                continue
        rest.append(line)
    query = "\n".join(rest).strip()
    return query, tools


def _compact_tool(obj: dict) -> str:
    params = obj.get("parameters") or {}
    bits = []
    for k, v in list(params.items())[:6]:
        t = v.get("type") if isinstance(v, dict) else "string"
        bits.append(f"{k}:{t}")
    return f"{obj.get('name')}({', '.join(bits)})" if bits else str(obj.get("name"))


def iter_ratchet(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_msgs = row.get("prompt") or []
            user = next(
                (m["content"] for m in prompt_msgs if m.get("role") == "user"), None
            )
            comp = row.get("completion") or []
            asst = next((m["content"] for m in comp if m.get("role") == "assistant"), None)
            if not user or not asst:
                continue
            query, tools = _ratchet_tools(user)
            try:
                call = json.loads(asst)
            except json.JSONDecodeError:
                continue
            used = call.get("name")
            shown = [t for t in tools if t.get("name") == used] or tools[:6]
            head = "Tools: " + "; ".join(_compact_tool(t) for t in shown) if shown else ""
            prompt = (head + "\n" + query).strip() if head else query
            # teach the exact one-object answer, compactly
            response = json.dumps(
                {"name": used, "arguments": call.get("arguments", {})},
                ensure_ascii=False,
                separators=(", ", ": "),
            )
            yield prompt, response


def iter_glm(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            code = row.get("code") or ""
            instr = row.get("instruction") or ""
            if code and instr:
                yield instr, code


def iter_qkf(path: Path):
    """(source label, prompt, response) for the Qwen3.8 / Fable-5 / Kimi-K3 mix."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = _chat_pair(row.get("messages") or [])
            if pair is None:
                continue
            yield str(row.get("source")), pair[0], pair[1]


def iter_gpt61(path: Path):
    """`answer` carries the reply; `reasoning` is a separate long field we do not
    have the bytes for (103M model, 1024-byte window)."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    for ex in blob.get("examples", []):
        msgs = ex.get("messages") or []
        user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        asst = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        body = (asst.get("answer") or asst.get("content") or "").strip()
        if body:
            yield user, body


def iter_openr1(path: Path):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["problem", "solution", "answer"])
    for problem, solution, answer in zip(
        t.column("problem").to_pylist(),
        t.column("solution").to_pylist(),
        t.column("answer").to_pylist(),
        strict=True,
    ):
        if not problem:
            continue
        text = (solution or "").strip()
        resp = text if 0 < len(text) <= RESPONSE_MAX else f"answer: {answer}"
        yield problem, resp


# ---------------------------------------------------------------- build


@dataclass
class Stat:
    seen: int = 0
    kept: int = 0
    bytes_out: int = 0
    drops: dict[str, int] = field(default_factory=lambda: {k: 0 for k in DROP_KEYS})

    def drop(self, why: str) -> None:
        self.drops[why] = self.drops.get(why, 0) + 1


def render_len(prompt: str, response: str) -> int:
    """The rendered byte count the model actually sees (prompt + markers + answer)."""
    return len(prompt.encode()) + len(response.encode()) + 3


def build(src: Path, out: Path, *, limit: int = 0, eval_path: Path | None = None) -> dict:
    guard = LeakGuard(eval_path) if eval_path and eval_path.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    global_exact: set[bytes] = set()
    manifest: dict = {
        "format": 1,
        "rules": {
            "prompt_max": PROMPT_MAX,
            "response_max": RESPONSE_MAX,
            "window": WINDOW,
            "near_dup_jaccard": NEAR_DUP_J,
            "leak_bytes": LEAK_BYTES,
            "holdout_rows": HOLDOUT_ROWS,
        },
        "eval_file": str(eval_path) if eval_path else None,
        "sources": {},
    }
    accepted: dict[str, list[dict]] = {n: [] for n, *_ in SOURCES}

    def consume(name: str, prompt: str, response: str) -> None:
        st: Stat = manifest["sources"][name]["stat"]
        st["seen"] += 1
        if has_control(prompt) or has_control(response):
            st["drops"]["control"] += 1
            return
        pb, rb = prompt.encode(), response.encode()
        if len(pb) > PROMPT_MAX:
            st["drops"]["prompt_long"] += 1
            return
        if len(rb) > RESPONSE_MAX:
            st["drops"]["response_long"] += 1
            return
        if lang_of(prompt + "\n" + response) not in ("en", "ru"):
            st["drops"]["language"] += 1
            return
        why = degenerate(prompt, response)
        if why:
            st["drops"]["degenerate"] += 1
            return
        if render_len(prompt, response) > WINDOW - 32:
            st["drops"]["render"] += 1
            return
        digest = hashlib.sha256(pb + b"\x00" + rb).digest()
        if digest in global_exact:
            st["drops"]["exact_dup"] += 1
            return
        sig = sketch(pb + b"\n" + rb)
        for bkey in buckets(sig):
            for prev_sig in manifest["sources"][name]["_buckets"].get(bkey, ())[:64]:
                if jaccard(sig, prev_sig) >= NEAR_DUP_J:
                    st["drops"]["near_dup"] += 1
                    return
        if guard is not None and guard.leaks(pb + b"\n" + rb):
            st["drops"]["leak"] += 1
            return
        global_exact.add(digest)
        for bkey in buckets(sig):
            manifest["sources"][name]["_buckets"].setdefault(bkey, []).append(sig)
        accepted[name].append(
            {
                "prompt": prompt,
                "response": response,
                "source": name,
                "domain": DOMAIN[name],
                "license": LICENSE[name],
                "license_risk": RISK[name],
            }
        )
        st["kept"] += 1

    for name, domain, lic, risk, quota in SOURCES:
        manifest["sources"][name] = {
            "stat": Stat().__dict__,
            "domain": domain,
            "license": lic,
            "license_risk": risk,
            "quota": quota,
            "_buckets": {},
        }

    feeds = {
        "tool_ratchet": lambda: iter_ratchet(src / "ratchet/data/train_v2.jsonl"),
        "code_glm": lambda: iter_glm(
            src / "glm17k/ox-alpha-glm-5.3-flash-distillation-coding-17k-raw.jsonl"
        ),
        "math_r1": lambda: (
            row
            for shard in ("train-00000-of-00010.parquet", "train-00001-of-00010.parquet")
            for row in iter_openr1(src / "or1/data" / shard)
        ),
        "reason_gpt61": lambda: iter_gpt61(src / "gpt61/Train.json"),
    }
    for name, feed in feeds.items():
        n = 0
        for prompt, response in feed():
            consume(name, prompt, response)
            n += 1
            if limit and n >= limit:
                break

    n = 0
    for label, prompt, response in iter_qkf(src / "qkf/curated.jsonl"):
        target = {"fable-5-trace": "code_fable", "kimi-k3-trace": "code_kimi"}.get(
            label, "reason_qwen38"
        )
        if label not in ("fable-5-trace", "kimi-k3-trace", "qwen3.8-max"):
            continue
        consume(target, prompt, response)
        n += 1
        if limit and n >= limit:
            break

    # quotas: keep the smallest hash, so the cap is not the head of the file
    for name in accepted:
        rows = accepted[name]
        quota = QUOTA[name]
        if len(rows) > quota:
            rows.sort(key=lambda r: key_of(name, r["prompt"].encode() + r["response"].encode()))
            manifest["sources"][name]["stat"]["drops"]["quota"] += len(rows) - quota
            accepted[name] = rows[:quota]

    # holdout: smallest hash over the rendered bytes, hash-disjoint by construction
    pool = [(name, r) for name in accepted for r in accepted[name]]
    pool.sort(key=lambda nr: key_of("holdout", render_key(nr[1])))
    # 300 rows, but never more than 5% of the pool (a tiny --limit run must still
    # leave a train file worth reading)
    want = min(HOLDOUT_ROWS, len(pool) // 20)
    hold = pool[: max(0, want)]
    hold_ids = {id(r) for _n, r in hold}

    files = {}
    for name in accepted:
        rows = [r for r in accepted[name] if id(r) not in hold_ids]
        rows.sort(key=lambda r: key_of(name, r["prompt"].encode() + r["response"].encode()))
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        st = manifest["sources"][name]["stat"]
        st["bytes_out"] = sum(
            len(r["prompt"].encode()) + len(r["response"].encode()) for r in rows
        )
        st["accounts"] = st["kept"] == sum(
            v for k, v in st["drops"].items() if k != "quota"
        ) + st["kept"]
        manifest["sources"][name].pop("_buckets", None)
        manifest["sources"][name]["rows"] = len(rows)

    hpath = out / "holdout.jsonl"
    with open(hpath, "w", encoding="utf-8") as f:
        for _name, r in hold:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    files[hpath.name] = hashlib.sha256(hpath.read_bytes()).hexdigest()

    clean_dir = out.parent / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name in accepted:
        if RISK[name] != "low":
            continue
        (clean_dir / f"{name}.jsonl").write_bytes((out / f"{name}.jsonl").read_bytes())
    clean_hold = [r for _n, r in hold if RISK[r["source"]] == "low"]
    with open(clean_dir / "holdout.jsonl", "w", encoding="utf-8") as f:
        for r in clean_hold:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    total_rows = sum(m["rows"] for m in manifest["sources"].values())
    manifest["totals"] = {
        "train_rows": total_rows,
        "holdout_rows": len(hold),
        "bytes": sum(m["stat"]["bytes_out"] for m in manifest["sources"].values()),
        "clean_rows": sum(
            m["rows"] for k, m in manifest["sources"].items() if RISK[k] == "low"
        ),
    }
    manifest["mix"] = {
        k: round(100.0 * m["rows"] / max(1, total_rows), 2)
        for k, m in manifest["sources"].items()
    }
    manifest["files"] = files
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (clean_dir / "manifest.json").write_text(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "files"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def render_key(row: dict) -> bytes:
    return (row["prompt"] + "\n" + row["response"]).encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/tmp/opencode/sftsrc", type=Path)
    ap.add_argument("--out", default="sft_pairs/v2/full", type=Path)
    ap.add_argument("--eval", default="/home/sehaxe/dormouse/data_local/mix4/web_eval.bin")
    ap.add_argument("--limit", default=0, type=int, help="rows per source (determinism gate)")
    args = ap.parse_args(argv)
    man = build(args.src, args.out, limit=args.limit, eval_path=Path(args.eval))
    print(json.dumps(man["totals"], indent=2))
    for name, m in man["sources"].items():
        print(
            f"  {name:16s} seen {m['stat']['seen']:6d}  rows {m['rows']:5d}  "
            f"risk {m['license_risk']:8s} drops {m['stat']['drops']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
