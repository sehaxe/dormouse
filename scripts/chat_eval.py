"""Score a served checkpoint on a fixed chat set — the honest alternative to vibes.

Reads the OpenAI-compatible endpoint (`dormouse serve`), asks every item, and
prints one line per item plus three numbers:

    hit      the answer contains one of the item's expected substrings
    bytes    the answer's length (the project's token-thrift axis)
    time     wall clock for the request

Why a fixed set and not a benchmark: the chat models in this repo are tiny, and
"looks like a sentence" is not a measurement. A frozen list of questions with a
keyword per question is crude, but it is comparable across checkpoints, which is
what an SFT recipe needs (AGENTS 1.2: an arm beats its own removal on a fixed
protocol, or it is deleted).

Usage:
    python scripts/chat_eval.py --url http://127.0.0.1:3783 --model sft13
    python scripts/chat_eval.py --url … --model sft12 --json /tmp/chat_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: (prompt, any-of these lowercase substrings counts as a hit). Kept SHORT on
#: purpose: a good answer at this model size is one clause, and a long expected
#: string would measure the checker's imagination rather than the model's.
ITEMS: list[tuple[str, tuple[str, ...]]] = [
    ("Hello! Who are you?", ("dormouse", "assistant", "model", "help")),
    ("Hi!", ("hi", "hello", "hey")),
    ("What is your name?", ("dormouse",)),
    ("What is 2+2?", ("4", "four")),
    ("What is 7*6?", ("42", "forty")),
    ("What is 10-3?", ("7", "seven")),
    ("Name a color.", ("red", "blue", "green", "black", "white", "yellow")),
    ("Name an animal.", ("cat", "dog", "mouse", "bird", "fish", "horse", "dormouse")),
    ("What is the capital of France?", ("paris",)),
    ("What is the capital of Russia?", ("moscow", "москв")),
    ("How many days are in a week?", ("7", "seven", "семь")),
    ("Say something short.", ("",)),
    ("Are you a human?", ("no", "not", "не", "нет")),
    ("Can you write code?", ("yes", "no", "can", "python", "def")),
    ("Привет! Как дела?", ("привет", "хорошо", "нормально", "дела")),
    ("Как тебя зовут?", ("dormouse", "соня", "зовут")),
    ("Сколько будет 2+2?", ("4", "четыре")),
    ("Назови цвет.", ("красн", "син", "зелён", "зелен", "чёрн", "черн", "бел")),
    ("Столица России?", ("москв",)),
    ("Что такое дерево?", ("дерев", "растен", "листв", "ствол")),
    ("Скажи что-нибудь короткое.", ("",)),
    ("Ты человек?", ("нет", "не ", "no")),
    ("What can you do?", ("help", "answer", "write", "can", "assist")),
    ("Tell me a joke.", ("",)),
    ("What is a byte?", ("8", "eight", "bit", "byte", "data")),
]


def ask(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[str, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload["choices"][0]["message"]["content"], time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:3783")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args(argv)

    rows, hits, total_bytes = [], 0, 0
    for prompt, expect in ITEMS:
        try:
            answer, dt = ask(a.url, a.model, prompt, a.max_tokens, a.timeout)
            err = ""
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            answer, dt, err = "", 0.0, f"{type(e).__name__}: {e}"
        low = answer.lower()
        hit = (not expect) or any(s in low for s in expect)
        # An empty answer is never a hit, even for the "anything" items: a model
        # that says nothing has not answered the question.
        if not answer.strip():
            hit = False
        hits += hit
        total_bytes += len(answer.encode())
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "hit": hit,
                "bytes": len(answer.encode()),
                "seconds": round(dt, 2),
                "error": err,
            }
        )
        print(
            f"{'HIT ' if hit else 'miss'} {len(answer.encode()):4d}B {dt:6.1f}s  "
            f"{prompt!r} -> {answer!r}"[:200]
        )
        if err:
            print(f"     error: {err}", file=sys.stderr)

    n = len(ITEMS)
    print(
        f"\n{a.model}: {hits}/{n} hits ({100 * hits / n:.0f}%), "
        f"{total_bytes / n:.0f} B/answer, {n} items"
    )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "model": a.model,
                    "hits": hits,
                    "items": n,
                    "mean_bytes": total_bytes / n,
                    "rows": rows,
                },
                f,
                indent=1,
            )
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
