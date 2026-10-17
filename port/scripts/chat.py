"""An interactive chat REPL against a dormouse checkpoint — no server needed.

One question per line, the answer streams in as it is sampled. Each line is a
FRESH single-turn record in the SFT template (`\\x01 Q\\n\\x02 A\\x03`) — the
exact shape the model was trained on, so multi-turn history is deliberately not
carried (a window must stay one example; the trainer's own rule).

    uv run --extra gpu python scripts/chat.py --ckpt-name sft7
    uv run --extra gpu python scripts/chat.py --ckpt-name sft7 --ask "Who are you?"
    uv run --extra gpu python scripts/chat.py --ckpt-name sft7 --spec-k 4

Commands: `/temp 0.4` sets the temperature, `/n 200` the byte cap, `/spec 4`
turns speculative decode on (0 = off), `/quit` exits. With `--spec-k` on, every
answer prints the acceptance rate and forwards/byte — the two numbers that say
whether the draft head is earning its keep on THIS checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dormouse.cli import _snapshot_run  # noqa: E402
from dormouse.instrument import loud  # noqa: E402
from dormouse.model import DormouseModel  # noqa: E402
from dormouse.model.decode import decode_bytes  # noqa: E402
from dormouse.model.specdecode import spec_decode_bytes  # noqa: E402

USER, ASSIST, END = 0x01, 0x02, 0x03


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="chat with a dormouse checkpoint")
    p.add_argument("--ckpt-name", default="sft7")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--n", type=int, default=90, help="max bytes to sample per answer")
    p.add_argument(
        "--spec-k",
        type=int,
        default=0,
        help="speculative decode: draft bytes per round (0 = off); needs a draft head",
    )
    p.add_argument("--ask", default="", help="one question, print the answer, exit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out_dir)
    run = _snapshot_run(out, args.ckpt_name)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DormouseModel(run.model)
    ck = torch.load(out / f"{args.ckpt_name}.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        loud(
            f"chat: the checkpoint and the preset disagree "
            f"(missing={list(missing)[:3]}, unexpected={list(unexpected)[:3]}). "
            f"Escape: the same preset the checkpoint was trained with."
        )
    model = model.to(device).eval()
    step = ck.get("step", "?")
    temp, n_new, spec_k = args.temp, args.n, args.spec_k

    def answer(question: str) -> bytes:
        prompt = bytes([USER]) + question.encode("utf-8") + b"\n" + bytes([ASSIST])
        t0 = time.perf_counter()
        with torch.no_grad():
            if spec_k:
                raw, stats = spec_decode_bytes(
                    model,
                    prompt,
                    n_new,
                    spec_k=spec_k,
                    temp=temp,
                    device=device,
                    stop=frozenset({END}),
                )
                rate = (
                    f"{stats.accepted}/{stats.drafted} accepted "
                    f"({stats.acceptance_rate:.0%}) · {stats.per_byte_forwards:.2f} fwd/byte"
                    f" · {stats.counters.line()}"
                )
            else:
                raw = decode_bytes(
                    model, prompt, n_new, temp=temp, device=device, stop=frozenset({END})
                )
                rate = "plain"
        dt = time.perf_counter() - t0
        body = raw[len(prompt) :]
        if body.endswith(bytes([END])):
            body = body[:-1]
        print(f"[{len(raw) - len(prompt)} B in {dt:.1f}s · {rate}]")
        return body

    if args.ask:
        sys.stdout.write(answer(args.ask).decode("utf-8", "replace") + "\n")
        return

    print(
        f"dormouse chat · {args.ckpt_name} (step {step}) · temp {temp} · "
        f"spec k={spec_k or 'off'} · /quit to exit"
    )
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line.startswith("/temp "):
            temp = float(line.split()[1])
            print(f"temp = {temp}")
            continue
        if line.startswith("/n "):
            n_new = int(line.split()[1])
            print(f"n = {n_new}")
            continue
        if line.startswith("/spec "):
            spec_k = int(line.split()[1])
            print(f"spec k = {spec_k or 'off'}")
            continue
        print(answer(line).decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
