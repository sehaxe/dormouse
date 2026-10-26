#!/usr/bin/env python3
"""Score a HuggingFace causal LM in **bits per byte** on a raw byte file.

Why this exists: our model's BPB and a tokenizer model's BPB are only
comparable when both are measured on the SAME bytes. A token model's
`loss` is per token, and a token covers several bytes, so the conversion is

    BPB = sum(-log2 p(token_i)) / n_bytes

i.e. the total code length in bits divided by the number of bytes that were
encoded — never per token, never averaged over a tokenizer's own split.

Usage:
    python scripts/bpb_hf.py <model> <file.bin> [--bytes N] [--stride N]

The file is read as UTF-8 with `errors="replace"` (the corpus is UTF-8); a
replacement byte is still one byte, so the denominator stays honest.
"""

from __future__ import annotations

import argparse
import math
import sys


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("file")
    ap.add_argument("--bytes", type=int, default=100_000, help="how many bytes to score")
    ap.add_argument("--stride", type=int, default=512, help="tokens per forward")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = open(args.file, "rb").read(args.bytes)
    text = raw.decode("utf-8", errors="replace")
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        print("empty slice", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total_bits = 0.0
    scored = 0
    with torch.no_grad():
        for start in range(0, len(ids) - 1, args.stride):
            chunk = ids[start : start + args.stride + 1]
            if chunk.numel() < 2:
                break
            logits = model(chunk[:-1].unsqueeze(0)).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            tgt = chunk[1:]
            lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            total_bits += float(-lp.sum()) / math.log(2)
            scored += int(tgt.numel())
            print(f"  {scored} tokens, {total_bits / n_bytes:.4f} BPB so far", flush=True)

    bpb = total_bits / n_bytes
    print(f"\nmodel={args.model}")
    print(f"file={args.file} ({n_bytes} bytes, {scored} tokens)")
    print(f"tokens/byte={scored / n_bytes:.3f}")
    print(f"BPB={bpb:.4f}  (perplexity per byte = {2 ** bpb:.2f}, compression = {8 / bpb:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
