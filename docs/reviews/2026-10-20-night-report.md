# Night report — 2026-10-19 → 2026-10-20

What ran while the owner slept, in the order it happened, with the numbers a
reader can check against `~/logs/`.

## 1. The quality loop closed at 7/10 (three rounds, the owner accepted it)

| round | score | what moved it |
|---|---|---|
| the code audit | 6/10 | an init the seed did not control; an SFT corpus that shared its answers with the RL/bench verifier |
| round 1 | 7/10 | two gates that could not fail; the new counter reason crashed `opt_graph` |
| round 2 | 6/10 | a sympy RCE in the math verifier and a jail bypass; two regressions the previous wave introduced |
| round 3 | 7/10 | the reward path died on a 23-byte answer; the LFB scope stopped at the pretrain trainer; a mixture resume rewound |

Every finding is in `docs/reviews/2026-10-19-findings-ledger.md` with its
status, and the fixes carry revert-checked gates. The three reports are in
`docs/reviews/2026-10-19-review-round-{1,2,3}.md`, the protocol and the final
verdict in `docs/reviews/2026-10-19-review-loop.md`.

The fix list from the last round, all landed:
`(2**1000)**1000` is refused before the power is computed (worst allowed
expression 0.31 ms, was 5.45 s); a verifier bug fails the sample, not the batch;
the SFT corpus holds no solution the verifier can score; the no-`--eval` tail is
sized in reads (20 windows, not 19); the LFB sidecar is LOUD only on a training
resume and `lfb_counts` restarts with the load; a mixture resume replays its
prefix by step index; and the SFT/RL checkpoint lines carry the nvfp4 tally.

## 2. Multi-GPU and heterogeneous hardware landed (`multigpu`, merged)

A device-capability layer (per-device dtype/kernel policy: bf16 → fp16 → fp32,
Triton-MoE needs bf16, FLA needs fp16/bf16), DDP verified on two CPU ranks
(|Δloss| vs one process 5.5e-3 at a 1e-2 tolerance, parameters 3e-8 apart),
FSDP2 for the AdamW arm and a LOUD refusal for Muon (measured cause:
`_foreach_add_` mixing Tensor and DTensor), a per-rank capacity profile with
token-weighted loss scaling (1.1e-5 against the hand-computed mean), and a
`--fit <GB>` knob that names the mechanisms it engages: small and medium fit a
**4 GB** card with none, b1 needs `moe_offload + engram_ram` on 16 GB (8.03 GiB).

Three fleet traps are LOUD refusals with the measurement behind them: host
Engram tables (never reduced), capacity-padded MoE (logits differ by 0.357 at
`capacity=1.0`, by 1.2e-6 at dropless) and the forward-mutating LFB controller.

What needs the second card is named in `docs/architecture/multi-gpu.md`,
including the one command that checks a 1650 when it arrives.

## 3. The base model: honest verdict on the chat

`sft12` (SFT on the 80k-step `mix1`, 4000 steps, corpus without verifier-scored
pairs) finished at **CE 1.437 / BPB 2.073** on the 32 768 B holdout, `first=`
29-36%. Its chat transcript (`~/logs/chat_sft12_transcript.txt`) is word salad.

The cause is not SFT: the **base** produces the same salad. `dormouse decode` on
`mix1` at `temp 0.05` gives "The capital of France is a family and the fact that
the fact that…" — the syntax of a model that predicts bytes well (eval 1.901
BPB over 196 608 B) and has seen **0.85% of the 230 GB mixture**. Both decode
arms (chunked and recurrent) agree, so the decoder is not the difference.

That is the size/data wall, stated plainly: 88M stored / ~20M active at 1.9 GB
seen does not speak.

## 4. The long pretrain (running)

`mix1` resumed to **170 000 steps** (from 80 000): +2.2 GB of the four-source
mixture, the stage plan moving with the budget (S2 at 119k, S3 decay at 153k),
all the measured optimizations on (Triton NS, optimizer CUDA-graph, bf16 Muon
state, dropless Triton MoE, Engram in host RAM, tf32).

`~/logs/mix1_long2.{log,jsonl}`, ~40.7k tok/s, ETA ~15 h. The `.lfb` break-glass
is the checkpoint's missing sidecar (`moe_lb_coef=0`, COUNTED on the load line).

When it finishes: `bench` on the new checkpoint, then `sft13` (bilingual corpus:
rudialog for RU, dolly + agentic for EN, the hand-written `.qa` files for tone;
short answers on purpose), then `serve` and a transcript.
