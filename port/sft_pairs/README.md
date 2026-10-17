# sft_pairs — the hand-written SFT set

Small, curated, **hand-written**: the character of the model lives here. The
big distill exports (breadth) are mixed in by the training command below; this
directory is the quality core.

## Files

| file | what |
|---|---|
| `identity.qa` | who the model is, what it can and cannot do, honesty |
| `smalltalk.qa` | greetings, politeness, small talk (EN+RU) |
| `humor.qa` | light jokes (EN+RU) |
| `math.qa` | hand-written arithmetic, `answer: N` format kept for the verifier |
| `facts.qa` | simple facts a tiny model must hold (EN+RU) |
| `code.qa` | small functions and one-line outputs |
| `style.qa` | brevity, uncertainty, corrections, refusals — the character |
| `ru.qa` | Russian: talk, facts, jokes, arithmetic |
| `mine.qa` | **your own pairs** — edit freely, they are mixed in like the rest |

## Format

One `Q:` line and one `A:` line per pair. `#` comments and blank lines are
free. Both halves must fit the window together (~500 bytes) — longer pairs are
dropped with a count, not truncated. Answers single-line, keep them short.

## The build command (mixed with the distill corpus + RU bank)

```bash
cd /home/sehaxe/dormouse/port
uv run --extra gpu dormouse sft medium \
  --data /home/sehaxe/dormouse/data_local/sft_chat.bin --build-sft \
  --chat-data /home/sehaxe/dormouse/data_local/distill_short.jsonl \
  --chat-data /home/sehaxe/dormouse/data_local/more_short.jsonl \
  --chat-data sft_pairs/identity.qa --chat-data sft_pairs/smalltalk.qa \
  --chat-data sft_pairs/humor.qa --chat-data sft_pairs/math.qa \
  --chat-data sft_pairs/facts.qa --chat-data sft_pairs/code.qa \
  --chat-data sft_pairs/style.qa --chat-data sft_pairs/ru.qa \
  --chat-data sft_pairs/mine.qa \
  --chat-cap 12000 \
  --init-from checkpoints/medium/init_full2.pt --ckpt-name sft6 --out-dir checkpoints/sft \
  --set steps=4000 --set seq_len=512 --device cuda
```

Every malformed line is a loud error naming `file:line` — a typo in a personal
file gets a message, not a silent skip.
