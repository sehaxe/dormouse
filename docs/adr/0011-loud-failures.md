# Loud failures: no silent data, no silent skips

Pretrain v21 trained on a constant `b'x'` stream: an unreadable corpus root
produced an empty file list, and `refill()` silently fabricated filler bytes
instead of failing. Train CE collapsed to 0.11 while everything looked like
learning; only the held-out eval (a separate stream) exposed it. The same
silent-failure shape existed elsewhere: a grad-shape mismatch skipped the
host-table update instead of failing, a short read was space-padded, a v1
sidecar was reported and then discarded, `max_ortho` drift and the drift check
were the only loud mechanisms.

Decision: NASA P10 Rule 5 (docs/research/2026-09-25-nasa-burn-rust-practices-renamed.md) —
assertion density >= 2 per non-trivial function on average, and every assertion
failure routes to a named recovery action (the `--guard` checkpoint-and-resume
is that action; a startup config error is a hard error with the offending path
named). Data that does not exist stops the run; it is never synthesized. Shape
mismatches stop the run; they are never skipped. Backwards-compat fallbacks
that discard trained state (sidecar relayout) are hard errors naming the
escape (delete the file or pick another `--ckpt-name`).

Landmarks: `collect_files`/`from_files` asserts + corpus size floor
(commit 9605896), the refill dry-assert and short-read assert (data/lib.rs),
host-grad shape assert and sidecar hard error (train/lib.rs), dead-field
validation removed rather than kept as false confidence
(docs/audit-2026-09-25.md §3.10).
