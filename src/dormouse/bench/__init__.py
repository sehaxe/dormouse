"""The downstream MC eval harness — "is the model smart", not "does it predict bytes".

Everything the repo measured until now was BPB: a held-out compression number.
BPB is necessary and not sufficient — a model can sharpen its byte distribution
and still be unable to pick the right answer among four. This package is the
second instrument: byte-level multiple choice, the GPT-2-era protocol, correct
for a vocab-256 model that has no tokenizer to hide behind.

**The score** ([`score.score_choices`]): for an item `(context C, choices
c_0..c_k, label)`, `score(c_i) = Σ_j log p(c_i[j] | C + c_i[:j])`, teacher-forced
in ONE forward per choice over `C + c_i`. Prediction = `argmax_i score(c_i)`.
Accuracy is the mean over items.

**Two accuracies, both printed** ([`runner`]): the raw sum and the per-byte mean.
They disagree when choices differ in length (a 1-byte choice and a 2-byte choice
are not comparable as sums), so the headline is the per-byte mean and the raw sum
rides next to it. A task report with only one of them is not a number — the
protocol it came from is missing.

**Honesty rules, the same three as the trainer's**: every printed number carries
its `n` and its bytes scored; an item whose `C + c_i` exceeds `max_seq_len` is
DROPPED with the `bench_dropped` counter, never truncated (a truncated choice is
a different item); a task that cannot be loaded is LOUD naming its URL — never
silently skipped, which would turn "the model cannot do ARC" into "the harness
did not run ARC".
"""

from .runner import TaskReport, run_bench, run_task
from .score import ChoiceScore, model_logits, score_choices
from .tasks import SYNTH_N, TASKS, MCItem, download, load_task, task_names

__all__ = [
    "SYNTH_N",
    "TASKS",
    "ChoiceScore",
    "MCItem",
    "TaskReport",
    "download",
    "load_task",
    "model_logits",
    "run_bench",
    "run_task",
    "score_choices",
    "task_names",
]
