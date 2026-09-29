#!/usr/bin/env python3
"""Fixture generator for the DSpark / DeepSpec oracle.

Reads the OFFICIAL configuration
(``deepseek-ai/DeepSpec@main:config/dspark/dspark_qwen3_4b.py``, vendored next
to this file) plus the DeepSpec source files that DEFINE what its fields mean,
and emits a flat fixture that ``src/dspark_oracle.rs`` pins our constants
against.

    python3 gen_dspark_oracle.py dspark_qwen3_4b.py dspark_loss.py \
        dspark_markov_head.py dspark_common.py ../fixtures/dspark_oracle.txt

The config alone is not enough, and that is the point: three of the four
questions this fixture exists to answer cannot be answered by reading a dict.

* What is ``block_size``? Not a stride. ``deepspec/modeling/dspark/common.py:19``
  calls it "number of draft positions per anchor", and its uses agree: the
  labels are ``anchor + arange(1, block_size + 1)`` (qwen3/modeling.py:432) and
  the position ids are ``anchor + arange(block_size)`` (common.py:257). So
  ``block_size`` is the LENGTH OF A DRAFT BLOCK - which is dormouse's
  ``dspark_k``, not ``dspark_stride``. The fixture records the evidence LINES,
  not just the conclusion, so a reader can check the read.

* Is the confidence head Markov-conditioned? ``confidence_head_with_markov=True``
  says yes, and ``loss.py:157`` feeds ``outputs.confidence_pred`` - a LOGIT -
  to ``binary_cross_entropy_with_logits``.

* Which Markov head? ``markov_head_type='vanilla'`` selects ``VanillaMarkov``
  in ``markov_head.py:294``, not ``RNNHead``.

Values are read with ``ast.literal_eval`` on the config's keyword arguments, so
this generator CANNOT execute the code it is quoting: the config imports
``deepspec.trainer`` at module scope, and an AST read needs nothing.
"""
from __future__ import annotations

import ast
import os
import re
import sys

WANTED = [
    "block_size", "num_draft_layers", "target_layer_ids", "mask_token_id",
    "num_anchors", "markov_rank", "markov_head_type", "confidence_head_alpha",
    "confidence_head_with_markov", "loss_decay_gamma", "ce_loss_alpha",
    "l1_loss_alpha",
]

EVIDENCE = {
    "loss": {
        "decay": r"decay_weights\s*=\s*torch\.exp",
        "bce": r"binary_cross_entropy_with_logits",
        "accept": r"accept_rate_3d\s*=\s*1\.0\s*-\s*0\.5",
        "mix": r"ce_loss_alpha \* ce_loss",
    },
    "markov": {
        "vanilla": r'markov_head_type == "vanilla"',
        "rnn": r"class RNNHead",
        "gated": r'markov_head_type == "gated"',
        "joint": r"joint_proj = nn\.Linear",
    },
    "common": {
        "block_size_doc": r"block_size: number of draft positions",
        "position_ids": r"offsets = torch\.arange\(block_size",
    },
}


def _literal(node):
    """`ast.literal_eval` for values, a marker for names like QWEN_3_4B."""
    if isinstance(node, ast.Name):
        return f"<{node.id}>"
    return ast.literal_eval(node)


def read_config(path: str) -> dict:
    src = open(path).read()
    model = None
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "model" for t in node.targets
        )):
            continue
        call = node.value
        if isinstance(call, ast.Dict):
            model = {k.value: _literal(v) for k, v in zip(call.keys, call.values)}
        elif isinstance(call, ast.Call) and getattr(call.func, "id", None) == "dict":
            model = {kw.arg: _literal(kw.value) for kw in call.keywords}
    assert model is not None, "no `model = dict(...)` in the official config"
    exp = re.search(r'exp_name\s*=\s*"([^"]+)"', src)
    return {"fields": {k: model[k] for k in WANTED if k in model},
            "exp_name": exp.group(1) if exp else "",
            "file": os.path.basename(path)}


def read_evidence(paths: dict) -> dict:
    """Pull the lines that DEFINE the fields, with their line numbers."""
    out = {}
    for key, (path, patterns) in paths.items():
        found = {}
        for i, line in enumerate(open(path).read().splitlines(), 1):
            for label, pat in patterns.items():
                if label not in found and re.search(pat, line):
                    found[label] = (i, line.strip())
        out[key] = found
    return out


def quote(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main() -> int:
    cfg_path, loss, markov, common = sys.argv[1:5]
    out_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "dspark_oracle.txt"
    )
    cfg = read_config(cfg_path)
    ev = read_evidence({"loss": (loss, EVIDENCE["loss"]),
                        "markov": (markov, EVIDENCE["markov"]),
                        "common": (common, EVIDENCE["common"])})

    lines = [
        "# dspark / deepspec oracle fixture - GENERATED, do not hand-edit.",
        "# config: deepseek-ai/DeepSpec@main config/dspark/dspark_qwen3_4b.py",
        f"# config file: {cfg['file']}, exp_name {cfg['exp_name']}",
        "# regenerate: python3 gen_dspark_oracle.py dspark_qwen3_4b.py \\",
        "#              dspark_loss.py dspark_markov_head.py dspark_common.py \\",
        "#              ../fixtures/dspark_oracle.txt",
        "# values are read with ast.literal_eval, so this file never executes",
        "# the code it quotes.",
        "",
    ]
    for k, v in sorted(cfg["fields"].items()):
        if isinstance(v, (list, tuple)):
            lines.append(f"config.{k}: " + ",".join(repr(x) for x in v))
        else:
            lines.append(f"config.{k}: {v!r}")
    for src, found in sorted(ev.items()):
        for label, (lineno, text) in sorted(found.items()):
            lines.append(f"evidence.{src}.{label}.line: {lineno}")
            lines.append(f"evidence.{src}.{label}.text: {quote(text)}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
