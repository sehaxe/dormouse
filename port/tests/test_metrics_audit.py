"""The 2026-11-07 metric audit's gates: BPB, bytes, counters and the FLOP ledger.

Each test pins ONE of the instruments every architecture comparison rests on.
They are designed to fail when the instrument is wrong, not when a number moves
for an honest reason:

a. `bpb_fin`/`bpb_mean` are `sum(CE nats) / (ln2 * scored bytes)` recomputed
   from the head's own logits (not from the trainer's reduction);
b. a masked eval divides by the positions it actually selected, so a variable
   mask count per batch cannot move the denominator away from the numerator;
c. `bytes_seen` is `step * batch * seq_len * grad_accum` on a single stream and
   on the staged mixture, and a resume continues it instead of re-counting;
d. the arm counters a reader can predict from the config equal their formula;
e. the analytic transformer ledger equals the measured matmul bill PLUS the
   attention core, at two `T`s (the FlopCounterMode-on-CPU blind spot is the
   T-dependent term; a ledger that omits it is wrong).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_run

from dormouse.instrument import Counters
from dormouse.model import DormouseModel
from dormouse.train.data import MaskedByteStream, eval_batches_for, open_streams
from dormouse.train.loop import _eval_loss, eval_row, kda_field, train

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LN2 = math.log(2.0)


# ---------------------------------------------------------------- helpers


def _load_ledger():
    name = "mod_ledger_audit"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "mod_ledger.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _logits_of(model: torch.nn.Module, x, y, mask=None, **kw):
    """Every head call's logits for one forward, in call order (T + final)."""
    calls: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda _m, _a, out: calls.append(out.detach().float())
    )
    try:
        model(x, y, loss_mask=mask, pure=True, breakdown=True, **kw)
    finally:
        hook.remove()
    return calls


def _ce_rows(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(logits, dim=-1).gather(1, y.reshape(-1, 1)).reshape(-1)


def _recompute_from_logits(model, stream, batches, *, masked: bool, max_iter: int):
    """Independent `sum(nats)/scored` for both readouts, from the head's logits."""
    stream.rewind()
    model.eval()
    fin_nats = 0.0
    mean_nats = 0.0
    scored = 0
    masks: list[int] = []
    with torch.no_grad():
        for _ in range(batches):
            if masked:
                x, y, mask = stream.next_batch_masked()
            else:
                x, y = stream.next_batch()
                mask = None
            rows = _logits_of(model, x, y, mask)
            assert len(rows) == max_iter + 1, (len(rows), max_iter + 1)
            selected = int(mask.sum()) if mask is not None else int(x.numel())
            masks.append(selected)
            scored += selected
            w = None if mask is None else mask.reshape(-1, 1).to(torch.float32)
            fin = _ce_rows(rows[-1], y)
            per = [_ce_rows(row, y) for row in rows[:-1]]
            if w is not None:
                fin = fin.reshape(-1, 1)
                per = [p.reshape(-1, 1) for p in per]
                fin_nats += float((fin * w).sum())
                mean_nats += float(((sum(per) + fin) * w).sum() / (max_iter + 1))
            else:
                fin_nats += float(fin.sum())
                mean_nats += float((sum(per) + fin).sum() / (max_iter + 1))
    return mean_nats / scored, fin_nats / scored, scored, masks


# ---------------------------------------------------------------- (a) BPB


def test_bpb_fin_and_mean_are_sum_nats_over_ln2_bytes(tmp_path):
    run = tiny_run(tmp_path, eval=True, eval_batches=2)
    run = dataclasses.replace(
        run, model=dataclasses.replace(run.model, max_iter=3)
    )
    model = DormouseModel(run.model)
    _train, stream = open_streams(run)
    assert stream is not None
    batches = eval_batches_for(run.train)

    ev, scored, fin, per = _eval_loss(model, stream, batches, None, None)
    mean_indep, fin_indep, scored_indep, _ = _recompute_from_logits(
        model, stream, batches, masked=False, max_iter=3
    )
    assert scored == scored_indep == batches * BATCH * SEQ
    assert ev == pytest.approx(mean_indep, abs=1e-6)
    assert fin == pytest.approx(fin_indep, abs=1e-6)

    row = eval_row(step=1, ev=ev, fin=fin, scored=scored, cfg=run.train, offset=0)
    assert row["bpb_fin"] == pytest.approx(fin_indep / LN2, abs=1e-6)
    assert row["bpb_mean"] == pytest.approx(mean_indep / LN2, abs=1e-6)
    assert row["bpb"] == row["bpb_fin"]
    # The depth mean is the average of the T+1 readouts, not a second reading
    # of the final one.
    assert len(per) == 3
    assert (sum(per) + fin) / 4.0 == pytest.approx(ev, abs=1e-6)


def test_a_masked_eval_divides_by_the_positions_it_selected(tmp_path):
    """Variable mask counts per batch are the case mean-of-means gets wrong."""
    run = tiny_run(tmp_path, eval=True, eval_batches=2, doc_masking=True)
    data = Path(run.train.data)
    n = 4096
    data.write_bytes(bytes((i * 7 + 3) % 256 for i in range(n)))
    Path(str(data) + ".docs.json").write_text(
        json.dumps({"version": 1, "size": n, "starts": [0, 200, 700, 1500, 2500, 3300]})
    )
    run = dataclasses.replace(
        run, model=dataclasses.replace(run.model, max_iter=3)
    )
    model = DormouseModel(run.model)
    _train, stream = open_streams(run)
    assert isinstance(stream, MaskedByteStream)
    batches = eval_batches_for(run.train)

    ev, scored, fin, _per = _eval_loss(model, stream, batches, None, None)
    mean_indep, fin_indep, scored_indep, masks = _recompute_from_logits(
        model, stream, batches, masked=True, max_iter=3
    )
    assert len(set(masks)) > 1, f"the fixture must vary the per-batch mask: {masks}"
    assert scored == scored_indep == sum(masks)
    assert ev == pytest.approx(mean_indep, abs=1e-6)
    assert fin == pytest.approx(fin_indep, abs=1e-6)

    row = eval_row(step=1, ev=ev, fin=fin, scored=scored, cfg=run.train, offset=0)
    assert row["bpb_fin"] == pytest.approx(fin_indep / LN2, abs=1e-6)
    assert row["bpb_mean"] == pytest.approx(mean_indep / LN2, abs=1e-6)


# ---------------------------------------------------------------- (b) bytes


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_bytes_seen_is_the_optimizer_step_product_and_resume_continues_it(tmp_path):
    first = tmp_path / "first.jsonl"
    run = tiny_run(
        tmp_path,
        steps=4,
        log_every=1,
        grad_accum=3,
        ckpt_every=2,
        json_log=str(first),
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)
    rows = _jsonl(first)
    logs = [r for r in rows if r["event"] == "log"]
    assert [r["step"] for r in logs] == [1, 2, 3, 4]
    want = BATCH * SEQ * 3
    for row in logs:
        assert row["bytes_seen"] == row["step"] * want
    assert logs[-1]["bytes_seen"] == 4 * want
    assert next(r for r in rows if r["event"] == "run")["bytes_per_step"] == want

    # Resume at the last checkpoint: the first resumed log continues at step 5
    # and does not re-count 1..4.
    second = tmp_path / "second.jsonl"
    run2 = dataclasses.replace(
        run,
        train=dataclasses.replace(
            run.train, steps=6, json_log=str(second)
        ),
    )
    train(DormouseModel(run2.model), run2, resume=True, log=lambda *_: None)
    rows2 = _jsonl(second)
    resumed = [r for r in rows2 if r["event"] == "log"]
    assert [r["step"] for r in resumed] == [5, 6]
    assert [r["bytes_seen"] for r in resumed] == [5 * want, 6 * want]
    assert next(r for r in rows2 if r["event"] == "run")["resumed"] == 4


def test_a_staged_mixture_counts_the_same_product(tmp_path):
    corpus = Path(tiny_run(tmp_path).train.data)
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "\n".join(
            [
                "max_epochs = 4.0",
                "seed = 1",
                "[sources]",
                f'tiny = "{corpus}"',
                "[[stage]]",
                "until_frac = 1.0",
                "[stage.weights]",
                "tiny = 1.0",
                "",
            ]
        )
    )
    log = tmp_path / "mix.jsonl"
    run = tiny_run(
        tmp_path,
        steps=3,
        log_every=1,
        grad_accum=2,
        mix_plan=str(plan),
        json_log=str(log),
    )
    lines: list[str] = []
    train(DormouseModel(run.model), run, log=lines.append)
    rows = _jsonl(log)
    logs = [r for r in rows if r["event"] == "log"]
    want = BATCH * SEQ * 2
    assert [r["step"] for r in logs] == [1, 2, 3]
    assert all(r["bytes_seen"] == r["step"] * want for r in logs)
    assert any("mix_plan=" in line and "steps=6 micro-batches" in line for line in lines), (
        "the mixture must be built for steps x grad_accum micro-batches"
    )


def test_a_staged_mixture_resume_continues_the_byte_count(tmp_path):
    corpus = Path(tiny_run(tmp_path).train.data)
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "\n".join(
            [
                "max_epochs = 4.0",
                "seed = 1",
                "[sources]",
                f'tiny = "{corpus}"',
                "[[stage]]",
                "until_frac = 1.0",
                "[stage.weights]",
                "tiny = 1.0",
                "",
            ]
        )
    )
    first = tmp_path / "mix_first.jsonl"
    run = tiny_run(
        tmp_path,
        steps=4,
        log_every=1,
        grad_accum=2,
        ckpt_every=2,
        mix_plan=str(plan),
        json_log=str(first),
    )
    train(DormouseModel(run.model), run, log=lambda *_: None)

    second = tmp_path / "mix_second.jsonl"
    run2 = dataclasses.replace(
        run,
        train=dataclasses.replace(run.train, steps=6, json_log=str(second)),
    )
    train(DormouseModel(run2.model), run2, resume=True, log=lambda *_: None)
    logs = [r for r in _jsonl(second) if r["event"] == "log"]
    want = BATCH * SEQ * 2
    assert [r["step"] for r in logs] == [5, 6]
    assert [r["bytes_seen"] for r in logs] == [5 * want, 6 * want]


# ------------------------------------------------------------- (c) counters


def _run_once(cfg, ids, targets):
    model = DormouseModel(cfg)
    counters = Counters()
    with torch.no_grad():
        model(ids, targets, counters=counters)
    counters.drain()
    return model, counters


def test_the_predictable_arm_counters_equal_their_config_formula():
    ids = torch.randint(0, 256, (BATCH, SEQ))
    targets = (ids + 1) % 256
    cfg = tiny_cfg(max_iter=3)
    model, c = _run_once(cfg, ids, targets)
    orders = len(model.loop.engram.orders) if model.loop.engram is not None else 0
    assert c["depth"] == c["iter"] == c["kda_sdpa"] == c["engram_arms"] == cfg.max_iter
    assert c["engram_rows"] == BATCH * SEQ * orders * cfg.max_iter
    assert c["moe_tokens"] == BATCH * SEQ * cfg.max_iter
    assert kda_field(c) == f"kda=sdpa,n={cfg.max_iter}"

    # Those are the same numbers for a full-attention arm, under a different
    # name: `kda=full` means the KDA arm did NOT run (the field is an arm label).
    _m, full = _run_once(tiny_cfg(max_iter=3, attn_kind="full"), ids, targets)
    assert full["attn_full_forward"] == cfg.max_iter
    assert "kda_sdpa" not in full
    assert kda_field(full) == f"kda=full,n={cfg.max_iter}"

    # Composition counters: one per window/forward, `mhc` once per iteration.
    _m, both = _run_once(
        tiny_cfg(max_iter=3, use_mhc=True, use_attnres=True), ids, targets
    )
    assert both["mhc"] == cfg.max_iter
    assert both["mhc_attnres_forward"] == 1

    # CSA: one fine call and one coarse call per iteration.
    _m, csa = _run_once(
        tiny_cfg(
            max_iter=3,
            use_csa=True,
            csa_chunk=8,
            csa_kv_dim=16,
            csa_topk=4,
            csa_coarse_chunk=16,
            csa_coarse_kv_dim=8,
        ),
        ids,
        targets,
    )
    assert csa["csa_forward"] == cfg.max_iter
    assert csa["csa_coarse_forward"] == cfg.max_iter

    # share_u: one SparseMoE forward per iteration.
    _m, shared = _run_once(
        tiny_cfg(max_iter=3, share_u=True, share_u_mode="plane", share_u_rank=16),
        ids,
        targets,
    )
    assert shared["share_u_forward"] == cfg.max_iter

    # LayerRoPE: P prelude + T iterations + C coda + 1 final read.
    _m, lr = _run_once(
        tiny_cfg(max_iter=3, layerrope=True, prelude_layers=2, coda_layers=1),
        ids,
        targets,
    )
    assert lr["layerrope_forward"] == 2 + cfg.max_iter + 1 + 1


def test_the_transformer_stack_counts_one_block_per_layer():
    ids = torch.randint(0, 256, (BATCH, SEQ))
    cfg = tiny_cfg(
        block_form="transformer",
        max_iter=1,
        n_layers=4,
        d_ffn=128,
        use_kda=False,
        use_csa=False,
        use_mhc=False,
        use_attnres=False,
        use_engram=False,
        layerrope=False,
        n_experts=1,
        moe_topk=0,
    )
    _model, c = _run_once(cfg, ids, (ids + 1) % 256)
    assert c["transformer_block"] == cfg.n_layers
    assert "depth" not in c and "iter" not in c


def test_moe_expert_calls_is_data_dependent_and_only_on_the_naive_path():
    """The one listed counter that is NOT a function of the config."""
    model = DormouseModel(tiny_cfg(max_iter=3, n_experts=4, moe_topk=2))
    moe = model.loop.moe
    ids = torch.randint(0, 256, (BATCH, SEQ))
    model.eval()
    with torch.no_grad():
        _, _, c1 = model(ids, counters=Counters())
        _, _, c2 = model((ids + 1) % 256, counters=Counters())
    assert c1["moe_naive_dispatch"] == c2["moe_naive_dispatch"] == 3
    # Two different batches, the same config: the count is the unique-expert
    # set the router chose, so it can move. It is not `steps x iterations`.
    assert "moe_expert_calls" in c1 and "moe_expert_calls" in c2
    assert c1["moe_expert_calls"] != c1["moe_tokens"]
    # The name is also a token count, not a forward count: the shared expert
    # runs once per MoE forward, and this field says 384 at B=2, T=64, T=3.
    assert c1["moe_shared_forwards"] == BATCH * SEQ * 3
    assert c1["moe_shared_forwards"] != 3

    # The production `_padded_dispatch` does not bump `moe_expert_calls` at
    # all: on CUDA the field is absent while the expert pass ran.
    x = torch.randn(BATCH * SEQ, model.cfg.d_model)
    index = torch.zeros(BATCH * SEQ, 2, dtype=torch.long)
    index[:, 1] = 1
    weight = torch.full((BATCH * SEQ, 2), 0.5)
    order, keep, _dropped = moe._permute(index)
    padded = Counters()
    moe._padded_dispatch(x, weight, index, order, keep, padded)
    assert "moe_expert_calls" not in padded
    naive = Counters()
    moe._naive_dispatch(x, weight, index, order, keep, naive)
    assert naive["moe_expert_calls"] == 2


# ------------------------------------------------------------- (d) the ledger


def test_the_transformer_ledger_includes_the_t_dependent_attention_core():
    """FlopCounterMode (CPU) counts matmuls, not SDPA; the ledger must add it."""
    led = _load_ledger()
    cfg = tiny_cfg(
        block_form="transformer",
        max_iter=1,
        n_layers=2,
        d_ffn=64,
        use_kda=False,
        use_csa=False,
        use_mhc=False,
        use_attnres=False,
        use_engram=False,
        layerrope=False,
        n_experts=1,
        moe_topk=0,
        bf16=False,
    )
    model = DormouseModel(cfg)
    layers, d = cfg.n_layers, cfg.d_model
    for seq in (32, 64):
        x = torch.randint(0, 256, (1, seq))
        y = torch.randint(0, 256, (1, seq))
        model.zero_grad(set_to_none=True)
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc:
            _logits, loss, _ = model(x, y)
            loss.backward()
        measured = fc.get_total_flops() / seq / 1e6  # MFLOP/byte, fwd+bwd
        led_seq = led.ledger(cfg, 1.0, seq_len=seq)
        core_fwd = 4.0 * layers * seq * d / 1e6
        body_head = 3.0 * (led_seq.routed + led_seq.fixed - core_fwd)
        assert measured == pytest.approx(body_head, rel=0.01), (
            seq,
            measured,
            body_head,
        )
        assert led_seq.total == pytest.approx(measured + 3.0 * core_fwd, rel=0.01), (
            seq,
            led_seq.total,
            measured + 3.0 * core_fwd,
        )
        assert led_seq.total > measured * 1.01, (
            "the core must be a positive, T-dependent term the counter omits"
        )
