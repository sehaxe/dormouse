"""`full_graph` on the LIVE m400b recipe: capture, counter, refusals, parity.

The recipe is `configs/m400b.toml`, copied from the running checkpoint's
`m400b.config.toml` (2026-11-05): 199M params, d 1536, E 512, top-k 2,
**dropless** (`moe_capacity=0`) with `moe_kernel=triton`, `moe_lb_mode=prop`,
`moe_combine=fused`, `moe_bank_epoch=true`, paper LayerRoPE, CSA, mHC, AttnRes,
`activation_recompute=none`.

What this file pins, one gate per claim:

1. **The recipe is legal for `full_graph`.** The old refusal set said
   "dropless" and "triton" and "LayerRoPE" without looking at the arm each
   name actually selects; the current recipe names all three and the dropped
   robots are the ones with static rows / fixed depth. `full_graph_refusal`
   must return `None` for the recipe and a LOUD `Escape:` for each arm that
   still cannot be captured.
2. **A capture is COUNTED.** `FullGraph.capture` increments
   `<arm>_captured`; a fallback increments `<arm>_fallback` and
   `opt_graph_fallback`. A run whose log says `full_graph` but whose counters
   carry neither has silently stayed eager — the defect class this project
   paid for twice (`opt_graph` "+0" for a month; `full_graph` never captured
   on the first try).
3. **The off path did not move.** `full_graph=false` is still the pinned
   5-step sha (the same constants `tests/test_full_graph.py` holds); the
   capture work touched no eager arithmetic.
4. **The refusals that remain are LOUD with an `Escape:`.** Recompute, ragged
   MoE arms and the fused route/permute are refused by name, not discovered
   mid-capture.
5. **The probe names an op.** `scripts/probe_capture.py` prints the last
   dispatched ops and names the first known host-read op; a failure that only
   prints "capture failed" is not a probe.
6. **GPU:** the recipe captures, replays 20 steps and reports its first-replay
   parity (1e-6) and 20-step drift. The 20-step drift is NOT bound at 1e-6:
   mHC's near-zero gains make Muon normalise float chaos, a measured property
   of the arm, not a wiring error (`.bulba/memory.md`, 2026-10-24).

The CPU half runs with `CUDA_VISIBLE_DEVICES=""`; the GPU gates skip with a
named reason when the card is owned (AGENTS 1.5).
"""

from __future__ import annotations

import dataclasses
import gc
import time

import pytest
import torch

from dormouse.config import TrainConfig, load_run, validate_train
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE
from dormouse.train import train
from dormouse.train.data import ByteStream
from dormouse.train.graph import FullGraph, full_graph_refusal

#: The train half the recipe's checkpoint carries, reduced to the keys a
#: capture gate can move. Everything else is the schema default the live run
#: resolved to (`checkpoints/m400b/m400b.config.toml`).
M400B_TRAIN = dict(
    batch=8,
    seq_len=1024,
    steps=26,
    log_every=1,
    eval=False,
    eval_every=10**9,
    ckpt_every=10**9,
    json_log="",
    lr=5e-4,
    wd=0.01,
    grad_clip=1.0,
    muon_state_dtype="fp32",
    ns_dtype="fp32",
    muon_chunk_elems=1 << 26,
    full_graph=True,
)


def _recipe(**over):
    ov = dict(M400B_TRAIN)
    ov.update(over)
    return load_run("m400b", train_overrides=ov)


# --- 1. the recipe is legal; the ragged arms are still refused -------------


def test_the_m400b_recipe_is_legal_for_full_graph():
    run = _recipe()
    why = full_graph_refusal(run.train, run.model)
    assert why is None, why


@pytest.mark.parametrize(
    "model_over, needle",
    [
        (dict(moe_capacity=1.0), "RAGGED"),  # triton at a capacity: order[keep]
        (dict(moe_capacity=1.0, moe_kernel="naive"), "unique"),
        (dict(moe_capacity=1.0, moe_kernel="bucketed"), "counts.tolist"),
        (dict(moe_kernel="batched"), "counts.max().item"),  # dropless
        (dict(moe_kernel="naive"), "unique"),
        (dict(moe_kernel="padded"), "fall through"),  # dropless padded
        (dict(moe_permute="count"), "tolist"),
        (dict(moe_route_fused=True), "reference fallback"),
        (dict(moe_gather="kernel"), "gather-fused"),
    ],
)
def test_the_recipe_refuses_each_non_capturable_arm(model_over, needle):
    run = _recipe()
    why = full_graph_refusal(run.train, dataclasses.replace(run.model, **model_over))
    assert why is not None, f"{model_over} must be refused"
    assert needle in why, why
    assert "Escape:" in why, why


def test_recompute_is_refused_loudly_with_an_escape():
    for over, needle in (
        (dict(activation_recompute="selective"), "a capture cannot see a recompute"),
        (dict(activation_recompute="full"), "a capture cannot see a recompute"),
        (dict(iter_checkpoint=True), "ONE launch sequence"),
    ):
        with pytest.raises(LoudError, match=needle):
            validate_train(TrainConfig(**over, full_graph=True))
        try:
            validate_train(TrainConfig(**over, full_graph=True))
        except LoudError as e:
            assert "Escape:" in str(e), str(e)
            assert "full_graph" in str(e), str(e)


def test_the_prop_controller_is_device_only(monkeypatch):
    """`moe_lb_mode=prop` -- the recipe's controller -- must not read the host.

    The OLD `sign` path's `lfb_counts.mean()` was the capture blocker; `prop`
    runs `load.mean()`, the normalised deviation and `clamp_` on the device.
    The gate traps every host-read method on `Tensor` and calls the controller
    with a CPU index: a `.item()`/`.tolist()`/`bool()` reintroduced here fails.
    """

    def boom(*_a, **_k):
        raise AssertionError("host read in the prop controller")

    cfg = dataclasses.replace(
        load_run("tiny", train_overrides=dict(batch=8, seq_len=64)).model,
        n_experts=8,
        moe_topk=2,
        rank=8,
        d_model=32,
        d_ffn=64,
        bf16=False,
        moe_lb_mode="prop",
        moe_lb_gamma=0.01,
        moe_lb_clamp=2.0,
    )
    moe = SparseMoE(cfg, device="cpu")
    moe.train()
    index = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)
    saved = {n: getattr(torch.Tensor, n) for n in ("item", "tolist", "__float__", "__bool__")}
    try:
        for name in saved:
            monkeypatch.setattr(torch.Tensor, name, boom)
        moe._lfb_step(index, Counters())
    finally:
        for name, fn in saved.items():
            monkeypatch.setattr(torch.Tensor, name, fn)
    assert float(moe.lfb_counts.sum()) > 0.0
    assert torch.isfinite(moe.lfb_bias).all()


# --- 2. the counter: present on capture, loud on fallback ------------------


class _FakeAdamW:
    param_groups: list = []


class _FakeOpt:
    """The minimum `FullGraph` needs, on CPU: no groups, no parameters."""

    def __init__(self) -> None:
        self.adamw = _FakeAdamW()
        self.muon = None
        self.model = torch.nn.Module()
        self.steps = 0
        self.counters = Counters()


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1

    def reset(self) -> None:
        pass


class _FakeCtx:
    def __init__(self, graph: _FakeGraph) -> None:
        self.graph = graph

    def __enter__(self) -> _FakeGraph:
        return self.graph

    def __exit__(self, *_exc) -> bool:
        return False


def _cpu_capture(monkeypatch, body, *, log=None, pre_capture=None) -> FullGraph:
    """A `FullGraph` whose CUDA surface is faked so the COUNTING is testable."""

    def empty_cache():
        return None

    def synchronize():
        return None

    def graph_ctx(graph, **_kwargs):
        return _FakeCtx(graph)

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", graph_ctx)
    monkeypatch.setattr(
        torch.autograd.graph, "set_override_stale_capture_stream", lambda _v: None
    )
    opt = _FakeOpt()
    graph = FullGraph(
        opt,
        body,
        batch=2,
        seq_len=4,
        device="cpu",
        warmup_steps=1,
        counters=opt.counters,
        log=log,
        pre_capture=pre_capture,
    )
    graph._refusals = lambda: None  # type: ignore[method-assign]
    return graph


def test_a_successful_capture_counts_full_graph_captured(monkeypatch):
    logs: list[str] = []
    body_calls = {"n": 0}

    def body() -> None:
        body_calls["n"] += 1

    graph = _cpu_capture(monkeypatch, body, log=logs.append)
    graph.step()  # warmup
    graph.step()  # capture
    assert graph._graph is not None
    assert graph.reason is None
    assert body_calls["n"] == 2, "one warmup body + the capture's one execution"
    assert graph.counters["full_graph_captured"] == 1
    assert "full_graph_fallback" not in graph.counters
    assert any("full_graph_captured=1" in line for line in logs), logs
    graph.step()  # replay
    assert body_calls["n"] == 2, "a replay must not run the Python body"


def test_the_capture_does_not_double_count_the_optimizer_counters(monkeypatch):
    """The body runs once at capture; the replay re-adds it once, not twice.

    `opt.counters` is merged by the trainer AFTER `full.step()` returns, so the
    snapshot taken inside `capture()` must clear the body's own counts before
    `_host_bookkeeping` re-adds them. Without the clear the capture step counts
    every optimizer arm twice; the first replay then reads a doubled total.
    """
    graph = _cpu_capture(monkeypatch, lambda: None)
    graph.opt.counters["ns_triton"] = 1
    graph.step()  # warmup
    graph.step()  # capture: body would add one more
    assert graph.counters["ns_triton"] == 1, dict(graph.counters)
    graph.step()  # replay
    assert graph.counters["ns_triton"] == 2, dict(graph.counters)


def test_a_forced_fallback_counts_and_never_claims_capture(monkeypatch):
    logs: list[str] = []
    calls = {"n": 0}

    def body() -> None:
        calls["n"] += 1
        if calls["n"] == 2:  # the capture's one execution
            raise RuntimeError("scratch: force the fallback")

    graph = _cpu_capture(monkeypatch, body, log=logs.append)
    graph.step()  # warmup
    graph.step()  # capture raises -> COUNTED fallback -> eager body
    assert graph._graph is None
    assert graph.reason is not None and "force the fallback" in graph.reason
    assert graph.counters["opt_graph_fallback"] == 1
    assert graph.counters["full_graph_fallback"] == 1
    assert "full_graph_captured" not in graph.counters
    assert any("capture failed" in line for line in logs), logs


def test_the_pre_capture_hook_runs_before_the_body(monkeypatch):
    """The bank/QAT caches must be dropped BEFORE the recorded forward."""
    order: list[str] = []

    def pre() -> None:
        order.append("pre")

    def body() -> None:
        order.append("body")

    graph = _cpu_capture(monkeypatch, body, pre_capture=pre)
    graph.step()
    graph.step()
    assert order == ["body", "pre", "body"], order


# --- 3. the off path did not move ------------------------------------------


def test_the_default_path_is_still_the_pinned_sha(tmp_path):
    from test_full_graph import _PRE_CHANGE_LOGITS_SHA, _PRE_CHANGE_PARAMS_SHA, _five_steps

    params, logits = _five_steps(tmp_path)
    assert params == _PRE_CHANGE_PARAMS_SHA, "full_graph=false moved the default parameters"
    assert logits == _PRE_CHANGE_LOGITS_SHA, "full_graph=false moved the default logits"


# --- 5. the probe can name the op ------------------------------------------


def test_the_probe_names_the_first_known_host_read():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_capture

    ops = [
        "aten.addmm.default  @ model/kda.py:100",
        "aten.index.Tensor  @ model/experts.py:1354",
        "aten.nonzero.default  @ model/experts.py:1354",
    ]
    culprit = probe_capture.name_the_culprit(ops)
    assert culprit is not None and "nonzero" in culprit, culprit
    assert probe_capture.name_the_culprit(ops[:2]) is None


# --- 6. GPU: the recipe captures and reports parity ------------------------

CARD_MIN_MIB = 4096


def _card_reason(min_free_mib: int = CARD_MIN_MIB) -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device (CUDA_VISIBLE_DEVICES is empty)"
    try:
        import subprocess

        busy = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate precondition, not a run
        return f"cannot check the card ({e})"
    if busy:
        return f"a compute process holds the card:\n{busy}"
    free, _total = torch.cuda.mem_get_info()
    if free < min_free_mib << 20:
        return f"only {free / 2**20:.0f} MiB VRAM free, this gate needs {min_free_mib}"
    return None


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # noqa: SLF001
    if clear is not None:
        clear()
        torch.cuda.empty_cache()


def _corpus(tmp_path, n: int = 8 << 20):
    p = tmp_path / "bytes.bin"
    p.write_bytes(bytes(range(256)) * (n // 256))
    return p


def _gpu_recipe(steps: int, **over):
    ov = dict(M400B_TRAIN)
    ov.update(steps=steps)
    ov.update(over)
    return load_run("m400b", train_overrides=ov)


def _run_gpu(run, data):
    """One `train()` run on a fresh model; `(report, seconds, logs)`."""
    torch.manual_seed(0)
    model = DormouseModel(run.model, device="cuda")
    logs: list[str] = []
    t0 = time.perf_counter()
    report = train(
        model,
        run,
        stream=ByteStream(data, run.train.batch, run.train.seq_len),
        device="cuda",
        log=logs.append,
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return report, seconds, logs


@pytest.mark.skipif(
    _card_reason() is not None, reason=_card_reason() or "GPU gate"
)
def test_the_m400b_recipe_captures_and_replays_twenty_steps(tmp_path):
    """26 steps: 5 eager, ONE capture, 20 replays, no COUNTED fallback."""
    run = _gpu_recipe(26)
    data = _corpus(tmp_path)
    report, seconds, logs = _run_gpu(run, data)
    try:
        counters = report["counters"]
        assert counters.get("full_graph_captured") == 1, (
            "the recipe did not capture; counters="
            f"{dict(counters)} why={getattr(counters, 'why', None)}"
        )
        assert "full_graph_fallback" not in counters, getattr(counters, "why", None)
        fallback = [line for line in logs if "capture failed" in line]
        assert not fallback, fallback
        losses = [line.split("loss=")[1].split()[0] for line in logs if line.startswith("step=")]
        assert len(set(losses[-20:])) >= 15, "the replay is not advancing the data"
        print(
            f"[m400b capture] {seconds * 1e3 / run.train.steps:.1f} ms/step incl. "
            f"capture; counters={dict(counters)}"
        )
    finally:
        _free_gpu()


@pytest.mark.skipif(
    _card_reason() is not None, reason=_card_reason() or "GPU gate"
)
def test_the_first_replay_is_the_arms_own_eager_program(tmp_path, monkeypatch):
    """Capture vs the same arm with `capture` disabled: <=1e-6 after ONE replay.

    The bound is the measured fp32 ulp the recording costs; a wrong LR, a
    stale bank (the `_banks` cache-hit trap) or a frozen optimizer state is
    O(1), not 1e-6.
    """
    data = _corpus(tmp_path)
    run = _gpu_recipe(6)

    # `train()` does not return the model; build it here so the state dict can
    # be compared, one process per arm (the existing pattern).
    def six(capture: bool):
        torch.manual_seed(0)
        model = DormouseModel(run.model, device="cuda")
        if not capture:
            monkeypatch.setattr(
                FullGraph,
                "capture",
                lambda self: (_ for _ in ()).throw(RuntimeError("capture disabled")),
            )
        report = train(
            model,
            run,
            stream=ByteStream(data, run.train.batch, run.train.seq_len),
            device="cuda",
            log=lambda *_a, **_k: None,
        )
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        del model
        _free_gpu()
        return report, sd

    rep_g, sd_g = six(True)
    assert rep_g["counters"].get("full_graph_captured") == 1
    rep_e, sd_e = six(False)
    worst = max(((sd_g[k] - sd_e[k]).abs().max().item(), k) for k in sd_g)
    print(f"[m400b first replay] worst |delta| = {worst[0]:.3e} @ {worst[1]}")
    assert worst[0] <= 1e-6, f"the capture moved the arithmetic: {worst}"
    assert rep_e["counters"].get("full_graph_fallback") == 1


@pytest.mark.skipif(
    _card_reason() is not None, reason=_card_reason() or "GPU gate"
)
def test_the_twenty_step_parity_is_reported_not_asserted(tmp_path, monkeypatch):
    """The drift over 20 replays is a READING, not a 1e-6 bound.

    mHC's gains sit near zero, and Muon normalises a near-zero gradient, so two
    programs that differ by one fp32 ulp decorrelate by ~1e-1 within 20 steps
    (`.bulba/memory.md`, 2026-10-24). Asserting 1e-6 here would be asserting
    bit-identity the arm does not have; the wiring bound is the first replay
    above, and this test reports the drift with a loose gross-error ceiling.
    """
    data = _corpus(tmp_path)
    run = _gpu_recipe(21)

    def run_one(capture: bool):
        torch.manual_seed(0)
        model = DormouseModel(run.model, device="cuda")
        if not capture:
            monkeypatch.setattr(
                FullGraph,
                "capture",
                lambda self: (_ for _ in ()).throw(RuntimeError("capture disabled")),
            )
        report = train(
            model,
            run,
            stream=ByteStream(data, run.train.batch, run.train.seq_len),
            device="cuda",
            log=lambda *_a, **_k: None,
        )
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        del model
        _free_gpu()
        return report, sd

    rep_g, sd_g = run_one(True)
    assert rep_g["counters"].get("full_graph_captured") == 1
    rep_e, sd_e = run_one(False)
    worst = max(((sd_g[k] - sd_e[k]).abs().max().item(), k) for k in sd_g)
    loss_delta = abs(rep_g["loss"] - rep_e["loss"])
    print(
        f"[m400b 20-step parity] worst |delta| = {worst[0]:.3e} @ {worst[1]}; "
        f"loss delta = {loss_delta:.3e}; within_1e-6={worst[0] <= 1e-6}"
    )
    assert worst[0] <= 1.0, f"O(1) wiring error, not float chaos: {worst}"
    assert loss_delta <= 1e-3, f"the loss does not track: {loss_delta}"
