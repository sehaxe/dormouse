"""The core guarantee: resume from step N is the same run as never having stopped.

Proof, not a smoke test. On the tiny model, CPU, fp32 (torch 2.13), a 6-step run
is compared bit-for-bit with: 3 steps -> checkpoint -> a FRESH model and a
FRESH `MixOptimizer` -> resume -> 3 more steps. The comparison is the loss
trace (every step), every parameter and buffer, and the final checkpoint's
model half, `moe` section and optimizer moments.

Three paths that carry their own state are exercised at the TRAINER level, not
on the stream alone:

* the staged mixture (`MixtureStream.skip_bytes`): the resumed run's served
  windows are compared tensor-for-tensor with the tail of the unbroken run's —
  the `test_skip_bytes_lands_where_an_unbroken_run_would_be` property, but
  through `train()`'s `start x (seq_len+1) x batch x grad_accum` fast-forward,
  a plan whose stage boundary the resume CROSSES, a `.bin` source and a shard
  DIRECTORY source (`ShardStream`, whose resume is modulo an epoch);
* `grad_accum=2`, where the skip is `start x grad_accum` micro-batches;
* `doc_masking`, whose fast-forward is by packed STEPS
  (`DocPackedStream.skip_steps`), not flat bytes.

The one arm that was NOT exact when this file was written (2026-11-08):
`moe_router_jitter > 0`. `SparseMoE.forward` draws its tie-breaking noise from
the process-global torch RNG and the checkpoint carries no RNG state, so
6 steps vs 3 + resume diverged 1.5e-4 relative at `embed.weight` (the noise
moved the router, the router moved everything). `train()`/`sft_train()` now
seed that stream from `(seed, step)` before each step
(`loop.seed_router_jitter`), which also makes two runs of one config agree;
the jitter test pins both halves.

Known breaks that are NOT regressions and have no fix in this lane:
`ternary_s3t` (the Bernoulli mask is a process-local `torch.Generator`; the
trainer prints the disclaimer every run, and `tsct.factor_qat_project`'s
docstring owns the statement) and any CUDA-graph capture (this lane is CPU).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from conftest import SEQ, tiny_cfg

from dormouse.config import RunConfig, TrainConfig
from dormouse.model import DormouseModel
from dormouse.train import train
from dormouse.train.data import doc_sidecar_path
from dormouse.train.mixture import MixPlan, MixStage, MixtureStream, check_epoch_cap

#: The tiny model is an MoE with an Engram — the arms that carry counters,
#: non-persistent buffers and a routing controller, i.e. the state a resume
#: can lose. `TFORM` is the same depth without them, for the data-path tests
#: where the model is not the subject.
TFORM = dict(
    block_form="transformer",
    n_layers=2,
    max_iter=1,
    loop_blocks=1,
    use_kda=False,
    use_csa=False,
    use_mhc=False,
    use_attnres=False,
    use_engram=False,
    layerrope=False,
    n_experts=1,
    moe_topk=0,
    use_tsct=False,
)

LOSS = re.compile(r"step=(\d+) loss=([-\d.eE+]+)")
STEPS = 6
CUT = 3  # the checkpoint is written at the END of optimizer step CUT


class _Stop(Exception):
    """The crash: raised on the log line AFTER the step-CUT checkpoint exists."""


def _write_corpus(path: Path, size: int = 4096) -> Path:
    path.write_bytes(bytes(range(16)) * (size // 16 + 1))
    return path


def _run_config(
    tmp: Path,
    *,
    ckpt_every: int = CUT,
    data: Path | None = None,
    model_over: dict | None = None,
    **over,
):
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        steps=STEPS,
        batch=2,
        seq_len=SEQ,
        lr=0.01,
        log_every=1,
        ckpt_every=ckpt_every,
        ckpt_name="t",
        out_dir=str(tmp / "ck"),
        data=str(data if data is not None else _write_corpus(tmp / "corpus.bin")),
    )
    if over:
        cfg = dataclasses.replace(cfg, **over)
    return RunConfig(source="resume-exact", model=tiny_cfg(**(model_over or {})), train=cfg)


def _model(seed: int = 1234, model_over: dict | None = None) -> DormouseModel:
    torch.manual_seed(seed)
    return DormouseModel(tiny_cfg(**(model_over or {})))


def _hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _go(model, run, *, stream=None, stop_at=None, resume=False, traces=None, fn=train):
    """Run a trainer; crash after the `stop_at` step log when asked (a real interrupt)."""

    def log(line: str) -> None:
        m = LOSS.search(line)
        if m is None:
            return
        step = int(m.group(1))
        if traces is not None:
            traces.append((step, float(m.group(2))))
        if stop_at is not None and step == stop_at:
            raise _Stop

    try:
        return fn(model, run, stream=stream, resume=resume, log=log)
    except _Stop:
        return None


def _assert_same_opt(a: dict, b: dict, path: str = "") -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: optimizer state keys moved: {set(a) ^ set(b)}"
        for key in a:
            _assert_same_opt(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same_opt(x, y, f"{path}[{i}]")
    elif torch.is_tensor(a):
        assert torch.equal(a, b), f"{path}: optimizer tensor differs"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_same_ckpt(x: Path, y: Path) -> None:
    a = torch.load(x, map_location="cpu", weights_only=False)
    b = torch.load(y, map_location="cpu", weights_only=False)
    assert a["step"] == b["step"]
    assert set(a["model"]) == set(b["model"]), "model half: keys moved"
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), f"model half: {key} differs"
    assert set(a["moe"]) == set(b["moe"]), "moe section: keys moved"
    for key in a["moe"]:
        assert torch.equal(a["moe"][key], b["moe"][key]), f"moe section: {key} differs"
    _assert_same_opt(a["opt"], b["opt"], "opt")


def test_three_plus_three_is_six_bit_exact(tmp_path):
    """6 steps continuous == 3 + checkpoint + FRESH model/opt + 3, bit for bit."""
    trace_a: list[tuple[int, float]] = []
    a = _model()
    rep_a = _go(a, _run_config(tmp_path / "ref", ckpt_every=STEPS), traces=trace_a)

    trace_cut: list[tuple[int, float]] = []
    run = _run_config(tmp_path / "cut", ckpt_every=CUT)
    _go(_model(), run, stop_at=CUT + 1, traces=trace_cut)
    assert CUT + 1 in [s for s, _ in trace_cut], "the fixture must run past the checkpoint"

    trace_r: list[tuple[int, float]] = []
    resumed = _model()  # a FRESH model: the process died, its memory is gone
    rep_r = _go(resumed, run, resume=True, traces=trace_r)

    assert len(trace_a) == STEPS and len(trace_r) == STEPS - CUT
    assert trace_a == trace_cut[:CUT] + trace_r, "the loss trace moved"
    assert rep_a["loss"] == rep_r["loss"]
    assert rep_a["opt_steps"] == rep_r["opt_steps"] == STEPS
    assert _hash(a) == _hash(resumed), "the resumed parameters/buffers are not the run's"
    # Not a coincidence of two summaries: the files both runs wrote at the same
    # step agree at the tensor level — model half, `moe` section, moments.
    _assert_same_ckpt(tmp_path / "ref" / "ck" / "t.pt", tmp_path / "cut" / "ck" / "t.pt")


# --- the mixture: the hardest resume path is the data schedule -----------------


def _mix_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    web = root / "web.bin"
    web.write_bytes(b"A" * 8192)
    code = root / "code"
    code.mkdir(exist_ok=True)
    (code / "a.txt").write_bytes(b"B" * 1700)
    (code / "b.txt").write_bytes(b"C" * 1300)
    return {"web": web, "code": code}


def _mix_plan(srcs: dict[str, Path]) -> MixPlan:
    """Two stages, switching at half the budget — the resume CROSSES the seam."""
    return MixPlan(
        (MixStage(0.5, {"web": 1.0}), MixStage(1.0, {"code": 1.0})),
        srcs,
        max_epochs=4.0,
        seed=1,
    )


class _Recorder(MixtureStream):
    """The real stream, plus every window and the stage it came from (a taped run)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.windows: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def next_batch(self):
        stage = self.stage
        x, y = super().next_batch()
        self.windows.append((stage, x.clone(), y.clone()))
        return x, y


def _mix_stream(srcs, grad_accum: int) -> _Recorder:
    plan = _mix_plan(srcs)
    micro = STEPS * grad_accum
    check_epoch_cap(plan, steps=micro, batch=2, seq_len=SEQ)
    return _Recorder(plan, 2, SEQ, micro)


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_mixture_resume_serves_the_same_bytes(tmp_path, grad_accum):
    """A resumed staged mixture continues on the bytes the unbroken run read."""
    srcs = _mix_sources(tmp_path / "data")
    micro = STEPS * grad_accum
    start = CUT * grad_accum

    def cfg(d):
        return _run_config(d, ckpt_every=CUT, grad_accum=grad_accum, model_over=TFORM)

    def model():
        return _model(model_over=TFORM)

    trace_a: list[tuple[int, float]] = []
    s_a = _mix_stream(srcs, grad_accum)
    a = model()
    rep_a = _go(a, cfg(tmp_path / "ref"), stream=s_a, traces=trace_a)

    trace_cut: list[tuple[int, float]] = []
    s_b = _mix_stream(srcs, grad_accum)
    run = cfg(tmp_path / "cut")
    _go(model(), run, stream=s_b, stop_at=CUT + 1, traces=trace_cut)

    trace_r: list[tuple[int, float]] = []
    s_r = _mix_stream(srcs, grad_accum)
    resumed = model()
    rep_r = _go(resumed, run, stream=s_r, resume=True, traces=trace_r)

    # The interrupted leg read up to the crashed step's windows; the resumed one
    # must continue from the CHECKPOINT, not from where the crash happened.
    assert len(s_a.windows) == micro
    assert len(s_b.windows) == (CUT + 1) * grad_accum
    assert len(s_r.windows) == micro - start, "the resume served the wrong window count"
    # The fixture is real: the last window before the checkpoint was stage 0,
    # the first one after it is stage 1 — the resume crosses the plan's seam.
    assert s_a.windows[start - 1][0] == 0 and s_a.windows[start][0] == 1
    for i, ((stage_a, x_a, y_a), (stage_r, x_r, y_r)) in enumerate(
        zip(s_a.windows[start:], s_r.windows, strict=True)
    ):
        assert stage_r == stage_a, f"window {start + i}: stage {stage_r} != {stage_a}"
        assert torch.equal(x_a, x_r), f"window {start + i}: the resumed inputs moved"
        assert torch.equal(y_a, y_r), f"window {start + i}: the resumed targets moved"

    assert trace_a == trace_cut[:CUT] + trace_r, "the loss trace moved"
    assert rep_a["loss"] == rep_r["loss"] and rep_a["opt_steps"] == rep_r["opt_steps"] == STEPS
    assert _hash(a) == _hash(resumed), "the mixture resume did not land on the same weights"


# --- the one arm that broke: an unseeded RNG in the forward --------------------


def test_router_jitter_resumes_exactly_and_repeats_across_runs(tmp_path):
    """`moe_router_jitter>0` used the global RNG with no state in the file.

    Measured before `seed_router_jitter`: 6 vs 3 + resume diverged 1.5e-4
    relative on `embed.weight` (the noise moved the router). With the per-step
    seed, the arm is (a) a resume-exact run and (b) the same run twice.
    """
    over = dict(moe_router_jitter=0.1)

    def cfg(d):
        return _run_config(d, ckpt_every=CUT, model_over=over)

    def model():
        return _model(model_over=over)

    trace_a: list[tuple[int, float]] = []
    a = model()
    rep_a = _go(a, cfg(tmp_path / "ref"), traces=trace_a)

    # (b) one config, two processes: the jitter draw is now pure `(seed, step)`.
    trace_b: list[tuple[int, float]] = []
    b = model()
    _go(b, cfg(tmp_path / "again"), traces=trace_b)
    assert trace_a == trace_b, "two runs of one jitter config drew different noise"
    assert _hash(a) == _hash(b)

    # (a) the guarantee itself.
    trace_cut: list[tuple[int, float]] = []
    run = cfg(tmp_path / "cut")
    _go(model(), run, stop_at=CUT + 1, traces=trace_cut)
    trace_r: list[tuple[int, float]] = []
    resumed = model()
    rep_r = _go(resumed, run, resume=True, traces=trace_r)
    assert trace_a == trace_cut[:CUT] + trace_r, "the jitter arm's resume moved the loss"
    assert rep_a["loss"] == rep_r["loss"]
    assert _hash(a) == _hash(resumed)


# --- doc_masking: the fast-forward counts PACKED STEPS, not flat bytes ---------


def _doc_corpus(tmp: Path) -> Path:
    docs = [b"alpha beta ", b"gamma delta epsilon ", b"zeta eta theta iota "]
    data = b"".join(docs * 40)
    path = tmp / "docs.bin"
    tmp.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    starts, at = [], 0
    for doc in docs * 40:
        starts.append(at)
        at += len(doc)
    doc_sidecar_path(path).write_text(
        json.dumps({"version": 1, "size": len(data), "starts": starts})
    )
    return path


def test_doc_masking_resume_is_bit_exact(tmp_path):
    """`DocPackedStream.skip_steps` replays the packing, so the windows match."""
    data = _doc_corpus(tmp_path)

    def cfg(d):
        return _run_config(d, ckpt_every=CUT, data=data, doc_masking=True, model_over=TFORM)

    def model():
        return _model(model_over=TFORM)

    trace_a: list[tuple[int, float]] = []
    a = model()
    rep_a = _go(a, cfg(tmp_path / "ref"), traces=trace_a)

    trace_cut: list[tuple[int, float]] = []
    run = cfg(tmp_path / "cut")
    _go(model(), run, stop_at=CUT + 1, traces=trace_cut)

    trace_r: list[tuple[int, float]] = []
    resumed = model()
    rep_r = _go(resumed, run, resume=True, traces=trace_r)

    assert trace_a == trace_cut[:CUT] + trace_r, "the doc-packed resume moved the loss"
    assert rep_a["loss"] == rep_r["loss"]
    assert _hash(a) == _hash(resumed)


# --- sft_train: the second step loop, with its own copy of the seed call -----


def test_sft_loop_resumes_exactly_with_router_jitter(tmp_path):
    """`sft_train`'s loop is a separate implementation; the fix had to land twice."""
    from dormouse.train.sft import build_sft_corpus, sft_train

    data = tmp_path / "sft.bin"
    build_sft_corpus(data, seed=1, seq_len=SEQ, n_tasks=32)
    over = dict(moe_router_jitter=0.1)

    def cfg(d):
        return _run_config(d, ckpt_every=CUT, data=data, model_over=over)

    def model():
        return _model(model_over=over)

    trace_a: list[tuple[int, float]] = []
    a = model()
    rep_a = _go(a, cfg(tmp_path / "ref"), traces=trace_a, fn=sft_train)

    trace_cut: list[tuple[int, float]] = []
    run = cfg(tmp_path / "cut")
    _go(model(), run, stop_at=CUT + 1, traces=trace_cut, fn=sft_train)

    trace_r: list[tuple[int, float]] = []
    resumed = model()
    rep_r = _go(resumed, run, resume=True, traces=trace_r, fn=sft_train)

    assert trace_a == trace_cut[:CUT] + trace_r, "the SFT resume moved the loss"
    assert rep_a["loss"] == rep_r["loss"]
    assert _hash(a) == _hash(resumed)
