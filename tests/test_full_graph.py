"""`full_graph` (the WHOLE step in one graph) and the two capture blockers it found.

The arm's claims, each with the gate that fails on revert:

1. **The default path cannot move.** `full_graph=false` is the pre-arm code: the
   sha256 of a 5-step tiny run's `state_dict` AND of its logits are pinned from a
   clean `main` worktree (`b36ef27`, two processes, identical output). If this
   goes red after a `loop.py`/`graph.py` edit, that edit moved the DEFAULT path.
2. **`torch.bincount` is not CUDA-graph capturable on this build, and the MoE's
   dispatch used it.** Measured 2026-10-23: `with torch.cuda.graph(g):
   torch.bincount(a, minlength=9)` raises "Cannot copy between CPU and CUDA
   tensors during CUDA graph capture unless the CPU tensor is pinned" — the op
   pulls a host-side value and the copy aborts the capture. `_padded_dispatch`
   and `_triton_dispatch` counted experts with it, so NO capture of a training
   step could complete at any MoE shape. `experts._bincount` is the same
   histogram without the host round trip; the gate below pins both halves (the
   eager VALUES agree, and the captured call does not raise) and goes RED on
   revert to `torch.bincount`.
3. **The refusals are LOUD, one per REAL conflict**, each naming its `Escape:`.
4. **The arm never kills a run**: a non-CUDA device is refused BEFORE the first
   step (the LR is bound as device tensors in the constructor, so a late
   fallback would not be the eager program), and a capture that cannot complete
   is COUNTED (`opt_graph_fallback` + the reason) with the run staying eager.
5. **The b24 s1024 shape replays 20 optimizer steps without a fault.** The shape
   the 2026-10-23 report names: `final_byte` (d 1024, iters 3, mHC + AttnRes +
   share_u + Engram), batch 24, seq 1024, `muon_state_dtype=bf16`,
   `ns_dtype=bf16`. It builds the REAL model, fakes the grads (the stage's own
   region is what faulted) and runs 20 captured replays. On revert of the
   `allow_frozen` change the capture is REFUSED (`loop.residual_scale` has no
   gradient under `use_mhc`) and this test fails on `graph._graph is not None`.
"""

from __future__ import annotations

import gc
import hashlib
import subprocess

import pytest
import torch
from conftest import tiny_cfg

from dormouse.config import TrainConfig, load_run, validate_train
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel
from dormouse.model.experts import SparseMoE, _bincount
from dormouse.train import train
from dormouse.train.data import ByteStream
from dormouse.train.graph import FullGraph, OptimizerGraph, full_graph_refusal
from dormouse.train.optim import MixOptimizer, clip_grads

# --- the pre-arm reference, captured from a CLEAN `main` worktree -----------
#
# `/home/sehaxe/dormouse-wt/golden` at `b36ef27`, `uv run python` on the script
# `_five_steps` repeats (the same 10 240-byte fixture), two processes, identical:
#   PARAMS d3001de29191f26c055ef8d9e9762e8ff69f98eb2ba1833c4648a02c199bb128
#   LOGITS b862230dffd0ab6ab0643b806218945eab2c3c80135df2e5367e88f8fd46463d
_PRE_CHANGE_PARAMS_SHA = "d3001de29191f26c055ef8d9e9762e8ff69f98eb2ba1833c4648a02c199bb128"
_PRE_CHANGE_LOGITS_SHA = "b862230dffd0ab6ab0643b806218945eab2c3c80135df2e5367e88f8fd46463d"


def _sha_state(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _five_steps(tmp_path, **train_over):
    """The pinned 5-step tiny run: `(params sha, logits sha)`, host device."""
    ov = dict(
        batch=8,
        seq_len=64,
        steps=5,
        log_every=10**9,
        eval=False,
        ckpt_every=10**9,
        lr=1e-3,
    )
    ov.update(train_over)
    run = load_run("tiny", train_overrides=ov)
    torch.manual_seed(0)
    model = DormouseModel(run.model, device="cpu")
    data = tmp_path / "bytes.bin"
    data.write_bytes(bytes(range(256)) * 40)
    stream = ByteStream(data, run.train.batch, run.train.seq_len)
    train(model, run, stream=stream, device=None, log=lambda *a, **k: None)
    x, y = stream.next_batch()
    with torch.no_grad():
        logits, _loss, _c = model(x, y)
    return _sha_state(model.state_dict()), _sha_tensor(logits)


def test_the_default_path_is_byte_identical(tmp_path):
    params, logits = _five_steps(tmp_path)
    assert params == _PRE_CHANGE_PARAMS_SHA, (
        "full_graph=false moved the default path's parameters — this arm must be a "
        "no-op when it is off"
    )
    assert logits == _PRE_CHANGE_LOGITS_SHA, "full_graph=false moved the default path's logits"


# --- the capture blocker: `torch.bincount` is not capturable ----------------

_BC = torch.tensor([0, 1, 1, 3, 3, 3, 7], dtype=torch.long)


def test_the_moe_histogram_matches_torch():
    """Same values, and the LENGTH rule the call sites rely on (all ids < n).

    `minlength` must cover the input's range — every call site counts expert ids
    whose sentinel is already `< minlength`. A short `minlength` is LOUD here
    (the scatter's own bounds check) where `torch.bincount` would silently
    return a longer array, and that difference is asserted, not hidden.
    """
    for n in (8, 9, 65):
        assert torch.equal(_bincount(_BC, n), torch.bincount(_BC, minlength=n)), n
    with pytest.raises(RuntimeError, match="out of bounds"):
        _bincount(_BC, 4)


def _card_reason(min_free_mib: int = 2048) -> str | None:
    """`None` when a GPU gate may run. The desktop holds ~1 GiB on this box
    permanently, so the threshold is a REAL second process, not the display."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        busy = subprocess.run(
            ["pgrep", "-f", "dormouse trai[n]"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a gate's precondition, not a run's
        return f"cannot check the card ({e})"
    if busy:
        return f"the live training run holds the card (pids {busy}) — one GPU thing at a time"
    mib = int(used.split()[0]) if used else -1
    if mib >= 4096:
        return f"the card already holds {mib} MiB"
    free, _total = torch.cuda.mem_get_info()
    if free < min_free_mib << 20:
        return f"only {free / 2**20:.0f} MiB VRAM free, this gate needs {min_free_mib}"
    return None


CUDA = pytest.mark.skipif(_card_reason(512) is not None, reason=_card_reason(512) or "GPU gate")


@CUDA
def test_bincount_aborts_a_capture_and_the_replacement_does_not():
    """THE gate for the fix: the torch call raises, `_bincount` captures.

    Reverting `experts._bincount` to `torch.bincount` re-breaks every capture of
    a training step (measured at every MoE shape), so this is not a style test.
    """
    a = torch.randint(0, 8, (4096,), device="cuda")
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="pinned|graph capture"):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.bincount(a, minlength=9)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        out = _bincount(a, 9)
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.bincount(a, minlength=9))


# --- the refusals: one per real conflict -----------------------------------


def _tiny_pair():
    run = load_run("tiny", train_overrides=dict(seq_len=64, batch=8))
    return run.train, run.model


@pytest.mark.parametrize(
    "train_over, model_over, needle",
    [
        (dict(opt_graph=True), {}, "TWO captures"),
        (dict(optim_8bit=True), {}, "bitsandbytes"),
        (dict(muon_sf=True), {}, "schedule-free"),
        (dict(muon_variant="normuon"), {}, "NorMuon"),
        ({}, dict(moe_offload=True), "host-offload"),
        (dict(grad_accum=2), {}, "accumulation loop is PYTHON"),
        (dict(rand_depth=True), {}, "sampled from `(seed, step)` on the HOST"),
        (dict(segment_len=16), {}, "PYTHON loop over segments"),
        ({}, dict(use_byteflow=True), "patch latents"),
        ({}, dict(halt_weight=1.0), "halt"),
        ({}, dict(use_msa=True), "MSA arm reads a device share back"),
        (dict(ema_decay=0.99), {}, "EAGERLY after the optimizer step"),
        (dict(engram_ram=True), {}, "device→host INDEX transfer"),
        ({}, dict(jepa=True, jepa_weight=0.05), "span mask"),
        ({}, dict(dspark_weight=0.1), "last_mtp"),
        ({}, dict(act_quant="int8"), "finiteness guard is a HOST read"),
        ({}, dict(dense_qat="int8"), "QAT arms"),
    ],
)
def test_full_graph_refuses_each_real_conflict(train_over, model_over, needle):
    import dataclasses

    train_cfg, model_cfg = _tiny_pair()
    train_cfg = dataclasses.replace(train_cfg, **train_over)
    model_cfg = dataclasses.replace(model_cfg, **model_over)
    why = full_graph_refusal(train_cfg, model_cfg)
    assert why is not None, "this conflict must be refused"
    assert needle in why, why
    assert "Escape:" in why, f"a refusal without an escape is not actionable: {why}"


def test_full_graph_is_legal_on_the_default_config():
    train_cfg, model_cfg = _tiny_pair()
    assert full_graph_refusal(train_cfg, model_cfg) is None


def test_validate_refuses_full_graph_with_accum_and_compile():
    for over, needle in (
        (dict(full_graph=True, grad_accum=2), "grad_accum"),
        (dict(full_graph=True, torch_compile=True), "torch_compile"),
        (dict(full_graph=True, opt_graph=True), "TWO captures"),
    ):
        cfg = TrainConfig(**over)
        with pytest.raises(LoudError, match=needle):
            validate_train(cfg)


def test_the_arm_refuses_a_non_cuda_device_up_front(tmp_path):
    """No CUDA: LOUD BEFORE the first step, not at capture.

    The arm's constructor binds the LR as device tensors; refusing only when the
    first capture fires would already have run the warmup steps on fp32 LR
    tensors, i.e. a different program wearing the fallback's name.
    """
    with pytest.raises(LoudError, match="CUDA-only"):
        _five_steps(tmp_path, full_graph=True)


# --- GPU: the b24 s1024 shape, 20 replays ----------------------------------


@pytest.fixture(scope="module")
def b24_model():
    """The `final_byte` model at the shape the 2026-10-23 report names."""
    why = _card_reason(10_000)
    if why is not None:
        pytest.skip(f"GPU busy — nothing measured: {why}")
    run = load_run(
        "final_byte",
        train_overrides=dict(
            batch=24,
            seq_len=1024,
            engram_ram=True,
            muon_state_dtype="bf16",
            ns_dtype="bf16",
            grad_clip=1.0,
        ),
        model_overrides=dict(bf16=True),
    )
    torch.manual_seed(0)
    try:
        model = DormouseModel(run.model, device="cuda")
    except torch.cuda.OutOfMemoryError:
        # The suite shares this card with every other gate; a full 93M model at
        # b24 s1024 plus its graph pool needs a quiet one. Nothing measured, so
        # there is nothing to report — the machine discipline of `_card_reason`.
        torch.cuda.empty_cache()
        pytest.skip("the card cannot hold the b24 s1024 model right now")
    return run, model


def test_the_b24_s1024_optimizer_stage_replays_20_times(b24_model):
    """The crash gate: the shape is in the name, and the capture must HAPPEN.

    On revert of `allow_frozen` the capture is refused (`loop.residual_scale`
    has no gradient under `use_mhc`) — this fails, and that IS the finding: the
    arm measured "+0 at b8 s512" for a month because it never captured.

    The GRADS ARE FAKED and no forward runs: this gate is the optimizer STAGE's
    (the region that was captured when the fault fired), and a real b24 s1024
    forward+backward puts the run's peak at 14.8 of 15.6 GiB — which the arm's
    own headroom rule then refuses, correctly, for a run that must keep
    training. The stage's shapes, chunking and LR seam are the real model's
    either way.
    """
    run, model = b24_model
    opt = MixOptimizer(model, lr=3e-4, wd=run.train.wd, muon_state_dtype="bf16", ns_dtype="bf16")
    gen = torch.Generator(device="cuda").manual_seed(7)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=gen, device="cuda") * 0.1

    def stage() -> None:
        clip_grads(model.parameters(), 1.0)
        opt.step()

    graph = OptimizerGraph(opt, stage, warmup_steps=1, allow_frozen=True)
    try:
        graph.step()
        graph.step()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("the card cannot hold the b24 s1024 graph pool right now")
    assert graph.reason is None, f"the capture must run at b24 s1024: {graph.reason}"
    assert graph._graph is not None, "the capture did not happen at b24 s1024"
    for _ in range(20):
        graph.step()
    torch.cuda.synchronize()
    assert graph.reason is None, graph.reason
    torch.cuda.empty_cache()


# --- the SECOND blocker: a boolean-mask index is `nonzero` ------------------
#
# Measured 2026-10-24: `order[keep]` (a BOOL mask on a device tensor) is
# `nonzero` underneath — a D2H read, and a kept-row count that changes with the
# routing, so a graph could neither record it nor replay it. The abort surfaces
# as `cudaErrorStreamCaptureInvalidated` at the NEXT call (measured: the capture
# died one op after `aten.index.Tensor @ experts.py:1354`), which is why the
# `TorchDispatchMode` bisect is `scripts/probe_capture.py`.
# `_padded_dispatch` now keeps `n·k` rows and moves the cap into a MASK; the
# three ragged arms are refused by `full_graph_refusal` instead of failing mid
# capture.


def _moe() -> SparseMoE:
    cfg = tiny_cfg(n_experts=4, moe_topk=2, rank=8, d_model=64, d_ffn=128, bf16=True)
    return SparseMoE(cfg, device="cuda")


def _kept(moe: SparseMoE, x: torch.Tensor) -> torch.Tensor:
    """The `keep` mask the forward's own router produces for `x`."""
    probs = torch.softmax(moe.router(x.float()), dim=-1)
    return moe._permute(probs.topk(moe.topk, dim=-1).indices)[1]


@CUDA
def test_a_boolean_mask_index_aborts_a_capture_and_the_static_rows_do_not():
    """The op, and the arm that no longer contains it.

    Revert `_padded_dispatch` to `rows = order[keep]` and the capture below
    raises the same `StreamCaptureInvalidated` the first half pins.
    """
    t = torch.arange(4096, device="cuda")
    mask = torch.rand(4096, device="cuda") > 0.5
    with pytest.raises(RuntimeError, match="pinned|graph capture|previous error"):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            t[mask]
    torch.cuda.synchronize()

    torch.manual_seed(0)
    moe = _moe()
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = moe(x)
        eager2 = moe(x2)
        keep1, keep2 = _kept(moe, x), _kept(moe, x2)
    assert not torch.equal(keep1, keep2), "the fixture must route differently"

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        out = moe(x)
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the replay must be the eager forward, bit for bit"
    x.copy_(x2)  # the SAME storage: `fill()`'s contract
    g2.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager2), "the replay must read the LIVE buffer, not the capture's"
    print(
        f"[moe capture] dropped rows: pass1={int((~keep1).sum())} "
        f"pass2={int((~keep2).sum())} -> the replay followed the routing"
    )
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "model_over, needle",
    [
        (dict(moe_capacity=0.0), "dropless"),
        (dict(moe_kernel="triton"), "RAGGED"),
        (dict(moe_kernel="naive"), "PYTHON loop"),
        (dict(moe_gather="kernel"), "PERMUTED"),
    ],
)
def test_full_graph_refuses_the_ragged_moe_arms(model_over, needle):
    """One refusal per arm, each naming the dynamic row count and its Escape.

    These four are refusals, not fixes, and the reason is the SAME one in all
    four: the arm is built around the kept-row count (`offs`, the per-expert
    loop's `unique`, the gather ABI's permuted rows), i.e. a device→host read
    and a shape that changes with the routing. The padded bank is the arm whose
    rows are static (`n·k`, cap as a mask).
    """
    import dataclasses

    train_cfg, model_cfg = _tiny_pair()
    model_cfg = dataclasses.replace(model_cfg, moe_topk=2, **model_over)
    why = full_graph_refusal(train_cfg, model_cfg)
    assert why is not None, "this arm must be refused"
    assert needle in why, why
    assert "Escape:" in why, why
    assert "moe_kernel='padded'" in why, why


# --- the whole step: forward + backward + firewall + optimizer in ONE graph -

_B8S512 = dict(
    batch=8,
    seq_len=512,
    eval=False,
    log_every=10**9,
    ckpt_every=10**9,
    muon_state_dtype="bf16",
    ns_dtype="bf16",
    grad_clip=1.0,
    full_graph=True,
)


@pytest.fixture(scope="module", autouse=True)
def _release_card():
    """Hand the card back when this module ends.

    The b24 fixture's model and every captured graph are dropped by the garbage
    collector, and a graph's POOL is mapped for as long as the graph lives — so
    the next module's `_card_reason` (a 1 GiB threshold in `test_opt_graph`) sees
    the residue and skips its three GPU gates. Measured: 11 GiB without this,
    1059 MiB with only the per-test free, under the 1 GiB bar after a collect
    that runs AFTER the module's fixtures are finalised.
    """
    yield
    _free_gpu()


def _free_gpu() -> None:
    """Release the graph POOL, not just the cache.

    A captured graph's pool is mapped for the GRAPH's life and the graph is
    dropped by the garbage collector, not by leaving `train()`. Without the
    collect, the NEXT module on this card sees ~11 GiB still reserved and its
    GPU gates skip (measured: `test_opt_graph`'s three gates skipped only when
    these tests ran first).

    The ccBLAS workspaces a capture allocates are HELD for the process's life
    (`cudaGraphCaptureMode` maps them outside the allocator), +226 MiB measured
    here — enough to push `test_opt_graph`'s 1 GiB card threshold over and skip
    three gates that pass on `main`. `_cuda_clearCublasWorkspaces` is torch's
    own call for exactly this (it runs around its captures); guarded because a
    CPU-only build need not expose it.
    """
    gc.collect()
    torch.cuda.empty_cache()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # noqa: SLF001
    if clear is not None:
        clear()
        torch.cuda.empty_cache()


def _final_byte_run(steps: int, **over):
    ov = dict(_B8S512, steps=steps)
    ov.update(over)
    return load_run("final_byte", train_overrides=ov, model_overrides=dict(bf16=True))


def _corpus(tmp_path, n: int = 1 << 16):
    p = tmp_path / "bytes.bin"
    p.write_bytes(bytes(range(256)) * (n // 256))
    return p


@CUDA
def test_the_whole_step_captures_and_replays_20_times_at_b8_s512(tmp_path):
    """`final_byte` b8 s512: 5 eager steps, ONE capture, 20 replays, no fault.

    A COUNTED fallback is the failure this gate exists for: the arm then looks
    identical from the outside (same loss, same counters) while every step runs
    eager. Revert either half of the fix — `_padded_dispatch`'s static rows or
    `FullGraph.capture`'s stale-stream override — and the log carries
    "full_graph: capture failed, staying EAGER".
    """
    run = _final_byte_run(26, log_every=1)
    torch.manual_seed(0)
    model = DormouseModel(run.model, device="cuda")
    logs: list[str] = []
    report = train(
        model,
        run,
        stream=ByteStream(_corpus(tmp_path), run.train.batch, run.train.seq_len),
        device="cuda",
        log=logs.append,
    )
    fallback = [ln for ln in logs if "capture failed" in ln]
    assert not fallback, fallback
    assert report["steps"] == 26
    assert torch.isfinite(torch.tensor(report["loss"]))
    # 20 replays must each read the LIVE batch: identical losses would mean the
    # graph replayed the capture step's data.
    losses = [ln for ln in logs if ln.startswith("step=")]
    tail = {ln.split("loss=")[1].split()[0] for ln in losses[-20:]}
    print(f"[whole step] {len(losses)} logged steps, {len(tail)} distinct replay losses")
    assert len(tail) >= 15, f"the replay is not advancing the data: {sorted(tail)}"
    del model
    _free_gpu()


@CUDA
def test_the_first_replay_is_the_eager_step(tmp_path, monkeypatch):
    """Capture vs the arm's OWN eager program, one process, same seed and data.

    `full_graph=false` is not the reference (the arm binds the LR as a device
    tensor and turns AdamW's `capturable` branch on, so it is a different
    program by design — measured 2e-7 after ONE step). The reference is the same
    arm with the capture removed, and the bound is the measured 1.5e-8 fp32 ulp
    the recording costs: a wrong LR, a stale buffer or a frozen optimizer state
    would be O(1) here.
    """
    data = _corpus(tmp_path)

    def six_steps(capture: bool) -> tuple[dict, str, list[str]]:
        run = _final_byte_run(6)
        torch.manual_seed(0)
        model = DormouseModel(run.model, device="cuda")
        logs: list[str] = []
        if not capture:
            monkeypatch.setattr(
                FullGraph,
                "capture",
                lambda self: (_ for _ in ()).throw(RuntimeError("capture disabled")),
            )
        report = train(
            model,
            run,
            stream=ByteStream(data, 8, 512),
            device="cuda",
            log=logs.append,
        )
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        loss = f"{report['loss']:.9g}"
        del model
        _free_gpu()
        return sd, loss, logs

    eager_sd, eager_loss, eager_logs = six_steps(capture=True)
    _free_gpu()
    graph_sd, graph_loss, graph_logs = six_steps(capture=False)
    _free_gpu()
    assert not any("capture failed" in ln for ln in eager_logs), eager_logs
    assert any("capture failed" in ln for ln in graph_logs), "the reference must be the EAGER arm"

    worst = max(((a - graph_sd[k]).abs().max().item(), k) for k, a in eager_sd.items())
    print(
        f"[first replay] worst |delta| = {worst[0]:.3e} @ {worst[1]}; "
        f"loss {eager_loss} vs {graph_loss}"
    )
    assert worst[0] <= 1e-6, f"the capture moved the arithmetic: {worst}"
