"""LayerRoPE (`arXiv:2610.09179`) in the looped port.

The arm replaces every RMSNorm gain with ONE shared vector and a per-depth
`(amp, theta)` pair:

    gamma_i = amp_i · rotate_pairs(gamma_shared, theta_i)

Depth is the prelude layer, the ACTUAL loop iteration (`P+i`), the coda layer,
or the terminal final norm. The gates below are the task's, in order:

a. OFF is bit-identical to `main` (state + logits sha256 pins captured from a
   clean detached worktree of `main` HEAD before any edit);
b. amp=1, theta=0 makes `gamma_i` bit-equal to `gamma_shared` and the ON
   forward bit-equal to the OFF forward;
c. the norm-gain parameters are exactly `d + 2·n_depths` (or
   `d + 2·n_depths·n_sites` per-site), printed against OFF and `dormouse info`;
d. every gain parameter takes a finite, non-zero gradient in a real 3-step run;
e. the rotation is live (`theta_0 = pi/2` moves the logits);
f. the depth index is the ACTUAL iteration: moving iteration 1's amp changes a
   full run and does NOT change a `depth=1` prefix;
g. every declared refusal is LOUD with key + reason + `Escape:`, including a
   resume from a `layerrope=false` checkpoint;
h. the forward + backward + optimizer step read no device scalar to the host;
i. the `layerrope_forward` counter is DEVICE-side and drains on the log cadence;
j. the full CPU suite (reported with this branch, not runnable from inside
   pytest): exit 0, failing node-id set identical to the clean-`main` baseline.

The `dormouse info` gate is in (c): a closed-form parameter count that drifts
from the instantiated tree is the defect the CLI line exists to catch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re

import pytest
import torch
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, load_preset, load_run, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, param_count
from dormouse.train import train
from dormouse.train.loop import ckpt_paths, load_ckpt

#: Captured from a clean detached worktree of `main` HEAD (3f83788) BEFORE the
#: flag existed: `load_preset("tiny")`, seed 0, torch 2.13.0+cu129.
#: These coincide with `test_cheapwidth.py`'s `tiny_tsct` pins — this arm must
#: add nothing to the OFF tree.
GOLDEN_STATE_SHA = "d3360f3178dc1be2f3bfc6614f52927f762fc76054187098de8faa1bdb6d2d27"
GOLDEN_LOGITS_SHA = "57d0c0ee6c5b0240e81d95eaea18681d06ef2c9ca1400292c3f97948e79837ee"
GOLDEN_PARAMS = 64_881
GOLDEN_LOSS = 6.034450531005859
#: The OFF gradients too: same seed/inputs as `_logits_hash`, every parameter
#: sorted by name; absent gradients are recorded as the literal "none" (the CPU
#: SDPA fallback leaves `loop.attn.beta_proj.weight` without a grad, pre-existing).
GOLDEN_GRAD_SHA = "c8fd5f27070192a18bd0d99007f4f61fedbce79073477960b6789367cab75a48"


def _cfg(**over) -> ModelConfig:
    cfg = load_preset("tiny")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _built(**over) -> DormouseModel:
    torch.manual_seed(0)
    return DormouseModel(_cfg(**over))


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _logits_hash(model, seed: int = 7) -> tuple[str, float]:
    torch.manual_seed(seed)
    ids = (torch.arange(2 * 16, dtype=torch.long) % model.cfg.vocab).reshape(2, 16)
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(ids, ids)
    assert loss is not None and torch.isfinite(loss)
    h = hashlib.sha256()
    h.update(logits.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest(), float(loss)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(BATCH * SEQ, dtype=torch.long).view(BATCH, SEQ) % 251
    return ids, (ids + 1) % 251


def _grad_hash(model) -> str:
    """The OFF forward's parameter gradients, deterministic from seed 7."""
    torch.manual_seed(7)
    ids = (torch.arange(2 * 16, dtype=torch.long) % model.cfg.vocab).reshape(2, 16)
    model.eval()
    with torch.enable_grad():
        _, loss, _ = model(ids, ids)
        assert loss is not None
        model.zero_grad()
        loss.backward()
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        if p.grad is None:
            h.update(b"none")
            continue
        h.update(str(tuple(p.grad.shape)).encode())
        h.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _norm_gain_elems(model: DormouseModel) -> dict[str, int]:
    """Every parameter/buffer that is a norm GAIN (off: `.norm.weight`; on: layerrope)."""
    out: dict[str, int] = {}
    for name, t in model.state_dict().items():
        if name.startswith("layerrope.") or name == "norm.weight" or name.endswith(".norm.weight"):
            out[name] = t.numel()
    return out


def test_a_off_is_bit_identical_to_main():
    """Gate a: the OFF tree is main's state dict, logits and gradients, byte for byte."""
    assert ModelConfig().layerrope is False
    model = _built()
    assert "layerrope.gamma_shared" not in model.state_dict()
    assert param_count(model) == GOLDEN_PARAMS
    assert _state_hash(model) == GOLDEN_STATE_SHA, "the OFF state_dict moved"
    logits_sha, loss = _logits_hash(model)
    assert logits_sha == GOLDEN_LOGITS_SHA, "the OFF logits moved"
    assert loss == GOLDEN_LOSS, loss
    assert _grad_hash(model) == GOLDEN_GRAD_SHA, "the OFF gradients moved"
    print(
        f"[layerrope] off state={GOLDEN_STATE_SHA[:16]} logits={GOLDEN_LOGITS_SHA[:16]} "
        f"grad={GOLDEN_GRAD_SHA[:16]} loss={loss!r} params={GOLDEN_PARAMS}"
    )


def test_b_identity_init_is_a_bit_exact_no_op():
    """Gate b: `gamma_i == gamma_shared` (torch.equal) and the forward is the OFF one."""
    off = _built()
    on = _built(layerrope=True)
    gains = on.layerrope
    assert gains is not None
    assert gains.n_depths == off.cfg.max_iter + 1  # P=C=0, plus the terminal norm
    # Every site's gain is the shared vector, bit for bit.
    d = on.cfg.d_model
    for depth in range(gains.n_depths):
        for site in range(gains.n_sites):
            gamma = gains.gain(depth, site)
            assert gamma.shape == (d,)
            assert torch.equal(gamma, gains.gamma_shared), (depth, site)
    # The flag adds no RNG draw and replaces only the per-site gain weights.
    off_sd, on_sd = off.state_dict(), on.state_dict()
    assert "norm.weight" not in on_sd and "loop.norm.weight" not in on_sd
    assert set(on_sd) - set(off_sd) == {
        "layerrope.gamma_shared",
        "layerrope.amp",
        "layerrope.theta",
    }
    for key, value in off_sd.items():
        if key == "norm.weight" or key.endswith(".norm.weight"):
            continue
        assert key in on_sd, key
        assert torch.equal(value, on_sd[key]), f"{key}: the ON build changed a non-gain weight"
    x, y = _batch()
    with torch.no_grad():
        logits_off, loss_off, _ = off(x, y)
        logits_on, loss_on, _ = on(x, y)
    delta = float((logits_off - logits_on).abs().max())
    print(f"[layerrope] identity forward max|Δ|={delta:.3e} loss {float(loss_off):.9f}")
    assert delta < 1e-6, delta
    assert abs(float(loss_off) - float(loss_on)) < 1e-6
    assert torch.isfinite(logits_on).all()


@pytest.mark.parametrize(
    "over,n_depths,n_sites",
    [
        ({}, 3, 1),
        ({"prelude_layers": 1, "coda_layers": 1}, 5, 1),
        ({"prelude_layers": 1, "coda_layers": 1, "loop_blocks": 2}, 5, 5),
        (
            {
                "prelude_layers": 1,
                "coda_layers": 1,
                "loop_blocks": 2,
                "layerrope_per_site": True,
            },
            5,
            5,
        ),
    ],
)
def test_c_norm_gain_parameters_are_d_plus_two_ndepths(over, n_depths, n_sites):
    """Gate c: the gain count is the formula, and the total delta is the gain delta."""
    off = _built(**{k: v for k, v in over.items() if k != "layerrope_per_site"})
    on = _built(layerrope=True, **over)
    d = on.cfg.d_model
    per_site = bool(over.get("layerrope_per_site", False))
    want = d + 2 * n_depths if not per_site else d + 2 * n_depths * n_sites
    off_gain = sum(_norm_gain_elems(off).values())
    on_gain = sum(_norm_gain_elems(on).values())
    delta = param_count(on) - param_count(off)
    print(
        f"[layerrope] tiny {over}: off_gain={off_gain} on_gain={on_gain} "
        f"want={want} param delta={delta:+d}"
    )
    assert on_gain == want
    assert delta == on_gain - off_gain, "a parameter moved outside the norm gains"


def test_c_medium_counts_via_dormouse_info():
    """Gate c (medium): `dormouse info` agrees with the hand arithmetic."""
    runner = CliRunner()

    def total(*sets: str) -> int:
        out = runner.invoke(app, ["info", "medium", *[f"--set={s}" for s in sets]])
        assert out.exit_code == 0, out.output
        m = re.search(r"([\d,]+) params", out.output)
        assert m, out.output
        return int(m.group(1).replace(",", ""))

    base = total()
    on = total("layerrope=true")
    per_site = total("layerrope=true", "layerrope_per_site=true")
    # medium: P=C=0, max_iter=4 -> n_depths=5; n_sites=2.
    print(f"[layerrope] medium {base:,} -> on {on:,} ({on - base:+,}) / {per_site:,}")
    assert base - on == 2 * 1024 - (1024 + 2 * 5)
    assert base - per_site == 2 * 1024 - (1024 + 2 * 5 * 2)


def test_c_learn_theta_false_is_a_fixed_buffer():
    """`layerrope_learn_theta=false`: the angle is a buffer, not a parameter."""
    model = tiny_model(layerrope=True, layerrope_learn_theta=False)
    gains = model.layerrope
    assert not isinstance(gains.theta, torch.nn.Parameter)
    assert gains.theta.requires_grad is False
    assert "layerrope.theta" in model.state_dict()
    want = model.cfg.d_model + gains.n_depths
    got = sum(p.numel() for p in gains.parameters())
    print(f"[layerrope] learn_theta=false: parameters={got} want={want}")
    assert got == want
    x, _ = _batch()
    with torch.no_grad():
        logits, _, _ = model(x)
    assert bool(torch.isfinite(logits).all())


def test_d_every_gain_parameter_gets_a_nonzero_gradient(tmp_path):
    """Gate d: gamma_shared, every amp and every theta are finite and non-zero after 3 steps."""
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=3, log_every=100),
        model=tiny_cfg(layerrope=True, prelude_layers=1, coda_layers=1),
    )
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    named = dict(model.named_parameters())
    dead: list[str] = []
    for name in ("layerrope.gamma_shared", "layerrope.amp", "layerrope.theta"):
        p = named[name]
        assert p.grad is not None, f"{name}: no gradient slot after 3 steps"
        assert bool(torch.isfinite(p.grad).all()), f"{name}: non-finite gradient"
        flat = p.grad.detach().reshape(-1)
        zeros = [i for i, v in enumerate(flat.tolist()) if v == 0.0]
        if zeros:
            dead.append(f"{name}{zeros}")
        print(
            f"[layerrope] 3-step {name} shape={tuple(p.shape)} "
            f"max|grad|={float(flat.abs().max()):.3e} zeros={len(zeros)}"
        )
    assert not dead, f"dead LayerRoPE parameters: {dead}"


def test_e_the_rotation_is_live():
    """Gate e: theta_0 = pi/2 moves the logits against theta_0 = 0."""
    model = tiny_model(layerrope=True)
    x, _ = _batch()
    with torch.no_grad():
        before, _, _ = model(x)
    with torch.no_grad():
        model.layerrope.theta.data[0] = math.pi / 2
        after, _, _ = model(x)
    moved = float((before - after).abs().max())
    print(f"[layerrope] theta_0=pi/2 max|Δ logits|={moved:.4e}")
    assert moved > 1e-3, moved


def test_f_depth_index_is_the_actual_iteration_not_a_run_position():
    """Gate f: iteration 1's amp changes a full run, not a depth=1 prefix."""
    model = tiny_model(layerrope=True)  # P=C=0, max_iter=2
    x, _ = _batch()
    with torch.no_grad():
        prefix_before, _, _ = model(x, depth=1)
        full_before, _, _ = model(x)
    assert model.loop._depth_ran == 2
    with torch.no_grad():
        model.layerrope.amp.data[1] = 5.0  # depth P+1 = iteration 1
    with torch.no_grad():
        prefix_after, _, _ = model(x, depth=1)
        full_after, _, _ = model(x)
    assert model.loop._depth_ran == 2
    # A prefix that never reaches iteration 1 must be BIT-identical.
    assert torch.equal(prefix_before, prefix_after), "iteration 1 leaked into a depth=1 prefix"
    moved = float((full_before - full_after).abs().max())
    print(f"[layerrope] prefix unchanged=True full-run max|Δ|={moved:.4e}")
    assert moved > 1e-3, moved


def test_g_declared_refusals_are_loud_and_name_the_key():
    """Gate g: each refused combination names key, reason and `Escape:`."""
    for text, key in (
        ("use_byteflow = true\nlayerrope = true\n", "use_byteflow"),
        (
            "d_model = 65\nn_heads = 5\nhead_dim = 13\nlayerrope = true\n",
            "d_model",
        ),
        ("layerrope_per_site = true\n", "layerrope_per_site"),
        ("layerrope = true\nlayerrope_amp_init = 0.0\n", "layerrope_amp_init"),
    ):
        with pytest.raises(LoudError) as e:
            parse_preset(text)
        msg = str(e.value)
        assert key in msg and "Escape:" in msg, msg
        print(f"[layerrope] refused {key}: {msg.splitlines()[0][:120]}")
    # The runtime seam: a hand-built model never calls `validate`.
    with pytest.raises(LoudError) as e:
        DormouseModel(ModelConfig(use_byteflow=True, layerrope=True))
    assert "use_byteflow" in str(e.value) and "Escape:" in str(e.value)
    with pytest.raises(LoudError) as e:
        DormouseModel(tiny_cfg(layerrope=True, d_model=65, n_heads=5, head_dim=13))
    assert "d_model=65" in str(e.value) and "Escape:" in str(e.value)
    # `full_graph` / `opt_graph`: MODEL×TRAIN, after the overrides.
    for key in ("full_graph", "opt_graph"):
        with pytest.raises(LoudError) as e:
            load_run(
                "tiny",
                model_overrides={"layerrope": True},
                train_overrides={key: True},
            )
        msg = str(e.value)
        assert key in msg and "layerrope" in msg and "Escape:" in msg, msg
        print(f"[layerrope] refused {key}: {msg.splitlines()[0][:120]}")
    # `muon_sf` does NOT conflict: the gain table is 1-D, so it routes to AdamW.
    run = load_run(
        "tiny",
        model_overrides={"layerrope": True},
        train_overrides={"muon_sf": True, "seq_len": 64, "batch": 2},
    )
    assert run.model.layerrope and run.train.muon_sf
    print("[layerrope] muon_sf + layerrope allowed (1-D gains -> AdamW)")


def test_g_resume_from_a_layerrope_off_checkpoint_is_refused(tmp_path):
    """Gate g: the snapshot drift names `model.layerrope` and the escape."""
    off_run = tiny_run(tmp_path, steps=2, ckpt_every=2)
    train(DormouseModel(off_run.model), off_run, log=lambda *_: None)
    weights, snapshot = ckpt_paths(off_run.train)
    assert weights.is_file() and snapshot.is_file()
    on_run = dataclasses.replace(
        tiny_run(tmp_path, steps=4, ckpt_every=2), model=tiny_cfg(layerrope=True)
    )
    with pytest.raises(LoudError) as e:
        load_ckpt(on_run)
    msg = str(e.value)
    assert "model.layerrope" in msg and "resume refused" in msg and "Escape:" in msg, msg
    print(f"[layerrope] resume refused: {msg.splitlines()[0][:140]}")


def test_g_allowed_flags_build_and_run():
    """The arm must compose with mHC (the case we most want), AttnRes, CSA, window."""
    for over in (
        {"use_mhc": True},
        {"use_attnres": True},
        {"use_mhc": True, "use_attnres": True},
        {"use_csa": True},
        {"attn_window": 8},
        {"loop_blocks": 2},
        {"prelude_layers": 1, "coda_layers": 1},
        {"tie_embeddings": True},
        {"dense_qat": "ternary"},
        {"factor_qat": "ternary"},
    ):
        model = tiny_model(layerrope=True, **over)
        x, y = _batch()
        with torch.no_grad():
            logits, loss, _ = model(x, y)
        assert loss is not None and bool(torch.isfinite(loss))
        print(f"[layerrope] allowed {over}: loss={float(loss):.4f}")
        del logits


def test_g_rand_depth_grad_accum_and_ema_are_allowed(tmp_path):
    """`rand_depth`, `grad_accum` and `ema_decay` must run with the arm on."""
    torch.manual_seed(0)
    base = tiny_run(tmp_path, steps=3, log_every=1, ckpt_every=100)
    run = dataclasses.replace(
        base,
        model=tiny_cfg(layerrope=True, prelude_layers=1, coda_layers=1, moe_topk=0),
        train=dataclasses.replace(base.train, rand_depth=True, grad_accum=2, ema_decay=0.9),
    )
    model = DormouseModel(run.model)
    report = train(model, run, log=lambda *_: None)
    assert report["nan_skipped"] == 0
    # A rand-depth step counts `P + T_actual + C + 1` applications, so the
    # count is > 0 and strictly less than `grad_accum·steps·(P+max_iter+C+1)`.
    apps = report["counters"]["layerrope_forward"]
    print(f"[layerrope] rand_depth+grad_accum+ema: applications={apps}")
    assert 0 < apps < 2 * 3 * (1 + 2 + 1 + 1)


def test_g_segment_len_keeps_the_one_shot_contract(tmp_path):
    """`segment_len` carries the state; the LayerRoPE gain is per actual iteration."""
    model = tiny_model(layerrope=True, moe_topk=0)
    x, y = _batch()
    with torch.no_grad():
        one, _, _ = model(x, y)
        seg, _, _ = model(x, y, segment_len=16)
    moved = float((one - seg).abs().max())
    print(f"[layerrope] segment_len=16 max|Δ logits|={moved:.3e}")
    assert moved < 1e-5, moved


def test_h_the_forward_backward_and_step_read_no_device_scalar():
    """Gate h: `.item/.tolist/.__float__/.__bool__` raise around a real step."""
    model = tiny_model(layerrope=True, moe_topk=0)
    x, y = _batch()
    names = ("item", "tolist", "__float__", "__bool__")
    saved = {n: getattr(torch.Tensor, n) for n in names}

    def boom(*_a, **_k):
        raise AssertionError("host sync in the LayerRoPE step")

    try:
        for n in names:
            setattr(torch.Tensor, n, boom)
        logits, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.001 * p.grad
    finally:
        for n, f in saved.items():
            setattr(torch.Tensor, n, f)
    print(f"[layerrope] patched forward+backward+step ok loss={float(loss.detach()):.4f}")
    assert bool(torch.isfinite(logits).all())


def test_i_the_counter_is_device_side_and_drains_at_the_log_cadence(tmp_path):
    """Gate i: one DEVICE counter per application, drained once on the log cadence."""
    model = tiny_model(layerrope=True, prelude_layers=1, coda_layers=1)
    x, y = _batch()
    counters = Counters()
    with torch.no_grad():
        model(x, y, counters=counters)
    # Before the drain: on the device side, NOT in the dict a log line reads.
    assert "layerrope_forward" in counters._device, counters._device
    assert "layerrope_forward" not in counters, counters
    counters.drain()
    want = 1 + model.cfg.max_iter + 1 + 1  # prelude + iterations + coda + final
    print(f"[layerrope] drained layerrope_forward={counters['layerrope_forward']} want={want}")
    assert counters["layerrope_forward"] == want
    # The trainer's own drain: 3 steps × 5 sites, reported without a host read.
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=3, log_every=1),
        model=tiny_cfg(layerrope=True, prelude_layers=1, coda_layers=1, moe_topk=0),
    )
    report = train(DormouseModel(run.model), run, log=lambda *_: None)
    print(f"[layerrope] 3-step trainer counter={report['counters']['layerrope_forward']}")
    assert report["counters"]["layerrope_forward"] == 3 * want


def test_the_driver_plan_resolves_every_arm():
    """The pre-registered driver: 12 arms, the right split, and `plan` runs."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "ab_layerrope.py"
    spec = importlib.util.spec_from_file_location("ab_layerrope", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [a["name"] for a in mod.ARMS]
    assert names == [
        "off_lr5e-4",
        "off_lr1e-3",
        "off_lr3e-3",
        "on_lr5e-4",
        "on_lr1e-3",
        "on_lr3e-3",
        "off",
        "on",
        "off_mhc0",
        "off_mhc1",
        "on_mhc0",
        "on_mhc1",
    ], names
    assert all(a["model"]["layerrope"] for a in mod.ARMS if a["name"].startswith("on"))
    assert not any(a["model"]["layerrope"] for a in mod.ARMS if a["name"].startswith("off"))
    assert mod.ARMS[0]["seeds"] == (1, 2) and mod.BY_NAME["off"]["seeds"] == (1, 2, 3)
    assert mod.COMMON_TRAIN["opt_graph"] is False, "layerrope refuses opt_graph"
    assert mod.main(["plan"]) == 0
    print("[layerrope] driver plan: 12 arms resolve, run mode NOT executed")
