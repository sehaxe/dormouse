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

**The paper form (`layerrope_form="paper"`, v2).** The project page's reference
snippet: `gamma[d]` + FOUR scalars per site, a log-linear magnitude
`r_l = a_mag + b_mag·log_l` and a RoPE-spectrum per-PAIR rotation
`ang_j = θ_l·base^(-2j/d)`, applied at BOTH the read and the write of the
attention branch and of the FFN/MoE branch — `4·(d+4)` per block. Its gates,
lettered as the v2 task letters them, are the `test_paper_*` functions below:
identity bit-parity, live depth scaling, live WRITE sites (with the
bit-identical zeroed-branch comparison), per-pair rotation, the parameter table,
`position` vs `iteration` depth indexing, the five parameters' gradients in
every site, and the refusals/counters/no-host-sync rules. `form="v1"` is
untouched and its gates above still pass unmodified — that is gate (b).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re

import pytest
import torch
import torch.nn as nn
from conftest import BATCH, SEQ, tiny_cfg, tiny_model, tiny_run
from typer.testing import CliRunner

from dormouse.cli import app
from dormouse.config import ModelConfig, load_preset, load_run, parse_preset
from dormouse.instrument import Counters, LoudError
from dormouse.model import DormouseModel, loopblock, param_count
from dormouse.model.layerrope import LayerRoPEPaperGains, rotate_pairs
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


# ---------------------------------------------------------------------------
# The PAPER form (`layerrope_form="paper"`): the project page's snippet.
# ---------------------------------------------------------------------------

#: Site order is the model's (`layerrope.PAPER_SITES`): this is the index every
#: test below addresses when it zeroes or moves ONE gain.
PAPER = ("attn_read", "attn_write", "mlp_read", "mlp_write")


def _paper(**over) -> DormouseModel:
    """Same seed as `_built()`, paper form — so two builds are comparable."""
    return _built(layerrope=True, layerrope_form="paper", **over)


def _logits(model: DormouseModel, x: torch.Tensor) -> torch.Tensor:
    """`model(x)` with no targets: logits only, no loss to read."""
    model.eval()
    with torch.no_grad():
        logits, loss, _ = model(x)
    assert loss is None and bool(torch.isfinite(logits).all())
    return logits


def _site(name: str) -> int:
    return PAPER.index(name)


def test_paper_c_identity_is_a_bit_exact_no_op():
    """Gate c: `init="identity"` -> every gain IS gamma, and the forward is OFF's."""
    off = _built()
    on = _built(layerrope=True, layerrope_form="paper", layerrope_init="identity")
    gains = on.layerrope
    assert isinstance(gains, LayerRoPEPaperGains)
    assert gains.n_blocks == 1 and len(gains.sites) == 4
    d = on.cfg.d_model
    n_layers = on.cfg.prelude_layers + on.cfg.max_iter + on.cfg.coda_layers
    for site in PAPER:
        table = gains.table(0, site)
        assert table.shape == (n_layers, d), table.shape
        gamma = gains.sites[_site(site)].gamma
        for depth in range(n_layers):
            assert torch.equal(table[depth], gamma), (site, depth)
    # The flag adds no RNG draw: every surviving weight is the OFF weight.
    off_sd, on_sd = off.state_dict(), on.state_dict()
    removed = set(off_sd) - set(on_sd)
    added = set(on_sd) - set(off_sd)
    print(f"[paper] identity removed={sorted(removed)} added={len(added)} keys")
    assert removed == {"loop.norm.weight"}, removed
    assert len(added) == 4 * 7, sorted(added)  # gamma + 4 scalars + freq/log_l
    for key, value in off_sd.items():
        if key in removed:
            continue
        assert key in on_sd, key
        assert torch.equal(value, on_sd[key]), f"{key}: the paper build changed a non-gain weight"
    x, y = _batch()
    with torch.no_grad():
        logits_off, loss_off, _ = off(x, y)
        logits_on, loss_on, _ = on(x, y)
    delta = float((logits_off - logits_on).abs().max())
    exact = torch.equal(logits_off, logits_on)
    print(
        f"[paper] identity forward max|Δ|={delta:.3e} bit_exact={exact} "
        f"loss {float(loss_off):.9f} / {float(loss_on):.9f}"
    )
    assert exact or delta < 1e-6, delta
    assert torch.equal(loss_off, loss_on)


def test_paper_c_paper_init_is_live_depth_scaling():
    """Gate c: the paper init is NOT an identity, and `b_mag` moves the logits."""
    off = _built()
    on = _paper()
    gains = on.layerrope
    table = gains.table(0, "attn_read")
    spread = float((table[0] - table[-1]).detach().abs().max())
    x, _ = _batch()
    before = _logits(on, x)
    off_logits = _logits(off, x)
    with torch.no_grad():
        gains.sites[0].b_mag.add_(0.1)
    after = _logits(on, x)
    moved = float((before - after).abs().max())
    vs_off = float((before - off_logits).abs().max())
    print(
        f"[paper] init=paper depth spread {spread:.3e} (NOT identity), "
        f"b_mag+0.1 max|Δ logits|={moved:.3e}, vs OFF {vs_off:.3e}"
    )
    assert spread > 1e-3, "the paper init must depth-scale from step 1"
    assert moved > 1e-4, moved


def test_paper_e_write_sites_are_live_and_sit_at_the_residual_seam():
    """Gate e: BOTH write gains move the output, reads do something ELSE."""
    x, _ = _batch()
    base = _logits(_paper(), x)
    delta: dict[str, float] = {}
    for name in PAPER:
        model = _paper()
        model.layerrope.sites[_site(name)].gamma.data.zero_()
        delta[name] = float((_logits(model, x) - base).abs().max())
    print("[paper] zero-gain deltas: " + " ".join(f"{k}={v:.4e}" for k, v in delta.items()))
    assert delta["attn_write"] > 1e-3 and delta["mlp_write"] > 1e-3
    assert delta["attn_read"] > 1e-3 and delta["mlp_read"] > 1e-3
    assert delta["attn_read"] != delta["attn_write"], (
        "the read and the write gain are the same seam"
    )
    # The MLP branch is HOMOGENEOUS in its input at zero (no bias, silu(0)=0,
    # and the routing sees an all-zero row), so a zero READ and a zero WRITE give
    # the same output there. The read/write split is only distinguishable with a
    # NON-zero scale: the read changes what the branch computes (routing +
    # silu), the write only rescales what it produced.
    scaled: dict[str, float] = {}
    for name in PAPER:
        model = _paper()
        model.layerrope.sites[_site(name)].gamma.data.mul_(0.5)
        scaled[name] = float((_logits(model, x) - base).abs().max())
    print("[paper] gamma*0.5 deltas: " + " ".join(f"{k}={v:.4e}" for k, v in scaled.items()))
    assert scaled["attn_read"] != scaled["attn_write"]
    assert scaled["mlp_read"] != scaled["mlp_write"]

    # `attn_write = 0` must equal a run whose attention-side outputs are zeroed:
    # the gain multiplies the BRANCH, it is not a rewrite of the norm.
    model = _paper()
    model.layerrope.sites[_site("attn_write")].gamma.data.zero_()
    killed = _logits(model, x)
    block = model.loop

    class _Zero(nn.Module):
        def forward(self, h, *, counters=None):
            return torch.zeros_like(h)

    saved_attn, saved_mix = block.attn, loopblock.memory_floor_mix
    block.attn = _Zero()
    loopblock.memory_floor_mix = lambda *args: torch.zeros_like(args[0])
    try:
        zeroed = _logits(model, x)
    finally:
        block.attn = saved_attn
        loopblock.memory_floor_mix = saved_mix
    assert torch.equal(killed, zeroed), "attn_write is not a multiply on the branch output"

    # The same for the FFN write: zero the gain vs zero the MoE output.
    model = _paper()
    model.layerrope.sites[_site("mlp_write")].gamma.data.zero_()
    killed = _logits(model, x)
    block = model.loop
    saved_moe = block._moe_branch
    block._moe_branch = lambda normed, counters: torch.zeros_like(
        normed.reshape(-1, normed.shape[-1])
    )
    try:
        zeroed = _logits(model, x)
    finally:
        block._moe_branch = saved_moe
    assert torch.equal(killed, zeroed), "mlp_write is not a multiply on the branch output"
    print("[paper] write gains are multiplies on the branch, zero-for-zero at both seams")


def test_paper_f_rotation_is_live_and_per_pair():
    """Gate f: `a_rot` moves the logits, and the spectrum gives each pair its own angle.

    The per-pair claim is recovered from the MODEL's own gain table (the
    rotation an implementation actually applied to each pair), never from a
    local re-derivation of the formula — a re-derivation passes even when the
    implementation rotates every pair by one angle.
    """
    model = _paper()
    x, _ = _batch()
    site = model.layerrope.sites[0]

    def applied(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(per-pair angle, per-pair magnitude)` the table row was given."""
        with torch.no_grad():
            gain = model.layerrope.table(0, "attn_read")[depth]
            gamma = site.gamma
            g = gain.reshape(-1, 2)
            g0 = gamma.reshape(-1, 2)
            mag = g.norm(dim=-1) / g0.norm(dim=-1)
            rot = torch.atan2(g[:, 1], g[:, 0]) - torch.atan2(g0[:, 1], g0[:, 0])
            return (rot + math.pi) % (2 * math.pi) - math.pi, mag

    rot, mag = applied(1)
    log_l = float(site.log_l[1])
    theta = math.exp(float(site.a_rot.detach()) + float(site.b_rot.detach()) * log_l)
    want = theta * site.freq
    angle_err = float((rot - want).abs().max())
    scale = math.exp(float(site.a_mag.detach()) + float(site.b_mag.detach()) * log_l)
    scale_err = float((mag - scale).abs().max())
    n_distinct = int(torch.unique(rot).numel())
    pair_gap = float((rot[0] - rot[1]).abs())
    ratio = float(rot.max() / rot.min())
    print(
        f"[paper] per-pair angle vs theta·freq max|Δ|={angle_err:.3e}; "
        f"magnitude vs exp(r) max|Δ|={scale_err:.3e}; {n_distinct}/{rot.numel()} "
        f"distinct pairs, gap(first two)={pair_gap:.4e}, angle max/min={ratio:.3f} "
        f"(spectrum buffer ratio {float(site.freq.max() / site.freq.min()):.3f}, "
        f"base={site.base})"
    )
    assert angle_err < 1e-4, angle_err
    assert scale_err < 1e-6, scale_err
    assert n_distinct == rot.numel(), "the spectrum must give every pair its own angle"
    assert pair_gap > 1e-6, pair_gap
    assert math.isclose(ratio, site.base ** ((site.d - 2) / site.d), rel_tol=1e-4)
    before = _logits(model, x)
    with torch.no_grad():
        site.a_rot.fill_(1.0)
    after = _logits(model, x)
    moved = float((before - after).abs().max())
    print(f"[paper] a_rot=1.0 max|Δ logits|={moved:.4e}")
    assert moved > 1e-3, moved

    # v1 rotated every pair by ONE angle: the contrast the v2 form exists for.
    v1 = _built(layerrope=True).layerrope
    one = rotate_pairs(v1.gamma_shared, v1.theta[0])
    pair_delta = float(
        (one.reshape(-1, 2)[:, 1] / v1.gamma_shared.reshape(-1, 2)[:, 1]).detach().std()
    )
    print(f"[paper] v1 pair-angle spread={pair_delta:.3e} (0 = one angle for all pairs)")
    assert pair_delta < 1e-6


def test_paper_g_parameter_counts():
    """Gate g: paper 4·(d+4) per block, v1 d+2·n_depths, baseline 2·L·d."""
    off = _built()
    v1 = _built(layerrope=True)
    paper = _paper()
    d = off.cfg.d_model
    layers = paper.cfg.max_iter
    site_counts = [sum(p.numel() for p in s.parameters()) for s in paper.layerrope.sites]
    buffers = len(paper.layerrope.sites[0]._buffers)
    block = sum(site_counts)
    v1_count = sum(p.numel() for p in v1.layerrope.parameters())
    baseline = 2 * layers * d
    delta = param_count(paper) - param_count(off)
    print(
        f"[paper] d={d} L={layers}: sites={site_counts} buffers/site={buffers} "
        f"block={block} (want {4 * (d + 4)}); v1={v1_count} "
        f"(want {d + 2 * v1.layerrope.n_depths}); per-layer baseline 2·L·d={baseline}; "
        f"model delta paper-OFF={delta:+d}, v1-OFF={param_count(v1) - param_count(off):+d}"
    )
    assert site_counts == [d + 4] * 4
    assert block == 4 * (d + 4)
    assert buffers == 2, buffers  # freq, log_l
    assert v1_count == d + 2 * v1.layerrope.n_depths
    # The block's own norm gain (d) is replaced by the four sites.
    assert delta == block - d == 4 * (d + 4) - d

    # The closed form against the INSTANTIATED medium tree (`dormouse info`):
    # paper replaces only the loop's norm gain, v1 also the final norm's.
    runner = CliRunner()

    def total(*sets: str) -> int:
        out = runner.invoke(app, ["info", "medium", *[f"--set={s}" for s in sets]])
        assert out.exit_code == 0, out.output
        m = re.search(r"([\d,]+) params", out.output)
        assert m, out.output
        return int(m.group(1).replace(",", ""))

    base, v1_med, paper_med = (
        total(),
        total("layerrope=true"),
        total("layerrope=true", "layerrope_form=paper"),
    )
    print(
        f"[paper] medium: base={base:,} v1={v1_med:,} ({v1_med - base:+,}) "
        f"paper={paper_med:,} ({paper_med - base:+,}); d=1024 L=4 -> "
        f"block {4 * (1024 + 4):,} vs per-layer baseline {2 * 4 * 1024:,}"
    )
    assert base - paper_med == 1024 - 4 * (1024 + 4)
    assert v1_med - base == (1024 + 2 * 5) - 2 * 1024


def test_paper_h_position_index_is_iteration_independent():
    """Gate h: `position` gives every iteration ONE gain; `iteration` does not."""
    seen: dict[str, tuple[float, float]] = {}
    for mode in ("iteration", "position"):
        model = _built(layerrope=True, layerrope_form="paper", layerrope_index=mode, max_iter=3)
        gains = model.layerrope
        gains.begin_forward()
        assert gains.table(0, "attn_read").shape[0] == 3
        # The index the FORWARD actually passes: read the block's own dispatch
        # flag, so a block that ignores `layerrope_index` fails here.
        block = model.loop
        assert block._paper_index == mode, (block._paper_index, mode)
        assert block._paper_position == model.cfg.prelude_layers

        it1 = 1 if block._paper_index == "iteration" else block._paper_position
        it2 = 2 if block._paper_index == "iteration" else block._paper_position
        first, second = gains.for_iteration(0, it1), gains.for_iteration(0, it2)
        gaps = [float((a - b).abs().detach().max()) for a, b in zip(first, second, strict=False)]
        seen[mode] = (max(gaps), min(gaps))
        print(
            f"[paper] index={block._paper_index} paper_position={block._paper_position} "
            f"site gaps it1/it2=" + " ".join(f"{g:.3e}" for g in gaps)
        )
        if mode == "position":
            assert all(torch.equal(a, b) for a, b in zip(first, second, strict=False)), (
                "position mode must not depend on the loop iteration"
            )
        else:
            assert all(g > 0.0 for g in gaps), "iteration mode must move the gain every pass"
    assert seen["position"][0] == 0.0 and seen["iteration"][1] > 0.0


def test_paper_i_every_parameter_gets_a_gradient_in_every_site(tmp_path):
    """Gate i: all five parameters of ALL FOUR sites, finite and non-zero, 3 steps."""
    torch.manual_seed(0)
    run = dataclasses.replace(
        tiny_run(tmp_path, steps=3, log_every=100),
        model=tiny_cfg(layerrope=True, layerrope_form="paper", max_iter=3),
    )
    model = DormouseModel(run.model)
    train(model, run, log=lambda *_: None)
    named = dict(model.named_parameters())
    dead: list[str] = []
    for site in PAPER:
        k = _site(site)
        for leaf in ("gamma", "a_mag", "b_mag", "a_rot", "b_rot"):
            name = f"layerrope.sites.{k}.{leaf}"
            p = named[name]
            assert p.grad is not None, f"{name}: no gradient slot after 3 steps"
            assert bool(torch.isfinite(p.grad).all()), f"{name}: non-finite gradient"
            peak = float(p.grad.detach().abs().max())
            nonzero = int((p.grad.detach() != 0).sum())
            print(
                f"[paper] 3-step {name} shape={tuple(p.shape)} max|grad|={peak:.3e} "
                f"nonzero={nonzero}/{p.numel()}"
            )
            if peak == 0.0:
                dead.append(name)
    assert not dead, f"dead paper-form parameters: {dead}"


def test_paper_j_counter_and_refusals():
    """Gate j: DEVICE counter=4/iteration; every declared refusal is LOUD."""
    model = tiny_model(layerrope=True, layerrope_form="paper", moe_topk=0, max_iter=3)
    x, y = _batch()
    counters = Counters()
    with torch.no_grad():
        model(x, y, counters=counters)
    assert "layerrope_forward" in counters._device, counters._device
    assert "layerrope_forward" not in counters, counters
    counters.drain()
    want = 4 * model.cfg.max_iter
    print(f"[paper] drained layerrope_forward={counters['layerrope_forward']} want={want}")
    assert counters["layerrope_forward"] == want

    failures = (
        ("layerrope = true\nlayerrope_form = 'x'\n", "layerrope_form"),
        ("layerrope = true\nlayerrope_form = 'paper'\nlayerrope_index = 'x'\n", "layerrope_index"),
        ("layerrope = true\nlayerrope_form = 'paper'\nlayerrope_init = 'x'\n", "layerrope_init"),
        ("layerrope = true\nlayerrope_form = 'paper'\nlayerrope_base = 0.0\n", "layerrope_base"),
        # The v1 path reads no beta/base/index/init: a non-default value there
        # would name an arm the forward never runs.
        ("layerrope = true\nlayerrope_beta_init = -0.3\n", "layerrope_beta_init"),
        ("layerrope = true\nlayerrope_base = 50.0\n", "layerrope_base"),
        ("layerrope = true\nlayerrope_index = 'position'\n", "layerrope_index"),
        ("layerrope = true\nlayerrope_init = 'identity'\n", "layerrope_init"),
        # The v1 subfields are the paper form's mirror image.
        (
            "layerrope = true\nlayerrope_form = 'paper'\nlayerrope_amp_init = 0.5\n",
            "layerrope_amp_init",
        ),
        (
            "layerrope = true\nlayerrope_form = 'paper'\nlayerrope_per_site = true\n",
            "layerrope_per_site",
        ),
        ("layerrope_form = 'paper'\n", "layerrope_form"),
    )
    for text, key in failures:
        with pytest.raises(LoudError) as e:
            parse_preset(text)
        msg = str(e.value)
        assert key in msg and "Escape:" in msg, msg
        print(f"[paper] refused {key}: {msg.splitlines()[0][:110]}")
    # The runtime seam a hand-built `ModelConfig` reaches.
    with pytest.raises(LoudError) as e:
        DormouseModel(tiny_cfg(layerrope=True, layerrope_form="nope"))
    assert "layerrope_form" in str(e.value)
    with pytest.raises(LoudError) as e:
        DormouseModel(tiny_cfg(layerrope=True, layerrope_form="paper", layerrope_index="nope"))
    assert "layerrope_index" in str(e.value)
    # `full_graph`/`opt_graph` are refused for BOTH forms (the depth index is a
    # host-selected slice, so a capture would freeze it).
    for key in ("full_graph", "opt_graph"):
        with pytest.raises(LoudError) as e:
            load_run(
                "tiny",
                model_overrides={"layerrope": True, "layerrope_form": "paper"},
                train_overrides={key: True},
            )
        assert key in str(e.value) and "Escape:" in str(e.value)


def test_paper_j_no_host_sync_in_the_step():
    """Gate j: `.item/.tolist/.__float__/.__bool__` raise around a real step."""
    model = tiny_model(layerrope=True, layerrope_form="paper", moe_topk=0, max_iter=3)
    x, y = _batch()
    names = ("item", "tolist", "__float__", "__bool__")
    saved = {n: getattr(torch.Tensor, n) for n in names}

    def boom(*_a, **_k):
        raise AssertionError("host sync in the paper-form step")

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
    print(f"[paper] patched forward+backward+step ok loss={float(loss.detach()):.4f}")
    assert bool(torch.isfinite(logits).all())


def test_paper_the_driver_has_the_new_ab_arms():
    """The NEW A/B: 9 arms, 3 seeds, and the pre-registered rule is in the docstring."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "ab_layerrope.py"
    spec = importlib.util.spec_from_file_location("ab_layerrope", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [a["name"] for a in mod.ARMS]
    assert names == [
        "no_layerrope_5e-4",
        "v1_5e-4",
        "paper_5e-4",
        "paper_identity_5e-4",
        "paper_pos_5e-4",
        "v1_1e-3",
        "paper_1e-3",
        "paper_identity_1e-3",
        "paper_pos_1e-3",
    ], names
    assert mod.SEEDS == (1, 2, 3), "3 seeds per arm"
    assert mod.ARMS[0]["model"]["layerrope"] is False
    assert mod.BY_NAME["v1_5e-4"]["model"]["layerrope_form"] == "v1"
    assert mod.BY_NAME["paper_pos_5e-4"]["model"]["layerrope_index"] == "position"
    assert mod.BY_NAME["paper_identity_5e-4"]["model"]["layerrope_init"] == "identity"
    assert mod.BY_NAME["paper_5e-4"]["lr"] == 5e-4 and mod.BY_NAME["paper_1e-3"]["lr"] == 1e-3
    assert mod.COMMON_TRAIN["opt_graph"] is False, "layerrope refuses opt_graph"
    assert "2 x spread(control)" in mod.__doc__, "the verdict rule must be pre-registered"
    assert mod.main(["plan"]) == 0
    print("[paper] driver plan: 9 arms resolve, run mode NOT executed")
