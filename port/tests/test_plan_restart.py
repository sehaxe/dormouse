"""The restart planner is a command and a budget, not prose.

The planner builds the model on `meta` only when `plan()` is called; these tests
pin the pure parts — command shape, the E/param budget and the size warning.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_command_is_the_settled_recipe_for_two_e_values():
    plan = _load("plan_restart")
    one = plan.build_command("configs/m400.toml", 512, "triton", "prop")
    two = plan.build_command("configs/m400.toml", 256, "triton", "prop")

    assert "--resume" not in one
    assert "dormouse train configs/m400.toml" in one
    for flag in (
        "batch=8",
        "seq_len=1024",
        "grad_accum=1",
        "lr=5e-4",
        "decay_frac=0.4",
        "min_lr_frac=0.01",
        "use_msa=false",
        "use_csa=true",
        "csa_chunk=8",
        "csa_coarse_chunk=256",
        "use_mhc=true",
        "use_attnres=true",
        "share_u=true",
        "use_engram=true",
        "engram_ram=false",
        "attn_latent_rank=256",
        "moe_kernel=triton",
        "doc_masking=false",
        "layerrope=true",
        "layerrope_form=paper",
        "layerrope_index=iteration",
        "layerrope_init=paper",
        "rank=64",
        "seed=1",
        "opt_graph=false",
        "iter_checkpoint=false",
        "moe_lb_mode=prop",
    ):
        assert f"--set {flag}" in one, flag

    left, right = one.split(), two.split()
    assert len(left) == len(right)
    diff = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
    assert diff == [("n_experts=512", "n_experts=256")]


def test_the_eta_is_a_number():
    plan = _load("plan_restart")
    params = 199_275_551
    b = plan.budget(params)
    assert b["chinchilla_tokens"] == 20 * params
    assert b["one_pass_tokens"] == 22_200_000_000
    assert isinstance(b["eta_s"], float) and b["eta_s"] > 0
    assert isinstance(b["one_pass_eta_s"], float) and b["one_pass_eta_s"] > 0


def test_the_oversized_bank_warns_only_above_four_times_live():
    plan = _load("plan_restart")
    warning = plan.oversize_warning(512, 80)
    assert warning and "WARNING" in warning and "6.4x" in warning
    assert plan.oversize_warning(512, 128) is None  # exactly 4x is allowed
    assert plan.oversize_warning(512, None) is None
