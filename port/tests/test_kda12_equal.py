"""`configs/kda12_equal.toml` — the KDA block at the gated control's parameter budget.

The blocker this arm exists to clear: `TransformerBlock` has ONE FFN and refuses
`n_experts > 1`, so every routed arm (MoE, a TSCT expert bank, 1B-stored) lives
on `block_form="kda"` — and the KDA block's QUALITY has never been measured
against the live transformer at equal parameters and equal data.

Four gates, each able to fail on its own:

a. the parameter count is within 1% of `arch_gated_swiglu` (95,035,200), priced
   by instantiating BOTH presets on `meta` — not by a closed form;
b. the depth is 12 DISTINCT bodies each visited ONCE (`loop_blocks == max_iter
   == 12`, iteration `i` runs body `i % 12`), not one body run twelve times;
c. `ffn_gated=true` is refused on this form — the loop's `DenseFFN` already IS
   the gated GLU, which is why the config leaves the flag off rather than
   silently getting an ungated FFN;
d. the config differs from the control in the block form's own spelling and the
   FFN width, and NOTHING else — a changed `use_mhc`/`use_attnres`/data knob
   would make the A/B measure two variables at once.
"""

from __future__ import annotations

import dataclasses

import pytest

from dormouse.config import load_preset, load_run, validate
from dormouse.instrument import LoudError
from dormouse.model import DormouseModel, param_count

#: `dormouse info configs/arch_gated_swiglu.toml` — the control.
CONTROL_PARAMS = 95_035_200
#: `dormouse info configs/kda12_equal.toml` — d_ffn=1760 pays back the loop's
#: wider attention side: -118,644 = -0.125%.
KDA12_PARAMS = 94_916_556

#: Every field the two presets MAY disagree on: the block form's spelling
#: (`block_form`/`n_layers`/`loop_blocks`/`max_iter`/`use_kda`), the FFN's
#: (`ffn_form`/`ffn_gated`/`d_ffn_gelu`) and the width that pays the delta.
ALLOWED_DIFF = {
    "block_form",
    "d_ffn",
    "d_ffn_gelu",
    "ffn_form",
    "ffn_gated",
    "loop_blocks",
    "max_iter",
    "n_layers",
    "use_kda",
}


def _meta(name: str) -> DormouseModel:
    return DormouseModel(load_run(name).model, device="meta")


def test_kda12_holds_the_controls_parameter_budget_within_one_percent():
    """Gate (a): both counts instantiated, and the delta under 1%."""
    control, kda = _meta("arch_gated_swiglu"), _meta("kda12_equal")
    assert param_count(control) == CONTROL_PARAMS
    assert param_count(kda) == KDA12_PARAMS
    delta = abs(param_count(kda) - param_count(control)) / param_count(control)
    assert delta < 0.01, f"kda12 is {delta:+.2%} off the control"


def test_twelve_distinct_bodies_each_visited_once():
    """Gate (b): 12 unique layers, not one body looped twelve times."""
    cfg = load_preset("kda12_equal")
    assert (cfg.loop_blocks, cfg.max_iter) == (12, 12)
    model = _meta("kda12_equal")
    bodies = (model.loop, *model.loop_extra)
    assert len(bodies) == cfg.loop_blocks == 12
    # `LoopBlock._run`: iteration i runs `bodies[i % len(bodies)]`, so 12
    # iterations over 12 bodies visits each exactly once.
    assert [i % len(bodies) for i in range(cfg.max_iter)] == list(range(12))
    # Each body owns its own attention and FFN — shared weights would be a loop.
    assert len({id(b.attn) for b in bodies}) == 12
    assert len({id(b.moe) for b in bodies}) == 12
    # Body 0 is the only one carrying the per-iteration slots: `max_iter·d_model`.
    per_body = [sum(p.numel() for p in b.parameters()) for b in bodies]
    assert per_body[0] - per_body[1] == cfg.max_iter * cfg.d_model == 9_984, per_body


def test_ffn_gated_is_refused_on_the_kda_form():
    """Gate (c): the loop's `DenseFFN` is already gated; the flag would refuse."""
    cfg = load_preset("kda12_equal")
    assert cfg.ffn_form == "dense" and not cfg.ffn_gated
    with pytest.raises(LoudError) as e:
        validate(dataclasses.replace(cfg, ffn_gated=True, d_ffn_gelu=2640))
    msg = str(e.value)
    assert "block_form" in msg and "Escape:" in msg, msg


def test_only_the_block_form_and_the_ffn_width_differ_from_the_control():
    """Gate (d): the A/B moves one thing — the attention block, at one budget."""
    a, b = load_preset("arch_gated_swiglu"), load_preset("kda12_equal")
    diff = {f.name for f in dataclasses.fields(a) if getattr(a, f.name) != getattr(b, f.name)}
    assert diff == ALLOWED_DIFF, diff
