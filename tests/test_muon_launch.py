"""The Muon step's launch count: the gates on the lockstep classic chain.

The change this file pins has three claims, and each one has a way to fail:

1. **the arithmetic did not move** — the post-step state is byte-identical to a
   clean `main` worktree's, for the default arm, the `colrow` arm and a group
   that hits the Triton fallback. The hashes below were captured from a CLEAN
   `b36ef27` worktree BEFORE the edit (`/tmp/opencode/capture_muon_golden.py`,
   reproduced twice); `batched_step_main` is the same tree's step transcribed
   into `scripts/bench_muon_launch.py`, and it is checked against those hashes
   too, so the reference cannot drift into "whatever this branch does".
2. **the launch count fell** — `LaunchCounter` models the two CUDA rules a CPU
   dispatch count does not see (metadata ops enqueue nothing; `_foreach_*`
   fuses up to 64 dense same-dtype tensors and falls back to one launch per
   tensor on a mixed dtype). Both arms are measured with the same counter on
   the same model, the totals are pinned, and a revert raises the number.
3. **every fallback is named** — the per-group table's path/reason is asserted
   for the `ns_dtype` rung, the `MAX_RANK` cap and the CPU reason; the step's
   own `ns_triton_fallback` counter still counts one per refused chunk.

The lockstep chain itself has an independent gate
(`test_the_classic_lockstep_chain_is_bit_identical_to_the_per_matrix_chain`):
it compares `_classic_ns_many` against `zeropower_via_ns` across 2-D/3-D,
tall/wide/square and fp32/bf16, so a bug does not have to hide behind the
optimizer's state to be found.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from dormouse.instrument import LoudError
from dormouse.train import optim
from dormouse.train.ns_kernels import path_reason
from dormouse.train.optim import (
    MixOptimizer,
    Muon,
    _classic_ns_many,
    _foreach_add_mixed_,
    _foreach_is_cheaper,
    _stack_grads,
    zeropower_via_ns,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_bench():
    """`scripts/bench_muon_launch.py` by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "bench_muon_launch", ROOT / "scripts" / "bench_muon_launch.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()

#: The fixture: a square batched group, a tall one (transposed NS), a wide one,
#: and four singletons — one of them `(80, 80)`, past `MAX_RANK`, so the
#: fallback arm runs a shape the fused chain refuses.
SHAPES = {(64, 64): 3, (128, 64): 2, (8, 128): 2, (2, 3): 1, (96, 96): 1, (80, 80): 1}

#: `hashlib.sha256` of each parameter's bytes, then each momentum tensor's,
#: 10 steps of `Muon.batched_step_from_model`, grads from `Generator(7)`.
#: From the clean `b36ef27` worktree; params 0-9, momentum 10-19.
GOLDEN_DEFAULT = (
    "7e58f36c2b99caf853064e6ae2f4f6eed857bf3bb3292b529f7b2fd761ca8667",
    "c71e73d5efd873ed840a2e06ea812d59ea78a4a0852ac36770fa0e38580b0cd4",
    "fcd24654f131970ca7e5581afe7e4c03c56f82dee4bbdb7cca9a1c6a1e1c5d29",
    "dd7a71e7232ceceb458e25e496d66a5099e223c3ec0502b6c518a1faaec69798",
    "2d1da970d47ebc1e1c3d7aaec86b151da6bb2532d8dfce73cb61db2c2cb0862b",
    "88a92500dd407d4ae4a184ba3cccb074fb596bb8869da9b0435b259227f92437",
    "141ced3c12b7fcd92d89fe044669b1b1d8c3fe4ce376759fd475cb70724f58c9",
    "85926caf01610c0e66d54e4e5954fab975e678f7136f586a4461abd46678392f",
    "6cab7d1ee315c9730563fd240cef25195b40b1fdebd8ae163ca1ba866a55d2f6",
    "f5146d9bf050106c2732f6262461d0e33548fce38560c3f42eca7fbe90e8cac6",
    "2a05ffd54e7ebe11be79835eed47996abbb9d37fc53cb282773116831d82a0ab",
    "cebaddb7b6d4a00e07efabe2d94507b8a532db96c52223abd79caa89979fbb09",
    "88f46e05ebd8feeb4a497b4e80e2419ae76124e5de608ffcfc28cb309247e377",
    "30690082ce96c9fcde5d77ab1ba9c34466bb97d3da89322afd7eeacc8a637ebe",
    "cc70a9d694aae3882d972ecf59dd4be80821869c87c48d335dc868abeb1364c1",
    "18bb9c8b68507205ca6593efbf8833c3a3e0de8d903a1cf86db63a0db5ef34d0",
    "3a8916dc9c69fe3795299b40f1135487163b9022e03472326293c02865550c34",
    "027b5dfae9885b7c397088329dd784144e3e5d9b4ab9c009b098dd2da5be6ac5",
    "597a5845a41509f1af195c3389bea18d0b666dc734ea220688b412d72858edfc",
    "ba837068ed353e179bef7f269cd9c619428d90021243c44c5eb1d94aadecc2f9",
)

#: The same fixture with bf16 parameters, `ns_kind="triton"`, `ns_dtype="bf16"`,
#: `state_dtype="bf16"` — every chunk falls back on the dtype rung.
GOLDEN_FALLBACK = (
    "6258b82eb55a23b044e55a11110d1c2a17e23bc7b26c2058b04844fb8c08d518",
    "c7a3206932eaec1d0f5253580c93ae6c18fd95eae257ff60cab23df889a694b1",
    "404f54114812efba2c6b1f3a4a38c7040753e935a3c2d36b65f38241ceaa9111",
    "99e88bf7e4c9e83d20b996e85e0127831747b9ec672b614b75f9cdfe0988e19a",
    "bbcd16f507b54000817dc7447aac620f5ed315476e433aa06714f6ea5b9801a1",
    "45bc3388ef985ab22b8a0f1215e5f817188d106b53a27ffcd4fdf44671ec5f0f",
    "f26ba101aff698399bc5fd7590cb7803c5b90a84e4924c51cedd99ab418d4fe3",
    "da5ca57e7aa661f60936ac92dc7f6e784e0230f983dae82bd705d10d46273aeb",
    "888ecaf81ac2e3623181c483a1baab628e7e58bc3c840a4943ba0c7e80bc6d61",
    "e0cdc562e0b36718975c78db26ff411a7e5fcca6c46fb09549e8b77946b1de8f",
    "39e8faabefaf15005e9e0bae968aaa0957f56a5c6e5145ab4ab78d7458966d64",
    "52395aa3b6d4b637c48bd5b73da40144866615e7983585ba191773aef1b4e04b",
    "e16e516b781a1ed126ef74ef136fb616a462fda5cd4f052eacbd930e9f4dff8d",
    "4461f18f4390fe5904d5486fc29462be04542e4fa72bb182571c0ea10ade7c65",
    "2b15947a9337e3b3ba7c91dc7a975c0a51f5df54bd34547b22d35c0cb7a723cb",
    "d38a21af09d0cceea4525fd9895aade5ae5cf29700348b6a601a479b12c5775e",
    "5a0bfd2d3b1bdf2e2c4711acc39e61cbfcf4c9eafc1461f618160e789cfcaf84",
    "ff904d2630cfe9a10b556c7d8ba96581a47a483d5ec1226505457f938049878e",
    "cfe991a372f0bb41230dd0b3b08f4485ef5fbef856edc6bbed9c2d4bcd6d27bd",
    "2908480cb73771d7bbfaa91f060061a8ef611993409a63de5f34c4f288e7f742",
)

#: The `MixOptimizer` arm (`opt=mix`, `muon_colrow_3d="colrow"`, AdamW on a 1-D
#: parameter and a name-routed `embed`): params + every state tensor, 5 steps.
GOLDEN_MIX = (
    "param p0: 06e4393f02e2b50ef88559e614e6c9472260a0002166d5ec1673773a2607f4d5",
    "state p0.momentum: 95f15048cf65507371b3a7c4150afdc51020df7bc190e17e727510d1716c0fc7",
    "param p1: 9143ec68ca61081212eaaa05cc3065aa70874d8e49e60933fae9b8ef41713b09",
    "state p1.momentum: e43fc43d0e9eab00768f03241a0719a6c207baf457147f596ad6e8fafce5426d",
    "param p2: 6f5c06bf3382c59472b184e3e546f8e2d9519ee8cfeb3a03c0389ba64695a2a5",
    "state p2.momentum: 89d22d5a81c3d53adb1b34695187a4b1d6e91d955e709213c95e41fd029e48f4",
    "param p3: c180dad6835b9284d1ed01447aab34fe7e0f543b3641054ad38f2da347558583",
    "state p3.momentum: 5f353c8f97eac05c26203112868732f8fb648ee44ff7da2b175dbae6a78df06b",
    "param p4: a6611b548ef042215266836abaf821f284b7cbd0cbe8060d6b718eb7bc93f684",
    "state p4.momentum: d73d7c3c4b83be6e5353266c9d4a789e388c3d0bfe8f28bb22e15717b7842557",
    "param p5: 6d3440dcf9487e480baeb2d7fed1d6292946a0cc238203824cf77d355e3a6049",
    "state p5.exp_avg: 98569de1ef0f40ba42581a38188bdc37957dc196b76690faf3a163d501f8e46e",
    "state p5.exp_avg_sq: 6df0154a5d52619200cef121befe7432aa567525b2f1a241739f302ef2f94d8c",
    "state p5.step: fca31f1667a6aa1bba12fca4e4ea1becd503379d80da3213af07f6cc5702828d",
    "param embed: a1aed08e84cdaa7b841e4822727f121eda330145e3d8c4ab85f56940f754efe4",
    "state embed.exp_avg: fbfb04027a730f7b738ac56c34ed2cfb6afabdd39bdc6af26d2213a8fbd0424a",
    "state embed.exp_avg_sq: a5e52569967ca44994cc4b1f699e7f0188b000f11fef069de71eef3b0c36ff0a",
    "state embed.step: fca31f1667a6aa1bba12fca4e4ea1becd503379d80da3213af07f6cc5702828d",
)

#: The measured launch totals on this fixture (both arms, same counter):
#: default fp32 and the bf16 fallback arm. Pinned so a revert cannot pass.
MAIN_LAUNCHES = {"default": 317, "fallback": 358}
NEW_LAUNCHES = {"default": 192, "fallback": 232}


def shape_model(shapes, dtype=torch.float32, seed: int = 0) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    i = 0
    for shape, count in shapes.items():
        for _ in range(count):
            p = torch.randn(*shape, generator=gen) * 0.05
            mod.register_parameter(f"p{i}", nn.Parameter(p.to(dtype)))
            i += 1
    return mod


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def run_hashes(muon: Muon, model: nn.Module, steps: int, dtype) -> tuple[str, ...]:
    gen = torch.Generator().manual_seed(7)
    for _ in range(steps):
        for p in model.parameters():
            p.grad = (torch.randn(*p.shape, generator=gen) * 0.1).to(dtype)
        muon.batched_step_from_model(model)
    return tuple(
        sha(t)
        for t in list(model.parameters()) + [muon.state[p]["momentum"] for p in model.parameters()]
    )


def grads(model: nn.Module, dtype=torch.float32, seed: int = 7) -> None:
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = (torch.randn(*p.shape, generator=gen) * 0.1).to(dtype)


# --- 1. the arithmetic --------------------------------------------------------


@pytest.mark.parametrize("dtype,ns", [(torch.float32, "fp32"), (torch.bfloat16, "bf16")])
@pytest.mark.parametrize(
    "shape", [(3, 64, 128), (3, 128, 64), (3, 64, 64), (64, 128), (128, 64), (64, 64)]
)
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_the_classic_lockstep_chain_is_bit_identical_to_the_per_matrix_chain(
    dtype, ns, shape, steps
):
    torch.manual_seed(0)
    x = (torch.randn(*shape) * 1e-3).to(dtype)
    ref = zeropower_via_ns(x, steps, "classic", ns)
    got = _classic_ns_many([x], steps, ns)[0]
    assert torch.equal(ref, got), f"{dtype} {shape} steps={steps} moved the chain"


def test_the_lockstep_chain_is_the_per_matrix_chain_on_a_MIXED_shapes_list():
    torch.manual_seed(0)
    xs = [
        torch.randn(3, 64, 128) * 1e-3,
        torch.randn(5, 8, 128) * 1e-3,
        torch.randn(64, 128) * 1e-3,
    ]
    refs = [zeropower_via_ns(x, 5, "classic") for x in xs]
    gots = _classic_ns_many(xs, 5)
    assert all(torch.equal(a, b) for a, b in zip(refs, gots, strict=True))


def test_the_foreach_route_is_only_taken_when_it_saves_a_launch():
    """One big chunk is 2 launches through foreach and 1 through `x * b`.

    The 513-tensor medium group's chunk is 33.5M elements; routing it through
    `_foreach_mul` would ADD a launch. The many-small-matrix batch is the case
    the route exists for.
    """
    assert not _foreach_is_cheaper([33_554_432])
    assert not _foreach_is_cheaper([33_554_432, 33_554_432])
    assert _foreach_is_cheaper([4096] * 7)
    assert _foreach_is_cheaper([8_388_608] * 8)
    assert _foreach_is_cheaper([1] * 64)
    assert not _foreach_is_cheaper([])


def test_the_memory_cap_splits_the_batch_without_moving_a_value(monkeypatch):
    """The lockstep batch is capped; the split must keep VALUES and ORDER.

    The cap is patched small so the fixture forces three sub-batches with the
    biggest-first sort REORDERING the work: the outputs still land at their
    input indices and are bit-identical to the per-matrix chain.
    """
    monkeypatch.setattr(optim, "MUON_CHUNK_ELEMS", 3 * 64 * 64)
    torch.manual_seed(0)
    xs = [
        torch.randn(2, 64, 64) * 1e-3,
        torch.randn(4, 64, 64) * 1e-3,
        torch.randn(64, 64) * 1e-3,
        torch.randn(1, 64, 64) * 1e-3,
    ]
    refs = [zeropower_via_ns(x, 2, "classic") for x in xs]
    gots = _classic_ns_many(xs, 2)
    assert all(torch.equal(a, b) for a, b in zip(refs, gots, strict=True))


@pytest.mark.local_pin
def test_the_default_step_is_byte_identical_to_clean_main():
    model = shape_model(SHAPES)
    muon = Muon(model.parameters(), lr=1e-3, colrow_3d="colrow")
    assert run_hashes(muon, model, 10, torch.float32) == GOLDEN_DEFAULT


@pytest.mark.local_pin
def test_the_fallback_step_is_byte_identical_to_clean_main():
    model = shape_model(SHAPES, dtype=torch.bfloat16)
    muon = Muon(
        model.parameters(),
        lr=1e-3,
        ns_kind="triton",
        ns_dtype="bf16",
        state_dtype="bf16",
        colrow_3d="colrow",
    )
    assert run_hashes(muon, model, 10, torch.bfloat16) == GOLDEN_FALLBACK


@pytest.mark.parametrize(
    "dtype,kwargs,golden",
    [
        (torch.float32, {"colrow_3d": "colrow"}, GOLDEN_DEFAULT),
        (
            torch.bfloat16,
            {"ns_kind": "triton", "ns_dtype": "bf16", "state_dtype": "bf16"},
            GOLDEN_FALLBACK,
        ),
    ],
    ids=["default", "fallback"],
)
@pytest.mark.local_pin
def test_the_transcribed_main_step_is_byte_identical_to_the_same_golden(dtype, kwargs, golden):
    """The reference is main, not this branch: if it drifts, the comparison lies."""
    model = shape_model(SHAPES, dtype=dtype)
    muon = Muon(model.parameters(), lr=1e-3, **kwargs)
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        for p in model.parameters():
            p.grad = (torch.randn(*p.shape, generator=gen) * 0.1).to(dtype)
        bench.batched_step_main(muon, model)
    got = tuple(
        sha(t)
        for t in list(model.parameters()) + [muon.state[p]["momentum"] for p in model.parameters()]
    )
    assert got == golden


@pytest.mark.local_pin
def test_the_mix_arm_is_byte_identical_to_clean_main():
    gen = torch.Generator().manual_seed(0)
    model = nn.Module()
    for name, shape in (
        ("p0", (64, 64)),
        ("p1", (64, 64)),
        ("p2", (64, 64)),
        ("p3", (128, 64)),
        ("p4", (8, 128)),
    ):
        model.register_parameter(name, nn.Parameter(torch.randn(*shape, generator=gen) * 0.05))
    model.register_parameter("p5", nn.Parameter(torch.randn(16, generator=gen) * 0.05))
    model.register_parameter("embed", nn.Parameter(torch.randn(8, 8, generator=gen) * 0.05))
    opt = MixOptimizer(model, lr=1e-3, wd=0.01, opt="mix", colrow_3d="colrow")
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        for p in model.parameters():
            p.grad = torch.randn(*p.shape, generator=g) * 0.1
        opt.step()
    out = []
    for name, p in model.named_parameters():
        out.append(f"param {name}: {sha(p)}")
        st: dict = {}
        if p in opt.muon.state:
            st.update(opt.muon.state[p])
        if p in opt.adamw.state:
            st.update(opt.adamw.state[p])
        for k, v in sorted(st.items()):
            if torch.is_tensor(v):
                out.append(f"state {name}.{k}: {sha(v)}")
    assert tuple(out) == GOLDEN_MIX


# --- 2. the launch count ------------------------------------------------------


def measure(arm: str, dtype: torch.dtype, **kw) -> int:
    model = shape_model(SHAPES, dtype=dtype)
    muon = Muon(model.parameters(), lr=1e-3, **kw)
    grads(model, dtype)
    c = bench.LaunchCounter()
    with c:
        if arm == "main":
            bench.batched_step_main(muon, model)
        else:
            muon.batched_step_from_model(model)
    return c.launches


def test_the_launch_count_is_below_main_and_pinned():
    old = measure("main", torch.float32, colrow_3d="colrow")
    new = measure("new", torch.float32, colrow_3d="colrow")
    print(f"[default] launches {old} -> {new}")
    assert old == MAIN_LAUNCHES["default"], f"the reference moved: {old}"
    assert new == NEW_LAUNCHES["default"], f"the new step moved: {new}"
    assert new < old


def test_the_fallback_arms_launch_count_is_below_main_and_pinned():
    old = measure("main", torch.bfloat16, ns_kind="triton", ns_dtype="bf16", state_dtype="bf16")
    new = measure("new", torch.bfloat16, ns_kind="triton", ns_dtype="bf16", state_dtype="bf16")
    print(f"[fallback] launches {old} -> {new}")
    assert old == MAIN_LAUNCHES["fallback"], f"the reference moved: {old}"
    assert new == NEW_LAUNCHES["fallback"], f"the new step moved: {new}"
    assert new < old


def test_the_step_counts_every_refused_chunk():
    """The lockstep batch must not swallow the fallback counter (ADR-0019)."""
    model = shape_model(SHAPES, dtype=torch.bfloat16)
    muon = Muon(model.parameters(), lr=1e-3, ns_kind="triton", ns_dtype="bf16", state_dtype="bf16")
    grads(model, torch.bfloat16)
    muon._build_groups()
    n_jobs = sum(len(bench.chunk_bounds(g.shape, len(g.params))) for g in muon._groups)
    muon.batched_step_from_model(model)
    assert muon.counters["ns_triton_fallback"] == n_jobs, dict(muon.counters)
    assert "ns_dtype" in muon.counters.why


# --- 3. the table names every path --------------------------------------------


def test_the_table_prints_every_group_and_its_path():
    model = shape_model(SHAPES, dtype=torch.bfloat16)
    muon = Muon(model.parameters(), lr=1e-3, ns_kind="triton", ns_dtype="bf16", state_dtype="bf16")
    rows = bench.table(muon, "new")
    print("\n" + "\n".join(str(r) for r in rows))
    assert len(rows) == len(SHAPES)  # one row per shape group, singletons included
    assert all(path == "classic" for _, _, _, path, _, _ in rows)
    assert all("ns_dtype" in why for _, _, _, _, why, _ in rows)


def test_every_refusal_reason_is_the_one_the_chain_reports():
    assert "ns_dtype" in path_reason(torch.empty(64, 64), "bf16")
    assert "MAX_RANK" in path_reason(torch.empty(80, 80), "fp32")
    assert "not CUDA" in path_reason(torch.empty(64, 64), "fp32")
    assert path_reason(torch.empty(64, 64), "fp32") is not None  # CPU: never eligible here


def test_a_classic_arm_says_so():
    model = shape_model(SHAPES)
    muon = Muon(model.parameters(), lr=1e-3)
    rows = bench.table(muon, "new")
    assert all(path == "classic" and "ns_kind" in why for _, _, _, path, why, _ in rows)


# --- 4. LOUD refusals ---------------------------------------------------------


def test_mixed_gradient_dtypes_are_refused_not_promoted():
    a = nn.Parameter(torch.zeros(4, 4))
    b = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    a.grad = torch.zeros(4, 4)
    b.grad = torch.zeros(4, 4, dtype=torch.bfloat16)
    with pytest.raises(LoudError, match="mixed dtypes"):
        _stack_grads([a, b])
    with pytest.raises(LoudError, match="mixed dtypes"):
        optim._add_grads_(torch.zeros(2, 4, 4), [a, b], 0.05)


def test_mixed_parameter_dtypes_are_refused_not_promoted():
    a = nn.Parameter(torch.zeros(4, 4))
    b = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(LoudError, match="mixes parameter dtypes"):
        _foreach_add_mixed_([a, b], torch.zeros(2, 4, 4))


def test_an_unknown_update_dtype_is_refused():
    p = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(LoudError, match="fp32-update"):
        _foreach_add_mixed_([p], torch.zeros(1, 4, 4, dtype=torch.float16))


# --- 5. the singleton tail is the per-parameter definition --------------------


def test_a_singleton_group_keeps_step_one_arithmetic():
    """The lockstep batch must not route a one-member shape through 3-D ColRow."""
    shapes = {(2, 3): 1, (96, 96): 1, (80, 80): 1}
    a = shape_model(shapes)
    b = shape_model(shapes)
    b.load_state_dict(a.state_dict())
    oa = Muon(a.parameters(), lr=1e-3)
    ob = Muon(b.parameters(), lr=1e-3)
    grads(a)
    grads(b)
    ob.batched_step_from_model(b)
    oa._build_groups()
    for grp in oa._groups:
        for p in grp.params:
            oa._step_one(p, 1e-3, 0.95, 5, max(1.0, grp.shape[0] / grp.shape[1]) ** 0.5)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters(), strict=True))
    assert all(
        torch.equal(oa.state[p]["momentum"], ob.state[q]["momentum"])
        for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )
