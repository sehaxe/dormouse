"""The OpenAI-shaped server: real socket, real requests, tiny CPU checkpoint.

The checkpoint is written directly (a `state_dict` plus its snapshot) rather
than trained first — the server's contract is the snapshot/weights pair, and a
4-step train would only re-test the trainer (`test_cli.py`'s job). The server
loads ONCE for the module, so the tests exercise the shared-model path, not a
fresh load per request.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from conftest import tiny_cfg, tiny_model

from dormouse.config import RunConfig, TrainConfig
from dormouse.instrument import LoudError
from dormouse.serve import assemble, build_server

CKPT = "tiny"
CAP = 8
MSG = [{"role": "user", "content": "Hello"}]


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory):
    """The module's one server: a checkpoint on disk, port 0, loaded once."""
    d = tmp_path_factory.mktemp("ck")
    _write_ckpt(d, tiny_cfg(), tiny_model())
    lines: list[str] = []
    httpd = build_server(d, CKPT, port=0, device="cpu", max_tokens_cap=CAP, log=lines.append)
    httpd.startup_log = lines  # the arm line the decision printed, for the tests
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=10)


def _write_ckpt(d: Path, cfg, model) -> None:
    """The snapshot + weights pair a server loads (written directly, not trained)."""
    run = RunConfig(source="tiny", model=cfg, train=TrainConfig(ckpt_name=CKPT, out_dir=str(d)))
    (d / f"{CKPT}.config.toml").write_text(run.snapshot_toml())
    torch.save({"model": model.state_dict(), "opt": {}, "step": 0}, d / f"{CKPT}.pt")


def _req(url: str, payload=None) -> urllib.request.Request:
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )


def _raw_status(req: urllib.request.Request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _chat(served, **body) -> dict:
    payload = {"messages": MSG, **body}
    req = _req(served[0] + "/v1/chat/completions", payload)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_models_endpoint_names_the_checkpoint(served):
    with urllib.request.urlopen(served[0] + "/v1/models", timeout=60) as r:
        body = json.loads(r.read())
    assert [m["id"] for m in body["data"]] == [CKPT]
    assert body["data"][0]["object"] == "model"


def test_chat_completion_has_the_openai_shape(served):
    body = _chat(served, max_tokens=CAP)
    assert body["model"] == CKPT and body["object"] == "chat.completion"
    (choice,) = body["choices"]
    produced = body["usage"]["completion_tokens"]
    # The usage reports what was PRODUCED, not the requested cap (round 1 N2).
    assert 1 <= produced <= CAP
    assert choice["finish_reason"] == ("length" if produced == CAP else "stop")
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + produced


def test_stream_is_sse_and_ends_with_done(served):
    payload = {"messages": MSG, "max_tokens": CAP, "stream": True}
    req = _req(served[0] + "/v1/chat/completions", payload)
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode()
    events = [e.strip() for e in text.split("\n\n") if e.strip()]
    assert events and events[-1] == "data: [DONE]"
    assert all(e.startswith("data: ") for e in events)
    chunks = [json.loads(e[len("data: ") :]) for e in events[:-1]]
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    # one seeded generator over a growing context == one call: same sampling path
    assert streamed == _chat(served, max_tokens=CAP)["choices"][0]["message"]["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_recurrent_is_the_default_arm_and_the_line_says_so(served):
    """The default arm is the state-carry decoder, named BEFORE the socket binds."""
    _, httpd = served
    assert httpd.state.recurrent is True
    assert any("decode arm = recurrent" in line for line in httpd.startup_log)
    # The arm is not decoration: a served request runs the torch recurrence
    # (FLA is CUDA-only, so on this CPU server every step counts `kda_step_torch`).
    before = httpd.state.counters.get("kda_step_torch", 0)
    _chat(served, max_tokens=CAP)
    assert httpd.state.counters.get("kda_step_torch", 0) > before


def test_a_model_that_cannot_run_recurrent_falls_back_counted(tmp_path: Path):
    """MSA on → the recurrent arm is refused; plain decode is COUNTED + named.

    The decision happens in `build_server`, before the socket binds, so the
    startup line a reader sees already names the arm every request will use.
    """
    d = tmp_path / "ck_msa"
    d.mkdir()
    _write_ckpt(d, tiny_cfg(use_msa=True), tiny_model(use_msa=True))
    lines: list[str] = []
    httpd = build_server(d, CKPT, port=0, device="cpu", log=lines.append)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        assert httpd.state.recurrent is False
        assert httpd.state.counters.get("recurrent_fallback") == 1
        (line,) = [line for line in lines if "recurrent_fallback" in line]
        assert "use_msa" in line and "Escape" in line
        # The fallback is usable, not just declared: the request is served.
        body = _chat((f"http://127.0.0.1:{httpd.server_address[1]}", httpd), max_tokens=CAP)
        assert body["choices"][0]["message"]["role"] == "assistant"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)


def test_no_recurrent_serves_plain_without_a_fallback(tmp_path: Path):
    """`--no-recurrent` is a choice, not a degradation: no fallback counter."""
    d = tmp_path / "ck_plain"
    d.mkdir()
    _write_ckpt(d, tiny_cfg(), tiny_model())
    lines: list[str] = []
    httpd = build_server(d, CKPT, port=0, device="cpu", recurrent=False, log=lines.append)
    try:
        assert httpd.state.recurrent is False
        assert "recurrent_fallback" not in httpd.state.counters
        assert any("--no-recurrent" in line for line in lines)
    finally:
        httpd.server_close()


def test_serve_cli_wires_the_recurrent_flag(tmp_path: Path, monkeypatch):
    """`dormouse serve --no-recurrent` reaches `build_server` (the typer seam)."""
    from typer.testing import CliRunner

    import dormouse.cli as cli

    d = tmp_path / "ck_cli"
    d.mkdir()
    _write_ckpt(d, tiny_cfg(), tiny_model())
    seen: dict[str, object] = {}

    def fake_serve(httpd):
        seen["arm"] = httpd.state.recurrent
        httpd.server_close()

    monkeypatch.setattr(cli, "serve_forever", fake_serve)
    r = CliRunner().invoke(
        cli.app,
        [
            "serve",
            "--out-dir",
            str(d),
            "--ckpt-name",
            CKPT,
            "--port",
            "0",
            "--no-recurrent",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert seen["arm"] is False
    assert "decode arm = plain" in r.output


def test_max_tokens_above_the_cap_is_clamped_and_counted(served):
    _, httpd = served
    before = httpd.state.counters.get("max_tokens_clamped", 0)
    body = _chat(served, max_tokens=10_000)
    # The cap bounds the budget; usage still reports what was produced (N2).
    assert 1 <= body["usage"]["completion_tokens"] <= CAP
    assert body["choices"][0]["message"]["content"]
    assert httpd.state.counters.get("max_tokens_clamped", 0) == before + 1


def test_usage_reports_the_bytes_produced_and_stops_at_nine(served, monkeypatch):
    """N2's gate: `max_tokens=64`, a reply that ends at 9 B, usage says 9."""
    import dormouse.serve as serve_mod

    def nine_then_stop(model, ctx, k, **kw):
        return ctx + b"123456789\x03"  # answer + the SFT end-of-answer byte

    monkeypatch.setattr(serve_mod, "recurrent_decode", nine_then_stop)
    body = _chat(served, max_tokens=64)
    choice = body["choices"][0]
    assert choice["message"]["content"] == "123456789"
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 9
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 9


def test_finish_reason_is_length_when_the_budget_runs_out(served, monkeypatch):
    """The other half of N2: `n` bytes and no stop byte is not `stop`."""
    import dormouse.serve as serve_mod

    def always_more(model, ctx, k, **kw):
        return ctx + b"x" * k

    monkeypatch.setattr(serve_mod, "recurrent_decode", always_more)
    body = _chat(served, max_tokens=CAP)
    assert body["usage"]["completion_tokens"] == CAP
    assert body["choices"][0]["finish_reason"] == "length"


def test_unknown_model_is_a_404_json_error(served):
    payload = {"model": "ghost", "messages": MSG}
    code, body = _raw_status(_req(served[0] + "/v1/chat/completions", payload))
    assert code == 404 and "ghost" in body["error"]["message"]


def test_malformed_json_is_a_400_json_error(served):
    req = urllib.request.Request(served[0] + "/v1/chat/completions", data=b"{oops", method="POST")
    code, body = _raw_status(req)
    assert code == 400 and "JSON" in body["error"]["message"]


def test_unknown_role_is_a_400_json_error(served):
    payload = {"messages": [{"role": "tool", "content": "x"}]}
    code, body = _raw_status(_req(served[0] + "/v1/chat/completions", payload))
    assert code == 400 and "tool" in body["error"]["message"]


def test_unknown_path_is_a_404_json_error(served):
    code, body = _raw_status(_req(served[0] + "/v1/nope"))
    assert code == 404 and "/v1/models" in body["error"]["message"]


def test_missing_checkpoint_refuses_loudly(tmp_path: Path):
    with pytest.raises(LoudError, match="no checkpoint pair"):
        build_server(tmp_path, "ghost", device="cpu")


def test_a_rung_checkpoint_serves(tmp_path: Path):
    """serve rides the same `load_model_half` as eval/decode (audit MAJOR 6).

    The fixture above writes an UNPACKED model half; a `quant=fp8` file carries
    packed factor records and used to die in `load_state_dict` with a
    RuntimeError naming neither the rung nor the escape.
    """
    from dormouse.model.tsct import is_factor_key, pack_tsct
    from dormouse.serve import load_checkpoint

    d = tmp_path / "ck"
    d.mkdir()
    run = RunConfig(
        source="tiny",
        model=tiny_cfg(quant="fp8"),
        train=TrainConfig(ckpt_name=CKPT, out_dir=str(d)),
    )
    (d / f"{CKPT}.config.toml").write_text(run.snapshot_toml())
    sd = tiny_model(quant="fp8").state_dict()
    packed = {k: (pack_tsct(v, "fp8") if is_factor_key(k) else v) for k, v in sd.items()}
    torch.save({"model": packed, "opt": {}, "step": 0, "model_quant_rung": "fp8"}, d / f"{CKPT}.pt")
    model, _run, _dev = load_checkpoint(d, CKPT, device="cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_clamp_line_is_counted():
    """The clamp is COUNTED, not just applied: the serve line carries the number."""
    from dormouse.serve import ServeState, _max_tokens

    lines: list[str] = []
    state = ServeState(
        model=None,
        seed=1,
        device="cpu",
        ckpt_name=CKPT,
        max_tokens_cap=CAP,
        temp=0.8,
        log=lines.append,
    )
    assert _max_tokens({"max_tokens": 99}, state) == CAP
    assert len(lines) == 1 and f"{CAP} max_tokens_clamped=1" in lines[0]
    assert _max_tokens({}, state) == CAP and len(lines) == 1  # default = the cap, no clamp


def test_assemble_renders_the_sft_template():
    """The prompt IS the SFT template: `\\x01 instruction \\n\\x02 ` — the
    model was trained on those markers, and a `User:/Assistant:` prompt was
    measured answering with noise (the first chat smoke, 2026-10-17)."""
    got = assemble([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert got == b"\x01 s\nu \n\x02 "
    # multi-turn: the last user message is the instruction, earlier ones are context
    got2 = assemble(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": "second"},
        ]
    )
    assert got2 == b"\x01 first\nsecond \n\x02 "


def test_a_stop_byte_ends_the_reply(tmp_path):
    """The SFT template's 0x03 ends a reply: the streamed and non-streamed
    content both stop at it instead of running to max_tokens."""
    from dormouse.model.decode import decode_bytes

    model = tiny_model()
    gen = torch.Generator().manual_seed(0)
    out = decode_bytes(
        model, b"\x01 hi \n\x02 ", 64, temp=1.0, generator=gen, stop=frozenset({0x03})
    )
    assert len(out) <= 64 + 8  # bounded by n_new; the early exit only shortens
    out2 = decode_bytes(model, b"\x01 hi \n\x02 ", 64, temp=1.0, generator=gen)
    assert len(out2) == 64 + len(b"\x01 hi \n\x02 ")  # default: no early exit


def test_recurrent_streaming_is_one_seeded_call():
    """The SSE re-call trick under the recurrent arm: chunked `_generate` == one call.

    `_generate` re-calls the decoder on a growing context under ONE seeded
    generator (the serve contract). On CPU the prefill is byte-by-byte, so the
    state after re-prefilling the grown context is identical to the carried
    state a single call would have — the emitted bytes must match exactly, stop
    byte included.
    """
    from dormouse.model.decode import recurrent_decode
    from dormouse.serve import _STOP, ServeState, _generate

    model = tiny_model(moe_topk=0)
    state = ServeState(
        model=model,
        seed=7,
        device=torch.device("cpu"),
        ckpt_name=CKPT,
        max_tokens_cap=64,
        temp=0.8,
        recurrent=True,
    )
    prompt, n = b"\x01 hi \n\x02 ", 24
    text = "".join(_generate(state, prompt, n))
    gen = torch.Generator().manual_seed(7)
    raw = recurrent_decode(model, prompt, n, temp=0.8, generator=gen, stop=_STOP)
    body = raw[len(prompt) :]
    if body and body[-1] in _STOP:
        body = body[:-1]  # the stop byte ends the reply and is never emitted as text
    assert text == body.decode("utf-8", "replace")
