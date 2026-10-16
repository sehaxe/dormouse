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
    run = RunConfig(
        source="tiny", model=tiny_cfg(), train=TrainConfig(ckpt_name=CKPT, out_dir=str(d))
    )
    (d / f"{CKPT}.config.toml").write_text(run.snapshot_toml())
    torch.save({"model": tiny_model().state_dict(), "opt": {}, "step": 0}, d / f"{CKPT}.pt")
    httpd = build_server(d, CKPT, port=0, device="cpu", max_tokens_cap=CAP)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=10)


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
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str) and choice["message"]["content"]
    assert body["usage"]["completion_tokens"] == CAP
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + CAP


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


def test_max_tokens_above_the_cap_is_clamped_and_counted(served):
    _, httpd = served
    before = httpd.state.counters.get("max_tokens_clamped", 0)
    body = _chat(served, max_tokens=10_000)
    assert body["usage"]["completion_tokens"] == CAP
    assert body["choices"][0]["message"]["content"]
    assert httpd.state.counters.get("max_tokens_clamped", 0) == before + 1


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


def test_clamp_line_is_counted():
    """The clamp is COUNTED, not just applied: the serve line carries the number."""
    from types import SimpleNamespace

    from dormouse.serve import ServeState, _max_tokens

    lines: list[str] = []
    state = ServeState(
        model=None,
        run=SimpleNamespace(train=SimpleNamespace(seed=1)),
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
