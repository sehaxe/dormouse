"""An OpenAI-shaped HTTP server over a trained checkpoint (stdlib only).

Two decisions keep this small, and both are calls into code that already
exists rather than copies of it:

* **Generation is `dormouse.model.decode`, called, not re-implemented.** The
  server renders messages into a byte prompt and asks the decoder for bytes.
  The arm is decided ONCE, before the socket binds: `recurrent_decode` (the
  state-carry, `O(1)`/byte path) when the model supports it and the caller did
  not pass `--no-recurrent`, else `decode_bytes` — a requested recurrent decode
  on a model that cannot run it falls back COUNTED (`recurrent_fallback`) with
  the reason printed at startup. A `stream=true` response calls the SAME
  function on a growing context with one seeded `torch.Generator`, so a
  streamed reply and one `max_tokens`-long call draw the identical sample
  sequence — there is no second sampler that could drift from the CLI's. (The
  recurrent arm's chunked prefill on CUDA is the one summation-order deviation:
  streamed and non-streamed stay equal within the documented class, not
  bit-for-bit; on CPU the byte prefill keeps them bit-identical.)
* **Loading is the CLI's own path, called, not copied** (`cli._snapshot_run`
  for resolution — the checkpoint's snapshot is authority, as in `eval`/
  `decode` — plus `cli._device`, the GPU preflight, so a card held by another
  process refuses at startup exactly as a training launch does).

Degradations are LOUD or COUNTED (AGENTS §1.1). A bad request is answered with
a JSON error body naming the cause and never with a silent empty completion;
a `max_tokens` above `--max-tokens-cap` is clamped and prints
`max_tokens_clamped=N` on the serve line.
"""

from __future__ import annotations

import codecs
import json
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from .instrument import Counters, LoudError, loud
from .model.decode import (
    DEFAULT_TEMP,
    decode_bytes,
    encode,
    recurrent_decode,
    recurrent_refusal,
)

#: The hard ceiling on one reply, in bytes (this model's tokens ARE bytes).
DEFAULT_MAX_TOKENS = 512

#: Bytes per SSE event. One TCP segment's worth of text keeps the stream a
#: stream; the sampling sequence is independent of it.
STREAM_BYTES = 8

#: The SFT template's markers (`train/sft.py`: `0x01` opens the user turn,
#: `0x02` the assistant turn, `0x03` closes the answer, `0x00` pads). The
#: server renders prompts in THAT template: the model was trained on it, and
#: the first chat smoke showed a `User:/Assistant:` prompt answers with noise.
TURN_OPEN, TURN_ANSWER = 0x01, 0x02
#: The model speaks no chat template beyond that: the prompt IS the dialogue.
ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}


def assemble(messages: Any) -> bytes:
    """`[{role, content}]` → the SFT-template prompt, UTF-8 bytes.

    Unknown role or non-string content is loud: a prompt the model half
    understood would produce a confident reply to a different question.

    The model is SINGLE-TURN (the SFT corpus is instruction→answer pairs), so
    the prompt is the LAST user message — with any system text and earlier
    user turns prepended as context — wrapped in the markers the model trained
    on: `\\x01 {instruction} \\n\\x02 `.
    """
    if not isinstance(messages, list) or not messages:
        loud('messages: need a non-empty list of {"role", "content"} objects')
    system: list[str] = []
    users: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            loud(f"message {m!r}: expected an object with role and content")
        role, content = m.get("role"), m.get("content")
        if role not in ROLES:
            loud(f"message role {role!r}: this model renders {sorted(ROLES)} only")
        if not isinstance(content, str):
            loud(f"message {role}: content must be a string, got {type(content).__name__}")
        if role == "system":
            system.append(content)
        elif role == "user":
            users.append(content)
        # an assistant turn is history; the corpus has no multi-turn dialogue
        # to learn from, so it is not re-fed (the last user message is the
        # instruction the model answers).
    if not users:
        loud("messages: no user turn — the model answers an instruction, and there is none")
    instruction = "\n".join([*system, *users[:-1], users[-1]])
    return bytes([TURN_OPEN]) + b" " + encode(instruction) + b" \n" + bytes([TURN_ANSWER]) + b" "


def load_checkpoint(
    out_dir: Path | str, ckpt_name: str, device: str = "auto"
) -> tuple[Any, Any, Any]:
    """Resolve through the snapshot and load the weights ONCE (`eval`'s mechanics).

    The CLI import is function-local: this module must stay importable without
    the typer/rich front, and the CLI's front imports it.
    """
    from .cli import _device, _freeze_gc, _model, _snapshot_run
    from .train.loop import load_model_half, load_model_state

    if not ckpt_name:
        loud("serve needs --ckpt-name")
    base = Path(out_dir)
    run = _snapshot_run(base, ckpt_name)
    dev = _device(device)
    model = _model(run).to(dev)
    _freeze_gc()
    # The ONE loader `eval`, `decode` and `load_ckpt` share: a storage-rung
    # checkpoint's factor records are unpacked, and an `engram_ram` checkpoint
    # switches the model to host tables and restores its `.ngram` sibling — a
    # raw `load_state_dict` died on both.
    load_model_state(model, run.train, load_model_half(base / f"{ckpt_name}.pt"))
    return model, run, dev


def load_artifact(path: Path | str, device: str = "auto") -> tuple[Any, int, Any, str]:
    """`(model, seed, device, name)` from a `.dormouse` artifact — the other source.

    The browser demo reads the same file, so a served model and a demo model are
    the same weights by construction rather than by a promise. The seed is the
    one the header carries (the run's), which is what a served reply's sampling
    reproducibility rests on.
    """
    from .cli import _device, _freeze_gc
    from .model.export import load_export, read_export

    path = Path(path)
    f = read_export(path)
    dev = _device(device)
    model = load_export(f, device=dev)
    _freeze_gc()
    return model, int(f.header.get("seed") or 1), dev, path.name


class ServeState:
    """What every request shares: the ONE model, its device, its decode arm.

    One lock covers the forwards (`torch` modules and the seeded generator are
    not concurrent), and it is held per decode call — never across a socket
    write, or one slow reader would stall every other request. `recurrent` is
    the arm `build_server` decided BEFORE the socket bound: `True` runs
    `recurrent_decode` (state-carry, `O(1)`/byte), `False` runs `decode_bytes`.
    """

    def __init__(
        self,
        model: Any,
        seed: int,
        device: torch.device,
        ckpt_name: str,
        max_tokens_cap: int,
        temp: float,
        log: Any = print,
        recurrent: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.ckpt_name = ckpt_name
        self.max_tokens_cap = max_tokens_cap
        self.temp = temp
        self.seed = seed
        self.log = log
        self.recurrent = recurrent
        self.counters = Counters()
        self.lock = threading.Lock()


def resolve_arm(model: Any, recurrent: bool) -> tuple[bool, str | None]:
    """The serve arm decision, made BEFORE the socket binds: `(use_recurrent, why_not)`.

    `recurrent=False` is the caller's choice and carries no reason; a requested
    recurrent decode on a model that cannot run it (`recurrent_refusal`)
    returns the reason, and the caller COUNTS `recurrent_fallback` and serves
    `decode_bytes` instead of failing on the first request.
    """
    if not recurrent:
        return False, None
    why = recurrent_refusal(model)
    return (True, None) if why is None else (False, why)


#: Bytes that end a reply: the SFT template's end-of-answer, plus its pad.
_STOP = frozenset({0x03, 0x00})


def _generate(state: ServeState, prompt: bytes, n: int, chunk: int = STREAM_BYTES) -> Iterator[str]:
    """`n` bytes past `prompt`, decoded incrementally, as text pieces.

    The arm is `state.recurrent`, decided once in `build_server` before the
    socket bound. Either way the SAME call is re-made on a growing context
    under ONE seeded generator: on the recurrent arm each re-call re-prefills
    the context and carries on from the state a single call would have (byte
    prefill on CPU, so byte-identical; chunked prefill on CUDA, equal within
    the documented summation-order class).
    """
    if not prompt:
        loud("empty prompt: the first window IS the prompt (model.decode's rule)")
    if n < 1:
        loud(f"max_tokens={n}: must be >= 1")
    gen = torch.Generator(device=state.device).manual_seed(state.seed)
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    ctx, done = prompt, 0
    while done < n:
        k = min(chunk, n - done)
        before = len(ctx)
        with state.lock:
            if state.recurrent:
                ctx = recurrent_decode(
                    state.model,
                    ctx,
                    k,
                    temp=state.temp,
                    device=state.device,
                    generator=gen,
                    # The SFT template's end-of-answer byte (0x03) ENDS a reply:
                    # without it the model pads a finished answer with its
                    # neighbours' tails, which is what the first chat smoke showed.
                    stop=_STOP,
                    counters=state.counters,
                )
            else:
                ctx = decode_bytes(
                    state.model,
                    ctx,
                    k,
                    temp=state.temp,
                    device=state.device,
                    generator=gen,
                    stop=_STOP,
                )
        delta = ctx[before:]
        hit = next((i for i, b in enumerate(delta) if b in _STOP), None)
        if hit is not None:
            # The reply ENDED: emit the bytes up to the marker, then stop —
            # the caller must not read a finished answer's padding as text.
            text = dec.decode(bytes(delta[:hit]))
            if text:
                yield text
            tail = dec.decode(b"", True)
            if tail:
                yield tail
            return
        done += k
        text = dec.decode(bytes(delta))
        if text:
            yield text
    tail = dec.decode(b"", True)
    if tail:
        yield tail


def _max_tokens(body: dict, state: ServeState) -> int:
    """`max_tokens`, clamped to the cap with a COUNTED log line."""
    asked = body.get("max_tokens", state.max_tokens_cap)
    if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
        loud(f"max_tokens={asked!r}: must be an int >= 1")
    if asked <= state.max_tokens_cap:
        return asked
    state.counters.count("max_tokens_clamped")
    state.log(
        f"serve: max_tokens={asked} > cap={state.max_tokens_cap} -> {state.max_tokens_cap} "
        f"{state.counters.line()}"
    )
    return state.max_tokens_cap


def make_handler(state: ServeState) -> type[BaseHTTPRequestHandler]:
    """The request handler class, bound to `state` (the server's one model)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"serve {self.address_string()} {fmt % args}")

        # -- plumbing ------------------------------------------------------

        def _json(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, code: int, msg: str) -> None:
            kind = "invalid_request_error" if code < 500 else "server_error"
            self._json(code, {"error": {"message": msg, "type": kind, "code": code}})

        def _body(self) -> Any:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return None
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError as e:
                loud(f"malformed JSON body: {e}")

        # -- routes --------------------------------------------------------

        def do_GET(self) -> None:
            if urlsplit(self.path).path.rstrip("/") == "/v1/models":
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": state.ckpt_name, "object": "model", "owned_by": "dormouse"}
                        ],
                    },
                )
            self._error(404, f"unknown path {self.path!r}: GET /v1/models")

        def do_POST(self) -> None:
            if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
                return self._error(404, f"unknown path {self.path!r}: POST /v1/chat/completions")
            try:
                body = self._body()
            except LoudError as e:
                return self._error(400, str(e))
            if not isinstance(body, dict):
                return self._error(400, "body: expected a JSON object with `messages`")
            model_id = body.get("model", state.ckpt_name)
            if model_id != state.ckpt_name:
                return self._error(
                    404,
                    f"unknown model {model_id!r}: this server serves {state.ckpt_name!r}",
                )
            try:
                n = _max_tokens(body, state)
                prompt = assemble(body.get("messages"))
            except LoudError as e:
                return self._error(400, str(e))
            cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
            if body.get("stream"):
                return self._stream(prompt, n, cid, created)
            try:
                content = "".join(_generate(state, prompt, n))
            except LoudError as e:
                return self._error(400, str(e))
            self._json(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": state.ckpt_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    # bytes ARE this model's tokens; nothing is estimated here.
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": n,
                        "total_tokens": len(prompt) + n,
                    },
                },
            )

        def _stream(self, prompt: bytes, n: int, cid: str, created: int) -> None:
            pieces = _generate(state, prompt, n)
            try:  # the first piece BEFORE the 200: a broken request still gets JSON
                first = next(pieces)
            except StopIteration:
                first = ""
            except LoudError as e:
                return self._error(400, str(e))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")  # no Content-Length: the close delimits
            self.end_headers()
            self.close_connection = True

            def event(delta: dict, finish: str | None = None) -> None:
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.ckpt_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            event({"role": "assistant", "content": first})
            for text in pieces:
                event({"content": text})
            event({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class ServeServer(ThreadingHTTPServer):
    """`state` is the server's one loaded model — the seam the tests read."""

    state: ServeState


def build_server(
    out_dir: Path | str,
    ckpt_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    device: str = "auto",
    max_tokens_cap: int = DEFAULT_MAX_TOKENS,
    temp: float = DEFAULT_TEMP,
    recurrent: bool = True,
    export: str | None = None,
    log: Any = print,
) -> ServeServer:
    """Load the model ONCE, pick the decode arm, bind, and return the ready server.

    The arm decision happens HERE, before the socket binds, so the startup
    lines name the arm that will serve every request — and a requested
    recurrent decode on a model that cannot run it is COUNTED
    (`recurrent_fallback`) with the reason, instead of failing on the first
    request. `port=0` takes an ephemeral one, so a test can run the real
    socket path.

    `export` is the OTHER source: a `.dormouse` artifact, read by the same
    reader the browser demo uses. It and `ckpt_name` are mutually exclusive at
    the CLI layer; here an `export` simply wins because the caller has already
    checked.
    """
    if max_tokens_cap < 1:
        loud(f"--max-tokens-cap {max_tokens_cap}: must be >= 1")
    if temp <= 0:
        loud(f"--temp {temp}: use a positive temperature (decode's rule)")
    if export:
        model, seed, dev, name = load_artifact(export, device)
    else:
        model, run, dev = load_checkpoint(out_dir, ckpt_name, device)
        seed, name = run.train.seed, ckpt_name
    use_recurrent, why = resolve_arm(model, recurrent)
    state = ServeState(model, seed, dev, name, max_tokens_cap, temp, log, recurrent=use_recurrent)
    if why is not None:
        state.counters.count("recurrent_fallback")
        log(
            f"serve: decode arm = plain (decode_bytes) — recurrent unavailable "
            f"({why}); {state.counters.line()}"
        )
    elif use_recurrent:
        log("serve: decode arm = recurrent (state-carry, O(1)/byte)")
    else:
        log("serve: decode arm = plain (decode_bytes; --no-recurrent)")
    httpd = ServeServer((host, port), make_handler(state))
    httpd.state = state
    return httpd


def serve_forever(httpd: ServeServer) -> None:
    """Print the line a reader of the log needs, then serve forever."""
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"serve: listening on http://{host}:{port}/v1 (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("serve: stopped")
    finally:
        httpd.server_close()
