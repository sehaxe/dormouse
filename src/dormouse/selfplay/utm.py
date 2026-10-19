"""The Brainf*ck-like universal machine `U` — arXiv:2609.30063 **Appendix E**.

Every training sequence in *Self-Play Pretraining with Zero Data* is the output
of a program executed on a fixed universal Turing machine, and the generator is
a distribution over programs for it (paper §2.1). This module is that machine,
transcribed from Appendix E ("Brainf*ck details"); §2.3 gives the alphabet.

The eight primitives (Appendix E): `<`/`>` move the head between cells, `+`/`-`
increment and decrement the cell under it, `[`/`]` loop, `,` reads a byte from
the random input tape ω, `.` emits the cell. `F` is the explicit end-of-program
token. Appendix E's Table 4 adds ten single-character macros, expanded here into
exactly the pure Brainf*ck they are defined as::

    Z [-]                     clear the cell
    R [->+<]                  clear and add x into the right neighbour
    L [->+++<]                clear and add 3x into the right neighbour
    N [-<->]                  clear and subtract x from the left neighbour
    C [->+>+<<]               clear and add x into the two right cells
    G [>]                     scan right to the next zero cell
    H [<]                     scan left to the next zero cell
    W [[-]>+<]                if the cell is non-zero: increment right, clear it
    V [.>]                    print the stored string until a zero cell
    X [-]++++++++++++++++     set the cell to 16

Four properties of Appendix E make the machine total, and each is deliberate
here rather than incidental:

1. **Every string over the alphabet is executable.** "The only way a program can
   be malformed is through unmatched brackets, which we treat as no-ops", so an
   unmatched `[` or `]` gets no jump target and falls through. Bytes outside the
   alphabet are no-ops too: the paper promises executability for `x ∈ A^{≤L}`
   only, and this module extends it to *every* byte string so that no caller can
   construct a program that faults.
2. **The tape is circular and cells are modulo `m = 256`**, so a head move off
   either end wraps and `+`/`-` wrap with it: "there are no out of bounds
   errors".
3. **Execution always terminates** — at the step budget, the end of the program,
   the `F` token, or the T-th emitted byte, whichever comes first.
4. **The output is the first T bytes emitted, zero-padded if the program halts
   early.** T is the learner's context length (the paper fixes it at 4096); it
   belongs to the batch builder that knows the window, so `run` returns the raw
   emitted bytes unless the caller passes `out_len`.

Appendix E names "a bounded step and memory budget" without a value, and never
fixes the tape size; [`TAPE_SIZE`] and [`DEFAULT_MAX_STEPS`] are this port's
choices and are parameters of `run`, not constants of the paper.

Pure Python on purpose: correctness before speed, and one function-shaped seam
([`run`]) so a Rust/PyO3 port can replace the body without touching a caller.
"""

from __future__ import annotations

__all__ = [
    "ALPHABET",
    "DEFAULT_MAX_STEPS",
    "END",
    "INSTRUCTIONS",
    "MACROS",
    "MODULUS",
    "PROGRAM_TOKENS",
    "TAPE_SIZE",
    "run",
]

#: Byte values are the tape's alphabet, so `m = 256` (Appendix E).
MODULUS = 256

#: The eight primitives of Appendix E. `,` and `.` read and emit.
INSTRUCTIONS = b"<>+-[],."

#: Appendix E's Table 4 — the ten single-character macros.
MACROS = b"ZRLNCGHWVX"

#: The tokens a mutation may insert: the 18 *executable* tokens, without `F`.
ALPHABET = INSTRUCTIONS + MACROS

#: The end-of-program token (Appendix E). Also the last token of every program.
END = b"F"

#: The generator's vocabulary (§2.3): 8 instructions + 10 macros + `F` = 19.
#: `|A| = 19` is what makes the Solomonoff prior `g0(x) = |A|^{-ℓ(x)}` a proper
#: distribution — the last token is always `F`, so the masses sum to one.
PROGRAM_TOKENS = ALPHABET + END

#: Finite memory, in cells. Appendix E does not give a value.
TAPE_SIZE = 256

#: The step budget's default. Appendix E does not give a value.
DEFAULT_MAX_STEPS = 100_000

# Appendix E's Table 4, as byte-string expansions. `bytes` keys because the
# program arrives as bytes; the expansion is what actually executes.
_EXPANSIONS = {
    ord("Z"): b"[-]",
    ord("R"): b"[->+<]",
    ord("L"): b"[->+++<]",
    ord("N"): b"[-<->]",
    ord("C"): b"[->+>+<<]",
    ord("G"): b"[>]",
    ord("H"): b"[<]",
    ord("W"): b"[[-]>+<]",
    ord("V"): b"[.>]",
    ord("X"): b"[-]++++++++++++++++",
}

_OPEN, _CLOSE, _END = ord("["), ord("]"), ord("F")


def _expand(program: bytes) -> bytes:
    """Rewrite every macro into the pure Brainf*ck of Appendix E's Table 4.

    Expansion happens BEFORE bracket matching, which is what keeps the pairing
    right: each expansion is itself balanced, so a bracket the author left
    unmatched stays unmatched and an author's pair stays a pair.
    """
    if not any(c in _EXPANSIONS for c in program):
        return program
    out = bytearray()
    for c in program:
        out += _EXPANSIONS.get(c, bytes((c,)))
    return bytes(out)


def _jump_table(code: bytes) -> list[int]:
    """For each bracket, the index of its partner; `-1` when unmatched.

    Appendix E: an unmatched bracket is a **no-op**. The stack never pairs
    across an unmatched one, so a lone `[` gets `-1` and falls through into its
    body instead of jumping, and a lone `]` gets `-1` and falls through.
    """
    jump = [-1] * len(code)
    stack: list[int] = []
    for i, c in enumerate(code):
        if c == _OPEN:
            stack.append(i)
        elif c == _CLOSE and stack:
            j = stack.pop()
            jump[j] = i
            jump[i] = j
    return jump


def _execute(
    code: bytes,
    tape_input: bytes,
    max_steps: int,
    out_len: int | None,
    tape_size: int,
) -> tuple[bytes, int]:
    """The machine's core: returns `(output, steps)`.

    The instruction pointer is the only control state, so "at the step budget,
    the end of the program, the `F` token, or the T-th emitted byte" are four
    loop conditions and nothing else. `jump[ip] == -1` (unmatched bracket) is
    the no-op case of Appendix E.
    """
    n = len(code)
    jump = _jump_table(code)
    tape = bytearray(tape_size)
    head = 0
    ip = 0
    steps = 0
    read = 0
    emitted = bytearray()
    while ip < n and steps < max_steps and (out_len is None or len(emitted) < out_len):
        c = code[ip]
        steps += 1
        if c == 0x2B:  # + increment, wrapping modulo m
            tape[head] = (tape[head] + 1) % MODULUS
        elif c == 0x2D:  # - decrement, wrapping modulo m
            tape[head] = (tape[head] - 1) % MODULUS
        elif c == 0x3E:  # > move right, wrapping around the circular tape
            head = (head + 1) % tape_size
        elif c == 0x3C:  # < move left, wrapping around the circular tape
            head = (head - 1) % tape_size
        elif c == 0x2E:  # . emit the cell
            emitted.append(tape[head])
        elif c == 0x2C:  # , read the random input tape ω
            if tape_input:
                tape[head] = tape_input[read % len(tape_input)]
                read += 1
            else:
                tape[head] = 0
        elif c == _OPEN and tape[head] == 0 and jump[ip] >= 0:
            ip = jump[ip]
        elif c == _CLOSE and tape[head] != 0 and jump[ip] >= 0:
            ip = jump[ip]
        elif c == _END:
            break
        # every other byte — a macro that survived expansion, a byte outside
        # the alphabet, an unmatched bracket — is a no-op.
        ip += 1
    if out_len is not None and len(emitted) < out_len:
        # "zero-padded if it halts early" (Appendix E).
        emitted += bytes(out_len - len(emitted))
    return bytes(emitted), steps


def run(
    program: bytes,
    tape_input: bytes = b"",
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    out_len: int | None = None,
    tape_size: int = TAPE_SIZE,
) -> bytes:
    """Execute `program` on `U` with input tape `tape_input`; return its output.

    This is the module's one entry point and the seam a Rust/PyO3 port replaces.

    `out_len` is Appendix E's T: when given, the result is exactly T bytes
    (zero-padded on an early halt, truncated at the T-th emitted byte). Left
    `None`, the result is the raw emitted bytes and T stays the batch builder's
    decision — a program that emits nothing returns `b""`, which is a bounded
    output but not a trainable sequence.

    `tape_input` is the paper's random tape ω. The paper draws ω as an i.i.d.
    uniform byte stream; a finite tape is **cycled** here so that a program may
    read any number of times and the output stays a distribution over the tape
    the caller supplied. An empty tape reads zeros.
    """
    assert max_steps >= 0, f"max_steps must be >= 0, got {max_steps}"
    assert tape_size >= 1, f"tape_size must be >= 1, got {tape_size}"
    assert out_len is None or out_len >= 0, f"out_len must be >= 0, got {out_len}"
    code = _expand(bytes(program))
    return _execute(code, bytes(tape_input), max_steps, out_len, tape_size)[0]
