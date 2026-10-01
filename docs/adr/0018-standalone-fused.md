# ADR-0018 — dormouse-fused is a standalone project, with three hard rules

Date: 2026-09-27. Status: accepted, partially landed.

## The decision

`dormouse-fused` (currently `vendor/dormouse-fused/`, 26 crates, all ours) is a
**standalone project**, not a dependency of the model. It has its own
workspace, its own test suite, its own gate, and its own benchmarks. Work on it
does not require dormouse to compile, and a change in it must not be able to
break the model's build.

The concrete failure that forced this: a half-written instrumentation file in
`dormouse-gdn2` (`alloc_trace.rs`, two compile errors) made `cargo check
-p dormouse-core` fail, which blocked twelve agents' commits and made the model
unbuildable for the duration. A technology library must not be able to take the
model down.

Concretely:
- every crate's tests run from inside the fork's own workspace, against its own
  reference, with no `dormouse-*` crate in the dependency graph;
- the fork's gate is `cargo test` over its own members, not dormouse's suite;
- the model consumes the library; the library never knows the model exists.

## The three hard rules

### 1. Strictly per the research paper, and bit-for-bit against the original source

Every mechanism is a port of a published method, and the port is verified
bit-for-bit (or to a stated tolerance with the tolerance justified) against the
ORIGINAL implementation wherever the authors shipped code: FLA for the
gated-delta / linear-attention family, the authors' repos for everything else.
Where no reference code exists, the crate says so in its doc comment instead of
implying verification it does not have.

This is a rule about honesty as much as about numerics: a doc comment claiming a
fused kernel, a published result, or a bit-for-bit harness is a claim that must
be true or absent. The inventory found at least one crate whose headline numbers
are verbatim-correct from a paper's table but omit the number that actually
governs our A/B, and one whose doc comment justifies a cache that its own call
pattern thrashes.

### 2. Zero host-device synchronization

No CPU in the hot path. No `try_into_scalar`, no `into_data`, no
`blocking_read`, no `try_for_each`, no host-side branch on a device value - not
in a fused kernel's forward or backward, and not in the training step. Every
host-visible quantity is produced by a device counter or a device flag and read
at a declared cadence (log steps, checkpoint steps), never per step.

This is not a micro-optimization. A per-step scalar read was added to the NaN
firewall on 2026-09-27 and removed the same day, because it serializes the CPU
against the GPU on every step, and the whole workload is launch-bound: the
firewall now masks the loss on device, sanitizes the gradients on device, and
derives its own accounting from the gradient norm the host already reads at log
cadence. Where a decision genuinely depends on a device value, the decision is
made ON the device (see the zero-gradient rule in `dormouse-muon-plus`: a zero
gradient means zero update, computed with a `mask_fill` rather than a host
branch).

### 3. Beat the original on speed and on VRAM

Bit-for-bit identical output is the FLOOR, not the goal. Each crate carries a
benchmark against its reference on the shapes we actually run (batch 10 x seq
512, 12 heads, K=V=64, chunk 16, fp32 and bf16), reporting step time and peak
VRAM. A port that matches the reference's numbers exactly has not earned its
existence; the target is fewer bytes moved and fewer kernel launches than the
reference on the same output.

The first measured case is already the argument for the rule: our gated-delta
path allocates 17 fresh tensors / 248 MB of scratch per call, ~1 GB per
training step, against a pool that is high-water and never frees - which is why
KDA is ~80% of a step while the fused kernels themselves have an ~80-100 us
design floor. The reference (FLA) avoids materialising a 63 MB per-chunk state
export by recomputing it. Matching the reference's arithmetic while keeping its
avoidable memory traffic is a 200x effect, and no amount of kernel tuning
reaches it.

## What this forbids

- Writing a kernel before the roofline arithmetic says the kernel is the
  bottleneck (documented: `docs/research/2026-09-27-kda-sota-ceiling.md` - the op is
  memory bound at 3.1 FLOP/byte against a machine balance of 24).
- Implementing a mechanism that already exists in the fork. `dormouse-mor` and
  `dormouse-attnres` were both fully implemented while two agents were about to
  write them from scratch.
- Touching `dormouse-core` from a library crate, in any direction.
