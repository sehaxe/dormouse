#!/usr/bin/env bash
# THE GPU gate for the fused library. One command, on the machine that has the
# GPU:
#
#     vendor/dormouse-fused/tools/gpu-gate.sh
#
# WHY IT IS A SCRIPT AND NOT A WORKFLOW. `.github/workflows/fused-library.yml`
# used to carry a `cuda-tests` job on `runs-on: [self-hosted, gpu, linux]` with
# no runner registered, so it never ran, and its `cuda-compiles` sibling only
# did `--no-run` — which proves the crate COMPILES and cannot fail on a
# CUDA-only bug. Every GPU-only claim in this library was therefore unverified,
# for a year, in a config file that read like a gate. A runner was not
# registered instead: this workstation is the only GPU the project has and it is
# also the trainer (a resumed 48M-slot engram run holds ~37 GB RSS), the
# benchmark box and the agent workstation. A GitHub runner polling it would
# take a release CUDA build next to that training run and get OOM-killed by
# `ram-guard` — a gate that is red for reasons unrelated to the diff teaches
# people to re-run it. Minting a registration token is the owner's call, not a
# code change's.
#
# This script therefore does what a runner would have done, and it fails loudly:
#   1. refuses to pass without a visible CUDA device,
#   2. sets BURN_DEVICE=cuda, without which burn-rope's and burn-situ's cuda
#      tests `return` early and report PASS having asserted nothing,
#   3. asserts the expected number of tests actually RAN, so a silently
#      cfg'd-out or `#[ignore]`d test cannot read as a pass,
#   4. reports pass/fail per test, and exits non-zero on any red.
set -uo pipefail
cd "$(dirname "$0")/.."

EXPECTED_TESTS=1

fail() { echo "FAIL  $*" >&2; exit 1; }

# 1. the device. A green run on a machine without the GPU is the exact failure
#    this script exists to end.
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: this gate needs the CUDA device"
nvidia-smi -L | grep -q . || fail "nvidia-smi sees no GPU: this gate cannot pass here"
echo "── device ──"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. the repo's own build doctrine (AGENTS.md): one heavy thing at a time. A
#    release CUDA build beside a 37 GB training run gets us OOM-killed by
#    ram-guard, and the resulting error says nothing about the code.
avail=$(awk '/^MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "── host: ${avail}G available RAM ──"
if pgrep -x train >/dev/null; then
  fail "a 'train' process is running: AGENTS.md says one heavy thing at a time. Stop it or run this later."
fi
[ "$avail" -ge 12 ] || fail "only ${avail}G available RAM; a release CUDA build needs ~12G. Close something (ram-guard will kill the heaviest process otherwise)"

export BURN_DEVICE=cuda
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12050}"

# 3+4. compile AND run AND count. `--no-run` is a compile check, not a
#      verification, and must never be reported as one.
echo "── gated_delta_chunk_path_runs_at_the_production_shape (B=10 H=12 T=512 k=v=64, chunk 64) ──"
out=$(mktemp)
cargo test -p burn-fused --release --features cuda,autodiff \
  --test gpu_production_shape -- --nocapture 2>&1 | tee "$out"
rc=${PIPESTATUS[0]}
rm -f "$out"
[ "$rc" -eq 0 ] || fail "cargo test exited $rc"

# 5. did anything actually assert? `0 passed` / `0 measured` reads as green.
ran=$(grep -oE 'test result: ok\. [0-9]+ passed' <<<"$out" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
[ "${ran:-0}" -eq "$EXPECTED_TESTS" ] \
  || fail "expected $EXPECTED_TESTS test(s) to run and pass, got ${ran:-0}. A test that silently skips must not read as a gate."

echo
echo "── what this gate does NOT cover (do not mistake it for the whole library) ──"
cat <<'EOF'
  * the other 12 cuda crates' GPU tests (burn-attnres, burn-bitnet, burn-kda,
    burn-mhc, burn-mor, burn-muon-plus, burn-rmsnorm, burn-rope, burn-sct,
    burn-situ, burn-spectral, burn-swiglu). They have never run in any
    executed job either; several are known-red (burn-spectral's three
    `retract` panics, burn-attnres self-records BROKEN), so folding them in
    now would make this gate permanently red and therefore not run.
  * burn-sct's reference comparison. `tests/cmp_reference.rs` still cannot run:
    no `tests/gen_reference.py`, no `tests/ref_data/*.bin`, and `binary-tests`
    is not a default feature. See crates/burn-sct/.gitignore.
  * a performance gate. `vendor/dormouse-fused/.github/workflows/bench.yml` was
    deleted with the rest of the never-runnable workflows: it needed a
    self-hosted GPU runner and it `git push`ed from the runner. The honest
    local equivalent is `cargo run -p burn-fused-benches --release` and reading
    the numbers; nothing compares them to bench/baselines.json automatically.
EOF
echo
echo "PASS  gpu-gate: the fused gated-delta chunk path ran on the GPU at the production shape."
