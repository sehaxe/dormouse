//! # RoPE on the KDA q/k path, against FLA's OWN rotary reference.
//!
//! **Tier (a), and a separate file on purpose.** The KDA oracle proper is
//! `kda_oracle.rs`; this arm is additive and touches nothing in it, because the
//! fixture that feeds both is shared and the two lanes (`wt/rope-kda2` and the
//! kda-gradflow lane) would otherwise collide on one file. The shared fixture
//! gained three `rope` blocks and **zero numeric edits to any pre-existing
//! block** — regenerating it with the RoPE section added left every existing
//! red/green verdict exactly where it was.
//!
//! ## What produced every expected number
//!
//! | what was run | where | how it is pinned |
//! |---|---|---|
//! | `fla/modules/rotary.py::rotate_half` (`:21`) and `::rotary_embedding_ref` (`:30`) — the pure-PyTorch rotary reference, twin of the Triton `rotary_embedding_kernel` in the same file | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) at `9f38d24980c46d46bd38614e743cdacd21906578` (2026-09-29) | `oracle/upstream/fla_modules_rotary.py`, byte-identical, sha256 asserted by the generator before anything is imported |
//! | `fla/modules/rotary.py::RotaryEmbedding._compute_inv_freq` (`:410`) and `._update_cos_sin_cache` (`:419`) — the cos/sin | same commit, same file | **transcribed as arithmetic**, not extracted: they are methods of a class that cannot be exec'd without triton. `gen_kda_oracle.py`'s VACUITY GUARD 4a/4b is the check on that transcription, and both halves fire. |
//! | `fla/ops/kda/naive.py::naive_chunk_kda` fed the ROTATED q/k | same commit, already pinned by the KDA oracle | same file as above |
//!
//! Fetched and run 2026-10-01, CPU only, `torch==2.14.0+cpu` (+`einops`,
//! `numpy`). Generator `oracle/gen_kda_oracle.py`, values
//! `fixtures/kda_oracle.txt`, **this test needs no network**.
//!
//! ## What the sources say, in one paragraph, because the test names it
//!
//! **FLA's official KDA layer has no RoPE at all.** `fla/layers/kda.py` at
//! `9f38d249` contains zero `rotary`/`rope` occurrences: the layer is
//! `q = F.silu(q_proj(x))` (`:248`), `k = F.silu(k_proj(x))` (`:249`), a head
//! rearrange (`:255`), then straight into `chunk_kda(..., use_qk_l2norm_in_kernel=True)`
//! (`:262`, `:272`). Position in a Kimi-Linear-style hybrid comes from the
//! *interleaved full-attention* layers instead — `fla/layers/attn.py:83,125`,
//! with `rope_theta` carried on the hybrid spec (`fla/models/hybrid.py:17-23`,
//! defaulted at `:103`) — and `get_hybrid_attention_spec`'s docstring says
//! unassigned layers "retain their model's native mixer" (`:190-200`), which is
//! the NoPE statement for the KDA half. Putting rope on a KDA layer is
//! therefore a **cross-family transplant**, justified by post-training
//! (Qwen3.8 playbook, AGENTS.md §3.5 item 5: NoPE breaks SFT/RLVR), not by
//! pretrain parity. It is off by default and adds no parameters.
//!
//! ## Tolerances
//!
//! `TOL_ROPE = 1e-5` on the rotation itself, `ATOL_ROPE = 1e-6`. Same class as
//! `kda_oracle.rs`'s `TOL_GATE`: both sides evaluate one `cos`, one `sin`, two
//! products and a difference, in the same order, on the same f32 inputs, so a
//! few ulp is the whole expectation (`f32::EPS = 1.19e-7`; 1e-5 is ~85 ulp).
//! The rotation is norm-preserving, so its output scale is the input scale
//! (O(1) here, max |q| ~ 3), which is what makes a mostly-relative comparison
//! meaningful with a small absolute floor.
//!
//! `TOL_CHUNK = 1e-3` for the end-to-end arm, **inherited unchanged** from
//! `kda_oracle.rs` and for its already-derived reason: the chunked form forms
//! `k / exp(cumsum(g))`, and over a 16-token tile with `g ∈ (-5, 0)` the
//! cumsum reaches -80, so the reciprocal reaches `e^80`. Measured worst 1.4e-4
//! on the un-rotated arm and 2.1e-4 here; the rotation does not change that
//! bound, it only changes which numbers come out.

use burn::tensor::{Device, Tensor};
use burn_kda::{apply_rope, KdaConfig, KdaModule};
use std::collections::BTreeMap;
use std::sync::Mutex;

const TOL_ROPE: f32 = 1e-5;
const ATOL_ROPE: f32 = 1e-6;
const TOL_CHUNK: f32 = 1e-3;
const ATOL_CHUNK: f32 = 1e-6;

fn dev() -> Device {
    Device::ndarray()
}

// ── fixture: the `rope` blocks only ────────────────────────────────────────

struct Block {
    fields: BTreeMap<String, String>,
}

impl Block {
    fn dims(&self) -> Vec<usize> {
        self.fields["shape"]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
    fn nums(&self, key: &str) -> Vec<f32> {
        self.fields[key]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect()
    }
}

/// A `rope` block: `rope <name>` then its `key=value` lines. The parser keeps
/// only this kind, so it cannot accidentally start reading the KDA oracle's
/// blocks and does not depend on their layout.
fn rope_blocks() -> BTreeMap<String, Block> {
    let mut out = BTreeMap::new();
    let mut key: Option<String> = None;
    let mut fields: BTreeMap<String, String> = BTreeMap::new();
    for line in include_str!("fixtures/kda_oracle.txt").lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            fields.insert(k.to_string(), v.to_string());
            continue;
        }
        if let Some(k) = key.take() {
            out.insert(k, Block { fields: std::mem::take(&mut fields) });
        }
        let mut it = t.split_whitespace();
        let kind = it.next().expect("empty block header");
        key = if kind == "rope" {
            Some(it.collect::<Vec<_>>().join(" "))
        } else {
            None
        };
    }
    if let Some(k) = key.take() {
        out.insert(k, Block { fields });
    }
    assert_eq!(out.len(), 3, "expected 3 rope blocks, parsed {}", out.len());
    out
}

fn get(rope: &BTreeMap<String, Block>, name: &str) -> Block {
    rope.get(name)
        .unwrap_or_else(|| panic!("fixture has no rope {name}"))
        .clone_owned()
}

trait CloneOwned {
    fn clone_owned(&self) -> Block;
}
impl CloneOwned for &Block {
    fn clone_owned(&self) -> Block {
        Block {
            fields: self.fields.clone(),
        }
    }
}

fn t3(v: &[f32], d: [usize; 3]) -> Tensor<3> {
    Tensor::<3, _>::from_data(burn::tensor::TensorData::new(v.to_vec(), d), &dev())
}

fn t4(v: &[f32], d: [usize; 4]) -> Tensor<4> {
    Tensor::<4, _>::from_data(burn::tensor::TensorData::new(v.to_vec(), d), &dev())
}

/// `[B, T, H, D]` (FLA's layout) -> `[B, H, T, D]` (ours).
fn bthd_to_bhtd(t: Tensor<4>, b: usize, h: usize, tt: usize, d: usize) -> Tensor<4> {
    t.permute([0, 2, 1, 3]).reshape([b, h, tt, d])
}

/// `|a-b| <= atol + rtol*|b|`, reported as the normalised ratio so every
/// assertion reads `<= 1.0`. Copied from `kda_oracle.rs` (same definition, same
/// reason for the absolute term) rather than re-derived.
fn num_diff(got: &[f32], want: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "length {} vs {}", got.len(), want.len());
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs() / (atol + rtol * w.abs());
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// Our rotation on the fixture's raw q, returned in FLA's own `[B, T, H, K]`
/// layout — which is the layout the fixture's `qrot`/`krot` rows are stored in,
/// so the comparison below is element-for-element in the same order.
///
/// `apply_rope` takes the flat `[B, T, H*HD]` that `KdaModule::project` has at
/// that point (`lib.rs`, `project`: `q_act` is `[B, T, H*HD]`), so the reshape
/// into and out of `[B, T, H, K]` is the same crossing the module does rather
/// than a convenience. **No permute before the comparison**: a permuted
/// tensor's `into_data()` is in the permuted logical order, so permuting here
/// would compare our `[B, T, H, K]` numbers against a `[B, T, H, K]` file
/// offset by the head axis, and every element would disagree.
fn our_rope(blk: &Block, which: &str) -> Tensor<4> {
    let d = blk.dims(); // B T H HV K V BT_ours BT_fla
    let (b, t, h, k) = (d[0], d[1], d[2], d[4]);
    apply_rope(t3(&blk.nums(which), [b, t, h * k]), h, k).reshape([b, t, h, k])
}

// ── 1. the rotation, against upstream's own reference ─────────────────────

/// **The load-bearing test.** `apply_rope` is ours; `qrot`/`krot` in the
/// fixture are FLA's `rotary_embedding_ref` output on the same raw q/k. A wrong
/// angle convention, a dropped factor of 2 in the frequency, an interleaved
/// half-split instead of a contiguous one, or a rotation applied across the
/// head boundary instead of within it, each moves at least one entry by O(1)
/// and goes red here.
///
/// `rope_t38` is the case that catches the head-boundary error on its own: it
/// is the only two-head case, and a rotation that treated `[B,T,H*HD]` as one
/// 16-wide vector would leave `rope_short` and `rope_t64` (both H=1, K=8,
/// T*D=128 and 512) green.
#[test]
fn rope_rotation_matches_fla_reference() {
    let rope = rope_blocks();
    let mut report = String::new();
    for name in ["rope_short", "rope_t38", "rope_t64"] {
        let blk = get(&rope, name);
        let d = blk.dims();
        assert_eq!(d[2], d[3], "the rope cases carry no GVA, so H == HV");
        for (ours, theirs) in [("q", "qrot"), ("k", "krot")] {
            let got: Vec<f32> = our_rope(&blk, ours)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let (r, at) = num_diff(&got, &blk.nums(theirs), ATOL_ROPE, TOL_ROPE);
            report.push_str(&format!("  {name} {ours}: worst normalised {r:.3e} at {at}\n"));
            assert!(
                r <= 1.0,
                "apply_rope disagrees with FLA's rotary_embedding_ref\n  \
                 case {name} {ours}: worst normalised {r:.3e} at {at} \
                 (ours {} want {})",
                got[at],
                blk.nums(theirs)[at]
            );
        }
    }
    println!("{report}");
}

/// The generator's GUARD 4b, asserted on this side of the wire: the fixture's
/// rope cases must separate from the UNROTATED answers, or the tests above are
/// blind to a rope that does nothing. It is computed here from the same blocks
/// rather than trusted, because a vacuity guard that only exists in the
/// generator cannot see a fixture edited by hand.
#[test]
fn rope_fixture_separates_from_the_unrotated_answer() {
    use burn_gdn2::ChunkPath;
    static ARM: Mutex<()> = Mutex::new(());
    let _guard = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let rope = rope_blocks();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        for name in ["rope_short", "rope_t38", "rope_t64"] {
            let blk = get(&rope, name);
            let want = blk.nums("o");
            let amax = want.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let with = run_chunk(&blk, true).0;
            let without = run_chunk(&blk, false).0;
            let sep = with
                .iter()
                .zip(&without)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / amax;
            assert!(
                sep > 1e-2,
                "rope {name} on arm {path:?}: the rotation moves the output by \
                 only {sep:.3e} relative, so a no-op rope would pass the \
                 comparison against FLA",
            );
        }
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
}

// ── 2. the chunked forward, fed our rotation ──────────────────────────────

/// The end-to-end arm: our rotated q/k through `chunk_wy_forward`, against
/// FLA's `naive_chunk_kda` fed the SAME rotated q/k.
///
/// Three shapes, each for a different reason (see `gen_kda_oracle.py`'s
/// `build_rope_cases` docstring, which is the authority):
/// `rope_short` is one tile with no boundary; `rope_t38` drives our path at
/// BT=16 over 38 tokens, so the last chunk is zero-padded
/// (`burn-gdn2/src/forward.rs:177-178`) against a BT=2 reference that is
/// nearly exact — the measured difference is therefore OUR f32 noise at a
/// non-multiple T, which is the number worth having; `rope_t64` is the
/// exact-multiple case, three boundaries, the one the padding path does not
/// cover. Both chunk arms are walked, like `kda_oracle.rs` does.
#[test]
fn chunked_wy_with_rope_matches_fla_chunk() {
    use burn_gdn2::ChunkPath;
    static ARM: Mutex<()> = Mutex::new(());
    let _guard = ARM.lock().unwrap_or_else(|e| e.into_inner());
    let rope = rope_blocks();
    let mut report = String::new();
    for path in [ChunkPath::Batched, ChunkPath::Loop] {
        burn_gdn2::set_chunk_path(path);
        for name in ["rope_short", "rope_t38", "rope_t64"] {
            let blk = get(&rope, name);
            let (got, got_s) = run_chunk(&blk, true);
            for (what, mine, theirs) in [
                ("o", &got, blk.nums("o")),
                ("S", &got_s, blk.nums("S")),
            ] {
                let (r, at) = num_diff(mine, &theirs, ATOL_CHUNK, TOL_CHUNK);
                report.push_str(&format!(
                    "  arm {path:?} case {name} {what}: worst normalised {r:.3e} at {at}\n"
                ));
                assert!(
                    r <= 1.0,
                    "chunk_wy_forward on rope'd q/k disagrees with FLA naive_chunk_kda \
                     on the same rotated q/k\n  arm {path:?} case {name} {what}: worst \
                     normalised {r:.3e} at {at} (ours {} want {})",
                    mine[at],
                    theirs[at]
                );
            }
        }
    }
    burn_gdn2::set_chunk_path(ChunkPath::Batched);
    println!("{report}");
}

/// The WIRING, at the seam the trainer uses: `KdaModule::project` with
/// `use_rope` on must emit `rope(l2(u))` where `u` is the projected
/// activation, and — because the rotation is orthogonal and the norm's
/// denominator is a function of `u` only — that is **bit-for-bit the same
/// function as `l2(rope(u))`**. So
///
/// ```text
/// project_rope(x) == apply_rope(project_off(x))
/// ```
///
/// exactly, and that identity is the whole placement argument from the module
/// docs, asserted at the level a trainer change would break. It pins three
/// things at once, and each has a distinct failure: the flag reaches
/// `project` at all; **both** q and k are rotated (rotating only q satisfies
/// the identity for q and breaks it for k, which is `falsify.sh`'s A7); and
/// the norm is applied to the rotated tensor or the un-rotated one
/// consistently — a half-applied rotation, e.g. normalising one half of the
/// head, cannot satisfy it.
///
/// The identity is about OUR two arms, so it is tier-(c) at worst and is
/// **not** a substitute for the FLA comparison above; it is here because the
/// FLA comparison cannot reach `project` (it feeds the rotation in directly,
/// since `q_proj` is a real `Linear` and pinning it would mean pinning weights
/// that are not the thing under test).
#[test]
fn module_projects_both_q_and_k_rotated() {
    use burn::module::Module;
    let dev = dev();
    let (b, t, d, h, k) = (2, 20, 32, 2, 8);
    let mut v = Vec::new();
    for i in 0..(b * t * d) {
        v.push(((i * 37 % 23) as f32 - 11.0) / 7.0);
    }
    let x = t3(&v, [b, t, d]);
    let cfg = |rope: bool| KdaConfig {
        hidden_size: d,
        num_heads: h,
        head_dim: k,
        use_short_conv: false,
        use_rope: rope,
        ..Default::default()
    };
    // TWO BUILDS OF THE SAME CONFIG ARE NOT THE SAME MODEL. `KdaModule::new`
    // initialises every projection from the device RNG and `Device::seed()`
    // does not rewind a consumed stream (AGENTS.md §3.7, the
    // `two_models_one_seed_are_bit_identical` withdrawal), so the first version
    // of this test compared two different networks and was red for exactly
    // that reason. The flag is `#[module(skip)]`, so the two records are
    // interchangeable and the weights are made equal the honest way: build
    // one, hand its record to the other.
    let off = KdaModule::new(&cfg(false), 0.0, &dev);
    let on = match KdaModule::new(&cfg(true), 0.0, &dev).try_load_record(off.clone().into_record()) {
        Ok(m) => m,
        Err(e) => panic!("the two configs must have interchangeable records: {e}"),
    };
    // `project` returns `[B, H, T, D]` (its `to_4d` permutes), while
    // `apply_rope` takes the flat `[B, T, H*HD]`. So the identity has to be
    // stated across that crossing, and both sides are compared in `[B,H,T,D]`
    // order. Getting this wrong is the same mistake twice in this file: the
    // first version of the test reshaped `project`'s `[B,H,T,D]` output
    // straight into `[B,T,H*D]`, which reads the head axis as the time axis.
    let from4 = |z: Tensor<4>| z.permute([0, 2, 1, 3]).reshape([b, t, h * k]);
    let to4 = |z: Tensor<3>| z.reshape([b, t, h, k]).permute([0, 2, 1, 3]);
    let proj = |m: &KdaModule| {
        let (q, kk, ..) = m.project_for_test(x.clone());
        (q, kk)
    };
    let (q_off, k_off) = proj(&off);
    let (q_on, k_on) = proj(&on);

    for (what, on_, off_) in [("q", &q_on, &q_off), ("k", &k_on, &k_off)] {
        let want = to4(apply_rope(from4(off_.clone()), h, k))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let got = on_.clone().into_data().to_vec::<f32>().unwrap();
        let (r, at) = num_diff(&got, &want, ATOL_ROPE, TOL_ROPE);
        assert!(
            r <= 1.0,
            "KdaModule::project with use_rope=true does not emit \
             rope(l2(u)) for {what}\n  worst normalised {r:.3e} at {at} \
             (got {} want {})",
            got[at],
            want[at]
        );
    }

    // And the arm is not a no-op: a rotation that leaves q untouched would
    // satisfy the identity above trivially, so the gap is stated.
    let qo = q_on.into_data().to_vec::<f32>().unwrap();
    let qf = q_off.into_data().to_vec::<f32>().unwrap();
    let moved = qo
        .iter()
        .zip(&qf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-3,
        "use_rope=true changed q by at most {moved:.3e}, so the arm is inert"
    );
}

// ── 3. the placement claim: rope and the L2 norm commute ──────────────────

/// **Why the arm is placed before the L2 norm, and why that is free.** RoPE is
/// a block-diagonal rotation with unit-modulus entries, so it preserves the
/// L2 norm exactly and
/// `l2(rope(x)) = rope(x)/‖rope(x)‖ = rope(x)/‖x‖ = rope(l2(x))`.
/// That is arithmetic, and this is the test for it — at OUR norm, which is
/// `l2_normalize` over the flat `H*HD` axis (`burn-gdn2/src/l2norm.rs:4-7`),
/// not a per-head one, and at the per-head norm as well, because the claim has
/// to hold for the norm this crate actually uses and for the one the reference
/// uses.
///
/// A **partial** rotation is where the order stops being free: upstream's
/// reference carries an un-rotated tail (`fla/modules/rotary.py:33-35`) and an
/// L2 norm over all head dims then mixes rotated and un-rotated coordinates.
/// `apply_rope` therefore takes no fraction, and this test is stated at
/// fraction 1.0 only.
#[test]
fn rope_commutes_with_l2norm() {
    use burn_gdn2::l2_normalize;
    let (b, t, h, k) = (2, 20, 3, 8);
    // A fixed, non-degenerate input: alternating magnitudes and signs, so no
    // entry is zero and no head is a copy of another.
    let mut v = Vec::new();
    for i in 0..(b * t * h * k) {
        v.push(((i * 37 % 23) as f32 - 11.0) / 7.0);
    }
    let x = t3(&v, [b, t, h * k]);

    // (a) the crate's own norm: `l2_normalize` over the flat H*HD axis
    //     (`burn-gdn2/src/l2norm.rs:4-7`), which is what `project` calls.
    let norm_then_rope = apply_rope(l2_normalize(x.clone(), 1e-6), h, k);
    let rope_then_norm = l2_normalize(apply_rope(x.clone(), h, k), 1e-6);
    let (r, at) = num_diff(
        &rope_then_norm.into_data().to_vec::<f32>().unwrap(),
        &norm_then_rope.into_data().to_vec::<f32>().unwrap(),
        ATOL_ROPE,
        TOL_ROPE,
    );
    assert!(
        r <= 1.0,
        "rope does not commute with our L2 norm: {r:.3e} at {at}"
    );

    // (b) PER HEAD, the norm the reference applies
    //     (`fla/ops/kda/chunk.py:56-60`, `l2norm_fwd`). The claim has to hold
    //     for the norm this crate uses AND for the reference's, or it is only
    //     half an argument.
    let to4 = |z: Tensor<3>| z.reshape([b, t, h, k]).permute([0, 2, 1, 3]);
    let from4 = |z: Tensor<4>| z.permute([0, 2, 1, 3]).reshape([b, t, h * k]);
    let per_head = |z: Tensor<4>| {
        let s = z.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-6).sqrt();
        z / s
    };
    // `to4` takes the flat `[B,T,H*HD]` form, which is what `apply_rope`
    // returns, so it serves for both directions of the identity.
    let n_then_r = to4(apply_rope(from4(per_head(to4(x.clone()))), h, k));
    let r_then_n = per_head(to4(apply_rope(x.clone(), h, k)));
    let (r, at) = num_diff(
        &r_then_n.into_data().to_vec::<f32>().unwrap(),
        &n_then_r.into_data().to_vec::<f32>().unwrap(),
        ATOL_ROPE,
        TOL_ROPE,
    );
    assert!(
        r <= 1.0,
        "rope does not commute with the per-head L2 norm: {r:.3e} at {at}"
    );

    // (c) THE MECHANISM, stated on its own: the rotation preserves each head's
    //     L2 norm. If this is what breaks, (a) and (b) are both red for a
    //     reason that has nothing to do with the norm -- which is exactly what
    //     falsify.sh's A6 (frequency loses its factor of 2) does, and why this
    //     test is green under A6 while the FLA comparison is not.
    let hn = |z: Tensor<4>| z.clone().powf_scalar(2.0).sum_dim(3).sqrt();
    let (r, at) = num_diff(
        &hn(to4(apply_rope(x.clone(), h, k)))
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        &hn(to4(x.clone())).into_data().to_vec::<f32>().unwrap(),
        ATOL_ROPE,
        TOL_ROPE,
    );
    assert!(r <= 1.0, "the rotation changed a head's L2 norm: {r:.3e} at {at}");
}

// ── 4. the zero default ───────────────────────────────────────────────────

/// `use_rope = false` must be **bitwise** today's forward, not close to it.
/// The flag is off in `KdaConfig::default()` and `KdaModule` skips it, so the
/// only thing that could change the numbers is a changed constant; this reads
/// the q/k the module actually projects and compares the raw bytes, at a shape
/// where every head is non-degenerate.
///
/// An explicit `use_rope: false` is compared against the default too, so a
/// default that drifts away from the flag cannot hide.
#[test]
fn rope_off_is_bitwise_identical() {
    use burn::module::Module;
    let dev = dev();
    let (b, t, d) = (2, 20, 32);
    let mut v = Vec::new();
    for i in 0..(b * t * d) {
        v.push(((i * 37 % 23) as f32 - 11.0) / 7.0);
    }
    let x = t3(&v, [b, t, d]);
    let qk = |m: &KdaModule| {
        let (q, k, ..) = m.project_for_test(x.clone());
        (
            q.into_data().to_vec::<f32>().unwrap(),
            k.into_data().to_vec::<f32>().unwrap(),
        )
    };
    // The DEFAULT, with the field not mentioned at all -- so if the default
    // ever drifts to true this is the arm that catches it.
    let by_default = KdaModule::new(
        &KdaConfig {
            hidden_size: d,
            num_heads: 2,
            head_dim: 8,
            use_short_conv: false,
            ..Default::default()
        },
        0.0,
        &dev,
    );
    assert!(!by_default.use_rope, "use_rope must default to false");
    // Same weights, or the comparison is between two networks. See the note in
    // `module_projects_both_q_and_k_rotated`: two builds of one config differ
    // because the init draws from the device RNG.
    let explicit = match KdaModule::new(
        &KdaConfig {
            hidden_size: d,
            num_heads: 2,
            head_dim: 8,
            use_short_conv: false,
            use_rope: false,
            ..Default::default()
        },
        0.0,
        &dev,
    )
    .try_load_record(by_default.clone().into_record())
    {
        Ok(m) => m,
        Err(e) => panic!("the two configs must have interchangeable records: {e}"),
    };
    let (qd, kd) = qk(&by_default);
    let (qe, ke) = qk(&explicit);
    assert_eq!(qd, qe, "default and explicit use_rope=false disagree on q");
    assert_eq!(kd, ke, "default and explicit use_rope=false disagree on k");
}

// ── the chunk runner ──────────────────────────────────────────────────────

/// `use_rope: true` rotates q/k with `apply_rope` before the chunked call,
/// which is the order `KdaModule::project` uses. `false` feeds the raw q/k, so
/// the same fixture block answers both the with-rope and the without-rope
/// question (guard 4b, and the falsify mutant's control). Returns
/// `(o [B,T,HV,V], final state [B,HV,K,V])`, FLA's layout.
fn run_chunk(blk: &Block, rope: bool) -> (Vec<f32>, Vec<f32>) {
    let d = blk.dims(); // B T H HV K V BT_ours BT_fla
    let (b, t, h, hv, k, v) = (d[0], d[1], d[2], d[3], d[4], d[5]);
    let bt = d[6];
    // `our_rope` already crosses the layout the module crosses: the fixture is
    // FLA's `[B,T,H,K]`, `apply_rope` takes our `[B,T,H*HD]`, and the result
    // comes back as `[B,H,T,K]` for `chunk_wy_forward`.
    let qk = |key: &str| {
        if rope {
            bthd_to_bhtd(our_rope(blk, key), b, h, t, k)
        } else {
            bthd_to_bhtd(t4(&blk.nums(key), [b, t, h, k]), b, h, t, k)
        }
    };
    let (q, kk) = (qk("q"), qk("k"));
    let vv = bthd_to_bhtd(t4(&blk.nums("v"), [b, t, hv, v]), b, hv, t, v);
    let g = bthd_to_bhtd(t4(&blk.nums("g"), [b, t, hv, k]), b, hv, t, k);
    let beta = t3(&blk.nums("beta"), [b, t, hv])
        .permute([0, 2, 1])
        .reshape([b, hv, t, 1]);
    let b_k = beta.clone().repeat(&[1, 1, 1, k]);
    let w_gate = beta.repeat(&[1, 1, 1, v]);
    let state = Tensor::<4>::zeros([b, hv, k, v], &dev());
    let (o, s) = burn_gdn2::chunk_wy_forward(q, kk, vv, g, b_k, w_gate, state, 1.0, bt);
    (
        o.permute([0, 2, 1, 3]).into_data().to_vec::<f32>().unwrap(),
        s.into_data().to_vec::<f32>().unwrap(),
    )
}
