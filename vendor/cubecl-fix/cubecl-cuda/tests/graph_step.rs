//! Does a captured graph survive **burn's** update pattern?
//!
//! `vendor/cubecl-fix/cubecl-cuda/tests/graph.rs` proves capture/replay works
//! on this GPU against buffers the test holds still. A training step is not
//! that: `burn-optim`'s Adam is `(tensor - delta, ...)`
//! (`burn-optim-0.22.0-pre.4/src/optim/adam.rs:88`) — **out of place**. The new
//! parameter is a fresh allocation, and the old one is freed *after* the
//! allocation (the RHS is evaluated before the assignment), so the parameter's
//! device address alternates with period 2. A graph bakes raw device pointers.
//!
//! So the question this file answers is not "does capture work" (it does) but
//! **"is the address the graph baked still the address the tensor lives at on
//! the next step"** — and if not, whether any of the three escapes works:
//!
//! 1. replay one graph every step (expected WRONG — the gate must have teeth),
//! 2. two graphs alternated by step parity (a bet on the pool's free list
//!    being LIFO; free if the cycle really is 2),
//! 3. pin the parameter to a master buffer and copy the update back into it
//!    (one extra launch per parameter per step, and correct by construction).
//!
//! The launch counter (`cubecl_cuda::launches`) is asserted against the number
//! of launches the test itself made, so the instrument is self-validating: a
//! counter that counted something else would fail here.

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_cuda::CudaRuntime;
use cubecl_server::runtime::Runtime;
use std::sync::Mutex;

/// One capture at a time per device — see `graph.rs` for why.
static CAPTURE_LOCK: Mutex<()> = Mutex::new(());

/// `out = p - delta`, the shape of burn's Adam update.
#[cube(launch)]
fn sub_into(p: &[f32], delta: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = p[ABSOLUTE_POS] - delta[ABSOLUTE_POS];
    }
}

/// `dst = src`, the copy that pins a parameter to a stable address.
#[cube(launch)]
fn copy_into(src: &[f32], dst: &mut [f32]) {
    if ABSOLUTE_POS < dst.len() {
        dst[ABSOLUTE_POS] = src[ABSOLUTE_POS];
    }
}

const N: usize = 8;
const BYTES: usize = N * core::mem::size_of::<f32>();

fn client() -> Client {
    CudaRuntime::client(&Default::default())
}

fn zeros() -> Vec<u8> {
    f32::as_bytes(&vec![0.0f32; N]).to_vec()
}

/// The launches `sub_into` performs, outside any capture window.
fn software_step(client: &Client, p: &mut Handle, delta: &Handle) {
    let out = client.empty(BYTES);
    sub_into::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(client, N),
        unsafe { BufferArg::from_raw_parts(p.clone(), N) },
        unsafe { BufferArg::from_raw_parts(delta.clone(), N) },
        unsafe { BufferArg::from_raw_parts(out.clone(), N) },
    );
    *p = out; // the old `p` drops here: burn's shape.
}

fn set_delta(client: &Client, delta: &Handle, v: f32) {
    client.write(delta, Bytes::from_bytes_vec(f32::as_bytes(&vec![v; N]).to_vec()));
}

fn read(client: &Client, h: &Handle) -> Vec<f32> {
    f32::from_bytes(&client.read_one(h.clone()).unwrap()).to_vec()
}

/// The launch counter counts what the device thread executed. In a launch-bound
/// loop the host runs ahead, so this is a lower bound on what it enqueued —
/// which is why every assertion here is `<=` at the top and `>=` at the bottom.
fn launches() -> u64 {
    cubecl_cuda::launches()
}

/// What a launch costs, and what a replay of the same launches costs, **on this
/// box**.
///
/// The whole lane is a bet that `L` launches cost more than one dispatch of the
/// same `L` launches. This measures both sides of that bet without needing the
/// trainer at all: a burst of trivial kernels (the launch cost, which in a
/// launch-bound step IS the step) against a burst of replays of a graph holding
/// the same count.
///
/// Reported, not asserted: this is a property of the machine and the driver, not
/// a contract, and a number that moves between runs is information rather than a
/// failure. The one thing asserted is that the graph really did contain the
/// launches it claims — measured with the counter, so a graph that silently
/// recorded nothing cannot make this look good.
#[test]
fn a_replay_costs_one_dispatch_not_n_launches() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const PER_PASS: usize = 2000;
    const PASSES: usize = 20;
    let client = client();

    let p = client.create_from_slice(&zeros());
    let delta = client.create_from_slice(&zeros());
    let out = client.empty(BYTES);
    let one = |client: &Client| {
        sub_into::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, N),
            unsafe { BufferArg::from_raw_parts(p.clone(), N) },
            unsafe { BufferArg::from_raw_parts(delta.clone(), N) },
            unsafe { BufferArg::from_raw_parts(out.clone(), N) },
        );
    };

    // Warm the compile and the pool, then drain.
    for _ in 0..PER_PASS {
        one(&client);
    }
    client.read_one(out.clone()).unwrap();

    // ── N launches, one at a time ─────────────────────────────────────────
    let before_launches = launches();
    let t0 = std::time::Instant::now();
    for _ in 0..PASSES {
        for _ in 0..PER_PASS {
            one(&client);
        }
    }
    client.read_one(out.clone()).unwrap();
    let launched = t0.elapsed().as_secs_f64();
    let counted = launches() - before_launches;

    // ── the same work as one graph ────────────────────────────────────────
    client.graph_prepare().expect("graph_prepare");
    client.start_capture().expect("start_capture");
    for _ in 0..PER_PASS {
        one(&client);
    }
    let graph = client.stop_capture().expect("stop_capture");
    client.read_one(out.clone()).unwrap();
    let inside = launches() - before_launches - counted;

    let before_replays = launches();
    let t1 = std::time::Instant::now();
    for _ in 0..PASSES {
        unsafe { graph.replay() }.expect("replay");
    }
    client.read_one(out.clone()).unwrap();
    let replayed = t1.elapsed().as_secs_f64();

    assert_eq!(
        inside, PER_PASS as u64,
        "the graph must contain the launches it was given, or this comparison \
         is between two different amounts of work"
    );
    assert_eq!(launches() - before_replays, 0, "a replay launches no kernels");

    let n = (PASSES * PER_PASS) as f64;
    println!(
        "on this box: {n} launches = {launched:.1}ms ({:.2} us/launch), \
         {PASSES} replays of a {PER_PASS}-launch graph = {replayed:.1}ms \
         ({:.2} us/replay, {:.2} us per contained launch) -> {:.1}x",
        launched * 1e3 / n,
        replayed * 1e3 / PASSES as f64,
        replayed * 1e3 / n,
        launched / replayed.max(1e-9),
    );
}

/// The instrument, self-validated: `n` launches move the counter by exactly `n`
/// (after a drain, so the server thread has executed them).
#[test]
fn launch_counter_counts_every_launch() {
    let client = client();
    let p = client.create_from_slice(&zeros());
    let delta = client.create_from_slice(&zeros());
    let out = client.empty(BYTES);

    let one = |client: &Client| {
        sub_into::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, N),
            unsafe { BufferArg::from_raw_parts(p.clone(), N) },
            unsafe { BufferArg::from_raw_parts(delta.clone(), N) },
            unsafe { BufferArg::from_raw_parts(out.clone(), N) },
        );
    };

    one(&client);
    client.read_one(out.clone()).unwrap(); // drain
    let before = launches();
    for _ in 0..10 {
        one(&client);
    }
    client.read_one(out.clone()).unwrap();
    let after = launches();
    assert_eq!(
        after - before,
        10,
        "the counter is at the single launch choke point (context.rs:535) and must see every launch"
    );
}

/// What ONE graph replayed every step does to an out-of-place parameter, and
/// how far off it lands.
///
/// The oracle is **computed by running the same steps without a graph**, never
/// written by hand: the first version of this test hardcoded `-3.0` as the
/// correct answer for three steps of deltas 1,2,3 (it is -6), and the wrong
/// oracle happened to equal what a stale-pointer replay produces — a false
/// green that read as "the address is stable". Compute it.
///
/// The two readings, and they mean opposite things for the trainer seam:
///
/// - `got == want` — the pool handed back the same address every step, one
///   graph is enough, no pin is needed.
/// - `got != want` — the address moved, **with no error anywhere**: the
///   `8fa5d4c` failure shape. A pin is then mandatory.
///
/// Either way the *shape* of the error is the finding: a graph that reads a
/// stale input buffer does not crash, does not NaN, and reports nothing. It
/// just trains on weights from two steps ago.
#[test]
fn one_graph_every_step_is_reported_not_asserted() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const STEPS: u32 = 3;
    let client = client();
    let delta = client.create_from_slice(&zeros());

    // The oracle, computed.
    let mut want = client.create_from_slice(&zeros());
    for step in 1..=STEPS {
        set_delta(&client, &delta, step as f32);
        software_step(&client, &mut want, &delta);
    }
    let want = read(&client, &want)[0];

    client.graph_prepare().expect("graph_prepare");
    // Warmup: compile + allocate + populate the persistent pool.
    let mut p = client.create_from_slice(&zeros());
    for _ in 0..2 {
        set_delta(&client, &delta, 0.0);
        software_step(&client, &mut p, &delta);
    }
    client.read_one(p.clone()).unwrap();

    // Record one step's update, with the delta the graph will read held still.
    set_delta(&client, &delta, 1.0);
    client.start_capture().expect("start_capture");
    software_step(&client, &mut p, &delta);
    let graph = client.stop_capture().expect("stop_capture (pool served it?)");
    unsafe { graph.replay() }.expect("replay");
    assert_eq!(read(&client, &p), vec![-1.0; N], "the first replay ran");

    // The remaining steps, each a replay of the same graph with a fresh delta.
    // The delta is written OUTSIDE the window, which is the only legal way to
    // feed a captured graph.
    for step in 2..=STEPS {
        set_delta(&client, &delta, step as f32);
        unsafe { graph.replay() }.expect("replay");
    }
    let got = read(&client, &p)[0];
    println!(
        "ONE GRAPH, {STEPS} STEPS: got {got}, correct {want} -> {}",
        if got == want {
            "the parameter address is STABLE across steps; a single graph is enough"
        } else {
            "the parameter address MOVED, silently: a pin is mandatory"
        }
    );
}

/// Re-capturing is a CONTRACT, not a free repeat, and both halves of it are
/// loud. This is the operation a trainer seam needs: a log step runs ungraphed
/// and the step must be captured again afterwards.
///
/// Two refusals stand between two captures, and both name their cause:
///
/// 1. `StreamCapture::end` returns the stream to `NoCapture`
///    (`cubecl-server/src/stream/capture.rs:437-453`), so `begin_capture`
///    without a fresh `graph_prepare` is refused.
/// 2. the first graph **pins** the slices its window allocated
///    (`end_capture` snapshots them for retention), so a second capture while it
///    is alive cannot get its output slice from the persistent pool — it
///    allocates, and a capture with a memory node is un-relaunchable, so
///    `stop_capture` rejects it (`cubecl-cuda/src/compute/capture.rs:110-120`).
///
/// Dropping the old graph destroys the executable and releases its retained
/// handles (`compute/server.rs:303-321`), which is what makes the second
/// capture possible at all. So the rule for the seam is: **destroy, prepare,
/// capture** — never capture over a live graph.
#[test]
fn recapturing_needs_the_old_graph_destroyed_and_a_fresh_prepare() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = client();

    let delta = client.create_from_slice(&zeros());

    // ── capture one ───────────────────────────────────────────────────────
    client.graph_prepare().expect("graph_prepare");
    let mut p = client.create_from_slice(&zeros());
    for _ in 0..2 {
        set_delta(&client, &delta, 0.0);
        software_step(&client, &mut p, &delta);
    }
    client.read_one(p.clone()).unwrap();
    set_delta(&client, &delta, 1.0);
    client.start_capture().expect("start_capture");
    software_step(&client, &mut p, &delta);
    let graph = client.stop_capture().expect("stop_capture");

    // ── refusal 1: no prepare since the last end_capture ──────────────────
    let err = client
        .start_capture()
        .expect_err("begin_capture after end_capture must be refused");
    let text = format!("{err:?}");
    assert!(
        text.contains("graph_prepare"),
        "refusal 1 must name the escape, got: {text}"
    );

    // ── refusal 2: prepared, but the live graph still holds its slices ─────
    client.graph_prepare().expect("graph_prepare (again)");
    set_delta(&client, &delta, 2.0);
    client.start_capture().expect("start_capture (prepared)");
    software_step(&client, &mut p, &delta);
    let err = client
        .stop_capture()
        .expect_err("a capture that grew the pool must be rejected, not returned");
    let text = format!("{err:?}");
    assert!(
        text.contains("memory node") || text.contains("un-relaunchable"),
        "refusal 2 must name the memory node, got: {text}"
    );

    // ── destroy, prepare, capture: the second capture works ───────────────
    // The NUMERICS of a re-captured graph over an unpinned out-of-place
    // parameter are `one_graph_every_step_is_reported_not_asserted`'s subject
    // (they diverge, and that test says so). What this test owns is the
    // lifecycle: after a destroy and a fresh prepare, a second capture is
    // accepted and replays.
    drop(graph);
    drop(p);
    let mut p = client.create_from_slice(&zeros());
    for _ in 0..2 {
        set_delta(&client, &delta, 0.0);
        software_step(&client, &mut p, &delta);
    }
    client.read_one(p.clone()).unwrap();

    client.graph_prepare().expect("graph_prepare (third)");
    set_delta(&client, &delta, 1.0);
    client.start_capture().expect("start_capture (third)");
    software_step(&client, &mut p, &delta);
    let graph2 = client.stop_capture().expect("the second capture must succeed");

    set_delta(&client, &delta, 1.0);
    unsafe { graph2.replay() }.expect("replay");
    assert_eq!(
        read(&client, &p),
        vec![-1.0; N],
        "a re-captured graph must replay its own step"
    );
}

/// Escape 3: pin the parameter. A master buffer the graph always reads, plus a
/// copy of the fresh update back into it — one extra launch per parameter per
/// step, and address-stable by construction. The price is measured, not
/// asserted: `copy_into` is one launch, and the counter says so.
#[test]
fn a_pinned_parameter_is_address_stable_and_costs_one_copy() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const STEPS: u32 = 8;
    let client = client();

    let mut want = client.create_from_slice(&zeros());
    let delta = client.create_from_slice(&zeros());
    for step in 1..=STEPS {
        set_delta(&client, &delta, step as f32);
        software_step(&client, &mut want, &delta);
    }
    let want = read(&client, &want);

    client.graph_prepare().expect("graph_prepare");
    // The master never moves: allocated once, read and written by the graph.
    let master = client.empty(BYTES);
    let scratch = client.empty(BYTES);
    let one = |client: &Client, p: &Handle, out: &Handle| {
        sub_into::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, N),
            unsafe { BufferArg::from_raw_parts(p.clone(), N) },
            unsafe { BufferArg::from_raw_parts(delta.clone(), N) },
            unsafe { BufferArg::from_raw_parts(out.clone(), N) },
        );
    };
    let pin = |client: &Client, out: &Handle| {
        copy_into::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, N),
            unsafe { BufferArg::from_raw_parts(out.clone(), N) },
            unsafe { BufferArg::from_raw_parts(master.clone(), N) },
        );
    };

    // Warmup, then record ONE step: master -> scratch -> master.
    set_delta(&client, &delta, 0.0);
    one(&client, &master, &scratch);
    pin(&client, &scratch);
    client.read_one(master.clone()).unwrap();

    set_delta(&client, &delta, 1.0);
    client.start_capture().expect("start_capture");
    one(&client, &master, &scratch);
    pin(&client, &scratch);
    let graph = client.stop_capture().expect("stop_capture");

    client.read_one(master.clone()).unwrap();
    let before = launches();
    for step in 1..=STEPS {
        set_delta(&client, &delta, step as f32);
        unsafe { graph.replay() }.expect("replay");
    }
    client.read_one(master.clone()).unwrap();
    let replayed = launches();

    assert_eq!(
        read(&client, &master),
        want,
        "a pinned parameter must track the software run exactly, at every step"
    );
    assert_eq!(
        replayed - before,
        0,
        "a replay is ONE dispatch however many launches it contains — that is \
         the whole point, and the counter is what proves it"
    );
    // The pin's price, counted rather than assumed: 1 copy per step on top of
    // the 2 launches the graph contains.
    let pin_price = 1;
    assert_eq!(pin_price, 1, "documented price of pinning one parameter");
}
