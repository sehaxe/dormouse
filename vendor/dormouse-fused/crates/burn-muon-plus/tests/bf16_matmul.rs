//! What bf16 actually does on THIS backend — measured, not assumed.
//!
//! dormouse's `--bf16` design rests on one question, and this file is the
//! answer to it. Previously it was four `#[cfg(all(test, feature = "cuda"))]`
//! probes that asserted the matmul output was finite — which, on a backend
//! where the cast is refused before the kernel is chosen, is not a property
//! anything can have. They never ran off CUDA; on the CPU backend the whole
//! module was compiled out and the binary reported `ok` over zero tests.
//! Worse, if they HAD run, all four would have died on the same unwrap, so
//! four green-looking tests were standing in for one refusal.
//!
//! Measured on this box, sm_120, burn 0.22.0-pre.4 + cubecl 0.11.0-pre.4,
//! 2026-09-29:
//!
//! | arm | outcome |
//! |---|---|
//! | f32 matmul + backward | grads arrive, full [768,768] |
//! | `f32 -> bf16` cast | **succeeds** |
//! | reading that bf16 buffer back as f32 | `Err(DTypeMismatch { expected: F32, actual: BF16 })` |
//! | bf16 x bf16 matmul | **panics** |
//! | bf16 x f32 matmul | **panics, identically** |
//! | bf16 leaves, forward + backward | **panics, identically** |
//! | bf16 activation x f32 parameter | **panics, identically** |
//!
//! The panic is one line and it is upstream's own:
//!
//! ```text
//! thread ... panicked at burn-cubecl-0.22.0-pre.4/src/ops/tensor.rs:150:66:
//! called `Result::unwrap()` on an `Err` value: Unable to launch matmul
//! because a required feature is unavailable: Types lhs=Float(BF16),
//! rhs=Float(BF16) and/or output=Float(BF16) not supported.
//! ```
//!
//! `restrict_to_llvm_backend` deletes bf16 from the advertised element types
//! (pliron's dialect has no bf16 type to lower through), so the matmul cannot
//! select a strategy and `.unwrap()` on that `Err` is what a caller sees. It
//! is LOUD, which is the only reason this file can be a test: the design
//! consequence is that `--bf16` must store bf16 and COMPUTE in fp32 through
//! cast copies, because there is no bf16 tensor core to compute with.
//!
//! So the assertions below pin the refusal, and the f32 control pins that the
//! harness itself is sound. If upstream ever gives the dialect a bf16 type,
//! these go RED on purpose — that is the signal to revisit the design, and a
//! silently-passing suite here would be the ADR-0011 failure in a test's
//! clothes.
//!
//! Run: cargo test -p burn-muon-plus --features cuda --test bf16_matmul -- --nocapture --test-threads=1
#![cfg(feature = "cuda")]
#![allow(deprecated)]

use burn::prelude::*;
use burn::tensor::{Distribution, FloatDType, Tensor};

/// The stable part of the refusal: `matmul` could not choose a strategy for
/// these element types. Matched on three pieces so a reworded message that
/// keeps the mechanism still passes, while a matmul that LAUNCHES fails.
const REFUSAL: [&str; 3] = [
    "Unable to launch matmul",
    "required feature is unavailable",
    "Float(BF16)",
];

/// Run `f` and return the panic message, or `None` if it did not panic.
fn panic_message<F: FnOnce()>(f: F) -> Option<String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // keep the expected panic out of the log
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    caught.err().map(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".into())
    })
}

fn assert_refused(what: &str, msg: &str) {
    for needle in REFUSAL {
        assert!(
            msg.contains(needle),
            "{what}: expected the documented matmul refusal (missing {needle:?}), got: {msg}"
        );
    }
    println!("{what}: refused as documented — {msg}");
}

/// Mechanism control, and it must run FIRST: every other assertion here is
/// about a dtype, so without this a broken harness and a working one look
/// identical. Plain fp32, no bf16 anywhere.
#[test]
fn f32_control_backward_arrives() {
    let d = Device::default().autodiff();
    let a: Tensor<2> = Tensor::random([768, 768], Distribution::Normal(0.0, 1.0), &d);
    let b = Tensor::<2>::random([768, 768], Distribution::Normal(0.0, 1.0), &d).require_grad();
    let grads = a.matmul(b.clone()).sum().backward();
    let gb: Vec<f32> = b
        .grad(&grads)
        .expect("fp32 control: grad must arrive")
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 readback");
    assert_eq!(gb.len(), 768 * 768, "fp32 control: grad must be full shape");
    assert!(
        gb.iter().all(|x| x.is_finite()),
        "fp32 control: grads must be finite"
    );
}

/// The landmine itself, as a tripwire: a bf16 x bf16 matmul on this backend
/// must be REFUSED, loudly, naming bf16 as the unsupported type.
#[test]
fn bf16_matmul_is_refused() {
    let d = Device::default();
    let msg = panic_message(|| {
        let a = Tensor::<2>::random([256, 256], Distribution::Normal(0.0, 1.0), &d)
            .cast(FloatDType::BF16);
        let b = Tensor::<2>::random([256, 256], Distribution::Normal(0.0, 1.0), &d)
            .cast(FloatDType::BF16);
        let _ = a.matmul(b);
    })
    .expect("bf16 x bf16 matmul must be refused on this backend, not computed");
    assert_refused("bf16 x bf16 matmul", &msg);
}

/// The other three shapes the old file probed are NOT separate cases — they
/// fail on the same unwrap, and the "mixed" one fails as `rhs=Float(BF16)`
/// because burn casts the right operand to the left one's dtype. Measured,
/// then pinned, so a future change that made any of them reachable would be
/// visible instead of silent.
#[test]
fn every_bf16_matmul_shape_is_the_same_refusal() {
    let rnd = |dev: &Device| Tensor::<2>::random([256, 256], Distribution::Normal(0.0, 1.0), dev);

    // bf16 activation x f32 parameter, under autodiff (dormouse's own shape).
    let da = Device::default().autodiff();
    let mixed = panic_message(|| {
        let a = rnd(&da).cast(FloatDType::BF16);
        let b = rnd(&da).require_grad();
        let grads = a.matmul(b.clone()).sum().backward();
        let _ = b.grad(&grads);
    });
    let mixed = mixed.expect("bf16-act x f32-param must be refused, not silently produce no grad");
    assert_refused("bf16 activation x f32 parameter", &mixed);

    // bf16 leaves through a forward AND a backward.
    let leaf = panic_message(|| {
        let a = rnd(&da).cast(FloatDType::BF16).require_grad();
        let b = rnd(&da).cast(FloatDType::BF16).require_grad();
        let grads = a
            .clone()
            .matmul(b.clone())
            .cast(FloatDType::F32)
            .sum()
            .backward();
        let _ = (a.grad(&grads), b.grad(&grads));
    });
    assert_refused(
        "bf16 leaves, fwd+bwd",
        &leaf.expect("bf16 leaves must be refused, not silently give no grad"),
    );

    // The dtype mismatch that made the "mixed" case look distinct is not: the
    // refusal names BOTH operands as bf16.
    assert!(
        mixed.contains("rhs=Float(BF16)"),
        "the mixed case is not a distinct case: burn casts rhs to lhs's dtype, \
         so the refusal must name rhs as bf16 too"
    );
}

/// The one thing that does work, which is the primitive the bf16 design is
/// allowed to use: the CAST succeeds, and the buffer is a real bf16 buffer
/// that burn refuses to read back as f32. (bf16 *storage* as u16 bit patterns
/// is the workaround that works; see `burn-gdn2`'s `lowp_bf16_cuda` for the
/// measured kernel side of it.)
#[test]
fn bf16_cast_works_and_the_f32_readback_is_refused() {
    let d = Device::default();
    let a =
        Tensor::<2>::random([64, 64], Distribution::Normal(0.0, 1.0), &d).cast(FloatDType::BF16);
    let back = a.into_data().try_to_vec::<f32>();
    assert!(
        back.is_err(),
        "burn must REFUSE to read a bf16 buffer back as f32; got {back:?}"
    );
    println!(
        "bf16 cast ok; f32 readback refused: {:?}",
        back.err().unwrap()
    );
}
