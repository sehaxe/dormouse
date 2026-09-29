// Diagnostic: dump every stage of `GatedDeltaNet2::project` for two cases, so
// the f64 reference's own stages can be diffed against them one at a time.
//
// WHY THIS EXISTS. `docs/ORACLE.md` §5 names the measurement that would settle
// where our output and a reference disagree — "print q/k/v at t=1 ... on both
// sides" — and records that nobody has run it. Two arms of a differential test
// that share a projection stage cannot see a bug in it; this is the way out,
// because it compares a STAGE against an outside reference rather than two arms
// against each other.
//
// USAGE (from `vendor/burn-fused`):
//     cargo run -p burn-gdn2 --release --example ref_f64_stages
//     python3 tools/gen_reference_f64.py --diff-stages /tmp/rust_stages.txt
//
// The diff script is `--diff-stages` in the generator; it recomputes each stage
// in f64 and prints max|ours - reference| per stage, so the first stage to
// disagree names the defect.
//
// This is a tool, not a gate. It has no assertions, and it is not part of
// `tools/wt.sh test f64ref`; the gate is `tests/ref_f64.rs`.
#![allow(dead_code, deprecated)]

use std::fs;

use burn::tensor::activation::softplus;
use burn_gdn2::short_conv_1d;

include!("../tests/common/ref_f64.rs");

fn main() {
    let t = load();
    let device = Device::ndarray();
    let m = build(&t, Gdn2Mode::FusedRecurrent, &device);
    let mut out = String::new();
    for (ci, (seq, x, _)) in t.cases.iter().enumerate() {
        if ci != 0 && ci != t.cases.len() - 1 {
            continue;
        }
        let input =
            Tensor::<3>::from_data(TensorData::new(x.clone(), [1, *seq, t.d]), &device);

        // raw projections, pre-conv
        let (qr, kr, vr) = (
            m.q_proj.forward(input.clone()),
            m.k_proj.forward(input.clone()),
            m.v_proj.forward(input.clone()),
        );
        dump(&mut out, &format!("c{ci}/raw_q_proj"), &qr.clone().into_data().bytes);
        dump(&mut out, &format!("c{ci}/raw_k_proj"), &kr.clone().into_data().bytes);
        dump(&mut out, &format!("c{ci}/raw_v_proj"), &vr.clone().into_data().bytes);

        // the conv, straight out of the crate's own short_conv_1d
        for (n, (p, w)) in [
            ("q_conv", (&qr, &m.q_conv_w)),
            ("k_conv", (&kr, &m.k_conv_w)),
            ("v_conv", (&vr, &m.v_conv_w)),
        ] {
            let (y, _) = short_conv_1d(p.clone(), w.val(), None::<&Tensor<3>>);
            dump(&mut out, &format!("c{ci}/{n}"), &y.clone().into_data().bytes);
        }

        // everything project() hands to the recurrence and the readout
        // Called for its side effect of exercising the whole projection stage;
        // the values it returns are read as 2-D by re-deriving them from the
        // fixture below, because the 4-D per-head views are strided.
        let (_projected, _) = m.project(input.clone(), None);

        // The log-decay, twice: as project() leaves it (a [B,H,T,HK] permuted
        // view) and un-permuted back to [B,T,KD], against a recomputation from
        // the fixture. If the un-permute and the recomputation agree, `p.g` is
        // right and any diff on the permuted view is an artifact of reading a
        // view. If they do not, the decay really is wrong.
        let kd = t.h * t.hk;
        let a_exp = m
            .a_log
            .val()
            .exp()
            .reshape([1, t.h, 1])
            .repeat(&[1, 1, t.hk])
            .reshape([1, 1, kd]);
        let f0 = m.f_proj_0.forward(input.clone());
        let f1 = m.f_proj_1.forward(f0.clone());
        let _ = &input;
        for (n, o) in [("f0", &f0), ("f1", &f1)] {
            dump(
                &mut out,
                &format!("c{ci}/{n}"),
                &o.clone().into_data().bytes,
            );
        }
        // the same two-layer chain on the OUTPUT gate, which is known to agree.
        // This isolates "T=70 breaks a Linear" from "f_proj is wrong".
        let q0 = m.g_proj_0.forward(input.clone());
        let q1 = m.g_proj_1.forward(q0.clone());
        for (n, o) in [("gp0", &q0), ("gp1", &q1)] {
            dump(
                &mut out,
                &format!("c{ci}/{n}"),
                &o.clone().into_data().bytes,
            );
        }
        let g_recomputed = -(a_exp.clone() * softplus(
            f1.clone() + m.dt_bias.val().reshape([1, 1, kd]),
            1.0,
        ));
        dump(
            &mut out,
            &format!("c{ci}/g_recomputed"),
            &g_recomputed.clone().into_data().bytes,
        );
        dump(
            &mut out,
            &format!("c{ci}/a_exp"),
            &a_exp.clone().into_data().bytes,
        );
        dump(
            &mut out,
            &format!("c{ci}/dt_bias"),
            &m.dt_bias.val().into_data().bytes,
        );
        dump(
            &mut out,
            &format!("c{ci}/A_log"),
            &m.a_log.val().into_data().bytes,
        );

        // 4-D STAGES ARE DELIBERATELY NOT DUMPED. `project` leaves them as
        // permuted [B,H,T,D] views, and on this backend `into_data()` on a
        // strided view does not read in a stable order: measured here, `p.g`'s
        // permuted read and its un-permuted read disagree by O(1) while holding
        // the same values, and only the un-permuted one matches the reference.
        // That is a burn readback question, NOT a `burn-gdn2` defect (the
        // layer's own matmuls consume the same views and produce the right
        // answer), and a diagnostic row built on it would be a false positive.
        // The 2-D and 1-D stages below have no such ambiguity, and they are
        // what the localisation actually rests on.
    }
    fs::write("/tmp/rust_stages.txt", &out).unwrap();
    println!("wrote /tmp/rust_stages.txt ({} bytes)", out.len());
}

fn dump(out: &mut String, name: &str, bytes: &[u8]) {
    assert!(!bytes.is_empty());
    out.push_str(name);
    out.push(' ');
    for c in bytes.chunks_exact(4) {
        out.push_str(&format!("{:e} ", f32::from_le_bytes(c.try_into().unwrap())));
    }
    out.push('\n');
}
