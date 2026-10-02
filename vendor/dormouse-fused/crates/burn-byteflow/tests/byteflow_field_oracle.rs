//! # The Canon gate — eq. (10) against torch cpu.
//!
//! Tier **(b)**, same oracle class as `byteflow_rate_oracle.rs` (no author
//! code; pdf p.19 + the lane's empty GitHub search). What this file adds
//! over the in-crate unit test (`canon_layer_matches_reference_formula`,
//! which re-derives eq. (10) from ITS OWN reading) is the SECOND
//! implementation route: `tests/oracle/gen_byteflow_field_oracle.py`
//! computes the SAME layer with torch cpu in f64 — a different arithmetic
//! stack, same formula text — and this test compares our f32 forward
//! through burn against it at a bar the f32 round-trip of the fixture
//! values leaves (2e-7 absolute at |h|≲2).
//!
//! The fixture carries NO RNG: every gate and every input cell is a
//! deterministic formula, so a torch-vs-burn disagreement names the
//! transcription (pad rule, tap order, gate layout) and not a seed.
//!
//! Generator `tests/oracle/gen_byteflow_field_oracle.py`, values
//! `tests/fixtures/byteflow_field_oracle.txt`, **this test needs no network**.

use burn::module::Param;
use burn::tensor::{Device, Tensor};
use burn_byteflow::CanonLayer;

const FIXTURE: &str = include_str!("fixtures/byteflow_field_oracle.txt");
const BAR: f32 = 2e-7;

fn dev() -> Device {
    Device::ndarray()
}

fn scalar(key: &str) -> f64 {
    FIXTURE
        .lines()
        .map(str::trim)
        .find_map(|l| {
            let mut it = l.split_whitespace();
            let k = it.next()?;
            (k == key).then(|| it.next().unwrap().parse().unwrap())
        })
        .unwrap_or_else(|| panic!("fixture scalar `{key}` missing"))
}

/// The FLOAT rows immediately following the named section header line.
fn matrix(section: &str) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::new();
    let mut started = false;
    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            if started {
                break;
            }
            continue;
        }
        if !started {
            started |= line == section;
            continue;
        }
        let row: Option<Vec<f32>> = line
            .split_whitespace()
            .map(|v| v.parse::<f32>())
            .collect::<Result<_, _>>()
            .ok();
        match row {
            Some(r) => out.push(r),
            None => break,
        }
    }
    assert!(!out.is_empty(), "fixture section `{section}` empty");
    out
}

#[test]
fn canon_layer_matches_torch_field_oracle() {
    let d = scalar("d") as usize;
    let t = scalar("t") as usize;
    assert_eq!(d, 8, "pinned by the generator header");
    assert_eq!(t, 12);

    let gates = matrix("gates_f32");
    let input = matrix("input_f32");
    let want = matrix("output_f32");
    assert_eq!(gates.len(), 4, "four taps, eq. (10)");
    assert!(gates.iter().all(|r| r.len() == d));

    let device = dev();
    let mut layer = CanonLayer::new(d, &device);
    let flat: Vec<f32> = gates.iter().flat_map(|r| r.iter().copied()).collect();
    layer.gates =
        Param::from_tensor(Tensor::<1>::from_floats(flat.as_slice(), &device).reshape([4, d]));

    let flat: Vec<f32> = input.iter().flat_map(|r| r.iter().copied()).collect();
    let h = Tensor::<1>::from_floats(flat.as_slice(), &device).reshape([1, t, d]);

    let got: Vec<f32> = layer
        .forward(h)
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    assert_eq!(got.len(), t * d);
    assert_eq!(want.len(), t);
    for (i, (g, w)) in got.iter().zip(want.iter().flatten()).enumerate() {
        assert!(
            (g - w).abs() < BAR,
            "canon[{}]: {g:.10} vs torch {w:.10}",
            i
        );
    }
}
