//! bench — gate, bpb, ppl, all-bf16
pub fn bpb(ce: f32) -> f32 { ce / std::f32::consts::LN_2 }
pub fn ppl(bpb: f32) -> f32 { 2_f32.powf(bpb) }
pub fn gate(baseline: f32, current: f32, tol: f32) -> bool { (current - baseline).abs() <= tol }
