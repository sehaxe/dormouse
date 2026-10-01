//! Inference benchmark: dense fp32 vs packed ternary matmul on the CPU.
//! Run: cargo run -p burn-tsct --release --example infer_bench
fn main() {
    dormouse_spectral::infer::bench();
}
