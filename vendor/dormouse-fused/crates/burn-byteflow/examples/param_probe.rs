//! Prints ByteFlowNet::num_params for candidate `d_global` values on CPU.
//! Pure shape arithmetic at init — no training, no GPU. Lane bfconfirm:
//! the params-matched confirm needs the actual count before any GPU turn.
use burn::module::Module as _;
use burn_byteflow::{ByteFlowConfig, ByteFlowNet, RateMode};
use burn_ndarray::NdArrayDevice;

fn cfg(d_global: usize) -> ByteFlowConfig {
    ByteFlowConfig {
        d_local: 96,
        d_global,
        k_tokens: 128,
        e_layers: 2,
        g_layers: 2,
        n_heads_local: 4,
        n_heads_global: 8,
        w_local: 256,
        d_ff_local: 256,
        d_ff_global: 2048,
        bins: 16,
        eps2: 0.5,
        rate_mode: RateMode::L2,
        max_bytes: 512,
        ..Default::default()
    }
}

fn main() {
    let device: burn::tensor::Device = NdArrayDevice::Cpu.into();
    for d_global in [768usize, 640, 512, 504, 496, 480, 464] {
        let net = ByteFlowNet::init(cfg(d_global), &device);
        println!("d_global={d_global} params={}", net.num_params());
    }
    println!("control target: 9197454");
}
