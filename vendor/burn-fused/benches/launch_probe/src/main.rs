use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::Runtime;
use std::time::Instant;

#[cube(launch_unchecked)]
fn tiny_kernel<F: Float>(out: &mut [F], v: u32) {
    let i = UNIT_POS_X as usize;
    out[i] = F::cast_from(v);
}

fn main() {
    type R = CudaRuntime;
    let client = R::client(&Default::default());
    let n = 16usize;
    let out = client.empty(n * 4);
    let dim = CubeDim::new_3d(16, 1, 1);
    // warmup
    for _ in 0..5 {
        unsafe {
            tiny_kernel::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                dim,
                BufferArg::from_raw_parts(out.clone(), n),
                1,
            );
        }
    }
    let _ = client.read_one(out.clone()).unwrap();
    // time 100 launches (async queue, one sync at end)
    let t0 = Instant::now();
    for i in 0..100 {
        unsafe {
            tiny_kernel::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                dim,
                BufferArg::from_raw_parts(out.clone(), n),
                i,
            );
        }
    }
    let _ = client.read_one(out).unwrap(); // sync
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "100 tiny launches + sync: {:.4} ms -> {:.4} ms/launch",
        dt * 1000.0,
        dt * 1000.0 / 100.0
    );
}
