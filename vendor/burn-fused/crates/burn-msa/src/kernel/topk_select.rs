use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube(launch_unchecked)]
fn exp_free_topk_kernel<F: Float>(
    scores: &[F],
    indices_out: &mut [u32],
    #[comptime] n_blocks: u32,
    #[comptime] topk: u32,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let step = CUBE_DIM_X as usize;
    let n_blocks = n_blocks as usize;
    let topk = topk as usize;

    let mut heap_scores = Shared::<[F]>::new_slice(topk);
    let mut heap_indices = Shared::<[u32]>::new_slice(topk);

    for i in 0..topk {
        heap_scores[i] = F::new(f32::NEG_INFINITY);
        heap_indices[i] = u32::MAX;
    }

    let row_base = row * n_blocks;
    let mut j = lane;
    while j < n_blocks {
        let score = scores[row_base + j];
        if score > heap_scores[0] {
            heap_scores[0] = score;
            heap_indices[0] = j as u32;
            let mut idx = 0;
            loop {
                let mut smallest = idx;
                let left = 2 * idx + 1;
                let right = 2 * idx + 2;
                if left < topk && heap_scores[left] < heap_scores[smallest] {
                    smallest = left;
                }
                if right < topk && heap_scores[right] < heap_scores[smallest] {
                    smallest = right;
                }
                if smallest == idx {
                    break;
                }
                let tmp_s = heap_scores[idx];
                let tmp_i = heap_indices[idx];
                heap_scores[idx] = heap_scores[smallest];
                heap_indices[idx] = heap_indices[smallest];
                heap_scores[smallest] = tmp_s;
                heap_indices[smallest] = tmp_i;
                idx = smallest;
            }
        }
        j += step;
    }

    for k in 0..topk {
        let mut best_score = heap_scores[0];
        let mut best_index = heap_indices[0];
        for offset in 0..5u32 {
            let mask = 1u32 << offset;
            let peer_score = plane_shuffle_down(best_score, mask);
            let peer_index = plane_shuffle_down(best_index, mask);
            if peer_score > best_score {
                best_score = peer_score;
                best_index = peer_index;
            }
        }
        best_index = plane_shuffle(best_index, 0u32);
        if lane == 0 {
            indices_out[row * topk + k] = best_index;
        }
        for i in 0..topk {
            if heap_indices[i] == best_index {
                heap_scores[i] = F::new(f32::NEG_INFINITY);
                heap_indices[i] = u32::MAX;
            }
        }
        let mut idx = 0;
        loop {
            let mut smallest = idx;
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            if left < topk && heap_scores[left] < heap_scores[smallest] {
                smallest = left;
            }
            if right < topk && heap_scores[right] < heap_scores[smallest] {
                smallest = right;
            }
            if smallest == idx {
                break;
            }
            let tmp_s = heap_scores[idx];
            let tmp_i = heap_indices[idx];
            heap_scores[idx] = heap_scores[smallest];
            heap_indices[idx] = heap_indices[smallest];
            heap_scores[smallest] = tmp_s;
            heap_indices[smallest] = tmp_i;
            idx = smallest;
        }
    }
}

/// Launch `exp_free_topk_kernel` on the bare CUDA backend: one warp per row,
/// the k largest scores of each row written to `indices_out` in descending
/// order.
///
/// # Safety
///
/// The handles must reference buffers large enough for `n_rows * n_blocks`
/// (scores) and `n_rows * topk` (indices) elements; lifetimes are not tracked.
pub unsafe fn launch_exp_free_topk(
    client: &ComputeClient<cubecl::cuda::CudaRuntime>,
    scores_handle: &Handle,
    indices_handle: &Handle,
    n_rows: u32,
    n_blocks: u32,
    topk: u32,
) {
    let cube_dim = CubeDim::new_3d(32, 1, 1);
    let cube_count = CubeCount::Static(n_rows, 1, 1);
    exp_free_topk_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
        client,
        cube_count,
        cube_dim,
        BufferArg::from_raw_parts(scores_handle.clone(), (n_rows * n_blocks) as usize),
        BufferArg::from_raw_parts(indices_handle.clone(), (n_rows * topk) as usize),
        n_blocks,
        topk,
    );
}
