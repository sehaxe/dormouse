// Shared fixture loader + module builder for the f64 reference layer.
// Included by `tests/ref_f64.rs`, by the breadth/chunk tests, and by the
// `ref_f64_stages` example; NOT a test target of its own (a file in `tests/`
// that is not a declared target would be).
//
// `ref_f64*.bin` layout (written by `tools/gen_reference_f64.py`):
//   "GDN2F64\0" | u32 d,h,hk,hv | f64 expand_v | u8 use_sc, allow_neg, n_tensors
//   | { u32 namelen, name, u32 ndim, u32 numel, i32[ndim] shape, f32[numel] }*
//   | u32 n_cases | { u32 t, f32[d*t] x, f64[d*t] y }*
// Linear weights are `[d_input, d_output]`, burn's on-disk layout.
//
// TWO FIXTURES, ONE FORMAT, ONE LOADER. `ref_f64.bin` is the 18-case
// hand-picked length matrix, and its wrong-formula companion
// `ref_f64_faults.bin` is committed against one specific case of it;
// `ref_f64_broad.bin` is the 1000-case breadth sweep. The sweep is a separate
// FILE rather than a longer case list precisely so the matrix's pinned case
// index keeps its meaning; see `BROAD_CASES` in the generator.
//
// Included, not compiled on its own, so it can carry `use` statements without
// colliding with the includer's. No inner attributes here: `include!` does not
// allow them.

use std::io::{Cursor, Read};

// `NdArray` is needed by the test's `forward::<NdArray>` but not by the
// example, which only builds the module; the includer cannot re-import it.
#[allow(unused_imports)]
use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData};
use burn_gdn2::{GatedDeltaNet2, Gdn2Config, Gdn2Mode};

pub struct Fixture {
    pub d: usize,
    pub h: usize,
    pub hk: usize,
    pub hv: usize,
    pub expand_v: f32,
    pub use_short_conv: bool,
    pub allow_neg_eigval: bool,
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub cases: Vec<(usize, Vec<f32>, Vec<f64>)>,
}

fn rd_u32(c: &mut Cursor<&[u8]>) -> u32 {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn rd_u8(c: &mut Cursor<&[u8]>) -> u8 {
    let mut b = [0u8; 1];
    c.read_exact(&mut b).unwrap();
    b[0]
}
fn rd_name(c: &mut Cursor<&[u8]>) -> String {
    let n = rd_u32(c) as usize;
    let mut b = vec![0u8; n];
    c.read_exact(&mut b).unwrap();
    String::from_utf8(b).unwrap()
}
fn rd_f32_vec(c: &mut Cursor<&[u8]>, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    c.read_exact(bytes).unwrap();
    v
}
fn rd_f64_vec(c: &mut Cursor<&[u8]>, n: usize) -> Vec<f64> {
    let mut v = vec![0f64; n];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 8) };
    c.read_exact(bytes).unwrap();
    v
}
fn rd_f32_tensor(c: &mut Cursor<&[u8]>) -> (String, Vec<usize>, Vec<f32>) {
    let name = rd_name(c);
    let ndim = rd_u32(c) as usize;
    let n = rd_u32(c) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(rd_u32(c) as usize);
    }
    (name, shape, rd_f32_vec(c, n))
}

/// Read a `GDN2F64` fixture. `what` only names the file in a panic message.
pub fn load_bytes(what: &str, data: &[u8]) -> Fixture {
    let mut c = Cursor::new(data);
    let mut magic = [0u8; 8];
    c.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"GDN2F64\0", "{what} is not this format");
    let d = rd_u32(&mut c) as usize;
    let h = rd_u32(&mut c) as usize;
    let hk = rd_u32(&mut c) as usize;
    let hv = rd_u32(&mut c) as usize;
    let mut eb = [0u8; 8];
    c.read_exact(&mut eb).unwrap();
    let expand_v = f64::from_le_bytes(eb) as f32;
    let use_short_conv = rd_u8(&mut c) != 0;
    let allow_neg_eigval = rd_u8(&mut c) != 0;
    let n_tensors = rd_u8(&mut c) as usize;
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        tensors.push(rd_f32_tensor(&mut c));
    }
    let n_cases = rd_u32(&mut c) as usize;
    let mut cases = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let t = rd_u32(&mut c) as usize;
        let x = rd_f32_vec(&mut c, t * d);
        let y = rd_f64_vec(&mut c, t * d);
        cases.push((t, x, y));
    }
    assert_eq!(
        c.position() as usize,
        data.len(),
        "trailing bytes in {what}"
    );
    Fixture {
        d,
        h,
        hk,
        hv,
        expand_v,
        use_short_conv,
        allow_neg_eigval,
        tensors,
        cases,
    }
}

/// The 18-case hand-picked length matrix.
pub fn load() -> Fixture {
    load_bytes("ref_f64.bin", include_bytes!("../ref_f64.bin"))
}

/// The 1000-case breadth sweep, same format, same loader.
pub fn load_broad() -> Fixture {
    load_bytes("ref_f64_broad.bin", include_bytes!("../ref_f64_broad.bin"))
}

pub fn get(t: &Fixture, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (_, s, d) = t
        .tensors
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("no tensor `{name}` in the fixture"));
    (s.clone(), d.clone())
}

fn param(t: &Fixture, name: &str, device: &Device) -> Param<Tensor<1>> {
    let (shape, data) = get(t, name);
    Param::from_tensor(Tensor::from_data(TensorData::new(data, shape), device))
}

fn lin_w(t: &Fixture, name: &str, device: &Device) -> Linear {
    let (shape, data) = get(t, name);
    let [d_input, d_output] = shape.as_slice() else {
        panic!("{name} is not 2-D")
    };
    let (d_input, d_output) = (*d_input, *d_output);
    let w = Tensor::from_data(TensorData::new(data, shape), device);
    let mut lin = LinearConfig::new(d_input, d_output)
        .with_bias(false)
        .init(device);
    lin.weight = Param::from_tensor(w);
    lin
}

fn lin_wb(t: &Fixture, name: &str, bias: &str, device: &Device) -> Linear {
    let (shape, data) = get(t, name);
    let [d_input, d_output] = shape.as_slice() else {
        panic!("{name} is not 2-D")
    };
    let (d_input, d_output) = (*d_input, *d_output);
    let (bshape, bdata) = get(t, bias);
    let w = Tensor::from_data(TensorData::new(data, shape), device);
    let b = Tensor::from_data(TensorData::new(bdata, bshape), device);
    let mut lin = LinearConfig::new(d_input, d_output)
        .with_bias(true)
        .init(device);
    lin.weight = Param::from_tensor(w);
    lin.bias = Some(Param::from_tensor(b));
    lin
}

/// `chunk_size` is a parameter, not a constant, because the chunk-sweep test
/// has to build the SAME weights at five chunk sizes. `self.config.chunk_size`
/// is read at forward time (`src/module.rs:341`, `:451`), but passing it in
/// makes the test say what it means rather than rely on that.
pub fn build(t: &Fixture, mode: Gdn2Mode, chunk_size: usize, device: &Device) -> GatedDeltaNet2 {
    GatedDeltaNet2 {
        q_proj: lin_w(t, "q_proj", device),
        k_proj: lin_w(t, "k_proj", device),
        v_proj: lin_w(t, "v_proj", device),
        f_proj_0: lin_w(t, "f_proj_0", device),
        f_proj_1: lin_w(t, "f_proj_1", device),
        b_proj: lin_w(t, "b_proj", device),
        w_proj: lin_w(t, "w_proj", device),
        g_proj_0: lin_w(t, "g_proj_0", device),
        g_proj_1: lin_wb(t, "g_proj_1", "g_proj_1_b", device),
        a_log: param(t, "A_log", device),
        dt_bias: param(t, "dt_bias", device),
        o_norm_weight: param(t, "o_norm_w", device),
        o_proj: lin_w(t, "o_proj", device),
        q_conv_w: Param::from_tensor({
            let (s, d) = get(t, "q_conv_w");
            Tensor::from_data(TensorData::new(d, s), device)
        }),
        k_conv_w: Param::from_tensor({
            let (s, d) = get(t, "k_conv_w");
            Tensor::from_data(TensorData::new(d, s), device)
        }),
        v_conv_w: Param::from_tensor({
            let (s, d) = get(t, "v_conv_w");
            Tensor::from_data(TensorData::new(d, s), device)
        }),
        config: Gdn2Config {
            hidden_size: t.d,
            num_heads: t.h,
            head_dim: t.hk,
            num_v_heads: Some(t.hv),
            expand_v: t.expand_v,
            use_short_conv: t.use_short_conv,
            allow_neg_eigval: t.allow_neg_eigval,
            norm_eps: 1e-5,
            mode,
            chunk_size,
            min_decay: None,
        },
        decay_factors: None,
    }
}

pub fn to_f32_vec(x: &Tensor<3>) -> Vec<f32> {
    x.clone()
        .into_data()
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, b| a.max(b.abs()))
}
