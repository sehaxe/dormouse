//! cuBLAS proof of concept: can dormouse's big GEMM go through cuBLAS with
//! f16 inputs and an fp32 accumulator, and what does routing cost?
//!
//! Why: measured on this exact GPU (RTX 5060 Ti, 5120x2048x8192,
//! research/cublas_probe/cublas_combos.cu) cuBLAS f16/fp32-accumulate runs at
//! 43.7 TFLOP/s against 13.2 for cuBLAS fp32 and 3.5-7.6 for cubecl's own f32
//! matmul path. A hand-rolled WMMA kernel measured 8.4 and LOST to plain SGEMM,
//! so the answer is cuBLAS, not a new kernel.
//!
//! The blocker is that a burn tensor's buffer is a cubecl `Handle`, not a
//! device pointer, and cubecl-runtime 0.11.0-pre.4 has no client API that
//! resolves one (research/2026-09-27-cublas-integration-poc.md has the patch
//! and the measured zero-copy result). This probe measures what the UNBLOCKED
//! version can do, which is what decides whether the patch is worth its risk:
//!
//!   1. the raw cuBLAS call (f16 in, fp32 accumulate) on device buffers this
//!      process allocated, checked against a host f32 matmul and timed on
//!      production shapes - is the 43.7 TFLOP/s real through this path?
//!   2. the same GEMM as a burn operation with the operands staged through the
//!      host (`into_data` -> f16 -> H2D -> GemmEx -> D2H -> a burn tensor),
//!      checked against `a.matmul(b)` on the same inputs;
//!   3. the same GEMM with the operands already f16 on the device and staged
//!      with D2D copies instead of host round trips - the shape the pointer
//!      patch would buy, minus the pointer itself.
//!
//! Run: cargo run --release -p cublas-poc
//!
//! Section 4 (zero-copy on cubecl's own stream) needs the ~110-line cubecl
//! patch in research/2026-09-27-cublas-integration-poc.md, which is
//! deliberately NOT in the tree, so it is behind the off-by-default
//! `cublas-native` feature: `cargo run --release -p cublas-poc --features
//! cublas-native`. Sections 0-3 need no patch and run either way.

#[cfg(feature = "cublas-native")]
mod poc {
    use burn::tensor::{Device, Distribution, FloatDType, Tensor, TensorData};
    use cudarc::driver::sys::{
        CUdevice, CUdeviceptr, CUstream, cuCtxSetCurrent, cuDeviceGet, cuDevicePrimaryCtxRetain,
        cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoD_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
        cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize,
    };
    use half::f16;
    use std::ffi::{c_char, c_int, c_void};
    use std::ptr;

    // cuBLAS enum values, from cudarc 0.19's `cublas::sys` (the values are
    // checked there, not guessed: FAST_16BF is 75 and FAST_TF32 is 77, and
    // both wrong values come back as CUBLAS_STATUS_NOT_SUPPORTED).
    const OP_N: u32 = 0;
    const OP_T: u32 = 1;
    const R_32F: u32 = 0;
    const R_16F: u32 = 2;
    const R_16BF: u32 = 14;
    const COMPUTE_32F: u32 = 68;
    const COMPUTE_32F_FAST_16F: u32 = 74;
    const COMPUTE_32F_FAST_16BF: u32 = 75;
    const COMPUTE_32F_FAST_TF32: u32 = 77;
    /// `CUBLAS_GEMM_DEFAULT` is `CUBLAS_GEMM_DFALT`, i.e. -1.
    const GEMM_DEFAULT: u32 = -1i32 as u32;

    type GemmEx = unsafe extern "C" fn(
        handle: *mut c_void,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        atype: u32,
        lda: c_int,
        b: *const c_void,
        btype: u32,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ctype: u32,
        ldc: c_int,
        compute: u32,
        algo: u32,
    ) -> c_int;

    /// A cuBLAS handle bound to this thread's CUDA context and a private
    /// non-blocking stream.
    ///
    /// cudarc 0.19 does ship a full `cublas` module, but it sits behind a
    /// `cublas` cargo feature this tree does not enable, and enabling it
    /// recompiles every cubecl crate in the tree (cudarc's metadata hash feeds
    /// all of them). Four dlopen'd symbols cost nothing and keep the probe
    /// buildable against the tree as it stands.
    struct Blas {
        gemm_ex: GemmEx,
        /// Only the patched zero-copy variant needs this: re-pointing cuBLAS at
        /// cubecl's own stream is the whole reason for the pointer patch. Kept
        /// here (and in `on_stream`) so the patched build of this probe is a
        /// one-line change - research/2026-09-27-cublas-integration-poc.md.
        #[allow(dead_code)]
        set_stream: unsafe extern "C" fn(*mut c_void, CUstream) -> c_int,
        destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
        handle: *mut c_void,
        stream: CUstream,
    }

    impl Blas {
        fn load() -> Self {
            unsafe extern "C" {
                fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            const RTLD_NOW: c_int = 2;

            unsafe {
                // A `&str` in a `&[&str]` is NOT NUL-terminated in memory (the
                // literals sit back to back in .rodata), so the name has to be
                // copied into a CString - passing `s.as_ptr()` hands dlopen
                // whatever string literal follows it.
                let c_name = |name: &str| std::ffi::CString::new(name).expect("no NUL in name");
                let lib = [
                    "/opt/cuda/lib64/libcublas.so.13",
                    "libcublas.so.13",
                    "libcublas.so.12",
                    "libcublas.so",
                ]
                .iter()
                .find_map(|name| {
                    let handle = dlopen(c_name(name).as_ptr(), RTLD_NOW);
                    (!handle.is_null()).then_some(handle)
                })
                .unwrap_or_else(|| panic!("libcublas is not loadable"));
                let sym = |name: &str| dlsym(lib, c_name(name).as_ptr());
                // cuBLAS renamed every entry point to _v2; which of the two
                // names a given libcublas exports is not documented, so try
                // both before giving up.
                let pick = |short: &str, v2: &str| {
                    let found = sym(short);
                    let found = if found.is_null() { sym(v2) } else { found };
                    assert!(!found.is_null(), "libcublas has neither {short} nor {v2}");
                    found
                };
                let create: unsafe extern "C" fn(*mut *mut c_void) -> c_int =
                    std::mem::transmute(pick("cublasCreate", "cublasCreate_v2"));
                let mut handle = ptr::null_mut();
                let status = create(&mut handle);
                assert_eq!(status, 0, "cublasCreate -> {status}");

                Self {
                    gemm_ex: std::mem::transmute(pick("cublasGemmEx", "cublasGemmEx_v2")),
                    set_stream: std::mem::transmute(pick(
                        "cublasSetStream",
                        "cublasSetStream_v2",
                    )),
                    destroy: std::mem::transmute(pick("cublasDestroy", "cublasDestroy_v2")),
                    handle,
                    stream: ptr::null_mut(),
                }
            }
        }

        /// Create a private stream and point the handle at it. Needs a current
        /// context, so it runs after `enter_primary_context`.
        fn attach_stream(&mut self) {
            unsafe {
                cuStreamCreate(&mut self.stream, 1 /* CU_STREAM_NON_BLOCKING */)
                    .result()
                    .expect("private stream");
            }
            self.on_stream(self.stream);
        }

        /// Point cuBLAS at `stream` (a raw `CUstream`, or the one cubecl's
        /// server owns once the pointer patch lands - that is the whole point
        /// of the patch: enqueue in the graph's stream order, no barrier).
        #[allow(dead_code)]
        fn on_stream(&self, stream: CUstream) {
            let status = unsafe { (self.set_stream)(self.handle, stream) };
            assert_eq!(status, 0, "cublasSetStream");
        }

        /// `C[M,N] = A[M,K] @ B[K,N]` for ROW-major buffers, which is what a
        /// burn tensor's memory is.
        ///
        /// cuBLAS is column-major: it computes `C'[m,n] = op(A)[m,k] @ op(B)[k,n]`
        /// over column-major buffers, where `C'[r,c]` lives at `r + c*ldc`. A
        /// row-major `C[M,N]` is the column-major `C'[N,M]` (r = column j, c =
        /// row i, so `ldc = N`), and `C[M,N] = A@B` transposed is
        /// `C'[N,M] = B^T @ A^T`, so the gemm's first operand is B and its
        /// second is A. Hence `m = N, n = M, k = K, ldc = N`, and:
        ///
        ///   * `lda` is B's row length, **N** - not K. (B is K x N row-major =
        ///     N x K column-major with ld = N.) `lda = K` is accepted by the
        ///     driver and silently computes the wrong thing, which is what
        ///     `convention_probe` below is there to prove.
        ///   * `ldb` is A's row length, K, and `OP_T` on that operand would need
        ///     `ldb >= M` for a M x K column-major operand, so OP_T there is
        ///     refused with an illegal ldb whenever M > K.
        ///
        /// `transa`/`lda` are parameters only so the probe can measure which
        /// combinations the driver accepts; everything else uses `row_major`.
        #[allow(clippy::too_many_arguments)]
        fn gemm_ex_raw(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: CUdeviceptr,
            b: CUdeviceptr,
            c: CUdeviceptr,
            ab_type: u32,
            compute: u32,
            transa: u32,
            transb: u32,
            lda: usize,
            ldb: usize,
        ) -> Result<(), i32> {
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status = unsafe {
                (self.gemm_ex)(
                    self.handle,
                    transa,
                    transb,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha as *const f32 as *const c_void,
                    b as *const c_void,
                    ab_type,
                    lda as c_int,
                    a as *const c_void,
                    ab_type,
                    ldb as c_int,
                    &beta as *const f32 as *const c_void,
                    c as *mut c_void,
                    R_32F,
                    n as c_int,
                    compute,
                    GEMM_DEFAULT,
                )
            };
            (status == 0).then_some(()).ok_or(status)
        }

        /// The verified row-major call: `A[M,K] @ B[K,N] -> C[M,N]`, all
        /// row-major. See `gemm_ex_raw` for why `lda = N`.
        fn gemm_ex(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: CUdeviceptr,
            b: CUdeviceptr,
            c: CUdeviceptr,
            ab_type: u32,
            compute: u32,
        ) -> Result<(), i32> {
            self.gemm_ex_raw(m, k, n, a, b, c, ab_type, compute, OP_N, OP_N, n, k)
        }
    }

    impl Drop for Blas {
        fn drop(&mut self) {
            unsafe {
                cuStreamSynchronize(self.stream).result().ok();
                (self.destroy)(self.handle);
                cuStreamDestroy_v2(self.stream).result().ok();
            }
        }
    }

    /// The cubecl server runs on its own thread with the primary context
    /// current *there*. Every driver/cuBLAS call below is made from this
    /// thread, and a CUDA context is per-thread, so the primary context has to
    /// be pushed here too - a device pointer means nothing without it. This is
    /// the same context (the primary one is shared process-wide), which is why
    /// the buffers cubecl allocated are reachable from here.
    fn enter_primary_context(device_index: i32) {
        unsafe {
            cuInit(0).result().expect("cuInit");
            let mut dev = std::mem::MaybeUninit::<CUdevice>::uninit();
            cuDeviceGet(dev.as_mut_ptr(), device_index)
                .result()
                .expect("cuDeviceGet");
            let mut ctx = std::mem::MaybeUninit::uninit();
            cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev.assume_init())
                .result()
                .expect("primary ctx retain");
            cuCtxSetCurrent(ctx.assume_init())
                .result()
                .expect("cuCtxSetCurrent");
        }
    }

    struct Buf(CUdeviceptr);

    impl Buf {
        fn new(bytes: usize) -> Self {
            let mut ptr = 0 as CUdeviceptr;
            unsafe { cuMemAlloc_v2(&mut ptr, bytes) }
                .result()
                .unwrap_or_else(|e| panic!("cuMemAlloc({bytes}): {e:?}"));
            Self(ptr)
        }
        fn download(&self, out: &mut [u8]) {
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.0, out.len()) }
                .result()
                .expect("D2H");
        }
        fn download_f32(&self, len: usize) -> Vec<f32> {
            let mut raw = vec![0u8; len * 4];
            self.download(&mut raw);
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }

    impl Drop for Buf {
        fn drop(&mut self) {
            unsafe { cuMemFree_v2(self.0) }.result().ok();
        }
    }

    fn upload_f32(ptr: CUdeviceptr, host: &[f32]) {
        let raw: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        unsafe {
            cuMemcpyHtoD_v2(ptr, raw.as_ptr() as *const c_void, raw.len())
        }
        .result()
        .expect("H2D f32");
    }

    fn upload_f16(ptr: CUdeviceptr, host: &[f32]) {
        let raw: Vec<u8> = host
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        unsafe {
            cuMemcpyHtoD_v2(ptr, raw.as_ptr() as *const c_void, raw.len())
        }
        .result()
        .expect("H2D f16");
    }

    /// Deterministic host data, so two runs are comparable.
    fn lcg(n: usize, scale: f32) -> Vec<f32> {
        let mut s = 0x2545_f491u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0 - 1.0) * scale
            })
            .collect()
    }

    /// max |diff| and the same normalized by the reference's scale. Per-element
    /// relative error is useless on gaussian data (half the elements are near
    /// zero), so this is the honest AMP-fidelity number.
    fn err(got: &[f32], want: &[f32]) -> (f32, f32) {
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let max = got
            .iter()
            .zip(want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
        (max, max / scale)
    }

    fn compute_name(compute: u32) -> String {
        match compute {
            COMPUTE_32F => "32F".into(),
            COMPUTE_32F_FAST_16F => "32F/16".into(),
            COMPUTE_32F_FAST_16BF => "32F/bf".into(),
            COMPUTE_32F_FAST_TF32 => "32F/tf32".into(),
            other => format!("{other}"),
        }
    }

    fn ms_of(iters: usize, mut f: impl FnMut()) -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        cudarc::driver::result::ctx::synchronize().expect("ctx sync");
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    }

    fn sync() {
        cudarc::driver::result::ctx::synchronize().expect("ctx sync");
    }

    /// `C[M,N] = A[M,K] @ B[K,N]` on the host, row-major, plain f32. Only for
    /// the small shape: 5120x2048x8192 is 30 s on one core.
    fn host_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for (i, row) in c.chunks_mut(n).enumerate() {
            for (j, out) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                *out = acc;
            }
        }
        c
    }

    pub fn run() {
        // cuBLAS is opened BEFORE anything CUDA: once cudarc has initialized
        // the driver on cubecl's device thread, dlopen of the CUDA math
        // libraries fails in this process with ENOENT (measured; the plain
        // probe binary dlopens it fine). A cuBLAS handle is context-agnostic
        // until its first call, so opening it early costs nothing.
        let mut blas = Blas::load();
        let device = Device::cuda(0);
        // Start the cubecl device thread first: it is the thread that retains
        // the primary context, and this probe wants that same context.
        let _: Tensor<2> = Tensor::zeros([4, 4], &device);
        enter_primary_context(0);
        blas.attach_stream();

        // ---- 0. the row-major call convention, measured not assumed ----
        // cuBLAS is column-major and a burn tensor is row-major, so the call
        // has to transpose the problem into cuBLAS's view. There are four
        // (transa, transb) pairs and each has two plausible leading
        // dimensions; this walks all of them against a host f32 reference and
        // prints which are accepted and which are actually right, so the
        // convention in `gemm_ex` is the measured one.
        println!("== 0. row-major call convention (C[M,N] = A[M,K] @ B[K,N]) ==");
        {
            let (m, k, n) = (256usize, 128usize, 512usize);
            let a = lcg(m * k, 1.0);
            let b = lcg(k * n, 0.05);
            let want = host_gemm(&a, &b, m, k, n);
            let da = Buf::new(m * k * 4);
            let db = Buf::new(k * n * 4);
            let dc = Buf::new(m * n * 4);
            upload_f32(da.0, &a);
            upload_f32(db.0, &b);
            println!(
                "  M={m} K={k} N={n} (M>K, N>K, N>M), reference max|C| = {:.3}",
                want.iter().fold(0.0f32, |x, v| x.max(v.abs()))
            );
            println!(
                "  {:>4} {:>4} {:>5} {:>5} {:>26}",
                "trA", "trB", "lda", "ldb", "result"
            );
            for (transa, an) in [(OP_N, "OP_N"), (OP_T, "OP_T")] {
                for (transb, bn) in [(OP_N, "OP_N"), (OP_T, "OP_T")] {
                    for lda in [k, n] {
                        for ldb in [k, m] {
                            let r = blas.gemm_ex_raw(
                                m, k, n, da.0, db.0, dc.0, R_32F, COMPUTE_32F, transa,
                                transb, lda, ldb,
                            );
                            let msg = match r {
                                Err(status) => format!("rejected, status {status}"),
                                Ok(()) => {
                                    sync();
                                    let (maxabs, rel) = err(&dc.download_f32(m * n), &want);
                                    if rel < 1e-4 {
                                        format!("MATCH  maxabs {maxabs:.2e} rel {rel:.1e}")
                                    } else {
                                        format!("accepted but WRONG: maxabs {maxabs:.2e} rel {rel:.1e}")
                                    }
                                }
                            };
                            println!("  {an:>4} {bn:>4} {lda:>5} {ldb:>5} {msg:>26}");
                        }
                    }
                }
            }
        }

        // ---- 1. the raw call, on buffers this process allocated ----
        println!("== 1. raw cuBLAS (f16 in, fp32 accumulate), own buffers ==");
        println!(
            "{:<20} {:>5} {:>8} {:>9} {:>8} {:>9} {:>9}",
            "shape", "A/B", "compute", "ms", "TFLOP/s", "maxabs", "rel"
        );
        for (m, k, n) in [
            (256usize, 128usize, 128usize),
            (5120, 768, 2048),
            (5120, 2048, 8192),
        ] {
            let a = lcg(m * k, 1.0);
            let b = lcg(k * n, 0.05);
            // Host f32 reference, small shape only: 5120x2048x8192 on one core
            // is 30 s, which is not a probe.
            let want: Option<Vec<f32>> = (m * n <= 1 << 22)
                .then(|| host_gemm(&a, &b, m, k, n));

            for (ab_type, compute, ab_name) in [
                (R_32F, COMPUTE_32F, "f32"),
                (R_32F, COMPUTE_32F_FAST_TF32, "f32"),
                (R_32F, COMPUTE_32F_FAST_16F, "f32"),
                (R_16F, COMPUTE_32F, "f16"),
                (R_16F, COMPUTE_32F_FAST_16F, "f16"),
                (R_16BF, COMPUTE_32F, "bf16"),
                (R_16BF, COMPUTE_32F_FAST_16BF, "bf16"),
            ] {
                let narrow = ab_type != R_32F;
                let da = Buf::new(m * k * if narrow { 2 } else { 4 });
                let db = Buf::new(k * n * if narrow { 2 } else { 4 });
                let dc = Buf::new(m * n * 4);
                if narrow {
                    upload_f16(da.0, &a);
                    upload_f16(db.0, &b);
                } else {
                    upload_f32(da.0, &a);
                    upload_f32(db.0, &b);
                }
                let (maxabs, rel) = match blas.gemm_ex(m, k, n, da.0, db.0, dc.0, ab_type, compute) {
                    Err(status) => {
                        println!(
                            "{:<20} {:>5} {:>8} rejected (status {status})",
                            format!("{m}x{k}x{n}"),
                            ab_name,
                            compute_name(compute)
                        );
                        continue;
                    }
                    Ok(()) => {
                        let got = dc.download_f32(m * n);
                        want.as_ref().map_or((f32::NAN, f32::NAN), |w| err(&got, w))
                    }
                };
                let ms = ms_of(20, || {
                    blas.gemm_ex(m, k, n, da.0, db.0, dc.0, ab_type, compute)
                        .expect("gemm_ex");
                });
                let flops = 2.0 * m as f64 * k as f64 * n as f64;
                println!(
                    "{:<20} {:>5} {:>8} {:>9.3} {:>8.1} {:>9.2e} {:>9.1e}",
                    format!("{m}x{k}x{n}"),
                    ab_name,
                    compute_name(compute),
                    ms,
                    flops / (ms * 1e-3) / 1e12,
                    maxabs,
                    rel
                );
            }
        }

        // ---- 2. the same GEMM as a burn op, staged through the host ----
        // This is the only shape that works with no patch at all: read the f32
        // operands out, convert to f16, upload, GemmEx, download, write back
        // into a burn tensor. Checked against `a.matmul(b)` on the same
        // tensors, so the error column is the real one.
        println!("\n== 2. staged through the host (buildable with no patch) ==");
        println!(
            "{:<20} {:>10} {:>10} {:>9} {:>9} {:>9}",
            "shape", "burn ms", "staged ms", "TFLOP/s", "maxabs", "rel"
        );
        for (m, k, n) in [(256usize, 128usize, 128usize), (5120, 768, 2048), (5120, 2048, 8192)] {
            let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &device);
            let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.05), &device);
            let want: Vec<f32> = a
                .clone()
                .matmul(b.clone())
                .into_data()
                .try_to_vec()
                .expect("read the burn reference");

            // The `want` matmul above already paid the kernel JIT, so this is
            // warm. `into_scalar` is the drain: it makes the timed region cover
            // the queued work, not just the enqueue.
            let burn_ms = ms_of(20, || {
                let c = a.clone().matmul(b.clone());
                let s: f32 = c.sum().into_scalar();
                std::hint::black_box(s);
            });

            let da = Buf::new(m * k * 2);
            let db = Buf::new(k * n * 2);
            let dc = Buf::new(m * n * 4);
            let mut got;
            let once = || {
                let ah: Vec<f32> = a.clone().into_data().try_to_vec().expect("read a");
                let bh: Vec<f32> = b.clone().into_data().try_to_vec().expect("read b");
                upload_f16(da.0, &ah);
                upload_f16(db.0, &bh);
                blas.gemm_ex(m, k, n, da.0, db.0, dc.0, R_16F, COMPUTE_32F)
                    .expect("gemm_ex");
                sync();
                let out = dc.download_f32(m * n);
                // The last step any real caller needs: the result AS a burn
                // tensor, so the rest of the graph can consume it.
                let back =
                    Tensor::<2>::from_data(TensorData::new(out.clone(), [m, n]), &device);
                std::hint::black_box(back.into_data());
                out
            };
            got = once(); // warm: first-touch allocations in the read path
            let staged_ms = ms_of(5, || {
                got = once();
            });
            let (maxabs, rel) = err(&got, &want);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{:<20} {:>10.3} {:>10.3} {:>9.1} {:>9.2e} {:>9.1e}",
                format!("{m}x{k}x{n}"),
                burn_ms,
                staged_ms,
                flops / (staged_ms * 1e-3) / 1e12,
                maxabs,
                rel
            );
        }

        // ---- 3. f16 operands, D2D staging: what the pointer patch buys ----
        // The f32->f16 conversion is a cubecl kernel (a device-side pass at
        // device bandwidth), and what is left is D2D copies, also at device
        // bandwidth. This is the staged variant that can still beat burn's f32
        // matmul, and the only one that does.
        println!("\n== 3. f16 operands + D2D staging (the pointer-patch design) ==");
        println!(
            "{:<20} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "shape", "cast ms", "gemm ms", "d2d ms", "total ms", "TFLOP/s"
        );
        for (m, k, n) in [(5120usize, 768usize, 2048usize), (5120, 2048, 8192)] {
            let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &device);
            // burn's f32->bf16 cast round trip, which is the conversion the
            // design pays. The D2H inside makes it an upper bound on the
            // device-only cast.
            let c0 = a.clone().cast(FloatDType::BF16);
            std::hint::black_box(c0.into_data());
            let cast_ms = ms_of(20, || {
                let c = a.clone().cast(FloatDType::BF16);
                std::hint::black_box(c.into_data());
            });

            let da = Buf::new(m * k * 2);
            let db = Buf::new(k * n * 2);
            let dc = Buf::new(m * n * 4);
            // Same byte volume, from a different buffer, so the copy is real.
            let (sa, sb, sc) = (Buf::new(m * k * 2), Buf::new(k * n * 2), Buf::new(m * n * 4));
            let d2d_ms = ms_of(20, || {
                for (dst, src, bytes) in [
                    (da.0, sa.0, m * k * 2),
                    (db.0, sb.0, k * n * 2),
                    (dc.0, sc.0, m * n * 4),
                ] {
                    unsafe { cuMemcpyDtoD_v2(dst, src, bytes) }
                        .result()
                        .expect("D2D");
                }
            });
            let gemm_ms = ms_of(20, || {
                blas.gemm_ex(m, k, n, da.0, db.0, dc.0, R_16F, COMPUTE_32F)
                    .expect("gemm_ex");
            });
            let total = cast_ms + d2d_ms + gemm_ms;
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{:<20} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.1}",
                format!("{m}x{k}x{n}"),
                cast_ms,
                gemm_ms,
                d2d_ms,
                total,
                flops / (total * 1e-3) / 1e12
            );
        }
        #[cfg(feature = "cublas-native")]
        zero_copy(&blas);
        println!("\nprobe done");
    }

    /// ---- 4. zero-copy, the real integration (needs the pointer patch) ----
    /// A burn tensor's buffer resolved to a device address, a cuBLAS call
    /// enqueued on cubecl's own stream, and the result left in a burn tensor.
    /// No copy in, no copy out, no barrier: the call is ordered by the same
    /// stream the rest of the graph runs on.
    #[cfg(feature = "cublas-native")]
    fn zero_copy(blas: &Blas) {
        use cubecl::cuda::CudaServer;

        let device = Device::cuda(0);
        for (m, k, n) in [(256usize, 128usize, 256usize), (5120, 768, 2048)] {
            let a = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &device);
            let b = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.05), &device);
            let out = Tensor::<2>::zeros([m, n], &device);

            // burn's reference for the same inputs, and the drain that makes
            // every number below cover queued work rather than enqueues.
            let want: Vec<f32> = a
                .clone()
                .matmul(b.clone())
                .into_data()
                .try_to_vec()
                .expect("read the burn reference");

            // The whole patch: three handles -> three addresses, plus the
            // server's stream, in one round trip.
            let (a_p, b_p, c_p) = (
                a.clone().try_into_primitive().unwrap().handle.clone(),
                b.clone().try_into_primitive().unwrap().handle.clone(),
                out.clone().try_into_primitive().unwrap().handle.clone(),
            );
            let (ptrs, stream) = a
                .clone()
                .try_into_primitive().unwrap()
                .client
                .native_handles::<CudaServer>(vec![a_p.clone(), b_p, c_p])
                .expect("native_handles");
            let ptr = |i: usize| ptrs[i].expect("the backend resolved the handle");
            let stream = stream.expect("CUDA has a stream") as CUstream;
            println!(
                "\n== 4. zero-copy on cubecl's stream (m={m} k={k} n={n}) ==\n  \
                 a={:#x} b={:#x} c={:#x} stream={:#x}",
                ptr(0),
                ptr(1),
                ptr(2),
                stream as u64
            );
            // Same context (the primary one, which both threads share), so the
            // server's stream and handle are ours to enqueue on.
            blas.on_stream(stream);

            // f32 in, f32 out: this is the plumbing check. The f16 numbers are
            // section 1; the buffer dtype is a burn cast away.
            blas.gemm_ex(m, k, n, ptr(0), ptr(1), ptr(2), R_32F, COMPUTE_32F)
                .expect("gemm_ex on cubecl's stream");
            sync();
            let got: Vec<f32> = out
                .clone()
                .into_data()
                .try_to_vec()
                .expect("read the cuBLAS result");
            let (maxabs, rel) = err(&got, &want);
            let ms = ms_of(20, || {
                blas.gemm_ex(m, k, n, ptr(0), ptr(1), ptr(2), R_32F, COMPUTE_32F)
                    .expect("gemm_ex");
                sync();
            });
            let burn_ms = ms_of(20, || {
                let c = a.clone().matmul(b.clone());
                let s: f32 = c.sum().into_scalar();
                std::hint::black_box(s);
            });
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "  f32 in, f32 out: maxabs {maxabs:.2e} rel {rel:.1e} | \
                 cublas {ms:.3} ms ({:.1} TFLOP/s) vs burn {burn_ms:.3} ms ({:.1} TFLOP/s)",
                flops / (ms * 1e-3) / 1e12,
                flops / (burn_ms * 1e-3) / 1e12
            );

            // The production shape of the same call: the operands are f16
            // *in burn's own storage* (a cast, not a copy this probe made), the
            // result is f32, and nothing moves. If burn's f16 tensors work on
            // this stack this is the whole integration in one number.
            let a16 = a.clone().cast(FloatDType::F16);
            let b16 = b.clone().cast(FloatDType::F16);
            let want16: Vec<f32> = a16
                .clone()
                .matmul(b16.clone())
                .cast(FloatDType::F32)
                .into_data()
                .try_to_vec()
                .expect("read the f16 burn reference");
            let (a16_p, b16_p, out16_p) = (
                a16.clone().try_into_primitive().unwrap().handle.clone(),
                b16.clone().try_into_primitive().unwrap().handle.clone(),
                out.clone().try_into_primitive().unwrap().handle.clone(),
            );
            let (p16, _) = a16
                .clone()
                .try_into_primitive().unwrap()
                .client
                .native_handles::<CudaServer>(vec![a16_p, b16_p, out16_p])
                .expect("native_handles");
            let p16 = |i: usize| p16[i].expect("resolved");
            blas.gemm_ex(m, k, n, p16(0), p16(1), p16(2), R_16F, COMPUTE_32F)
                .expect("gemm_ex f16");
            sync();
            let got16: Vec<f32> = out
                .clone()
                .into_data()
                .try_to_vec()
                .expect("read the f16 cuBLAS result");
            let (maxabs16, rel16) = err(&got16, &want16);
            let ms16 = ms_of(20, || {
                blas.gemm_ex(m, k, n, p16(0), p16(1), p16(2), R_16F, COMPUTE_32F)
                    .expect("gemm_ex f16");
                sync();
            });
            let burn16_ms = ms_of(20, || {
                let c = a16.clone().matmul(b16.clone()).cast(FloatDType::F32);
                let s: f32 = c.sum().into_scalar();
                std::hint::black_box(s);
            });
            println!(
                "  f16 in, f32 out: maxabs {maxabs16:.2e} rel {rel16:.1e} | \
                 cublas {ms16:.3} ms ({:.1} TFLOP/s) vs burn(cast+matmul+cast) \
                 {burn16_ms:.3} ms ({:.1} TFLOP/s)",
                flops / (ms16 * 1e-3) / 1e12,
                flops / (burn16_ms * 1e-3) / 1e12
            );
        }
    }
}

fn main() {
    #[cfg(feature = "cublas-native")]
    poc::run();
    #[cfg(not(feature = "cublas-native"))]
    eprintln!(
        "cublas-poc: sections 0-3 measured without any patch.\n\
         Section 4 (zero-copy on cubecl's stream) needs the cubecl patch in \
         research/2026-09-27-cublas-integration-poc.md:\n\
         cargo run --release -p cublas-poc --features cublas-native"
    );
}
