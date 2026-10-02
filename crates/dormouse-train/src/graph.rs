//! CUDA-graph capture of the forward+backward window (`--graph-capture`).
//!
//! The card idles ~87% of a warm step because the step is **launch-bound**: the
//! host walks the runtime one kernel launch at a time (34.4 us of HOST time per
//! launch against 0.1 us of DEVICE time to run it —
//! `vendor/cubecl-fix/cubecl-cuda/tests/graph_step.rs`), and a graph replay is
//! ONE dispatch however many launches it contains. So most of a step's cost is
//! the host, and replaying the host's own launches is the only lever that
//! attacks it directly. Nothing here rewrites the math: a captured step runs
//! exactly the kernels burn already ran, in the order it already ran them.
//!
//! # What the window is, and why it is not the whole step
//!
//! `window = forward -> loss (+aux) -> NaN-mask -> backward -> sanitize_grads`.
//! Outside it: `opt -> retract -> ema`, ~70 ms of a ~460 ms step
//! (`first_run_100000`: `opt=46 retr=21 ema=2.5`), so the window holds ~85% of
//! the launches. The tail is outside for one reason that is not about time:
//! burn's optimizer is **out of place**
//! (`burn-optim-0.22.0-pre.4/src/optim/adam.rs:88`, `(tensor - delta, Some(state))`),
//! so it allocates a fresh parameter every step and hands the old buffer back to
//! the pool — which is precisely what a captured address cannot survive. The
//! pin below fixes that for the parameter for one launch per parameter; getting
//! the optimizer's own state right is a bigger surface for a smaller share of
//! the step, and v1 does not pretend otherwise.
//!
//! # Why anything outside the window must not move
//!
//! Three classes of tensor are read by the window and are NOT allocated inside
//! it, so a capture bakes an address the next step may not honour:
//!
//! | what | why it moves | the pin |
//! |---|---|---|
//! | every parameter | `optim.step` returns a fresh tensor and frees the old one | one `copy_into` per parameter per step, then the model's `Param` is re-pointed at the master |
//! | the EMA teacher | `ema_update` rebuilds every teacher `Param` | the same, on the teacher |
//! | `x`, `y`, `hashed_ids` | built fresh from host bytes every step | the same, into pre-allocated buffers |
//!
//! Everything else the window allocates is RETAINED by that capture
//! (`PersistentPool::retain_touched`) and is therefore stable by construction —
//! including the **gradients**, which is why the graph may keep rewriting them
//! and why the optimizer is fed from ONE `Gradients` object for the life of the
//! capture ([`grads_params`]).
//!
//! # The re-capture contract
//!
//! A capture window refuses stream reads, syncs and handle writes
//! (`cubecl-runtime/src/client.rs:1288-1296`), so every step that reads
//! anything back (log cadence, host-adam cadence, timers) and every step that
//! drains the pool (`memory_cleanup`, `max_ortho`, every 500th) runs UNGRAPHED
//! and forces a re-capture. `stop_capture` refuses a window that allocated — a
//! memory node makes the graph un-relaunchable
//! (`cubecl-cuda/src/compute/capture.rs:110-120`) — and a second capture over a
//! live graph is refused for the same reason, so the seam always destroys,
//! prepares, captures.
//!
//! # Loud, never silent (ADR-0011)
//!
//! - `--graph-capture` beside an arm that changes the window's SHAPE
//!   (`--rand-depth`), or that feeds it a per-step tensor this version does not
//!   pin (`--engram-ram`'s row gather, `--jepa-targets`), is a hard error naming
//!   the arm — in [`check`], before any GPU work.
//! - A refused capture is COUNTED and its reason printed once verbatim; the run
//!   continues ungraphed, which is correct and as slow as before. A capture that
//!   failed quietly would be the `8fa5d4c` shape.
//! - The report line names replays, captures, refusals and the pin's price, so a
//!   graph that captured once and replayed nothing reads as the null it is.
//!
//! Off CUDA the seam still owns the `Gradients` (one code path for the caller)
//! and never captures.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    tensor::{Gradients, Tensor},
};
use dormouse_core::DormouseModel;

/// A pinned master buffer, rank-erased **without ever reshaping**: the copy and
/// the re-point both happen inside `map_float::<D>`, which is monomorphised
/// per rank, so the rank is a compile-time fact at every use and the downcast
/// below can only fail for a rank this enum does not carry (which is a loud
/// refusal, never a wrong-shaped copy).
enum Master {
    R1(Tensor<1>),
    R2(Tensor<2>),
    R3(Tensor<3>),
    R4(Tensor<4>),
}

impl Master {
    fn new<const D: usize>(t: &Tensor<D>) -> Option<Self> {
        let any = t as &dyn Any;
        if D == 1 {
            any.downcast_ref::<Tensor<1>>().map(|t| Master::R1(t.clone()))
        } else if D == 2 {
            any.downcast_ref::<Tensor<2>>().map(|t| Master::R2(t.clone()))
        } else if D == 3 {
            any.downcast_ref::<Tensor<3>>().map(|t| Master::R3(t.clone()))
        } else {
            any.downcast_ref::<Tensor<4>>().map(|t| Master::R4(t.clone()))
        }
    }

    /// `master <- now`, one launch. `false` if the ranks disagree (they cannot
    /// for a given `ParamId`) or the copy primitive refused.
    ///
    /// Both sides go through `dyn Any` rather than a const-generic pair, because
    /// the enum arm fixes ONE rank while the caller's `D` is only known as a
    /// type parameter. Matching them inside the arm is what makes this four
    /// monomorphic calls instead of four mismatched-type errors.
    ///
    /// **Both sides go through [`as_constant_float`], not just the source.** The
    /// master was built once at `Arm` time from the live parameter, so its own
    /// kind is `DispatchTensorKind::Autodiff(..)` just like the parameter's, and
    /// `try_into_primitive` refuses it with
    /// `KindMismatch("Expected Float primitive, got variant: Autodiff")`. That
    /// is what the first float-pin run said, with the source already unwrapped:
    /// the fix had been applied to one side of a two-sided copy.
    fn put(&self, now: &dyn Any) -> Result<(), String> {
        macro_rules! arm {
            ($t:ty, $v:expr) => {{
                match (
                    (now as &dyn Any).downcast_ref::<$t>(),
                    ($v as &dyn Any).downcast_ref::<$t>(),
                ) {
                    (Some(a), Some(b)) => copy_into(&as_constant_float(a), &as_constant_float(b)),
                    _ => Err("master and parameter are not the same rank".into()),
                }
            }};
        }
        match self {
            Master::R1(t) => arm!(Tensor<1>, t),
            Master::R2(t) => arm!(Tensor<2>, t),
            Master::R3(t) => arm!(Tensor<3>, t),
            Master::R4(t) => arm!(Tensor<4>, t),
        }
    }

    fn get<const D: usize>(&self) -> Option<Tensor<D>> {
        macro_rules! arm {
            ($v:expr) => {{
                match ($v as &dyn Any).downcast_ref::<Tensor<D>>() {
                    Some(t) => Some(t.clone()),
                    None => None,
                }
            }};
        }
        match self {
            Master::R1(t) => arm!(t),
            Master::R2(t) => arm!(t),
            Master::R3(t) => arm!(t),
            Master::R4(t) => arm!(t),
        }
    }

    /// The rank this master carries. Read by the tests that assert the enum
    /// round-trips; not on the step's path.
    #[allow(dead_code)]
    fn rank(&self) -> usize {
        match self {
            Master::R1(_) => 1,
            Master::R2(_) => 2,
            Master::R3(_) => 3,
            Master::R4(_) => 4,
        }
    }
}

/// The pinned copies of one module's float parameters, keyed by [`ParamId`].
#[derive(Default)]
pub struct Pins {
    masters: HashMap<ParamId, Master>,
    /// Why a pin could not be built, if one could not be. Loud, never silent.
    pub unsupported_rank: Option<usize>,
}

impl Pins {
    /// One `copy_into` per parameter per step: the pin's price, counted here
    /// rather than estimated in a comment.
    pub fn copies(&self) -> usize {
        self.masters.len()
    }

    pub fn armed(&self) -> bool {
        !self.masters.is_empty() && self.unsupported_rank.is_none()
    }
}

/// What [`PinMapper`] does to each parameter it meets.
#[derive(Clone, Copy, PartialEq)]
enum PinAction {
    /// Allocate a master per parameter and point the parameter at it.
    Arm,
    /// Copy the current tensor into its master and re-point the parameter.
    Refresh,
}

struct PinMapper<'a> {
    action: PinAction,
    pins: &'a mut Pins,
    /// Set by [`PinAction::Refresh`] if any copy was refused, and WHY.
    failed: bool,
    why: Option<String>,
}

impl ModuleMapper for PinMapper<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        match self.action {
            PinAction::Arm => {
                let master = match Master::new(&tensor) {
                    Some(m) => {
                        let t = m.get::<D>().expect("just built at this rank");
                        self.pins.masters.insert(id, m);
                        t
                    }
                    None => {
                        self.pins.unsupported_rank = Some(D);
                        return Param::from_mapped_value(id, tensor, mapper);
                    }
                };
                Param::from_mapped_value(id, master, mapper)
            }
            PinAction::Refresh => {
                let Some(master) = self.pins.masters.get(&id) else {
                    self.failed = true;
                    return Param::from_mapped_value(id, tensor, mapper);
                };
                if let Err(why) = master.put(&as_constant_float(&tensor)) {
                    self.failed = true;
                    self.why = Some(format!("parameter {id:?}: {why}"));
                    return Param::from_mapped_value(id, tensor, mapper);
                }
                match master.get::<D>() {
                    Some(t) => Param::from_mapped_value(id, t, mapper),
                    None => {
                        self.failed = true;
                        Param::from_mapped_value(id, tensor, mapper)
                    }
                }
            }
        }
    }
}

/// The gradients of one window, borrowed rather than REMOVED into the
/// optimizer — the reason the whole thing works.
///
/// burn's own converter is `GradientsParams::from_module`, which calls
/// `Tensor::grad_remove`: it takes the gradient OUT of the container. A replay
/// runs no host code, so there is exactly ONE `Gradients` object for the life
/// of the capture, and removing from it would leave the second replay with an
/// empty optimizer input — **a silently frozen parameter set**, the shape of
/// `8fa5d4c`. `Tensor::grad` returns the same handle without removing it, so
/// this visitor is safe to call on every replay.
struct GradBorrow<'a> {
    grads: &'a Gradients,
    out: &'a mut burn::optim::GradientsParams,
}

impl ModuleVisitor for GradBorrow<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !param.is_require_grad() {
            return;
        }
        if let Some(g) = param.val().grad(self.grads) {
            self.out.register(param.id, g);
        }
    }
}

/// `GradientsParams` for `module`, WITHOUT consuming `grads`.
pub fn grads_params<M: Module>(grads: &Gradients, module: &M) -> burn::optim::GradientsParams {
    let mut out = burn::optim::GradientsParams::new();
    let mut v = GradBorrow { grads, out: &mut out };
    module.visit(&mut v);
    out
}

/// Steps replayed / captured / refused, and why the last refusal happened.
#[derive(Default)]
pub struct Stats {
    pub replays: u64,
    pub captures: u64,
    pub refusals: u64,
    /// Set after [`MAX_REFUSALS`]: the capture is not attempted again. A refused
    /// capture has already grown the persistent pool by one working set (the
    /// priming pass keeps every slice it touched), so a run that retries every
    /// step and is refused every step would grow that pool until it OOMs.
    /// Stopping is loud — it is in the report line — and correct, because an
    /// ungraphed step trains exactly what a replayed one would.
    pub disabled: bool,
    pub last_refusal: Option<String>,
}

impl Stats {
    pub fn line(&self) -> String {
        format!(
            "graph: captured {} replayed {} refused {}{}{}",
            self.captures,
            self.replays,
            self.refusals,
            if self.disabled { " DISABLED after repeated refusals" } else { "" },
            match &self.last_refusal {
                Some(r) => format!(" last refusal: {r}"),
                None => String::new(),
            }
        )
    }
}

/// How many refusals before the seam stops trying. See [`Stats::disabled`].
const MAX_REFUSALS: u64 = 20;

/// The same BUFFER as a float `t`, with burn's autodiff context marked `Disabled`, so
/// `Tensor::try_into_primitive` will hand out its raw cubecl handle.
///
/// **Why this is needed, exactly.** `try_into_primitive` refuses any tensor
/// whose autodiff context is not `Disabled`
/// (`burn-dispatch/src/tensor.rs:481`), and every tensor the trainer holds is
/// `Enabled(Disabled)`: a parameter needs gradients, and an Int input tensor is
/// created on `Device::cuda(0).autodiff()`. burn's Int tensors have no
/// `detach()` (it is float-only), so there is no other route to a handle - which
/// is why the first two versions of the pin refused with
/// `BackendMismatch("Expected concrete Cube backend with disabled autodiff
/// context")`.
///
/// **Why this is sound.** `DispatchTensor.autodiff` is a `pub` field and
/// `Tensor::into_dispatch` / `from_dispatch` are public, so this is the same
/// thing burn itself does to an op input: a `Disabled` context means "treat
/// this buffer as a constant" (the context-merge rule, `tensor.rs:216-220`).
/// Nothing about the data, the device or the allocation changes; the returned
/// tensor is never fed back into a burn op, only to a raw kernel that copies the
/// same bytes. If a future burn makes the field private this stops compiling,
/// loudly, which is the right way to find out.
///
/// The unwraps below are a read of burn's type, not a trick: they take the
/// handle out of the two enums that wrap it and put the SAME handle back in the
/// variant a float belongs in. The buffer, its address and its contents are
/// untouched — `into_primitive` moves the `CubeTensor` out and hands it straight
/// back.
fn as_constant_float<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    // Destructured and rebuilt, not mutated in place: the `kind` arm MOVES out
    // of the enum, so the struct is partially moved by the time
    // `from_dispatch` wants it whole.
    let burn_dispatch::DispatchTensor { kind, mut autodiff } = t.clone().into_dispatch();
    autodiff = burn_dispatch::DispatchAutodiffContext::Disabled;
    // TWO nested unwraps, and the first version of this did one of them and
    // still failed, which is what the gate history in the review records. A
    // tracked float is `Autodiff(Box<Kind>)` — and what is INSIDE the box is
    // `Cube(BackendTensor<Cube>)`, whose `BackendTensor` is ITSELF an enum with
    // an `Autodiff` variant. So a tracked float is three enums deep:
    //
    //   Autodiff(Box<Autodiff>)   <- DispatchTensorKind, the box
    //     -> Cube(BackendTensor::Autodiff(FloatTensor<Autodiff<Cube>>))
    //                                     <- BackendTensor, a second enum
    //
    // and `try_into_primitive` matches on the INNERMOST one, refusing anything
    // that is not `BackendTensor::Float`
    // (`burn-tensor/src/tensor/api/extension.rs:167`). Unwrapping only the box
    // leaves `BackendTensor::Autodiff` and the error is exactly
    // `KindMismatch("Expected Float primitive, got variant: Autodiff")` —
    // the message the run reported. An Int tensor is `BackendTensor::Int` at
    // the innermost level already (ints do not track), which is why
    // `as_constant_int` has nothing to unwrap and the Int half of the pin was
    // green while the float half was red: the same call, two answers.
    let kind = match kind {
        burn_dispatch::DispatchTensorKind::Autodiff(inner) => *inner,
        other => return Tensor::from_dispatch(burn_dispatch::DispatchTensor { kind: other, autodiff }),
    };
    let kind = match kind {
        burn_dispatch::DispatchTensorKind::Cube(burn_dispatch::BackendTensor::Autodiff(f)) => {
            burn_dispatch::DispatchTensorKind::Cube(burn_dispatch::BackendTensor::Float(
                f.into_primitive(),
            ))
        }
        other => other,
    };
    Tensor::from_dispatch(burn_dispatch::DispatchTensor { kind, autodiff })
}

/// [`as_constant_float`] for Int tensors. One helper per kind, not one generic:
/// burn's `Basic` bound is `pub(crate)`, so a caller cannot name a bound that
/// covers both kinds and `into_dispatch` is defined per kind.
fn as_constant_int<const D: usize>(t: &Tensor<D, burn::tensor::Int>) -> Tensor<D, burn::tensor::Int> {
    let mut d = t.clone().into_dispatch();
    d.autodiff = burn_dispatch::DispatchAutodiffContext::Disabled;
    Tensor::from_dispatch(d)
}

/// The seam's own unit: one float-pin copy, exactly what [`Master::put`]
/// executes every step — both sides through [`as_constant_float`], then
/// `copy_into`. Public for the integration gate that pins this path
/// (`float_pin_roundtrips_through_as_constant`); a copy that launches and
/// writes elsewhere would otherwise surface only as silently-frozen masters.
#[doc(hidden)]
pub fn float_pin_copy<const D: usize>(src: &Tensor<D>, dst: &Tensor<D>) -> Result<(), String> {
    copy_into(&as_constant_float(src), &as_constant_float(dst))
}

/// Copy `src` into `dst`'s existing buffer: one launch, no allocation.
///
/// A CUDA-graph pin is exactly this operation and nothing else — the graph has
/// baked `dst`'s address, so a new parameter has to be moved into it. burn has
/// no copy-into-existing-storage that can be relied on to stay in place:
/// `slice_assign` returns whatever the backend allocated, and `reshape` is a
/// metadata op whose buffer behaviour is a backend detail. `false` off CUDA,
/// where there is no graph to pin for.
#[cfg(feature = "cuda")]
pub fn copy_into<const D: usize>(src: &Tensor<D>, dst: &Tensor<D>) -> Result<(), String> {
    burn_muon_plus::fused_kernels::copy_into_cuda(src, dst)
}
#[cfg(not(feature = "cuda"))]
pub fn copy_into<const D: usize>(_src: &Tensor<D>, _dst: &Tensor<D>) -> Result<(), String> {
    Err("no CUDA backend, so no graph to pin for".into())
}

/// The same pin for the Int inputs: `x`, the shifted `y`, the hashed keys.
#[cfg(feature = "cuda")]
pub fn copy_into_int<const D: usize>(
    src: &Tensor<D, burn::tensor::Int>,
    dst: &Tensor<D, burn::tensor::Int>,
) -> Result<(), String> {
    burn_muon_plus::fused_kernels::copy_into_i32_cuda(src, dst)
}
#[cfg(not(feature = "cuda"))]
pub fn copy_into_int<const D: usize>(
    _src: &Tensor<D, burn::tensor::Int>,
    _dst: &Tensor<D, burn::tensor::Int>,
) -> Result<(), String> {
    Err("no CUDA backend, so no graph to pin for".into())
}

/// The per-step input tensors, each pinned to a buffer that never moves.
///
/// `x`, `y` and `hashed_ids` are built from host bytes every step, so a
/// captured graph would read last step's batch — the `8fa5d4c` shape again, one
/// layer down. These are allocated ONCE and filled by copy: one launch per
/// tensor per step against the window's 21,433.
///
/// **Both sides of the copy go through [`as_constant_float`] / [`as_constant_int`]**,
/// because a raw cubecl
/// `Handle` is only reachable from a tensor whose autodiff context is
/// `Disabled`, and neither an Int input nor a parameter built by the optimizer
/// is. That is the whole reason this pin has a helper in it.
///
/// `h` is `None` when the window does not CONSUME keys: a key buffer nothing
/// reads would be ceremony on the `--no-engram` shape.
///
/// # The keys gate keys on consumption, not supply (2026-10-02)
///
/// The byte stream ALWAYS produces FNV keys (`ByteStream::next_batch` computes
/// `hashes_raw` unconditionally, ORDERS=[2,3,4]), so every step arrives with
/// keys in hand whether or not the model reads them. The window's key-ness is
/// therefore a fact about the MODEL (`keys_consumed` — the Engram arm fed from
/// the in-VRAM tables), not about the batch:
///
/// - `keys_consumed = true` and a step arrives with keys — the normal run,
///   one copy per step.
/// - `keys_consumed = true` and a step arrives with none — **a loud refusal**
///   (`feed`). There is no correct way to run it: the captured forward reads a
///   key buffer, so feeding nothing means the memory arm trains on whatever
///   the buffer already held (the `7adda92` shape — a measured eval running a
///   different program than the trained one).
/// - `keys_consumed = false` and keys are supplied — dropped, not refused.
///   The model never reads them (`loop_block` takes its `use_engram` branch
///   before `hashed_ids`), so `None` keys ARE the correct forward, and the
///   first graph-mode run that had this backwards refused every capture with
///   "this step has hashed keys and the pinned window has none" while the
///   run's config was `--no-engram`. The armed line prints
///   `key window FED|none` once, so the reader of the log can tell which arm
///   ran without re-deriving the config (ADR-0019).
pub struct InputPins {
    pub x: Tensor<2, burn::tensor::Int>,
    pub y: Tensor<2, burn::tensor::Int>,
    pub h: Option<Tensor<3, burn::tensor::Int>>,
    /// Whether the window reads the keys, fixed at arm time from the run's
    /// own consumption (`use_engram` and not the host-offload path).
    keys_consumed: bool,
}

impl InputPins {
    /// Allocate the buffers from the first batch's shapes. `h` is the first
    /// batch's keys (the stream always supplies them — it is the shape
    /// carrier here); `keys_consumed` decides whether the window pins a key
    /// buffer at all, and that verdict is the model's, not the stream's.
    pub fn new(
        first: &Tensor<2, burn::tensor::Int>,
        y: &Tensor<2, burn::tensor::Int>,
        h: &Tensor<3, burn::tensor::Int>,
        keys_consumed: bool,
    ) -> Self {
        let x = Tensor::zeros(first.dims(), &first.device());
        let yp = Tensor::zeros(y.dims(), &y.device());
        let hp = keys_consumed.then(|| Tensor::zeros(h.dims(), &h.device()));
        Self { x, y: yp, h: hp, keys_consumed }
    }

    /// Copy this step's batch into the pinned buffers and hand them over.
    /// `Err` names what could not be pinned or consumed, because a graph fed
    /// a buffer that moves — or handed keys a window refuses to read — is the
    /// silent failure this module exists to prevent.
    pub fn feed(
        &mut self,
        x: &Tensor<2, burn::tensor::Int>,
        y: &Tensor<2, burn::tensor::Int>,
        h: Option<&Tensor<3, burn::tensor::Int>>,
    ) -> Result<(Tensor<2, burn::tensor::Int>, Tensor<2, burn::tensor::Int>, Option<Tensor<3, burn::tensor::Int>>), String> {
        if x.dims() != self.x.dims() {
            return Err(format!(
                "graph capture: the input shape changed ({} vs the pinned {}) — a captured \
                 window has one shape. Batch/seq must be constant.",
                x.dims()[1] * x.dims()[0],
                self.x.dims()[1] * self.x.dims()[0]
            ));
        }
        if let Err(e) = copy_into_int(&as_constant_int(x), &as_constant_int(&self.x)) {
            return Err(format!("graph capture: could not pin x: {e}"));
        }
        if let Err(e) = copy_into_int(&as_constant_int(y), &as_constant_int(&self.y)) {
            return Err(format!("graph capture: could not pin y: {e}"));
        }
        if !self.keys_consumed {
            // The model reads no keys this run (no Engram arm, or the arm fed
            // from host rows): the supplied keys are dead weight and the
            // memory-less forward is the CORRECT program, so drop them. The
            // armed line already printed `key window none` — the reader can
            // tell which arm every step took.
            return Ok((self.x.clone(), self.y.clone(), None));
        }
        match h {
            Some(src) => {
                if let Err(e) = copy_into_int(
                    &as_constant_int(src),
                    &as_constant_int(self.h.as_ref().expect("keyed arm built the buffer")),
                ) {
                    return Err(format!("graph capture: could not pin hashed_ids: {e}"));
                }
            }
            None => {
                return Err(
                    "graph capture: the window consumes hashed keys but this step supplies \
                     none — replaying it would train a memory-less model on a keyed run"
                        .into(),
                );
            }
        }
        Ok((
            self.x.clone(),
            self.y.clone(),
            Some(self.h.clone().expect("keyed arm built the buffer")),
        ))
    }
}

/// How many times the window runs BEFORE the recorded pass.
///
/// Not a tuning knob: it is the capture protocol. `graph_prepare` opens a
/// PRIMING window in which every allocation is forced into the persistent pool
/// and RETAINED (`capture_touch`), so one pass grows the pool to that pass's
/// full working set instead of its transient peak. `start_capture` ends priming
/// (`Window::begin` → `capture_priming_end`) and releases those slices as free,
/// and the recorded pass then REUSES them.
///
/// Without the warmup the recorded pass allocates everything, every allocation
/// is a memory node, and `stop_capture` refuses the capture with the reason
/// spelled out — which is the failure mode this constant exists to prevent.
/// One pass is the documented minimum (`graph_step.rs` uses two on a toy);
/// if a capture is refused anyway the next attempt runs one more, so the pool
/// keeps whatever it grew and the seam heals instead of failing.
const CAPTURE_WARMUP: usize = 1;

/// What the seam needs from a captured graph; the `cfg` split exists only so
/// the CPU build carries no CUDA type.
pub trait GraphReplay {
    fn replay(&self) -> Result<(), String>;
}

#[cfg(feature = "cuda")]
impl GraphReplay for cubecl_runtime::client::Graph {
    fn replay(&self) -> Result<(), String> {
        // SAFETY: the handle is a live `cuGraphExec`; `replay` dispatches it on
        // the stream it was captured on.
        unsafe { self.replay() }.map_err(|e| format!("{e:?}"))
    }
}

#[cfg(not(feature = "cuda"))]
mod nothing {}

/// The graph seam of one run: the pins, the captured graph, the retained
/// gradients and the counters.
#[derive(Default)]
pub struct Seam {
    #[cfg(feature = "cuda")]
    client: Option<cubecl_runtime::client::Client>,
    graph: Option<Arc<dyn GraphReplay>>,
    grads: Option<Gradients>,
    pub model_pins: Pins,
    pub teacher_pins: Pins,
    pub stats: Stats,
    /// Launches the pin has cost over the run: one copy per parameter, model
    /// and teacher, per step.
    pub pin_launches: u64,
}

impl Seam {
    /// `client` is `None` off CUDA, where `step` always runs the body. Off CUDA
    /// there is no client to keep, so it is dropped rather than stored in a
    /// field that would exist only to be `None`.
    pub fn new(client: Option<cubecl_runtime::client::Client>) -> Self {
        #[cfg(feature = "cuda")]
        {
            Self { client, ..Default::default() }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = client;
            Self::default()
        }
    }

    /// Pin the model and, when there is one, the teacher: every address the
    /// window reads becomes one nothing else can be handed.
    pub fn arm(
        &mut self,
        model: DormouseModel,
        teacher: Option<DormouseModel>,
    ) -> (DormouseModel, Option<DormouseModel>) {
        (map(model, &mut self.model_pins, PinAction::Arm), teacher.map(|t| map(t, &mut self.teacher_pins, PinAction::Arm)))
    }

    /// Copy every parameter of `module` into its master and re-point it there.
    ///
    /// `false` = the pin could not be built or maintained (an unrepresentable
    /// rank, or a copy primitive that refused). The caller must then stop
    /// replaying: a graph whose inputs move is silent-wrong training, and a
    /// refused pin is the one signal that says so.
    pub fn refresh(
        &mut self,
        module: DormouseModel,
        teacher: bool,
    ) -> (DormouseModel, Result<(), String>) {
        let pins = if teacher { &mut self.teacher_pins } else { &mut self.model_pins };
        self.pin_launches += pins.copies() as u64;
        let unsupported = pins.unsupported_rank;
        let mut m = PinMapper { action: PinAction::Refresh, pins, failed: false, why: None };
        let module = module.map(&mut m);
        if m.failed {
            let why = unsupported
                .map(|r| format!("parameter of rank {r} has no pin representation"))
                .or(m.why)
                .unwrap_or_else(|| "a copy_into was refused".into());
            return (module, Err(why));
        }
        (module, Ok(()))
    }

    pub fn armed(&self) -> bool {
        self.model_pins.armed()
    }

    pub fn grads(&self) -> Option<&Gradients> {
        self.grads.as_ref()
    }

    /// One step's window: replay the graph, or run `body` and capture it.
    ///
    /// `ungraphed` is the caller's cadence verdict — a step that reads anything
    /// back cannot be inside a capture window. The window's gradients stay in
    /// the seam so a replay hands the optimizer the same ones with no host work
    /// at all.
    ///
    /// `body` is `Fn`, not `FnOnce`: a capture runs it [`CAPTURE_WARMUP`]
    /// times to populate the pool and then once more inside the recording.
    pub fn step(
        &mut self,
        ungraphed: bool,
        body: impl Fn() -> Gradients,
    ) -> Result<(), String> {
        if !ungraphed {
            if let Some(graph) = &self.graph {
                graph.replay().map_err(|e| {
                    format!("graph replay failed: {e} — a run cannot continue on a graph it cannot drive")
                })?;
                self.stats.replays += 1;
                return Ok(());
            }
        }
        // Ungraphed. The graph is invalid from here (a step that read something
        // back may have used the pool), so it goes first: destroy, prepare,
        // capture, never capture over a live graph.
        self.graph = None;
        #[cfg(feature = "cuda")]
        {
            if !ungraphed && !self.stats.disabled {
                if let Some(client) = &self.client {
                    if let Err(e) = client.graph_prepare() {
                        self.grads = Some(body());
                        return self.refused(format!("graph_prepare: {e:?}"));
                    }
                    // The warmup, OUTSIDE the recording: this is the pass that
                    // grows the persistent pool to the window's working set, so
                    // the recorded pass reuses slices instead of allocating
                    // (an allocation inside the window is a memory node, and a
                    // memory node makes the graph un-relaunchable).
                    for _ in 0..CAPTURE_WARMUP {
                        let _ = body();
                    }
                    if let Err(e) = client.start_capture() {
                        self.grads = Some(body());
                        return self.refused(format!("start_capture: {e:?}"));
                    }
                    // The body always runs (the closure is Fn): its grads are
                    // the buffers' pre-capture contents until the replay below
                    // executes the recording, and a capture refused at
                    // `stop_capture` must still hand the caller SOMETHING for
                    // the step it just consumed.
                    self.grads = Some(body());
                    return match client.stop_capture() {
                        Ok(graph) => {
                            self.graph = Some(Arc::new(graph));
                            self.stats.captures += 1;
                            // CUDA stream capture RECORDS work; it does not
                            // execute it. The gradients `body()` handed back
                            // are the buffers' PRE-capture contents — measured
                            // 2026-10-02 (`probe_replay_matches_fresh_window`):
                            // iter_embed's grad read exactly 0.0 (a
                            // never-written page) and embedding's 190x a real
                            // gradient, while the SAME graph's first replay
                            // wrote gradients equal to a fresh window's to six
                            // decimals. Execute the recorded step now: one
                            // replay launches the recorded kernels and writes
                            // the real results into the buffers this seam
                            // already handed to the caller.
                            if let Some(g) = &self.graph {
                                g.replay()
                                    .map_err(|e| format!("graph replay after capture failed: {e}"))?;
                                self.stats.replays += 1;
                            }
                            Ok(())
                        }
                        Err(e) => self.refused(format!("stop_capture: {e:?}")),
                    };
                }
            }
        }
        self.grads = Some(body());
        Ok(())
    }

    fn refused(&mut self, reason: String) -> Result<(), String> {
        self.stats.refusals += 1;
        if self.stats.refusals == 1 {
            println!(
                "graph: capture REFUSED ({reason}). Continuing UNGRAPHED: correct, and as slow \
                 as before --graph-capture. A refusal that names a memory node means the window \
                 grew the pool — it must allocate nothing new, so capture later in the run."
            );
        }
        if self.stats.refusals == MAX_REFUSALS {
            self.stats.disabled = true;
            println!(
                "graph: {MAX_REFUSALS} refusals, so the capture is DISABLED for the rest of this \
                 run. Each attempt grew the persistent pool by a working set; retrying forever \
                 would OOM. The run continues ungraphed and trains exactly what it would have."
            );
        }
        self.stats.last_refusal = Some(reason);
        Ok(())
    }

    pub fn report(&self) -> String {
        format!(
            "{} ({} replayed of {} graphed steps, pin={} launches)",
            self.stats.line(),
            self.stats.replays,
            self.stats.replays + self.stats.captures,
            self.pin_launches
        )
    }
}

fn map(module: DormouseModel, pins: &mut Pins, action: PinAction) -> DormouseModel {
    module.map(&mut PinMapper { action, pins, failed: false, why: None })
}

/// Which arms `--graph-capture` cannot be combined with, named.
///
/// Pure so it is testable without a GPU: a flag that changes the window's SHAPE
/// or feeds it an unpinned per-step tensor would make the captured graph a
/// different computation than the run claims to be doing, silently.
pub fn check(cfg: &crate::RunCfg) -> Result<(), String> {
    if !cfg.train.graph_capture {
        return Ok(());
    }
    // The arm refusals come FIRST so each one is reachable (and testable) off
    // CUDA: on a CPU build the "needs the cuda feature" refusal would otherwise
    // swallow all three and the arm gates would never run.
    if cfg.train.rand_depth {
        return Err(
            "--graph-capture refuses --rand-depth: the captured graph bakes ONE loop depth, and \
             the sampled depth changes the window's shape every step."
                .into(),
        );
    }
    if cfg.train.engram_ram {
        return Err(
            "--graph-capture refuses --engram-ram: host_rows is gathered per batch (a variable \
             number of rows) and uploaded per step, which v1 does not pin. With \
             --host-adam-every 1 that is also a host round-trip on EVERY step, so no step would \
             be replayable anyway."
                .into(),
        );
    }
    if cfg.train.jepa_targets.is_some() {
        return Err(
            "--graph-capture refuses --jepa-targets: the target tensor is looked up and uploaded \
             per chunk, which v1 does not pin."
                .into(),
        );
    }
    #[cfg(not(feature = "cuda"))]
    return Err(
        "--graph-capture needs the cuda feature: there is no graph on another backend, and \
         silently training without one would be a flag that does nothing."
            .into(),
    );
    #[cfg(feature = "cuda")]
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_line_names_the_three_numbers_and_the_refusal() {
        let mut s = Stats::default();
        assert!(s.line().contains("captured 0 replayed 0 refused 0"), "{}", s.line());
        s.captures = 2;
        s.replays = 7;
        s.refusals = 1;
        s.last_refusal = Some("stop_capture: memory node".into());
        let line = s.line();
        assert!(line.contains("captured 2 replayed 7 refused 1"), "{line}");
        assert!(line.contains("stop_capture: memory node"), "{line}");
    }

    /// A master's rank survives the round trip through the rank-erased enum, and
    /// a rank it does not carry is a refusal rather than a wrong-shaped copy.
    #[test]
    fn a_master_keeps_its_rank_and_refuses_one_it_cannot_carry() {
        let dev = burn::tensor::Device::default();
        let t: Tensor<2> = Tensor::ones([4, 3], &dev);
        let m = Master::new(&t).expect("rank 2 is carried");
        assert_eq!(m.rank(), 2);
        assert_eq!(m.get::<2>().expect("same rank back").dims(), [4, 3]);
        assert!(m.get::<3>().is_none(), "rank 3 must not come back as rank 2");
        // The put/get pair is what the pin does every step. Off CUDA the copy
        // primitive refuses (there is no cubecl handle) and that refusal is the
        // honest answer; on CUDA it must succeed. Either way it may not panic.
        #[cfg(not(feature = "cuda"))]
        assert!(m.put(&t).is_err(), "off CUDA a copy_into cannot succeed");
        #[cfg(feature = "cuda")]
        m.put(&t).expect("on CUDA the pin copy must succeed");
    }
}
