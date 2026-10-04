//! `plain9m` — the plain byte Transformer control of the controlled trio
//! (plain byte Transformer vs byteflow vs dormouse, one recipe, equal bytes;
//! `docs/reviews/trio-prep-2026-10-04.md`).
//!
//! The parameter count of a dormouse run is NOT the number of parameters that
//! train, and the gap is large enough to re-frame every architecture
//! comparison in this repo's archive. `LoopBlock::new` and `AuxHeads::new`
//! build their subtrees UNCONDITIONALLY — the KDA arm, the Engram tables with
//! their key/value projections, `mem_dense`, the MoR router and the three aux
//! heads — so a run with `--no-engram` and every aux weight at 0 still carries
//! all of them as parameters, in the checkpoint, and in the optimizer's
//! groups, while contributing nothing to any gradient.
//!
//! So this file MEASURED it instead of arguing it: one forward + backward per
//! preset, and every parameter whose `Param::grad` is `None` is reported by
//! path and by size. `Param::grad` returning `None` is the same question the
//! optimizer asks when it decides not to step a parameter, so the "dead" set is
//! what the trainer leaves at its initial value, not a proxy for it.
//!
//! Run: `cargo test -p dormouse-core --test plain9m_control`
//! Full width (minutes on this box's CPU backend):
//!     cargo test -p dormouse-core --test plain9m_control -- --ignored --nocapture
use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::autodiff::Autodiff;
use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use dormouse_core::config::{load_config, validate};
use dormouse_core::{fnv_hash, DormouseConfig, DormouseModel};

#[allow(deprecated)] // the alias the train crate uses for `--features cpu`
type B = Autodiff<burn::backend::Flex, BalancedCheckpointing>;

#[allow(deprecated)]
fn device() -> Device {
    Device::flex().autodiff()
}

const BATCH: usize = 2;
const SEQ: usize = 32;

fn preset(name: &str) -> DormouseConfig {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../configs/{name}.toml"));
    let c =
        load_config(path.to_str().expect("utf-8 path")).unwrap_or_else(|e| panic!("{name}: {e}"));
    validate(&c).unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    c
}

/// The preset with its WIDTHS shrunk — the same rule `preset_exec`'s fixture
/// follows, because a 9.2M-parameter forward is not a CPU test. Everything
/// that decides what executes is the preset's own value.
fn fixture(c: &DormouseConfig) -> DormouseConfig {
    DormouseConfig {
        d_model: 64,
        n_heads: 4,
        head_dim: 16,
        d_ffn: 128,
        rank: 16,
        max_seq_len: SEQ,
        engram_rows: 512,
        ..c.clone()
    }
}

/// PURE CE on any preset: every auxiliary objective at weight 0, nothing else
/// touched. This is the trio's recipe (one objective for all three arms, so no
/// arm pays for a second forward) and the archived controls' recipe.
fn pure_ce(mut c: DormouseConfig) -> DormouseConfig {
    c.jepa_weight = 0.0;
    c.dspark_weight = 0.0;
    c.aux_fb_weight = 0.0;
    c.mor_bce_weight = 0.0;
    c
}

/// `small` as the ARCHIVED control ran it (docs/reviews/byteflow-ab-2026-10-02.md):
/// pure CE AND the memory arm off — at the PRESET's own widths and table
/// budget, because the point is that the tables are still counted when the arm
/// is off. Widths shrink only where a test asks for it, via [`fixture`].
fn archived_control_recipe(c: &DormouseConfig) -> DormouseConfig {
    let mut c = pure_ce(c.clone());
    c.use_engram = false;
    c
}

fn bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 37 + 11) as u8).collect()
}

fn input_ids(b: &[u8], dev: &Device) -> Tensor<2, Int> {
    let v: Vec<i64> = b.iter().map(|&x| x as i64).collect();
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ]), dev)
}

/// Next-byte targets, the train loop's shift (per row, last entry wrapped).
fn targets(b: &[u8], dev: &Device) -> Tensor<2, Int> {
    let mut v = Vec::with_capacity(BATCH * SEQ);
    for r in 0..BATCH {
        let row = &b[r * SEQ..(r + 1) * SEQ];
        v.extend(row.iter().skip(1).map(|&x| x as i64));
        v.push(row[0] as i64);
    }
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ]), dev)
}

fn hashed_ids(b: &[u8], dev: &Device) -> Tensor<3, Int> {
    let mut v = Vec::with_capacity(BATCH * SEQ * 3);
    for r in 0..BATCH {
        let row = &b[r * SEQ..(r + 1) * SEQ];
        for p in 0..SEQ {
            let e = p + 1;
            for n in [2usize, 3, 4] {
                v.push((fnv_hash(&row[e.saturating_sub(n)..e]) as u32) as i32 as i64);
            }
        }
    }
    Tensor::from_data(TensorData::new(v, [BATCH, SEQ, 3]), dev)
}

/// One parameter, by path and size, with the verdict the backward gave it.
#[derive(Debug)]
struct Row {
    path: String,
    n: usize,
    live: bool,
}

/// Which subtree a path belongs to, for the summary. The buckets are the ones a
/// reader of a run header needs: what trains, and which of the always-built
/// subtrees does not.
fn bucket(path: &str) -> &'static str {
    if path.starts_with("loop_block.plain_attn") {
        "dense attention (the control's arm)"
    } else if path.starts_with("loop_block.shared_attn") {
        "KDA arm (built whether or not it runs)"
    } else if path.starts_with("loop_block.engram") || path.starts_with("loop_block.mem_dense") {
        "hashed memory subtree + mem_dense"
    } else if path.starts_with("aux.") {
        "aux heads (built whether or not they run)"
    } else if path.starts_with("loop_block.mor_router") {
        "MoR router"
    } else if path == "norm.weight" {
        "the model's final RMSNorm (see §4 of the review)"
    } else if path.starts_with("loop_block.expert_ffns") || path.starts_with("loop_block.out_proj") {
        "FFN experts + readout"
    } else {
        "everything else (embedding, head, controller, norms)"
    }
}

/// One forward + backward over `cfg`, and the per-parameter verdict.
fn measure(cfg: &DormouseConfig) -> Vec<Row> {
    let dev = device();
    let model = DormouseModel::new(cfg, &dev);
    let b = bytes(BATCH * SEQ);
    let (_logits, rec, ..) = model.forward_with_hidden::<B>(
        input_ids(&b, &dev),
        Some(hashed_ids(&b, &dev)),
        None,
        Some(targets(&b, &dev)),
        None,
    );
    assert!(
        rec.clone().into_scalar::<f32>().is_finite(),
        "the loss must be finite before its gradient set means anything"
    );
    let grads = rec.backward();
    let mut rows = Vec::new();
    let mut v = Walker {
        grads: &grads,
        stack: Vec::new(),
        out: Vec::new(),
    };
    model.visit(&mut v);
    rows.append(&mut v.out);
    rows
}

struct Walker<'g> {
    grads: &'g burn::tensor::Gradients,
    stack: Vec<String>,
    out: Vec<Row>,
}

impl ModuleVisitor for Walker<'_> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.stack.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.stack.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let n: usize = param.val().clone().dims().iter().product();
        self.out.push(Row {
            path: self.stack.join("."),
            n,
            live: param.grad(self.grads).is_some(),
        });
    }
}

/// `(total, live, dead)` and the per-bucket breakdown, printed.
fn report(label: &str, rows: &[Row]) -> (usize, usize) {
    let total: usize = rows.iter().map(|r| r.n).sum();
    let live: usize = rows.iter().filter(|r| r.live).map(|r| r.n).sum();
    println!(
        "\n{label}: {} params counted, {live} TRAIN, {} never train ({:.1}% of the count)",
        rows.iter().map(|r| r.n).sum::<usize>(),
        total - live,
        (total - live) as f64 * 100.0 / total as f64
    );
    let mut buckets: Vec<(&'static str, usize, usize)> = Vec::new();
    for r in rows {
        let b = bucket(&r.path);
        match buckets.iter_mut().find(|(name, ..)| *name == b) {
            Some(e) => {
                e.1 += r.n;
                e.2 += usize::from(!r.live);
            }
            None => buckets.push((b, r.n, usize::from(!r.live))),
        }
    }
    for (name, n, dead) in &buckets {
        println!("  {n:>9}  dead {dead:>9}  {name}");
    }
    for r in rows.iter().filter(|r| !r.live) {
        println!("    DEAD {:>9}  {}", r.n, r.path);
    }
    (total, live)
}

/// 1. The control is a control: every mechanism off, one dense expert, depth
///    one, no auxiliary objective. Asserted on the RESOLVED config, so a schema
///    default that moved cannot quietly switch an arm back on.
#[test]
fn plain9m_declares_no_mechanism() {
    let c = preset("plain9m");
    let off: [(&str, bool); 12] = [
        ("use_kda", c.use_kda),
        ("use_engram", c.use_engram),
        ("use_tsct", c.use_tsct),
        ("use_byteflow", c.use_byteflow),
        ("use_msa", c.use_msa),
        ("use_gr", c.use_gr),
        ("use_attnres", c.use_attnres),
        ("use_mhc", c.use_mhc),
        ("use_mor", c.use_mor),
        ("use_situ", c.use_situ),
        ("bf16", c.bf16),
        ("act_quant.is_some()", c.act_quant.is_some()),
    ];
    let on: Vec<&str> = off.iter().filter(|(_, v)| *v).map(|(n, _)| *n).collect();
    assert!(on.is_empty(), "the CONTROL has mechanisms on: {on:?}");
    assert!(c.use_plain_attn, "the arm under test is the dense attention");
    assert_eq!(c.max_iter, 1, "depth > 1 IS the weight-shared loop under test");
    assert_eq!(c.n_experts, 1, "one expert = a dense FFN, not a mixture");
    assert_eq!(c.moe_topk, 0, "moe_topk > 0 routes; the control blends densely");
    assert_eq!(c.jepa_weight, 0.0);
    assert_eq!(c.dspark_weight, 0.0);
    assert_eq!(c.aux_fb_weight, 0.0);
    assert_eq!(c.mor_bce_weight, 0.0);
    // The Engram tables are built whatever the arm says, so the control zeroes
    // them rather than shipping a count dominated by rows nobody reads.
    assert_eq!(c.engram_rows, 1, "the control must not ship memory rows");
}

/// 2. THE MEASUREMENT, on the control, at fixture width. Three claims:
///
///    * the dense attention arm's four projections all train (the `8fa5d4c`
///      gate — the control's attention must not be the thing that trains
///      nothing);
///    * the hashed-memory subtree and `mem_dense` take NO gradient, which is
///      the whole reason this file exists;
///    * the memory subtree is not the only dead weight: the always-built KDA
///      arm and the always-built aux heads are dead in this config too, and a
///      control whose headline count is 34% rows is still a count with 15%
///      more dead KDA in it.
#[test]
fn the_control_trains_its_attention_and_nothing_else_silently() {
    let rows = measure(&fixture(&preset("plain9m")));
    let (total, live) = report("plain9m (fixture width)", &rows);

    let attn: Vec<&Row> = rows
        .iter()
        .filter(|r| r.path.starts_with("loop_block.plain_attn"))
        .collect();
    assert_eq!(
        attn.len(),
        8,
        "four projections x (weight, bias): {} parameter tensors, so the arm is not what this \
         test thinks",
        attn.len()
    );
    for r in &attn {
        assert!(
            r.live,
            "{} received no gradient: the control's attention is not training, and the whole \
             trio is calibrated against this arm",
            r.path
        );
    }

    for want in [
        "loop_block.engram.key_projs",
        "loop_block.engram.value_proj",
        "loop_block.engram.memory",
        "loop_block.mem_dense",
        "loop_block.shared_attn.gdn2.q_proj",
        "aux.jepa_pred",
        "loop_block.mor_router",
    ] {
        assert!(
            rows.iter().any(|r| r.path.starts_with(want) && !r.live),
            "expected {want} to receive NO gradient in a config with the arm off"
        );
    }
    for want in ["loop_block.expert_ffns", "loop_block.out_proj", "embedding", "lm_head"] {
        assert!(
            rows.iter().any(|r| r.path.starts_with(want) && r.live),
            "expected {want} to train"
        );
    }
    // The share is the finding, so it is pinned rather than printed: a control
    // whose printed parameter count is mostly dead weight cannot be compared to
    // another net by that number.
    assert!(
        (total - live) * 100 / total > 25,
        "only {:.1}% of plain9m's {total} counted parameters are dead - this file's claim is that \
         the always-built subtrees dominate, and that has stopped being true",
        (total - live) as f64 * 100.0 / total as f64
    );
}

/// 3. THE SAME MEASUREMENT ON THE ARCHIVED CONTROL. `small` with the memory
///    arm off and pure CE is what every control in this repo's A/B archive
///    actually trained (docs/reviews/byteflow-ab-2026-10-02.md), and its
///    headline "9 197 454 params" counts 3 145 728 n-gram rows that run never
///    read. This asserts the shape of that gap so a future edit cannot quietly
///    close it and leave a stale number in a document.
#[test]
fn the_archived_controls_count_was_never_its_training_count() {
    let rows = measure(&fixture(&archived_control_recipe(&preset("small"))));
    let (total, live) = report("small --no-engram, pure CE (the ARCHIVED control)", &rows);
    assert!(
        (total - live) * 100 / total > 35,
        "the archived control's dead share fell to {:.1}% of {total} - the retraction in \
         docs/reviews/trio-prep-2026-10-04.md §3 would need re-measuring",
        (total - live) as f64 * 100.0 / total as f64
    );
}

/// 4. THE FULL-WIDTH NUMBERS, the ones a trio table would quote.
///    `#[ignore]`d by policy like the other full-width CPU gates
///    (`preset_exec::the_wide_presets_cost_what_they_say_they_cost`).
#[test]
#[ignore = "slow: full-width instantiation on the CPU backend takes minutes"]
fn the_full_width_counts() {
    let plain = preset("plain9m");
    let (pt, pl) = report("plain9m FULL WIDTH", &measure(&plain));
    let small = pure_ce(preset("small"));
    let (st, sl) = report("small, pure CE, FULL WIDTH (the TRIO's dormouse arm)", &measure(&small));
    let archived = archived_control_recipe(&preset("small"));
    let (at, al) = report(
        "small --no-engram, pure CE, FULL WIDTH (the ARCHIVED control)",
        &measure(&archived),
    );
    println!(
        "\nTRIO TABLE, full width, both dormouse-path arms with the memory arm off and pure CE:\n  \
         plain9m   counted {pt:>10}   live {pl:>10}\n  small     counted {st:>10}   live {sl:>10}\n  \
         byteflow  counted  15923968   live  15923968 (its loop builds nothing it does not read)"
    );
    // THE MATCHING RULE, asserted: the control and the arm are the same size in
    // parameters that TRAIN. Not in parameters counted - the two nets carry
    // different always-built subtrees (plain9m's dead KDA arm, small's dead
    // Engram subtree), so equal counted totals would mean shrinking the FFN to
    // ~813 to pay for weight neither net trains.
    assert!(
        (pl as f64 - sl as f64).abs() / (sl as f64) < 0.02,
        "plain9m trains {pl} parameters against the archived control's {sl} - more than 2% apart, \
         so the trio's control column is not a params-matched comparison"
    );
    // ... and the counted-vs-live gap is asserted too, because it is the number
    // a run header prints and the one a reader would quote by mistake.
    assert!(
        (pt - pl) * 100 / pt > 30,
        "plain9m's counted {pt} is within 30% of its live {pl} - the dead-weight finding this \
         file exists for has changed shape and the review needs re-measuring"
    );
    // The retraction this lane exists for: the ARCHIVED control trained
    // {al} live parameters while its document quoted {at} counted ones, so
    // byteflow's 15_923_968 was compared against a network ~3.5x smaller in
    // parameters that learn, not the 1.73x gap that document states.
    assert!(
        al * 3 < 15_923_968 / 2,
        "the archived control trains {al} of its counted {at}; byteflow's 15_923_968 against that \
         is a {:.1}x gap in parameters that learn - the retraction in the review needs \
         re-measuring",
        15_923_968.0 / al as f64
    );
}
