//! ONE optimizer policy: the gate that the live install and the declaration
//! agree.
//!
//! The two used to disagree and the run never said so. `dormouse_core::routing`
//! declared the policy from `ParamId`s and was exercised only by core's own
//! tests; the trainer built the groups from a second, string-based copy of
//! the rules, and its validator checked that second copy against itself. The
//! copies differed on one reachable leaf: with `--set use_tsct=false` a dense
//! expert's `[d,d]` weight went to Muon+ in the trainer while the declaration
//! put it on the base optimizer - the fp32 Newton-Schulz case AGENTS.md §2.3
//! records as solved. Every log line agreed with the strings, so the only way
//! to see it was to compare the two policies on the same model.
//!
//! That comparison is this file. It runs the INSTALLED `ParamGroup`s - the
//! objects the optimizer really holds, obtained through the same call
//! `build_optim` makes - against the declaration, for every parameter of the
//! model, over the arms that make the policy non-trivial (spectral and dense
//! experts, with and without `--factors-fallback`, with and without head-wise
//! Q/K). A marker set, a `contains` predicate, or a hand-written id list
//! reinstalled in `build_optim` turns every case red on the first parameter
//! it moves.

use dormouse_core::routing::{routing, Group};
use dormouse_core::{DormouseConfig, DormouseModel};
use dormouse_train::{optimizer_groups, param_paths, TrainCfg};

fn device() -> burn::tensor::Device {
    burn::tensor::Device::flex().autodiff()
}

/// Small enough to build in a second on the CPU backend, wide enough that
/// every arm exists: 2 experts (so the per-expert factor count is not
/// confusable with a constant), a rank, and an Engram table.
fn cfg() -> DormouseConfig {
    DormouseConfig {
        d_model: 32,
        n_heads: 2,
        head_dim: 16,
        d_ffn: 64,
        max_iter: 2,
        n_experts: 2,
        rank: 8,
        engram_rows: 64,
        ..DormouseConfig::default()
    }
}

fn train_cfg(factors_fallback: bool, qk_heads: Option<usize>) -> TrainCfg {
    TrainCfg {
        factors_fallback,
        qk_heads,
        ..TrainCfg::default()
    }
}

/// THE gate. For every parameter of the live model, the group it is
/// actually installed into must be the group the declaration names.
#[test]
fn installed_groups_are_the_declared_groups() {
    for use_tsct in [true, false] {
        for factors_fallback in [false, true] {
            for qk_heads in [None, Some(2)] {
                let c = DormouseConfig { use_tsct, ..cfg() };
                let model = DormouseModel::new(&c, &device());
                let tcfg = train_cfg(factors_fallback, qk_heads);
                let g = optimizer_groups(&model, &tcfg).unwrap_or_else(|e| {
                    panic!("use_tsct={use_tsct} ff={factors_fallback} qk={qk_heads:?}: {e}")
                });
                let r = routing(&model, factors_fallback);
                let what = format!("use_tsct={use_tsct} ff={factors_fallback} qk={qk_heads:?}");

                let mut checked = 0;
                for (path, id, _rank) in param_paths(&model) {
                    let declared = r.group_of_id(&id);
                    let mut installed = Vec::new();
                    if g.muon.matches(&id, Some(&path)) {
                        installed.push(Group::Muon);
                    }
                    if g.qk.as_ref().is_some_and(|q| q.matches(&id, Some(&path))) {
                        installed.push(Group::QkHeadWise);
                    }
                    if g.table.matches(&id, Some(&path)) {
                        installed.push(Group::Table);
                    }
                    match declared {
                        // Declared but deliberately not installed: `Rest` is
                        // the base optimizer (it installs no group), and Q/K
                        // without `qk_heads` train there too - the documented
                        // meaning of the flag being unset.
                        Some(Group::Rest) => assert!(
                            installed.is_empty(),
                            "{what} {path}: the base optimizer owns it; an installed group claims it",
                        ),
                        Some(Group::QkHeadWise) if qk_heads.is_none() => {
                            assert!(installed.is_empty(), "{what} {path}: head-wise Q/K declared but the group was installed")
                        }
                        Some(declared) => assert_eq!(
                            installed,
                            vec![declared],
                            "{what} {path}: the live install and the declared policy disagree",
                        ),
                        None => panic!("{what} {path}: no declared group (routing::check should have failed)"),
                    }
                    checked += 1;
                }
                assert!(
                    checked > 10,
                    "{what}: only {checked} params - the fixture is not the model this test thinks"
                );
            }
        }
    }
}

/// The bug itself, named: a dense expert's `[d,d]` weight is a real map, and
/// fp32 Newton-Schulz on one is the ~40 s/step case the policy exists to
/// avoid. It is the leaf the marker copy got wrong, so it gets its own
/// assertion rather than riding inside the loop above.
#[test]
fn a_dense_expert_weight_never_reaches_muon() {
    let c = DormouseConfig {
        use_tsct: false,
        ..cfg()
    };
    let model = DormouseModel::new(&c, &device());
    let tcfg = train_cfg(false, Some(2));
    let g = optimizer_groups(&model, &tcfg).expect("declared");
    let r = routing(&model, false);

    let mut dense_weights = 0;
    for (path, id, rank) in param_paths(&model) {
        if !path.contains("expert_ffns") || !path.ends_with("Dense.weight") {
            continue;
        }
        dense_weights += 1;
        assert_eq!(rank, 2, "{path}: a dense expert weight is a [d,d] map");
        assert!(
            !g.muon.matches(&id, Some(&path)),
            "{path}: a [d,d] weight reached the Muon+ group - fp32 Newton-Schulz on it is the \
             ~40 s/step case; escape is to fix the group, not to raise --no-tsct's speed budget"
        );
        assert_eq!(
            r.group_of_id(&id),
            Some(Group::Rest),
            "{path}: declared on the base optimizer"
        );
    }
    assert_eq!(dense_weights, 2 * c.n_experts, "gate_up + down, per expert");
    // And the spectral arm DOES put its factors there, or the test above is
    // passing for the wrong reason (an empty Muon+ group).
    let spectral = DormouseModel::new(&cfg(), &device());
    let sg = optimizer_groups(&spectral, &train_cfg(false, Some(2))).expect("declared");
    let factors = param_paths(&spectral)
        .into_iter()
        .filter(|(p, id, _)| {
            p.contains("expert_ffns") && p.ends_with("Tsct.u") && sg.muon.matches(id, Some(p))
        })
        .count();
    assert!(
        factors > 0,
        "the spectral expert factors must be on Muon+ - otherwise the dense gate is vacuous"
    );
}

/// Loudness: the install is verified against the live tree, and the
/// verification is reachable from outside the crate so it cannot rot into a
/// private convention. `optimizer_groups` is the call the trainer makes, and
/// it returns `Err` rather than installing a group set that does not match
/// the declaration - there is no "build it anyway" path.
#[test]
fn the_group_build_is_loud_and_reachable() {
    let model = DormouseModel::new(&cfg(), &device());
    let g = optimizer_groups(&model, &train_cfg(false, Some(2)))
        .expect("the declared install is valid");
    let r = routing(&model, false);
    // Every parameter the declaration does NOT call `Rest` is claimed by
    // exactly one installed group, and no `Rest` parameter is claimed at all.
    // That is the coverage the gate enforces; a group set that forgot an arm
    // fails here, at the call, naming the parameter.
    let mut owned = 0;
    let mut rest = 0;
    for (path, id, _) in param_paths(&model) {
        let claimed = g.muon.matches(&id, Some(&path))
            || g.qk.as_ref().is_some_and(|q| q.matches(&id, Some(&path)))
            || g.table.matches(&id, Some(&path));
        match r.group_of_id(&id) {
            Some(Group::Rest) => {
                assert!(
                    !claimed,
                    "{path}: a base-optimizer parameter is claimed by an installed group"
                );
                rest += 1;
            }
            Some(_) => {
                assert!(claimed, "{path}: a routed parameter is in no installed group - it would train on the base optimizer by accident");
                owned += 1;
            }
            None => panic!("{path}: no declared group"),
        }
    }
    assert!(owned > 0 && rest > 0, "the fixture split {owned} owned / {rest} rest - one side is empty, so the coverage check is half-blind");
}

/// One walker, one counts type. Both used to be declared twice — a second
/// `GroupCounts` in `optim.rs` and a second `ModuleVisitor` doing the identical
/// stack-then-join over the same tree — which is the twin
/// `docs/reviews/dedup-optimizer-2026-10-01.md` is about.
///
/// Asserting the TYPE IDENTITY is the gate, because a re-declaration
/// compiles silently and every test above would keep passing while the two
/// copies drift: that is precisely how the path-marker policy survived a whole
/// refactor with the tests green and the runtime wrong. Re-adding either
/// declaration in `optim.rs` turns this red.
#[test]
fn the_declaration_is_declared_once() {
    // The same type under both names, so a second `struct GroupCounts` in
    // `optim.rs` is a compile error at the `use`, not a silent shadow.
    let _same_counts: fn(&dormouse_core::routing::GroupCounts) -> &dormouse_train::GroupCounts =
        |c| c;
    // The same function under both names: two `ModuleVisitor` impls are two
    // walks, and the second one is free to disagree about which parameters
    // exist. Equality is decided on the walk, not on the signature.
    let model = DormouseModel::new(&cfg(), &device());
    let from_train = param_paths(&model);
    let from_core = dormouse_core::routing::param_paths(&model);
    assert_eq!(
        from_train.len(),
        from_core.len(),
        "two module walkers disagree on how many parameters the model has"
    );
    for ((p1, i1, r1), (p2, i2, r2)) in from_train.iter().zip(&from_core) {
        assert_eq!(
            (p1, i1, r1),
            (p2, i2, r2),
            "two module walkers disagree on a parameter"
        );
    }
    // And the walk still finds the arms, so an empty result cannot make the
    // equality above vacuously true.
    assert!(
        from_core.len() > 10,
        "the walk returned {} params",
        from_core.len()
    );
    assert!(
        from_core.iter().any(|(_, _, r)| *r == 2) && from_core.iter().any(|(_, _, r)| *r == 1),
        "no rank-1 parameter: the fixture is not the model this test thinks"
    );
}
