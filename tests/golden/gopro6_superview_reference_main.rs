// Driver for tests/golden/gopro6_superview.json.
//
// `gopro6_superview_reference.rs` is a byte-for-byte copy of
// opensource/gyroflow/src/core/stabilization/distortion_models/gopro6_superview.rs
// and is compiled unchanged. It opens with
//
//     use crate::{ stabilization::KernelParams, lens_profile::LensProfile };
//
// so the two modules below stand in for upstream's real ones: only the four
// fields the model actually reads are given, with the same types (i32 for the
// sizes, usize for the calibration dimension, String for the lens model).
// That is why the copy can stay verbatim — everything upstream-specific is
// either unused or satisfied by these shims.
//
// Build:
//     mkdir -p /tmp/gp6ref/src
//     cp tests/golden/gopro6_superview_reference.Cargo.toml /tmp/gp6ref/Cargo.toml
//     cp tests/golden/gopro6_superview_reference.rs       /tmp/gp6ref/src/gopro6_superview.rs
//     cp tests/golden/gopro6_superview_reference_main.rs  /tmp/gp6ref/src/main.rs
//     cd /tmp/gp6ref && cargo build --release --offline
//     /tmp/gp6ref/target/release/gp6ref <repo>/tests/golden/gopro6_superview.json
//
// With `--check` it reports differences instead of writing.

use serde::{Deserialize, Serialize};
use std::fs;

mod gopro6_superview;
use gopro6_superview::GoPro6Superview;

// -- shims for the two upstream modules the copied file imports ------------

pub mod stabilization {
    #[derive(Default, Clone)]
    pub struct KernelParams {
        pub width: i32,
        pub height: i32,
        pub output_width: i32,
        pub output_height: i32,
    }
}

pub mod lens_profile {
    #[derive(Default, Clone)]
    pub struct CalibDimension {
        pub w: usize,
        pub h: usize,
    }

    #[derive(Default, Clone)]
    pub struct LensProfile {
        pub calib_dimension: CalibDimension,
        pub lens_model: String,
    }
}

// -- fixture ---------------------------------------------------------------

#[derive(Deserialize, Serialize)]
struct Case {
    name: String,
    width: i32,
    height: i32,
    probes: Vec<[f64; 2]>,
    #[serde(default)]
    undistorted: Vec<[f64; 2]>,
    #[serde(default)]
    distorted: Vec<[f64; 2]>,
}

#[derive(Deserialize, Serialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gp6ref <fixture.json> [--check]");
        std::process::exit(2);
    }
    let check = args.iter().any(|a| a == "--check");

    let mut fixture: Fixture =
        serde_json::from_str(&fs::read_to_string(&args[1]).expect("read fixture"))
            .expect("parse fixture");

    let model = GoPro6Superview::default();
    let mut mismatches = 0usize;
    let mut total = 0usize;

    for case in fixture.cases.iter_mut() {
        let params = stabilization::KernelParams {
            width: case.width,
            height: case.height,
            output_width: case.width,
            output_height: case.height,
        };

        // The f32 -> f64 widening is deliberate: the model is f32, the
        // fixture is JSON, so every value crosses that boundary here.
        let undistorted: Vec<[f64; 2]> = case
            .probes
            .iter()
            .map(|p| {
                let (x, y) = model
                    .undistort_point((p[0] as f32, p[1] as f32), &params)
                    .expect("undistort_point returned None");
                [x as f64, y as f64]
            })
            .collect();
        let distorted: Vec<[f64; 2]> = case
            .probes
            .iter()
            .map(|p| {
                let (x, y) = model.distort_point(p[0] as f32, p[1] as f32, 1.0, &params);
                [x as f64, y as f64]
            })
            .collect();

        if check && !case.undistorted.is_empty() {
            for (index, (got, want)) in undistorted
                .iter()
                .zip(case.undistorted.iter())
                .enumerate()
            {
                total += 1;
                // Relative, not exact: recompiling the same source shifts some
                // of these by an ULP, and f32 -> f64 is not bit-preserving.
                let close = (0..2).all(|i| {
                    let scale = want[i].abs().max(1.0);
                    (got[i] - want[i]).abs() / scale < 1e-12
                });
                if !close {
                    mismatches += 1;
                    println!(
                        "UNDISTORT MISMATCH {name}[{index}]: {got:?} vs {want:?}",
                        name = case.name,
                        got = got,
                        want = want
                    );
                }
            }
            for (index, (got, want)) in distorted
                .iter()
                .zip(case.distorted.iter())
                .enumerate()
            {
                total += 1;
                let close = (0..2).all(|i| {
                    let scale = want[i].abs().max(1.0);
                    (got[i] - want[i]).abs() / scale < 1e-12
                });
                if !close {
                    mismatches += 1;
                    println!(
                        "DISTORT MISMATCH {name}[{index}]: {got:?} vs {want:?}",
                        name = case.name,
                        got = got,
                        want = want
                    );
                }
            }
            continue;
        }

        case.undistorted = undistorted;
        case.distorted = distorted;
        println!("{}: {} probes", case.name, case.probes.len());
    }

    if check {
        println!("{}/{} values match", total - mismatches, total);
        std::process::exit(if mismatches == 0 { 0 } else { 1 });
    }

    fs::write(
        &args[1],
        serde_json::to_string_pretty(&fixture).expect("serialize") + "\n",
    )
    .expect("write fixture");
    println!("wrote {}", args[1]);
}
