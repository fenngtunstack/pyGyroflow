//! Driver + shims for `tests/golden/undistort_points.json`.
//!
//! The subject under test is `src/cpu_undistort_points.rs`, which the staging
//! step extracts **verbatim** from
//! `opensource/gyroflow/src/core/stabilization/cpu_undistort.rs` lines 649-803
//! (the `// Ported from OpenCV` comment through `undistort_points`, which is
//! the last function in that file). It is `include!`d rather than copied so
//! the file cannot drift from upstream while looking authoritative.
//!
//! The distortion models are staged verbatim too: staging copies
//! `distortion_models/` and `gyro_source/splines.rs` out of the read-only
//! upstream tree unchanged.
//!
//! What is *not* verbatim is the set of shims below. They replace the parts of
//! the crate that `undistort_points` never touches: the GPU pipeline, the
//! stabilizer, the gyro source, the real `KeyframeManager`. Each one is
//! documented with exactly what it drops and why that cannot change the
//! result. If a shim is wrong, the fixture is wrong — so the surface is kept
//! as small as the function allows and every field keep is justified by a
//! read in the copied slice.

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::fs;

// ===========================================================================
// Shims: the crate-root names the staged files expect.
// ===========================================================================

pub mod shim {
    // -- KernelParams -------------------------------------------------------
    //
    // Upstream: `stabilization/mod.rs:103-151`. Field-for-field the same, with
    // two deliberate differences:
    //
    //   * `distortion_model` / `digital_lens` are `stabilize_spirv::DistortionModel`
    //     upstream, a GPU-side enum that only exists so the struct can be
    //     uploaded to a shader. Here they are the local `DistortionModelType`
    //     with the same variant names. `undistort_points` writes them via
    //     `..Default::default()` and never reads them.
    //   * `#[repr(C, packed(4))]` is dropped, and with it the `bytemuck::Pod`
    //     impls. The attribute exists to pin the GPU upload layout; nothing
    //     here uploads. `#[repr(C)]` is kept so the field order stays explicit.
    #[derive(Default, Copy, Clone)]
    pub enum DistortionModelType {
        #[default]
        OpenCVFisheye,
        OpenCVStandard,
        Poly3,
        Poly5,
        PtLens,
        Insta360,
        Sony,
        GoProSuperview,
        GoProHyperview,
        DigitalStretch,
    }

    pub fn distort(_x: f32, _y: f32) -> (f32, f32) {
        (0.0, 0.0)
    }

    #[repr(C)]
    #[derive(Default, Copy, Clone)]
    pub struct KernelParams {
        pub width: i32,
        pub height: i32,
        pub stride: i32,
        pub output_width: i32,
        pub output_height: i32,
        pub output_stride: i32,
        pub matrix_count: i32,
        pub interpolation: i32,
        pub background_mode: i32,
        pub flags: i32,
        pub bytes_per_pixel: i32,
        pub pix_element_count: i32,
        pub background: [f32; 4],
        pub f: [f32; 2],
        pub c: [f32; 2],
        pub k: [f32; 12],
        pub fov: f32,
        pub r_limit: f32,
        pub lens_correction_amount: f32,
        pub input_vertical_stretch: f32,
        pub input_horizontal_stretch: f32,
        pub background_margin: f32,
        pub background_margin_feather: f32,
        pub canvas_scale: f32,
        pub input_rotation: f32,
        pub output_rotation: f32,
        pub translation2d: [f32; 2],
        pub translation3d: [f32; 4],
        pub source_rect: [i32; 4],
        pub output_rect: [i32; 4],
        pub digital_lens_params: [f32; 4],
        pub safe_area_rect: [f32; 4],
        pub max_pixel_value: f32,
        pub distortion_model: DistortionModelType,
        pub digital_lens: DistortionModelType,
        pub pixel_value_limit: f32,
        pub light_refraction_coefficient: f32,
        pub plane_index: i32,
        pub reserved1: f32,
        pub reserved2: f32,
        pub ewa_coeffs_p: [f32; 4],
        pub ewa_coeffs_q: [f32; 4],
    }

    // -- LensProfile --------------------------------------------------------
    //
    // Only the four fields the distortion models and `undistort_points` read
    // (`lens_profile.rs:16,31,34,43-44`). The types are upstream's.
    #[derive(Default, Clone)]
    pub struct Dimensions {
        pub w: usize,
        pub h: usize,
    }

    #[derive(Default, Clone)]
    pub struct LensProfile {
        pub lens_model: String,
        pub calib_dimension: Dimensions,
        pub input_horizontal_stretch: f64,
        pub input_vertical_stretch: f64,
    }

    // -- KeyframeManager ----------------------------------------------------
    //
    // Upstream interpolates a piecewise-linear curve. That interpolation is
    // *not* what this reference is testing — `undistort_points` only asks for
    // one value and falls back to `params.light_refraction_coefficient` when
    // there is none. So the shim answers only at the exact timestamps the
    // fixture lists and returns `None` otherwise, which is upstream's
    // "not keyframed" answer. Fixtures therefore only ever use exact
    // timestamps; there is nothing here that could interpolate differently.
    #[derive(Default, Clone)]
    pub struct KeyframeManager {
        pub light_refraction: Vec<(f64, f64)>,
    }

    impl KeyframeManager {
        pub fn value_at_video_timestamp(
            &self,
            typ: &crate::KeyframeType,
            timestamp_ms: f64,
        ) -> Option<f64> {
            match typ {
                crate::KeyframeType::LightRefractionCoeff => self
                    .light_refraction
                    .iter()
                    .find(|(ts, _)| (ts - timestamp_ms).abs() < 1e-9)
                    .map(|(_, v)| *v),
            }
        }
    }

    // -- ComputeParams ------------------------------------------------------
    //
    // The seven fields `undistort_points` reads, listed by where they are read:
    //   width/height/output_width/output_height  -> the KernelParams literal
    //   keyframes                                -> the refraction lookup
    //   light_refraction_coefficient             -> the same lookup's fallback
    //   lens.input_{horizontal,vertical}_stretch -> the per-point pre-scale
    //   distortion_model, digital_lens           -> the two model calls
    // Upstream's struct (compute_params.rs:14-...) has ~50 more; none is
    // reachable from this function.
    #[derive(Default, Clone)]
    pub struct ComputeParams {
        pub width: usize,
        pub height: usize,
        pub output_width: usize,
        pub output_height: usize,
        pub keyframes: KeyframeManager,
        pub lens: LensProfile,
        pub lens_correction_amount: f64,
        pub light_refraction_coefficient: f64,
        pub distortion_model: crate::distortion_models::DistortionModel,
        pub digital_lens: Option<crate::distortion_models::DistortionModel>,
    }

    // -- KeyframeType -------------------------------------------------------
    //
    // Upstream's is a large enum carrying colour and label data for the UI.
    // Only the one variant is reachable from `undistort_points`.
    pub enum KeyframeType {
        LightRefractionCoeff,
    }

    // -- map_coord ----------------------------------------------------------
    //
    // `util.rs::map_coord`, copied because it is three lines and the mesh
    // branch reads it unqualified.
    pub fn map_coord(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
        (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    }
}

// The staged `distortion_models/mod.rs` says `use super::KernelParams;`, and
// the model files say `crate::stabilization::KernelParams` — both resolve to
// these re-exports, which is why the staged files need no edits.
pub use shim::{ComputeParams, KernelParams, KeyframeType, LensProfile};

pub mod stabilization {
    pub use crate::shim::KernelParams;
}

pub mod lens_profile {
    pub use crate::shim::LensProfile;
}

pub mod util {
    pub use crate::shim::map_coord;
}

pub mod gyro_source {
    use nalgebra::Vector2;

    /// Verbatim from `gyro_source/sony.rs:581-587`, the one function the mesh
    /// branch of `undistort_points` reaches. Upstream has it in `sony.rs` and
    /// re-exports it from `gyro_source/mod.rs:11`; the only edit is the
    /// `splines::` path, because the staged `splines.rs` is a crate-level
    /// module here rather than `gyro_source::splines`.
    pub fn interpolate_mesh(x: f64, y: f64, size: (f64, f64), mesh: &[f64]) -> Vector2<f64> {
        let grid_spline = crate::splines::BivariateSpline::new(mesh[1] as usize, mesh[2] as usize);
        Vector2::new(
            grid_spline.interpolate(size.0, size.1, mesh, 0, x, y),
            grid_spline.interpolate(size.0, size.1, mesh, 1, x, y),
        )
    }
}

pub mod distortion_models;
pub mod splines;

// -- the subject ------------------------------------------------------------

pub mod points {
    #![allow(unused_imports)] // stand-ins for cpu_undistort.rs's own header
    use crate::distortion_models::DistortionModel;
    use crate::util::map_coord;
    use crate::{ComputeParams, KernelParams};
    use nalgebra::Matrix3;

    // Verbatim: opensource/gyroflow/src/core/stabilization/cpu_undistort.rs
    // lines 649-803. The `use` lines above stand in for the file's own header.
    include!("cpu_undistort_points.rs");
}

pub use points::undistort_points;

// ===========================================================================
// Fixture + driver
// ===========================================================================

#[derive(Deserialize, Serialize, Clone)]
// Container-level: a missing field takes `Params::default()`'s value, not the
// field type's zero. A per-field `#[serde(default)]` would make an omitted
// `width` mean 0 rather than 1920.
#[serde(default)]
struct Params {
    width: usize,
    height: usize,
    output_width: usize,
    output_height: usize,
    light_refraction_coefficient: f64,
    input_horizontal_stretch: f64,
    input_vertical_stretch: f64,
    distortion_model: String,
    digital_lens: Option<String>,
    /// `[[timestamp_ms, value], ...]`, answered only at exact timestamps.
    light_refraction_keyframes: Vec<[f64; 2]>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            width: 1920,
            height: 1080,
            output_width: 1920,
            output_height: 1080,
            light_refraction_coefficient: 1.0,
            input_horizontal_stretch: 1.0,
            input_vertical_stretch: 1.0,
            distortion_model: "opencv_fisheye".into(),
            digital_lens: None,
            light_refraction_keyframes: Vec::new(),
        }
    }
}

impl Params {
    fn build(&self) -> ComputeParams {
        let mut lens = LensProfile::default();
        lens.calib_dimension = shim::Dimensions { w: self.width, h: self.height };
        lens.input_horizontal_stretch = if self.input_horizontal_stretch == 0.0 {
            1.0
        } else {
            self.input_horizontal_stretch
        };
        lens.input_vertical_stretch = if self.input_vertical_stretch == 0.0 {
            1.0
        } else {
            self.input_vertical_stretch
        };

        let mut keyframes = shim::KeyframeManager::default();
        keyframes.light_refraction = self
            .light_refraction_keyframes
            .iter()
            .map(|kv| (kv[0], kv[1]))
            .collect();

        ComputeParams {
            width: self.width,
            height: self.height,
            output_width: self.output_width,
            output_height: self.output_height,
            keyframes,
            lens,
            lens_correction_amount: 1.0,
            light_refraction_coefficient: self.light_refraction_coefficient,
            distortion_model: distortion_models::DistortionModel::from_name(&self.distortion_model),
            digital_lens: self
                .digital_lens
                .as_ref()
                .map(|name| distortion_models::DistortionModel::from_name(name)),
        }
    }
}

#[derive(Deserialize, Serialize)]
struct Case {
    name: String,
    #[serde(default)]
    params: Params,
    distorted: Vec<[f32; 2]>,
    camera_matrix: [[f64; 3]; 3],
    distortion_coeffs: [f64; 12],
    rotation: [[f64; 3]; 3],
    #[serde(default)]
    p: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    rot_per_point: Option<Vec<[[f64; 3]; 3]>>,
    #[serde(default = "one")]
    lens_correction_amount: f64,
    #[serde(default)]
    timestamp_ms: f64,
    #[serde(default)]
    shift_per_point: Option<Vec<[f32; 5]>>,
    #[serde(default)]
    mesh: Option<Vec<f64>>,
    /// Relative bound for this case over a 1.0 floor. Defaults to 1e-5, which
    /// is ~30x the measured f32/f64 disagreement (worst 3.7e-7 across the
    /// well-conditioned cases). A case raises it only with a stated reason.
    #[serde(default = "default_tolerance")]
    tolerance: f64,
    /// `Option` rather than `f32` because the light-refraction branch can
    /// produce a NaN coordinate (upstream's `sqrt` of a negative), and JSON
    /// has no NaN — `serde_json` writes it as `null`. A `null` here means
    /// "upstream returned NaN", which is a value the port has to match.
    #[serde(default)]
    expected: Vec<[Option<f32>; 2]>,
}

fn one() -> f64 {
    1.0
}

fn default_tolerance() -> f64 {
    1e-5
}

fn matrix3(m: &[[f64; 3]; 3]) -> Matrix3<f64> {
    // nalgebra's `new` is row-major, which is how the fixture stores it.
    Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    )
}

impl Case {
    fn run(&self) -> Vec<(f32, f32)> {
        let params = self.params.build();
        // Upstream takes `&[(f32, f32)]`; JSON has no tuple, so the fixture
        // stores `[x, y]` arrays and they are converted here.
        let distorted: Vec<(f32, f32)> = self.distorted.iter().map(|p| (p[0], p[1])).collect();
        undistort_points(
            &distorted,
            matrix3(&self.camera_matrix),
            &self.distortion_coeffs,
            matrix3(&self.rotation),
            self.p.as_ref().map(matrix3),
            self.rot_per_point
                .as_ref()
                .map(|v| v.iter().map(matrix3).collect()),
            &params,
            self.lens_correction_amount,
            self.timestamp_ms,
            self.shift_per_point
                .as_ref()
                .map(|v| v.iter().map(|s| (s[0], s[1], s[2], s[3], s[4])).collect()),
            self.mesh.clone(),
        )
    }
}

#[derive(Deserialize, Serialize)]
struct Fixture {
    #[serde(default)]
    description: String,
    #[serde(default)]
    _provenance: String,
    cases: Vec<Case>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: upref <fixture.json> [--check]");
        std::process::exit(2);
    }
    let check = args.iter().any(|a| a == "--check");

    let mut fixture: Fixture =
        serde_json::from_str(&fs::read_to_string(&args[1]).expect("read fixture"))
            .expect("parse fixture");

    let mut mismatches = 0usize;
    let mut total = 0usize;

    for case in fixture.cases.iter_mut() {
        let got = case.run();

        if check && !case.expected.is_empty() {
            if got.len() != case.expected.len() {
                println!(
                    "LENGTH MISMATCH {}: {} vs {}",
                    case.name,
                    got.len(),
                    case.expected.len()
                );
                mismatches += 1;
                continue;
            }
            let mut worst = 0.0f64;
            for (index, (g, w)) in got.iter().zip(case.expected.iter()).enumerate() {
                total += 1;
                // Relative over a 1.0 floor: the sentinel is (-1e6, -1e6) and
                // a pixel coordinate is O(1000), so a pure relative test would
                // be meaninglessly tight near the origin and meaninglessly
                // loose at the sentinel.
                let got_xy = [g.0, g.1];
                let mut bad = false;
                for i in 0..2 {
                    // `null` in the fixture is upstream's NaN; NaN is not a
                    // number and is not inside any tolerance, so it is an
                    // equality test, not a distance one.
                    let Some(want) = w[i] else {
                        if !got_xy[i].is_nan() {
                            bad = true;
                        }
                        continue;
                    };
                    if got_xy[i].is_nan() {
                        bad = true;
                        continue;
                    }
                    let scale = (want as f64).abs().max(1.0);
                    let err = (got_xy[i] as f64 - want as f64).abs() / scale;
                    worst = worst.max(err);
                    if err > case.tolerance {
                        bad = true;
                    }
                }
                if bad {
                    mismatches += 1;
                    println!("MISMATCH {}[{index}]: {got_xy:?} vs {w:?}", case.name);
                }
            }
            println!("{}: {} points, worst rel {:.3e}", case.name, got.len(), worst);
            continue;
        }

        case.expected = got
            .iter()
            .map(|(x, y)| {
                [
                    if x.is_nan() { None } else { Some(*x) },
                    if y.is_nan() { None } else { Some(*y) },
                ]
            })
            .collect();
        println!("{}: {} points", case.name, case.distorted.len());
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
