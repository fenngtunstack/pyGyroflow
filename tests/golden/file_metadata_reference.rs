// Reference generator for the CBOR layout of Gyroflow's FileMetadata.
//
// Struct definitions are copied field-for-field from
//   opensource/gyroflow/src/core/gyro_source/file_metadata.rs
//   opensource/gyroflow/src/core/camera_identifier.rs
//   vendor/telemetry-parser/src/util.rs
// with three substitutions, because nalgebra is not available here:
//   Quat64                    -> [f64; 4]   (UnitQuaternion<f64> is a 4-sequence)
//   Vector3<f64>              -> [f64; 3]
//   CatmullRom<Vector3<f64>>  -> CatmullRom<[f64; 3]>
// Everything else, including every attribute and field order, is verbatim.
// `serde_json` is built with `preserve_order`, matching what a real Gyroflow
// export shows (an unsorted `lens_profile`).
//
// Reads tests/golden/cbor_file_metadata.json, encodes each case's `input` with
// ciborium, and writes the resulting bytes back into the case as `cbor`.
// `--check` only reports differences.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

type Quat64 = [f64; 4];
type TimeIMU = IMUData;
type TimeQuat = BTreeMap<i64, Quat64>;
type TimeVec = BTreeMap<i64, [f64; 3]>;

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
pub struct IMUData {
    pub timestamp_ms: f64,
    pub gyro: Option<[f64; 3]>,
    pub accl: Option<[f64; 3]>,
    pub magn: Option<[f64; 3]>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LensParams {
    pub focal_length: Option<f32>,
    pub pixel_pitch: Option<(u32, u32)>,
    pub sensor_size_px: Option<(u32, u32)>,
    pub capture_area_origin: Option<(f32, f32)>,
    pub capture_area_size: Option<(f32, f32)>,
    pub pixel_focal_length: Option<f32>,
    pub distortion_coefficients: Vec<f64>,
    pub focus_distance: Option<f32>,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct CameraIdentifier {
    pub brand: String,
    pub model: String,
    pub lens_model: String,
    pub lens_info: String,
    pub focal_length: Option<f64>,
    pub camera_setting: String,
    pub fps: usize,
    pub video_width: usize,
    pub video_height: usize,
    pub additional: String,
    pub identifier: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CatmullRom<T> {
    points: Vec<(f64, T)>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraStabData {
    pub offset: f64,
    pub sensor_size: (u32, u32),
    pub crop_area: (f32, f32, f32, f32),
    pub pixel_pitch: (u32, u32),
    pub ibis_spline: CatmullRom<[f64; 3]>,
    pub ois_spline: CatmullRom<[f64; 3]>,
}

#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ReadoutDirection {
    #[default]
    TopToBottom = 0,
    BottomToTop = 1,
    LeftToRight = 2,
    RightToLeft = 3,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FileMetadata {
    pub imu_orientation: Option<String>,
    pub raw_imu: Vec<TimeIMU>,
    pub quaternions: TimeQuat,
    pub gravity_vectors: Option<TimeVec>,
    pub image_orientations: Option<TimeQuat>,
    pub detected_source: Option<String>,
    pub frame_readout_time: Option<f64>,
    pub frame_readout_direction: ReadoutDirection,
    pub frame_rate: Option<f64>,
    pub camera_identifier: Option<CameraIdentifier>,
    pub lens_profile: Option<Value>,
    pub lens_positions: BTreeMap<i64, f64>,
    pub lens_params: BTreeMap<i64, LensParams>,
    pub digital_zoom: Option<f64>,
    pub has_accurate_timestamps: bool,
    pub additional_data: Value,
    pub per_frame_time_offsets: Vec<f64>,
    pub camera_stab_data: Vec<CameraStabData>,
    pub mesh_correction: Vec<(Vec<f64>, Vec<f32>)>,
}

fn hex(buf: &[u8]) -> String {
    buf.iter().map(|b| format!("{:02x}", b)).collect()
}

fn encode<T: Serialize>(value: &T) -> String {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf).unwrap();
    hex(&buf)
}

fn case_bytes(op: &str, input: &Value) -> String {
    match op {
        "imu_list" => encode(&serde_json::from_value::<Vec<IMUData>>(input.clone()).unwrap()),
        "lens_params_map" => {
            encode(&serde_json::from_value::<BTreeMap<i64, LensParams>>(input.clone()).unwrap())
        }
        "lens_positions" => {
            encode(&serde_json::from_value::<BTreeMap<i64, f64>>(input.clone()).unwrap())
        }
        "gravity_vectors" => {
            encode(&serde_json::from_value::<Option<TimeVec>>(input.clone()).unwrap())
        }
        "image_orientations" => {
            encode(&serde_json::from_value::<Option<TimeQuat>>(input.clone()).unwrap())
        }
        "camera_identifier" => encode(
            &serde_json::from_value::<Option<CameraIdentifier>>(input.clone()).unwrap(),
        ),
        "json_value" => encode(input),
        "per_frame_time_offsets" => {
            encode(&serde_json::from_value::<Vec<f64>>(input.clone()).unwrap())
        }
        "mesh_correction" => encode(
            &serde_json::from_value::<Vec<(Vec<f64>, Vec<f32>)>>(input.clone()).unwrap(),
        ),
        "camera_stab_data" => encode(
            &serde_json::from_value::<Vec<CameraStabData>>(input.clone()).unwrap(),
        ),
        "file_metadata" => encode(
            &serde_json::from_value::<FileMetadata>(input.clone()).unwrap(),
        ),
        other => panic!("unknown op {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let check = args.iter().any(|a| a == "--check");
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "tests/golden/cbor_file_metadata.json".to_string());

    let text = std::fs::read_to_string(&path).expect("read fixture");
    let mut doc: Value = serde_json::from_str(&text).expect("parse fixture");
    let mut mismatches = 0;
    let total = {
    let cases = doc.get_mut("cases").and_then(|c| c.as_array_mut()).expect("cases");
    for case in cases.iter_mut() {
        let op = case.get("op").and_then(|v| v.as_str()).unwrap().to_string();
        let name = case.get("name").and_then(|v| v.as_str()).unwrap().to_string();
        let input = case.get("input").cloned().unwrap_or(Value::Null);
        let computed = case_bytes(&op, &input);
        match case.get("cbor").and_then(|v| v.as_str()) {
            Some(existing) => {
                if existing != computed {
                    eprintln!("MISMATCH {name}: fixture {} bytes, ciborium {} bytes",
                              existing.len() / 2, computed.len() / 2);
                    mismatches += 1;
                }
            }
            None => {}
        }
        case["cbor"] = Value::String(computed);
    }
    cases.len()
    };
    if !check {
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n")
            .expect("write fixture");
    }
    if mismatches > 0 {
        eprintln!("{mismatches} case(s) did not match");
        std::process::exit(1);
    }
    println!("{} case(s) ok", total);
}
