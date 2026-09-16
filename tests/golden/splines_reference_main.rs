// Reference generator for upstream Gyroflow's gyro_source/splines.rs.
// The file is copied verbatim into src/splines.rs; only main() is new.
//
// Reads tests/golden/splines.json, evaluates each case with the real
// implementation and writes the results back as `expected`.
// `--check` only reports differences.
mod splines;

use serde_json::Value;
use splines::{BivariateSpline, CatmullRom};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let check = args.iter().any(|a| a == "--check");
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "tests/golden/splines.json".to_string());

    let text = std::fs::read_to_string(&path).expect("read fixture");
    let mut doc: Value = serde_json::from_str(&text).expect("parse fixture");

    let mut mismatches = 0;
    let total = {
        let cases = doc.get_mut("cases").and_then(|c| c.as_array_mut()).expect("cases");
        for case in cases.iter_mut() {
            let op = case.get("op").and_then(|v| v.as_str()).unwrap().to_string();
            let name = case.get("name").and_then(|v| v.as_str()).unwrap().to_string();
            let input = case.get("input").cloned().unwrap_or(Value::Null);

            let computed: Vec<Value> = match op.as_str() {
                "catmull" => {
                    let points = input.get("points").and_then(|v| v.as_array()).unwrap();
                    let probes = input.get("probes").and_then(|v| v.as_array()).unwrap();
                    let mut spline: CatmullRom<f64> = CatmullRom::new();
                    for pair in points {
                        let p = pair[0].as_f64().unwrap();
                        let v = pair[1].as_f64().unwrap();
                        spline.add_point(p, v);
                    }
                    probes
                        .iter()
                        .map(|t| {
                            let t = t.as_f64().unwrap();
                            match spline.interpolate(t) {
                                Some(x) => Value::from(x),
                                None => Value::Null,
                            }
                        })
                        .collect()
                }
                "bivariate" => {
                    let gw = input.get("grid_w").and_then(|v| v.as_u64()).unwrap() as usize;
                    let gh = input.get("grid_h").and_then(|v| v.as_u64()).unwrap() as usize;
                    let mesh: Vec<f64> = input
                        .get("mesh")
                        .and_then(|v| v.as_array())
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    let size_x = input.get("size_x").and_then(|v| v.as_f64()).unwrap();
                    let size_y = input.get("size_y").and_then(|v| v.as_f64()).unwrap();
                    let offset = input.get("mesh_offset").and_then(|v| v.as_u64()).unwrap() as usize;
                    let spline = BivariateSpline::new(gw, gh);
                    input
                        .get("probes")
                        .and_then(|v| v.as_array())
                        .unwrap()
                        .iter()
                        .map(|p| {
                            let x = p[0].as_f64().unwrap();
                            let y = p[1].as_f64().unwrap();
                            Value::from(spline.interpolate(size_x, size_y, &mesh, offset, x, y))
                        })
                        .collect()
                }
                other => panic!("unknown op {other}"),
            };

            if let Some(existing) = case.get("expected").and_then(|v| v.as_array()) {
                // Compared with a tolerance, not for equality: the upstream
                // code is not bit-reproducible across compilations (a rebuild
                // shifts some values by one ULP, presumably LLVM contracting
                // a multiply-add differently), so an exact comparison would
                // report a diff on every fresh build.
                for (i, (a, b)) in existing.iter().zip(computed.iter()).enumerate() {
                    let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) else {
                        if a != b {
                            eprintln!("MISMATCH {name}[{i}]: null vs value");
                            mismatches += 1;
                        }
                        continue;
                    };
                    let rel = if y == 0.0 { (x - y).abs() } else { ((x - y) / y).abs() };
                    if rel > 1e-12 {
                        eprintln!("MISMATCH {name}[{i}]: {x:.17e} vs {y:.17e} rel={rel:.3e}");
                        mismatches += 1;
                    }
                }
            }
            case["expected"] = Value::Array(computed);
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
    println!("{total} case(s) ok");
}
