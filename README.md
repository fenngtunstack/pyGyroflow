# PyGyroFlow

Python port of Gyroflow video stabilization.

Numerical cross-check: 6/6 IMU integrators (simple_gyro, simple_gyro_accel,
mahony, madgwick, complementary, VQF) match the independent Rust port
(msgyro-imu-integration) to machine precision. Telemetry parsing (GoPro
GPMF, DJI protobuf) is bit-exact against the upstream telemetry-parser
reference dumps. Auto-sync offset search and lens-database auto-match are
verified on real footage (see docs/08-optimization-roadmap.md).
