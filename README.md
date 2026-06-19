# PyGyroFlow

Python port of Gyroflow video stabilization.

Numerical cross-check: 4/6 IMU integrators (simple_gyro, simple_gyro_accel,
mahony, madgwick) match the independent Rust port (msgyro-imu-integration) to
machine precision (max_err < 1e-14). Complementary and VQF share the golden
test suite but their Python ports diverge from the Rust algorithms and are
tracked as xfail until aligned (see tests/test_rust_golden.py).

