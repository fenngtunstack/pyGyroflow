"""IMU integration algorithms for converting raw sensor data to quaternion orientations.

Available integrators:
- SimpleGyroIntegrator: Pure gyroscope dead reckoning (no drift correction)
- SimpleGyroAccelIntegrator: Gyro + simple accelerometer gravity correction
- MahonyIntegrator: Mahony PI-controller complementary filter
- MadgwickIntegrator: Madgwick gradient-descent filter
- ComplementaryIntegrator: Basic complementary filter
- VQFIntegrator: VQF (Versatile Quaternion Filter) — offline gyro+accel+mag fusion

Utilities:
- QuaternionConverter: Re-integrate with a different method and apply SLERP correction
"""

from pygyroflow.imu_integration.base import GyroIntegrator
from pygyroflow.imu_integration.simple_gyro import SimpleGyroIntegrator
from pygyroflow.imu_integration.simple_gyro_accel import SimpleGyroAccelIntegrator
from pygyroflow.imu_integration.mahony import MahonyIntegrator
from pygyroflow.imu_integration.madgwick import MadgwickIntegrator
from pygyroflow.imu_integration.complementary import ComplementaryIntegrator
from pygyroflow.imu_integration.vqf import VQFIntegrator
from pygyroflow.imu_integration.converter import QuaternionConverter

__all__ = [
    "GyroIntegrator",
    "SimpleGyroIntegrator",
    "SimpleGyroAccelIntegrator",
    "MahonyIntegrator",
    "MadgwickIntegrator",
    "ComplementaryIntegrator",
    "VQFIntegrator",
    "QuaternionConverter",
]
