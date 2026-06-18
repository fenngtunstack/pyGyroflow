"""Quaternion converter — re-integrates with a different method and applies SLERP-smoothed correction.

Port of Gyroflow's QuaternionConverter (Rust).
"""

from __future__ import annotations

from pygyroflow.types.quaternion import Quat64
from pygyroflow.types.time_types import TimeIMU, TimeQuat


class QuaternionConverter:
    """Convert quaternions from one integration method to another.

    When the user switches the integration method, this converter:
    1. Re-integrates all IMU data with the new method
    2. For each original timestamp, computes a correction quaternion
       that maps from the old orientation to the new one
    3. SLERP-smooths the corrections to avoid sudden jumps
    4. Applies the smoothed corrections to the original quaternions

    The SLERP strategy is:
    - First sample: use full correction (factor = 1.0) for fast alignment
    - Subsequent samples: use very small factor (0.005) for slow tracking

    This preserves the original timestamp structure while gradually
    morphing orientations toward the new integration result.
    """

    # Method index -> integrator class mapping
    _METHOD_MAP: dict[int, str] = {
        0: "complementary",
        1: "vqf",
        2: "simple_gyro_accel",
        3: "mahony",
        4: "madgwick",
    }

    @staticmethod
    def convert(
        method: int,
        org_quaternions: TimeQuat,
        image_orientations: TimeQuat,
        imu_data: list[TimeIMU],
        duration_ms: float,
    ) -> TimeQuat:
        """Convert original quaternions to match a different integration method.

        Args:
            method: Integration method index:
                0 = ComplementaryIntegrator
                1 = VQFIntegrator
                2 = SimpleGyroAccelIntegrator
                3 = MahonyIntegrator
                4 = MadgwickIntegrator
            org_quaternions: Original quaternion sequence (timestamp_us -> Quat64).
            image_orientations: Image orientation quaternions from video metadata
                (e.g., GoPro ISG data). May be empty.
            imu_data: Raw IMU samples.
            duration_ms: Data duration in milliseconds.

        Returns:
            Corrected quaternion sequence with the same timestamps as org_quaternions.
        """
        # Lazy import to avoid circular imports
        integrator = QuaternionConverter._get_integrator(method)
        integrated_quats = integrator.integrate(imu_data, duration_ms)

        # Build sorted lists for efficient lookup
        integrated_ts_sorted = sorted(integrated_quats.keys())
        image_ts_sorted = sorted(image_orientations.keys())

        def _find_nearest(ts_sorted: list[int], quats: TimeQuat, target_ts: int) -> Quat64:
            """Find quaternion at or immediately after target_ts."""
            # Binary search for first timestamp >= target
            lo, hi = 0, len(ts_sorted)
            while lo < hi:
                mid = (lo + hi) // 2
                if ts_sorted[mid] < target_ts:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(ts_sorted):
                return quats[ts_sorted[lo]]
            return Quat64.identity()

        identity = Quat64.identity()
        boost = 1  # First sample gets full correction
        corr_sm = Quat64.identity()
        result: TimeQuat = {}

        for org_ts in sorted(org_quaternions.keys()):
            org_quat = org_quaternions[org_ts]

            # Find new integration result at this timestamp
            n_quat = _find_nearest(integrated_ts_sorted, integrated_quats, org_ts)

            # Find image orientation at this timestamp
            io_quat = _find_nearest(image_ts_sorted, image_orientations, org_ts)

            # Correction: corr = n_quat * inverse(org_quat * inverse(io_quat))
            # Simplification: corr = n_quat * io_quat * inverse(org_quat)
            corr = n_quat * io_quat * org_quat.inverse()

            # SLERP smooth the correction
            if boost > 0:
                slerp_factor = 1.0
                boost -= 1
            else:
                slerp_factor = 0.005

            corr_sm = corr_sm.slerp(corr, slerp_factor)

            # Apply smoothed correction
            result[org_ts] = corr_sm * org_quat

        return result

    @staticmethod
    def _get_integrator(method: int):
        """Instantiate the integrator for the given method index."""
        from pygyroflow.imu_integration.complementary import ComplementaryIntegrator
        from pygyroflow.imu_integration.vqf import VQFIntegrator
        from pygyroflow.imu_integration.simple_gyro_accel import SimpleGyroAccelIntegrator
        from pygyroflow.imu_integration.mahony import MahonyIntegrator
        from pygyroflow.imu_integration.madgwick import MadgwickIntegrator

        integrators = {
            0: ComplementaryIntegrator,
            1: VQFIntegrator,
            2: SimpleGyroAccelIntegrator,
            3: MahonyIntegrator,
            4: MadgwickIntegrator,
        }
        cls = integrators.get(method, VQFIntegrator)
        return cls()
