"""ARRSAC + eight-point: the real pose method 2 (``eight_point.rs``).

Upstream's ``PoseEightPoint`` is not OpenCV's ``findEssentialMat`` — it is
the ``arrsac`` crate (v0.11, from Raguram et al., "A Comparative Analysis
of RANSAC Techniques Leading to Adaptive Real-Time Random Sample
Consensus") driving the ``eight-point`` estimator from rust-cv. Ported
here file by file:

* :class:`Arrsac` — ``arrsac-0.11.0/src/lib.rs``: SPRT-filtered adaptive
  sample consensus. The hyperparameters (256 initialization hypotheses, 4
  initialization blocks, 64 max candidates / estimations / block size,
  likelihood-ratio threshold 1e3) and the algorithm structure (initial
  hypothesis generation to estimate ε/δ, per-block scoring,
  ``populate_hypotheses_sprt``, halving candidate retention) are
  upstream's verbatim; so are its documented quirks — the initialization
  deviates from the paper to estimate ε/δ from best/worst generated
  hypotheses, and the retention shift is the crate author's corrected
  placement.
* :class:`EightPointEstimator` — ``eight-point/src/lib.rs``: the epipolar
  constraint stacked 8×9, smallest-eigenvector essential matrix, SVD
  decomposition into 4 possible poses (U·W·Vᵀ / U·Wᵀ·Vᵀ with positive
  determinants, translation = U's third column).
* The ``CameraToCamera`` residual (``cv-core/src/pose.rs:249-292``): a
  two-view triangulation-based Sampson-ish error — least-squares point
  per view, cheirality-weighted half-space residual.

``estimate_pose_arrsac`` mirrors upstream ``eight_point.rs``: undistorted
unit-ray matches, thresholds [1e-10, 1e-8, 1e-6] tried in order, the
rotation extracted from the winning pose's isometry.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# eight-point estimator (eight-point/src/lib.rs + cv-pinhole essential.rs)
# ---------------------------------------------------------------------------


class _Pose:
    """``CameraToCamera``: an SE(3) isometry (rotation + translation)."""

    __slots__ = ("rotation", "translation")

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float64)  # (3, 3)
        self.translation = np.asarray(translation, dtype=np.float64)  # (3,)

    @staticmethod
    def identity() -> "_Pose":
        return _Pose(np.eye(3), np.zeros(3))

    def transform(self, point: np.ndarray) -> np.ndarray:
        """Camera point (x, y, z) through the pose (homogeneous)."""
        return self.rotation @ point + self.translation


class EightPointEstimator:
    """The eight-point algorithm as an ARRSAC estimator.

    ``estimate(sample)`` returns the 4 possible poses from the essential
    matrix (``possible_unscaled_poses``); ``residual`` is the triangulated
    two-view cheirality error.
    """

    MIN_SAMPLES = 8

    def __init__(self, epsilon: float = 1e-12, iterations: int = 1000) -> None:
        self.epsilon = epsilon
        self.iterations = iterations

    def _essential_from_matches(self, rays_a: np.ndarray,
                                rays_b: np.ndarray) -> np.ndarray | None:
        """``encode_epipolar_equation`` + smallest eigenvector
        (``eight-point/src/lib.rs:9-60``)."""
        if len(rays_a) < 8:
            return None
        constraint = np.zeros((8, 9))
        for i in range(8):
            ap = rays_a[i] / rays_a[i][2]
            bp = rays_b[i] / rays_b[i][2]
            row = np.concatenate([ap[j] * bp for j in range(3)])
            constraint[i] = row
        eet = constraint.T @ constraint
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(eet)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(eigenvalues).all():
            return None
        smallest = int(np.argmin(eigenvalues))
        # nalgebra's Matrix3::from_iterator fills column-major; the numpy
        # equivalent of that reshape is Fortran order.
        return eigenvectors[:, smallest].reshape(3, 3, order="F")

    def _possible_poses(self, essential: np.ndarray) -> list[_Pose]:
        """``possible_unscaled_poses`` (``cv-pinhole/src/essential.rs:
        114-165, 217-231``): SVD, positive-determinant U/Vᵀ, the two
        rotations U·W·Vᵀ and U·Wᵀ·Vᵀ, translation ±U's third column."""
        try:
            u, s, vt = np.linalg.svd(essential)
        except np.linalg.LinAlgError:
            return []
        if not (np.isfinite(u).all() and np.isfinite(vt).all()):
            return []
        if np.linalg.det(u) < 0.0:
            u[:, 2] *= -1.0
        if np.linalg.det(vt) < 0.0:
            vt[2, :] *= -1.0
        w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rot_a = u @ w @ vt
        rot_b = u @ w.T @ vt
        t = u[:, 2].copy()
        return [
            _Pose(rot_a, t), _Pose(rot_b, t),
            _Pose(rot_a, -t), _Pose(rot_b, -t),
        ]

    def estimate(self, indices: list[int],
                 rays_a: np.ndarray, rays_b: np.ndarray) -> list[_Pose]:
        """Estimator::estimate — the models one minimal sample yields."""
        essential = self._essential_from_matches(
            rays_a[indices], rays_b[indices]
        )
        if essential is None:
            return []
        return self._possible_poses(essential)

    def residual(self, pose: _Pose, a: np.ndarray, b: np.ndarray) -> float:
        """``Model::residual`` for ``CameraToCamera``
        (``cv-core/src/pose.rs:249-292``): triangulate the 3-D point in
        both views by null-space of the (pose − b·bᵀ·pose) design, then
        the cheirality-weighted residual; 2.0 on failure."""
        return float(self.residuals_batch(pose, a[None, :], b[None, :])[0])

    def residuals_batch(self, pose: _Pose, rays_a: np.ndarray,
                        rays_b: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`residual` over point arrays (the per-point
        4×4 eigen-decomposition runs as one stacked ``eigh``)."""
        rays_a = np.atleast_2d(rays_a)
        rays_b = np.atleast_2d(rays_b)
        n = len(rays_a)
        out = np.full(n, 2.0)

        design = np.zeros((n, 4, 4))
        for rotation, translation, bearings in (
            (np.eye(3), np.zeros(3), rays_a),
            (pose.rotation, pose.translation, rays_b),
        ):
            pose34 = np.column_stack([rotation, translation])          # (3,4)
            # term_i = pose34 - b_i b_iᵀ pose34 ; design += termᵀ term
            proj = bearings @ pose34                                    # b_iᵀ P (n,4)
            outer = np.einsum("ni,nj->nij", bearings, proj)             # b_i (b_iᵀP) (n,3,4)
            term = pose34[None, :, :] - outer
            design += np.einsum("nij,nik->njk", term, term)

        finite = np.isfinite(design).all(axis=(1, 2))
        if not finite.any():
            return out
        try:
            eigvals, eigvecs = np.linalg.eigh(design[finite])
        except np.linalg.LinAlgError:
            return out
        ix = np.argmin(np.abs(eigvals), axis=1)
        points_h = eigvecs[np.arange(len(ix)), :, ix]                   # (m,4)
        good = np.isfinite(points_h).all(axis=1) & (points_h[:, 3] != 0.0)
        points = points_h[good, :3] / points_h[good, 3:4]
        ok = np.isfinite(points).all(axis=1)
        points = points[ok]

        a_ok = rays_a[finite][good][ok]
        b_ok = rays_b[finite][good][ok]
        na = np.linalg.norm(points, axis=1, keepdims=True)
        na[na == 0.0] = 1.0
        bearing_a = points / na
        moved = points @ pose.rotation.T + pose.translation
        nb = np.linalg.norm(moved, axis=1, keepdims=True)
        nb[nb == 0.0] = 1.0
        bearing_b = moved / nb

        res = 0.5 * ((1.0 - np.einsum("ij,ij->i", a_ok, bearing_a))
                     + (1.0 - np.einsum("ij,ij->i", b_ok, bearing_b)))
        out[np.flatnonzero(finite)[good][ok]] = res
        return out


# ---------------------------------------------------------------------------
# ARRSAC (arrsac-0.11.0/src/lib.rs)
# ---------------------------------------------------------------------------


class Arrsac:
    """Adaptive Real-Time Random Sample Consensus.

    Structure and hyperparameters are the crate's verbatim; the embedded
    SPRT (sequential probability ratio test) rejects unlikely hypotheses
    after few residuals instead of scoring every point.
    """

    def __init__(self, inlier_threshold: float, rng: np.random.Generator
                 ) -> None:
        self.initialization_hypotheses = 256
        self.initialization_blocks = 4
        self.max_candidate_hypotheses = 64
        self.estimations_per_block = 64
        self.block_size = 64
        self.likelihood_ratio_threshold = 1e3
        self.inlier_threshold = float(inlier_threshold)
        self._rng = rng
        self._random_samples: list[int] = []

    # -- helpers (lib.rs:247-330) ------------------------------------------

    def _count_inliers(self, model, data, upto: int) -> int:
        rays_a, rays_b = data
        resids = self._estimator.residuals_batch(model, rays_a[:upto], rays_b[:upto])
        return int((resids < self.inlier_threshold).sum())

    def _inliers(self, model, data, upto: int) -> list[int]:
        rays_a, rays_b = data
        resids = self._estimator.residuals_batch(model, rays_a[:upto], rays_b[:upto])
        return np.flatnonzero(resids < self.inlier_threshold).tolist()

    def _populate_samples(self, num: int, length: int) -> None:
        """``populate_samples``: distinct random indices in [0, length)."""
        if length < num:
            raise ValueError(
                f"cannot use arrsac without having enough samples ({length} < {num})"
            )
        chosen: set[int] = set()
        while len(chosen) < num:
            chosen.add(int(self._rng.integers(0, length)))
        self._random_samples = sorted(chosen)

    def _generate_random_hypotheses(self, data, upto: int) -> list[_Pose]:
        n = len(data[0])
        self._populate_samples(self._estimator.MIN_SAMPLES, n)
        return self._estimator.estimate(self._random_samples, *data)

    def _generate_random_hypotheses_subset(self, data,
                                           subset: list[int]) -> list[_Pose]:
        self._populate_samples(self._estimator.MIN_SAMPLES, len(subset))
        rays_a, rays_b = data
        sample_a = rays_a[[subset[i] for i in self._random_samples]]
        sample_b = rays_b[[subset[i] for i in self._random_samples]]
        return self._estimator.estimate(list(range(len(sample_a))),
                                        sample_a, sample_b)

    def _asprt(self, model, data, upto: int,
               positive_ratio: float, negative_ratio: float) -> int | None:
        """Algorithm 1 of "Randomized RANSAC with SPRT": accept/reject a
        hypothesis by its running likelihood ratio."""
        rays_a, rays_b = data
        resids = self._estimator.residuals_batch(model, rays_a[:upto], rays_b[:upto])
        likelihood = 1.0
        inliers = 0
        for r in resids:
            if r < self.inlier_threshold:
                inliers += 1
                likelihood *= positive_ratio
            else:
                likelihood *= negative_ratio
            if likelihood > self.likelihood_ratio_threshold \
                    or math.isnan(likelihood):
                return None
        return inliers if inliers >= self._estimator.MIN_SAMPLES else None

    def _populate_hypotheses_sprt(self, hypotheses, data, upto: int,
                                  delta: float,
                                  num_hypotheses: int) -> None:
        """Refine hypotheses from the best model's inliers, SPRT-filtered."""
        epsilon = hypotheses[0][1] / upto
        positive_ratio = delta / epsilon
        # Rust float division: a perfect epsilon gives inf (every outlier
        # rejects instantly), not an exception.
        denom = 1.0 - epsilon
        negative_ratio = (1.0 - delta) / denom if denom != 0.0 else float("inf")
        inliers = self._inliers(hypotheses[0][0], data, upto)
        for _ in range(num_hypotheses):
            for model in self._generate_random_hypotheses_subset(data, inliers):
                kept = self._asprt(model, data, upto,
                                   positive_ratio, negative_ratio)
                if kept is not None:
                    hypotheses.append((model, kept))

    def _initial_hypotheses(self, data):
        """The crate's rewritten Algorithm 3: estimate ε/δ from the best
        and worst of the initial random hypotheses."""
        n = len(data[0])
        initial_datapoints = min(
            self.initialization_blocks * self.block_size, n
        )
        hypotheses: list[tuple[_Pose, int]] = []
        for _ in range(self.initialization_hypotheses):
            for model in self._generate_random_hypotheses(data, n):
                hypotheses.append(
                    (model, self._count_inliers(model, data, initial_datapoints))
                )
        if not hypotheses:
            return [], 0.0

        hypotheses.sort(key=lambda h: -h[1])
        epsilon = hypotheses[0][1] / initial_datapoints
        min_inliers = hypotheses[-1][1]
        floor = self._estimator.MIN_SAMPLES
        delta = (floor if min_inliers < floor else min_inliers) \
            / initial_datapoints
        if epsilon < delta:
            hypotheses.clear()
            return hypotheses, delta

        self._populate_hypotheses_sprt(
            hypotheses, data, initial_datapoints, delta,
            self.initialization_hypotheses,
        )
        hypotheses.sort(key=lambda h: -h[1])
        keep = self.max_candidate_hypotheses >> (self.initialization_blocks - 1)
        del hypotheses[keep:]
        return hypotheses, delta

    # -- entry point (lib.rs:400-466) --------------------------------------

    def model(self, estimator: EightPointEstimator, rays_a: np.ndarray,
              rays_b: np.ndarray) -> tuple[_Pose, list[int]] | None:
        """``Consensus::model_inliers``: the winning pose + inlier indices."""
        self._estimator = estimator
        data = (np.asarray(rays_a), np.asarray(rays_b))
        n = len(data[0])
        if n < estimator.MIN_SAMPLES:
            return None

        hypotheses, delta = self._initial_hypotheses(data)
        if not hypotheses:
            return None

        block = self.initialization_blocks
        while True:
            start = block * self.block_size
            end = start + self.block_size
            if start >= n:
                break
            end_eff = min(end, n)
            resids_all = [
                estimator.residuals_batch(model, rays_a[start:end_eff],
                                          rays_b[start:end_eff])
                for model, _count in hypotheses
            ]
            for sample in range(start, end_eff):
                for k in range(len(hypotheses)):
                    if resids_all[k][sample - start] < self.inlier_threshold:
                        model, count = hypotheses[k]
                        hypotheses[k] = (model, count + 1)
            if end_eff < end:
                break  # reached the last datapoint

            hypotheses.sort(key=lambda h: -h[1])
            self._populate_hypotheses_sprt(
                hypotheses, data, end, delta, self.estimations_per_block
            )
            hypotheses.sort(key=lambda h: -h[1])
            keep = self.max_candidate_hypotheses >> block
            del hypotheses[keep:]
            if len(hypotheses) <= 1:
                break
            block += 1

        best = max(hypotheses, key=lambda h: h[1])
        inliers = self._inliers(best[0], data, n)
        return best[0], inliers


# ---------------------------------------------------------------------------
# The pose method itself (eight_point.rs)
# ---------------------------------------------------------------------------


def estimate_pose_arrsac(
    rays_a: np.ndarray, rays_b: np.ndarray,
    thresholds: tuple[float, ...] = (1e-10, 1e-8, 1e-6),
    seed: int = 0,
) -> np.ndarray | None:
    """Recover the inter-frame rotation via ARRSAC over the eight-point
    estimator (``eight_point.rs:20-61``).

    Inputs are undistorted **unit rays** (as upstream's
    ``undistort_points_for_optical_flow`` produces); the thresholds are
    tried in order until a model is found.
    """
    if len(rays_a) < EightPointEstimator.MIN_SAMPLES:
        return None

    rng = np.random.default_rng(seed)  # Xoshiro256PlusPlus::seed_from_u64(0)
    arrsac = Arrsac(thresholds[0], rng)
    estimator = EightPointEstimator()

    for threshold in thresholds:
        arrsac.inlier_threshold = float(threshold)
        result = arrsac.model(estimator, rays_a, rays_b)
        if result is not None:
            pose, _inliers = result
            return pose.rotation
    logger.warning("couldn't find model")
    return None
