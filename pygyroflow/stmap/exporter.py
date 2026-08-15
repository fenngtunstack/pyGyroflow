"""ST-Map exporter -- per-pixel coordinate mapping export.

Generates an ST-coordinate map (undistortion lookup table) that encodes the
full stabilization transform as a pixel-to-pixel mapping.  Each output pixel's
R channel stores the normalized source X coordinate, G channel stores the
normalized source Y coordinate.  This can be consumed by external compositing
tools (Nuke, Fusion, After Effects) to reproduce the stabilization without
running the full Gyroflow pipeline.

Port of Gyroflow's ``core/stmap.rs``.

Output format follows Gyroflow convention:
  - R = source_x / width    (normalized [0, 1])
  - G = 1.0 - source_y / height  (Y-flipped for EXR coordinate system)
  - B = 0.0  (unused)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from pygyroflow.stabilization.frame_transform import FrameTransform
from pygyroflow.stabilization.compute_params import ComputeParams
from pygyroflow.stabilization.cpu_undistort import _rotate_and_distort
from pygyroflow.stabilization.distortion_models import from_name as _dm_from_name

logger = logging.getLogger(__name__)


class STMapFormat(str, Enum):
    """Supported output formats for the ST-Map."""
    EXR = "exr"
    NPZ = "npz"       # NumPy compressed archive
    PNG16 = "png16"   # 16-bit PNG, R=X normalized, G=Y normalized


class STMapExporter:
    """ST-Map (Stabilization Map) exporter.

    Generates per-pixel coordinate mapping for the stabilization transform.
    Two map types are supported:

    - **Undistort map**: for each output pixel, stores the source pixel
      coordinate in the original (unstabilized) frame.  Used by GPU/CPU
      renderers to sample the input image.
    - **Distort map**: for each source pixel, stores where it maps to in
      the stabilized output frame.  Used by compositing tools to apply
      the stabilization in forward direction.

    Usage::

        exporter = STMapExporter(compute_params)
        exporter.export("undistort.exr", map_type="undistort")
        exporter.export("distort.exr", map_type="distort")
    """

    def __init__(
        self,
        params: ComputeParams,
        transform_factory: Optional[Callable[
            [ComputeParams, float, int], FrameTransform
        ]] = None,
    ):
        """Initialize the exporter with stabilization parameters.

        Args:
            params: ComputeParams snapshot with lens, gyro, and FOV data.
            transform_factory: Optional override for FrameTransform.at_timestamp.
                Defaults to FrameTransform.at_timestamp.
        """
        self._params = params
        self._transform_factory = (
            transform_factory or FrameTransform.at_timestamp
        )

    def compute_undistort_map(
        self,
        timestamp_ms: float = 0.0,
        frame: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """Compute the undistort coordinate map (output -> input).

        For each output pixel (x, y), computes the source coordinate in the
        original unstabilized frame using the stabilization transform.

        Args:
            timestamp_ms: Frame timestamp in milliseconds.
            frame: Frame index for FOV lookup.
            width: Override output width (defaults to params width).
            height: Override output height (defaults to params height).

        Returns:
            Float32 array of shape (H, W, 2) where [..., 0] is source X
            and [..., 1] is source Y in pixel coordinates.
        """
        params = self._params
        w = width or params.width
        h = height or params.height

        transform = self._transform_factory(params, timestamp_ms, frame)

        # Override dimensions to match requested output size
        transform.kernel_params.width = w
        transform.kernel_params.height = h
        transform.kernel_params.output_width = w
        transform.kernel_params.output_height = h

        kp = transform.kernel_params
        matrices = np.asarray(transform.matrices, dtype=np.float64)
        matrix_count = len(matrices)
        model = _dm_from_name(transform.distortion_model_name)

        # Output: (H, W, 2) -- source (x, y) per output pixel
        coord_map = np.zeros((h, w, 2), dtype=np.float32)

        # Full-image coordinate grids (vectorized).
        ys_grid, xs_grid = np.mgrid[0:h, 0:w].astype(np.float64)

        if matrix_count > 1:
            # Rolling shutter: trial-map with the center matrix to estimate
            # each pixel's source line, then evaluate with the per-pixel
            # matrix of that line — same semantics as the CPU renderer.
            horizontal_rs = (kp.flags & 16) == 16
            sx0, sy0, ok0 = _rotate_and_distort(
                xs_grid, ys_grid, matrices[matrix_count // 2], kp, model
            )
            est = np.rint(sx0 if horizontal_rs else sy0)
            fallback = xs_grid if horizontal_rs else ys_grid
            idx = np.clip(
                np.where(ok0, est, fallback), 0, matrix_count - 1
            ).astype(np.int64)
            src_x, src_y, valid = _rotate_and_distort(
                xs_grid, ys_grid, matrices[idx], kp, model
            )
        else:
            src_x, src_y, valid = _rotate_and_distort(
                xs_grid, ys_grid, matrices[0], kp, model
            )

        coord_map[:, :, 0] = np.where(valid, src_x, 0.0).astype(np.float32)
        coord_map[:, :, 1] = np.where(valid, src_y, 0.0).astype(np.float32)

        return coord_map

    def compute_distort_map(
        self,
        timestamp_ms: float = 0.0,
        frame: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """Compute the distort coordinate map (input -> output).

        For each source pixel in the original frame, computes where it maps
        to in the stabilized output.  This is the forward-direction mapping
        that compositing tools use to push pixels.

        Implementation: computes an undistort map at the input resolution,
        then the forward map is the inverse relationship.  For simplicity,
        we compute it by evaluating the transform at each input pixel.

        Args:
            timestamp_ms: Frame timestamp in milliseconds.
            frame: Frame index for FOV lookup.
            width: Override source width (defaults to params width).
            height: Override source height (defaults to params height).

        Returns:
            Float32 array of shape (H, W, 2) where [..., 0] is output X
            and [..., 1] is output Y in pixel coordinates.
        """
        params = self._params
        w = width or params.width
        h = height or params.height

        # Build a modified params for forward mapping:
        # Input dimensions are the source, output is the stabilized result
        from copy import deepcopy
        fwd_params = deepcopy(params)
        fwd_params.width = w
        fwd_params.height = h
        fwd_params.output_width = w
        fwd_params.output_height = h

        transform = self._transform_factory(fwd_params, timestamp_ms, frame)
        transform.kernel_params.width = w
        transform.kernel_params.height = h
        transform.kernel_params.output_width = w
        transform.kernel_params.output_height = h

        kp = transform.kernel_params
        matrices = transform.matrices

        # For the distort map we iterate source pixels and apply the
        # inverse of the undistort transform.  In Gyroflow's Rust code,
        # this uses undistort_points().  Here we use a simplified approach:
        # compute the stabilization matrix inverse to map source -> output.
        coord_map = np.zeros((h, w, 2), dtype=np.float32)

        # Extract the 3x3 inverse transform from matrices[0]
        # The matrix stored is (new_k @ R)^-1, so to go forward we need
        # to invert it back: new_k @ R applied to source coords gives output.
        if len(matrices) > 0:
            m = matrices[0]
            # Reconstruct the 3x3 forward matrix (inverse of what's stored)
            inv_mat = np.array([
                [m[0], m[1], m[2]],
                [m[3], m[4], m[5]],
                [m[6], m[7], m[8]],
            ], dtype=np.float64)

            try:
                fwd_mat = np.linalg.inv(inv_mat)
            except np.linalg.LinAlgError:
                fwd_mat = np.eye(3, dtype=np.float64)

            for y in range(h):
                for x in range(w):
                    # Apply forward transform: output = fwd_mat @ [x, y, 1]
                    homogeneous = np.array([x, y, 1.0], dtype=np.float64)
                    result = fwd_mat @ homogeneous

                    if result[2] > 0.0:
                        out_x = result[0] / result[2]
                        out_y = result[1] / result[2]
                        coord_map[y, x, 0] = out_x
                        coord_map[y, x, 1] = out_y

        return coord_map

    def export(
        self,
        output_path: str,
        map_type: str = "undistort",
        fmt: Optional[str] = None,
        timestamp_ms: float = 0.0,
        frame: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Export ST-map to file.

        Args:
            output_path: Destination file path.  Format is auto-detected from
                extension (.exr, .npz, .png) unless ``fmt`` is given.
            map_type: "undistort" (output->input) or "distort" (input->output).
            fmt: Force output format ("exr", "npz", "png16").  None = auto-detect.
            timestamp_ms: Frame timestamp in milliseconds.
            frame: Frame index for FOV lookup.
            width: Override width (defaults to params dimensions).
            height: Override height (defaults to params dimensions).
        """
        # Compute coordinate map
        if map_type == "undistort":
            coord_map = self.compute_undistort_map(
                timestamp_ms, frame, width, height
            )
        elif map_type == "distort":
            coord_map = self.compute_distort_map(
                timestamp_ms, frame, width, height
            )
        else:
            raise ValueError(
                f"map_type must be 'undistort' or 'distort', got '{map_type}'"
            )

        # Determine format
        output_fmt = self._resolve_format(fmt, output_path)

        # Export
        if output_fmt == STMapFormat.EXR:
            self._write_exr(coord_map, output_path)
        elif output_fmt == STMapFormat.NPZ:
            self._write_npz(coord_map, output_path)
        elif output_fmt == STMapFormat.PNG16:
            self._write_png16(coord_map, output_path)
        else:
            raise ValueError(f"Unsupported format: {output_fmt}")

    def export_bytes(
        self,
        map_type: str = "undistort",
        fmt: str = "exr",
        timestamp_ms: float = 0.0,
        frame: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bytes:
        """Export ST-map to in-memory bytes.

        Same as :meth:`export` but returns the serialized data instead of
        writing to disk.  Only supports EXR and NPZ formats.

        Args:
            map_type: "undistort" or "distort".
            fmt: "exr" or "npz".
            timestamp_ms: Frame timestamp.
            frame: Frame index.
            width: Override width.
            height: Override height.

        Returns:
            Serialized ST-map bytes.
        """
        if map_type == "undistort":
            coord_map = self.compute_undistort_map(
                timestamp_ms, frame, width, height
            )
        elif map_type == "distort":
            coord_map = self.compute_distort_map(
                timestamp_ms, frame, width, height
            )
        else:
            raise ValueError(f"Invalid map_type: {map_type}")

        output_fmt = STMapFormat(fmt)

        if output_fmt == STMapFormat.EXR:
            return self._encode_exr_bytes(coord_map)
        elif output_fmt == STMapFormat.NPZ:
            return self._encode_npz_bytes(coord_map)
        else:
            raise ValueError(f"In-memory export not supported for {output_fmt}")

    # ------------------------------------------------------------------ #
    #  Format detection                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_format(
        fmt: Optional[str], path: str
    ) -> STMapFormat:
        """Determine output format from explicit flag or file extension."""
        if fmt is not None:
            return STMapFormat(fmt.lower())

        ext = os.path.splitext(path)[1].lower()
        if ext == ".exr":
            return STMapFormat.EXR
        elif ext == ".npz":
            return STMapFormat.NPZ
        elif ext == ".png":
            return STMapFormat.PNG16
        else:
            # Default to NPZ for unknown extensions
            return STMapFormat.NPZ

    # ------------------------------------------------------------------ #
    #  EXR export (via OpenCV or manual)                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_exr_image(coord_map: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert (H, W, 2) coord map to (H, W, 3) EXR image.

        Gyroflow convention:
          R = source_x / width  (normalized X)
          G = 1.0 - source_y / height  (Y-flipped, normalized)
          B = 0.0  (unused)
        """
        h, w = coord_map.shape[:2]
        exr_img = np.zeros((h, w, 3), dtype=np.float32)

        exr_img[:, :, 0] = coord_map[:, :, 0] / float(w)
        exr_img[:, :, 1] = 1.0 - coord_map[:, :, 1] / float(h)
        # B channel stays 0

        return exr_img

    def _write_exr(
        self, coord_map: NDArray[np.float32], path: str
    ) -> None:
        """Write coordinate map as OpenEXR file."""
        exr_img = self._build_exr_image(coord_map)

        # Try OpenCV with EXR env flag
        try:
            self._write_exr_opencv(exr_img, path)
            return
        except Exception as exc:
            logger.debug("OpenCV EXR write failed: %s", exc)

        # Fallback: write as NPZ with .exr extension note in metadata
        logger.warning(
            "EXR write unavailable (OpenCV EXR disabled?). "
            "Falling back to NPZ format at: %s",
            path + ".npz",
        )
        self._write_npz(coord_map, path + ".npz")

    def _write_exr_opencv(
        self, exr_img: NDArray[np.float32], path: str
    ) -> None:
        """Write EXR via OpenCV (requires OPENCV_IO_ENABLE_OPENEXR=1)."""
        import cv2

        # Temporarily set env var if not already set
        old_val = os.environ.get("OPENCV_IO_ENABLE_OPENEXR")
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        try:
            success = cv2.imwrite(path, exr_img)
            if not success:
                raise RuntimeError("cv2.imwrite returned False for EXR")
        finally:
            if old_val is not None:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = old_val
            else:
                os.environ.pop("OPENCV_IO_ENABLE_OPENEXR", None)

        logger.info("Wrote EXR ST-map: %s (%dx%d)", path, *exr_img.shape[:2])

    def _encode_exr_bytes(
        self, coord_map: NDArray[np.float32]
    ) -> bytes:
        """Encode EXR to bytes via OpenCV's imencode."""
        import cv2

        exr_img = self._build_exr_image(coord_map)

        old_val = os.environ.get("OPENCV_IO_ENABLE_OPENEXR")
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        try:
            success, buf = cv2.imencode(".exr", exr_img)
            if not success:
                raise RuntimeError("cv2.imencode failed for EXR")
        finally:
            if old_val is not None:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = old_val
            else:
                os.environ.pop("OPENCV_IO_ENABLE_OPENEXR", None)

        return buf.tobytes()

    # ------------------------------------------------------------------ #
    #  NPZ export                                                          #
    # ------------------------------------------------------------------ #

    def _write_npz(
        self, coord_map: NDArray[np.float32], path: str
    ) -> None:
        """Write coordinate map as NumPy compressed archive.

        Saves:
          - "coords": (H, W, 2) float32 array [source_x, source_y]
          - "metadata": structured array with width, height, map_type info
        """
        h, w = coord_map.shape[:2]
        metadata = np.array(
            [(w, h)], dtype=[("width", "i4"), ("height", "i4")]
        )

        np.savez_compressed(path, coords=coord_map, metadata=metadata)
        logger.info("Wrote NPZ ST-map: %s (%dx%d)", path, w, h)

    @staticmethod
    def _encode_npz_bytes(coord_map: NDArray[np.float32]) -> bytes:
        """Encode NPZ to in-memory bytes."""
        import io

        h, w = coord_map.shape[:2]
        metadata = np.array(
            [(w, h)], dtype=[("width", "i4"), ("height", "i4")]
        )

        buf = io.BytesIO()
        np.savez_compressed(buf, coords=coord_map, metadata=metadata)
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    #  16-bit PNG export                                                   #
    # ------------------------------------------------------------------ #

    def _write_png16(
        self, coord_map: NDArray[np.float32], path: str
    ) -> None:
        """Write coordinate map as 16-bit PNG.

        Encodes normalized coordinates as 16-bit unsigned integers:
          R = uint16(source_x / width * 65535)
          G = uint16(source_y / height * 65535)
          B = 0

        Note: 16-bit PNG has limited precision (~1e-5 relative).  For
        high-precision work prefer EXR or NPZ.
        """
        import cv2

        h, w = coord_map.shape[:2]
        png_img = np.zeros((h, w, 3), dtype=np.uint16)

        # Normalize to [0, 1] then scale to 16-bit
        x_norm = np.clip(coord_map[:, :, 0] / float(w), 0.0, 1.0)
        y_norm = np.clip(coord_map[:, :, 1] / float(h), 0.0, 1.0)

        png_img[:, :, 0] = (x_norm * 65535).astype(np.uint16)
        png_img[:, :, 1] = (y_norm * 65535).astype(np.uint16)

        success = cv2.imwrite(path, png_img)
        if not success:
            raise RuntimeError(f"Failed to write PNG: {path}")

        logger.info("Wrote 16-bit PNG ST-map: %s (%dx%d)", path, w, h)
