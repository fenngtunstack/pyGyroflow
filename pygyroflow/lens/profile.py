"""Lens profile dataclass — camera calibration parameters and interpolation.

Ported from Gyroflow's core/lens_profile.rs. Stores intrinsics (camera matrix,
distortion coefficients), compatible resolution settings, per-focal-length
interpolation data, and metadata (brand, model, readout direction, etc.).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def _parse_dimensions(raw: Any) -> dict[str, int]:
    """Normalise a dimension value to {"w": int, "h": int}."""
    if isinstance(raw, dict):
        return {"w": int(raw.get("w", 0)), "h": int(raw.get("h", 0))}
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return {"w": int(raw[0]), "h": int(raw[1])}
    return {"w": 0, "h": 0}


def _parse_camera_matrix(raw: Any) -> list[list[float]]:
    """Parse camera_matrix from JSON (3x3 row-major list)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [[float(c) for c in row] for row in raw]
    return []


def _parse_distortion_coeffs(raw: Any) -> list[float]:
    """Parse distortion_coeffs from JSON."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(v) for v in raw]
    return []


@dataclass
class LensProfile:
    """Complete camera / lens calibration profile.

    Mirrors Gyroflow's ``LensProfile`` Rust struct.  All fields have defaults
    so an empty profile can be created and populated via :meth:`from_json`.
    """

    # ---- Identity -----------------------------------------------------------
    name: str = ""
    note: str = ""
    calibrated_by: str = ""
    camera_brand: str = ""
    camera_model: str = ""
    lens_model: str = ""
    camera_setting: str = ""
    identifier: str = ""

    # ---- Calibration metadata -----------------------------------------------
    calib_dimension: dict[str, int] = field(default_factory=lambda: {"w": 0, "h": 0})
    orig_dimension: dict[str, int] = field(default_factory=lambda: {"w": 0, "h": 0})
    output_dimension: dict[str, int] | None = None

    # ---- Timing / readout ---------------------------------------------------
    frame_readout_time: float | None = None
    frame_readout_direction: str = "TopToBottom"
    gyro_lpf: float | None = None
    fps: float = 0.0

    # ---- Stretch / crop -----------------------------------------------------
    input_horizontal_stretch: float = 0.0
    input_vertical_stretch: float = 0.0
    crop: float | None = None

    # ---- Intrinsics ---------------------------------------------------------
    # camera_matrix: 3x3 row-major (list of 3 lists of 3 floats)
    camera_matrix: list[list[float]] = field(default_factory=list)
    distortion_coeffs: list[float] = field(default_factory=list)
    radial_distortion_limit: float | None = None
    rms_error: float = 0.0

    # ---- Distortion model ---------------------------------------------------
    distortion_model: str | None = None
    digital_lens: str | None = None
    digital_lens_params: list[float] | None = None

    # ---- Multi-resolution / compatible settings -----------------------------
    compatible_settings: list[dict] = field(default_factory=list)
    sync_settings: dict | None = None

    # ---- Per-focal-length interpolation (raw JSON + parsed) -----------------
    interpolations_raw: dict[str, dict] | None = None
    _parsed_interpolations: dict[int, LensProfile] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ---- Additional metadata ------------------------------------------------
    focal_length: float | None = None
    crop_factor: float | None = None
    global_shutter: bool = False
    official: bool = False
    asymmetrical: bool = False
    num_images: int = 0
    calibrator_version: str = ""
    date: str = ""

    # ---- Runtime-only fields ------------------------------------------------
    path_to_file: str = ""
    optimal_fov: float | None = None
    is_copy: bool = False
    rating: float | None = None
    checksum: str | None = None

    # --------------------------------------------------------------------- #
    #  Construction helpers                                                   #
    # --------------------------------------------------------------------- #

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LensProfile:
        """Create a LensProfile from a parsed JSON dict.

        Unknown keys are silently ignored.  Nested structures
        (``calib_dimension``, ``camera_matrix``, etc.) are normalised.
        """
        p = cls()
        _str_fields = (
            "name", "note", "calibrated_by", "camera_brand", "camera_model",
            "lens_model", "camera_setting", "identifier",
            "frame_readout_direction", "distortion_model", "digital_lens",
            "calibrator_version", "date", "path_to_file",
        )
        for f in _str_fields:
            if f in data and data[f] is not None:
                setattr(p, f, str(data[f]))

        _float_fields = (
            "fps", "input_horizontal_stretch", "input_vertical_stretch",
            "rms_error", "optimal_fov", "crop_factor", "focal_length",
            "gyro_lpf", "frame_readout_time",
        )
        for f in _float_fields:
            if f in data and data[f] is not None:
                setattr(p, f, float(data[f]))

        _bool_fields = ("global_shutter", "official", "asymmetrical", "is_copy")
        for f in _bool_fields:
            if f in data and data[f] is not None:
                setattr(p, f, bool(data[f]))

        _int_fields = ("num_images",)
        for f in _int_fields:
            if f in data and data[f] is not None:
                setattr(p, f, int(data[f]))

        # Dimensions
        for dim_key in ("calib_dimension", "orig_dimension"):
            if dim_key in data:
                setattr(p, dim_key, _parse_dimensions(data[dim_key]))

        if "output_dimension" in data and data["output_dimension"] is not None:
            p.output_dimension = _parse_dimensions(data["output_dimension"])

        # Crop
        if "crop" in data and data["crop"] is not None:
            p.crop = float(data["crop"])

        # Intrinsics — may be nested under "fisheye_params" or flat
        fisheye = data.get("fisheye_params", {})
        if isinstance(fisheye, dict):
            p.camera_matrix = _parse_camera_matrix(
                fisheye.get("camera_matrix", data.get("camera_matrix"))
            )
            p.distortion_coeffs = _parse_distortion_coeffs(
                fisheye.get("distortion_coeffs", data.get("distortion_coeffs"))
            )
            p.radial_distortion_limit = fisheye.get(
                "radial_distortion_limit",
                data.get("radial_distortion_limit"),
            )
            p.rms_error = float(fisheye.get("RMS_error", fisheye.get("rms_error", data.get("rms_error", 0.0))))
        else:
            p.camera_matrix = _parse_camera_matrix(data.get("camera_matrix"))
            p.distortion_coeffs = _parse_distortion_coeffs(data.get("distortion_coeffs"))

        # Digital lens params
        dlp = data.get("digital_lens_params")
        if dlp is not None and isinstance(dlp, list):
            p.digital_lens_params = [float(v) for v in dlp]

        # Compatible settings (list of dicts)
        cs = data.get("compatible_settings")
        if isinstance(cs, list):
            p.compatible_settings = [dict(s) if isinstance(s, dict) else s for s in cs]

        # Sync settings
        ss = data.get("sync_settings")
        if isinstance(ss, dict):
            p.sync_settings = ss

        # Interpolations (raw JSON object)
        interp = data.get("interpolations")
        if isinstance(interp, dict):
            p.interpolations_raw = interp

        # Rating
        if "rating" in data and data["rating"] is not None:
            p.rating = float(data["rating"])

        # Checksum
        if "checksum" in data and data["checksum"] is not None:
            p.checksum = str(data["checksum"])

        return p

    # --------------------------------------------------------------------- #
    #  Camera matrix / distortion accessors                                   #
    # --------------------------------------------------------------------- #

    def get_camera_matrix(self, size: tuple[int, int] | None = None,
                          invert_h: bool = False) -> np.ndarray:
        """Return the 3x3 camera intrinsic matrix as a numpy array.

        Parameters
        ----------
        size:
            ``(width, height)`` of the video frame.  Used as a fallback when
            no calibration data is present (produces a default matrix).
        invert_h:
            When *True* and the profile is asymmetric, flip the vertical
            principal-point coordinate.
        """
        if len(self.camera_matrix) == 3:
            mat = np.array(self.camera_matrix, dtype=np.float64)
            if not self.asymmetrical:
                mat[0, 2] = self.calib_dimension["w"] / 2.0
                mat[1, 2] = self.calib_dimension["h"] / 2.0
            elif invert_h:
                mat[1, 2] = self.calib_dimension["h"] - mat[1, 2]
            if self.crop is not None and self.crop > 0:
                mat[0, 0] /= self.crop
                mat[1, 1] /= self.crop
            return mat

        # Default camera matrix from frame size
        w, h = size if size else (self.calib_dimension["w"], self.calib_dimension["h"])
        mat = np.eye(3, dtype=np.float64)
        mat[0, 0] = w * 0.8
        mat[1, 1] = w * 0.8
        mat[0, 2] = w / 2.0
        mat[1, 2] = h / 2.0
        return mat

    def get_camera_matrix_at_timestamp(self, timestamp_ms: float) -> np.ndarray:
        """Get interpolated 3x3 camera matrix for a given timestamp.

        Delegates to :meth:`get_interpolated_profile_at` for interpolation
        logic, then extracts the matrix.
        """
        interp = self.get_interpolated_profile_at(timestamp_ms)
        return interp.get_camera_matrix()

    def get_distortion_coeffs(self) -> list[float]:
        """Return distortion coefficients, zero-padded to length 12."""
        coeffs = list(self.distortion_coeffs)
        while len(coeffs) < 12:
            coeffs.append(0.0)
        return coeffs[:12]

    def get_distortion_coeffs_at_timestamp(self, timestamp_ms: float) -> list[float]:
        """Get interpolated distortion coefficients for a given timestamp."""
        interp = self.get_interpolated_profile_at(timestamp_ms)
        return interp.get_distortion_coeffs()

    # --------------------------------------------------------------------- #
    #  Interpolation                                                          #
    # --------------------------------------------------------------------- #

    def resolve_interpolations(self, database: LensProfileDatabase | None = None,
                               focal_length: float | None = None) -> None:
        """Parse raw interpolation JSON into the internal sorted map.

        After loading, ``interpolations_raw`` contains a dict mapping
        focal-length keys (as strings) to parameter overrides or profile
        identifiers.  This method resolves references against *database*
        and builds ``_parsed_interpolations`` for fast lookup.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._parsed_interpolations:
            return

        if self.interpolations_raw is None:
            return

        parsed: dict[int, LensProfile] = {}
        for key_str, overrides in self.interpolations_raw.items():
            try:
                key_val = float(key_str)
            except (ValueError, TypeError):
                continue
            # Key is focal length * 1_000_000 (matches Rust's i64 scheme)
            key_int = round(key_val * 1_000_000)

            # Start from a referenced profile or clone self
            new_profile = self._clone_shallow()
            if database is not None and "identifier" in overrides:
                ref = database.get_by_id(overrides["identifier"])
                if ref is not None:
                    new_profile = ref._clone_shallow()
            new_profile.interpolations_raw = None

            # Override camera matrix
            cm = overrides.get("camera_matrix")
            if isinstance(cm, list):
                new_profile.camera_matrix = _parse_camera_matrix(cm)

            # Override distortion coefficients
            dc = overrides.get("distortion_coeffs")
            if isinstance(dc, list):
                new_profile.distortion_coeffs = _parse_distortion_coeffs(dc)

            # Override focal length
            fl = overrides.get("focal_length")
            if fl is not None:
                new_profile.focal_length = float(fl)

            parsed[key_int] = new_profile

        self._parsed_interpolations = parsed

    def get_interpolated_profile_at(self, value: float) -> LensProfile:
        """Return a LensProfile interpolated at *value* (e.g. focal length).

        If the parsed-interpolations map is empty, returns a shallow clone of
        ``self``.  Exact matches are returned directly; values between two
        keys are linearly interpolated.
        """
        if not self._parsed_interpolations:
            return self._clone_shallow()

        key = round(value * 1_000_000)

        # Exact match
        if key in self._parsed_interpolations:
            return self._parsed_interpolations[key]._clone_shallow()

        sorted_keys = sorted(self._parsed_interpolations.keys())
        first, last = sorted_keys[0], sorted_keys[-1]
        lookup = max(first + 1, min(last - 1, key))

        # Find surrounding keys
        p1_key: int | None = None
        p2_key: int | None = None
        for k in sorted_keys:
            if k <= lookup:
                p1_key = k
            if k >= lookup and p2_key is None:
                p2_key = k

        if p1_key is None or p2_key is None:
            return self._clone_shallow()

        if p1_key == p2_key:
            return self._parsed_interpolations[p1_key]._clone_shallow()

        l1 = self._parsed_interpolations[p1_key]
        l2 = self._parsed_interpolations[p2_key]
        time_delta = p2_key - p1_key
        if time_delta == 0:
            return l1._clone_shallow()

        fract = (key - p1_key) / time_delta
        cpy = self._clone_shallow()

        # Interpolate camera matrix (rows 0,1 elements 0,1,2)
        if len(cpy.camera_matrix) >= 2 and len(l1.camera_matrix) >= 2 and len(l2.camera_matrix) >= 2:
            for r in (0, 1):
                for c in (0, 1, 2):
                    v1 = l1.camera_matrix[r][c] if c < len(l1.camera_matrix[r]) else 0.0
                    v2 = l2.camera_matrix[r][c] if c < len(l2.camera_matrix[r]) else 0.0
                    cpy.camera_matrix[r][c] = v1 * (1.0 - fract) + v2 * fract

        # Interpolate distortion coefficients
        if len(l1.distortion_coeffs) == len(l2.distortion_coeffs):
            cpy.distortion_coeffs = [
                a * (1.0 - fract) + b * fract
                for a, b in zip(l1.distortion_coeffs, l2.distortion_coeffs)
            ]

        # Interpolate crop
        c1 = l1.crop if l1.crop is not None else 1.0
        c2 = l2.crop if l2.crop is not None else 1.0
        cpy.crop = c1 * (1.0 - fract) + c2 * fract

        # Interpolate focal length
        if l1.focal_length is not None and l2.focal_length is not None:
            cpy.focal_length = l1.focal_length * (1.0 - fract) + l2.focal_length * fract

        # Interpolate dimensions
        cpy.calib_dimension = {
            "w": round(l1.calib_dimension["w"] * (1.0 - fract) + l2.calib_dimension["w"] * fract),
            "h": round(l1.calib_dimension["h"] * (1.0 - fract) + l2.calib_dimension["h"] * fract),
        }

        # Interpolate stretch factors
        cpy.input_horizontal_stretch = (
            l1.input_horizontal_stretch * (1.0 - fract) + l2.input_horizontal_stretch * fract
        )
        cpy.input_vertical_stretch = (
            l1.input_vertical_stretch * (1.0 - fract) + l2.input_vertical_stretch * fract
        )

        return cpy

    # --------------------------------------------------------------------- #
    #  Compatible settings — derived profiles                                 #
    # --------------------------------------------------------------------- #

    def get_all_matching_profiles(self) -> list[LensProfile]:
        """Expand this profile plus all compatible settings into a list.

        For each compatible setting, a derived copy is created with scaled
        dimensions, camera matrix, and optional parameter overrides
        (fps, crop, digital_lens, etc.).
        """
        result = [self._clone_shallow()]

        for setting in self.compatible_settings:
            if not isinstance(setting, dict):
                continue
            cpy = self._clone_shallow()
            cpy.compatible_settings = []

            w_val = setting.get("width")
            h_val = setting.get("height")
            if w_val is not None and h_val is not None:
                new_w = int(w_val)
                new_h = int(h_val)
                if new_w > 0 and new_h > 0:
                    ratiow = new_w / cpy.calib_dimension["w"]
                    ratioh = new_h / cpy.calib_dimension["h"]
                    dl = setting.get("digital_lens", "")
                    if dl in ("gopro_superview", "gopro6_superview"):
                        ratiow /= 1.33333333333
                    elif dl == "gopro_hyperview":
                        ratiow /= 1.55555555555

                    def _scale(val: int, ratio: float) -> int:
                        scaled = round(val * ratio)
                        return scaled - (scaled % 2)  # ensure even
                    cpy.calib_dimension = {
                        "w": _scale(cpy.calib_dimension["w"], ratiow),
                        "h": _scale(cpy.calib_dimension["h"], ratioh),
                    }
                    cpy.orig_dimension = {
                        "w": _scale(cpy.orig_dimension["w"], ratiow),
                        "h": _scale(cpy.orig_dimension["h"], ratioh),
                    }

                    if len(cpy.camera_matrix) > 1 and abs(ratiow - ratioh) < 0.001:
                        cpy.camera_matrix[0][0] *= ratiow
                        cpy.camera_matrix[0][2] *= ratiow
                        cpy.camera_matrix[1][1] *= ratioh
                        cpy.camera_matrix[1][2] *= ratioh

                    if cpy.output_dimension is not None:
                        cpy.output_dimension = {
                            "w": _scale(cpy.output_dimension["w"], ratiow),
                            "h": _scale(cpy.output_dimension["h"], ratioh),
                        }

            # Apply optional overrides from compatible setting
            if "frame_readout_time" in setting:
                cpy.frame_readout_time = float(setting["frame_readout_time"])
            if "fps" in setting:
                cpy.fps = float(setting["fps"])
            if "crop" in setting:
                cpy.crop = float(setting["crop"])
            if "interpolations" in setting:
                interp = setting["interpolations"]
                if isinstance(interp, dict):
                    cpy.interpolations_raw = interp
                    cpy._parsed_interpolations = {}
            if "digital_lens" in setting:
                cpy.digital_lens = str(setting["digital_lens"])
            if "focal_length" in setting:
                cpy.focal_length = float(setting["focal_length"])
            if "crop_factor" in setting:
                cpy.crop_factor = float(setting["crop_factor"])
            if "input_horizontal_stretch" in setting:
                cpy.input_horizontal_stretch = float(setting["input_horizontal_stretch"])
            if "input_vertical_stretch" in setting:
                cpy.input_vertical_stretch = float(setting["input_vertical_stretch"])
            if "lens_model" in setting:
                cpy.lens_model = str(setting["lens_model"])
            if "identifier" in setting:
                cpy.identifier = str(setting["identifier"])

            dc = setting.get("distortion_coeffs")
            if isinstance(dc, list):
                for i, v in enumerate(dc):
                    if i < len(cpy.distortion_coeffs):
                        cpy.distortion_coeffs[i] = float(v)

            odim = setting.get("output_dimension")
            if isinstance(odim, dict) and "w" in odim and "h" in odim:
                cpy.output_dimension = {"w": int(odim["w"]), "h": int(odim["h"])}

            cpy.is_copy = True
            result.append(cpy)

        return result

    # --------------------------------------------------------------------- #
    #  Display helpers                                                        #
    # --------------------------------------------------------------------- #

    def get_aspect_ratio(self) -> str:
        """Return a human-readable aspect ratio string (e.g. ``"16:9"``)."""
        w = self.calib_dimension["w"]
        h = self.calib_dimension["h"]
        if w == 0 or h == 0:
            return ""
        known = [
            (1.0,       "1:1"),
            (3.0 / 2.0, "3:2"), (2.0 / 3.0, "2:3"),
            (4.0 / 3.0, "4:3"), (3.0 / 4.0, "3:4"),
            (8.0 / 7.0, "8:7"), (7.0 / 8.0, "7:8"),
            (16.0 / 9.0, "16:9"), (9.0 / 16.0, "9:16"),
        ]
        ratio = w / h
        best_diff, best_str = min(
            ((abs(r - ratio), s) for r, s in known),
            key=lambda x: x[0],
        )
        if best_diff < 0.05:
            return best_str
        g = _gcd(w, h)
        r1, r2 = w // g, h // g
        if r1 >= 20 or r2 >= 20:
            return f"{ratio:.2}:1"
        return f"{r1}:{r2}"

    def get_size_str(self) -> str:
        """Return a video size label (``"4k"``, ``"1080p"``, etc.)."""
        w = self.calib_dimension["w"]
        h = self.calib_dimension["h"]
        if w >= 8000:   return "8k"
        if w >= 6000:   return "6k"
        if w >= 5000:   return "5k"
        if w >  4000:   return "C4k"
        if w >= 3840:   return "4k"
        if w >= 2700:   return "2.7k"
        if w >= 2500:   return "2.5k"
        if w >= 2000:   return "2k"
        if w == 1920 and h == 1440: return "1440p"
        if w >= 1920:   return "1080p"
        if w >= 1280:   return "720p"
        if w >= 640:    return "480p"
        return ""

    def get_display_name(self) -> str:
        """User-friendly display name (brand model size ratio lens fps)."""
        w = self.calib_dimension["w"]
        h = self.calib_dimension["h"]
        if w == 0 or h == 0:
            return "---"

        all_sizes: set[int] = set()
        all_fps: set[int] = set()
        all_sizes.add(w * 10000 + h)
        if self.fps > 0.0:
            all_fps.add(round(self.fps * 10000))

        for s in self.compatible_settings:
            if isinstance(s, dict):
                sw, sh = s.get("width"), s.get("height")
                if sw and sh:
                    all_sizes.add(int(sw) * 10000 + int(sh))
                sfps = s.get("fps")
                if sfps is not None:
                    all_fps.add(round(float(sfps) * 10000))

        include_size = len(all_sizes) <= 1
        include_fps = len(all_fps) <= 1 or (
            len(all_fps) == 2 and min(all_fps) >= 2_000_000
        )

        parts = [p for p in (self.camera_brand, self.camera_model) if p]
        name = " ".join(parts)

        if include_size:
            name += " " + self.get_size_str()
        name += " " + self.get_aspect_ratio()

        detail_parts = [p for p in (self.lens_model, self.camera_setting, self.note) if p]
        if detail_parts:
            name += " " + _cleanup_name(" ".join(detail_parts))

        if include_size:
            name += f" {w}x{h}"
        if include_fps and self.fps > 0.0:
            name += f" {self.fps:.2f}fps"

        return name

    # --------------------------------------------------------------------- #
    #  Swap (portrait / landscape)                                            #
    # --------------------------------------------------------------------- #

    def swapped(self) -> LensProfile:
        """Return a copy with width/height swapped (portrait <-> landscape).

        Also swaps the principal-point coordinates in the camera matrix.
        """
        ret = self._clone_shallow()

        # Swap dimensions
        ret.calib_dimension = {"w": self.calib_dimension["h"], "h": self.calib_dimension["w"]}
        ret.orig_dimension = {"w": self.orig_dimension["h"], "h": self.orig_dimension["w"]}
        ret.input_horizontal_stretch, ret.input_vertical_stretch = (
            self.input_vertical_stretch, self.input_horizontal_stretch,
        )

        if ret.output_dimension is not None:
            ret.output_dimension = {"w": self.output_dimension["h"], "h": self.output_dimension["w"]}

        # Swap camera matrix principal point
        if len(ret.camera_matrix) == 3:
            m = ret.camera_matrix
            m[0][0], m[1][1] = m[1][1], m[0][0]
            m[0][2], m[1][2] = m[1][2], m[0][2]

        # Swap compatible settings
        for s in ret.compatible_settings:
            if isinstance(s, dict) and "width" in s and "height" in s:
                s["width"], s["height"] = s["height"], s["width"]

        # Swap parsed interpolations
        swapped_interp = {}
        for k, v in ret._parsed_interpolations.items():
            swapped_interp[k] = v.swapped()
        ret._parsed_interpolations = swapped_interp

        return ret

    # --------------------------------------------------------------------- #
    #  Internal                                                               #
    # --------------------------------------------------------------------- #

    def _clone_shallow(self) -> LensProfile:
        """Return a deep copy (needed because dataclass copy_deep is safe here)."""
        return copy.deepcopy(self)


def _cleanup_name(name: str) -> str:
    """Remove common resolution/aspect-ratio substrings and underscores."""
    for token in (
        ".json", "4_3", "4:3", "4by3", "16:9", "169", "16_9", "16*9",
        "16/9", "16by9", "2_7K", "2,7K", "2.7K", "4K", "5K",
    ):
        name = name.replace(token, "")
    return name.replace("_", " ").strip()


# Forward reference for type hint — resolved at runtime via TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pygyroflow.lens.database import LensProfileDatabase
