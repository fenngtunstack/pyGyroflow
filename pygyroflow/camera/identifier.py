"""Camera identifier — brand and model detection from telemetry metadata.

Port of Gyroflow's src/core/camera_identifier.rs.
Extracts camera brand, model, lens info from video telemetry metadata
and generates a unique identifier string for auto-loading lens profiles.
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CameraIdentifier:
    """Camera identification info for auto-loading lens profiles.

    Extracted from video file telemetry metadata. The ``identifier`` field
    is a normalized string used to match against known lens profiles.

    Brand-specific extraction logic:
    - GoPro: EIS status, FOV mode (Wide/Super/Hyper/Linear/Narrow/Max)
    - Sony: focal length, lens name, lens distortion hash
    - Insta360: FOV type, FlowState status
    - Others: generic — focal length, lens name, resolution format
    """

    brand: str = ""
    model: str = ""
    lens_model: str = ""
    lens_info: str = ""
    focal_length: float | None = None
    camera_setting: str = ""
    fps: int = 0  # stored as fps * 1000 (integer)
    video_width: int = 0
    video_height: int = 0
    additional: str = ""
    identifier: str = ""

    # ── Construction from telemetry metadata ──────────────────────────

    @classmethod
    def from_metadata(
        cls,
        *,
        brand: str = "",
        model: str = "",
        video_width: int = 0,
        video_height: int = 0,
        fps: float = 0.0,
        samples: list[dict[str, Any]] | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> CameraIdentifier:
        """Build a CameraIdentifier from parsed telemetry metadata.

        This is the Python equivalent of the Rust ``from_telemetry_parser``.
        It accepts pre-parsed dicts rather than a telemetry_parser Input
        object, since the native bridge (if available) returns dicts.

        Args:
            brand: Camera brand string (e.g. "GoPro", "Sony").
            model: Camera model string (e.g. "HERO11 Black").
            video_width: Video resolution width.
            video_height: Video resolution height.
            fps: Video frame rate (float, will be stored as fps*1000 int).
            samples: List of sample dicts with tag_map entries. Each sample
                is ``{"tag_map": {group_id: {tag_id: value, ...}, ...}}``.
            raw_metadata: Flat metadata dict from the native bridge with
                keys like "lens_info", "focal_length", "lens_type", etc.
        """
        fps_int = round(fps * 1000.0)
        samples = samples or []
        raw_metadata = raw_metadata or {}

        obj = cls(
            brand=brand,
            model=model,
            video_width=video_width,
            video_height=video_height,
            fps=fps_int,
        )

        # Brand-specific defaults
        if obj.brand.lower() in ("runcam", "caddx"):
            obj.lens_info = "wide"

        # Strip brand prefix from model (GoPro HERO11 Black -> HERO11 Black)
        if obj.brand and obj.model:
            obj.model = obj.model.replace(obj.brand, "").strip()

        # Dispatch to brand-specific extraction
        brand_lower = obj.brand.lower()
        if brand_lower == "gopro":
            _extract_gopro(obj, samples)
        elif brand_lower == "sony":
            _extract_sony(obj, samples)
        elif brand_lower == "insta360":
            _extract_insta360(obj, samples)
        else:
            _extract_generic(obj, samples, raw_metadata, obj.brand)

        obj.identifier = obj._build_identifier()
        log.debug("CameraIdentifier: %s", obj)
        return obj

    # ── Identifier generation ─────────────────────────────────────────

    def _build_identifier(self) -> str:
        """Build the normalized identifier string.

        Format: ``brand-model-lens_model-lens_info-WxH@fps-additional``
        All lowercased, spaces and duplicate dashes removed.

        RED cameras omit fps (no sensor crop per framerate).
        """
        if not self.brand or not self.model or not self.lens_info:
            return ""

        # RED cameras don't crop sensor by framerate
        fps_val = 0 if self.brand in ("RED", "RED RAW") else self.fps

        raw_id = (
            f"{self.brand}-{self.model}-{self.lens_model}"
            f"-{self.lens_info}-{self.video_width}x{self.video_height}"
            f"@{fps_val}-{self.additional}"
        )
        # Normalize: strip spaces, collapse dashes, trim edges, lowercase
        result = raw_id.replace(" ", "")
        result = result.replace("--", "-")
        result = result.replace("--", "-")
        result = result.strip("- ")
        return result.lower()

    def get_identifier_for_autoload(self) -> str:
        """Return identifier for auto-loading lens profiles.

        Maps GoPro 12/13 and Hero11 Black Mini to GoPro 11 profiles
        (they share the same lens configuration).
        """
        result = self.identifier
        result = result.replace("hero12", "hero11")
        result = result.replace("hero13", "hero11")
        result = result.replace("hero11blackmini", "hero11black")
        return result

    # ── Serialization ─────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> CameraIdentifier:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ── Camera identifiers DB ─────────────────────────────────────────

    @staticmethod
    def load_identifiers_db(path: str) -> dict[str, Any]:
        """Load a Gyroflow camera_identifiers.json database file.

        The database maps identifier strings to lens profile metadata.
        Returns an empty dict if the file doesn't exist or is invalid.
        """
        import os

        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load camera identifiers DB %s: %s", path, exc)
            return {}

    def find_in_db(self, db: dict[str, Any]) -> dict[str, Any] | None:
        """Look up this identifier in a loaded camera_identifiers database.

        Tries the autoload identifier first, then the raw identifier.
        """
        autoload_id = self.get_identifier_for_autoload()
        if autoload_id in db:
            return db[autoload_id]
        if self.identifier in db:
            return db[self.identifier]
        return None

    # ── Display ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CameraIdentifier(brand={self.brand!r}, model={self.model!r}, "
            f"lens_info={self.lens_info!r}, id={self.identifier!r})"
        )


# ── Brand-specific extraction helpers ─────────────────────────────────


def _get_tag(
    samples: list[dict[str, Any]],
    group_id: str,
    tag_id: str | int,
) -> Any | None:
    """Retrieve a tag value from sample tag_maps.

    Args:
        samples: List of sample dicts with ``tag_map``.
        group_id: Group to look in (e.g. "Default", "Lens").
        tag_id: Tag key — string name or int code.
    """
    for sample in samples:
        tag_map = sample.get("tag_map")
        if not tag_map:
            continue
        group = tag_map.get(group_id)
        if not group:
            continue
        val = group.get(tag_id)
        if val is not None:
            return val
    return None


def _extract_gopro(obj: CameraIdentifier, samples: list[dict[str, Any]]) -> None:
    """Extract GoPro-specific fields: EIS status, FOV mode."""
    # EIS (tag 0x45495341 = "EISA")
    eisa = _get_tag(samples, "Default", "EISA")
    if eisa is not None:
        eisa_str = str(eisa)
        if eisa_str != "N/A":
            obj.additional = f"EIS-{eisa_str}" if eisa_str in ("Y", "N") else eisa_str

    # EIS extension (tag 0x45495345 = "EISE")
    if not obj.additional:
        eise = _get_tag(samples, "Default", "EISE")
        if eise is not None:
            obj.additional = f"EIS-{eise}"

    if obj.additional == "EIS-N":
        obj.additional = "NO-EIS"

    # Video FOV (tag 0x56464f56 = "VFOV")
    vfov = _get_tag(samples, "Default", "VFOV")
    if vfov is not None:
        fov_map = {
            "X": "Max",
            "W": "Wide",
            "S": "Super",
            "H": "Hyper",
            "L": "Linear",
            "N": "Narrow",
            "M": "Medium",
        }
        vfov_str = str(vfov)
        obj.lens_info = fov_map.get(vfov_str, vfov_str)

    # Zoom FOV (tag 0x5a464f56 = "ZFOV") — overrides Linear -> Narrow if < 80
    zfov = _get_tag(samples, "Default", "ZFOV")
    if zfov is not None:
        try:
            if obj.lens_info == "Linear" and float(zfov) < 80.0:
                obj.lens_info = "Narrow"
        except (ValueError, TypeError):
            pass

    # Projection type (tag 0x50524a54 = "PRJT")
    prjt = _get_tag(samples, "Default", "PRJT")
    if prjt is not None and str(prjt) == "GPMW":
        obj.lens_info = "Max Wide"


def _format_focal_length_mm(value: float) -> str:
    """Format focal length like Rust's {:.2} for f32.

    Rust's {:.2} uses *precision*=2, which for floats means 2 significant
    digits after the decimal for the formatted representation. For whole
    numbers like 24.0 it prints "24". We match that behaviour.
    """
    if value == int(value):
        return f"{int(value)}"
    # Use up to 2 decimal places, stripping trailing zeros
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _extract_sony(obj: CameraIdentifier, samples: list[dict[str, Any]]) -> None:
    """Extract Sony-specific fields: focal length, lens name, distortion hash."""
    # Focal length from Lens group
    fl = _get_tag(samples, "Lens", "FocalLength")
    if fl is not None:
        try:
            fl_val = float(fl)
            obj.lens_info = f"{_format_focal_length_mm(fl_val)} mm"
            obj.focal_length = fl_val
        except (ValueError, TypeError):
            pass

    # Lens display name
    dn = _get_tag(samples, "Lens", "DisplayName")
    if dn is not None:
        obj.lens_model = str(dn)

    # Lens distortion hash — only if lens_info still empty
    if not obj.lens_info:
        ld = _get_tag(samples, "LensDistortion", "Data")
        if ld is not None and isinstance(ld, dict):
            if "focal_length_nm" in ld:
                # Replicate the Rust hashing: build a JSON with specific keys
                hash_input = json.dumps({
                    "unk1": [ld["focal_length_nm"], ld.get("effective_sensor_height_nm")],
                    "unk2": ld.get("unk1"),
                    "unk3": ld.get("coeff_scale"),
                    "unk4": ld.get("coeffs"),
                }, sort_keys=True)
                crc = zlib.crc32(hash_input.encode("utf-8")) & 0xFFFFFFFF
                obj.lens_info = format(crc, "x")


def _extract_insta360(obj: CameraIdentifier, samples: list[dict[str, Any]]) -> None:
    """Extract Insta360-specific fields: FOV type, FlowState."""
    meta = _get_tag(samples, "Default", "Metadata")
    if meta is None or not isinstance(meta, dict):
        return

    fov_type = meta.get("fov_type")
    if fov_type is not None:
        obj.lens_info = str(fov_type).replace("FovType", "")

    fov = meta.get("fov")
    if fov is not None:
        try:
            fov_val = float(fov)
            if fov_val > 0.0:
                obj.lens_info = f"{obj.lens_info} {fov_val:.0f}".strip()
        except (ValueError, TypeError):
            pass

    flowstate = meta.get("is_flowstate_online")
    if flowstate is not None:
        obj.additional = "EIS" if bool(flowstate) else "NO-EIS"


def _extract_generic(
    obj: CameraIdentifier,
    samples: list[dict[str, Any]],
    raw_metadata: dict[str, Any],
    brand: str,
) -> None:
    """Generic extraction: focal length, lens name, resolution format.

    Falls through to raw_metadata dict for keys like lens_info,
    focal_length, lens_type, resolution_format_name.
    """
    lens_info_found = False

    for sample in samples:
        tag_map = sample.get("tag_map")
        if not tag_map:
            continue

        # Focal length from Lens group
        fl = _get_tag([sample], "Lens", "FocalLength")
        if fl is not None:
            try:
                fl_val = float(fl)
                obj.lens_info = f"{_format_focal_length_mm(fl_val)} mm"
                obj.focal_length = fl_val
            except (ValueError, TypeError):
                pass

        # Lens name (skip Runcam — uses generic lens_info)
        if brand != "Runcam":
            ln = _get_tag([sample], "Lens", "Name")
            if ln is not None:
                obj.lens_model = str(ln)

        # Metadata from Default group
        meta = _get_tag([sample], "Default", "Metadata")
        if meta is not None and isinstance(meta, dict):
            log.debug(
                "Camera ID Brand: %s, Model: %s, Metadata: %s",
                obj.brand, obj.model, meta,
            )

            if "lens_info" in meta:
                obj.lens_info = str(meta["lens_info"])

            if "focal_length" in meta:
                fl_meta = meta["focal_length"]
                if isinstance(fl_meta, (int, float)):
                    fl_val = float(fl_meta)
                    obj.lens_info = f"{_format_focal_length_mm(fl_val)}mm"
                    obj.focal_length = fl_val
                elif isinstance(fl_meta, str):
                    obj.lens_info = fl_meta
                    obj.focal_length = _safe_parse_float(fl_meta.replace("mm", ""))

            if "lens_type" in meta:
                obj.lens_model = str(meta["lens_type"])

            if "resolution_format_name" in meta:
                obj.camera_setting = str(meta["resolution_format_name"])

        # If we found lens_info, stop; otherwise try next sample
        if obj.lens_info:
            lens_info_found = True
            break

    # Last resort: check raw_metadata flat dict
    if not lens_info_found and raw_metadata:
        _apply_raw_metadata(obj, raw_metadata)


def _apply_raw_metadata(obj: CameraIdentifier, raw: dict[str, Any]) -> None:
    """Apply fields from a flat raw_metadata dict (native bridge output)."""
    if "lens_info" in raw:
        obj.lens_info = str(raw["lens_info"])

    if "focal_length" in raw:
        fl = raw["focal_length"]
        if isinstance(fl, (int, float)):
            fl_val = float(fl)
            obj.lens_info = f"{_format_focal_length_mm(fl_val)}mm"
            obj.focal_length = fl_val
        elif isinstance(fl, str):
            obj.lens_info = fl
            obj.focal_length = _safe_parse_float(fl.replace("mm", ""))

    if "lens_type" in raw:
        obj.lens_model = str(raw["lens_type"])

    if "resolution_format_name" in raw:
        obj.camera_setting = str(raw["resolution_format_name"])


def _safe_parse_float(s: str) -> float | None:
    """Parse a float from string, returning None on failure."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None
