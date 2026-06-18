"""Keyframe interpolation manager.

Port of Gyroflow's src/core/keyframes.rs KeyframeManager.

Maintains a two-level dict: KeyframeType -> (timestamp_us -> Keyframe).
Timestamps are in microseconds (integer), matching the Rust BTreeMap<i64, Keyframe>.

Key behaviors ported from Rust:
- Snap to nearest existing keyframe within +/-1ms (get_closest_timestamp)
- Default easing on new keyframes is EaseInOut
- Easing resolution combines adjacent keyframe easings via Easing::get()
- Custom providers take priority over stored keyframes
- timestamp_scale multiplies video timestamps before lookup
"""

from __future__ import annotations

import bisect
import json
import random
from typing import Callable, Optional

from pygyroflow.keyframes.types import Easing, Keyframe, KeyframeType

# Range for random keyframe IDs, matching Rust's fastrand::u32(1..2147483640)
_ID_RANGE = (1, 2147483639)


def _new_id() -> int:
    return random.randint(*_ID_RANGE)


class KeyframeManager:
    """Manages keyframes for all keyframe types with interpolation.

    Thread safety note: Unlike the Rust version which uses Arc<Mutex> for
    the custom provider, this Python implementation is NOT thread-safe.
    Callers must handle synchronization if needed.
    """

    def __init__(self) -> None:
        # KeyframeType -> sorted list of (timestamp_us, Keyframe)
        # We keep timestamps sorted in a list for bisect-based lookup,
        # plus a dict for O(1) access by timestamp.
        self._timestamps: dict[KeyframeType, list[int]] = {}
        self._keyframes: dict[KeyframeType, dict[int, Keyframe]] = {}

        # Custom value provider: called before looking up stored keyframes.
        # Signature: (manager, key_type, timestamp_ms) -> Optional[float]
        self._custom_provider: Optional[
            Callable[[KeyframeManager, KeyframeType, float], Optional[float]]
        ] = None

        # Timestamp scale factor (maps video time to keyframe time)
        self.timestamp_scale: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_closest_timestamp(
        self, key: KeyframeType, timestamp_us: int
    ) -> int:
        """Snap to nearest existing keyframe within +/-1ms.

        Mirrors Rust's get_closest_timestamp. If an existing keyframe is
        within 1000 microseconds (1ms), return that timestamp instead.
        """
        ts_list = self._timestamps.get(key)
        if not ts_list:
            return timestamp_us

        # Binary search for the insertion point
        idx = bisect.bisect_left(ts_list, timestamp_us)

        # Check exact match
        if idx < len(ts_list) and ts_list[idx] == timestamp_us:
            return timestamp_us

        # Check neighbor within +/-1000us
        for check_idx in (idx - 1, idx):
            if 0 <= check_idx < len(ts_list):
                if abs(ts_list[check_idx] - timestamp_us) <= 1000:
                    return ts_list[check_idx]

        return timestamp_us

    def _ensure_type(self, key: KeyframeType) -> None:
        """Initialize storage for a keyframe type if not present."""
        if key not in self._keyframes:
            self._keyframes[key] = {}
            self._timestamps[key] = []

    def _insert_sorted(self, key: KeyframeType, timestamp_us: int) -> None:
        """Insert timestamp into the sorted list (maintain sort order)."""
        ts_list = self._timestamps[key]
        idx = bisect.bisect_left(ts_list, timestamp_us)
        if idx >= len(ts_list) or ts_list[idx] != timestamp_us:
            ts_list.insert(idx, timestamp_us)

    def _remove_sorted(self, key: KeyframeType, timestamp_us: int) -> None:
        """Remove timestamp from the sorted list."""
        ts_list = self._timestamps[key]
        idx = bisect.bisect_left(ts_list, timestamp_us)
        if idx < len(ts_list) and ts_list[idx] == timestamp_us:
            ts_list.pop(idx)

    # ------------------------------------------------------------------
    # Public API: keyframe CRUD
    # ------------------------------------------------------------------

    def set_keyframe(
        self,
        key: KeyframeType,
        timestamp_us: int,
        value: float,
        easing: Easing = Easing.EaseInOut,
    ) -> int:
        """Set a keyframe at the given timestamp.

        If a keyframe already exists within +/-1ms, updates its value.
        Otherwise creates a new one.

        Args:
            key: Keyframe type.
            timestamp_us: Timestamp in microseconds.
            value: Keyframe value.
            easing: Easing function (default: EaseInOut, matching Rust).

        Returns:
            The keyframe ID.
        """
        timestamp_us = self._get_closest_timestamp(key, timestamp_us)
        self._ensure_type(key)

        if timestamp_us in self._keyframes[key]:
            # Update existing
            kf = self._keyframes[key][timestamp_us]
            kf.value = value
            return kf.id
        else:
            # New keyframe
            kf = Keyframe(id=_new_id(), value=value, easing=easing)
            self._keyframes[key][timestamp_us] = kf
            self._insert_sorted(key, timestamp_us)
            return kf.id

    def remove_keyframe(self, key: KeyframeType, timestamp_us: int) -> None:
        """Remove a keyframe at the given timestamp (snaps to +/-1ms)."""
        timestamp_us = self._get_closest_timestamp(key, timestamp_us)
        if key in self._keyframes and timestamp_us in self._keyframes[key]:
            del self._keyframes[key][timestamp_us]
            self._remove_sorted(key, timestamp_us)

    def set_easing(
        self, key: KeyframeType, timestamp_us: int, easing: Easing
    ) -> None:
        """Set the easing for a keyframe at the given timestamp."""
        timestamp_us = self._get_closest_timestamp(key, timestamp_us)
        kfs = self._keyframes.get(key)
        if kfs and timestamp_us in kfs:
            kfs[timestamp_us].easing = easing

    def get_easing(
        self, key: KeyframeType, timestamp_us: int
    ) -> Optional[Easing]:
        """Get the easing for a keyframe at the given timestamp."""
        timestamp_us = self._get_closest_timestamp(key, timestamp_us)
        kfs = self._keyframes.get(key)
        if kfs and timestamp_us in kfs:
            return kfs[timestamp_us].easing
        return None

    def set_timestamp(
        self, key: KeyframeType, keyframe_id: int, new_timestamp_us: int
    ) -> None:
        """Move a keyframe (identified by its id) to a new timestamp."""
        kfs = self._keyframes.get(key)
        if not kfs:
            return

        old_ts = None
        for ts, kf in kfs.items():
            if kf.id == keyframe_id:
                old_ts = ts
                break

        if old_ts is not None and old_ts != new_timestamp_us:
            kf = kfs.pop(old_ts)
            self._remove_sorted(key, old_ts)
            kfs[new_timestamp_us] = kf
            self._insert_sorted(key, new_timestamp_us)

    def get_keyframe_id(
        self, key: KeyframeType, timestamp_us: int
    ) -> Optional[int]:
        """Get the ID of the keyframe at the given timestamp."""
        timestamp_us = self._get_closest_timestamp(key, timestamp_us)
        kfs = self._keyframes.get(key)
        if kfs and timestamp_us in kfs:
            return kfs[timestamp_us].id
        return None

    # ------------------------------------------------------------------
    # Public API: custom provider
    # ------------------------------------------------------------------

    def set_custom_provider(
        self,
        provider: Callable[
            [KeyframeManager, KeyframeType, float], Optional[float]
        ],
    ) -> None:
        """Set a custom value provider.

        The provider is called before stored keyframes during value lookup.
        Signature: (manager, key_type, timestamp_ms) -> Optional[float]
        If it returns a value, that value is used instead of interpolation.
        """
        self._custom_provider = provider

    # ------------------------------------------------------------------
    # Public API: value lookup
    # ------------------------------------------------------------------

    def value_at_timestamp(
        self, key: KeyframeType, timestamp_us: int
    ) -> Optional[float]:
        """Get interpolated value at a timestamp (microseconds).

        Lookup order matches Rust:
        1. Custom provider
        2. No keyframes -> None
        3. Single keyframe -> its value
        4. Clamp to keyframe range, then interpolate between neighbors
        """
        # 1. Custom provider
        if self._custom_provider is not None:
            scale = self.timestamp_scale if self.timestamp_scale is not None else 1.0
            ts_ms = timestamp_us / 1000.0 * scale
            result = self._custom_provider(self, key, ts_ms)
            if result is not None:
                return result

        kfs = self._keyframes.get(key)
        if not kfs:
            return None

        ts_list = self._timestamps[key]
        n = len(ts_list)

        if n == 0:
            return None

        if n == 1:
            return kfs[ts_list[0]].value

        # Clamp timestamp to keyframe range
        first_ts = ts_list[0]
        last_ts = ts_list[-1]
        lookup_ts = max(first_ts, min(last_ts, timestamp_us))

        # Find the keyframe at or before lookup_ts
        idx = bisect.bisect_right(ts_list, lookup_ts) - 1
        if idx < 0:
            idx = 0

        if ts_list[idx] == lookup_ts:
            return kfs[lookup_ts].value

        # Interpolate between ts_list[idx] and ts_list[idx + 1]
        if idx + 1 >= n:
            return kfs[ts_list[idx]].value

        ts_a = ts_list[idx]
        ts_b = ts_list[idx + 1]
        kf_a = kfs[ts_a]
        kf_b = kfs[ts_b]

        time_delta = ts_b - ts_a
        if time_delta == 0:
            return kf_a.value

        alpha = (timestamp_us - ts_a) / time_delta
        # Clamp alpha to [0, 1] for safety
        alpha = max(0.0, min(1.0, alpha))

        return Easing.interpolate(kf_a.easing, kf_b.easing, kf_a.value, kf_b.value, alpha)

    def value_at_video_timestamp(
        self, key: KeyframeType, timestamp_ms: float
    ) -> Optional[float]:
        """Get interpolated value at a video timestamp (milliseconds).

        Converts ms to us and applies timestamp_scale, matching Rust's
        value_at_video_timestamp.
        """
        scale = self.timestamp_scale if self.timestamp_scale is not None else 1.0
        timestamp_us = round(timestamp_ms * 1000.0 * scale)
        return self.value_at_timestamp(key, timestamp_us)

    def value_at_gyro_timestamp(
        self, key: KeyframeType, timestamp_ms: float
    ) -> Optional[float]:
        """Get interpolated value at a gyro timestamp (milliseconds).

        In Rust, this applies gyro offset before calling value_at_video_timestamp.
        The Python version currently delegates directly without offset,
        since gyro offset handling is not yet implemented.
        """
        return self.value_at_video_timestamp(key, timestamp_ms)

    # ------------------------------------------------------------------
    # Public API: queries
    # ------------------------------------------------------------------

    def is_keyframed(self, key: KeyframeType) -> bool:
        """Check if a keyframe type has any keyframes (including custom provider)."""
        if self._custom_provider is not None:
            scale = self.timestamp_scale if self.timestamp_scale is not None else 1.0
            result = self._custom_provider(self, key, 0.0 * scale)
            if result is not None:
                return True
        return self.is_keyframed_internally(key)

    def is_keyframed_internally(self, key: KeyframeType) -> bool:
        """Check stored keyframes only (no custom provider)."""
        kfs = self._keyframes.get(key)
        return bool(kfs)

    def get_all_keys(self) -> list[KeyframeType]:
        """Return all keyframe types that have stored keyframes."""
        return [
            key for key, kfs in self._keyframes.items() if kfs
        ]

    def get_keyframes(self, key: KeyframeType) -> Optional[dict[int, Keyframe]]:
        """Get all keyframes for a type (timestamp_us -> Keyframe)."""
        return self._keyframes.get(key)

    # ------------------------------------------------------------------
    # Public API: navigation
    # ------------------------------------------------------------------

    def next_keyframe(
        self, ts_us: int, key: Optional[KeyframeType] = None
    ) -> Optional[tuple[KeyframeType, int, Keyframe]]:
        """Find the next keyframe after the given timestamp.

        If key is specified, searches only that type.
        Otherwise searches all types and returns the closest next keyframe.

        Returns:
            (KeyframeType, timestamp_us, Keyframe) or None.
        """
        if key is not None:
            ts_list = self._timestamps.get(key, [])
            kfs = self._keyframes.get(key, {})
            # Find first timestamp > ts_us
            idx = bisect.bisect_right(ts_list, ts_us)
            if idx < len(ts_list):
                t = ts_list[idx]
                return (key, t, kfs[t])
            return None

        # Search all types
        candidates = []
        for kt in self._keyframes:
            result = self.next_keyframe(ts_us, kt)
            if result is not None:
                candidates.append(result)

        if not candidates:
            return None
        return min(candidates, key=lambda r: abs(r[1] - ts_us))

    def prev_keyframe(
        self, ts_us: int, key: Optional[KeyframeType] = None
    ) -> Optional[tuple[KeyframeType, int, Keyframe]]:
        """Find the previous keyframe before the given timestamp.

        If key is specified, searches only that type.
        Otherwise searches all types and returns the closest previous keyframe.

        Returns:
            (KeyframeType, timestamp_us, Keyframe) or None.
        """
        if key is not None:
            ts_list = self._timestamps.get(key, [])
            kfs = self._keyframes.get(key, {})
            # Find last timestamp < ts_us
            idx = bisect.bisect_left(ts_list, ts_us)
            if idx > 0:
                t = ts_list[idx - 1]
                return (key, t, kfs[t])
            return None

        # Search all types
        candidates = []
        for kt in self._keyframes:
            result = self.prev_keyframe(ts_us, kt)
            if result is not None:
                candidates.append(result)

        if not candidates:
            return None
        return min(candidates, key=lambda r: abs(r[1] - ts_us))

    # ------------------------------------------------------------------
    # Public API: bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all keyframes and reset state."""
        self._keyframes.clear()
        self._timestamps.clear()
        # Keep custom provider and timestamp_scale

    def clear_type(self, key: KeyframeType) -> None:
        """Remove all keyframes of a specific type."""
        self._keyframes.pop(key, None)
        self._timestamps.pop(key, None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize all keyframes to a JSON-compatible dict."""
        result = {}
        for key, kfs in self._keyframes.items():
            result[key.name] = {
                str(ts): {"id": kf.id, "value": kf.value, "easing": kf.easing.value}
                for ts, kf in kfs.items()
            }
        return result

    def deserialize(self, data: dict) -> None:
        """Deserialize keyframes from a dict, replacing current state."""
        self._keyframes.clear()
        self._timestamps.clear()

        for type_name, ts_map in data.items():
            try:
                key = KeyframeType[type_name]
            except KeyError:
                continue

            self._ensure_type(key)
            for ts_str, kf_data in ts_map.items():
                ts = int(ts_str)
                kf = Keyframe(
                    id=kf_data["id"],
                    value=kf_data["value"],
                    easing=Easing(kf_data["easing"]),
                )
                self._keyframes[key][ts] = kf
                self._insert_sorted(key, ts)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.serialize())

    def from_json(self, json_str: str) -> None:
        """Deserialize from JSON string."""
        self.deserialize(json.loads(json_str))
