"""Lens profile database — loading, searching and managing calibration profiles.

Ported from Gyroflow's core/lens_profile_database.rs.  Supports loading from
individual JSON files, directories of profiles, and the bundled CBOR+Gzip
format used by Gyroflow's ``profiles.cbor.gz``.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import zlib
from pathlib import Path
from typing import Any

from pygyroflow.lens.profile import LensProfile

logger = logging.getLogger(__name__)

CBOR_BUNDLE_NAME = "profiles.cbor.gz"


def default_lens_profile_dir() -> str:
    """User-writable lens profile directory.

    Follows the XDG base-dir spec (``$XDG_DATA_HOME``, falling back to
    ``~/.local/share``) so that synced profiles live outside the installed
    package and survive upgrades.
    """
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "pygyroflow", "lens_profiles")


def default_git_checkout_dir() -> str:
    """Default target of a raw-repository checkout (``sync clone``).

    A sibling of the release dir rather than a subdirectory of it: a git
    working tree is not a drop-in replacement for the bundle dir (which holds
    a single ``profiles.cbor.gz``), and a clone needs an empty destination.
    """
    return default_lens_profile_dir() + "_repo"


def bundled_lens_profile_dirs() -> list[str]:
    """Candidate locations of the profiles shipped inside the package.

    The bundled copy is a snapshot taken when the package was built; it is
    the fallback when the user has not synced a newer database.  The upward
    walk covers source checkouts of the surrounding Rust workspace, where
    ``profiles.cbor.gz`` lives next to the Python package.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(here, "..", "resources", "camera_presets")]

    parent = os.path.normpath(os.path.join(here, "..", "..", ".."))
    for _ in range(3):
        if os.path.isfile(os.path.join(parent, "resources", CBOR_BUNDLE_NAME)):
            dirs.append(os.path.join(parent, "resources"))
            break
        if os.path.isdir(os.path.join(parent, "resources", "camera_presets")):
            dirs.append(os.path.join(parent, "resources", "camera_presets"))
            break
        parent = os.path.dirname(parent)

    return [os.path.normpath(d) for d in dirs]


def lens_profile_search_paths() -> list[str]:
    """Ordered search path: user data dirs first, bundled snapshot last.

    Mirrors upstream ``load_all`` — user data wins on key collisions, and
    exactly one source ends up loaded.  The release bundle and the git
    checkout are both user data; the bundle comes first because it is the
    one a plain ``sync update`` produces.
    """
    return [
        default_lens_profile_dir(),
        default_git_checkout_dir(),
        *bundled_lens_profile_dirs(),
    ]


class LensProfileDatabase:
    """Collection of lens profiles with search capabilities.

    Profiles are stored as a list of ``(key, LensProfile)`` pairs where
    *key* is typically the filename or profile identifier.  Provides
    text search with brand-alias expansion, aspect-ratio filtering, and
    favourite / preset prioritised sorting.
    """

    def __init__(self) -> None:
        self.profiles: list[tuple[str, LensProfile]] = []
        self._by_key: dict[str, LensProfile] = {}
        self._version: int = 0
        self.loaded: bool = False

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def load_from_cbor(self, path: str) -> None:
        """Load profiles from a CBOR+Gzip file (Gyroflow's bundled format).

        The file is a gzip-compressed CBOR array of ``(str, dict)`` pairs.
        Special keys starting with ``"__"`` are metadata (e.g. ``"__version"``).

        Requires the ``cbor2`` package.  Missing file or parse errors are
        logged and silently ignored.
        """
        try:
            import cbor2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("cbor2 package not installed — cannot load %s", path)
            return

        if not os.path.isfile(path):
            logger.debug("CBOR file does not exist: %s", path)
            return

        try:
            with open(path, "rb") as fh:
                compressed = fh.read()
            decompressed = gzip.decompress(compressed)
            array: list[tuple[str, Any]] = cbor2.loads(decompressed)
        except Exception as exc:
            logger.error("Failed to read CBOR file %s: %s", path, exc)
            return

        if not isinstance(array, list):
            logger.error("CBOR root is not an array in %s", path)
            return

        for item in array:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            fname, profile_data = item
            if not isinstance(fname, str):
                continue

            # Metadata keys
            if fname == "__version":
                try:
                    self._version = int(profile_data)
                except (TypeError, ValueError):
                    pass
                continue
            if fname.startswith("__"):
                continue

            self._load_single_profile(profile_data, fname)

        self._resolve_all_interpolations()
        self.loaded = True

    def load_from_directory(self, directory: str) -> None:
        """Recursively load ``.json`` profiles from *directory*.

        The relative path (``"GoPro/foo.json"``) is passed as the key stem —
        the same string the prebuilt ``profiles.cbor.gz`` stores for that file,
        since the bundle is built by walking the repo from its root.  Profiles
        that carry an explicit ``identifier`` are keyed by it instead (see
        :meth:`_load_single_profile`), so a raw clone of the upstream
        ``lens_profiles`` repo and the bundled snapshot end up with *identical*
        key sets and dedupe against each other.  ``path_to_file`` keeps the
        absolute path so profiles stay reloadable from disk.

        ``.gyroflow`` project files are loaded as stub profiles (name only, no
        calibration data).  A ``.cbor.gz`` bundle found in the tree is loaded
        instead of the loose JSON files.  Hidden directories (``.git``) are
        skipped.
        """
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            logger.debug("Directory does not exist: %s", directory)
            return

        bundle_loaded = False
        for root, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for fname in sorted(files):
                if fname.startswith("."):
                    continue
                fpath = os.path.join(root, fname)
                key = os.path.relpath(fpath, directory).replace(os.sep, "/")

                if fname.endswith(".gyroflow"):
                    stub = LensProfile()
                    stub.name = Path(fname).stem
                    stub.path_to_file = fpath
                    # lens_profile_database.rs:81 — the preset's checksum
                    # is the crc32 of its path, so favourites keyed by it
                    # survive the file's contents changing.
                    stub.checksum = format(
                        zlib.crc32(fpath.encode("utf-8")) & 0xFFFFFFFF, "08x"
                    )
                    self._insert(key, stub)
                    continue

                if fname.endswith(".json"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as fh:
                            raw = fh.read()
                    except OSError as exc:
                        logger.error("Cannot read %s: %s", fpath, exc)
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        logger.error("Invalid JSON in %s: %s", fpath, exc)
                        continue
                    self._load_single_profile(data, key, path_to_file=fpath)

                elif not bundle_loaded and fname.endswith(".cbor.gz"):
                    self.load_from_cbor(fpath)
                    bundle_loaded = True

        self._resolve_all_interpolations()
        self.loaded = True

    def load_all(self, extra_dirs: list[str] | None = None) -> None:
        """Load every profile source, in priority order.

        Mirrors the Rust ``load_all``:
        1. Explicit ``extra_dirs`` (all of them, in order)
        2. The user data dir (:func:`default_lens_profile_dir`)
        3. The snapshot bundled with the package

        Steps 2 and 3 load the *first* source that exists and is non-empty,
        matching upstream's "exactly one bundle" rule.  An empty user dir
        therefore never shadows the bundled profiles.
        """
        for d in extra_dirs or []:
            self.load_from_directory(d)

        for candidate in lens_profile_search_paths():
            if self._load_first_available(candidate):
                break

        self.loaded = True

    def _load_first_available(self, candidate: str) -> bool:
        """Load the CBOR bundle (preferred) or the JSON tree at *candidate*.

        Returns True when something was loaded.  An existing but empty
        directory returns False so the next candidate gets a chance.
        """
        cbor_path = os.path.join(candidate, CBOR_BUNDLE_NAME)
        if os.path.isfile(cbor_path):
            self.load_from_cbor(cbor_path)
            return True
        if os.path.isdir(candidate) and os.listdir(candidate):
            self.load_from_directory(candidate)
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def get_by_name(self, name: str) -> LensProfile | None:
        """Return profile by exact *name* field."""
        for _key, prof in self.profiles:
            if prof.name == name:
                return prof
        return None

    def get_by_id(self, identifier: str) -> LensProfile | None:
        """Return profile by its unique key / identifier."""
        return self._by_key.get(identifier)

    def find(self, filename_or_id: str) -> LensProfile | None:
        """Look up by key, then by substring match on path_to_file."""
        hit = self._by_key.get(filename_or_id)
        if hit is not None:
            return hit
        needle = filename_or_id.replace("\\", "/")
        for _key, prof in self.profiles:
            if needle in prof.path_to_file.replace("\\", "/"):
                return prof
        return None

    def search(
        self,
        text: str,
        favorites: set[str] | None = None,
        aspect_ratio: float | None = None,
        limit: int = 200,
    ) -> list[LensProfile]:
        """Search profiles by *text* with brand-alias support.

        All whitespace-separated words must appear in the profile's display
        name or author.  Results are sorted by priority:
        1. Favourited or preset profiles
        2. Aspect-ratio match
        3. Alphabetical by display name

        Parameters
        ----------
        text:
            Search query.  Brand aliases are expanded automatically.
        favorites:
            Set of profile checksums to treat as favourites.
        aspect_ratio:
            Optional ``width/height`` ratio to boost matching profiles.
        limit:
            Maximum number of results (default 200).
        """
        favorites = favorites or set()
        expanded = self._expand_search_text(text)
        words = [w.strip().lower() for w in expanded.split() if w.strip()]
        if not words:
            return []

        # Compute aspect-ratio integer (ratio * 1000, matching Rust)
        ar_int = round(aspect_ratio * 1000) if aspect_ratio is not None else 0

        # Build candidate list: (display_name, key, profile, aspect_ratio_int)
        candidates: list[tuple[str, str, LensProfile, int]] = []
        for key, prof in self.profiles:
            display = prof.get_display_name()
            name_lower = display.lower()
            author_lower = prof.calibrated_by.lower()

            if all(w in name_lower or w in author_lower for w in words):
                # Compute aspect ratio for this profile
                cw = prof.calib_dimension["w"]
                ch = prof.calib_dimension["h"]
                hstretch = max(prof.input_horizontal_stretch, 0.01) if prof.input_horizontal_stretch > 0.01 else 1.0
                vstretch = max(prof.input_vertical_stretch, 0.01) if prof.input_vertical_stretch > 0.01 else 1.0
                prof_ar = round((cw / hstretch) / (max(ch, 1) / vstretch) * 1000) if cw > 0 and ch > 0 else 0
                candidates.append((display, key, prof, prof_ar))

        # Sort by priority
        def _sort_key(item: tuple[str, str, LensProfile, int]) -> tuple:
            display, key, prof, prof_ar = item
            is_priority = key.endswith(".gyroflow") or (prof.checksum in favorites if prof.checksum else False)
            ar_match = prof_ar != 0 and ar_int == prof_ar
            return (0 if is_priority else 1, 0 if ar_match else 1, display.lower())

        candidates.sort(key=_sort_key)
        return [prof for _d, _k, prof, _ar in candidates[:limit]]

    @property
    def version(self) -> int:
        """Database version from the CBOR ``__version`` metadata key."""
        return self._version

    def __len__(self) -> int:
        return len(self.profiles)

    def _load_single_profile(self, data: Any, fname: str, path_to_file: str | None = None) -> None:
        """Parse a single profile dict and insert it (with compatible copies).

        *fname* is the key stem: profiles that declare an ``identifier`` are
        keyed by it, the rest fall back to *fname* — which is the path relative
        to the source root, matching what the CBOR bundle records.  It doubles
        as the default ``path_to_file``; *path_to_file* overrides it when the
        profile came from a directory scan, preserving the absolute location.
        """
        if not isinstance(data, dict):
            return
        try:
            profile = LensProfile.from_json(data)
            _assign_profile_checksum(profile)
        except Exception as exc:
            logger.error("Error parsing lens profile %s: %s", fname, exc)
            return

        profile.path_to_file = path_to_file if path_to_file is not None else fname
        for derived in profile.get_all_matching_profiles():
            key = derived.identifier if derived.identifier else fname
            self._insert(key, derived)

    def _insert(self, key: str, profile: LensProfile) -> None:
        """Add or replace a profile entry."""
        if key in self._by_key:
            if not self.loaded:
                logger.debug("Duplicate profile key: %s", key)
            return
        self._by_key[key] = profile
        self.profiles.append((key, profile))

    def _resolve_all_interpolations(self) -> None:
        """Resolve interpolation data for every loaded profile."""
        for _key, prof in self.profiles:
            prof.resolve_interpolations(database=self)

    # ------------------------------------------------------------------ #
    #  Brand aliases                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand_search_text(text: str) -> str:
        """Expand brand aliases in the search text.

        Matches Gyroflow's alias table — GoPro models, Sony ILCE/ILME codes,
        Blackmagic abbreviations, etc.
        """
        t = text.lower()

        # Multi-word replacements first (longer matches)
        replacements = [
            ("bmpcc4k",  "blackmagic pocket cinema camera 4k"),
            ("bmpcc6k",  "blackmagic pocket cinema camera 6k"),
            ("bmpcc",    "blackmagic pocket cinema camera"),
            ("gopro5",   "hero5 black"),  ("gopro 5",  "hero5 black"),
            ("gopro6",   "hero6 black"),  ("gopro 6",  "hero6 black"),
            ("gopro7",   "hero7 black"),  ("gopro 7",  "hero7 black"),
            ("gopro8",   "hero8 black"),  ("gopro 8",  "hero8 black"),
            ("gopro9",   "hero9 black"),  ("gopro 9",  "hero9 black"),
            ("gopro10",  "hero10 black"), ("gopro 10", "hero10 black"),
            ("gopro11",  "hero11 black"), ("gopro 11", "hero11 black"),
            ("gopro12",  "hero11 black"), ("gopro 12", "hero11 black"),
            ("gopro13",  "hero11 black"), ("gopro 13", "hero11 black"),
            ("session5", "hero5 session"), ("session 5", "hero5 session"),
            ("a73", "a7iii"), ("a74", "a7iv"), ("a75", "a7v"),
            ("a7r3", "a7riii"), ("a7r4", "a7riv"), ("a7r5", "a7rv"),
            ("a7s2", "a7sii"), ("a7s3", "a7siii"),
        ]
        for old, new in replacements:
            t = t.replace(old, new)

        # Also handle comma/semicolon as space
        t = t.replace(",", " ").replace(";", " ")
        return t

    @staticmethod
    def _replace_brand_aliases(text: str) -> str:
        """Public alias: replace common brand abbreviations in a single token."""
        aliases = {
            "gopro": "hero",
            "gopro2": "hero2", "gopro3": "hero3", "gopro4": "hero4",
            "gopro5": "hero5 black", "gopro6": "hero6 black",
            "gopro7": "hero7 black", "gopro8": "hero8 black",
            "gopro9": "hero9 black", "gopro10": "hero10 black",
            "gopro11": "hero11 black", "gopro12": "hero12 black",
            "sony": "ILCE",
            "fx3": "ILME-FX3", "fx6": "ILME-FX6",
            "a7": "ILCE-7", "a7s": "ILCE-7S", "a7r": "ILCE-7RM",
            "a7iv": "ILCE-7M4", "a7iii": "ILCE-7M3",
        }
        return aliases.get(text.lower(), text)


def _assign_profile_checksum(profile: LensProfile) -> None:
    """The crc32 a JSON profile gets on load (lens_profile_database.rs:
    112-130): ``{identifier}|{w}{h}|{fx:.8}{fy:.8}|{cx:.8}{cy:.8}|{4 coeffs
    :.8}``. Floats print with 8 decimals — Rust's ``{:.8}`` — and the
    string is hashed as raw bytes. A profile without a usable camera
    matrix keeps ``None`` (upstream's closure bails the same way)."""
    matrix = profile.camera_matrix
    if not isinstance(matrix, (list, tuple)) or len(matrix) < 2:
        return
    try:
        fx = float(matrix[0][0]); fy = float(matrix[1][1])
        cx = float(matrix[0][2]); cy = float(matrix[1][2])
    except (TypeError, IndexError, ValueError):
        return
    coeffs = [float(c) for c in (profile.distortion_coeffs or [])[:4]]
    coeffs += [0.0] * (4 - len(coeffs))
    w = profile.calib_dimension.get("w", 0)
    h = profile.calib_dimension.get("h", 0)
    payload = (
        f"{profile.identifier or ''}"
        f"|{w}{h}"
        f"|{fx:.8f}{fy:.8f}"
        f"|{cx:.8f}{cy:.8f}"
        f"|" + "".join(f"{c:.8f}" for c in coeffs)
    )
    profile.checksum = format(
        zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF, "08x"
    )
