# -*- coding: utf-8 -*-
"""Lens profile database sync — pull the upstream ``gyroflow/lens_profiles`` data.

Two sources are supported:

``release`` (default)
    Download the prebuilt ``profiles.cbor.gz`` attached to the latest GitHub
    release of ``gyroflow/lens_profiles``.  ~1.5 MB, byte-identical to the file
    that ships inside the package, and its ``__version`` is the repo's CI run
    number — which is exactly what the release tag (``v41``) says.

``git``
    Clone / fast-forward the raw JSON repository.  Thousands of small files
    (slower to load, ~10x the disk) but lets you pin a commit, diff two
    releases, or work from a local mirror.

Synced data lands outside the installed package, in the user data dirs that
:func:`~pygyroflow.lens.database.lens_profile_search_paths` consults before the
bundled snapshot: ``update`` writes ``lens_profiles/profiles.cbor.gz``, and
``clone`` checks out into ``lens_profiles_repo/``.  Whichever of those exists
first wins; an empty directory never shadows the bundled profiles.

Command line::

    python -m pygyroflow.lens.sync status
    python -m pygyroflow.lens.sync check
    python -m pygyroflow.lens.sync update
    python -m pygyroflow.lens.sync clone

Nothing here raises on network failure — every entry point returns a result
dict carrying ``ok`` plus an ``error`` string, so callers (and the CLI) can
report problems without a traceback.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from pygyroflow.lens.database import (
    CBOR_BUNDLE_NAME,
    bundled_lens_profile_dirs,
    default_git_checkout_dir,
    default_lens_profile_dir,
)

logger = logging.getLogger(__name__)

REPO_SLUG = "gyroflow/lens_profiles"
REPO_URL = f"https://github.com/{REPO_SLUG}.git"
RELEASE_API = f"https://api.github.com/repos/{REPO_SLUG}/releases/latest"
RELEASE_ASSET = f"https://github.com/{REPO_SLUG}/releases/latest/download/{CBOR_BUNDLE_NAME}"
METADATA_NAME = "sync.json"

DEFAULT_TIMEOUT = 30.0
_USER_AGENT = "pygyroflow-lens-sync/0.1"


# --------------------------------------------------------------------------- #
#  Local state                                                                 #
# --------------------------------------------------------------------------- #


def read_cbor_version(path: str) -> int | None:
    """Read the ``__version`` metadata entry from a ``profiles.cbor.gz``.

    Returns ``None`` when the file is missing, unreadable, or has no version
    entry — the version is metadata, never a hard requirement.
    """
    if not os.path.isfile(path):
        return None
    try:
        import cbor2  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("cbor2 not installed — cannot read version from %s", path)
        return None
    try:
        with open(path, "rb") as fh:
            data = cbor2.loads(gzip.decompress(fh.read()))
    except Exception as exc:  # gzip/BadGzipFile/cbor decode
        logger.warning("Cannot read %s: %s", path, exc)
        return None
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "__version":
            try:
                return int(item[1])
            except (TypeError, ValueError):
                return None
    return None


def read_metadata(directory: str) -> dict:
    """Read the ``sync.json`` sidecar written by :func:`update`."""
    path = os.path.join(directory, METADATA_NAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def local_version(directory: str | None = None) -> int | None:
    """Version of the synced bundle in *directory* (``None`` if absent)."""
    directory = directory or default_lens_profile_dir()
    return read_cbor_version(os.path.join(directory, CBOR_BUNDLE_NAME))


def local_status(directory: str | None = None) -> dict:
    """Describe what is currently on disk, for both sources.

    ``bundled_version`` is the snapshot inside the installed package, so a
    user can see at a glance whether syncing would actually gain anything.
    """
    directory = directory or default_lens_profile_dir()
    bundle = os.path.join(directory, CBOR_BUNDLE_NAME)
    meta = read_metadata(directory)

    checkout = default_git_checkout_dir() if directory == default_lens_profile_dir() else None
    git_commit = _git_commit(checkout) if checkout else None

    return {
        "ok": True,
        "directory": directory,
        "has_bundle": os.path.isfile(bundle),
        "version": read_cbor_version(bundle),
        "size_bytes": os.path.getsize(bundle) if os.path.isfile(bundle) else 0,
        "downloaded_at": meta.get("downloaded_at"),
        "source_url": meta.get("url"),
        "bundled_version": _bundled_version(),
        "git_checkout": checkout,
        "git_commit": git_commit,
    }


def _bundled_version() -> int | None:
    """Version of the snapshot shipped with the package."""
    for directory in bundled_lens_profile_dirs():
        version = read_cbor_version(os.path.join(directory, CBOR_BUNDLE_NAME))
        if version is not None:
            return version
    return None


# --------------------------------------------------------------------------- #
#  Remote queries                                                              #
# --------------------------------------------------------------------------- #


def _http_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """GET *url* and return the body.  Raises on HTTP/URL errors."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def remote_version(timeout: float = DEFAULT_TIMEOUT) -> int | None:
    """Newest published release number, or ``None`` if it cannot be read.

    Release tags are ``v<CI run number>``, the same counter that ends up in
    the bundle's ``__version``.
    """
    try:
        payload = json.loads(_http_get(RELEASE_API, timeout))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Cannot reach the lens profile release API: %s", exc)
        return None
    tag = str(payload.get("tag_name", "")).lstrip("vV")
    try:
        return int(tag)
    except ValueError:
        logger.warning("Unexpected release tag: %r", payload.get("tag_name"))
        return None


def check_update(directory: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Compare the local bundle against the newest published release.

    Falls back to the bundled snapshot when no user copy exists, since that is
    what :meth:`LensProfileDatabase.load_all` would actually use.
    """
    directory = directory or default_lens_profile_dir()
    current = local_version(directory)
    origin = "user" if current is not None else "bundled"
    if current is None:
        current = _bundled_version()

    latest = remote_version(timeout)
    return {
        "ok": latest is not None,
        "directory": directory,
        "local_version": current,
        "local_source": origin,
        "remote_version": latest,
        "update_available": latest is not None and (current is None or latest > current),
        "error": None if latest is not None else "release API unreachable",
    }


# --------------------------------------------------------------------------- #
#  Update                                                                      #
# --------------------------------------------------------------------------- #


def update(
    directory: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> dict:
    """Download the latest ``profiles.cbor.gz`` into *directory*.

    The download is staged in a temporary file and moved into place only after
    it parses and its version matches the release tag, so an interrupted sync
    can never leave a corrupt bundle behind.  Returns a result dict; ``ok`` is
    False on any failure.
    """
    directory = directory or default_lens_profile_dir()
    dest = os.path.join(directory, CBOR_BUNDLE_NAME)

    current = read_cbor_version(dest)
    latest = remote_version(timeout)
    if latest is None:
        return _fail(directory, "release API unreachable — check the network")
    if current is not None and latest <= current and not force:
        return {
            "ok": True,
            "directory": directory,
            "version": current,
            "updated": False,
            "message": f"already up to date (v{current})",
        }

    try:
        os.makedirs(directory, exist_ok=True)
        blob = _http_get(RELEASE_ASSET, timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return _fail(directory, f"download failed: {exc}")

    staged = f"{dest}.tmp{os.getpid()}"
    try:
        with open(staged, "wb") as fh:
            fh.write(blob)
        staged_version = read_cbor_version(staged)
        if staged_version is None:
            raise ValueError("downloaded file is not a readable profiles.cbor.gz")
        # The tag is the CI run number that also lands in __version; a mismatch
        # means the asset and the API answer disagree (mirror lag, stale cache).
        if staged_version != latest:
            logger.warning(
                "Release tag v%s but bundle __version=%s", latest, staged_version
            )
        os.replace(staged, dest)
    except (OSError, ValueError) as exc:
        _unlink(staged)
        return _fail(directory, f"download rejected: {exc}")

    result = {
        "ok": True,
        "directory": directory,
        "version": staged_version,
        "updated": True,
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "url": RELEASE_ASSET,
        "downloaded_at": _now_iso(),
    }
    _write_metadata(directory, result)
    logger.info("Lens profiles updated to v%s (%d bytes)", staged_version, len(blob))
    return result


def _write_metadata(directory: str, result: dict) -> None:
    """Persist a small sidecar describing where the bundle came from."""
    payload = {
        "version": result.get("version"),
        "sha256": result.get("sha256"),
        "size_bytes": result.get("size_bytes"),
        "url": result.get("url"),
        "downloaded_at": result.get("downloaded_at"),
    }
    path = os.path.join(directory, METADATA_NAME)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Cannot write %s: %s", path, exc)


# --------------------------------------------------------------------------- #
#  Git source                                                                  #
# --------------------------------------------------------------------------- #


def _git_commit(directory: str) -> str | None:
    """Short HEAD sha of a checkout, or ``None``."""
    if not os.path.isdir(os.path.join(directory, ".git")):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def clone_or_pull(directory: str | None = None, timeout: float = 300.0) -> dict:
    """Clone the raw JSON repo, or fast-forward an existing checkout.

    Uses ``--depth 1``: the history is not needed to read the profiles, and a
    shallow clone is a fraction of the download.
    """
    directory = directory or default_git_checkout_dir()
    if shutil.which("git") is None:
        return _fail(directory, "git is not installed")

    is_repo = os.path.isdir(os.path.join(directory, ".git"))
    cmd = (
        ["git", "-C", directory, "pull", "--ff-only"]
        if is_repo
        else ["git", "clone", "--depth", "1", REPO_URL, directory]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _fail(directory, f"git failed: {exc}")

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return _fail(directory, f"git failed: {detail[-1] if detail else proc.returncode}")

    files = sum(1 for _r, _d, fs in os.walk(directory) for f in fs if f.endswith(".json"))
    commit = _git_commit(directory)
    logger.info("Lens profile checkout %s at %s (%d json files)", "updated" if is_repo else "created", commit, files)
    return {
        "ok": True,
        "directory": directory,
        "action": "pull" if is_repo else "clone",
        "commit": commit,
        "json_files": files,
    }


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _fail(directory: str, message: str) -> dict:
    logger.error("%s", message)
    return {"ok": False, "directory": directory, "error": message}


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pygyroflow.lens.sync",
        description="Sync the gyroflow/lens_profiles database used by pyGyroFlow.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "check", "update", "clone"],
        help="status: show local state; check: compare with upstream; "
             "update: download the latest release bundle; clone: git checkout",
    )
    parser.add_argument("--dir", default=None, help="target directory (default: the XDG user data dir)")
    parser.add_argument("--force", action="store_true", help="re-download even if already current")
    parser.add_argument("--timeout", type=float, default=None, help="network timeout in seconds")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")

    if args.command == "status":
        result = local_status(args.dir)
    elif args.command == "check":
        result = check_update(args.dir, args.timeout or DEFAULT_TIMEOUT)
    elif args.command == "update":
        result = update(args.dir, force=args.force, timeout=args.timeout or 60.0)
    else:
        result = clone_or_pull(args.dir, timeout=args.timeout or 300.0)

    if not args.quiet:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
