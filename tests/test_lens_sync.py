# -*- coding: utf-8 -*-
"""Tests for the lens profile sync mechanism and user-dir load priority."""

from __future__ import annotations

import gzip
import json
import os
import urllib.error

import pytest

from pygyroflow.lens import LensProfileDatabase, default_lens_profile_dir, sync
from pygyroflow.lens.database import bundled_lens_profile_dirs, lens_profile_search_paths

CBOR = "profiles.cbor.gz"


# --------------------------------------------------------------------------- #
#  Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #


def _profile_entry(name: str, **overrides) -> dict:
    entry = {
        "name": name,
        "camera_brand": "TestBrand",
        "camera_model": name,
        "calib_dimension": {"w": 1920, "h": 1080},
        "fps": 30.0,
        "fisheye_params": {"camera_matrix": [[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]]},
    }
    entry.update(overrides)
    return entry


def _write_bundle(path: str, version: int, entries: list[tuple[str, dict]]) -> bytes:
    """Write a synthetic ``profiles.cbor.gz`` and return its raw bytes."""
    cbor2 = pytest.importorskip("cbor2")
    blob = gzip.compress(cbor2.dumps([("__version", version), *entries]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return blob


@pytest.fixture
def user_root(tmp_path, monkeypatch):
    """Point the XDG data dir at a temp dir and return it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    root = tmp_path / "pygyroflow" / "lens_profiles"
    return root


# --------------------------------------------------------------------------- #
#  Version / metadata reading                                                  #
# --------------------------------------------------------------------------- #


class TestReadCborVersion:
    def test_reads_version(self, tmp_path):
        path = str(tmp_path / CBOR)
        _write_bundle(path, 7, [("Brand/a.json", _profile_entry("A"))])
        assert sync.read_cbor_version(path) == 7

    def test_missing_file_is_none(self, tmp_path):
        assert sync.read_cbor_version(str(tmp_path / CBOR)) is None

    def test_corrupt_file_is_none(self, tmp_path):
        path = tmp_path / CBOR
        path.write_bytes(b"not a gzip stream")
        assert sync.read_cbor_version(str(path)) is None

    def test_bundle_without_version_entry(self, tmp_path):
        cbor2 = pytest.importorskip("cbor2")
        path = tmp_path / CBOR
        path.write_bytes(gzip.compress(cbor2.dumps([("Brand/a.json", {})])))
        assert sync.read_cbor_version(str(path)) is None

    def test_bundled_snapshot_has_a_version(self):
        # The snapshot we ship must stay readable by the sync tooling.
        assert sync._bundled_version() is not None


# --------------------------------------------------------------------------- #
#  Search path / load priority                                                 #
# --------------------------------------------------------------------------- #


class TestLoadPriority:
    def test_user_dir_is_first(self, user_root):
        assert lens_profile_search_paths()[0] == str(user_root)

    def test_bundled_dirs_are_the_fallback(self, user_root):
        paths = lens_profile_search_paths()
        for bundled in bundled_lens_profile_dirs():
            assert bundled in paths
            assert paths.index(bundled) > 0

    def test_user_bundle_wins_over_bundled(self, user_root):
        _write_bundle(str(user_root / CBOR), 99, [("Brand/custom.json", _profile_entry("Custom"))])

        db = LensProfileDatabase()
        db.load_all()

        assert db.version == 99
        assert len(db) == 1
        assert db.get_by_id("Brand/custom.json") is not None

    def test_empty_user_dir_does_not_shadow_bundle(self, user_root):
        # A directory that exists but holds nothing must not satisfy the search,
        # otherwise merely creating ~/.local/share/pygyroflow/lens_profiles
        # would silently drop every lens profile.
        user_root.mkdir(parents=True)
        db = LensProfileDatabase()
        assert db._load_first_available(str(user_root)) is False
        assert len(db) == 0

    def test_git_checkout_is_searched(self, user_root):
        # `sync clone` writes to a sibling dir; without it in the search path
        # the git source would download data nothing ever reads.
        checkout = user_root.parent / "lens_profiles_repo" / "GoPro"
        checkout.mkdir(parents=True)
        (checkout / "cam.json").write_text(json.dumps(_profile_entry("Cloned")), encoding="utf-8")

        db = LensProfileDatabase()
        db.load_all()

        assert db.get_by_id("GoPro/cam.json") is not None

    def test_extra_dirs_merge_before_search_path(self, user_root, tmp_path):
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "one.json").write_text(json.dumps(_profile_entry("Extra")), encoding="utf-8")
        _write_bundle(str(user_root / CBOR), 5, [("Brand/kept.json", _profile_entry("Kept"))])

        db = LensProfileDatabase()
        db.load_all(extra_dirs=[str(extra)])

        assert {key for key, _p in db.profiles} == {"one.json", "Brand/kept.json"}

    def test_missing_everything_leaves_database_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "nope"))
        monkeypatch.setattr(
            "pygyroflow.lens.database.bundled_lens_profile_dirs", lambda: [str(tmp_path / "also-gone")]
        )
        db = LensProfileDatabase()
        db.load_all()
        assert len(db) == 0
        assert db.loaded


# --------------------------------------------------------------------------- #
#  Directory keys (relpath parity with the bundled bundle)                     #
# --------------------------------------------------------------------------- #


class TestDirectoryKeys:
    def test_keys_are_relative_paths(self, tmp_path):
        brand = tmp_path / "GoPro"
        brand.mkdir()
        (brand / "cam.json").write_text(json.dumps(_profile_entry("Cam")), encoding="utf-8")

        db = LensProfileDatabase()
        db.load_from_directory(str(tmp_path))

        assert db.get_by_id("GoPro/cam.json") is not None
        # the absolute location is preserved for reloads
        assert db.get_by_id("GoPro/cam.json").path_to_file == str(brand / "cam.json")

    def test_hidden_directories_are_skipped(self, tmp_path):
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "junk.json").write_text(json.dumps(_profile_entry("Junk")), encoding="utf-8")
        (tmp_path / "real.json").write_text(json.dumps(_profile_entry("Real")), encoding="utf-8")

        db = LensProfileDatabase()
        db.load_from_directory(str(tmp_path))

        assert db.get_by_id("real.json") is not None
        assert all(not key.startswith(".git") for key, _p in db.profiles)

    def test_directory_keys_match_bundle_keys(self, tmp_path):
        # Same profile via bundle and via directory scan -> same keys, so the
        # two sources dedupe instead of doubling every entry.  Covers both key
        # paths: identifier-keyed profiles and the relpath fallback.
        identified = _profile_entry(
            "Identified",
            identifier="brand-identified-1920x1080@30000",
            compatible_settings=[{"width": 1280, "height": 720, "identifier": "brand-identified-1280x720@30000"}],
        )
        anonymous = _profile_entry("Anonymous")
        entries = [("GoPro/cam.json", identified), ("GoPro/plain.json", anonymous)]

        bundle_path = str(tmp_path / "bundle" / CBOR)
        _write_bundle(bundle_path, 1, entries)

        loose = tmp_path / "loose" / "GoPro"
        loose.mkdir(parents=True)
        for rel, data in entries:
            (tmp_path / "loose" / rel).write_text(json.dumps(data), encoding="utf-8")

        from_bundle = LensProfileDatabase()
        from_bundle.load_from_cbor(bundle_path)
        from_dir = LensProfileDatabase()
        from_dir.load_from_directory(str(tmp_path / "loose"))

        bundle_keys = [k for k, _p in from_bundle.profiles]
        dir_keys = [k for k, _p in from_dir.profiles]
        assert sorted(bundle_keys) == sorted(dir_keys)
        assert "brand-identified-1920x1080@30000" in dir_keys
        assert "GoPro/plain.json" in dir_keys


# --------------------------------------------------------------------------- #
#  Remote comparison                                                           #
# --------------------------------------------------------------------------- #


class TestCheckUpdate:
    def test_newer_release_is_reported(self, user_root, monkeypatch):
        _write_bundle(str(user_root / CBOR), 10, [])
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: 11)
        result = sync.check_update()
        assert result["ok"] and result["update_available"]
        assert result["local_version"] == 10

    def test_same_release_is_not_an_update(self, user_root, monkeypatch):
        _write_bundle(str(user_root / CBOR), 41, [])
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: 41)
        assert sync.check_update()["update_available"] is False

    def test_falls_back_to_bundled_version(self, user_root, monkeypatch):
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: 41)
        result = sync.check_update()
        assert result["local_source"] == "bundled"
        assert result["local_version"] == sync._bundled_version()

    def test_unreachable_api_is_not_ok(self, user_root, monkeypatch):
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: None)
        result = sync.check_update()
        assert result["ok"] is False
        assert result["update_available"] is False
        assert result["error"]


# --------------------------------------------------------------------------- #
#  Update                                                                      #
# --------------------------------------------------------------------------- #


class TestUpdate:
    def _fake_get(self, monkeypatch, blob: bytes, remote: int):
        def _get(url, timeout=None):
            if url == sync.RELEASE_ASSET:
                return blob
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(sync, "_http_get", _get)
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: remote)

    def test_downloads_and_records_metadata(self, user_root, monkeypatch):
        blob = _write_bundle(str(user_root / "staging" / CBOR), 41, [("Brand/a.json", _profile_entry("A"))])
        self._fake_get(monkeypatch, blob, 41)

        result = sync.update()

        assert result["ok"] and result["updated"]
        assert result["version"] == 41
        assert result["sha256"] == sync.hashlib.sha256(blob).hexdigest()
        assert os.path.isfile(user_root / CBOR)
        assert sync.local_version(str(user_root)) == 41

        meta = sync.read_metadata(str(user_root))
        assert meta["version"] == 41
        assert meta["url"] == sync.RELEASE_ASSET

    def test_no_download_when_current(self, user_root, monkeypatch):
        _write_bundle(str(user_root / CBOR), 41, [])
        self._fake_get(monkeypatch, b"", 41)

        result = sync.update()

        assert result["ok"] and result["updated"] is False
        assert "up to date" in result["message"]

    def test_force_redownloads(self, user_root, monkeypatch):
        _write_bundle(str(user_root / CBOR), 41, [])
        blob = _write_bundle(str(user_root / "staging2" / CBOR), 41, [("Brand/b.json", _profile_entry("B"))])
        self._fake_get(monkeypatch, blob, 41)

        assert sync.update(force=True)["updated"] is True

    def test_rejects_non_bundle_payload(self, user_root, monkeypatch):
        self._fake_get(monkeypatch, b"garbage", 41)

        result = sync.update()

        assert result["ok"] is False
        assert not os.path.exists(user_root / CBOR)

    def test_leaves_no_partial_file_on_failure(self, user_root, monkeypatch):
        def _boom(url, timeout=None):
            raise urllib.error.URLError("network down")

        monkeypatch.setattr(sync, "_http_get", _boom)
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: 41)

        result = sync.update()

        assert result["ok"] is False
        assert "download failed" in result["error"]
        assert not os.path.exists(user_root / CBOR)
        assert not any(p.name.startswith(CBOR + ".tmp") for p in user_root.iterdir())

    def test_unreachable_api_aborts(self, user_root, monkeypatch):
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: None)
        result = sync.update()
        assert result["ok"] is False
        assert not os.path.exists(user_root / CBOR)

    def test_interrupted_stage_is_not_promoted(self, user_root, monkeypatch):
        # Simulate a truncated write: gzip magic present, stream incomplete.
        self._fake_get(monkeypatch, b"\x1f\x8b\x08\x00truncated", 41)

        result = sync.update()

        assert result["ok"] is False
        assert not os.path.exists(user_root / CBOR)


# --------------------------------------------------------------------------- #
#  Git source / CLI                                                            #
# --------------------------------------------------------------------------- #


class TestGitSource:
    def test_missing_git_is_reported(self, monkeypatch):
        monkeypatch.setattr(sync.shutil, "which", lambda _name: None)
        result = sync.clone_or_pull("/tmp/does-not-matter")
        assert result["ok"] is False
        assert "git is not installed" in result["error"]

    def test_leaves_non_repo_directory_untouched(self, tmp_path, monkeypatch):
        # A directory that exists but is not a checkout must be rejected, not
        # silently declared synced.
        monkeypatch.setattr(sync.shutil, "which", lambda _name: "/usr/bin/git")

        def _fail(cmd, **_kwargs):
            class _Proc:
                returncode = 128
                stdout = ""
                stderr = "fatal: destination path already exists"

            return _Proc()

        monkeypatch.setattr(sync.subprocess, "run", _fail)
        result = sync.clone_or_pull(str(tmp_path))

        assert result["ok"] is False
        assert "already exists" in result["error"]


class TestCli:
    def test_status_exits_zero_and_prints_json(self, user_root, capsys):
        _write_bundle(str(user_root / CBOR), 3, [])
        assert sync.main(["status"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] and payload["version"] == 3

    def test_check_failure_exits_nonzero(self, user_root, monkeypatch, capsys):
        monkeypatch.setattr(sync, "remote_version", lambda *_a, **_k: None)
        assert sync.main(["check", "--quiet"]) == 1
        assert capsys.readouterr().out == ""

    def test_status_reports_missing_bundle(self, user_root, capsys):
        assert sync.main(["status"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["has_bundle"] is False
        assert payload["directory"] == str(user_root)


# --------------------------------------------------------------------------- #
#  Environment                                                                 #
# --------------------------------------------------------------------------- #


class TestDefaultDir:
    def test_xdg_env_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert default_lens_profile_dir() == str(tmp_path / "pygyroflow" / "lens_profiles")

    def test_home_fallback(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/tester")
        assert default_lens_profile_dir() == "/home/tester/.local/share/pygyroflow/lens_profiles"

    def test_blank_xdg_falls_back_to_home(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "   ")
        monkeypatch.setenv("HOME", "/home/tester")
        assert default_lens_profile_dir().startswith("/home/tester/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
