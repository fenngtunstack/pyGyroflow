"""merge_json + the video-extension whitelist (C-15)."""
import pytest
from pygyroflow.util import merge_json, filename_has_video_extension


class TestMergeJson:
    def test_objects_merge_keywise(self):
        a = {"x": 1, "nested": {"p": 1}}
        b = {"nested": {"q": 2}, "new": 5}
        assert merge_json(a, b) == {"x": 1, "nested": {"p": 1, "q": 2}, "new": 5}

    def test_arrays_concatenate(self):
        assert merge_json({"arr": [1, 2]}, {"arr": [3]}) == {"arr": [1, 2, 3]}

    def test_object_into_array_appends(self):
        assert merge_json([1], {"k": 1}) == [1, {"k": 1}]

    def test_scalars_replace(self):
        assert merge_json({"x": 1}, {"x": 2}) == {"x": 2}
        assert merge_json("a", "b") == "b"

    def test_missing_key_inserts_b_value(self):
        # Rust: entry(k).or_insert(Null) then merge(null, v) -> v.
        assert merge_json({"a": None}, {"b": 5}) == {"a": None, "b": 5}

    def test_b_is_not_aliased(self):
        """Rust clones on merge; mutating the result must not reach *b*."""
        b = {"nested": {"q": 2}}
        out = merge_json({"x": 1}, b)
        out["nested"]["q"] = 99
        assert b["nested"]["q"] == 2

    def test_deeply_nested(self):
        a = {"s": {"of_method": 0, "params": {"a": 1}}}
        b = {"s": {"params": {"b": 2}}}
        assert merge_json(a, b) == {
            "s": {"of_method": 0, "params": {"a": 1, "b": 2}}
        }


class TestVideoExtensionWhitelist:
    @pytest.mark.parametrize("name", [
        "clip.mp4", "CLIP.MOV", "raw.braw", "one.insv", "two.360", "x.mxf",
        "/path/to/a.mp4",
    ])
    def test_accepted(self, name):
        assert filename_has_video_extension(name)

    @pytest.mark.parametrize("name", ["clip.txt", "clip.mp4.bak", "noext", "clip.avi"])
    def test_refused(self, name):
        assert not filename_has_video_extension(name)
