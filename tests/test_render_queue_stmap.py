"""The render queue's ST-map job (G-06's reachability half)."""

from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av", reason="av required")

from pygyroflow.rendering.render_queue import (  # noqa: E402
    RenderJob,
    RenderJobType,
    RenderQueue,
)


def _tiny_clip(path, width=64, height=48, frames=6):
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 0] = (i * 31) % 256
        img[height // 4: height // 2, width // 4: width // 2, 1] = 200
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return str(path)


class TestStmapJob:
    def test_undistort_only_by_default(self, tmp_path):
        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        queue = RenderQueue()
        queue.add_job(RenderJob(
            RenderJobType.STMap, clip, str(out),
        ))
        queue.process_all()
        assert queue.failures == []
        assert (tmp_path / "out-undistort.exr").exists()
        assert not (tmp_path / "out-redistort.exr").exists()

    def test_both_maps_when_asked(self, tmp_path):
        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        queue = RenderQueue()
        queue.add_job(RenderJob(
            RenderJobType.STMap, clip, str(out),
            options={"map_type": "both"},
        ))
        queue.process_all()
        assert queue.failures == []
        assert (tmp_path / "out-undistort.exr").exists()
        assert (tmp_path / "out-redistort.exr").exists()

    def test_the_written_exr_is_a_real_image(self, tmp_path):
        """The map loads as an EXR with the clip's dimensions — not an
        empty or placeholder payload."""
        clip = _tiny_clip(tmp_path / "in.mp4")
        out = tmp_path / "out.mp4"
        queue = RenderQueue()
        queue.add_job(RenderJob(RenderJobType.STMap, clip, str(out)))
        queue.process_all()
        with av.open(str(tmp_path / "out-undistort.exr")) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.width == 64
            assert stream.codec_context.height == 48
