"""Trim ranges at render time (gap item C-04).

`params.trim_ranges` used to drive the smoothing window and nothing else —
every render wrote the whole clip regardless. Upstream feeds the ranges to
the processor as `ranges_ms`, which seeks to each range's start, cuts at its
end and rebases the output timestamps so the kept spans end up back-to-back
(rendering/mod.rs, ffmpeg_processor.rs).

These tests render real files: frame counts and pixel content, not mocks.

A frame's identity is written into it as four black-or-white quadrants (two
in red, two in blue, one bit each). A flat colour ramp would survive H.264
fine but not the stabilization warp, which samples a little outside the
frame; quadrants a quarter of the image tall are robust to that.
"""

import fractions

import numpy as np
import pytest

av = pytest.importorskip("av")

from pygyroflow.rendering.ffmpeg_processor import (  # noqa: E402
    normalise_ranges,
    output_path_for_range,
    split_range_ms,
)

FPS = 30.0
FRAMES = 12
# 12 frames at 30 fps.
DURATION_MS = FRAMES * 1000.0 / FPS  # 400.0

# (bit, channel) for each of the four identity quadrants.
_BITS = [(0, 0), (1, 0), (2, 2), (3, 2)]
_GRAY = 128


def write_source(path, frames=FRAMES, fps=FPS, width=64, height=48, audio=False):
    """A clip whose frames carry their own index."""
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=int(fps))
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    # The audio stream has to exist before the first mux: the container
    # header is written then, and a stream added later has a zero time base
    # and cannot be muxed ("Cannot rebase to zero time").
    audio_stream = None
    if audio:
        audio_stream = container.add_stream("aac", rate=48000)
        audio_stream.layout = "stereo"
    half = height // 2
    for i in range(frames):
        img = np.full((height, width, 3), _GRAY, np.uint8)
        for bit, channel in _BITS:
            rows = slice(0, half) if bit % 2 == 0 else slice(half, height)
            img[rows, :, channel] = 255 if (i >> bit) & 1 else 0
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in video.encode(frame):
            container.mux(pkt)
    for pkt in video.encode():
        container.mux(pkt)

    if audio_stream is not None:
        # AAC wants a fixed 1024-sample frame, so the tail is padded rather
        # than shortened — a short final frame is rejected by the encoder.
        total = int(48000 * frames / fps)
        for index in range((total + 1023) // 1024):
            t = (np.arange(1024) + index * 1024) / 48000.0
            tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            aframe = av.AudioFrame.from_ndarray(
                np.stack([tone, tone]), format="fltp", layout="stereo"
            )
            aframe.sample_rate = 48000
            aframe.pts = index * 1024
            aframe.time_base = fractions.Fraction(1, 48000)
            for pkt in audio_stream.encode(aframe):
                container.mux(pkt)
        for pkt in audio_stream.encode():
            container.mux(pkt)

    container.close()


def frame_index_of(img: np.ndarray) -> int:
    """Decode the identity a fixture frame was written with."""
    half = img.shape[0] // 2
    index = 0
    for bit, channel in _BITS:
        rows = slice(0, half) if bit % 2 == 0 else slice(half, img.shape[0])
        if float(img[rows, :, channel].mean()) > 128.0:
            index |= 1 << bit
    return index


def frame_count(path):
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def frame_pts(path):
    """The presentation timestamp of every decoded frame, in order."""
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            out.append(int(frame.pts))
    return out


def frame_indices(path):
    """The source index of every decoded frame, in order."""
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            out.append(frame_index_of(frame.to_ndarray(format="rgb24")))
    return out


def stream_duration_ms(path, kind="audio"):
    with av.open(str(path)) as container:
        streams = container.streams.audio if kind == "audio" else container.streams.video
        if not streams:
            return 0.0
        if streams[0].duration is not None:
            return float(streams[0].duration * streams[0].time_base) * 1000.0
        # Some containers carry no per-stream duration; fall back to the
        # container's own figure.
        return float(container.duration) / 1000.0


def new_manager(with_gyro=True):
    """A manager sized to the fixture clip.

    With *with_gyro* a synthetic rotating IMU is injected. With no
    quaternions at all ``cpu_undistort`` returns an empty frame, so a render
    needs a gyro timeline even though these tests are about trimming.
    """
    from pygyroflow.manager import StabilizationManager

    mgr = StabilizationManager()
    mgr.init_from_video_data(DURATION_MS, FPS, FRAMES, (64, 48))
    mgr.set_size(64, 48)
    mgr.set_output_size(64, 48)
    if with_gyro:
        from pygyroflow.gyro_source import FileMetadata
        from pygyroflow.types.time_types import TimeIMU

        md = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
        md.raw_imu = [
            TimeIMU(timestamp_ms=i * 5.0, gyro=np.array([0.0, 0.0, 20.0]), accl=None)
            for i in range(int(DURATION_MS / 5.0) + 1)
        ]
        mgr.gyro.init_from_params(mgr.params.get_scaled_duration_ms())
        mgr.gyro.load_from_telemetry(md)
        mgr.recompute_blocking()
    return mgr


def render(tmp_path, name, options=None, ranges=None):
    """Render the standard source with *ranges* and return the output path."""
    src = tmp_path / "src.mp4"
    if not src.exists():
        write_source(src)
    out = tmp_path / name
    mgr = new_manager()
    mgr.params.trim_ranges = list(ranges or [])
    mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": False, **(options or {})})
    return out


# ----------------------------------------------------------------------
# Unit level
# ----------------------------------------------------------------------


class TestNormaliseRanges:
    def test_empty(self):
        assert normalise_ranges([], DURATION_MS) == []
        assert normalise_ranges(None, DURATION_MS) == []

    def test_full_clip_becomes_all_none(self):
        """Both bounds at their extremes mean "no cut and no seek".

        Collapsing them to 0.0/1.0 instead would tell the processor to cut
        at the very last millisecond of the clip.
        """
        assert normalise_ranges([(0.0, 1.0)], DURATION_MS) == [(None, None)]

    def test_partial_becomes_milliseconds(self):
        assert normalise_ranges([(0.25, 0.75)], DURATION_MS) == [(100.0, 300.0)]

    def test_half_open_ends(self):
        assert normalise_ranges([(0.0, 0.5)], DURATION_MS) == [(None, 200.0)]
        assert normalise_ranges([(0.5, 1.0)], DURATION_MS) == [(200.0, None)]

    def test_each_range_independently(self):
        assert normalise_ranges(
            [(0.0, 0.25), (0.75, 1.0)], DURATION_MS
        ) == [(None, 100.0), (300.0, None)]

    def test_split_range_keeps_concrete_bounds(self):
        assert split_range_ms([(0.0, 0.5)], DURATION_MS) == [(0.0, 200.0)]


class TestOutputPathForRange:
    @pytest.mark.parametrize(
        "path,index,expected",
        [
            ("out.mp4", 0, "out-001.mp4"),
            ("out.mp4", 1, "out-002.mp4"),
            ("/tmp/dir/out.mp4", 11, "/tmp/dir/out-012.mp4"),
            ("out", 0, "out-001"),
            ("seq/frame_%05d.png", 0, "seq/frame_%05d-001.png"),
        ],
    )
    def test_naming(self, path, index, expected):
        """Upstream inserts `-{:0>3}` before the extension."""
        assert output_path_for_range(path, index) == expected


# ----------------------------------------------------------------------
# Frame selection
# ----------------------------------------------------------------------


class TestFrameSelection:
    """Frame i sits at i*33.333 ms, and a frame is kept while ts <= end."""

    @pytest.mark.parametrize(
        "ranges,expected",
        [
            ([], 12),
            ([(0.0, 1.0)], 12),
            ([(0.25, 0.75)], 7),      # ts <= 300 -> frames 3..9
            ([(0.0, 0.5)], 7),        # ts <= 200 -> frames 0..6
            ([(0.5, 1.0)], 6),        # ts >= 200 -> frames 6..11
            ([(0.25, 0.5)], 4),       # 100 <= ts <= 200 -> frames 3..6
            ([(0.0, 0.25), (0.75, 1.0)], 7),   # frames 0..3 then 9..11
            ([(0.25, 0.3), (0.4, 0.5)], 3),    # frame 3, then frames 5 and 6
        ],
    )
    def test_kept_frame_count(self, tmp_path, ranges, expected):
        out = render(tmp_path, "out.mp4", ranges=ranges)
        assert frame_count(out) == expected

    def test_renders_everything_without_ranges(self, tmp_path):
        """The pre-existing behaviour has to survive untouched."""
        out = render(tmp_path, "out.mp4")
        assert frame_count(out) == FRAMES

    def test_output_has_no_gap_where_a_range_was_cut(self, tmp_path):
        """Two ranges must land back-to-back, not 200 ms of nothing.

        Asserted on the output pts rather than on a duration: evenly spaced
        pts is exactly "no gap", and it does not depend on how the muxer
        rounds the track duration. The spacing itself is in the encoder's
        own time base (1/15360 for mp4 here), so only the deltas are
        meaningful.
        """
        out = render(tmp_path, "out.mp4", ranges=[(0.0, 0.25), (0.75, 1.0)])
        pts = frame_pts(out)
        assert len(pts) == 7
        deltas = {b - a for a, b in zip(pts, pts[1:])}
        assert len(deltas) == 1, pts
        assert min(pts) == 0

    def test_content_comes_from_the_kept_ranges(self, tmp_path):
        """Frame identity, not just the count: the head then the tail."""
        out = render(tmp_path, "out.mp4", ranges=[(0.0, 0.25), (0.75, 1.0)])
        assert frame_indices(out) == [0, 1, 2, 3] + [9, 10, 11]

    def test_a_middle_range_keeps_middle_content(self, tmp_path):
        out = render(tmp_path, "out.mp4", ranges=[(0.5, 1.0)])
        assert frame_indices(out) == [6, 7, 8, 9, 10, 11]


class TestCallbackSeesTheSourceTimeline:
    """A kept frame's transform has to come from where it sat originally.

    If the callback were handed the output position instead, every frame
    after a cut would be stabilized with its neighbour's transform — the
    kind of error that produces a plausible-looking but wrong render.
    """

    def test_timestamps_and_indices_are_the_original_ones(self, tmp_path):
        src = tmp_path / "src.mp4"
        write_source(src)

        seen = []
        mgr = new_manager()
        mgr.params.trim_ranges = [(0.25, 0.75)]

        from pygyroflow.rendering import FfmpegProcessor

        proc = FfmpegProcessor()
        proc.open_input(str(src))
        proc.create_output(
            str(tmp_path / "out.mp4"), 64, 48, 30.0, codec="H.264/AVC"
        )
        proc.process_frames(
            lambda img, ts, idx: (seen.append((ts, idx)) or img),
            ranges_ms=normalise_ranges(mgr.params.trim_ranges, DURATION_MS),
        )
        proc.close()

        assert [idx for _, idx in seen] == [3, 4, 5, 6, 7, 8, 9]
        for ts, idx in seen:
            assert ts == pytest.approx(idx * 1000.0 / FPS, abs=0.1)


# ----------------------------------------------------------------------
# Per-range export
# ----------------------------------------------------------------------


class TestSeparateRangeExport:
    def test_one_file_per_range(self, tmp_path):
        src = tmp_path / "src.mp4"
        write_source(src)
        mgr = new_manager()
        mgr.params.trim_ranges = [(0.0, 0.25), (0.75, 1.0)]
        mgr.render(
            str(src),
            str(tmp_path / "out.mp4"),
            {"codec": "H.264/AVC", "audio": False, "export_trims_separately": True},
        )

        first = tmp_path / "out-001.mp4"
        second = tmp_path / "out-002.mp4"
        assert first.exists() and second.exists()
        assert not (tmp_path / "out.mp4").exists()
        assert frame_count(first) == 4
        assert frame_count(second) == 3

    def test_single_range_is_not_split(self, tmp_path):
        """Upstream only rewrites the name when there is more than one."""
        src = tmp_path / "src.mp4"
        write_source(src)
        mgr = new_manager()
        mgr.params.trim_ranges = [(0.25, 0.75)]
        mgr.render(
            str(src),
            str(tmp_path / "out.mp4"),
            {"codec": "H.264/AVC", "audio": False, "export_trims_separately": True},
        )
        assert (tmp_path / "out.mp4").exists()
        assert frame_count(tmp_path / "out.mp4") == 7


# ----------------------------------------------------------------------
# The flags that disable trimming
# ----------------------------------------------------------------------


class TestTrimmingDisabled:
    @pytest.mark.parametrize("flag", ["pad_with_black", "preserve_other_tracks"])
    def test_keeps_the_whole_clip(self, tmp_path, flag):
        """Upstream gates ranges_ms on exactly these two flags."""
        out = render(tmp_path, "out.mp4", options={flag: True}, ranges=[(0.25, 0.75)])
        assert frame_count(out) == FRAMES


# ----------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------


class TestAudioTrim:
    def _render(self, tmp_path, ranges, name):
        src = tmp_path / "av.mp4"
        write_source(src, audio=True)
        out = tmp_path / name
        mgr = new_manager()
        mgr.params.trim_ranges = list(ranges)
        mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": True})
        return out

    def test_untrimmed_audio_keeps_its_length(self, tmp_path):
        out = self._render(tmp_path, [], "full.mp4")
        assert stream_duration_ms(out, "audio") == pytest.approx(400.0, abs=60.0)

    def test_trimmed_audio_is_cut_too(self, tmp_path):
        """A trimmed render must not carry the audio of the parts it cut."""
        out = self._render(tmp_path, [(0.0, 0.5)], "half.mp4")
        duration = stream_duration_ms(out, "audio")
        assert duration == pytest.approx(200.0, abs=60.0)
        assert duration < 320.0

    def test_audio_matches_the_video_length(self, tmp_path):
        """Both tracks are trimmed the same way, so they stay in step."""
        out = self._render(tmp_path, [(0.0, 0.5)], "half.mp4")
        audio_ms = stream_duration_ms(out, "audio")
        video_ms = frame_count(out) * 1000.0 / FPS
        assert abs(audio_ms - video_ms) <= 60.0


class TestAudioRebaser:
    """Unit-level check of the timestamp fold the stream-copy path uses."""

    @staticmethod
    def _rebaser(ranges):
        from pygyroflow.rendering.audio_resampler import _make_rebaser

        return _make_rebaser(ranges)

    def test_no_ranges_is_a_passthrough(self):
        keep, rebase = self._rebaser(None)
        assert keep is None and rebase is None

    def test_single_range_shifts_to_zero(self):
        keep, rebase = self._rebaser([(100.0, 300.0)])
        assert keep(99.0) is False
        assert keep(100.0) is True
        assert keep(300.0) is True
        assert keep(300.1) is False
        assert rebase(100.0) == pytest.approx(0.0)
        assert rebase(250.0) == pytest.approx(150.0)

    def test_second_range_starts_where_the_first_ended(self):
        keep, rebase = self._rebaser([(0.0, 100.0), (300.0, 400.0)])
        assert keep(100.0) is True
        assert keep(100.1) is False
        assert keep(299.9) is False
        assert keep(300.0) is True
        # Range 0 contributes 100 ms, so range 1 starts at 100 ms.
        assert rebase(300.0) == pytest.approx(100.0)
        assert rebase(400.0) == pytest.approx(200.0)

    def test_open_ended_range(self):
        keep, rebase = self._rebaser([(200.0, None)])
        assert keep(10.0) is False
        assert keep(10_000.0) is True
        assert rebase(200.0) == pytest.approx(0.0)
        assert rebase(250.0) == pytest.approx(50.0)
