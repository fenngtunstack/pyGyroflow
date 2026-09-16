"""Video speed and frame-rate scaling at render time (gap item C-05).

`video_speed` and `fps_scale` used to reach the smoothing and zooming maths
and nothing else: every render wrote one output frame per input frame. They
are two different mechanisms and both are covered here.

* ``video_speed`` changes how many frames come out — up to 1 drops them,
  below 1 duplicates them. Upstream drives it through ``rate_control``.
* ``fps_scale`` changes which frame of the gyro timeline each frame is looked
  up against (a 240 fps recording written into a 60 fps container), and
  leaves the frame count alone. Upstream divides ``timestamp_us`` by it.

The tests render real files and count real frames.
"""

from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av")

from pygyroflow.rendering.ffmpeg_processor import FrameRateControl  # noqa: E402

FPS = 30.0
FRAMES = 12
DURATION_MS = FRAMES * 1000.0 / FPS  # 400.0

_BITS = [(0, 0), (1, 0), (2, 2), (3, 2)]


def write_clip(path, frames=FRAMES, fps=FPS, width=64, height=48):
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=int(fps))
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    half = height // 2
    for i in range(frames):
        img = np.full((height, width, 3), 128, np.uint8)
        for bit, channel in _BITS:
            rows = slice(0, half) if bit % 2 == 0 else slice(half, height)
            img[rows, :, channel] = 255 if (i >> bit) & 1 else 0
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for pkt in video.encode(frame):
            container.mux(pkt)
    for pkt in video.encode():
        container.mux(pkt)
    container.close()


def frame_count(path):
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def frame_indices(path):
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            half = img.shape[0] // 2
            index = 0
            for bit, channel in _BITS:
                rows = slice(0, half) if bit % 2 == 0 else slice(half, img.shape[0])
                if float(img[rows, :, channel].mean()) > 128.0:
                    index |= 1 << bit
            out.append(index)
    return out


def new_manager(**params):
    from pygyroflow.gyro_source import FileMetadata
    from pygyroflow.manager import StabilizationManager
    from pygyroflow.types.time_types import TimeIMU

    mgr = StabilizationManager()
    mgr.init_from_video_data(DURATION_MS, FPS, FRAMES, (64, 48))
    mgr.set_size(64, 48)
    mgr.set_output_size(64, 48)

    metadata = FileMetadata(detected_source="Synthetic", imu_orientation="XYZ")
    metadata.raw_imu = [
        TimeIMU(
            timestamp_ms=float(i),
            gyro=np.array([0.0, 0.0, 20.0]),
            accl=None,
        )
        for i in range(int(DURATION_MS) + 1)
    ]
    mgr.gyro.init_from_params(mgr.params.get_scaled_duration_ms())
    mgr.gyro.load_from_telemetry(metadata)
    for key, value in params.items():
        setattr(mgr.params, key, value)
    mgr.recompute_blocking()
    return mgr


def render(tmp_path, name, audio=False, **params):
    src = tmp_path / "src.mp4"
    if not src.exists():
        write_clip(src)
    out = tmp_path / name
    mgr = new_manager(**params)
    mgr.render(
        str(src), str(out), {"codec": "H.264/AVC", "audio": audio},
    )
    return out


def repeat_profile(speed, frames=FRAMES, fps=FPS):
    """Which frames survive, and how many copies each gets."""
    control = FrameRateControl(fps, speed)
    out = []
    for i in range(frames):
        out.append(control.repeats(i * 1000.0 / fps))
    return out


# ----------------------------------------------------------------------
# The rate controller on its own
# ----------------------------------------------------------------------


class TestFrameRateControl:
    def test_disabled_when_speed_is_none(self):
        control = FrameRateControl(FPS, None)
        assert not control.active
        assert [control.repeats(i * 1000.0 / FPS) for i in range(5)] == [1] * 5

    def test_speed_one_keeps_almost_every_frame(self):
        """1.0 is not the same path as None: it still runs the gate, and the
        gate opens only once the ramped clock has advanced half an interval,
        so the first frame of the clip is always dropped. That phase is
        upstream's (rendering/mod.rs) — this pins it rather than hiding it.
        """
        profile = repeat_profile(1.0)
        assert profile[0] == 0
        assert all(p == 1 for p in profile[1:])
        assert sum(profile) == FRAMES - 1

    def test_double_speed_drops_every_other_frame(self):
        profile = repeat_profile(2.0)
        assert sum(profile) == FRAMES // 2
        assert [i for i, n in enumerate(profile) if n] == [1, 3, 5, 7, 9, 11]

    def test_half_speed_duplicates_frames(self):
        profile = repeat_profile(0.5)
        assert all(n == 2 for n in profile[1:])
        assert sum(profile) == (FRAMES - 1) * 2

    def test_no_frame_is_written_twice_in_a_row_at_high_speed(self):
        """Above 1.0 the contract is decimation, not repetition: a repeated
        frame would mean the temporal spacing went wrong."""
        profile = repeat_profile(3.0)
        assert sum(profile) == 4
        assert max(profile) == 1

    def test_keyframed_speed_is_read_per_frame(self):
        """A callable is queried with the frame's own timestamp."""
        seen = []

        def speed(ts):
            seen.append(ts)
            return 2.0

        control = FrameRateControl(FPS, speed)
        for i in range(4):
            control.repeats(i * 1000.0 / FPS)
        assert seen == [i * 1000.0 / FPS for i in range(4)]

    def test_non_positive_speed_falls_back_to_one(self):
        """A zero or negative speed would divide by zero or run the clock
        backwards; upstream's slider cannot produce one, but a corrupt
        project file can."""
        control = FrameRateControl(FPS, 0.0)
        assert [control.repeats(i * 1000.0 / FPS) for i in range(4)] == [0, 1, 1, 1]

    def test_zero_fps_does_not_divide_by_zero(self):
        control = FrameRateControl(0.0, 2.0)
        assert control.repeats(0.0) == 0  # first frame is always gated out
        assert control.repeats(1000.0) >= 1


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


class TestVideoSpeedRender:
    def test_default_speed_writes_every_frame(self, tmp_path):
        out = render(tmp_path, "out.mp4")
        assert frame_count(out) == FRAMES
        assert frame_indices(out) == list(range(FRAMES))

    def test_double_speed_halves_the_output(self, tmp_path):
        out = render(tmp_path, "fast.mp4", video_speed=2.0)
        assert frame_count(out) == FRAMES // 2
        # The survivors are the odd-numbered source frames, not just any six.
        assert frame_indices(out) == [1, 3, 5, 7, 9, 11]

    def test_half_speed_duplicates_frames(self, tmp_path):
        out = render(tmp_path, "slow.mp4", video_speed=0.5)
        assert frame_count(out) == (FRAMES - 1) * 2
        # Each surviving source frame appears twice, in order.
        assert frame_indices(out) == [i for i in range(1, FRAMES) for _ in range(2)]

    def test_four_times_speed_keeps_a_quarter(self, tmp_path):
        out = render(tmp_path, "v4.mp4", video_speed=4.0)
        assert frame_count(out) == FRAMES // 4

    def test_speed_change_drops_audio(self, tmp_path):
        """A stream copy cannot follow a speed change; upstream clears the
        audio codec rather than emit a drifting track."""
        src = tmp_path / "src.mp4"
        write_clip(src)
        out = tmp_path / "fast.mp4"
        mgr = new_manager(video_speed=2.0)
        mgr.render(str(src), str(out), {"codec": "H.264/AVC", "audio": True})
        with av.open(str(out)) as container:
            assert len(container.streams.audio) == 0
        assert frame_count(out) == FRAMES // 2


class TestFpsScaleRender:
    def test_frame_count_is_untouched(self, tmp_path):
        """fps_scale is about which gyro sample a frame maps to, not how many
        frames there are."""
        out = render(tmp_path, "scaled.mp4", fps_scale=2.0)
        assert frame_count(out) == FRAMES

    def test_lookup_timestamp_is_divided_by_the_scale(self, tmp_path):
        """The observable contract: the transform for a frame is looked up at
        timestamp / scale, so a 240 fps recording in a 60 fps container lines
        up with its gyro."""
        seen = []

        def spy(self, timestamp_ms, frame, compute_params=None):
            seen.append(timestamp_ms)
            return original(self, timestamp_ms, frame, compute_params)

        from pygyroflow.manager import StabilizationManager

        original = StabilizationManager.get_frame_transform
        src = tmp_path / "src.mp4"
        write_clip(src)
        mgr = new_manager(fps_scale=2.0)
        StabilizationManager.get_frame_transform = spy
        try:
            mgr.render(
                str(src), str(tmp_path / "scaled.mp4"),
                {"codec": "H.264/AVC", "audio": False},
            )
        finally:
            StabilizationManager.get_frame_transform = original

        # Frame i sits at i * 33.333 ms and must be looked up at half that.
        assert seen == pytest.approx([i * 1000.0 / FPS / 2.0 for i in range(FRAMES)])

    def test_render_matches_an_unscaled_run_of_half_the_rate(self, tmp_path):
        """Same thing said in pixels: halving the lookup times is the same as
        playing the clip at half speed against the same gyro."""
        src = tmp_path / "src.mp4"
        write_clip(src)
        scaled = new_manager(fps_scale=2.0)
        scaled.render(
            str(src), str(tmp_path / "scaled.mp4"),
            {"codec": "H.264/AVC", "audio": False},
        )
        # A clip declared at 15 fps puts 30 fps content at twice the spacing,
        # which is the same lookup times as fps_scale=2 on the 30 fps clip.
        slower = new_manager()
        slower.params.fps = FPS / 2.0
        slower.recompute_blocking()
        slower.render(
            str(src), str(tmp_path / "slow.mp4"),
            {"codec": "H.264/AVC", "audio": False},
        )

        def first_frame(path):
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    return frame.to_ndarray(format="rgb24")
            raise AssertionError("no frames")

        a, b = first_frame(tmp_path / "scaled.mp4"), first_frame(tmp_path / "slow.mp4")
        assert a.shape == b.shape
        assert float(np.abs(a.astype(int) - b.astype(int)).mean()) < 6.0


class TestSpeedAndTrimTogether:
    def test_trim_first_then_speed(self, tmp_path):
        """Both filters run in the same loop, and the trim has to come first:
        a frame outside every range is gone regardless of the speed."""
        out = render(
            tmp_path, "both.mp4", video_speed=2.0,
        )
        assert frame_count(out) == FRAMES // 2

        src = tmp_path / "src.mp4"
        mgr = new_manager(video_speed=2.0)
        mgr.params.trim_ranges = [(0.0, 0.5)]
        trimmed = tmp_path / "trimmed.mp4"
        mgr.render(
            str(src), str(trimmed), {"codec": "H.264/AVC", "audio": False}
        )
        # ts <= 200 keeps frames 0..6; double speed then drops every other
        # one of those, starting with the first.
        assert frame_indices(trimmed) == [1, 3, 5]
