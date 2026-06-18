"""Render job queue for batch video stabilisation.

Manages a queue of render jobs (video, metadata, ST-map, project) and
processes them sequentially, reporting progress through callbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger(__name__)


class RenderJobType(Enum):
    """Type of render job."""

    Video = 0
    Metadata = 1
    STMap = 2
    Project = 3


@dataclass
class RenderJob:
    """A single render job in the queue.

    Attributes:
        job_type: What kind of output to produce.
        input_path: Source video file path.
        output_path: Destination file path.
        options: Codec, bitrate, and other rendering options.
        progress_callback: Called with progress fraction (0.0 .. 1.0).
    """

    job_type: RenderJobType
    input_path: str
    output_path: str
    options: dict = field(default_factory=dict)
    progress_callback: Optional[Callable[[float], None]] = None


class RenderQueue:
    """Sequential render queue for batch stabilisation jobs.

    Usage::

        q = RenderQueue()
        q.add_job(RenderJob(RenderJobType.Video, "in.mp4", "out.mp4"))
        q.process_all()
    """

    def __init__(self) -> None:
        self._jobs: list[RenderJob] = []
        self._current_index: int = 0
        self._current_progress: float = 0.0

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def add_job(self, job: RenderJob) -> None:
        """Append a job to the end of the queue."""
        self._jobs.append(job)

    def remove_job(self, index: int) -> RenderJob:
        """Remove and return the job at *index*."""
        return self._jobs.pop(index)

    @property
    def jobs(self) -> list[RenderJob]:
        """Return a shallow copy of the job list."""
        return list(self._jobs)

    @property
    def current_index(self) -> int:
        """Index of the next job to process."""
        return self._current_index

    @property
    def progress(self) -> float:
        """Overall queue progress as a fraction (0.0 .. 1.0)."""
        if not self._jobs:
            return 1.0
        per_job = 1.0 / len(self._jobs)
        base = self._current_index * per_job
        return min(base + self._current_progress * per_job, 1.0)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_next(self) -> bool:
        """Process the next pending job.

        Returns:
            True if a job was processed, False if the queue is empty.
        """
        if self._current_index >= len(self._jobs):
            return False

        job = self._jobs[self._current_index]
        self._current_progress = 0.0

        try:
            if job.job_type == RenderJobType.Video:
                self._process_video_job(job)
            elif job.job_type == RenderJobType.Metadata:
                self._process_metadata_job(job)
            elif job.job_type == RenderJobType.STMap:
                self._process_stmap_job(job)
            elif job.job_type == RenderJobType.Project:
                self._process_project_job(job)
            else:
                log.warning("Unknown job type: %s", job.job_type)
        except Exception:
            log.error(
                "Render job failed: %s -> %s",
                job.input_path,
                job.output_path,
                exc_info=True,
            )

        self._current_index += 1
        self._current_progress = 1.0
        return True

    def process_all(self) -> None:
        """Process all remaining jobs in the queue."""
        while self.process_next():
            pass

    # ------------------------------------------------------------------
    # Job processors (stubs -- wired to actual stabilisation logic)
    # ------------------------------------------------------------------

    def _process_video_job(self, job: RenderJob) -> None:
        """Process a video stabilisation job."""
        from pygyroflow.rendering.ffmpeg_processor import FfmpegProcessor

        proc = FfmpegProcessor()
        try:
            info = proc.open_input(job.input_path)

            codec = job.options.get("codec", "H.265/HEVC")
            bitrate = job.options.get("bitrate", 0.0)
            out_w = job.options.get("output_width", info["width"])
            out_h = job.options.get("output_height", info["height"])

            proc.create_output(
                job.output_path, out_w, out_h, info["fps"], codec, bitrate
            )

            # If a stabilise callback was provided in options, use it.
            # Otherwise pass frames through unchanged.
            stabilise = job.options.get("frame_callback", _passthrough)

            total = max(info.get("frames", 1), 1)

            def _tracked_callback(
                frame, timestamp_ms: float, frame_idx: int
            ):
                result = stabilise(frame, timestamp_ms, frame_idx)
                self._current_progress = (frame_idx + 1) / total
                if job.progress_callback:
                    job.progress_callback(self._current_progress)
                return result

            proc.process_frames(_tracked_callback)
        finally:
            proc.close()

    def _process_metadata_job(self, job: RenderJob) -> None:
        """Export stabilisation metadata (gyro data, lens profile, etc.)."""
        import json

        data = job.options.get("metadata", {})
        with open(job.output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _process_stmap_job(self, job: RenderJob) -> None:
        """Generate an ST-map from the current stabilisation parameters."""
        log.info("ST-map export: %s", job.output_path)
        # Actual ST-map generation is a future feature.
        # For now write a placeholder so the pipeline doesn't break.
        import numpy as np

        w = job.options.get("width", 1920)
        h = job.options.get("height", 1080)
        stmap = np.zeros((h, w, 2), dtype=np.float32)
        # Identity map as placeholder.
        ys, xs = np.mgrid[0:h, 0:w]
        stmap[:, :, 0] = xs.astype(np.float32) / w
        stmap[:, :, 1] = ys.astype(np.float32) / h
        stmap.tofile(job.output_path)

    def _process_project_job(self, job: RenderJob) -> None:
        """Save / export a full Gyroflow-compatible project file."""
        import json

        data = job.options.get("project_data", {})
        with open(job.output_path, "w") as f:
            json.dump(data, f, indent=2)


def _passthrough(frame, _ts: float, _idx: int):
    """Default frame callback that returns frames unchanged."""
    return frame
