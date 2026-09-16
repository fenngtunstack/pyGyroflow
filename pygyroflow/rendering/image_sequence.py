# -*- coding: utf-8 -*-
"""Image sequence input — resolve a directory / pattern / single frame to an
FFmpeg ``image2`` input.

Upstream Gyroflow treats image sequences as an FFmpeg demuxer concern: the
input is a printf path (``/shots/frame%04d.exr``) plus two ``avformat``
options, ``start_number`` and ``framerate``.  Nothing else in the pipeline
changes — the frames decode into the same RGBA/planar buffers as a video.

This module reproduces that: it normalises the three natural ways a user
points at a sequence into the pattern + options FFmpeg wants.

* ``/shots/frame_%04d.exr`` — used as-is (the padding width and start number
  are still verified against the files on disk)
* ``/shots/frame_0001.exr`` — the trailing digit run gives padding and start
* ``/shots/``               — the first frame in natural sort order is taken
  as the template for the series

Sequences carry no frame rate (EXR and PNG have no notion of time), so the
rate must come from the caller; FFmpeg silently assumes 25 fps otherwise,
which would put the gyro timeline at the wrong speed.  :func:`format_options`
is where that decision is made explicit.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from pygyroflow.types.errors import VideoIOError

# Extensions FFmpeg's image2 demuxer can open.  Deliberately a list rather
# than "anything that is not a video": a stray .txt or .json in the folder
# must not be mistaken for a frame.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dpx", ".ppm", ".pgm", ".webp"}
)

# FFmpeg's own default when no framerate option is given.
FFMPEG_DEFAULT_FPS = 25.0

_NUMBERED = re.compile(r"^(?P<prefix>.*?)(?P<digits>\d+)$")


@dataclass(frozen=True)
class ImageSequence:
    """A resolved image sequence (or a single still)."""

    pattern: str
    directory: str
    start_number: int
    frame_count: int
    pad_width: int
    extension: str
    is_sequence: bool

    @property
    def first_file(self) -> str:
        """Path of the first frame (only meaningful for a single still)."""
        return self.pattern


def is_image_file(path: str) -> bool:
    """True when *path* looks like a still image by extension."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def looks_like_image_sequence(path: str) -> bool:
    """True when *path* points at an image sequence rather than a video.

    Accepts a printf pattern, a single image file, or a directory whose first
    entry is an image.
    """
    if "%" in os.path.basename(path) and is_image_file(path.replace("%", "0", 1)):
        # e.g. frame_%04d.exr -> frame_0000.exr has a known extension
        return True
    if is_image_file(path):
        return True
    return os.path.isdir(path) and _first_image_in(path) is not None


def resolve_image_sequence(path: str) -> ImageSequence:
    """Normalise *path* into a pattern FFmpeg's ``image2`` demuxer accepts.

    Raises:
        FileNotFoundError: no usable image was found.
        VideoIOError: the frames are not a contiguous run — FFmpeg's image2
            demuxer aborts mid-stream on a gap, so it is better to say so
            before opening the sequence than to fail several frames in.
    """
    if os.path.isdir(path):
        template = _first_image_in(path)
        if template is None:
            raise FileNotFoundError(f"No image files in directory: {path}")
    else:
        template = path
        if not is_image_file(template):
            raise FileNotFoundError(f"Not an image file: {path}")
        if "%" in template:
            return _resolve_pattern(template)

    directory = os.path.dirname(os.path.abspath(template))
    basename = os.path.basename(template)
    stem, extension = os.path.splitext(basename)

    match = _NUMBERED.match(stem)
    if match is None:
        if not os.path.isfile(template):
            raise FileNotFoundError(f"File not found: {template}")
        # A single still: image2 opens it by literal name.
        return ImageSequence(
            pattern=template,
            directory=directory,
            start_number=0,
            frame_count=1,
            pad_width=0,
            extension=extension.lower(),
            is_sequence=False,
        )

    prefix = match.group("prefix")
    digits = match.group("digits")
    pattern = os.path.join(directory, f"{prefix}%0{len(digits)}d{extension}")
    return _resolve_pattern(pattern, extension=extension)


def _resolve_pattern(pattern: str, extension: str | None = None) -> ImageSequence:
    """Verify a printf pattern against the files on disk."""
    directory = os.path.dirname(os.path.abspath(pattern))
    basename = os.path.basename(pattern)
    extension = extension or os.path.splitext(basename)[1]
    stem = os.path.splitext(basename)[0]
    match = re.search(r"%0(\d+)d", stem)
    if match is None:
        raise FileNotFoundError(f"Not a printf image pattern: {pattern}")
    pad_width = int(match.group(1))
    prefix = stem[: match.start()]

    numbered: dict[int, str] = {}
    for entry in os.listdir(directory):
        if not entry.startswith(prefix) or not entry.lower().endswith(extension.lower()):
            continue
        tail = os.path.splitext(entry[len(prefix):])[0]
        if len(tail) == pad_width and tail.isdigit():
            numbered[int(tail)] = entry
    if not numbered:
        raise FileNotFoundError(f"No frames match {pattern}")

    longest = max(numbered)
    start = min(numbered)
    count = longest - start + 1
    missing = [i for i in range(start, longest + 1) if i not in numbered]
    if missing:
        raise VideoIOError(
            f"Image sequence has a gap: {len(missing)} missing frame(s), first is "
            f"{prefix}{missing[0]:0{pad_width}d}{extension} in {directory}. "
            "FFmpeg's image2 demuxer aborts on a missing frame — renumber the "
            "sequence or fill the gap."
        )

    return ImageSequence(
        pattern=pattern,
        directory=directory,
        start_number=start,
        frame_count=count,
        pad_width=pad_width,
        extension=extension.lower(),
        is_sequence=True,
    )


def format_options(sequence: ImageSequence, fps: float | None = None) -> dict[str, str]:
    """Build the ``avformat`` options that drive the image2 demuxer.

    ``start_number`` must match the first file on disk: FFmpeg silently starts
    at the requested index and stops at the first missing file, so an
    off-by-one quietly drops the leading frame and shifts the whole clip
    against the gyro timeline.

    ``fps`` becomes ``framerate``.  When it is missing, FFmpeg assumes 25 fps;
    the caller is expected to have already warned (see
    :func:`pygyroflow.cli.main`).
    """
    options: dict[str, str] = {}
    if sequence.is_sequence:
        options["start_number"] = str(sequence.start_number)
    effective = fps if fps and fps > 0 else FFMPEG_DEFAULT_FPS
    options["framerate"] = fps_to_rational(effective)
    return options


def sequence_output_stem(path: str) -> str:
    """Derive an output basename stem for a sequence *path*.

    A trailing ``.exr`` in a pattern would otherwise be mistaken for the
    output extension (``shots/frame_%04d.exr`` -> ``shots/frame_%04d_stabilized.mp4``
    rather than ``shots/frame_%04d.exr_stabilized.mp4``), and a directory
    would get written into the frame folder.
    """
    if os.path.isdir(path):
        stripped = os.path.abspath(path.rstrip(os.sep))
        return os.path.join(os.path.dirname(stripped), os.path.basename(stripped))
    stem = re.sub(r"%0?\d*d", "", os.path.splitext(path)[0]).rstrip("_-. ")
    return stem or os.path.splitext(path)[0]


def fps_to_rational(fps: float) -> str:
    """Render *fps* as a rational FFmpeg accepts.

    Port of upstream ``rendering::fps_to_rational``: rates with a fractional
    part above 0.1 are expressed over 1001 (the NTSC family — 29.97 becomes
    ``30000/1001``), the rest are rounded to whole numbers.  Decimals like
    ``29.97`` are close enough to NTSC that FFmpeg's own containers use the
    ``/1001`` form, and the demuxer rate has to match the material.
    """
    if abs(fps) - math.floor(abs(fps)) > 0.1:  # upstream: fps.fract()
        return f"{round(abs(fps) * 1001)}/1001"
    return str(round(abs(fps)))


def _first_image_in(directory: str) -> str | None:
    """First image in *directory* under natural (numeric-aware) ordering."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    images = [e for e in entries if not e.startswith(".") and is_image_file(e)]
    if not images:
        return None
    images.sort(key=_natural_key)
    return os.path.join(directory, images[0])


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]
