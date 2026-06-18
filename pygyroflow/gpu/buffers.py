"""GPU buffer descriptors for the wgpu undistortion pipeline.

These dataclasses describe GPU resources needed for frame stabilization,
abstracting away the wgpu buffer creation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BufferDescription:
    """Description of a GPU buffer.

    Attributes:
        label: Human-readable label for debugging.
        size: Buffer size in bytes.
        usage: One of 'uniform', 'storage_read', 'storage_rw', 'texture'.
    """

    label: str
    size: int  # bytes
    usage: str  # 'uniform', 'storage_read', 'storage_rw', 'texture'

    def __post_init__(self) -> None:
        if self.usage not in ("uniform", "storage_read", "storage_rw", "texture"):
            raise ValueError(
                f"Invalid buffer usage '{self.usage}', "
                f"expected one of: uniform, storage_read, storage_rw, texture"
            )
        if self.size <= 0:
            raise ValueError(f"Buffer size must be positive, got {self.size}")


@dataclass
class BufferSource:
    """Source data for a GPU buffer.

    Wraps either raw bytes or a numpy array that will be uploaded
    to the GPU as a storage or uniform buffer.

    Attributes:
        data: The payload -- bytes or numpy array.
        label: Optional label for debugging.
    """

    data: bytes | np.ndarray
    label: str = ""

    def size_bytes(self) -> int:
        """Return the data size in bytes."""
        if isinstance(self.data, bytes):
            return len(self.data)
        return self.data.nbytes

    def as_bytes(self) -> bytes:
        """Convert to raw bytes for GPU upload."""
        if isinstance(self.data, bytes):
            return self.data
        return self.data.tobytes()
