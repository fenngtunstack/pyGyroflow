"""Background workers for the GUI (QThread-based).

Keeps long operations (render, auto-sync) off the UI thread; the previous
synchronous calls froze the whole window for the duration.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from PySide6.QtCore import QThread, Signal

    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover - GUI optional in headless CI
    _QT_AVAILABLE = False


if _QT_AVAILABLE:

    class RenderWorker(QThread):
        """Render a stabilized video off the UI thread."""

        finished_ok = Signal(str)
        finished_err = Signal(str)

        def __init__(self, manager, input_path: str, output_path: str) -> None:
            super().__init__()
            self._manager = manager
            self._input = input_path
            self._output = output_path

        def run(self) -> None:  # pragma: no cover - exercised via GUI
            try:
                self._manager.render(self._input, self._output)
                self.finished_ok.emit(self._output)
            except Exception as exc:  # noqa: BLE001 - reported via signal
                log.exception("Render worker failed")
                self.finished_err.emit(str(exc))

    class SyncWorker(QThread):
        """Run auto-sync (optical flow + RS offset search) off the UI thread."""

        finished_ok = Signal(object)  # float | None (offset in ms)
        finished_err = Signal(str)

        def __init__(self, manager) -> None:
            super().__init__()
            self._manager = manager

        def run(self) -> None:  # pragma: no cover - exercised via GUI
            try:
                offset = self._manager.synchronize()
                self.finished_ok.emit(offset)
            except Exception as exc:  # noqa: BLE001 - reported via signal
                log.exception("Sync worker failed")
                self.finished_err.emit(str(exc))

else:  # pragma: no cover
    class RenderWorker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PySide6 is required for GUI workers")

    class SyncWorker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PySide6 is required for GUI workers")
