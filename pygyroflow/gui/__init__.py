"""PyGyroFlow GUI — PySide6-based video stabilization interface.

Provides a Gyroflow-like Qt interface for loading video, configuring
stabilization parameters, lens profiles, and rendering output.

Usage::

    from pygyroflow.gui import MainWindow

    app = MainWindow.create_app()
    window = MainWindow()
    window.show()
    app.exec()
"""

try:
    from pygyroflow.gui.main_window import MainWindow
except ImportError as exc:
    raise ImportError(
        f"PySide6 is required for the GUI: {exc}. "
        "Install it with: pip install PySide6"
    ) from exc

__all__ = ["MainWindow"]
