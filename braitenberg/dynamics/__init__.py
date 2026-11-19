"""Nonlinear dynamics, phase portraits, Shannon ensemble connectomes, and Tri-Panel studio visuals.

See specs/dynamics.md for the mathematical model and CLI reference:
    python -m braitenberg.dynamics --help
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    _default_mpl = Path.home() / ".matplotlib"
    try:
        _default_mpl.mkdir(parents=True, exist_ok=True)
        _probe = _default_mpl / f".probe_{os.getpid()}"
        _probe.touch()
        _probe.unlink()
    except OSError:
        _fallback_mpl = Path(tempfile.gettempdir()) / "braitenberg_mpl"
        _fallback_mpl.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_fallback_mpl)

__all__ = [
    "themes",
    "diagram_mpl",
    "studio",
    "phase_portraits",
    "phase_atlas",
    "shannon",
    "phase_plane",
    "lagrangian",
    "neuro",
]
