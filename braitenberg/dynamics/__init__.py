"""Nonlinear dynamics, phase portraits, Lagrangian coherent structures, and studio visuals.

See specs/dynamics.md for the mathematical model and CLI reference:
    python -m braitenberg.dynamics --help
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    _default_mpl = Path.home() / ".matplotlib"
    if not os.access(_default_mpl if _default_mpl.exists() else Path.home(), os.W_OK):
        _fallback_mpl = Path(tempfile.gettempdir()) / "braitenberg_mpl"
        _fallback_mpl.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_fallback_mpl)

__all__ = [
    "themes",
    "studio",
    "phase_plane",
    "lagrangian",
    "neuro",
]
