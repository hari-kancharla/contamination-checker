"""Small, dependency-light statistical toolkit.

The functions here are deliberately written from first principles so that the
package does not need SciPy.  Every routine has a docstring stating the null
hypothesis or the quantity it estimates.  The module is the numerical
foundation for the evidence synthesis in :mod:`contamcheckr.calibration`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import itertools
import math
from statistics import NormalDist

import numpy as np


_EPS = 3.0e-14
_FPMIN = 1.0e-300
_NORMAL = NormalDist()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` to ``[lower, upper]``."""
    if lower > upper:
        raise ValueError("Lower bound cannot exceed upper bound.")
    return float(min(max(value, lower), upper))


def _as_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array
