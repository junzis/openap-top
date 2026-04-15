"""Shared type aliases for opentop.

Re-exported from modules that need polymorphic numeric / symbolic types.
"""
from __future__ import annotations

from typing import Callable, Union

import casadi as ca
import numpy as np

# Lat/Lon pair used as airport designator
LatLon = tuple[float, float]

# Polymorphic numeric / symbolic type: numbers in the pure path, CasADi in the NLP path.
Symbolic = Union[ca.MX, ca.SX, ca.DM, np.ndarray, float]

# Objective function signature. Takes (x, u, dt, **kwargs), returns a CasADi expression.
ObjectiveFn = Callable[..., ca.MX]
