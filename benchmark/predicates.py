"""Predicate helpers for LCL experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


@dataclass
class IntervalPredicate:
    a: float
    b: float

    def __call__(self, x: float) -> bool:
        return self.a < x < self.b


def adaptive_interval(gamma_values: Iterable[float], width: float = 0.05) -> IntervalPredicate:
    values = list(gamma_values)
    if not values:
        raise ValueError("gamma_values must be non-empty")
    median = float(np.median(values))
    lower = (1.0 - width) * median
    upper = (1.0 + width) * median
    return IntervalPredicate(a=lower, b=upper)


def gamma_from_eigs(eigvals: np.ndarray) -> Callable[[int], float]:
    def Gamma(N: int) -> float:
        return float(np.sum(eigvals ** N).real)

    return Gamma
