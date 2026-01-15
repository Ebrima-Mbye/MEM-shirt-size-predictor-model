from __future__ import annotations

from typing import Any, Tuple

import numpy as np


FEATURE_ORDER: Tuple[str, ...] = (
    "height_cm",
    "weight_kg",
    "age",
    "gender",
    "fit_preference",
    "build",
)


def rows_to_matrix(batch: Any) -> np.ndarray:
    """Convert an iterable of row dicts into a fixed-order 2D feature matrix.

    This is defined in an importable module so sklearn pipelines using it are
    picklable/unpicklable across processes (e.g., Uvicorn workers).
    """

    out = np.empty((len(batch), len(FEATURE_ORDER)), dtype=object)
    for i, row in enumerate(batch):
        for j, key in enumerate(FEATURE_ORDER):
            out[i, j] = row.get(key)
    return out
