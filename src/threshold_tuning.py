"""Ordinal threshold tuning: converts softmax-expected grade into discrete predictions
using cut points optimized for quadratic-weighted Kappa, instead of plain argmax.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from baseline import KLGradeModel, NUM_CLASSES

DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]  # equivalent to standard rounding


def get_expected_grades(
    model: KLGradeModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model and returns the softmax-weighted expected grade per sample."""
    model.eval()
    grades = torch.arange(NUM_CLASSES, dtype=torch.float32)
    expected: list[float] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            expected.extend((probs @ grades).tolist())
            targets.extend(labels.tolist())
    return np.array(expected), np.array(targets)


def apply_thresholds(expected: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Buckets expected grades into NUM_CLASSES discrete predictions via cut points."""
    return np.digitize(expected, thresholds)


def tune_thresholds(
    expected: np.ndarray,
    targets: np.ndarray,
    initial: list[float] | None = None,
    grid_step: float = 0.05,
    rounds: int = 4,
) -> list[float]:
    """Coordinate-ascent search over cut points to maximize quadratic-weighted Kappa.

    Each of the 4 cut points (between grades 0-1, 1-2, 2-3, 3-4) is optimized in turn,
    holding the others fixed, for a few rounds. This is a small enough search space
    (4 ordered scalars in [0, 4]) that coordinate ascent reliably finds a good optimum
    without needing a general-purpose optimizer.
    """
    thresholds = list(initial or DEFAULT_THRESHOLDS)
    grid = np.arange(0.1, 3.95, grid_step)

    def kappa_for(thr: list[float]) -> float:
        preds = apply_thresholds(expected, thr)
        return cohen_kappa_score(targets, preds, weights="quadratic")

    for _ in range(rounds):
        for i in range(len(thresholds)):
            lo = thresholds[i - 1] if i > 0 else 0.0
            hi = thresholds[i + 1] if i < len(thresholds) - 1 else 4.0
            best_value, best_score = thresholds[i], kappa_for(thresholds)
            for candidate in grid:
                if not (lo < candidate < hi):
                    continue
                trial = thresholds.copy()
                trial[i] = candidate
                trial_score = kappa_for(trial)
                if trial_score > best_score:
                    best_score, best_value = trial_score, candidate
            thresholds[i] = best_value

    return thresholds
