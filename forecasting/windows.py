"""
Leakage-safe sliding-window construction and binary labeling.

This is the core of the measurement instrument (Problem A). Everything else — the
positive control and the soup forecaster — flows through it, so its correctness is
what Gate 1 certifies.

Target definition (Addendum sec.1, strict binary classification, no regression):
    For a run with transition epoch T (or None if it never transitions), a window
    ENDING at epoch t is labeled
        Y = 1  if  t < T <= t + H        (transition falls in the next H epochs)
        Y = 0  otherwise.

Leakage rules enforced here:
  * A window ending at t covers epochs [t-W+1, t]. We require t < T for every kept
    window of a transitioning run, so the observed window lies STRICTLY before the
    transition — no post-event data ever enters the features.
  * The absolute epoch index t is NEVER a feature (it is returned separately, only
    for lead-time accounting).
  * Splitting is by run id (seed), never by window — callers must group on `run_id`.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class WindowSet:
    X: np.ndarray          # (n_windows, n_features) — features only, NO epoch index
    y: np.ndarray          # (n_windows,) int {0,1}
    run_id: np.ndarray     # (n_windows,) group key for seed-level splitting
    t_end: np.ndarray      # (n_windows,) epoch at which each window ends (for lead time)
    T: np.ndarray          # (n_windows,) transition epoch of the source run (-1 if none)
    feature_names: list


def make_windows(series, T, run_id, horizon, window, stride, feature_fn, feature_names):
    """Build labeled windows from ONE run's observable series.

    series     : 1-D array of the observed scalar/vector per epoch.
    T          : transition epoch (int) or None if the run never transitions.
    run_id     : identifier for this run (used later for seed-level splits).
    horizon H  : forecast horizon in epochs.
    window  W  : number of epochs of history each feature vector is computed from.
    stride     : step between consecutive window ends.
    feature_fn : callable(window_slice) -> 1-D feature array.
    """
    series = np.asarray(series)
    L = series.shape[0]
    Xs, ys, tes = [], [], []
    # A window ends at t and needs W samples of history: t in [W-1, L-1].
    # For a transitioning run we additionally require t < T (strictly pre-event).
    t_max = (T - 1) if (T is not None) else (L - 1)
    for t in range(window - 1, min(L - 1, t_max) + 1, stride):
        win = series[t - window + 1: t + 1]
        feats = feature_fn(win)
        if not np.all(np.isfinite(feats)):
            continue
        if T is not None and t < T <= t + horizon:
            y = 1
        else:
            y = 0
        Xs.append(feats)
        ys.append(y)
        tes.append(t)
    if not Xs:
        return None
    X = np.asarray(Xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.int64)
    te = np.asarray(tes, dtype=np.int64)
    return WindowSet(
        X=X, y=y,
        run_id=np.full(len(y), run_id),
        t_end=te,
        T=np.full(len(y), -1 if T is None else T),
        feature_names=list(feature_names),
    )


def stack_windowsets(wsets):
    """Concatenate per-run WindowSets into one, preserving run ids for grouping."""
    wsets = [w for w in wsets if w is not None]
    if not wsets:
        raise ValueError("no windows produced")
    return WindowSet(
        X=np.vstack([w.X for w in wsets]),
        y=np.concatenate([w.y for w in wsets]),
        run_id=np.concatenate([w.run_id for w in wsets]),
        t_end=np.concatenate([w.t_end for w in wsets]),
        T=np.concatenate([w.T for w in wsets]),
        feature_names=wsets[0].feature_names,
    )
