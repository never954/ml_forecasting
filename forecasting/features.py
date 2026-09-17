"""
Observation-map features.

Arm E1 (the representation classical early-warning-signal theory assumes): a small
set of critical-slowing-down (CSD) indicators computed from a window of a single
observable. Near a fold/saddle-node bifurcation the system's recovery rate falls,
so fluctuation variance and lag-1 autocorrelation rise before the transition.

The window is linearly detrended first so a drifting mean is not mistaken for
rising variance.
"""
import numpy as np

CSD_FEATURE_NAMES = ["variance", "lag1_autocorr", "std", "abs_return_mean", "range"]


def _detrend(win):
    n = win.shape[0]
    x = np.arange(n, dtype=np.float64)
    # least-squares line
    a, b = np.polyfit(x, win, 1)
    return win - (a * x + b)


def csd_features(win):
    """Critical-slowing-down indicators for a 1-D window."""
    win = np.asarray(win, dtype=np.float64)
    d = _detrend(win)
    var = float(np.var(d))
    std = float(np.sqrt(var))
    # lag-1 autocorrelation of the detrended window
    if var <= 1e-12 or d.shape[0] < 3:
        ac1 = 0.0
    else:
        ac1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
        if not np.isfinite(ac1):
            ac1 = 0.0
    abs_ret = float(np.mean(np.abs(np.diff(win))))
    rng = float(win.max() - win.min())
    return np.array([var, ac1, std, abs_ret, rng], dtype=np.float64)
