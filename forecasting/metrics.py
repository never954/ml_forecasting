"""Forecasting metrics for the binary target Y_t^H."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def core_metrics(y_true, p):
    """Discrimination + calibration metrics."""
    return {
        "auroc": float(roc_auc_score(y_true, p)),
        "auprc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "base_rate": float(np.mean(y_true)),
    }


def threshold_at_fpr(p_neg, target_fpr=0.05):
    """Operating threshold that yields ~target FPR on the negative windows."""
    if len(p_neg) == 0:
        return 0.5
    return float(np.quantile(p_neg, 1.0 - target_fpr))


def fpr_at_threshold(y_true, p, thr):
    y_true = np.asarray(y_true)
    neg = p[y_true == 0]
    if len(neg) == 0:
        return float("nan")
    return float(np.mean(neg >= thr))


def lead_times(run_id, t_end, T, p, thr):
    """Median lead time (epochs before T) of the FIRST alarm per positive run.

    A positive run is one with T >= 0. For each, find the earliest window whose
    predicted probability crosses `thr`; its lead time is T - t_end. Runs that
    never fire are recorded as misses (no lead time).
    """
    run_id = np.asarray(run_id)
    t_end = np.asarray(t_end)
    T = np.asarray(T)
    p = np.asarray(p)
    leads, fired, total = [], 0, 0
    for rid in np.unique(run_id):
        m = run_id == rid
        if T[m][0] < 0:
            continue                      # null run
        total += 1
        alarms = m & (p >= thr)
        if not np.any(alarms):
            continue
        earliest_t = t_end[alarms].min()
        leads.append(int(T[m][0] - earliest_t))
        fired += 1
    return {
        "median_lead_time": float(np.median(leads)) if leads else float("nan"),
        "warned_fraction": (fired / total) if total else float("nan"),
        "n_positive_runs": total,
    }
