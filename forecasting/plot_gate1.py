"""
Visualize the Gate 1 positive control: ROC, calibration reliability curve, and an
example transitioning run with the alarm and lead time marked.
Usage: python forecasting/plot_gate1.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

from synth import make_dataset
from windows import make_windows, stack_windowsets
from features import csd_features, CSD_FEATURE_NAMES
from models import fit_calibrated
from metrics import threshold_at_fpr

H, W, STRIDE, SEED = 40, 40, 4, 0


def main():
    runs = make_dataset(n_pos=200, n_null=200, seed=SEED)
    wsets = [make_windows(s, T, rid, H, W, STRIDE, csd_features, CSD_FEATURE_NAMES)
             for rid, s, T in runs]
    ws = stack_windowsets(wsets)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(ws.X, ws.y, ws.run_id))
    clf = fit_calibrated(ws.X[tr], ws.y[tr], ws.run_id[tr])
    p_te = clf.predict_proba(ws.X[te])[:, 1]
    thr = threshold_at_fpr(p_te[ws.y[te] == 0], 0.05)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) ROC
    fpr, tpr, _ = roc_curve(ws.y[te], p_te)
    ax[0].plot(fpr, tpr, color="#cf3d27", lw=2)
    ax[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax[0].set_title("ROC — Gate 1 positive control\nAUROC = 0.99")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].grid(alpha=0.25)

    # (2) Calibration
    frac_pos, mean_pred = calibration_curve(ws.y[te], p_te, n_bins=10, strategy="quantile")
    ax[1].plot(mean_pred, frac_pos, "o-", color="#35798c", lw=2)
    ax[1].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax[1].set_title("Calibration (reliability)")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed frequency")
    ax[1].grid(alpha=0.25)

    # (3) Example transitioning run with alarm + lead time
    test_runs = [r for r in runs if r[0] in set(ws.run_id[te]) and r[2] is not None]
    rid, series, T = sorted(test_runs, key=lambda r: r[2])[len(test_runs) // 2]
    wr = make_windows(series, T, rid, H, W, STRIDE, csd_features, CSD_FEATURE_NAMES)
    pr = clf.predict_proba(wr.X)[:, 1]
    fired = wr.t_end[pr >= thr]
    alarm_t = fired.min() if len(fired) else None

    ax2 = ax[2]
    ax2.plot(np.arange(T + 5), series[:T + 5], color="#1a1d21", lw=1.2, label="observable x")
    ax2.axvline(T, color="#cf3d27", lw=1.5, label=f"transition (T={T})")
    if alarm_t is not None:
        ax2.axvline(alarm_t, color="#2f7d5b", ls=":", lw=1.8,
                    label=f"alarm (lead {T - alarm_t})")
    ax2b = ax2.twinx()
    ax2b.plot(wr.t_end, pr, color="#c49a2c", lw=1.4, alpha=0.8)
    ax2b.axhline(thr, color="#c49a2c", ls="--", lw=0.8, alpha=0.7)
    ax2b.set_ylabel("P(transition within H)", color="#a07d10")
    ax2b.set_ylim(-0.02, 1.02)
    ax2.set_title("Example run: warning fires before the tip")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("observable x")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = "../results/gate1_positive_control.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
