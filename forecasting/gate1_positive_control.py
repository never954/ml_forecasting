"""
GATE 1 — positive control for the measurement instrument.

Runs the ENTIRE forecasting pipeline (windowing -> labeling -> grouped split ->
calibrated classifier -> metrics) on a system whose physics *guarantees* advance
information exists: a fold bifurcation with critical slowing down. Purpose is NOT
to prove ML works in the abstract, but to prove THIS pipeline does not leak and can
detect predictive information when it is present (Addendum sec.4).

Decision rule: AUROC >= 0.85 on held-out runs. If this fails, no conclusion about
the soup is admissible. If it passes and the soup later reads ~0.5, the event is
physically unpredictable at that horizon/representation — not a broken pipeline.

Usage: python forecasting/gate1_positive_control.py --horizon 40
"""
import argparse
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from synth import make_dataset
from windows import make_windows, stack_windowsets
from features import csd_features, CSD_FEATURE_NAMES
from models import fit_calibrated
from metrics import core_metrics, threshold_at_fpr, fpr_at_threshold, lead_times

GATE1_AUROC = 0.85


def build_windows(runs, horizon, window, stride):
    wsets = []
    for run_id, series, T in runs:
        wsets.append(make_windows(series, T, run_id, horizon, window, stride,
                                  csd_features, CSD_FEATURE_NAMES))
    return stack_windowsets(wsets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    print(f"[gate1] generating {args.n_pos} positive + {args.n_null} null runs ...")
    runs = make_dataset(n_pos=args.n_pos, n_null=args.n_null, seed=args.seed)
    n_tipped = sum(1 for _, _, T in runs if T is not None)
    print(f"[gate1] usable runs: {len(runs)} ({n_tipped} transitioning)")

    ws = build_windows(runs, args.horizon, args.window, args.stride)
    print(f"[gate1] windows: {len(ws.y)}  positives: {int(ws.y.sum())}  "
          f"base rate: {ws.y.mean():.3f}")

    # Seed-level split: no run appears in both train and test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    tr, te = next(gss.split(ws.X, ws.y, ws.run_id))
    assert set(ws.run_id[tr]).isdisjoint(set(ws.run_id[te])), "group leakage!"

    print("[gate1] fitting calibrated MLP (matched capacity) ...")
    clf = fit_calibrated(ws.X[tr], ws.y[tr], ws.run_id[tr])
    p_te = clf.predict_proba(ws.X[te])[:, 1]

    m = core_metrics(ws.y[te], p_te)
    thr = threshold_at_fpr(p_te[ws.y[te] == 0], args.target_fpr)
    m["fpr_at_thr"] = fpr_at_threshold(ws.y[te], p_te, thr)
    lt = lead_times(ws.run_id[te], ws.t_end[te], ws.T[te], p_te, thr)

    print("\n================ GATE 1 REPORT ================")
    print(f"horizon H              : {args.horizon} epochs")
    print(f"test windows           : {len(te)}  (base rate {m['base_rate']:.3f})")
    print(f"AUROC                  : {m['auroc']:.3f}   (gate: >= {GATE1_AUROC})")
    print(f"AUPRC                  : {m['auprc']:.3f}   (baseline = base rate {m['base_rate']:.3f})")
    print(f"Brier score            : {m['brier']:.4f}")
    print(f"operating threshold    : {thr:.3f}  (target FPR {args.target_fpr})")
    print(f"achieved FPR           : {m['fpr_at_thr']:.3f}")
    print(f"median lead time       : {lt['median_lead_time']} epochs")
    print(f"warned fraction        : {lt['warned_fraction']:.2f} of {lt['n_positive_runs']} pos runs")
    verdict = "PASS" if m["auroc"] >= GATE1_AUROC else "FAIL"
    print(f"VERDICT                : GATE 1 {verdict}")
    print("===============================================")
    return 0 if m["auroc"] >= GATE1_AUROC else 1


if __name__ == "__main__":
    raise SystemExit(main())
