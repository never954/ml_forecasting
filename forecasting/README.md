# Forecasting pipeline (Problem A / Stage 3)

The measurement instrument: leakage-safe sliding-window forecasting of a binary
target `Y_t^H = 1[t < T ≤ t+H]`, validated on a positive control before it is
applied to the soup.

## Modules
| File | Role |
|---|---|
| `windows.py` | Sliding-window construction + binary labeling. Enforces the leakage rules (windows end strictly before `T`; epoch index withheld; grouping by run id). |
| `features.py` | Arm **E1** — critical-slowing-down indicators (variance, lag-1 autocorrelation, …) on a detrended window. |
| `synth.py` | Known **fold-bifurcation** generator: transitioning runs (with critical slowing down) + matched null runs. The Gate 1 ground truth. |
| `models.py` | Calibrated **deep-MLP** classifier at fixed "matched capacity", with run-grouped calibration folds (Addendum §3). |
| `metrics.py` | AUROC, AUPRC, Brier, FPR at an operating threshold, median lead time. |
| `gate1_positive_control.py` | Runs the whole pipeline on the control. **Decision rule: AUROC ≥ 0.85.** |
| `test_windows.py` | Leakage/labeling unit tests (5/5). |

## Gate 1 — status: **PASS**
```
python forecasting/test_windows.py            # 5/5 leakage tests
python forecasting/gate1_positive_control.py  # AUROC 0.99 on held-out runs
python forecasting/plot_gate1.py              # -> results/gate1_positive_control.png
```
Result (H=40, 200 positive + 200 null runs, seed-level 70/30 split):

| metric | value |
|---|---|
| AUROC | **0.994** (gate ≥ 0.85) |
| AUPRC | 0.897 (base rate 0.031) |
| Brier | 0.0086 |
| FPR @ threshold | 0.055 (target 0.05) |
| median lead time | ~200 epochs |
| warned fraction | 1.00 of positive runs |

**What this certifies (Addendum §4):** the windowing, retrospective labeling,
seed-level splitting, and metric code do not leak and *do* detect predictive
information when the physics guarantees it exists. Therefore, if the soup later
reads AUROC ≈ 0.5, that is evidence the event is **physically unpredictable** at
that horizon/representation — not a broken pipeline.

## Next
Apply this exact pipeline (arm E1) to a soup corpus with a genuine pre-life phase
(larger soups than the Stage-1 demo runs), reporting AUROC/AUPRC/Brier/lead-time
for "self-replication within H epochs?" — the first real forecast of the emergence
event.
