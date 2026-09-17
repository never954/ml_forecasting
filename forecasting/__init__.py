"""Forecasting pipeline for the emergence-prediction project (Problem A / Stage 3).

Modules:
    windows   - leakage-safe sliding-window construction + binary labeling Y_t^H
    features  - observation-map features (Stage 6 arm E1 = critical-slowing-down indicators)
    synth     - known fold-bifurcation generator (Gate 1 positive control + null runs)
    models    - calibrated non-linear classifier at matched capacity (Addendum sec.3)
    metrics   - AUROC / AUPRC / Brier / lead-time / FPR
"""
