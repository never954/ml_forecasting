"""
Calibrated non-linear classifier at matched capacity (Addendum sec.3).

We standardize on ONE expressive family — a deep MLP — as the primary estimator,
with a fixed architecture reused across every representation arm so differences
reflect information content, not model power. Probabilities are calibrated because
the committor comparison and lead-time thresholds need probabilities that mean what
they say, not just a ranking.

Calibration respects run-level groups: the internal calibration folds are grouped
by run id, so no window from a run appears in both the fit and calibration sides.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold


# Fixed "matched capacity" architecture reused across all representation arms.
MATCHED_CAPACITY = dict(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    alpha=1e-3,              # L2 regularization
    max_iter=400,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=0,
)


def make_estimator():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**MATCHED_CAPACITY)),
    ])


def fit_calibrated(X, y, groups, n_folds=3):
    """Fit the MLP with grouped, calibrated probabilities."""
    n_groups = len(np.unique(groups))
    n_folds = max(2, min(n_folds, n_groups))
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    method = "isotonic" if np.bincount(y).min() >= 200 else "sigmoid"
    clf = CalibratedClassifierCV(estimator=make_estimator(), method=method, cv=splits)
    clf.fit(X, y)
    return clf
