"""
Leakage and labeling tests for the windowing core. This is what Gate 1 ultimately
certifies, so it is tested directly. Run: python forecasting/test_windows.py
"""
import numpy as np
from windows import make_windows, stack_windowsets


def _ident_last(win):
    # trivial feature: last value + window mean (both finite); shape (2,)
    return np.array([win[-1], win.mean()], dtype=np.float64)


def test_no_window_reaches_or_passes_T():
    series = np.arange(100, dtype=np.float64)
    T, H, W = 60, 10, 8
    ws = make_windows(series, T, "r0", horizon=H, window=W, stride=1,
                      feature_fn=_ident_last, feature_names=["last", "mean"])
    # every window ends strictly before T -> its features never include epoch >= T
    assert ws.t_end.max() < T, ws.t_end.max()


def test_label_matches_definition():
    series = np.arange(100, dtype=np.float64)
    T, H, W = 60, 10, 8
    ws = make_windows(series, T, "r0", horizon=H, window=W, stride=1,
                      feature_fn=_ident_last, feature_names=["last", "mean"])
    for t, y in zip(ws.t_end, ws.y):
        expected = 1 if (t < T <= t + H) else 0
        assert y == expected, (t, y, expected)
    # positives must be exactly the window ends in [T-H, T-1]
    pos_t = sorted(ws.t_end[ws.y == 1])
    assert pos_t == list(range(T - H, T)), pos_t


def test_null_run_all_negative():
    series = np.random.default_rng(0).normal(size=120)
    ws = make_windows(series, None, "n0", horizon=10, window=8, stride=1,
                      feature_fn=_ident_last, feature_names=["last", "mean"])
    assert set(np.unique(ws.y)).issubset({0})
    assert np.all(ws.T == -1)


def test_epoch_index_not_in_features():
    # features are (last_value, mean); neither equals the absolute epoch unless the
    # series itself encodes it. Guard: feature width is exactly what feature_fn returns.
    series = np.random.default_rng(1).normal(size=80)
    ws = make_windows(series, 50, "r1", horizon=5, window=6, stride=1,
                      feature_fn=_ident_last, feature_names=["last", "mean"])
    assert ws.X.shape[1] == 2


def test_grouping_preserved_on_stack():
    s1 = np.arange(60, dtype=np.float64)
    s2 = np.arange(60, dtype=np.float64)
    w1 = make_windows(s1, 40, "A", 5, 6, 1, _ident_last, ["last", "mean"])
    w2 = make_windows(s2, None, "B", 5, 6, 1, _ident_last, ["last", "mean"])
    ws = stack_windowsets([w1, w2])
    assert set(np.unique(ws.run_id)) == {"A", "B"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
