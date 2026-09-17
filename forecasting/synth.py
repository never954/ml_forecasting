"""
Known bifurcation generator — the Gate 1 positive control.

We integrate the saddle-node (fold) normal form with a slowly ramped control
parameter and additive noise:

    dx = (mu(t) - x^2) dt + sigma dW

For mu > 0 there is a stable fixed point at x* = +sqrt(mu) whose local recovery
rate 2*sqrt(mu) -> 0 as mu -> 0. That vanishing recovery rate is *critical slowing
down*: fluctuation variance and lag-1 autocorrelation rise before the tip. When
mu crosses 0 the fixed point disappears and the state collapses (the transition).

This gives ground truth: "positive" runs approach a real bifurcation and carry
detectable pre-transition information; "null" runs sit far from it and do not. If
our pipeline can tell imminent-transition windows from the rest, the instrument
works (Gate 1). We generate the data ourselves (rather than via ewstools) so the
ground truth is fully transparent.
"""
import numpy as np


def simulate_fold(rng, mu0=2.0, mu_end=-0.5, n_epochs=700, steps_per_epoch=10,
                  dt=0.01, sigma=0.05, tip_thresh=-0.5):
    """One transitioning run. Returns (series, T) where T is the tip epoch."""
    total_steps = n_epochs * steps_per_epoch
    ramp = np.linspace(mu0, mu_end, total_steps)
    x = np.sqrt(max(mu0, 1e-6))
    series = np.empty(n_epochs, dtype=np.float64)
    T = None
    sdt = np.sqrt(dt)
    for e in range(n_epochs):
        for s in range(steps_per_epoch):
            mu = ramp[e * steps_per_epoch + s]
            x += (mu - x * x) * dt + sigma * sdt * rng.standard_normal()
            if x < -5.0:
                x = -5.0
        series[e] = x
        if T is None and x < tip_thresh:
            T = e
    return series, T


def simulate_null(rng, mu_const=1.2, n_epochs=700, steps_per_epoch=10,
                  dt=0.01, sigma=0.05):
    """One non-transitioning run: parameter held far from the bifurcation."""
    x = np.sqrt(mu_const)
    series = np.empty(n_epochs, dtype=np.float64)
    sdt = np.sqrt(dt)
    for e in range(n_epochs):
        for s in range(steps_per_epoch):
            x += (mu_const - x * x) * dt + sigma * sdt * rng.standard_normal()
        series[e] = x
    return series, None


def make_dataset(n_pos=200, n_null=200, seed=0, **kw):
    """Return list of (run_id, series, T). Null-run lengths match positive runs."""
    rng = np.random.default_rng(seed)
    runs = []
    pos_lengths = []
    for i in range(n_pos):
        s, T = simulate_fold(rng, **{k: v for k, v in kw.items()
                                     if k in simulate_fold.__code__.co_varnames})
        # keep only runs that actually tipped with a usable pre-transition stretch
        if T is None or T < 60:
            continue
        runs.append((f"pos_{i}", s, int(T)))
        pos_lengths.append(T)
    # Null runs: length drawn to match the pre-transition span of positive runs,
    # so the negative class has a matched epoch distribution (no length shortcut).
    med_len = int(np.median(pos_lengths)) if pos_lengths else kw.get("n_epochs", 700)
    for j in range(n_null):
        n_ep = max(80, int(np.random.default_rng(seed + 1000 + j)
                           .integers(med_len // 2, med_len + med_len // 2 + 1)))
        s, _ = simulate_null(np.random.default_rng(seed + 2000 + j),
                             n_epochs=n_ep,
                             mu_const=kw.get("mu_const", 1.2),
                             sigma=kw.get("sigma", 0.05))
        runs.append((f"null_{j}", s, None))
    return runs


if __name__ == "__main__":
    # Self-check: do positive runs tip, and does variance rise before the tip?
    runs = make_dataset(n_pos=40, n_null=10, seed=1)
    pos = [r for r in runs if r[2] is not None]
    print(f"tipped {len(pos)}/40 positive runs")
    early_var, late_var = [], []
    for _, s, T in pos:
        if T > 120:
            early_var.append(np.var(s[T - 120:T - 80]))   # far from tip
            late_var.append(np.var(s[T - 40:T]))           # near tip
    print(f"mean variance far-from-tip:  {np.mean(early_var):.5f}")
    print(f"mean variance near-tip:      {np.mean(late_var):.5f}")
    print(f"variance ratio (near/far):   {np.mean(late_var)/np.mean(early_var):.2f}x")
