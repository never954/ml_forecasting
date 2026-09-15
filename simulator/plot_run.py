"""
Plot the high-order-entropy trajectory of a soup run and mark the detected
pre-life -> life transition. Usage: python simulator/plot_run.py ../results/run1
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_transition(epoch, hoe, k=6.0):
    """Transition = first crossing of (epoch-0 baseline + k*early-noise-std).

    Baseline is anchored on the initial random-soup value (epoch 0), so a fast
    takeover is not absorbed into the baseline window.
    """
    base = hoe[0]
    # Random soup sits at HOE ~ 0; a living soup at ~1+. A fixed margin above the
    # initial baseline is a robust, config-independent transition marker.
    thr = max(base + 0.30, 0.30)
    mask = hoe > thr
    idx = np.argmax(mask) if mask.any() else -1
    return (epoch[idx] if idx >= 0 and mask[idx] else None), thr


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "../results/run1"
    d = np.genfromtxt(stem + ".csv", delimiter=",", names=True)
    epoch, hoe, sh, ko = d["epoch"], d["hoe"], d["shannon"], d["kolmogorov"]

    t_star, thr = detect_transition(epoch, hoe)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})

    ax[0].plot(epoch, hoe, lw=1.8, color="#c0392b", label="high-order entropy")
    ax[0].axhline(thr, ls="--", lw=1, color="gray", label="detection threshold")
    if t_star is not None:
        ax[0].axvline(t_star, ls=":", lw=1.5, color="#2c3e50")
        ax[0].annotate(f"transition\n~epoch {int(t_star)}",
                       xy=(t_star, hoe.max() * 0.6),
                       xytext=(t_star + (epoch.max() - t_star) * 0.15, hoe.max() * 0.55),
                       arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax[0].set_ylabel("high-order entropy (bits/byte)")
    ax[0].set_title("BFF primordial soup: spontaneous emergence of self-replicators\n"
                    "(from-scratch reproduction of Aguera y Arcas et al., 2024)")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.25)

    ax[1].plot(epoch, sh, lw=1.4, color="#2980b9", label="Shannon entropy (bytes)")
    ax[1].plot(epoch, ko, lw=1.4, color="#27ae60", label="Kolmogorov (compressed)")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("bits/byte")
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    out = stem + "_emergence.png"
    fig.savefig(out, dpi=140)
    print("saved", out)
    if t_star is not None:
        print(f"detected transition at epoch ~{int(t_star)}")
    else:
        print("no transition detected yet in this window")


if __name__ == "__main__":
    main()
