"""
Render the soup as an image: each row is one program, each column a byte,
colour = byte value. Random soup looks like TV static (high entropy, no
structure); after self-replicators take over, visible horizontal/vertical
banding appears as copies of the same programs spread through the population.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="*_final_soup.npy")
    ap.add_argument("--plen", type=int, default=64)
    ap.add_argument("--rows", type=int, default=256, help="programs to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../results/soup_before_after.png")
    args = ap.parse_args()

    final = np.load(args.final)
    n = final.shape[0] // args.plen
    final_img = final.reshape(n, args.plen)[: args.rows]

    rng = np.random.default_rng(args.seed)
    before_img = rng.integers(0, 256, size=(args.rows, args.plen), dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, img, title in (
        (axes[0], before_img, "epoch 0 — random soup\n(no structure, incompressible)"),
        (axes[1], final_img, "after emergence\n(repeated programs = self-replicators)"),
    ):
        ax.imshow(img, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("byte position")
        ax.set_ylabel("program")
    fig.suptitle("BFF primordial soup: before vs after self-replicators emerge",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
