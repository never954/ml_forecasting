"""
Run the BFF primordial soup and log the high-order-entropy trajectory.
Reproduces the pre-life -> life transition (Stage 1 of the project).

Usage:
    python simulator/run_soup.py --n 4096 --epochs 5000 --out ../results/run
"""
import argparse
import time
import numpy as np

from bff import make_soup, run_epoch, mutate
from detector import high_order_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="number of programs (soup size)")
    ap.add_argument("--plen", type=int, default=64, help="program length in bytes")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=2048, help="instruction cap per interaction")
    ap.add_argument("--mutation", type=float, default=1.5e-4, help="per-byte background mutation rate")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="../results/run")
    args = ap.parse_args()

    soup, rng = make_soup(args.n, args.plen, seed=args.seed)

    # Warm up the JIT so timing/first epoch isn't skewed.
    order = np.arange(args.n, dtype=np.int64)
    run_epoch(soup.copy(), args.plen, order[:2].copy(), args.max_steps)

    rows = []
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        if epoch % args.log_every == 0:
            hoe, sh, ko = high_order_entropy(soup)
            rows.append((epoch, hoe, sh, ko))
            el = time.time() - t0
            print(f"epoch {epoch:6d}  HOE={hoe:6.3f}  shannon={sh:5.3f}  "
                  f"kolm={ko:5.3f}  ({el:5.1f}s)", flush=True)
            np.savetxt(args.out + ".csv", np.array(rows),
                       delimiter=",", header="epoch,hoe,shannon,kolmogorov",
                       comments="", fmt="%.6f")
        order = rng.permutation(args.n).astype(np.int64)
        run_epoch(soup, args.plen, order, args.max_steps)
        mutate(soup, rng, args.mutation)

    np.save(args.out + "_final_soup.npy", soup)
    print("done ->", args.out + ".csv")


if __name__ == "__main__":
    main()
