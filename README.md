# Predictability of Emergent Self-Replication — Project Repository

**Research question:** *What determines whether an emergent event is predictable?*
We study this through the spontaneous origin of self-replicating programs in a
random computational substrate. Forecasting is the **measuring instrument**;
**information** is the scientific object.

Full framing in the planning docs:
- [`01_learning_prerequisites.md`](01_learning_prerequisites.md) — what to learn to navigate the project
- [`02_implementation_plan.md`](02_implementation_plan.md) — the 9-stage build plan with gates
- [`03_feasibility_novelty_audit.md`](03_feasibility_novelty_audit.md) — doability / novelty / assumption check

---

## Current status: **Stage 1 complete** — substrate + detector reproduced

We have a from-scratch, tested reproduction of the BFF primordial soup from
*Computational Life* (Agüera y Arcas et al., 2024, arXiv:2406.19108) and its
compression-based transition detector. Self-replicators emerge spontaneously
from random bytes with **no fitness function and no seeded replicator** — the
labelled event this whole project is built to forecast.

### What's in `simulator/`
| File | What it is |
|---|---|
| `bff.py` | BFF interpreter (10-instruction extended Brainfuck) + primordial-soup epoch loop, numba-JIT accelerated (~200 epochs/s) |
| `detector.py` | High-order-entropy detector = Shannon entropy − normalized Kolmogorov (zlib) complexity |
| `test_bff.py` | Correctness suite (5/5 pass), incl. fuzz-equivalence vs. a transparent reference over 3,000 random tapes |
| `run_soup.py` | Runs a soup, logs the HOE trajectory |
| `plot_run.py` | Plots the emergence curve and marks the detected transition |

### Reproduce
```bash
pip install numpy numba matplotlib
cd simulator
python test_bff.py                                   # 5/5 correctness tests
python run_soup.py --n 4096 --epochs 20000 --out ../results/run1
python plot_run.py ../results/run1                   # -> results/run1_emergence.png
```

### Result
`results/run1_emergence.png` — high-order entropy jumps from ~0 (random,
incompressible soup) to a peak of ~1.8 bits/byte as self-replicators take over,
then settles into a "living" steady state. Shannon entropy falls 8.0 → 6.8 while
compressibility drops faster (Kolmogorov 8.0 → 6.2); the growing gap between them
*is* the emergent structure.

---

## Why this is the right first milestone
Everything downstream — forecasting the transition (Problem A) and measuring
*what information* makes it predictable (Problem B) — needs (1) a working substrate
that produces the event and (2) a trustworthy label for when it happens. Stage 1
delivers both. Next: build the labelled corpus (N ≥ 400 runs) and the positive
control (a known bifurcation at AUROC ≥ 0.85) — Gate 1 in the implementation plan.
