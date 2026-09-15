# What to Learn First — Navigation Guide

**Goal of this doc:** the *minimum* you need to read this project and know what every stage is doing. Not "learn all of ML." Just enough that no term in the proposal is a black box.

The project is: *can you forecast the spontaneous birth of self-replicating programs in a random-code "soup" before it happens, and — more importantly — what information makes that forecast possible?* Forecasting is the ruler; **information** is the thing being measured.

Everything below is tiered. **Tier 0 is non-negotiable.** Tier 1 you learn as you build. Tier 2 is for the hard back-half (Problem B).

---

## Tier 0 — Learn before you touch code (~2–3 weeks)

### 0.1 The substrate (read the actual paper first)
- **Agüera y Arcas et al., 2024, "Computational Life" (arXiv:2406.19108).** This *is* your world. You must understand it cold.
  - **BFF** = extended Brainfuck. Programs are **64 bytes**, instruction and data share one tape, programs modify themselves and each other. Only **10 of 256 byte-values are instructions**; the rest are data.
  - The **soup** = a large population (the paper uses **2¹⁷ ≈ 131,072** programs). Each epoch: pick random ordered pairs, concatenate, execute, split them back. No fitness function, no selection, no seeded replicator. Self-replication *emerges anyway* — that's the whole point.
  - The transition is detected **after the fact** using their **"high-order entropy"** metric (Shannon entropy minus normalized Kolmogorov complexity, approximated by a compressor like gzip). Your project turns this *post-hoc detector* into a *label* and asks whether you can see the event coming.
- Clone and run their open-source code before anything else. If you can't reproduce a run, nothing downstream matters.

### 0.2 Information theory (the spine of the whole project)
You need these five ideas fluently. A weekend with a good tutorial (e.g. the info-theory chapters of MacKay's free book, or 3Blue1Brown-level intuition + one problem set):
- **Entropy H(X)** — uncertainty in bits.
- **Mutual information I(X;Y)** — how many bits knowing X tells you about Y. *This is the single most important quantity in the project.*
- **Conditional MI I(X;Y|Z)** — bits X adds about Y *once you already know Z*. (This is exactly how "does history help beyond the present state?" is tested.)
- **KL divergence** — "distance" between distributions; underlies cross-entropy/log-loss.
- **Data Processing Inequality (DPI):** `I(φ(X); Y) ≤ I(X; Y)`. **Memorize this.** It's the mathematical backbone: no observer (representation φ) can extract more predictive information than exists. The whole "predictability is observer-relative" claim is just DPI.

### 0.3 Classification + calibration (your measuring instrument)
The "forecaster" is just a probabilistic binary classifier ("will the transition fire within H epochs? yes/no").
- **AUROC / AUPRC** — discrimination metrics. Learn *why AUPRC matters when positives are rare* (they are here).
- **Log-loss / cross-entropy** — this is your *primary MI estimator*, not just a training loss. Key fact to internalize: **a calibrated classifier's log-loss vs. the base-rate gives a valid lower bound on I(features; label)**. That's the trick that dodges the hard MI-estimation problem.
- **Calibration: Brier score, reliability diagrams.** You need probabilities that *mean* something, because later they're compared to a theoretical ceiling.

### 0.4 Experiment hygiene (this is where projects secretly die)
- **Temporal causality / no leakage:** every training window must end **≥ H epochs before** the event. Splits are by *simulation seed*, never by window.
- **Positive & negative controls:** a system with a *known* transition (to prove your pipeline can detect anything at all) and runs that *never* transition (to prove you're not just saying "yes").
- **Multiple-testing correction (Benjamini–Hochberg):** you'll run a grid of representations × horizons; you must correct for it.

---

## Tier 1 — Learn while building Problem A (~parallel with stages 1–5)

### 1.1 Early-Warning Signals (EWS) — the classical forecasting toolkit
- **Critical Slowing Down (CSD):** near a tipping point, variance and lag-1 autocorrelation of an observable tend to rise. This is the *baseline* forecaster (arm E1).
- **Tipping taxonomy (Ashwin et al. 2012):** bifurcation- vs. noise- vs. rate-induced. **Why you care:** CSD only reliably works for *bifurcation* tipping. If your soup's transition is *noise-induced nucleation*, CSD is expected to **fail** — and that failure is a result, not a bug.
- **Tool:** `ewstools` (Bury). Gives you CSD indicators + a pretrained deep-learning EWS classifier out of the box. Learn its API; it also supplies your **positive control** (known bifurcation models).

### 1.2 Rare-event theory — the "predictability ceiling"
- **The committor q(x):** probability that, starting from state x, the system hits the transition *before* returning to the pre-life basin. Intuition only is enough at first: **q is the ideal forecaster.** If you knew q exactly, no model could beat it.
- **Why it matters:** `Var[q(x)]` gives a **ceiling on predictability in bits** (via a Bayes/Fano bound). This converts "we found no signal" into "there *is* no signal, and here's the proven bound." Learn the *concept*; estimation (Li et al. 2019 — committor by deep learning) is a Tier-2 build.

### 1.3 Python/ML tooling
- **PyTorch** (classifiers, the committor net).
- **scikit-learn** (baselines, metrics, calibration).
- **numpy/pandas**, plus a compressor (`gzip`/`zlib`) for the high-order-entropy detector.
- Comfortable running **hundreds of simulations** and managing their outputs (this becomes a data-engineering problem — see the implementation plan on storage).

---

## Tier 2 — For Problem B (the science; stages 6–8)

### 2.1 MI estimation and its limits
- **The hard truth (McAllester & Stratos, 2020):** any distribution-free high-confidence *lower bound* on MI from N samples grows only like `O(ln N)`. Translation: **do not trust point estimates of MI in high dimensions.** This is *why* the project reports **bounds and ratios**, not numbers. Understand this or you'll over-claim.
- Know the estimator zoo *by name and role*: **log-loss lower bound (workhorse)**, plug-in/KSG (small-dim sanity checks), variational bounds like MINE/InfoNCE (corroboration only). You don't need to derive them; you need to know which to trust when.

### 2.2 Representation & GNNs
- The **central experiment (E1–E7)** is comparing *representations* of the same system — a global scalar → population statistics → local neighborhoods → the interaction graph → full micro-state → +history → +oracle internals.
- **Graph Neural Networks (PyTorch Geometric):** E4 encodes "which program interacted with which" as a graph. You need a *working* GNN, not GNN theory. One good tutorial + one toy project.

### 2.3 Local information dynamics (lightest touch)
- **JIDT (Java Information Dynamics Toolkit)** + concepts: **transfer entropy** (directed *predictive* relationship, **not causation** — burn that caveat in), active information storage. Used to ask *where in the soup* the predictive information sits and *when it concentrates*.
- **Computational mechanics (causal states, ε-machines; Crutchfield/Shalizi):** you only need the *idea* — the minimal statistic that captures all predictive information — because Stage 0 checks whether the project's "Minimal Predictive State" is just a rebranding of this. Conceptual reading only.

---

## What you can safely *skip*
- Deep learning architectures beyond MLPs + one GNN (no transformers, no fancy nets needed).
- Reinforcement learning entirely.
- Full measure-theoretic probability / heavy proofs of the info-theory bounds — use them as tools.
- Integrated Information Φ, Fisher information, PID internals — the proposal *deliberately rejects or demotes* these; don't rabbit-hole.

## Fastest path (if you only have a month)
1. Run the Computational Life code end-to-end. (Substrate)
2. Info theory: H, I, conditional I, DPI. (Spine)
3. Build one calibrated classifier + AUROC/AUPRC/Brier + a leakage-safe temporal split on toy data. (Instrument)
4. `ewstools` CSD on a known bifurcation → get AUROC ≥ 0.85. (Positive control = the go/no-go gate)

If you can do those four, you can start Stage 1 and learn the rest just-in-time.
