# Implementation Plan

Turns the proposal's 9 stages into an actual build order with tech, deliverables, and go/no-go gates. Two halves: **Problem A** (build + validate the forecasting instrument) → **Problem B** (use it to measure information). Problem A is publishable on its own; Problem B depends on it, not the reverse.

**Golden rule:** three gates test the *instrument*, not the hypothesis. Never interpret a null result until the relevant gate has passed — otherwise "no signal" is indistinguishable from a bug.

---

## Tech stack (all free/open-source)
| Purpose | Tool |
|---|---|
| Substrate | Agüera y Arcas et al. "Computational Life" repo (BFF) |
| EWS baseline + positive control | `ewstools` |
| Classifiers / committor net | PyTorch, scikit-learn |
| GNN (arm E4) | PyTorch Geometric |
| Local info dynamics / transfer entropy | JIDT |
| Detector | gzip/zlib (high-order entropy) |
| Experiment tracking | anything (MLflow / plain CSVs + git); **pre-register** the grid |

---

## Phase 0 — Setup & novelty gate (Month 1)

**Stage 0 — Systematic review + terminology decision.**
- Screen ≥120 records, ≥35 full-text. Find the 5 closest prior works and state the exact difference from each.
- **Decision:** is "Minimal Predictive State" (MPS) just causal states / information bottleneck / Markov blanket / PSR? Is "predictive microenvironment" just a thresholded local-causal-state field? If yes → **drop the names, keep the measurements.** (See audit doc — this is the highest-probability outcome and it's *fine*.)
- **Gate: terminology fixed before any implementation.**

**Stages 1–2 — Reproduce substrate + build labelled corpus.**
- Reproduce the BFF soup; verify the high-order-entropy detector fires on their runs.
- Generate the corpus: **N ≥ 400** runs on the primary instruction set, **≥100 each** on two more instruction sets (substrate-generality insurance), yielding **≥150 positive** (transition) and **≥200 negative** (never-transition) runs.
- **Dual-label a subsample:** run an independent lineage-tracing detector, not just the compression signature. Label disagreement is itself a finding (predictability is target-relative).
- **Pilot (25 seeds) to fix checkpoint cadence** *before* bulk generation — see Storage below. This is critical: a too-coarse cadence manufactures a fake "history helps" result.

**Deliverable:** reproducible soup + labelled corpus + fixed vocabulary. **Milestone: framing frozen.**

---

## Phase 1 — Problem A: build & validate the instrument (Months 2–6)

**Stage 3 — Calibrate the forecaster.**
- Build the labelled forecasting task: target `Y = 1[t < T ≤ t+H]`, horizons **H ∈ {100, 250, 500, 1000}** epochs.
- Baseline forecaster = classical EWS (variance + lag-1 autocorrelation, via `ewstools`).
- Metrics: **AUROC, AUPRC, Brier, false-positive rate on censored negatives, median lead time.** Kendall τ of indicators vs. negative controls.
- **GATE 1 (the big one): AUROC ≥ 0.85 on a known bifurcation (positive control).** If the pipeline can't detect a transition it *should* see, **no conclusion about the soup is admissible.** Fix the pipeline before proceeding.

**Stages 4–5 — Predictability ceiling, tipping class, estimator validation.**
- **Committor estimation:** train a neural committor q(x) (Li et al. 2019 style). Compute **Var[q(x)]** and the implied **Bayes/Fano bound in bits** — this is the *denominator* for everything in Problem B.
- **Tipping class:** fit waiting-time distribution — exponential vs. Weibull vs. gamma (KS test + AIC). Exponential ⇒ memoryless nucleation ⇒ CSD expected to fail (a prediction, testable).
- **GATE 2: estimator recovery on synthetic ground truth** — recover ν within ±0.05 and I within ±0.1 bits, and correctly reproduce known "history-helps" and "history-doesn't" cases. **No information-theoretic result on the soup is admissible until this passes.**

**Deliverable + milestone:** **Problem A manuscript** (instrument validated, ceiling measured) → submit to ALIFE/ECAL/GECCO ~Month 6. *This is your safe publishable checkpoint even if Problem B struggles.*

---

## Phase 2 — Problem B: measure the information (Months 6–10)

This is the science. Every arm scored as a **fraction of the Stage-4 ceiling**, reported as bounds/ratios (never point MI estimates — McAllester–Stratos).

**Stages 6–7 — Representation comparison + how much must be observed.**
Seven representations of the *same* system, capacity/tuning matched so differences reflect *information content*, not model power:

| Arm | Representation | Question it answers |
|---|---|---|
| E1 | Single global scalar/epoch (entropy or compressed size) | What classical EWS assumes |
| E2 | Population statistics (k-mer spectra, diversity, moments) | Does distribution-level info suffice? |
| E3 | Local interaction neighborhoods | The "how much" question, representationally |
| E4 | Interaction graph via GNN | Does relational structure add anything? |
| E5 | Full observable micro-state, single epoch | The Markov reference |
| E6 | E5 + k previous epochs | Does history add over present state? |
| E7 | Full trajectory + **oracle simulator internals** | Isolates normally-hidden variables |

- Compute per arm: **predictive efficiency η = I(φ(X);Y)/I(X;Y)** (fraction of ceiling captured), plus the accuracy-vs-information curve.
- **Minimal Predictive State:** smallest set of sites S retaining (1−ε) of the predictive information, for **ε ∈ {0.05, 0.1, 0.25}**, via **three selection routes** (report cross-route agreement as a reliability stat) + random-subset controls. Report **minimal predictive volume ν = |S|/|V|** — an *upper bound* (selection is greedy/NP-hard).

**Stage 8 — Localisation + state-vs-history.**
- **State vs. history:** compute **conditional MI I(X_{t−k:t−1}; Y | X_t)** for k ∈ {10,50,100,500,1000}. Near-zero ⇒ present state is predictively Markov ⇒ "you need the right *observables*, not memory." If history *does* help: is it (a) coarse observation [benefit shrinks as E1→E5 refine], (b) hidden variables [persists at E5, vanishes at E7 oracle], or (c) genuine memory [persists even at oracle — treated as a **red flag / bug**, since the simulator is Markov given full internal state]?
- **Localisation:** per-site marginal-contribution **Gini = L(t)**; test whether hot sites persist (consecutive-epoch **Jaccard vs. permutation null**) and whether info **concentrates as the event nears** (L(t) trend, Kendall τ vs. negative controls).

**Deliverable + milestone:** **Problem B manuscript** → Artificial Life / Chaos / Physical Review E.

---

## Phase 3 — Generality (Months 10–11, stretch — dropped first)

**Stage 9 — Cross-substrate.** Reduced repeat on Lenia or Biomaker CA. Is the representation ordering and the sign of the L(t) trend preserved? First test of "is this about *emergence* or just *this substrate*?" **Explicitly droppable** if schedule slips; report as dropped.

**Month 12:** code + labelled-data release, documented deviations from the pre-registered plan.

---

## Storage & compute budget (plan for this early — it's the real constraint)
- **Compute (Phase I): CPU-bound, modest.** ~7 epochs/sec/core; transitions ~12k–16k epochs; 400 runs ≈ **~230 core-hours.** Fine on a cluster.
- **Storage (Phase II): the binding constraint.** One BFF soup snapshot ≈ 2¹⁷ × 64 B ≈ **8 MB**. At 10-epoch cadence over ~16k epochs ≈ 1,600 snapshots ≈ **~13 GB/run before compression** → matches the proposal's "low tens of GB." Across the multi-substrate corpus this is large.
  - **Mitigation baked into the design:** *retrospective retention* — checkpoint everything, but only keep full pre-transition state once T is known; discard the rest. Cadence fixed by the Stage-1 pilot.
- **GPU:** ~150–400 GPU-hours over Months 2–3 (biggest model = GNN over a few thousand nodes). **Fallback if allocation is small:** one substrate, two horizons, reduced N — and *report the lost power*, don't hide it.

## Critical-path dependencies
```
Stage 0 (framing) ─► Stages 1–2 (corpus) ─► Stage 3 (GATE 1) ─► Stages 4–5 (GATE 2, ceiling)
                                                                      │
                                          ┌───────────────────────────┘
                                          ▼
                              Stages 6–7 (representations, ν) ─► Stage 8 (localisation, history) ─► Stage 9 (stretch)
```
Gates 1 and 2 are hard stops. Do not spend Phase-II effort until both are green.
