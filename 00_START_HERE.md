# 00 · Start Here — Project Bird's-Eye View

*The one-page reload. Read this first; the numbered docs go deeper.*
*Repo: github.com/never954/ml_forecasting · Status: Stage 1 complete.*

---

## The one-sentence version
We are studying **what makes an emergent event predictable at all** — treating
*forecasting as a measuring instrument* and *information as the thing being
measured* — and we use the **spontaneous birth of self-replicating programs** in a
random-code soup as a clean, fully observable test case.

## The core reframe (this is the whole idea)
Most work asks "can we predict X?" We instead ask **"what information makes X
predictable, for a given observer?"** Prediction is the ruler; information is the
object. This flips a forecasting project into a science-of-predictability project.

Three consequences we've committed to:
1. **Predictability is observer-relative.** It depends on what you observe and how
   you represent it, bounded by the data-processing inequality: `I(φ(X);Y) ≤ I(X;Y)`.
   Every claim is indexed by *event · horizon · observation map*.
2. **A null is a result.** "Nothing predicts it" becomes a *measured bound in bits*
   (via the committor / Fano), not a failure. This is what makes the project safe.
3. **Forecasting ≠ the contribution.** The contribution is the *measurement
   framework*, valid even if every forecasting model performs at chance.

## Two problems, one programme
- **Problem A — build & validate the instrument.** Can the transition be forecast
  before it happens, with calibrated accuracy/lead-time, against a measured ceiling?
  *(Publishable on its own ~month 6.)*
- **Problem B — the science.** *Which* representation carries the predictive
  information, *how much* of the system must be observed, does *history* add
  anything beyond the present state, and is the information *localised*?
  *(Depends on A; the real novelty lives here.)*

---

## What we've agreed on (the consensus)

**On novelty** — real but modest, and we state it honestly:
- Novel as an **integration + first application**: forecasting an emergence event
  and scoring representations against a measured predictability ceiling. *Not* a new
  theorem.
- Our two coined terms (**Minimal Predictive State**, **predictive
  microenvironment**) are *probably* rebrandings of existing constructs (information
  bottleneck / local causal states). **This is fine by design** — they sit at the
  bottom of the hierarchy, Stage 0 checks them first, and losing them costs
  *vocabulary, not experiments*.

**On assumptions** — the design's strength is that most "assumptions" are actually
**testable hypotheses with falsifiers**, not baked-in commitments:
- Whether classical early-warning-signals (variance/autocorrelation) work is *tested*,
  and expected to *fail* if the transition is noise-induced nucleation — a result,
  not a bug.
- "History helps" is not assumed; an oracle protocol distinguishes *incomplete
  observation* from *hidden variables* from *genuine memory*.

**On facts** — verified against the source paper (Agüera y Arcas et al., 2024):
- Substrate = **BFF** (extended Brainfuck), **64-byte** programs, ~**2¹⁷** soup,
  **10 instructions**, code and data share one tape. ✅
- Transition detected by **high-order entropy** (Shannon − compressed size). ✅
- The "~40% of runs transition within 16,000 epochs" figure is **plausible but
  unverified** — we treat it as a number to *reproduce in Stage 1*, not trust.

**On the real risks** (not correctness — practicality):
1. **Scope.** This is ~2–3 person-years done fully, as a 12-month solo project.
   Agreed strategy: **ship Problem A as the real deliverable; treat Problem B as
   best-effort; drop Stage 9 first.**
2. **The committor is the technical crux.** The whole Problem-B denominator (the
   predictability ceiling) depends on training a reliable committor `q(x)`.
3. **MI in high dimensions is untrustworthy as a point estimate** (McAllester–Stratos
   `O(ln N)` limit). Agreed: **report bounds and ratios**, use calibrated log-loss as
   the workhorse lower bound.

**Two hard gates we will not cross prematurely:**
- **Gate 1:** positive control must hit **AUROC ≥ 0.85** on a known transition, or no
  claim about the soup is admissible.
- **Gate 2:** estimators must recover known values on synthetic ground truth, or no
  information-theoretic result on the soup is admissible.

---

## Where we are right now
**Stage 1 is done.** We have a from-scratch, tested reproduction of the substrate:
- BFF interpreter + primordial-soup engine (numba, ~200 epochs/s), high-order-entropy
  detector — all in `simulator/`.
- **5/5 correctness tests** pass, including fuzz-equivalence vs. a transparent
  reference over 3,000 random tapes.
- **Self-replicators emerge spontaneously** from random bytes: high-order entropy
  jumps ~0 → 1.8 bits/byte at the transition, then settles into a "living" steady
  state. Plot: `results/run1_emergence.png`.
- Progress page for presenting: `progress_stage1.html`.

## The vision — the nine stages
| Stage | What | Status |
|---|---|---|
| 0 | Novelty review, fix terminology | ✅ done |
| 1–2 | Reproduce substrate ✓ · build labelled corpus (≥400 runs) | **half done — corpus next** |
| 3 | Calibrate the forecaster · **Gate 1** | next |
| 4–5 | Predictability ceiling (committor) · tipping class · **Gate 2** | ahead |
| 6–7 | Representation comparison (E1–E7) · how much must be observed | ahead |
| 8 | Localisation dynamics · state-vs-history | ahead |
| 9 | Cross-substrate generality | stretch (dropped first) |

**Endgame:** two papers — Problem A to ALIFE/ECAL/GECCO (~month 6), Problem B to
Artificial Life / Chaos / Physical Review E, both with pre-registration and open
code + data.

---

## Immediate next actions
1. **Build the labelled corpus** — scale up soup runs; capture pre-transition state;
   dual-label a subsample. *(Fix checkpoint cadence with a pilot first — a coarse
   cadence manufactures a fake "history helps" result.)*
2. **Stand up the positive control** — a known bifurcation via `ewstools`; drive a
   calibrated classifier to **Gate 1 (AUROC ≥ 0.85)**.
3. **Then** a first real forecaster on the soup: "transition within H epochs?" with
   AUROC/AUPRC/Brier/lead-time.

## The rest of the map
- [`01_learning_prerequisites.md`](01_learning_prerequisites.md) — what to learn, tiered.
- [`02_implementation_plan.md`](02_implementation_plan.md) — the 9 stages with gates, tech stack, storage/compute budget.
- [`03_feasibility_novelty_audit.md`](03_feasibility_novelty_audit.md) — the doability / novelty / assumptions check in full.
- [`README.md`](README.md) — repo overview + how to reproduce Stage 1.
