# Full Project Context — Handoff Brief for an LLM Collaborator

> **Purpose of this file.** You (an LLM) are being brought in as a collaborator on a
> research project with no prior conversation history. This document is your complete
> briefing: the scientific idea, the formal framework, the substrate mechanics, every
> decision already made, the current code state, and what remains. Treat it as ground
> truth. Where a fact is *verified against a source* it is marked ✅; where it is
> *assumed/plausible but unconfirmed* it is marked ⚠️ and must not be relied on without
> checking. The human is an undergraduate/early-grad student doing this as an individual
> 12-month research project; explanations should stay rigorous but not assume deep prior
> knowledge of information theory or artificial life.

---

## 0. TL;DR

We study **what makes an emergent event predictable at all**, reframing forecasting as a
*measurement instrument* and information as the *object of study*. The test system is the
**spontaneous emergence of self-replicating programs** in a random-code "primordial soup"
(from Agüera y Arcas et al., 2024, *Computational Life*, arXiv:2406.19108). The project
asks: before the transition happens, (A) can it be forecast, with what calibrated accuracy
and lead time, against a measured ceiling; and (B) *what information* makes any such
forecast possible — which representation carries it, how much of the system must be
observed, whether history helps, and where the information sits. **Problem A validates the
instrument; Problem B is the science.** Every outcome, including "nothing predicts
anything," maps to a stated, quantified conclusion (a bound in bits), so the project cannot
"fail" in the ordinary sense.

**Current state:** Stage 1 complete — the substrate and its detector are reproduced from
scratch, tested (5/5, incl. fuzz-equivalence over 3,000 random tapes), and
self-replicators demonstrably emerge. Code is public at
`github.com/never954/ml_forecasting` (MIT).

---

## 1. The central idea and why it is framed this way

Conventional framing asks "can we predict event X?" We instead ask **"what information
makes X predictable, and for which observer?"** The forecaster is treated as an instrument
that converts an *observation map* into a number; that number is only meaningful once you
know what the instrument was given. This turns a forecasting exercise into a *science of
predictability*.

**Governing question:** *What determines whether an emergent event is predictable?*

Three load-bearing commitments follow:

1. **Predictability is observer-relative** — an information-theoretic statement, not a
   philosophical one. For a target `Y` and representation `φ`, any predictor's performance
   is bounded by the **data-processing inequality**:
   `I(φ(X); Y) ≤ I(X; Y)`. Reality doesn't change with the observer; the *available*
   information does. Every predictability claim is indexed by **(target event, horizon,
   observation map)**; a claim missing any of the three is ill-formed. This is *why the
   representation comparison is the central experiment*, not an afterthought.

2. **A null result is a measurement, not a failure.** "Nothing predicts the event" is
   reported as a *measured upper bound in bits* (via committor variance and a Bayes/Fano
   bound), never asserted as an absence. This is the project's insurance policy.

3. **Forecasting performance is not the contribution.** The contribution is the
   *measurement framework*. Every component remains valid even if every forecasting model
   performs at chance.

### The five conceptual commitments (kept strictly apart)
- **Information** — a property of a joint distribution; model- and observer-independent.
- **Representation** — an observation map `φ` chosen by the observer; how much information
  it retains is never a property of the system alone.
- **Prediction** — a property of a model, representation, and dataset jointly.
- **Causation** — a property of the mechanism; testable only by intervention.
- **Explanation** — a human-facing account; interpretation, never a measurement.

Two consequences recur: (a) an event can have a complete causal account yet be
unpredictable (if the chain starts in a microscopic fluctuation amplified only afterward);
(b) a quantity can be strongly predictive while lying nowhere on the causal path.
**Information may be predictive without being causal.** Relatedly, **transfer entropy
quantifies directed *predictive* relationships, not causal effect** (Ay & Polani, 2008),
and is used here only in the predictive sense.

### The conceptual hierarchy (top = most fundamental)
```
Information         — what exists in the system, model-independently
 ↓ Predictability   — how much of it bears on the future event, for a given observer
 ↓ Prediction       — the instrument by which that is measured
 ↓ Representation    — which observation map carries the information
 ↓ Localisation      — how it is distributed across the system
 ↓ Minimal Predictive State (MPS)       — one method for "how much must be observed"
 ↓ Predictive microenvironment          — one frame for "where the information sits over time"
```
The bottom two are **instruments, not findings**. If they turn out to be existing
constructs, the framework above is unaffected (see §9).

---

## 2. The substrate (the test system) — mechanics and verified facts

The substrate is **BFF**, an extended Brainfuck, from Agüera y Arcas et al. (2024). It was
chosen because it has an **unambiguous emergent event, full observability, and
repeatability** — prebiotic chemistry and biological populations have none of these.

### 2.1 The BFF language ✅ (verified against the paper)
- Programs are **64-byte** tapes. Two programs are concatenated into one **128-byte** tape
  and executed together; both halves can be modified (self-modifying code — instructions
  and data share one tape).
- **10 valid instructions** out of 256 byte values (plus a "true zero" for loop exit); all
  other bytes are no-ops and act as data.
- Three pointers, all on the same tape: instruction pointer `ip`, read head `head0`,
  write head `head1`. Instruction set:

  | byte | char | action |
  |---|---|---|
  | 60 | `<` | head0 -= 1 |
  | 62 | `>` | head0 += 1 |
  | 123 | `{` | head1 -= 1 |
  | 125 | `}` | head1 += 1 |
  | 45 | `-` | tape[head0] -= 1 |
  | 43 | `+` | tape[head0] += 1 |
  | 46 | `.` | tape[head1] = tape[head0]  (copy read→write) |
  | 44 | `,` | tape[head0] = tape[head1]  (copy write→read) |
  | 91 | `[` | if tape[head0]==0: jump forward past matching `]` |
  | 93 | `]` | if tape[head0]!=0: jump back to matching `[` |

  Pointers wrap modulo tape length; byte arithmetic wraps modulo 256. Execution halts after
  a fixed instruction budget (paper: 2¹³ = 8192 reads) or on an unmatched bracket.

### 2.2 The primordial soup ("Turing gas") ✅
- A large population of programs (paper uses **2¹⁷ ≈ 131,072**), each 64 random bytes.
- **No programs are created or destroyed.** Change comes only from self-modification during
  interactions and a low rate of **background mutation**.
- **Each epoch:** random ordered pairs are selected, concatenated, executed, then split
  back into the population. **No fitness function, no selection operator, no seeded
  replicator.**
- Despite this, **self-replicators emerge spontaneously**: a long, apparently structureless
  "pre-life" phase, then a rapid takeover.

### 2.3 The transition detector: "high-order entropy" ✅
The paper's novel complexity metric, used here as the **ground-truth label** for when the
transition occurs:

`high_order_entropy = Shannon entropy per byte − normalized Kolmogorov complexity (K/n)`

Kolmogorov complexity is approximated by a real compressor (gzip/zlib) — a valid upper
bound on K. Intuition: it measures information explained *only by relations between bytes*.
For a random soup, bytes are ~uniform (Shannon ≈ 8 bits/byte) **and** incompressible
(K/n ≈ 8), so HOE ≈ 0. When replicators take over, the soup fills with repeated structure:
it compresses far below its byte entropy, K/n drops, and **HOE rises sharply**. That rise
is the transition signature.

### 2.4 Numbers to treat with care
- ⚠️ **"~40% of runs transition within 16,000 epochs."** Stated in the project proposal;
  **not** verified against the paper text in detail. Treat as a benchmark to *reproduce in
  Stage 1–2*, not as a given. (The paper's own figures show first-replicator events across a
  range of epochs and configurations.)
- ✅ Storage arithmetic: one 2¹⁷-program snapshot ≈ 2¹⁷ × 64 B ≈ **8 MB**; at a 10-epoch
  checkpoint cadence over ~16k epochs ≈ 1,600 snapshots ≈ **~13 GB/run** before
  compression. This is why **retrospective retention** (keep full pre-transition state only
  once `T` is known) is a requirement, not an optimization.
- ✅ Compute: paper-scale is CPU-bound and modest (~7 epochs/s/core; 400 runs ≈ ~230
  core-hours). Our from-scratch reproduction runs ~200 epochs/s at N=4096 with numba.

---

## 3. Formal framework (notation used throughout)

- Substrate = finite set of **sites** `V` (each holds a program tape). Micro-state at epoch
  `t` is `X_t`; history is `X_≤t`.
- `T` = first epoch at which the transition criterion fires.
- For horizon `H`, the **target** is `Y_t^H = 1[ t < T ≤ t+H ]` (does the transition happen
  within the next `H` epochs?).
- An **observation map** `φ` defines an observer. Information **present** = `I(X_≤t; Y_t^H)`;
  information **accessible** = `I(φ(X_≤t); Y_t^H)`. By DPI the latter ≤ the former. The
  **ratio** of the two is the quantity of interest; experiments sweep over `φ`.

### Coined operational constructs (explicitly *instruments*, not theory)
- **Minimal Predictive State (MPS):** the smallest subset of sites retaining a (1−ε)
  fraction of the available predictive information:
  `S*_ε(t) = argmin_{S⊆V} |S|  s.t.  I(φ_S(X_t); Y_t^H) ≥ (1−ε)·I(X_t; Y_t^H)`.
  - **Minimal predictive volume** `ν = |S*_ε| / |V|`. History-augmented variant gives
    `(S*, k*)`.
  - Three properties constrain it: (i) relative to target, horizon, and observation basis;
    (ii) the ε-tolerance makes it generally **non-sufficient** (so it sits *outside* the
    causal-state theorems); (iii) the optimization is **NP-hard**, so reported volumes are
    **upper bounds** (greedy selection).
- **Predictive microenvironment** `M_δ(t)`: the sites whose *marginal* contribution to
  predictive information exceeds `δ`. Defined by an information criterion, **not spatial
  adjacency** — members need not be neighbors. It is *not asserted* that microenvironments
  exist, are stable, or move coherently; persistence/merging/splitting/migration are
  empirical questions tested against a **permutation null**.

### Derived metrics
- **Predictive efficiency** `η = I(φ(X);Y) / I(X_≤t;Y)` — fraction of available information
  a given observer captures; the direct operationalization of observer-relative
  predictability. Reported as a ratio of bounds.
- **Minimal predictive volume** `ν` (above) — an upper bound.
- **Localisation index** `L(t)` = the **Gini coefficient** of per-site marginal
  contributions to predictive information.
- **Representation cost in bits observed** = the x-axis against which η is read (without it,
  efficiency comparisons are unanchored).

---

## 4. Research questions (RQ1–RQ8), each with a falsifier

- **RQ1** — Does the pre-event phase carry measurable information about whether/when the
  transition occurs? *Open. Falsified by chance-level AUROC with the positive-control gate
  passed.*
- **RQ2** — Which **tipping class** does the transition belong to, and does it predict which
  representations should work? *Hypothesis: waiting times closer to exponential than
  Weibull/gamma (memoryless nucleation) → low-dimensional critical-slowing-down (CSD)
  representations should fail. Falsified by a non-exponential fit plus trending CSD
  indicators.*
- **RQ3** — What is the intrinsic **predictability ceiling**, independent of representation
  and model? *No directional hypothesis; deliverable is `Var[q(x)]` and the implied
  Bayes/Fano bound in bits — the denominator for RQ4–RQ5.*
- **RQ4** — Which representation carries predictive information, and how much of the ceiling
  does each capture? *Hypothesis: efficiency monotone in refinement, diminishing beyond
  local neighborhoods. Falsified by non-monotonicity beyond CI, or no plateau.*
- **RQ5** — How much of the system must be observed? *Hypothesis: at ε=0.1, ν < 0.25 at
  horizons of 250–1000 epochs. Falsified if ν_0.1 > 0.75 at all horizons.*
- **RQ6** — Does history contribute information beyond the current observable state, and
  why? *Hypothesis: history helps under coarse representations only (an incomplete
  representation, not memory). Falsified if the benefit persists under oracle internals.*
- **RQ7** — Is predictive information **localised**, and does the localisation persist?
  *Hypothesis: consecutive-epoch Jaccard overlap exceeds the permutation null. Falsified if
  it does not.*
- **RQ8** — Does predictive information **concentrate** as the event approaches?
  *Hypothesis: `L(t)` rises over the final ~10% of the pre-transition window, Kendall τ
  above matched negative controls.*

RQ1–RQ3 = **Problem A** (instrument + ceiling). RQ4–RQ8 = **Problem B** (the science),
following the hierarchy of §1 in order: representation → how much → present-vs-history →
where it sits → how that changes.

---

## 5. The nine-stage programme (with decision gates)

| Stage | Objective | Measured / decision rule |
|---|---|---|
| **0** | Systematic review + novelty verdict | ≥120 records screened, ≥35 full-text; 5 closest prior works + the difference from each. **Gate: fix terminology before implementation.** |
| **1–2** | Reproduce substrate; build labelled corpus | Transition rate vs. the ~40%/16k benchmark; N≥400 runs + ≥100 each on two further instruction sets; ≥150 positive & ≥200 negative; dual-labelling on a subsample. |
| **3** | Calibrate the instrument (Problem A) | Kendall τ for variance & lag-1 autocorrelation vs. negative controls; AUROC/AUPRC/Brier/FPR/lead-time at K=100/250/500/1000. **Gate 1: AUROC ≥ 0.85 on a known bifurcation, else no conclusion about the soup is admissible.** |
| **4–5** | Predictability limits, tipping class, estimator validation | `Var[q(x)]` vs. null and the Bayes/Fano bound in bits; KS + AIC for exponential vs. Weibull vs. gamma; on synthetic ground truth recover ν within ±0.05 and I within ±0.1 bits, with provable history-helps / history-does-not variants. **Gate 2: no info-theoretic result on the soup is admissible until recovery passes.** |
| **6–7** | Representation comparison; how much must be observed | E1–E7 × four horizons at matched capacity; accuracy-vs-information curve and η per representation; ν_ε for ε∈{0.05,0.1,0.25} by three selection routes, with cross-route agreement + random-subset controls. |
| **8** | Localisation dynamics; state-vs-history | Jaccard persistence vs. permutation null; `L(t)` trend vs. negative controls; conditional MI `I(X_{t−k:t−1}; Y | X_t)` for k∈{10,50,100,500,1000}. |
| **9** | Cross-substrate generality (stretch) | Reduced repeat on Lenia or Biomaker CA; is the representation ordering and the sign of the `L(t)` trend preserved? First test of "about emergence vs. about this substrate." |

**Timeline (12 months):** framing + reproduced soup + pilot by month 1; full corpus by
months 2–3; positive control + EWS + forecaster (Gate 1) by months 3–4; committor + ceiling
+ tipping class + estimator validation (Gate 2) + **Problem A manuscript** by months 5–6;
E1–E7 + minimal-subset + localisation + state-vs-history by months 6–10; cross-substrate
(or dropped) months 10–11; **Problem B manuscript** + code/data release month 12.

---

## 6. The seven representations (E1–E7) — the central experiment

These are **not feature sets** but different *representations of the same underlying
information* — different answers to "what does the observer see" — varying along three axes
kept separate in analysis: **spatial extent, relational structure, temporal depth**.
Capacity and tuning budget are **matched across arms** so differences are attributable to
information content, not model power. Each arm is scored against chance and always-positive
baselines, and reported as a fraction of the Stage-4 ceiling.

- **E1** — a single global scalar per epoch (entropy or compressed size). *What classical
  EWS assumes.*
- **E2** — population statistics (k-mer spectra, diversity, moments); discards spatial &
  relational detail.
- **E3** — local interaction neighborhoods. *The representation-level form of the "how
  much" question.*
- **E4** — the interaction graph via a **GNN** over tapes and recent pairings; tests whether
  relational structure adds beyond site content.
- **E5** — the complete observable micro-state at a single epoch. *The Markov reference.*
- **E6** — complete state + k previous epochs.
- **E7** — full trajectory + **oracle simulator internals** (instruction pointers, execution
  traces, the pairing schedule); isolates variables normally hidden from an observer.

---

## 7. State-versus-history protocol (RQ6) — a key, subtle piece

**Position (tested, not assumed):** history is *not fundamentally necessary*. If the current
observable state forms a complete Markov representation, it screens off the past and
`I(X_{t−k:t−1}; Y | X_t) = 0`. So a measured benefit from history is, in the first instance,
**evidence about the observation map, not the dynamics.** Three competing explanations, made
distinguishable by the oracle:

- **(a) Incomplete/coarse observation** — the representation discards state that matters and
  history partially reconstructs it. *Signature: the benefit shrinks monotonically toward
  zero as representations refine from E1→E5.*
- **(b) Hidden variables** — the relevant state exists in the simulator but isn't part of
  what an ordinary observer sees. *Signature: benefit persists under E5 but vanishes when
  internals are exposed as the E7 oracle.* (Directly testable here in a way impossible in a
  natural system — a primary reason for choosing a simulated substrate.)
- **(c) Genuine system memory** — benefit persists even under oracle access. Since the
  simulator is a deterministic Markov update given full internal state + the pairing draw,
  this would indicate an **error** in the oracle definition or estimation — treated as a
  **diagnostic red flag, not a discovery.**

Stating this in advance prevents (a) or (b) from masquerading as (c).

---

## 8. Methodology: causality, leakage, estimation, metrics

**Causality & leakage (non-negotiable):**
- Every predictive window ends **≥ H epochs before `T`**.
- Splits are at the level of the **simulation seed**, never the window.
- Negative runs supply the negative class at **matched epoch distributions**; **epoch index
  is withheld from all models**.
- Labels are inherited from the published detector, with an independent lineage-tracing
  label on a subsample.
- Full state is checkpointed every ~10 epochs and retained retrospectively once `T` is
  known. **Checkpoint cadence is a controlled variable fixed by a pilot ablation** — a
  coarse cadence would manufacture a spurious "history helps" result (an artifact of the
  observation map masquerading as a property of the system). ⚠️ *This is one of the easiest
  ways to accidentally publish a wrong result; guard it.*

**Estimation (the hard part):**
- **McAllester & Stratos (2020):** any distribution-free high-confidence *lower* bound on MI
  from N samples is `O(ln N)`. **Therefore: report bounds and ratios, never trust point
  estimates in high dimensions.**
- **Workhorse estimator:** the achieved **log-loss of a calibrated predictor** — a valid MI
  lower bound via cross-entropy reduction against the base rate. Plug-in and KSG estimates
  alongside; variational bounds (MINE/InfoNCE) for corroboration only.
- Disagreements beyond confidence intervals are **reported, not resolved by preference**.
- Multiplicity across the representation × horizon grid handled by **Benjamini–Hochberg**.

**The predictability ceiling (the technical crux):**
- The **committor** `q(x)` = probability of hitting the transition before returning to the
  pre-life basin; it is the *ideal forecaster*. Estimated by deep learning (Li et al., 2019;
  Kang et al., 2024). `Var[q(x)]` yields a **Bayes/Fano bound in bits** — the denominator
  for all Problem-B efficiency numbers. ⚠️ **Everything in Problem B depends on getting this
  right; it is the single hardest engineering piece and gated by Gate 2.**

**Metrics:** AUROC + AUPRC (positive class is rare); Brier score + calibration curves (the
committor comparison needs probabilities, not just discrimination); median lead time (a
warning one epoch ahead is not a warning); false-positive rate on censored negative runs
(exposes an always-yes forecaster); `Var[q(x)]` + implied Bayes/Fano bound (converts "we
found nothing" into a ceiling in bits). Three derived comparison quantities: η, ν, L(t)
(§3).

**Tools rejected/demoted (do not rabbit-hole):** Integrated Information Φ (rejected —
variants disagree, intractable, not event-conditioned); Fisher information (rejected —
target is binary); compression-based complexity (retained *only* as the transition
detector, not as an information measure); partial information decomposition (demoted to
exploratory — redundancy/synergy split is non-unique).

---

## 9. Novelty audit (provisional verdicts, to be confirmed in Stage 0)

| Claimed contribution | Verdict | Reasoning |
|---|---|---|
| Integrated framework treating forecasting as an instrument for measuring the informational basis of predictability | **Possibly novel as an integration** | Every component exists separately (EWS, committors, IB, local information dynamics); the claim is only that they haven't been assembled into one measurement programme for an emergence event. |
| Forecasting the onset of spontaneous self-replication; tipping-class determination | **Likely novel** | Prior work detects post hoc (Agüera y Arcas) or seeds replicators (Tierra/Avida); EWS targets bifurcations elsewhere. Novelty of *application*, not method. |
| Representation comparison against a measured ceiling; localisation before an emergence event | **Possibly novel** | Representation ablations are routine in ML, but scoring them as fractions of an independently estimated predictability bound appears less common. Search incomplete. |
| MPS and predictive microenvironment as formal objects | **Probably existing** | MPS ≈ an information-bottleneck problem with a structured selector, adjacent to causal states / non-sufficient predictive memories; microenvironment ≈ a thresholded functional of a local-causal-state field. **Both are instruments — if confirmed existing, names are dropped, measurements retained.** |
| The memory/observation protocol; metrics η, ν, L | **Possibly novel** | The confusion is well-known but an explicit oracle-based protocol appears less common; each metric is standard individually, novelty claimed only for combined use on an accuracy–extent plane. |
| Committor & Bayes bounds; IB, transfer entropy, GNNs, MI estimators | **Clearly existing** | Standard apparatus adopted as tools; no methodological claim. |

**Biggest framing risk:** MPS and the microenvironment turn out to be renamings. Because
both sit at the *bottom* of the hierarchy and Stage 0 runs first with a pre-specified
decision rule, this is a **terminological loss, not a scientific one** — it costs
vocabulary, not experiments.

---

## 10. Risks and the agreed mitigation strategy

| Risk | Severity | Mitigation / what's lost if it bites |
|---|---|---|
| MI estimation in high dimensions | High | Log-loss lower bound as primary; the Stage-5 synthetic gate; all values as bounds. If the gate fails, scope narrows to E1–E3 and the narrowing is reported. |
| The transition may be genuinely unpredictable | High | Handled by design: positive control proves the pipeline works; committor + MI convert absence into a bound; waiting times supply the mechanism. A null is a result about information. |
| Constructs may be renamings; exact MPS selection intractable | Moderate | Stage 0 first, rule changes framing not experiments; three selection routes with cross-route agreement as a reliability statistic; ν stays an upper bound. |
| Storage-bound checkpointing; detector defines the event | Moderate | Retrospective retention + pilot fixes cadence; dual labelling with headline results re-run under both. A large label discrepancy is itself a finding. |
| Results may be substrate-specific | Moderate | Three instruction sets in phase I + the Stage-9 cross-substrate test; where generality can't be shown, conclusions stated as substrate-specific. |
| Two problems in one programme | Moderate | Stages 0–4 independently publishable by month 6; stages 5–8 depend on them but not conversely; Stage 9 dropped first. |

### Consensus reached with the human (the honest-scope position)
1. **This is ~2–3 person-years of work done fully, as a 12-month solo project. Ship Problem
   A as the real deliverable; treat Problem B as best-effort; drop Stage 9 first.**
2. **The committor is the crux** — respect Gate 2 before any Problem-B claim.
3. **Report MI as bounds/ratios, not point estimates.**
4. **Novelty is a careful integration + first application, stated honestly — not a new
   theorem.** Framing it this way reads as rigor and survives review better than
   over-claiming.
5. **Guard the two gates ruthlessly** — they are what make a null result informative rather
   than indistinguishable from a bug.
6. **Stage 1 deliberately reproduces published work** — the project's own contribution
   begins at Stage 3.

---

## 11. Current implementation state (what actually exists)

Repository: **`github.com/never954/ml_forecasting`** (public, MIT). Local path
`/Users/vedantghule/ml_forecasting`. **Note:** the local repo is an *isolated* git repo;
the user's home directory is (separately) a giant git repo — never operate on that.

### Files
- `00_START_HERE.md` — brief orientation (bird's-eye).
- `01_learning_prerequisites.md` — tiered learning path (info theory, EWS, committors, GNNs,
  JIDT), with a skip-list and a 1-month fast path.
- `02_implementation_plan.md` — the 9 stages as a build plan with gates, tech stack, storage
  budget, critical-path diagram.
- `03_feasibility_novelty_audit.md` — the doability/novelty/assumptions audit in full.
- `README.md` — repo overview + reproduce instructions.
- `progress_stage1.html` — a presentation page (the emergence result, tests, roadmap).
- `simulator/bff.py` — the BFF interpreter + primordial-soup epoch loop, **numba**-JIT
  (~200 epochs/s at N=4096). Implements exactly the instruction table in §2.1; `ip` wraps
  modulo tape length; halts on step cap or unmatched bracket.
- `simulator/detector.py` — high-order-entropy detector (Shannon − zlib-compressed
  bits/byte).
- `simulator/test_bff.py` — **5/5 tests pass**, including **fuzz-equivalence** of the fast
  numba interpreter against a transparent pure-Python reference over **3,000 random tapes**
  (byte-for-byte identical final states), plus targeted semantic tests (increment on a data
  cell, cross-head copy, bracket-matched clear loop, block copy).
- `simulator/run_soup.py` — runs a soup, logs the HOE trajectory to CSV, saves final soup.
- `simulator/plot_run.py` — plots the emergence curve + marks the detected transition.
- `results/run1_emergence.png`, `results/run1.csv`, `results/run2*` — reproduced results.

### Reproduced results (real, from this project's code)
- **run1** (N=4096, plen=64, epochs=20000, max_steps=2048, mutation=1.5e-4, seed=1): HOE
  rises from ≈0 (random soup) to a **peak of ~1.81 bits/byte** near epoch ~200, then settles
  into a "living" steady state around ~0.65. Shannon entropy falls 8.0→~6.8 while the
  compressed size falls further (Kolmogorov 8.0→~6.2) — the widening gap *is* the emergent
  structure.
- **run2** (N=8192, epochs=16000, mutation=2e-5, seed=7): same qualitative story, steady
  state ~0.4.
- ⚠️ The transitions here are **fast** (tens–hundreds of epochs) because the soup is small
  and mutation seeds diversity quickly. The paper's characteristic *long* pre-life phase
  (transition ~epoch 12k–16k) needs the full 2¹⁷ soup (~30 min/run) — not yet run. The
  *phenomenon* is reproduced; matching the exact pre-life duration is a Stage-1–2 task.

### Immediate next actions
1. **Build the labelled corpus** — scale runs; capture pre-transition state at a
   pilot-fixed cadence; dual-label a subsample.
2. **Stand up the positive control** via `ewstools` (a known bifurcation); drive a
   calibrated classifier to **Gate 1 (AUROC ≥ 0.85)**.
3. **First real forecaster on the soup** — "transition within H epochs?" reporting
   AUROC/AUPRC/Brier/lead-time, with strict seed-level splits and epoch index withheld.

---

## 12. Key references
- Agüera y Arcas, B., et al. (2024). *Computational Life: How Well-formed, Self-replicating
  Programs Emerge from Simple Interaction.* arXiv:2406.19108. **(the substrate)**
- Ashwin, P., et al. (2012). Tipping points in open systems. *Phil. Trans. R. Soc. A* 370.
  **(tipping taxonomy)**
- Ay, N., Polani, D. (2008). Information flows in causal networks. *Adv. Complex Syst.* 11.
- Boettiger, C., Hastings, A. (2013). No early warning signals for stochastic transitions.
  *Proc. R. Soc. B* 280.
- Bury, T. M., et al. (2021). Deep learning for early warning signals of tipping points.
  *PNAS* 118(39). **(+ `ewstools`, the positive control)**
- Crutchfield & Young (1989); Shalizi & Crutchfield (2001). **(causal states / comp.
  mechanics)**
- E & Vanden-Eijnden (2010); Li et al. (2019); Kang et al. (2024). **(committor / transition
  path theory)**
- Lizier, J. T., et al. (2008). Local information dynamics. *Phys. Rev. E* 77. **(+ JIDT)**
- Löhr & Ay (2009). Non-sufficient memories sufficient for prediction. **(why an ε-tolerant
  MPS is legitimate)**
- McAllester & Stratos (2020). Formal limitations on measuring mutual information. *AISTATS
  108.* **(why bounds, not point estimates)**
- Rupe & Crutchfield (2018). Local causal states. *Chaos* 28. **(the microenvironment's
  likely home)**
- Scheffer, M., et al. (2009). Early-warning signals for critical transitions. *Nature* 461.
- Tishby, Pereira, Bialek (1999). The information bottleneck. **(the MPS's likely home)**

---

## 13. Glossary (quick reference)
- **BFF** — extended Brainfuck; the 10-instruction self-modifying language of the substrate.
- **Soup / Turing gas** — the population of programs that interact pairwise each epoch.
- **High-order entropy (HOE)** — Shannon entropy − compressed size; the transition detector.
- **Transition / `T`** — the epoch self-replication takes over (pre-life → life).
- **Horizon `H`** — how far ahead the forecast looks; target `Y_t^H = 1[t<T≤t+H]`.
- **Observation map `φ`** — what/how an observer sees; defines the representation.
- **DPI** — data-processing inequality: `I(φ(X);Y) ≤ I(X;Y)`; the backbone of
  observer-relative predictability.
- **Committor `q(x)`** — probability of reaching the transition before returning to pre-life;
  the ideal forecaster; `Var[q]` → the predictability ceiling in bits.
- **EWS / CSD** — early-warning signals / critical slowing down (rising variance &
  autocorrelation near tipping); the E1 baseline.
- **MPS / ν** — minimal predictive state / minimal predictive volume ("how much must be
  observed").
- **Predictive microenvironment / `M_δ(t)` / `L(t)`** — where predictive information sits and
  how concentrated it is (Gini of per-site contributions).
- **η** — predictive efficiency = fraction of the available information (the ceiling) a
  representation captures.
- **Oracle (E7)** — access to hidden simulator internals; used to separate "incomplete
  observation" from "genuine memory."
- **Gate 1 / Gate 2** — AUROC≥0.85 on a positive control / estimator recovery on synthetic
  data; hard stops before trusting soup results.
