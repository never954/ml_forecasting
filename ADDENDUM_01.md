# ADDENDUM 01 — Clarifications & Strategic Updates (verified/corrected)

> **Status:** This is the reviewed, corrected version of a draft addendum. Read it as a
> patch to `LLM_CONTEXT.md`. Each section states the decision, then a **✅ verified / ⚠️
> corrected** note so you know what was changed and why. Where this addendum and the base
> brief differ, **this addendum wins**.

---

## 1. Task formulation: strict binary classification (no regression)

- **Decision.** The forecasting task is **supervised binary classification over a sliding
  window**, never regression on "time-to-transition." Run the sim, detect transition epoch
  `T`, then recursively label every prior snapshot: for snapshot at `t` and horizon `H`,
  `Y = 1 if t < T ≤ t+H, else 0`.
- **Reason.** Many runs **never** transition (the censored negative class); a time-to-event
  regression has no finite target for them.
- ✅ **Verified — matches the base target `Y_t^H = 1[t<T≤t+H]` exactly.**
- ⚠️ **Two clarifications (important):**
  1. `H` is **swept** — this is a *family* of binary tasks, one per horizon
     `H ∈ {100, 250, 500, 1000}`. Every metric and every η is reported per-horizon.
  2. The regression ban applies to the **forecasting instrument only**. The **committor
     `q(x)` is a probability in [0,1]** — estimating it (Stage 4–5, the ceiling) is a
     legitimate regression/probability-estimation task and is *not* covered by this ban.
     Do not conflate the two.

---

## 2. E4 (GNN) architecture: keep it plain — spectral encodings NOT adopted

- **Decision.** E4 is a **plain message-passing GNN** over the interaction graph: node
  features = program/tape content, edges = recent pairings. Its only job is to test whether
  **relational structure adds predictive information beyond node content (E5)** at matched
  capacity.
- **Rejected suggestion — Laplacian spectral positional encodings.** A draft proposed adding
  spectral PE (lowest Laplacian eigenvectors) to beat the 1-WL limit / over-squashing. **We
  are not adopting this**, because the premise fails for our substrate: the base BFF soup is
  **well-mixed** — pairs are drawn uniformly at random each epoch — so the interaction graph
  is essentially a **random matching with no stable macroscopic structure** for eigenvectors
  to encode. Spectral PE on a random graph encodes noise, can degrade the arm, and adds
  sign/basis ambiguity. It is a correct technique aimed at the wrong problem.
- **When (and only when) to revisit.** If E4's graph is later built from **accumulated
  interaction history** or a **spatial soup variant** (2D, neighbor-only interactions) — i.e.
  a graph with genuine geometry — positional encodings become worth testing, at which point
  they must use sign-invariant handling and a random-PE control. Not before.
- **If E4 ≈ E5 in the well-mixed soup, that is a finding, not a failure:** relational
  structure carries no extra predictive information beyond node content when the soup has no
  structure.

---

## 3. Estimator constraints: expressive, and matched *within one family*

- **Decision (draft).** Linear models (plain logistic regression) are insufficient; use
  non-linear estimators at matched capacity so a null means "no information," not "a linear
  boundary couldn't bend."
- ✅ **Verified reasoning.** Correct, and it matters doubly here: the **log-loss MI lower
  bound is only as tight as the estimator**, so a weak estimator *understates* the
  information and would make a representation look worse than it is.
- ⚠️ **Corrected — "matched capacity across different families" is ill-defined:**
  - MLP vs. tree-ensemble vs. kernel-SVM have different inductive biases; you cannot match
    "capacity" across them in a way that isolates the representation. **Standardize on ONE
    primary expressive family** — a **deep MLP** — with **matched capacity (width/depth/
    regularization/tuning budget) across all arms E1–E7.** Use tree-ensembles / kernel SVMs
    (small feature spaces) **only as robustness checks**, not as the primary comparison.
  - **Tightest-bound reporting:** the max of several valid lower bounds is still a valid lower
    bound, but selecting the best estimator on the same data inflates it — evaluate and select
    on **held-out** folds. Report the achieved log-loss lower bound with its CI.

---

## 4. Gate 1 as a positive control for the *pipeline*

- **Decision.** Gate 1 (a known bifurcation via `ewstools`, AUROC ≥ 0.85) is **not** a proof
  that ML classification works in principle. It is a **strict positive control for the
  measurement instrument**: it proves the specific rolling-window pipeline, retrospective
  labeling, splits, and metric calculators **do not leak** and **do** detect predictive
  information where the physics guarantees it exists.
- ✅ **Verified — exactly matches the base brief** ("two of three gates test the instrument,
  not the hypothesis"; a null is only informative once the instrument is known to work).
- ⚠️ **One addition:** the control must run through the **identical pipeline code path** as
  the soup (same windowing, labeling, seed-level splitting, epoch-index withholding, metric
  code). A separate/parallel implementation would test different code and defeat the purpose.
- **The payoff (state this to reviewers):** Gate 1 passes **and** the soup yields AUROC ≈ 0.5
  ⇒ we can claim the event is **physically unpredictable at that horizon/representation**,
  not that the pipeline is broken.

---

## 5. Gate 2: the predictability ceiling in bits

- **Decision.** The committor `q(x)` is the ground-truth probability that micro-state `x`
  transitions before returning to the pre-life basin. Use it to compute the maximum possible
  predictive information (the ceiling), in bits:
  `I(X; Y) = H(Y) − E[ H_b(q(X)) ]`,
  where `H(Y)` is the base-rate (marginal) entropy and `H_b(·)` is the **binary** entropy of
  the per-state probability. This ceiling is the denominator of the Y-axis; each model's
  `η = I(φ(X);Y)/I(X;Y)` is a percentage of it, plotted against representation cost (bits
  observed) on the X-axis.
- ✅ **Verified — the formula is correct.** It is the standard MI decomposition
  `I = H(Y) − H(Y|X)` with `H(Y|X=x) = H_b(q(x))` for a binary target. Using MI-in-bits is
  actually **better than the base brief's `Var[q]`**, because MI is the natural, consistent
  denominator for `η` (a ratio of informations). Adopt it as the primary ceiling.
- ⚠️ **Three corrections/clarifications:**
  1. **Horizon consistency.** The classical (transition-path-theory) committor is
     **horizon-free** — P(*ever* reaching life before returning). Our target is bounded to
     **within H**. So for each horizon you need the **finite-horizon committor**
     `q_H(x) = P(t < T ≤ t+H | X_t = x)`, and the ceiling is
     `I(X;Y^H) = H(Y^H) − E[H_b(q_H(X))]`. Do not plug the raw TPT committor into a
     finite-horizon target.
  2. **Keep the complementary reports.** Retain `Var[q_H(X)]` and the **Fano error-rate
     bound** from the base brief alongside the MI ceiling — they summarize predictability in
     error terms and cross-check the MI number.
  3. **Ceiling vs. achieved vs. Gate 2.** The committor gives the **ceiling** (an *upper*
     estimate of `I`, only as good as the fitted `q`); a **calibrated classifier's held-out
     log-loss** gives the **achieved** value (a *lower* bound on `I`). `η` is achieved/ceiling.
     Because a mis-estimated `q` corrupts the denominator, **Gate 2 (recover ν within ±0.05
     and I within ±0.1 bits on synthetic ground truth, incl. calibrated `q`) is exactly the
     guard** — no soup ceiling is admissible until it passes.

---

## Compatibility summary (what changed vs. the base brief)
- **Reinforced, no change:** §1 binary target; §4 Gate 1 as instrument control.
- **Improved:** §5 adopts **MI-in-bits** (via the committor) as the primary ceiling and η
  denominator — cleaner than `Var[q]`, which is retained as a complementary report.
- **Rejected:** §2 spectral positional encodings — dropped for the well-mixed soup (no graph
  structure to encode); E4 stays a plain GNN, revisited only if a spatial/history-accumulated
  graph is introduced.
- **Corrected:** §3 standardize on one expressive estimator family with matched capacity,
  others as robustness only; §5 use the **finite-horizon** committor `q_H`, not the
  horizon-free TPT committor.
- **Net effect on scope:** none of this enlarges the programme; it tightens E4, the estimator
  protocol, and the ceiling definition so the results survive review.
