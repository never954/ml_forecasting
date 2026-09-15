# Cross-Reference: Feasibility, Novelty & Assumption Check

Honest audit of the proposal against three questions: **is it doable, is it novel, and are any assumptions wrong?** Verified external facts are marked ✅ (checked against the source paper / library docs); ⚠️ = a real risk to manage; 🚩 = something to fix or watch.

**Headline verdict:** This is an unusually rigorous, self-critical proposal. It has *already done* an honest novelty audit on itself and builds falsifiers + fallbacks into every stage. Novelty is **real but modest and correctly stated** — it's an *integration + application*, not a new theorem. No fundamentally broken assumptions. The dominant risk is **scope for one person in 12 months**, not correctness.

---

## 1. Are the external facts right?

| Claim in proposal | Status | Notes |
|---|---|---|
| Substrate = spontaneous self-replicators in random code, no fitness/seed | ✅ | Matches Agüera y Arcas et al. 2024 exactly. |
| BFF = extended Brainfuck, programs 64 bytes, shared instruction/data tape | ✅ | Confirmed in the paper. |
| ~10 valid instructions of 256 byte-values | ✅ | Confirmed (10 instructions + a "true zero"). |
| Soup fully observable down to every byte, cheap to replay | ✅ | True — deterministic given the pairing draw; open source. |
| Compression-based "high-order entropy" detects the transition post-hoc | ✅ | This is *their own* introduced metric; using it as a label is sound. |
| "≈40% of runs transition within 16,000 epochs" | ⚠️ | Plausible and order-of-magnitude right, but I could **not** verify the exact 40%/16k figure from the paper text (their soup uses 2¹⁷ programs; first-replicator timing varies a lot by config). **Treat as a Stage-1 quantity to reproduce, not a given.** The proposal already makes this a measured benchmark, which is correct. |
| Storage "low tens of GB per run" | ✅ | Checks out: 2¹⁷ × 64 B ≈ 8 MB/snapshot × ~1,600 snapshots ≈ ~13 GB/run. Consistent. |
| Compute ~230 core-hours for 400 runs at 7 epochs/s/core | ✅ | Arithmetic is self-consistent (~0.6 core-hr/run). |
| `ewstools` provides CSD indicators + DL classifier + known-bifurcation models | ✅ | Confirmed (Bury; JOSS 2023). Good choice for the positive control. |
| McAllester–Stratos: distribution-free MI lower bounds are O(ln N) | ✅ | Correct, and correctly used to justify reporting bounds not point estimates. |
| Transfer entropy ≠ causation (Ay & Polani) | ✅ | Correctly stated and correctly restricted to predictive use. |

**No factual red flags.** One number (40%/16k) to confirm empirically rather than trust.

---

## 2. Is it novel?

The proposal's own novelty audit is well-calibrated — I largely agree with it:

| Contribution | Their verdict | My assessment |
|---|---|---|
| Forecasting as an *instrument* to measure the informational basis of predictability | "Possibly novel as an integration" | **Agree.** Every piece exists (EWS, committors, IB, local info dynamics); assembling them into one measurement programme for an emergence event does appear novel. The novelty is *the framing*, and they say so. |
| Forecasting *onset of spontaneous self-replication* + tipping-class determination | "Likely novel" | **Agree.** Prior ALife (Tierra/Avida) *seeds* replicators; Computational Life detects *post hoc*. Forecasting the onset before it happens is genuinely new. Novelty of *application*, not method. |
| Representation comparison scored as fractions of a measured predictability ceiling | "Possibly novel" | **Agree, with caution.** Representation ablations are routine in ML; scoring them against an *independently estimated committor ceiling* for a distant global event is uncommon. Defensible. |
| MPS + predictive microenvironment as formal objects | "Probably existing" | **Strongly agree** — likely an information-bottleneck problem with a structured selector / a thresholded local-causal-state field. **This is the single biggest novelty risk, and the design correctly neutralises it:** these sit at the *bottom* of the hierarchy, Stage 0 checks them first, and losing them costs *vocabulary, not experiments.* |
| η, ν, L metrics; oracle memory protocol | "Possibly novel" | Fair. Novelty claimed only for *combined* use. Reasonable. |
| Committors, IB, GNNs, MI estimators | "Clearly existing" | Correct — adopted as tools. |

**Bottom line on novelty:** solid enough for the target venues (ALIFE/ECAL/GECCO, Artificial Life/Chaos/PRE). It's an honest "novel integration + first application," *not* a claimed breakthrough — which is exactly the right register and is more likely to survive review than an over-claim.

---

## 3. Are any assumptions wrong? (the important part)

Most things I'd normally flag are **already handled as testable hypotheses with falsifiers**, not baked-in assumptions. That's the proposal's biggest strength. What remains:

**Genuinely well-handled (not assumptions — they're tested):**
- *Predictability is observer-relative* → this is just the Data Processing Inequality. Correct, not an assumption.
- *CSD/EWS should work* → **not** assumed; RQ2 predicts it may *fail* if the transition is noise-induced nucleation, and treats that as a result. ✅ Good.
- *History helps because of memory* → explicitly **rejected as the default**; the oracle protocol (E5 vs E7) distinguishes incomplete observation from hidden variables from genuine memory. ✅ Genuinely clever, and only possible *because* it's a simulator.
- *A null result = failure* → converted into a **measured bound in bits** via the committor. ✅ This is what makes the project robust to "nothing predicts anything."

**Real risks to manage (⚠️):**
1. **Scope.** ⚠️ *This is the #1 concern.* 9 stages, 3 instruction sets, ≥400 runs, a trained neural committor, 7 representations × 4 horizons, three MPS selection routes, localisation dynamics, plus a cross-substrate stretch — as an **individual** 12-month project. Realistically this is closer to 2–3 person-years done fully. **Mitigation is in the design** (Problem A ships at month 6 independently; Stage 9 drops first). Advice: **treat Problem A as the real deliverable and Problem B as best-effort.** Guard the two gates ruthlessly; they prevent months of wasted work on a broken instrument.
2. **Committor estimation is the technical crux.** ⚠️ Defining reactant/product basins and training a reliable q(x) on a discrete, high-dimensional, *non-stationary* pre-life soup is the hardest single piece, and the entire Problem-B denominator depends on it. Gate 2 (synthetic recovery) is the right safeguard — do not skip it.
3. **Rare-positive statistical power.** ⚠️ Binary, rare target + high-dim representations = the MI-estimation regime McAllester–Stratos warns about. The log-loss-lower-bound workhorse is the correct call, but expect **wide confidence intervals**; the honest output is often "η is bounded in [a,b]," not a crisp number. Plan sample sizes for *bounds*, not point estimates.
4. **Non-stationarity vs. causal-state machinery.** ⚠️ The pre-transition soup is explicitly non-stationary, which is *why* classical causal states don't cleanly apply — the proposal knows this (it's the stated gap) and the ε-tolerant MPS is deliberately non-sufficient. Just be aware some info-theory estimators assume stationarity; validate on the non-stationary synthetic in Gate 2.

**Potential 🚩 to watch:**
- 🚩 **Checkpoint cadence artifact.** If cadence is too coarse, you *manufacture* a fake "history helps" result (an artifact of the observation map masquerading as system property). The proposal catches this and fixes cadence via a Stage-1 pilot ablation — **make sure that pilot actually happens before bulk generation.** This is the easiest way to accidentally publish a wrong result.
- 🚩 **The 40%/16k benchmark** (see §1) — confirm empirically in Stage 1; don't design storage/compute around an unverified rate.
- 🚩 **Label definition = the detector.** Your event is *defined by* the compression detector. The dual-lineage labelling on a subsample is the right hedge; a large discrepancy is a finding, not an error — keep that framing.

---

## One-paragraph summary for a supervisor
The science is sound and the assumptions are, for the most part, *hypotheses with falsifiers* rather than unexamined commitments — the mark of a careful design. External facts check out (one transition-rate number to reproduce, not trust). Novelty is real but honestly scoped as integration-plus-first-application, appropriate for the named venues. The two things that will actually determine success are **(1) keeping scope honest — ship Problem A, treat Problem B as best-effort — and (2) getting a trustworthy committor through Gate 2.** If both gates are respected, even a fully negative outcome yields a publishable, quantified result. Recommend proceeding, with an explicit conversation about de-scoping Problem B/Stage 9 up front.
