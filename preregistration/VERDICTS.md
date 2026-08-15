# VERDICTS — Outcomes of All Preregistered Predictions and Gates

This file closes the freeze → run → adjudicate chain for every preregistration in this
directory. It reports **all** outcomes, including failed predictions. Effect sizes are
percentage points (pp) on the rule segment (1,000 rows) unless noted; full per-seed
tables are in the released bundle (`run_summaries/`, `per_row/`).

---

## 1. `prereg_w1_contrast_family.md` — confirmatory contrast family (m = 8)

Procedure: per-seed exact-binomial McNemar on paired per-row outcomes → Stouffer
combination across seeds 42/43/44 → Holm correction over m = 8.
Adoption rule: Holm-corrected p < 0.05 **and** consistent effect sign across all three
seeds. Adjudicated once, on the final verified backbone.

| Contrast | Metric | Mean Δ (pp) | Stouffer z | Holm p | Seed signs | Verdict |
|---|---|---:|---:|---:|---|---|
| C1 (B2₂₃ − gen60) | coarse | +17.63 | +19.31 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C1 | strict | +8.80 | +11.71 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C2a (A2 − A1) | coarse | +14.33 | +16.59 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C2a | strict | +10.73 | +14.28 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C2b (B2 − A2) | coarse | +5.87 | +8.93 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C2b | strict | +1.93 | +3.08 | 2.09×10⁻³ | +/+/− | **REJECTED** (seed-sign inconsistency; seed 44: −0.20pp) |
| C2c (B2 − A1) | coarse | +20.20 | +22.08 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |
| C2c | strict | +12.67 | +16.44 | <10⁻¹⁶ | +/+/+ | **ADOPTED** |

Adopted 7/8. Notes recorded at adjudication:
- C1 strict: seed-level paired t (df = 2) never cleared 0.05 (0.056–0.057 across
  conditions) while the row-level procedure did; per the frozen precedence rule the
  row-level verdict stands, and the disagreement is reported (paper §6.2) as a
  power finding, not suppressed.
- C2b strict clears Holm on combined p but is rejected on the sign rule — the frozen
  rule was applied against the numerically favourable outcome.
- Post-hoc sensitivity (disclosed, paper §6.6): one-third of the C1 strict effect
  (2.87 of 8.80pp) is attributable to the H clause alone; confirmatory weight rests on
  C1 coarse and C2a.

## 2. `prereg_w2_constrained_decoding.md` — projection control gate

Frozen: nearest-tuple projection (equal-weight normalised 5-D distance, deterministic
tie-breaking), scored by the unmodified benchmark scorer; three-way closure gate.

| Quantity | Preregistered expectation | Measured | Verdict |
|---|---|---|---|
| Closure (coarse) | 0.33–0.66 | **−0.004** | Expectation **MISSED** (recorded); gate branch "closure < 0.33" taken |
| C1′ (B2 vs projected gen60), coarse | adoption test | +17.70, z = 19.08 | **ADOPTED** |
| C1′ strict | adoption test | +9.27, z = 12.19 | **ADOPTED** |

Mechanism finding (paper §6.5): ~80% of generative outputs were already exactly valid
policy tuples (rows changed by projection: 19.0% / 20.5% for 23M / 60M; mean normalised
projection distance 0.03). The preregistration's premise — that generative models
frequently leave the valid set — was itself wrong; the failure mode is selection, not
validity. Side observation: projection slightly *reduced* strict (−0.43 / −0.47pp).

## 3. `prereg_contamination_delta.md` — input-only contamination effect (P1–P4)

Paired dirty/clean retraining over the (size × seed × fraction) grid; tokenizer, steps,
seeds, hyperparameters held fixed.

| Prediction | Criterion as frozen | Outcome | Verdict |
|---|---|---|---|
| P1 (rule Δ within ±3pp per cell) | cell-level | strict 18/36, coarse 27/36 cells exceeded | **CRITERION NOT MET**; mechanism supported (mean Δ −0.23 / −1.05pp, sign-balanced; paired ΔSD below independent-retraining noise expectation). Post-hoc analysis: the cell-level threshold was uncalibrated against per-cell seed SD (4.09 / 5.56pp) and was unattainable by construction |
| P2 (natural drops ≤5pp after cleaning) | direction + magnitude | no measurable drop: +0.26 / +0.64pp, within noise | **CRITERION NOT MET** as worded (no directional effect detected at all) |
| P3 (\|natural Δ\| > \|rule Δ\|) | cell-level comparison | failed; per-stage measurement noise differs (rule seed SD ≈ 4.1 vs natural ≈ 1.2) and the prediction did not account for it | **CRITERION NOT MET** |
| P4 (C1/C2 signs preserved on clean backbone) | sign preservation | all contrast signs preserved | **MET** |

Standing conclusion (paper Appendix B): label-unexposed, single-occurrence, input-only
pretraining contamination did not convert into measurable performance gain at this scale
(3M–60M, 5.1M tokens). This measurement does **not** rehabilitate the natural segment
for evaluation (exclusion stands on the label diagnostic, §5.2). Lesson recorded: do not
preregister cell-level pp thresholds on a 3-seed grid without a power calculation
against existing seed variance.

## 4. `prereg_teacher_consistency.md` — continuous-label path gates

| Gate | Frozen threshold | Measured | Verdict |
|---|---|---|---|
| Embedding–label correlation risk line | ρ ≥ −0.05 ⇒ risk | ρ = **+0.007** (10^5 random pairs) | Risk line crossed; continuous-mapping hypothesis **REJECTED** |
| Teacher re-query noise floor (300 sentences × 5) | withdraw if > 3.3 (50% of the 6.6 quantisation floor) | ΔE00 = **7.24** (SNR 0.91) | **WITHDRAWN** — regression-head path abandoned per the pre-committed criterion |

Consequences (paper §5.2, Appendix C): the natural segment's teacher labels carry no
usable continuous signal; the apparent diversity (17,545 unique tuples / 38,000 rows) is
prompt-induced within-range sampling. The natural segment is excluded from evaluation;
the paper's claims are confined to policy compliance on the rule segment.

---

*Scope: this file records outcomes only. Freeze commits, verification commands, and the
limitations of the time-stamp evidence are documented in [`PROVENANCE.md`](./PROVENANCE.md).
Per-row outputs, run summaries, and analysis scripts sufficient to recompute every number
in this file are in the released bundle.*
