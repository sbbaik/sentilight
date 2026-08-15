# Preregistration Directory

This directory contains the preregistration documents for the SentiLight study, frozen
**before** the corresponding results existed.

Three companion documents divide the record without overlap:
- [`PROVENANCE.md`](./PROVENANCE.md) — **time-stamp evidence**: the freeze-commit table,
  verification commands, and the honest limitations of git-based timestamping.
- [`VERDICTS.md`](./VERDICTS.md) — **outcomes** of every preregistered prediction and
  gate, including the ones that failed.
- This README — orientation and integrity rules only.

## Contents

| File | What was frozen | Paper section |
|---|---|---|
| `prereg_contamination_delta.md` | Predictions P1–P4 for the effect of removing input-only pretraining contamination, before the clean grid was trained | Appendix B |
| `prereg_teacher_consistency.md` | Withdrawal/adoption criteria for the continuous-label (regression-head) path: embedding-correlation risk line and teacher re-query noise-floor thresholds, before either diagnostic ran | §5.2, Appendix C |
| `prereg_w1_contrast_family.md` | The confirmatory contrast family (m = 8), the row-level McNemar → Stouffer → Holm procedure, and the adoption rule (Holm p < 0.05 ∧ per-seed sign consistency), before any row-level result existed | §5.5, §6.2 |
| `prereg_w2_constrained_decoding.md` | The nearest-tuple projection definition (equal-weight normalised distance, deterministic tie-breaking) and the three-way closure gate, before the projection ran | §6.5 |
| `eval_split_freeze.json` | SHA-256 freeze of all evaluation splits, before any model reported in the paper was trained on the verified corpus | §5.1 |

Freeze-commit hashes and verification commands for every file above: see
[`PROVENANCE.md`](./PROVENANCE.md). `SHA256SUMS` holds the freeze-time digests.

## Integrity rules

1. The four `prereg_*.md` files and `eval_split_freeze.json` are **content-frozen**.
   They may be moved or linked, never edited. Any correction is recorded in
   `VERDICTS.md` or the paper, not by rewriting the frozen document.
2. Git history is part of the evidence. This repository's history is never rebased,
   squashed, or force-pushed.
3. Where a preregistered criterion later proved miscalibrated (see P1–P3 in
   `VERDICTS.md`), the criterion is reported as failed and the miscalibration is
   analysed post hoc — the frozen document stands as written.
