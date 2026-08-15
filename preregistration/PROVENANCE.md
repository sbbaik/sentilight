# Preregistration provenance

Each protocol in this directory was committed to version control **before** the analysis it
governs was run. This file records the evidence for that ordering so a reader can ask for it
and an editor can check it.

## What the SHA-256 digests do and do not show

`SHA256SUMS` shows only that the files here are unmodified since they were written. On its
own a digest says nothing about *when* a document was written — a hash can be recomputed at
any time from any content.

The ordering claim rests on the version-control history below: each preregistration commit
precedes the commit that records the corresponding result, and git's commit graph chains
those commits cryptographically, so the sequence cannot be rearranged after the fact without
invalidating every descendant hash.

## Commit record

Repository: the authors' private working repository (not published — it contains internal
working notes and material excluded from release). Available to editors or reviewers on
request.

| Protocol | SHA-256 | Preregistration commit | Committed | Result commit | Committed | Gap |
|---|---|---|---|---|---|---|
| `prereg_contamination_delta.md` | `042a6865…6503` | `ab424ce` | 2026-08-05 18:02:03 +09 | `79fe7c6` | 2026-08-05 20:31:36 +09 | 2 h 30 m |
| `prereg_teacher_consistency.md` | `1968e03e…d3d3` | `cac3255` | 2026-08-05 18:23:13 +09 | `2ffc07e` | 2026-08-05 19:00:58 +09 | 38 m |
| `prereg_w1_contrast_family.md` | `f90ce79f…2d55` | `24ec122` | 2026-08-11 06:59:27 +09 | `e08f9a2` | 2026-08-11 07:02:32 +09 | 3 m 05 s |
| `prereg_w2_constrained_decoding.md` | `ea94b1d3…dcde` | `a000280` | 2026-08-11 07:25:46 +09 | `369fe87` | 2026-08-11 07:28:09 +09 | 2 m 23 s |

The remaining predictions in `prereg_contamination_delta.md` (P2 and P3, which depend on the
natural-segment evaluations) were adjudicated later, in commit `79c70c8`
(2026-08-11 05:17:53 +09), six days after the protocol was frozen.

Preregistration commit subjects, verbatim:

```
ab424ce  Preregister contamination-delta predictions before clean-corpus retraining
cac3255  Preregister teacher self-consistency gate and embedding diagnostic criteria
24ec122  Preregister W1 contrast family and Holm scope before running row-level tests
a000280  Preregister W2 headline gate; add W1 source decomposition; swap clean headline numbers
```

## Verification

The copies released here are byte-identical to the versions in the commits that introduced
them. Given access to the working repository, this is checked with:

```bash
git show ab424ce:<path>/prereg_contamination_delta.md   | sha256sum
git show cac3255:<path>/prereg_teacher_consistency.md   | sha256sum
git show 24ec122:<path>/prereg_w1_contrast_family.md    | sha256sum
git show a000280:<path>/prereg_w2_constrained_decoding.md | sha256sum
```

Each digest matches the corresponding line in `SHA256SUMS`. Confirmed for all four.

## An independent timestamp

The files in this directory are contained in the Zenodo archive of release v1.0.0,
**DOI [10.5281/zenodo.21930496](https://doi.org/10.5281/zenodo.21930496), published
2026-08-14**. Verified by downloading that record's archive and hashing its contents: the
four protocol documents inside it carry exactly the digests listed in `SHA256SUMS`.

Zenodo is operated by CERN and its publication date is not settable by the depositor, so it
is an independent attestation that this content existed in this form no later than that date
— which git commit timestamps, by themselves, are not.

Note what this does and does not establish. The Zenodo date is *later* than the results it
accompanies (2026-08-05 and 2026-08-11), so it anchors the **content**, not the ordering.
The ordering evidence remains the commit history above. The two are complementary: the
archive fixes what the documents say, the commit graph fixes when they were written relative
to the analysis.

## Limitations, stated plainly

1. **Git commit timestamps are set by the committer** and are not independently attested.
   They are evidence of ordering within a repository the author controls. The Zenodo archive
   above supplies an independent attestation, but only of content and only from 2026-08-14
   onward; no third-party timestamp covers the moment each protocol was written, because none
   was obtained at the time. Nothing added now could supply one retroactively.
2. **The gaps for the two row-level protocols are short** (2–3 minutes). That is what
   happened: the protocol was written immediately before the analysis was executed. The
   ordering is what the preregistration is for; the interval is reported rather than
   presented as further evidence.
3. **The working repository is not published.** It contains internal notes, planning
   documents and material excluded from this release. It can be provided for verification on
   request.

## Why the documents still reference unpublished files

The protocols cite internal working documents (`story_line.md`, `pre_result_NN.md`) that are
not part of this release, so those references do not resolve here. The files are left exactly
as written: editing them to tidy the links would change their digests and destroy the only
thing that ties them to the commits above.

A protocol that references the author's working notes is also, incidentally, evidence that it
is a genuine internal artifact rather than a document composed for publication after the
results were known.
