"""Single canonical source for the lighting value ranges and the evaluation
tolerances.

Before this module the same quantities were restated in five places and did not
agree. The two disagreements were:

  1. Hue normalization. backend/models/compact_llm/training_data.py normalized
     hue as `normalize_hue(clamp_int(h, 0, 360))` -- clamp first, then mod. That
     destroys wraparound: 370 collapses to 0 instead of 10, and -10 collapses to
     0 instead of 350. The evaluation side (backend/common/benchmark_eval.py)
     applies no normalization at all and instead mods inside hue_distance() and
     hue_coarse_bin(), which wraps correctly. Label generation and scoring
     therefore disagreed on any out-of-range hue.

  2. CT floor. The prompt shown to every model declares CT 153-500
     (backend/server/adapters/base.py) and the coarse binning uses
     COARSE_CT_MIN = 153, but label generation, both runtime clamp tables and
     the shared schema used 150.

Empirical impact of both, measured over all 162,025 per-row records in 164
prediction files before the fix: ZERO. Observed hue range was [0, 359] with no
out-of-range or negative values, and the observed minimum CT was exactly 153.
Both were latent defects, not result-changing ones. They are fixed here, and
tools/check_lighting_spec_consistency.py asserts the fix holds.

Canonical rule: hue is CIRCULAR and is normalized by modulo 360 (never clamped);
every other dimension is LINEAR and is clamped to its inclusive range.
"""
from __future__ import annotations

from typing import Any

# --- value ranges -----------------------------------------------------------
# Hue is deliberately absent from CLAMP_RANGES: it must never be clamped.
HUE_MODULUS = 360           # valid hue is 0..359 after normalization
CT_MIN = 153                # Tasmota mired floor, as declared in the model prompt
CT_MAX = 500

CLAMP_RANGES: dict[str, tuple[int, int]] = {
    "S": (0, 100),
    "B": (0, 100),
    "Dimmer": (0, 100),
    "CT": (CT_MIN, CT_MAX),
}

# --- policy constants -------------------------------------------------------
# resolve_case_output() snaps hue to the colour anchor when it drifts further
# than this. Note this is a SNAP-TO-ANCHOR, not a clamp to the boundary: a hue
# 25 degrees off the anchor becomes the anchor exactly, not anchor+18.
HUE_ANCHOR_SNAP_DEGREES = 18

# --- evaluation tolerances --------------------------------------------------
# Used by benchmark_eval.semantic_pass(). Hue uses circular distance; the rest
# use absolute difference. All bounds are INCLUSIVE (a violation requires >).
STRICT_TOLERANCES: dict[str, int] = {
    "H": 20,        # circular degrees
    "S": 20,
    "B": 20,
    "Dimmer": 20,
    "CT": 80,       # mireds
}

# Per-emotion constraints the prediction must additionally satisfy, as
# (dimension, minimum_or_None, maximum_or_None).
EMOTION_CONSTRAINTS: dict[str, tuple[tuple[str, int | None, int | None], ...]] = {
    "sadness":   (("B", None, 40), ("CT", 340, None)),
    "happiness": (("B", 70, None), ("CT", None, 290)),
    "anger":     (("S", 75, None), ("B", 60, None)),
    "calm":      (("S", 30, 70), ("B", 45, 80)),
    "anxiety":   (("B", None, 45), ("CT", 340, None)),
}
# Applied when the row names neither a base colour nor an emotion.
NEUTRAL_FALLBACK_CONSTRAINTS: tuple[tuple[str, int | None, int | None], ...] = (
    ("S", 15, 45), ("B", 40, 70),
)

# Coarse binning. NOTE: the metric is named "coarse_3bin" but only three of the
# five dimensions are three-way -- CT and Dimmer are BINARY. Bin *widths* are
# uniform (hue 120 degrees each; S/B even thirds; CT/Dimmer split at midpoint),
# so the misnomer is the naming, not the thresholds.
#
# What must be disclosed alongside any coarse number is the skew of the LABEL
# distribution, not the bin widths. On the 1000-row rule segment the reference
# bins are H {RED 64.3%, BLUE 25.3%, YELLOW 10.4%}, B {MID 42.9%, HIGH 40.9%,
# LOW 16.2%}. Consequently:
#     uniform-random all-5 match = (1/3)^3 * (1/2)^2 = 0.93%
#     best constant (majority-bin) prediction        = 3.10%
# Quote 3.10%, not 0.93%, as the trivial baseline for coarse.
COARSE_BINS_PER_DIMENSION = {"H": 3, "S": 3, "B": 3, "CT": 2, "Dimmer": 2}
COARSE_HUE_BINS = (("RED", "[300,360) u [0,60)"), ("YELLOW", "[60,180)"), ("BLUE", "[180,300)"))
COARSE_UNIFORM_RANDOM_ALL5 = 0.0093
COARSE_MAJORITY_CONSTANT_ALL5 = 0.0310
COARSE_CT_MIN = CT_MIN
COARSE_CT_MAX = CT_MAX
COARSE_DIMENSIONS = ("H", "S", "B", "CT", "Dimmer")


def normalize_hue(value: Any) -> int:
    """Circular normalization. Never clamps -- 370 -> 10, -10 -> 350."""
    return int(round(float(value))) % HUE_MODULUS


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))


def hue_distance(a: Any, b: Any) -> int:
    """Shortest circular distance in degrees, 0..180."""
    diff = abs(int(a) - int(b)) % HUE_MODULUS
    return min(diff, HUE_MODULUS - diff)
