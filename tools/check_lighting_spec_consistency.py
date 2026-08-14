"""Guard: assert the canonical lighting spec agrees with every code path that restates it.

backend/common/benchmark_eval.py is the scorer of record -- every published number
was produced by it -- so it is deliberately NOT refactored to import from
lighting_spec.py. Instead this check asserts the two agree, by exercising the
real scorer at each tolerance boundary. If someone edits one and not the other,
this fails.

Run: python tools/check_lighting_spec_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "backend" / "models"))

from common import benchmark_eval as be  # noqa: E402
from common import lighting_spec as spec  # noqa: E402
from compact_llm import inference as gen_inf  # noqa: E402
from compact_llm.training_data import normalize_output  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(label)


BASE = {"H": 100, "S": 50, "B": 50, "Dimmer": 50, "CT": 300}
ROW: dict = {"emotion": None, "base_color": "green"}   # colour set -> no emotion constraints


def perturb(dim: str, delta: int) -> dict:
    out = dict(BASE)
    out[dim] = BASE[dim] + delta
    return out


print("tolerance boundaries (scorer vs spec):")
for dim, tol in spec.STRICT_TOLERANCES.items():
    at = be.semantic_pass(BASE, perturb(dim, tol), ROW)
    beyond = be.semantic_pass(BASE, perturb(dim, tol + 1), ROW)
    check(f"{dim}: pass at +{tol}, fail at +{tol+1}", at and not beyond,
          f"at={at} beyond={beyond}")

print("\nCT range:")
check("spec CT floor == scorer COARSE_CT_MIN",
      spec.COARSE_CT_MIN == be.COARSE_CT_MIN, f"{spec.COARSE_CT_MIN} vs {be.COARSE_CT_MIN}")
check("spec CT ceiling == scorer COARSE_CT_MAX",
      spec.COARSE_CT_MAX == be.COARSE_CT_MAX, f"{spec.COARSE_CT_MAX} vs {be.COARSE_CT_MAX}")
check("generative runtime clamp CT floor == spec",
      gen_inf.FIELD_RANGES["CT"][0] == spec.CT_MIN,
      f"{gen_inf.FIELD_RANGES['CT'][0]} vs {spec.CT_MIN}")
check("label generation clamps CT to the spec floor",
      normalize_output({"h": 0, "s": 0, "b": 0, "dimmer": 0, "ct": 100})["CT"] == spec.CT_MIN)

print("\nhue is circular, never clamped:")
for raw, expected in ((370, 10), (-10, 350), (360, 0), (725, 5), (359, 359)):
    got = normalize_output({"h": raw, "s": 0, "b": 0, "dimmer": 0, "ct": 300})["H"]
    check(f"label normalize {raw} -> {expected}", got == expected, f"got {got}")
    check(f"spec normalize {raw} -> {expected}", spec.normalize_hue(raw) == expected)

print("\nhue distance wraps:")
for a, b, expected in ((350, 10, 20), (10, 350, 20), (0, 180, 180), (0, 181, 179)):
    check(f"hue_distance({a},{b}) == {expected}",
          be.hue_distance(a, b) == expected == spec.hue_distance(a, b),
          f"scorer={be.hue_distance(a,b)} spec={spec.hue_distance(a,b)}")

print("\ncoarse binning:")
counts = {"RED": 0, "YELLOW": 0, "BLUE": 0}
for h in range(360):
    counts[be.hue_coarse_bin(h)] += 1
check(f"hue bin widths uniform at 120 degrees {counts}",
      counts == {"RED": 120, "YELLOW": 120, "BLUE": 120})
observed = {d: len({be.coarse_bins({"H": h, "S": v, "B": v, "CT": c, "Dimmer": v})[d]
                    for h in range(0, 360, 7) for v in range(0, 101, 3) for c in range(153, 501, 11)})
            for d in ("H", "S", "B", "CT", "Dimmer")}
check(f"bins per dimension {observed} -- CT and Dimmer are binary despite the "
      f"'coarse_3bin' name", observed == spec.COARSE_BINS_PER_DIMENSION)

print("\nhue-anchor snap threshold:")
check("spec records the 18-degree snap used by resolve_case_output",
      spec.HUE_ANCHOR_SNAP_DEGREES == 18)
check("strict H tolerance (20) is looser than the anchor snap (18)",
      spec.STRICT_TOLERANCES["H"] > spec.HUE_ANCHOR_SNAP_DEGREES,
      "policy-conformant hues therefore always clear the H check")

print()
if failures:
    print(f"FAILED: {len(failures)} inconsistencies")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all lighting-spec consistency checks passed")
