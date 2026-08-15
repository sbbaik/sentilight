"""Generate F2-F5 from the artifacts of record.

F1 (system architecture) is a schematic and is not produced here.

Every value is read from a run summary or an adjudication JSON; nothing is
hard-coded. Three correctness constraints are enforced in code rather than left
to the person drawing the figure:

  F2  The tuple head has no data-fraction sweep (f100 only), so it is drawn as a
      horizontal band, never as a line across fractions. No 60M-vs-23M scale
      annotation is drawn under the head: its direction flips across seeds.
  F3  C2b strict was NOT adopted (seed-sign inconsistency), so it is hatched and
      labelled rather than shown as a plain bar beside adopted effects.
  F4  Isolated single-model latency (batch 1, CUDA-synchronized) and
      service-tier latency (5-model concurrent /predict_all) are DIFFERENT
      measurements and are never placed on one axis. Gemini latency is excluded
      entirely (997/1000 quota failures in the run it came from).

Usage: python tools/make_figures.py [--outdir results/figures] [--format pdf]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
SCALES = ["3M", "8M", "23M", "60M"]
FRACTIONS = [25, 50, 100]
SCALE_COLOR = {"3M": "#c44e52", "8M": "#dd8452", "23M": "#4c72b0", "60M": "#55a868"}
HEAD_COLOR = "#8172b3"

FAMILY_PATHS = {
    "A1_gen23": "backend/reports/scale_grid_clean/run_summaries/compactlm_23M_f100_s{seed}.json",
    "gen60": "backend/reports/scale_grid_clean/run_summaries/compactlm_60M_f100_s{seed}.json",
    "A2_policy23": "backend/reports/a2_policy_only_clean/run_summaries/a2_policy_lm_23M_s{seed}.json",
    "B2_head23": "backend/reports/tuple_head_clean/run_summaries/tuple_head_23M_f100_s{seed}.json",
    "B2_head60": "backend/reports/tuple_head_60M_clean/run_summaries/tuple_head_60M_f100_s{seed}.json",
}
FAMILY_LABEL = {
    # Descriptive names are primary; the design codes used by the preregistration,
    # VERDICTS.md and the per_row directories are kept in parentheses so a figure
    # can still be matched to the released artifacts.
    "A1_gen23": "Gen-Full 23M (A1)", "gen60": "Gen-Full 60M",
    "A2_policy23": "Gen-Policy 23M (A2)", "B2_head23": "Head-Policy 23M (B2)",
    "B2_head60": "Head-Policy 60M",
}


def caption(fig, text: str, y: float = -0.03, width: int = 110) -> None:
    """Footnote wrapped to a fixed width so bbox_inches='tight' cannot stretch the figure."""
    import textwrap
    fig.text(0.5, y, "\n".join(textwrap.wrap(text, width)),
             ha="center", fontsize=7.5, style="italic")


def metrics(path: Path) -> tuple[float, float]:
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return m["strict_semantic_pass_rate"] * 100, m["coarse_3bin_pass_rate"] * 100


def family_seed_values() -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for fam, pat in FAMILY_PATHS.items():
        s, c = [], []
        for seed in SEEDS:
            a, b = metrics(REPO / pat.format(seed=seed))
            s.append(a)
            c.append(b)
        out[fam] = {"strict": s, "coarse": c}
    return out


def grid_aggregates() -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    cells = collections.defaultdict(lambda: {"strict": [], "coarse": []})
    for f in sorted(glob.glob(str(REPO / "backend/reports/scale_grid_clean/run_summaries/compactlm_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        mid = r["model_id"]
        key = (mid.split("_")[1], int(mid.split("_f")[1].split("_")[0]))
        cells[key]["strict"].append(r["metrics"]["strict_semantic_pass_rate"] * 100)
        cells[key]["coarse"].append(r["metrics"]["coarse_3bin_pass_rate"] * 100)
    return {k: {m: (st.fmean(v[m]), st.stdev(v[m])) for m in ("strict", "coarse")}
            for k, v in cells.items()}


# ----------------------------------------------------------------- F2
def figure_f2(outdir: Path, fmt: str) -> None:
    agg = grid_aggregates()
    fam = family_seed_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, metric in zip(axes, ("strict", "coarse")):
        for scale in SCALES:
            means = [agg[(scale, f)][metric][0] for f in FRACTIONS]
            sds = [agg[(scale, f)][metric][1] for f in FRACTIONS]
            ax.errorbar(FRACTIONS, means, yerr=sds, marker="o", capsize=3, lw=1.8,
                        color=SCALE_COLOR[scale], label=f"Gen-Full {scale}")
        # tuple-head band: f100 only -> horizontal span, NOT a line across fractions
        for key, ls in (("B2_head23", "-"), ("B2_head60", "--")):
            vals = fam[key][metric]
            m, sd = st.fmean(vals), st.stdev(vals)
            ax.axhspan(m - sd, m + sd, color=HEAD_COLOR, alpha=0.16, zorder=0)
            ax.axhline(m, color=HEAD_COLOR, ls=ls, lw=2.0, zorder=1,
                       label=f"{FAMILY_LABEL[key]} (f100 only)")
        ax.set_xticks(FRACTIONS)
        ax.set_xlabel("SFT data fraction (%)")
        ax.set_ylabel(f"{metric} pass rate (%)")
        ax.set_title(f"{metric}", fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
    axes[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   framealpha=0.95, borderaxespad=0)
    fig.suptitle("F2  Clean scale grid (4 scales x 3 fractions x 3 seeds) with the Head-Policy band",
                 fontsize=11.5)
    caption(fig, 
             "Head-Policy bands are f100 only and have no fraction sweep. No 60M-vs-23M scale "
             "effect is annotated under the head: its direction flips across seeds.",)
    fig.tight_layout()
    fig.savefig(outdir / f"F2_scale_grid.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- F3
def figure_f3(outdir: Path, fmt: str) -> None:
    """2x2 decomposition with 95% CI from the three seed-level paired deltas."""
    fam = family_seed_values()
    contrasts = [
        ("C2a\ndata alignment\nGen-Policy \u2212 Gen-Full", "A1_gen23", "A2_policy23"),
        ("C2b\noutput structure\nHead-Policy \u2212 Gen-Policy", "A2_policy23", "B2_head23"),
        ("C2c\ncomposite\nHead-Policy \u2212 Gen-Full", "A1_gen23", "B2_head23"),
        ("C1\nreformulation vs scale\nHead-Policy \u2212 Gen-Full 60M", "gen60", "B2_head23"),
    ]
    # adoption verdicts from the row-level test output
    w1 = json.loads((REPO / "results/row_level_results.json").read_text(encoding="utf-8"))
    adopted = {(k.split("_")[0], k.split("_")[1]): v["adopted"]
               for k, v in w1["clean"]["primary"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, metric in zip(axes, ("strict", "coarse")):
        labels, means, errs, hatches, colors = [], [], [], [], []
        for label, base, treat in contrasts:
            d = [t - b for b, t in zip(fam[base][metric], fam[treat][metric])]
            m, sd = st.fmean(d), st.stdev(d)
            ci = 4.303 * sd / math.sqrt(len(d))   # t(0.975, df=2)
            key = label.split("\n")[0]
            ok = adopted.get((key, metric), True)
            labels.append(label)
            means.append(m)
            errs.append(ci)
            hatches.append("" if ok else "///")
            colors.append("#4c72b0" if ok else "#b0b0b0")
        bars = ax.bar(range(len(labels)), means, yerr=errs, capsize=4,
                      color=colors, edgecolor="black", lw=0.8)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        for i, (m, h) in enumerate(zip(means, hatches)):
            ax.text(i, m + (0.6 if m >= 0 else -1.6), f"{m:+.2f}", ha="center", fontsize=8.5)
            if h:
                lo = min(0.0, m - errs[i])
                ax.text(i, lo - abs(lo) * 0.10 - 1.4, "not adopted", ha="center", va="top",
                        fontsize=8, color="#333333", fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.0, linespacing=1.35)
        ax.set_ylabel("delta (pp)")
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.margins(y=0.18)
    axes[1].legend(handles=[
        Patch(facecolor="#4c72b0", edgecolor="black", label="adopted (Holm, signs consistent)"),
        Patch(facecolor="#b0b0b0", edgecolor="black", hatch="///", label="not adopted"),
    ], fontsize=7.5, loc="upper left")
    fig.suptitle("F3  2x2 decomposition, clean backbone (error bars: 95% CI over 3 seeds)",
                 fontsize=11.5)
    caption(fig, y=-0.13,
             text="Error bars are SEED-level (n=3, t(0.975,df=2)=4.303) and are wide by construction; "
             "adoption is the ROW-level verdict (exact McNemar -> Stouffer -> Holm, m=8), which "
             "is far better powered. A wide CI beside an adopted bar is the power finding, not a "
             "contradiction. C2b strict clears Holm at p=2.09e-3 but is rejected on seed-sign "
             "inconsistency.")
    fig.tight_layout()
    fig.savefig(outdir / f"F3_decomposition.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- F4
def figure_f4(outdir: Path, fmt: str) -> None:
    """Two panels: the two latency tiers are never placed on one axis."""
    fam = family_seed_values()
    (REPO / "results").mkdir(parents=True, exist_ok=True)
    lat_path = REPO / "results/latency_results.json"
    if not lat_path.exists():
        print("  figure_f4 -> skipped: results/latency_results.json not found. "
              "Run tools/run_latency_measurement.py first (needs a GPU and the checkpoints, "
              "which are distributed separately).", flush=True)
        return False
    lat = json.loads(lat_path.read_text(encoding="utf-8"))["families"]
    iso = {
        "A1_gen23": lat["A1_generative_23M"]["mean_ms"],
        "gen60": lat["generative_60M"]["mean_ms"],
        "B2_head23": lat["B2_tuple_head_23M"]["mean_ms"],
        "B2_head60": lat["B2_tuple_head_60M"]["mean_ms"],
    }
    seg = json.loads((REPO / "backend/reports/five_model_benchmark_v3_by_segment.json").read_text(encoding="utf-8"))
    service = {m["model_id"]: (m["latency"]["mean_ms"], m["strict_rate"] * 100, m["coarse_rate"] * 100)
               for m in seg["segments"]["rule_based_1000"]["models"]
               if m["model_id"] != "gemini_api"}          # Gemini latency excluded outright

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    for key, ms in iso.items():
        c = st.fmean(fam[key]["coarse"])
        marker = "s" if "head" in key else "o"
        ax.scatter(ms, c, s=90, marker=marker, zorder=3,
                   color=HEAD_COLOR if "head" in key else "#4c72b0", edgecolor="black", lw=0.7)
        ax.annotate(FAMILY_LABEL[key], (ms, c), textcoords="offset points",
                    xytext=(7, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("isolated latency (ms, log) — batch 1, CUDA-synchronized")
    ax.set_ylabel("coarse pass rate (%)")
    ax.set_title("(a) isolated single-model latency", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.margins(x=0.28, y=0.14)

    ax = axes[1]
    for mid, (ms, s, c) in service.items():
        ax.scatter(ms, c, s=90, zorder=3, color="#dd8452", edgecolor="black", lw=0.7)
        ax.annotate(mid, (ms, c), textcoords="offset points", xytext=(7, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("service latency (ms, log) — 5-model concurrent /predict_all")
    ax.set_ylabel("coarse pass rate (%)")
    ax.set_title("(b) deployed service tier", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.margins(x=0.30, y=0.14)

    fig.suptitle("F4  Latency-compliance, by measurement tier", fontsize=11.5)
    caption(fig, 
             "The two panels are DIFFERENT measurements and must not be merged onto one axis. "
             "Gemini latency is excluded (997/1000 quota failures in its source run). The tuple "
             "head is absent from panel (b): it was never in the service benchmark.")
    fig.tight_layout()
    fig.savefig(outdir / f"F4_latency_pareto.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- F5
def figure_f5(outdir: Path, fmt: str) -> None:
    fam = family_seed_values()
    order = ["A1_gen23", "gen60", "A2_policy23", "B2_head60", "B2_head23"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric in zip(axes, ("strict", "coarse")):
        for i, key in enumerate(order):
            vals = fam[key][metric]
            m, sd = st.fmean(vals), st.stdev(vals)
            ax.errorbar([i], [m], yerr=[sd], fmt="none", ecolor="#555555", capsize=6, lw=1.6, zorder=2)
            for j, v in enumerate(vals):
                ax.scatter([i], [v], s=42, zorder=3, alpha=0.85,
                           color=HEAD_COLOR if "head" in key else "#4c72b0",
                           marker=["o", "^", "s"][j], edgecolor="black", lw=0.5)
            ax.text(i, m + sd + 1.2, f"SD {sd:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([FAMILY_LABEL[k] for k in order], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"{metric} pass rate (%)")
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.margins(y=0.20)
    fig.suptitle("F5  Seed-to-seed stability (3 seeds; markers = seeds 42/43/44)", fontsize=11.5)
    caption(fig, 
             "Descriptive only — no variance-reduction claim is made. Note the spread "
             "collapses for the 23M head but NOT for the 60M head, whose SD exceeds "
             "Gen-Full 23M's on both metrics; any stability statement must name the 23M head.",)
    fig.tight_layout()
    fig.savefig(outdir / f"F5_seed_stability.{fmt}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=REPO / "results/figures")
    ap.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fn in (figure_f2, figure_f3, figure_f4, figure_f5):
        if fn(args.outdir, args.format) is not False:
            print(f"  {fn.__name__} -> ok", flush=True)
    print(f"figures written to {args.outdir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
