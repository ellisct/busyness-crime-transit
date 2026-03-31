#!/usr/bin/env python3
"""
rq_models.py

Research-question-specific Negative Binomial models with presentation outputs.

Tests two research questions about pedestrian busyness and crime at transit stops:
  RQ1: Does busyness change (directional and magnitude) predict crime counts?
  RQ2: Does deviation from expected busyness predict crime counts?
  Combined: Robustness check including all predictors.

For each specification and outcome (total, property, violent), the script fits
Negative Binomial (primary) and Poisson (comparison) GLMs, then produces:
  - Coefficient tables with IRR and 95% CIs
  - Predicted-count plots with confidence ribbons
  - A presentation-style summary table (IRR and % change)

Crime type grouping:
  Violent — Simple Assault, Aggravated Assault, Robbery
  Property — Vandalism, Larceny, Motor Vehicle Theft, Burglary, etc.
  (Unmapped crime types are excluded by default.)

Usage:
    python rq_models.py --panel path/to/panel_for_probcrime.csv.gz \\
                        --crimes path/to/agg_stop_dow_hod_ctype.csv.gz
    python rq_models.py --panel ... --crimes ... --out path/to/output/

Inputs:
    panel_for_probcrime.csv.gz
        Required columns: stop_id_core, dow, hod,
                          busyness_norm, d1_pos, d1_neg, sd_dev

    agg_stop_dow_hod_ctype.csv.gz
        Required columns: stop_id_core, dow, hod, ctype, crime_count

Outputs:
    summaries/            — Poisson + NB summaries and dispersion diagnostics
    coefs_all.csv         — Tidy coefficient table across all specs and outcomes
    figures/              — Predicted-count plots with ribbons
    tables/               — Presentation-style IRR tables (CSV + PNG)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy

from utils import (
    require_cols, coerce_numeric, safe_name,
    fit_poisson, fit_negbin, overdispersion, coef_table,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_COL = "stop_id_core"
DOW_COL = "dow"
HOD_COL = "hod"
DV = "crime_count"

PANEL_COLS = [STOP_COL, DOW_COL, HOD_COL, "busyness_norm", "d1_pos", "d1_neg", "sd_dev"]
CRIME_COLS = [STOP_COL, DOW_COL, HOD_COL, "ctype", DV]

CRIME_GROUP_MAP = {
    "Simple Assault": "Violent",
    "Aggravated Assault": "Violent",
    "Robbery": "Violent",
    "Destruction/Damage/Vandalism of Property": "Property",
    "All Other Larceny": "Property",
    "Motor Vehicle Theft": "Property",
    "Theft from Motor Vehicle": "Property",
    "Burglary/Breaking & Entering **": "Property",
    "Burglary/Breaking & Entering": "Property",
    "Theft from Building": "Property",
}

TERM_LABELS = {
    "busyness_norm": "Avg busyness",
    "d1": "Busyness change (signed)",
    "abs_d1": "Busyness change (magnitude)",
    "sd_dev": "Deviation from expected",
}

SPEC_LABELS = {
    "RQ1": "RQ1: Busyness change → crime",
    "RQ2": "RQ2: Busyness deviation → crime",
    "Combined": "Combined (robustness check)",
}

Z95 = 1.959963984540054
N_GRID = 80


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_merge(panel_path: Path, crimes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    crimes = pd.read_csv(crimes_path)

    require_cols(panel, PANEL_COLS, panel_path.name)
    require_cols(crimes, CRIME_COLS, crimes_path.name)

    coerce_numeric(panel, [DOW_COL, HOD_COL, "busyness_norm", "d1_pos", "d1_neg", "sd_dev"])
    coerce_numeric(crimes, [DOW_COL, HOD_COL, DV])

    panel = panel.dropna(subset=PANEL_COLS).copy()
    panel[DOW_COL] = panel[DOW_COL].astype(int)
    panel[HOD_COL] = panel[HOD_COL].astype(int)

    # Build cleaner change terms
    panel["d1"] = panel["d1_pos"].fillna(0) - panel["d1_neg"].fillna(0)
    panel["abs_d1"] = panel["d1_pos"].fillna(0) + panel["d1_neg"].fillna(0)

    crimes = crimes.dropna(subset=CRIME_COLS).copy()
    crimes[DOW_COL] = crimes[DOW_COL].astype(int)
    crimes[HOD_COL] = crimes[HOD_COL].astype(int)
    crimes[DV] = crimes[DV].astype(int)

    # Apply crime grouping
    crimes["ctype"] = crimes["ctype"].astype(str).str.strip()
    crimes["crime_group"] = crimes["ctype"].map(CRIME_GROUP_MAP)
    crimes = crimes.dropna(subset=["crime_group"]).copy()

    return panel, crimes


def build_outcome_df(panel: pd.DataFrame, crimes: pd.DataFrame,
                     group: str | None) -> pd.DataFrame:
    """Merge panel with crime counts for a given group (or all groups if None)."""
    if group is None:
        counts = crimes.groupby([STOP_COL, DOW_COL, HOD_COL], as_index=False)[DV].sum()
    else:
        counts = (crimes[crimes["crime_group"] == group]
                  .groupby([STOP_COL, DOW_COL, HOD_COL], as_index=False)[DV].sum())

    df = panel.merge(counts, on=[STOP_COL, DOW_COL, HOD_COL], how="left")
    df[DV] = df[DV].fillna(0).astype(int)
    return df.dropna(subset=["busyness_norm", "d1", "abs_d1", "sd_dev"]).copy()


# ---------------------------------------------------------------------------
# Prediction curves
# ---------------------------------------------------------------------------

def prediction_curve(result, df: pd.DataFrame, x_var: str,
                     all_preds: list[str]) -> pd.DataFrame:
    """
    Compute predicted counts and 95% CI ribbon over the range of x_var,
    holding all other predictors at their median.
    """
    lo_p, hi_p = np.percentile(df[x_var].dropna(), [1, 99])
    grid = np.linspace(lo_p, hi_p, N_GRID)

    medians = {c: float(np.nanmedian(df[c])) for c in all_preds}
    pred_df = pd.DataFrame({c: [medians[c]] * N_GRID for c in all_preds})
    pred_df[x_var] = grid

    design_info = result.model.data.design_info
    X = np.asarray(patsy.build_design_matrices([design_info], pred_df,
                                                return_type="dataframe")[0], dtype=float)
    beta = np.asarray(result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)

    eta = X @ beta
    se_eta = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, cov, X), 0))
    inv = result.model.family.link.inverse

    return pd.DataFrame({
        "x": grid,
        "mu": np.asarray(inv(eta), dtype=float),
        "lo": np.asarray(inv(eta - Z95 * se_eta), dtype=float),
        "hi": np.asarray(inv(eta + Z95 * se_eta), dtype=float),
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ribbon(curves: dict[str, pd.DataFrame], x_var: str,
                spec_label: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for outcome, curve in curves.items():
        ax.plot(curve["x"], curve["mu"], label=outcome, linewidth=2)
        ax.fill_between(curve["x"], curve["lo"], curve["hi"], alpha=0.15)
    ax.set_xlabel(TERM_LABELS.get(x_var, x_var), fontsize=12)
    ax.set_ylabel("Predicted incidents (per stop × day × hour)", fontsize=12)
    ax.set_title(f"{SPEC_LABELS.get(spec_label, spec_label)}\n"
                 f"Predicted counts vs. {TERM_LABELS.get(x_var, x_var)}", fontsize=13, pad=10)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Presentation table
# ---------------------------------------------------------------------------

def presentation_table(nb_models: dict[str, dict[str, object]],
                       preds_by_spec: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """
    Build a wide IRR + % change table for each specification.
    Rows = predictors; columns = outcome × (IRR, %Δ).
    """
    tables = {}
    for spec, outcome_models in nb_models.items():
        rows = []
        for term in preds_by_spec[spec]:
            row = {"Predictor": TERM_LABELS.get(term, term)}
            for outcome, res in outcome_models.items():
                b = float(res.params.get(term, np.nan))
                irr = np.exp(b) if np.isfinite(b) else np.nan
                pct = (irr - 1) * 100 if np.isfinite(irr) else np.nan
                p = float(res.pvalues.get(term, np.nan))
                row[f"{outcome} IRR"] = f"{irr:.3f}" if np.isfinite(irr) else ""
                row[f"{outcome} %Δ"] = f"{pct:.1f}" if np.isfinite(pct) else ""
                row[f"{outcome} p"] = f"{p:.3f}" if np.isfinite(p) else ""
            rows.append(row)
        tables[spec] = pd.DataFrame(rows)
    return tables


def save_table_png(df: pd.DataFrame, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, max(2.5, len(df) * 0.6 + 1.2)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    tbl = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(panel_path: Path, crimes_path: Path, out_dir: Path) -> None:
    panel, crimes = load_and_merge(panel_path, crimes_path)

    outcomes = {
        "Total": build_outcome_df(panel, crimes, None),
        "Property": build_outcome_df(panel, crimes, "Property"),
        "Violent": build_outcome_df(panel, crimes, "Violent"),
    }

    specs = {
        "RQ1": ["busyness_norm", "d1", "abs_d1"],
        "RQ2": ["busyness_norm", "sd_dev"],
        "Combined": ["busyness_norm", "d1", "abs_d1", "sd_dev"],
    }

    sum_dir = out_dir / "summaries"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    for d in [sum_dir, fig_dir, tab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_coefs = []
    nb_fitted: dict[str, dict[str, object]] = {s: {} for s in specs}

    for spec, preds in specs.items():
        formula = f"{DV} ~ {' + '.join(preds)}"

        for outcome, df in outcomes.items():
            pois, pois_l = fit_poisson(formula, df)
            nb, nb_l = fit_negbin(formula, df)

            fname = f"{safe_name(spec)}__{safe_name(outcome)}"
            (sum_dir / f"{fname}_summary.txt").write_text(
                f"Spec: {SPEC_LABELS[spec]}\nOutcome: {outcome}\nFormula: {formula}\n\n"
                f"--- {pois_l} ---\n{pois.summary().as_text()}\n\n"
                f"--- {nb_l} ---\n{nb.summary().as_text()}\n"
            )

            disp = overdispersion(pois)
            (sum_dir / f"{fname}_dispersion.txt").write_text(
                f"Spec: {spec} | Outcome: {outcome}\n"
                f"Pearson chi2 / df_resid = {disp['pearson_over_df']:.3f}\n"
            )

            for res, label in [(pois, pois_l), (nb, nb_l)]:
                all_coefs.append(coef_table(res, label, outcome=outcome, spec=spec))

            nb_fitted[spec][outcome] = nb

        # Prediction plots (grouped by predictor, overlaying outcomes)
        for x_var in preds:
            curves = {
                outcome: prediction_curve(nb_fitted[spec][outcome], outcomes[outcome],
                                          x_var, preds)
                for outcome in outcomes
            }
            plot_ribbon(curves, x_var,
                        spec_label=spec,
                        outpath=fig_dir / safe_name(spec) / f"{safe_name(x_var)}.png")

    # Save coefficient tables
    pd.concat(all_coefs, ignore_index=True).to_csv(out_dir / "coefs_all.csv", index=False)

    # Presentation tables
    pres = presentation_table(nb_fitted, specs)
    for spec, tdf in pres.items():
        tdf.to_csv(tab_dir / f"{safe_name(spec)}_table.csv", index=False)
        save_table_png(tdf,
                       title=f"{SPEC_LABELS[spec]} — IRR and % change",
                       outpath=tab_dir / f"{safe_name(spec)}_table.png")

    print("Done. Outputs:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ-specific crime count models with presentation outputs."
    )
    parser.add_argument("--panel", required=True, type=Path,
                        help="Path to panel_for_probcrime.csv.gz")
    parser.add_argument("--crimes", required=True, type=Path,
                        help="Path to agg_stop_dow_hod_ctype.csv.gz")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: panel directory / rq_models_<timestamp>)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (args.panel.parent / f"rq_models_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    main(args.panel, args.crimes, out_dir)
