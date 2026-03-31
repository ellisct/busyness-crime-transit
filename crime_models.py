#!/usr/bin/env python3
"""
crime_models.py

Poisson and Negative Binomial count models of crime at transit stops.

Models crime counts at the stop × day-of-week × hour-of-day level as a
function of three busyness predictors:
  - busyness_norm : normalized average busyness level
  - d1_pos / d1_neg : positive and negative components of busyness change
  - sd_dev        : deviation from expected (baseline) busyness

Both an all-crime model and per-crime-type models are fit. Negative
Binomial is the primary specification; Poisson with robust (HC0) standard
errors is included for comparison and overdispersion diagnostics.

Usage:
    python crime_models.py --panel path/to/panel_for_probcrime.csv.gz \\
                           --crimes path/to/agg_stop_dow_hod_ctype.csv.gz
    python crime_models.py --panel ... --crimes ... --out path/to/output/ \\
                           --min-events 200

Inputs:
    panel_for_probcrime.csv.gz
        Required columns: stop_id_core, dow, hod,
                          busyness_norm, d1_pos, d1_neg, sd_dev

    agg_stop_dow_hod_ctype.csv.gz
        Required columns: stop_id_core, dow, hod, ctype, crime_count

Outputs:
    allcrime_summary.txt          — Poisson and NB model summaries
    allcrime_coefs.csv            — Tidy coefficient table (all-crime)
    allcrime_dispersion.txt       — Overdispersion diagnostic
    ctypes_coefs.csv              — Tidy coefficient table (all crime types)
    summaries/<ctype>_summary.txt — Per-crime-type model summaries
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
PREDICTORS = ["busyness_norm", "d1_pos", "d1_neg", "sd_dev"]
FORMULA = f"{DV} ~ {' + '.join(PREDICTORS)}"

PANEL_COLS = [STOP_COL, DOW_COL, HOD_COL] + PREDICTORS
CRIME_COLS = [STOP_COL, DOW_COL, HOD_COL, "ctype", DV]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_cols(df, PANEL_COLS, path.name)
    coerce_numeric(df, [DOW_COL, HOD_COL] + PREDICTORS)
    df = df.dropna(subset=PANEL_COLS).copy()
    df[DOW_COL] = df[DOW_COL].astype(int)
    df[HOD_COL] = df[HOD_COL].astype(int)
    return df


def load_crimes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_cols(df, CRIME_COLS, path.name)
    coerce_numeric(df, [DOW_COL, HOD_COL, DV])
    df = df.dropna(subset=CRIME_COLS).copy()
    df[DOW_COL] = df[DOW_COL].astype(int)
    df[HOD_COL] = df[HOD_COL].astype(int)
    df[DV] = df[DV].astype(int)
    return df


def build_model_df(panel: pd.DataFrame, crimes: pd.DataFrame,
                   ctype: str | None = None) -> pd.DataFrame:
    """
    Merge panel busyness predictors with crime counts.
    If ctype is None, sums across all crime types.
    Zero-fills stops with no recorded crimes.
    """
    if ctype is None:
        counts = (crimes.groupby([STOP_COL, DOW_COL, HOD_COL], as_index=False)[DV].sum())
    else:
        counts = (crimes[crimes["ctype"] == ctype]
                  .groupby([STOP_COL, DOW_COL, HOD_COL], as_index=False)[DV].sum())

    merged = panel.merge(counts, on=[STOP_COL, DOW_COL, HOD_COL], how="left")
    merged[DV] = merged[DV].fillna(0).astype(int)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(panel_path: Path, crimes_path: Path, out_dir: Path,
         min_events: int) -> None:

    panel = load_panel(panel_path)
    crimes = load_crimes(crimes_path)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)

    all_coefs = []

    # --- All-crime model ---
    df_all = build_model_df(panel, crimes)

    pois, pois_label = fit_poisson(FORMULA, df_all)
    nb, nb_label = fit_negbin(FORMULA, df_all)

    (out_dir / "allcrime_summary.txt").write_text(
        f"Formula: {FORMULA}\n\n"
        f"--- {pois_label} ---\n{pois.summary().as_text()}\n\n"
        f"--- {nb_label} ---\n{nb.summary().as_text()}\n"
    )

    disp = overdispersion(pois)
    (out_dir / "allcrime_dispersion.txt").write_text(
        "Overdispersion diagnostic (Poisson)\n"
        f"Pearson chi2 / df_resid = {disp['pearson_over_df']:.3f}\n"
        "Values >> 1 indicate overdispersion; prefer Negative Binomial.\n"
    )

    for res, label in [(pois, pois_label), (nb, nb_label)]:
        all_coefs.append(coef_table(res, label, outcome="all_crime"))

    # --- Per-crime-type models ---
    totals = (crimes.groupby("ctype", as_index=False)[DV].sum()
              .sort_values(DV, ascending=False))
    eligible = totals[totals[DV] >= min_events]["ctype"].tolist()

    for ctype in eligible:
        df_c = build_model_df(panel, crimes, ctype=ctype)

        pois_c, pois_cl = fit_poisson(FORMULA, df_c)
        nb_c, nb_cl = fit_negbin(FORMULA, df_c)

        fname = safe_name(str(ctype))
        (out_dir / "summaries" / f"{fname}_summary.txt").write_text(
            f"Crime type: {ctype}\n"
            f"Total events: {int(totals.loc[totals['ctype'] == ctype, DV].values[0])}\n"
            f"Formula: {FORMULA}\n\n"
            f"--- {pois_cl} ---\n{pois_c.summary().as_text()}\n\n"
            f"--- {nb_cl} ---\n{nb_c.summary().as_text()}\n"
        )

        for res, label in [(pois_c, pois_cl), (nb_c, nb_cl)]:
            all_coefs.append(coef_table(res, label, outcome=str(ctype)))

    # Save all coefficients
    pd.concat(all_coefs, ignore_index=True).to_csv(out_dir / "ctypes_coefs.csv", index=False)

    # All-crime coefs separately for convenience
    (pd.concat([t for t in all_coefs if t["outcome"].eq("all_crime").all()], ignore_index=True)
     .to_csv(out_dir / "allcrime_coefs.csv", index=False))

    print("Done. Outputs:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Poisson and Negative Binomial models of crime at transit stops."
    )
    parser.add_argument("--panel", required=True, type=Path,
                        help="Path to panel_for_probcrime.csv.gz")
    parser.add_argument("--crimes", required=True, type=Path,
                        help="Path to agg_stop_dow_hod_ctype.csv.gz")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: panel directory / crime_models_<timestamp>)")
    parser.add_argument("--min-events", type=int, default=200,
                        help="Minimum total events for a crime type to be modeled (default: 200)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (args.panel.parent / f"crime_models_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    main(args.panel, args.crimes, out_dir, args.min_events)
