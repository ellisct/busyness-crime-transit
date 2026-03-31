#!/usr/bin/env python3
"""
busyness_patterns.py

Visualize descriptive busyness dynamics at San Diego transit stops.

For each stop, the input panel contains three busyness metrics per
hour-of-day (HOD) and day-of-week (DOW):
  - sd_dev   : deviation from expected (baseline) busyness
  - d1_pos   : positive component of the first derivative of smoothed busyness
  - d1_neg   : negative component of the first derivative of smoothed busyness

This script:
  1. Pools all stops and plots smoothed, standardized busyness dynamics
     by hour of day and by day of week.
  2. Produces interactive Folium heatmaps of per-stop mean dynamics
     across San Diego.

Usage:
    python busyness_patterns.py --data path/to/panel_for_probcrime.csv.gz
    python busyness_patterns.py --data path/to/panel.csv.gz --out path/to/output/

Inputs:
    panel_for_probcrime.csv.gz
        Required columns: stop_id_core, hod, dow, stop_lat, stop_lon,
                          sd_dev, d1_pos, d1_neg

Outputs:
    figures/busyness_by_hour.png
    figures/busyness_by_day.png
    maps/map_deviation.html
    maps/map_change_magnitude.html
    tables/by_hour.csv
    tables/by_day.csv
    tables/stop_means.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

from utils import require_cols, coerce_numeric, smooth_centered, zscore, safe_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = ["stop_id_core", "hod", "dow", "stop_lat", "stop_lon",
                 "sd_dev", "d1_pos", "d1_neg"]

SD_CENTER = (32.7157, -117.1611)
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SMOOTH_WINDOW_HOUR = 3
SMOOTH_WINDOW_DOW = 3


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dynamics(
    x: np.ndarray,
    series: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    outpath: Path,
    xtick_labels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for y, lab in zip(series, labels):
        ax.plot(x, y, linewidth=2.5, label=lab)
    ax.axhline(0, linewidth=0.8, color="grey", alpha=0.6)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Relative intensity (z-score)", fontsize=12)
    if xtick_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
    ax.legend(
        title="Busyness metric",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
    )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def heatmap_html(
    stop_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    weight_col: str,
    title: str,
    outpath: Path,
) -> None:
    m = folium.Map(location=list(SD_CENTER), zoom_start=11, tiles="CartoDB Positron")
    pts = [
        [float(r[lat_col]), float(r[lon_col]), float(r[weight_col])]
        for _, r in stop_df.iterrows()
        if all(np.isfinite(r[c]) for c in [lat_col, lon_col, weight_col])
    ]
    HeatMap(pts, name=title, radius=14, blur=18, min_opacity=0.25).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(outpath))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: Path, out_dir: Path) -> None:
    # Load and validate
    df = pd.read_csv(data_path)
    require_cols(df, REQUIRED_COLS, str(data_path))
    coerce_numeric(df, ["hod", "dow", "stop_lat", "stop_lon", "sd_dev", "d1_pos", "d1_neg"])
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["hod"] = df["hod"].astype(int)
    df["dow"] = df["dow"].astype(int)

    # Normalize DOW to 0–6 if encoded as 1–7
    if df["dow"].min() >= 1 and df["dow"].max() <= 7:
        df["dow"] -= 1

    # Derived metrics
    df["abs_sd_dev"] = df["sd_dev"].abs()
    df["abs_d1"] = df["d1_pos"].fillna(0) + df["d1_neg"].fillna(0)
    df["d1"] = df["d1_pos"].fillna(0) - df["d1_neg"].fillna(0)

    # Aggregate by HOD and DOW
    def agg(group_col: str, window: int) -> pd.DataFrame:
        agged = (
            df.groupby(group_col, as_index=False)
            .agg(abs_sd_dev=("abs_sd_dev", "mean"),
                 d1=("d1", "mean"),
                 abs_d1=("abs_d1", "mean"))
            .sort_values(group_col)
        )
        for col in ["abs_sd_dev", "d1", "abs_d1"]:
            agged[col] = zscore(smooth_centered(agged[col], window))
        return agged

    by_hour = agg("hod", SMOOTH_WINDOW_HOUR)
    by_day = agg("dow", SMOOTH_WINDOW_DOW)

    # Save tables
    tbl = out_dir / "tables"
    by_hour.to_csv(tbl / "by_hour.csv", index=False)
    by_day.to_csv(tbl / "by_day.csv", index=False)

    # Plots
    metric_labels = [
        "Deviation from expected busyness",
        "Directional change in busyness",
        "Magnitude of busyness change",
    ]

    plot_dynamics(
        x=by_hour["hod"].to_numpy(),
        series=[by_hour[c].to_numpy() for c in ["abs_sd_dev", "d1", "abs_d1"]],
        labels=metric_labels,
        title="Busyness dynamics by hour of day (all transit stops)",
        xlabel="Hour of day",
        outpath=out_dir / "figures" / "busyness_by_hour.png",
        xtick_labels=[str(h) for h in range(24)],
    )

    plot_dynamics(
        x=by_day["dow"].to_numpy(),
        series=[by_day[c].to_numpy() for c in ["abs_sd_dev", "d1", "abs_d1"]],
        labels=metric_labels,
        title="Busyness dynamics by day of week (all transit stops)",
        xlabel="Day of week",
        outpath=out_dir / "figures" / "busyness_by_day.png",
        xtick_labels=DOW_LABELS,
    )

    # Stop-level means for maps
    stop_means = (
        df.groupby("stop_id_core", as_index=False)
        .agg(stop_lat=("stop_lat", "first"),
             stop_lon=("stop_lon", "first"),
             mean_abs_sd_dev=("abs_sd_dev", "mean"),
             mean_abs_d1=("abs_d1", "mean"))
    )
    stop_means.to_csv(tbl / "stop_means.csv", index=False)

    heatmap_html(stop_means, "stop_lat", "stop_lon", "mean_abs_sd_dev",
                 "Mean deviation from expected busyness",
                 out_dir / "maps" / "map_deviation.html")

    heatmap_html(stop_means, "stop_lat", "stop_lon", "mean_abs_d1",
                 "Mean magnitude of busyness change",
                 out_dir / "maps" / "map_change_magnitude.html")

    print("Done. Outputs:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize busyness dynamics at transit stops.")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to panel_for_probcrime.csv.gz")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: data directory / busyness_patterns_<timestamp>)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (args.data.parent / f"busyness_patterns_{timestamp}")
    for sub in ["figures", "maps", "tables"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    main(args.data, out_dir)
