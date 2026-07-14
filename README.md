# Pedestrian Busyness, Volatility, and Crime Near Bus Stops

Replication code for the project titiled *Unpredictability and Risk: Pedestrian Busyness, Volatility, and Crime Near Bus Stops* (presented at the Western Society of Criminology Annual Conference, 2025).

This repository contains Python scripts for modeling the relationship between pedestrian busyness dynamics and crime counts at San Diego Metropolitan Transit System (MTS) bus stops, using a stop × day-of-week × hour-of-day panel dataset.

---

## Project Overview

Conventional crime research treats busyness as a static count of people present. This project takes a different approach: it measures how busyness changes (its volatility, directional shifts, and deviation from expected patterns) and tests whether those dynamics predict crime, above and beyond average activity levels.

**Research questions:**
- RQ1: Does busyness change (directional and magnitude) predict crime counts at transit stops?
- RQ2: Does deviation from expected busyness predict crime counts?

**Crime outcomes:** Total incidents, property crime, violent crime

**Methods:** Negative Binomial and Poisson GLMs; spatiotemporal visualization; interactive heatmaps

---

## Repository Structure

```
├── utils.py               # Shared utilities (data loading, model fitting, helpers)
├── busyness_patterns.py   # Descriptive busyness dynamics — plots and heatmaps
├── crime_models.py        # All-crime and per-crime-type count models
├── rq_models.py           # RQ-specific models with presentation outputs
└── README.md
```

---

## Data

The scripts expect two input files produced by a separate data-build pipeline (not included here, as the underlying data is proprietary):

| File | Description |
|------|-------------|
| `panel_for_probcrime.csv.gz` | Stop × DOW × HOD panel with busyness predictors |
| `agg_stop_dow_hod_ctype.csv.gz` | Aggregated crime counts by stop × DOW × HOD × crime type |

### Required columns

**`panel_for_probcrime.csv.gz`**

| Column | Description |
|--------|-------------|
| `stop_id_core` | Transit stop identifier |
| `hod` | Hour of day (0–23) |
| `dow` | Day of week (0 = Monday, 6 = Sunday; or 1–7, auto-normalized) |
| `stop_lat` / `stop_lon` | Stop coordinates |
| `busyness_norm` | Normalized average busyness level |
| `d1_pos` | Positive component of busyness first derivative |
| `d1_neg` | Negative component of busyness first derivative |
| `sd_dev` | Deviation from expected (baseline) busyness |

**`agg_stop_dow_hod_ctype.csv.gz`**

| Column | Description |
|--------|-------------|
| `stop_id_core` | Transit stop identifier |
| `hod` | Hour of day |
| `dow` | Day of week |
| `ctype` | Crime type label |
| `crime_count` | Count of incidents |

---

## Installation

```bash
pip install numpy pandas matplotlib folium statsmodels patsy
```

Python 3.9+ recommended.

---

## Usage

### 1. Descriptive busyness patterns

Produces smoothed time-series plots and interactive heatmaps of busyness dynamics across all stops.

```bash
python busyness_patterns.py --data path/to/panel_for_probcrime.csv.gz
```

**Outputs** (in `busyness_patterns_<timestamp>/`):
- `figures/busyness_by_hour.png` — Busyness dynamics by hour of day
- `figures/busyness_by_day.png` — Busyness dynamics by day of week
- `maps/map_deviation.html` — Interactive heatmap of mean deviation from expected busyness
- `maps/map_change_magnitude.html` — Interactive heatmap of mean busyness change magnitude
- `tables/` — Underlying aggregated data

---

### 2. All-crime and per-type count models

Fits Poisson and Negative Binomial models for total crime and each crime type meeting a minimum event threshold.

```bash
python crime_models.py \
  --panel path/to/panel_for_probcrime.csv.gz \
  --crimes path/to/agg_stop_dow_hod_ctype.csv.gz \
  --min-events 200
```

**Outputs** (in `crime_models_<timestamp>/`):
- `allcrime_summary.txt` — Poisson and NB model summaries
- `allcrime_coefs.csv` — Coefficient table with IRR and 95% CIs
- `allcrime_dispersion.txt` — Overdispersion diagnostic
- `ctypes_coefs.csv` — Coefficient table across all eligible crime types
- `summaries/<ctype>_summary.txt` — Per-type model summaries

---

### 3. RQ-specific models with presentation outputs

Fits three model specifications (RQ1: busyness change; RQ2: busyness deviation; Combined robustness) across total, property, and violent crime outcomes. Produces coefficient tables, prediction plots, and summary tables formatted for presentations.

```bash
python rq_models.py \
  --panel path/to/panel_for_probcrime.csv.gz \
  --crimes path/to/agg_stop_dow_hod_ctype.csv.gz
```

**Outputs** (in `rq_models_<timestamp>/`):
- `coefs_all.csv` — Tidy coefficient table across all specs and outcomes
- `summaries/` — Model summaries and overdispersion diagnostics
- `figures/<spec>/<predictor>.png` — Predicted-count plots with 95% ribbons
- `tables/<spec>_table.csv` — IRR and % change table
- `tables/<spec>_table.png` — Presentation-ready version of above

---

## Model Notes

- **Negative Binomial** is the primary specification. Crime count data are typically overdispersed; the Pearson chi-square / df diagnostic in each output folder quantifies this.
- **Poisson with HC0 robust standard errors** is included for comparison.
- All models are fit at the stop × DOW × HOD level with no fixed effects. Stops with zero recorded crimes are retained (zero-filled) to avoid selection bias.
- Crime types are grouped into **Property** and **Violent** categories (see `CRIME_GROUP_MAP` in `rq_models.py`); unmapped types are excluded.

---

## Citation

Ellis, C. (2025). Unpredictability and risk: Pedestrian busyness, volatility, and crime near bus stops. Paper presented at the *Western Society of Criminology Annual Conference*. Panel Chair: Temporal, Spatial, and Other Data Considerations for Crime Analysis.

---

## License

MIT License. See `LICENSE` for details.
