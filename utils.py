#!/usr/bin/env python3
"""
utils.py
Shared utilities for busyness-crime analysis scripts.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def safe_name(s: str, maxlen: int = 80) -> str:
    """Sanitize a string for use as a filename."""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def require_cols(df: pd.DataFrame, cols: list[str], source: str) -> None:
    """Raise KeyError if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """Coerce columns to numeric in-place, setting non-parseable values to NaN."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# ---------------------------------------------------------------------------
# Smoothing and scaling
# ---------------------------------------------------------------------------

def smooth_centered(s: pd.Series, window: int) -> pd.Series:
    """Apply a centered rolling mean."""
    w = max(int(window), 1)
    if w <= 1:
        return s
    return s.rolling(window=w, center=True, min_periods=max(1, w // 2)).mean()


def zscore(s: pd.Series) -> pd.Series:
    """Standardize a series to zero mean and unit variance."""
    mu, sd = float(s.mean()), float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_poisson(formula: str, df: pd.DataFrame) -> tuple:
    """
    Fit a Poisson GLM with HC0 robust standard errors.
    Falls back to non-robust if the installed statsmodels version does not
    support cov_type in GLM.
    Returns (result, label).
    """
    mod = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    try:
        return mod.fit(cov_type="HC0"), "Poisson (HC0 robust SE)"
    except TypeError:
        return mod.fit(), "Poisson (non-robust SE)"


def fit_negbin(formula: str, df: pd.DataFrame) -> tuple:
    """
    Fit a Negative Binomial GLM.
    Falls back to Poisson if NB fails (e.g. convergence issues).
    Returns (result, label).
    """
    try:
        mod = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
        return mod.fit(maxiter=250), "Negative Binomial"
    except Exception as exc:
        res = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
        return res, f"Poisson fallback (NB failed: {exc!r})"


def overdispersion(poisson_result) -> dict:
    """
    Return Pearson chi-square overdispersion diagnostic for a Poisson model.
    Values substantially above 1 suggest overdispersion; prefer NB in that case.
    """
    chi2 = float(getattr(poisson_result, "pearson_chi2", np.nan))
    df_r = float(getattr(poisson_result, "df_resid", np.nan))
    ratio = chi2 / df_r if np.isfinite(chi2) and df_r > 0 else np.nan
    return {"pearson_chi2": chi2, "df_resid": df_r, "pearson_over_df": ratio}


def coef_table(model, model_label: str, outcome: str = "", spec: str = "") -> pd.DataFrame:
    """
    Extract a tidy coefficient table from a fitted GLM.
    Includes log-IRR, SE, p-value, IRR, and 95% CI on the IRR scale.
    """
    Z = 1.959963984540054
    b = model.params.astype(float)
    se = model.bse.astype(float)
    p = model.pvalues.astype(float)
    lo = np.exp(b.values - Z * se.values)
    hi = np.exp(b.values + Z * se.values)

    return pd.DataFrame({
        "spec": spec,
        "outcome": outcome,
        "model": model_label,
        "term": b.index.astype(str),
        "coef": b.values,
        "se": se.values,
        "p": p.values,
        "IRR": np.exp(b.values),
        "IRR_CI95_low": lo,
        "IRR_CI95_high": hi,
    })
