#!/usr/bin/env python3
"""
Supplementary material: practical data-I/O implementation of Algorithm 3
for the paper

    Bivariate Survival Analysis for Truncated and Censored Astronomical Data:
    Applications to Galaxy Luminosity Functions

File name
---------
bivariate_survival_data_io.py

Purpose
-------
This script applies Algorithm 3 to a user-supplied bivariate astronomical
catalog.  It is the practical data-analysis counterpart of
``bivariate_survival_reference.py``.

The estimator itself is the same combined van der Laan--Dabrowska estimator
described as Algorithm 3 in the paper.  This data-I/O version removes
mock-data generation and built-in validation routines, while retaining the
same estimator, numerical outputs, and diagnostic figures.

Input columns
-------------
A CSV input file must contain

    x1_obs, x2_obs, y1, y2, delta1, delta2, region

where

* x1_obs, x2_obs are observed logarithmic luminosities or censoring limits;
* y1, y2 are logarithmic detection limits;
* delta1, delta2 are detection indicators, with 1 = detected and 0 = nondetected;
* region is one of A, B, C, or D.

Algorithm 3 uses rows in Regions A, B, and C.  Rows labeled D are allowed but
are removed before fitting, since Region D objects are unobserved by definition
and enter only through inverse observation-probability weighting.

Optional columns
----------------

    x1_true, x2_true

If these columns are present, additional comparison outputs against the
empirical truth are written.  These columns are intended only for simulations
or validation examples, not for real survey catalogs.

Outputs
-------
The output directory contains CSV files for the fitted CDF grid, PDF grid,
Algorithm 3 weights, weighted jump table, marginal Kaplan--Meier tables, and
summary diagnostics.  If plotting is enabled, PNG diagnostic figures are also
written under ``figures/``.

Examples
--------
Run Algorithm 3 on a user CSV:

    python bivariate_survival_data_io.py --input my_catalog.csv --outdir outputs

Run the example catalog distributed with the supplementary material:

    python bivariate_survival_data_io.py \
        --input example/example_input.csv \
        --outdir output_example

Run without figures:

    python bivariate_survival_data_io.py \
        --input example/example_input.csv \
        --outdir output_example \
        --no-plots

Set the evaluation grid explicitly:

    python bivariate_survival_data_io.py \
        --input my_catalog.csv \
        --outdir outputs \
        --x1-min 9.0 --x1-max 12.0 \
        --x2-min 8.8 --x2-max 12.2 \
        --grid 50

Jupyter notebook use
--------------------
If this file is imported or pasted into a notebook, call ``main([...])``
explicitly, for example

    main(["--input", "example/example_input.csv", "--outdir", "output_example"])
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # matplotlib is optional; numerical outputs still work.
    plt = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

Array = np.ndarray


# =============================================================================
# Utility functions
# =============================================================================

def _safe_log1p(x: Array | float, floor: float = 1e-14) -> Array | float:
    """Return log(1+x) with a lower floor for numerical safety."""
    return np.log(np.maximum(1.0 + x, floor))


def _check_required_columns(df: pd.DataFrame) -> None:
    required = ["x1_obs", "x2_obs", "y1", "y2", "delta1", "delta2", "region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def cdf_grid_to_cell_masses(F: Array) -> Array:
    """Convert a gridded bivariate CDF to rectangle cell masses.

    For a grid F[i,j] = F(x_i,y_j), the returned array M has the same shape and
    is defined by the discrete inclusion-exclusion increments

        M[i,j] = F[i,j] - F[i-1,j] - F[i,j-1] + F[i-1,j-1],

    using zero boundary values outside the lower-left corner.
    """
    F = np.asarray(F, dtype=float)
    M = F.copy()
    M[1:, :] -= F[:-1, :]
    M[:, 1:] -= F[:, :-1]
    M[1:, 1:] += F[:-1, :-1]
    return M


def cell_masses_to_cdf_grid(M: Array) -> Array:
    """Reconstruct a gridded bivariate CDF from nonnegative cell masses."""
    M = np.asarray(M, dtype=float)
    return np.cumsum(np.cumsum(M, axis=0), axis=1)


def _cdf_grid_repair(F: Array, enforce_rectangle: bool = True) -> Array:
    """Repair a diagnostic gridded CDF.

    Parameters
    ----------
    F:
        Raw gridded CDF values.
    enforce_rectangle:
        If True, convert the grid to cell masses, clip negative cell masses to
        zero, and reconstruct the CDF. This enforces rectangle positivity on the
        grid. If False, only coordinatewise monotonicity is enforced.

    Notes
    -----
    This function is used only for gridded diagnostic output and convergence
    monitoring. It does not alter the jump-set estimator itself.
    """
    out = np.clip(np.asarray(F, dtype=float), 0.0, 1.0)

    if enforce_rectangle:
        M = cdf_grid_to_cell_masses(out)
        M = np.maximum(M, 0.0)
        total = float(np.sum(M))
        if total > 1.0:
            M /= total
        out = cell_masses_to_cdf_grid(M)
    else:
        out = np.maximum.accumulate(out, axis=0)
        out = np.maximum.accumulate(out, axis=1)

    return np.clip(out, 0.0, 1.0)


def _grid_diagnostics(F: Array, tol: float = 1e-10) -> dict:
    """Return basic CDF and rectangle-positivity diagnostics on a grid."""
    F = np.asarray(F, dtype=float)
    M = cdf_grid_to_cell_masses(F)
    return {
        "F_min": float(np.nanmin(F)),
        "F_max": float(np.nanmax(F)),
        "F_finite": bool(np.isfinite(F).all()),
        "cdf_between_0_and_1": bool((F >= -tol).all() and (F <= 1.0 + tol).all()),
        "cdf_nondecreasing_x1": bool((np.diff(F, axis=0) >= -tol).all()),
        "cdf_nondecreasing_x2": bool((np.diff(F, axis=1) >= -tol).all()),
        "rectangle_min_cell_mass": float(np.nanmin(M)),
        "rectangle_negative_cells": int((M < -tol).sum()),
        "rectangle_positive": bool((M >= -tol).all()),
        "grid_total_mass": float(F[-1, -1]),
    }


# =============================================================================
# Region-A initialization
# =============================================================================

@dataclass
class StepCDF:
    """Discrete bivariate CDF represented by nonnegative point masses."""

    x1: Array
    x2: Array
    mass: Array

    def __post_init__(self) -> None:
        self.x1 = np.asarray(self.x1, dtype=float)
        self.x2 = np.asarray(self.x2, dtype=float)
        self.mass = np.asarray(self.mass, dtype=float)
        self.mass = np.maximum(self.mass, 0.0)
        total = float(np.sum(self.mass))
        if total <= 0 or not np.isfinite(total):
            self.mass = np.ones_like(self.x1, dtype=float) / max(len(self.x1), 1)
        else:
            self.mass = self.mass / total

    def __call__(self, x1q, x2q) -> Array:
        x1q = np.asarray(x1q, dtype=float)
        x2q = np.asarray(x2q, dtype=float)
        bshape = np.broadcast_shapes(x1q.shape, x2q.shape)
        q1 = np.broadcast_to(x1q, bshape).ravel()
        q2 = np.broadcast_to(x2q, bshape).ravel()
        out = np.empty_like(q1, dtype=float)

        for start in range(0, len(q1), 2048):
            end = min(len(q1), start + 2048)
            mask = (
                (self.x1[None, :] <= q1[start:end, None])
                & (self.x2[None, :] <= q2[start:end, None])
            )
            out[start:end] = mask @ self.mass
        return out.reshape(bshape)


def van_der_laan_initial_estimator(df_abc: pd.DataFrame) -> tuple[StepCDF, pd.DataFrame, dict]:
    """Reference Region-A initialization.

    The initialization assigns mass to Region-A detections using inverse
    bivariate comparable-set sizes. This is a deliberately simple, stable
    reference initialization for Algorithm 3. The subsequent iteration updates
    observation weights and applies the weighted Dabrowska estimator to A+B+C.
    """
    _check_required_columns(df_abc)
    a = df_abc[(df_abc["delta1"] == 1) & (df_abc["delta2"] == 1)].copy()
    if len(a) == 0:
        raise ValueError("Region A is empty; initialization cannot be constructed.")

    x1 = a["x1_obs"].to_numpy(float)
    x2 = a["x2_obs"].to_numpy(float)
    y1 = a["y1"].to_numpy(float)
    y2 = a["y2"].to_numpy(float)

    keep = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y1) & np.isfinite(y2)
    keep &= (x1 >= y1) & (x2 >= y2)
    x1, x2, y1, y2 = x1[keep], x2[keep], y1[keep], y2[keep]

    if len(x1) == 0:
        raise ValueError("No valid Region-A objects remain after filtering.")

    comparable_size = np.empty(len(x1), dtype=float)
    for i, (xx1, xx2) in enumerate(zip(x1, x2)):
        # Comparable set in Section 3: Y1_j <= x1_i <= X1_j and
        # Y2_j <= x2_i <= X2_j.
        comparable_size[i] = np.sum(
            (y1 <= xx1) & (xx1 <= x1) & (y2 <= xx2) & (xx2 <= x2)
        )

    mass = 1.0 / np.maximum(comparable_size, 1.0)
    mass = mass / np.sum(mass)

    cdf = StepCDF(x1=x1, x2=x2, mass=mass)
    mass_df = pd.DataFrame({"x1": x1, "x2": x2, "mass": mass})
    info = {
        "n_region_A": int(len(x1)),
        "vdlaan_initialization": "inverse_comparable_set_reference",
    }
    return cdf, mass_df, info


# =============================================================================
# Weighted Dabrowska estimator on J_n = U_n x V_n
# =============================================================================

def _weighted_margin_table(z: Array, event: Array, w: Array, eps: float = 1e-12) -> pd.DataFrame:
    """One-dimensional weighted Kaplan--Meier log factors."""
    z = np.asarray(z, dtype=float)
    event = np.asarray(event, dtype=bool)
    w = np.asarray(w, dtype=float)

    times = np.array(sorted(pd.unique(z[event & np.isfinite(z)])), dtype=float)
    rows: list[dict] = []
    log_km = 0.0

    for u in times:
        risk = float(np.sum(w[z >= u]))
        num = float(np.sum(w[(z == u) & event]))
        hazard = 0.0 if risk <= eps else num / risk
        hazard = float(np.clip(hazard, 0.0, 1.0 - eps))
        log_km += math.log1p(-hazard)
        rows.append({"z": u, "risk_w": risk, "event_w": num, "hazard_w": hazard, "log_km": log_km})

    return pd.DataFrame(rows)


@dataclass
class WeightedDabrowskaJumpEstimator:
    """Weighted bivariate Dabrowska estimator on J_n = U_n x V_n."""

    margin1_table: pd.DataFrame
    margin2_table: pd.DataFrame
    jump_table: pd.DataFrame
    U: Array
    V: Array
    log1p_Q_grid: Array
    prefix_log1p_Q: Array
    eps: float = 1e-12

    def _marginal_log_factor(self, table: pd.DataFrame, tq: Array) -> Array:
        tq = np.asarray(tq, dtype=float)
        if len(table) == 0:
            return np.zeros_like(tq, dtype=float)

        t = table["z"].to_numpy(float)
        log_cum = table["log_km"].to_numpy(float)
        idx = np.searchsorted(t, tq, side="right") - 1
        out = np.zeros_like(tq, dtype=float)
        ok = idx >= 0
        out[ok] = log_cum[idx[ok]]
        return out

    def _bivariate_log_factor(self, q1: Array, q2: Array) -> Array:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        if len(self.U) == 0 or len(self.V) == 0:
            return np.zeros_like(q1, dtype=float)

        i = np.searchsorted(self.U, q1, side="right") - 1
        j = np.searchsorted(self.V, q2, side="right") - 1
        out = np.zeros_like(q1, dtype=float)
        ok = (i >= 0) & (j >= 0)
        out[ok] = self.prefix_log1p_Q[i[ok], j[ok]]
        return out

    def survival_z(self, z1q, z2q) -> Array:
        """Evaluate S_Z(z1,z2) by the Dabrowska product integral."""
        z1q = np.asarray(z1q, dtype=float)
        z2q = np.asarray(z2q, dtype=float)
        bshape = np.broadcast_shapes(z1q.shape, z2q.shape)
        q1 = np.broadcast_to(z1q, bshape).ravel()
        q2 = np.broadcast_to(z2q, bshape).ravel()

        log_survival = self._marginal_log_factor(self.margin1_table, q1)
        log_survival += self._marginal_log_factor(self.margin2_table, q2)
        log_survival += self._bivariate_log_factor(q1, q2)
        return np.clip(np.exp(log_survival).reshape(bshape), 0.0, 1.0)

    def cdf_x(self, x1q, x2q) -> Array:
        """Luminosity-space CDF: F_X(x1,x2) = S_Z(-x1,-x2)."""
        return self.survival_z(-np.asarray(x1q, dtype=float), -np.asarray(x2q, dtype=float))


def _exact_event_indices(U: Array, V: Array, z1_sub: Array, z2_sub: Array) -> tuple[Array, Array, Array]:
    i = np.searchsorted(U, z1_sub, side="left")
    j = np.searchsorted(V, z2_sub, side="left")
    ok = (i >= 0) & (i < len(U)) & (j >= 0) & (j < len(V))
    ok_exact = np.zeros_like(ok, dtype=bool)
    valid = np.where(ok)[0]
    ok_exact[valid] = (U[i[valid]] == z1_sub[valid]) & (V[j[valid]] == z2_sub[valid])
    return i, j, ok_exact


def weighted_dabrowska_jump_estimator(
    df_abc: pd.DataFrame,
    weights: Optional[Array] = None,
    eps: float = 1e-12,
) -> WeightedDabrowskaJumpEstimator:
    """Construct the weighted Dabrowska estimator on J_n = U_n x V_n.

    Section 3.6 correspondence
    --------------------------
    Z_b = -X_b^obs:
        The bivariate CDF in luminosity space is evaluated as
        F_X(x1,x2) = S_Z(-x1,-x2).

    Eq. (dabrowska_jump_set):
        U_n = {u: Z_1i = u, Delta_1i = 1},
        V_n = {v: Z_2i = v, Delta_2i = 1},
        J_n = U_n x V_n.

    Eq. (weighted_bivariate_risk):
        R_w(u,v) = sum_i w_i I(Z_1i >= u, Z_2i >= v).

    Eqs. (weighted_N11), (weighted_N10), (weighted_N01):
        Weighted joint and one-sided counting processes on J_n.

    Eqs. (weighted_DeltaLambda11)--(weighted_DeltaLambda01):
        Delta Lambda_ab,w(u,v) = N_ab,w(u,v) / R_w(u,v).

    Eq. (weighted_gamma):
        Gamma_w(u,v) = [Delta Lambda_11 - Delta Lambda_10 Delta Lambda_01]
        / [(1 - Delta Lambda_10)(1 - Delta Lambda_01)].
    """
    _check_required_columns(df_abc)
    dat = df_abc[df_abc["region"].isin(["A", "B", "C"])].copy().reset_index(drop=True)
    if len(dat) == 0:
        raise ValueError("No A+B+C objects are available.")

    z1 = -dat["x1_obs"].to_numpy(float)
    z2 = -dat["x2_obs"].to_numpy(float)
    d1 = dat["delta1"].to_numpy(int) == 1
    d2 = dat["delta2"].to_numpy(int) == 1

    finite = np.isfinite(z1) & np.isfinite(z2)
    z1, z2, d1, d2 = z1[finite], z2[finite], d1[finite], d2[finite]
    if len(z1) == 0:
        raise ValueError("No finite observed times remain after filtering.")

    if weights is None:
        w = np.ones(len(z1), dtype=float) / len(z1)
    else:
        w0 = np.asarray(weights, dtype=float)
        if len(w0) != len(dat):
            raise ValueError("weights must have one entry per A+B+C object.")
        w = np.maximum(w0[finite], 0.0)
        total_weight = float(np.sum(w))
        if total_weight <= 0 or not np.isfinite(total_weight):
            raise ValueError("weights must have a positive finite sum.")
        w = w / total_weight

    # Eq. (dabrowska_jump_set): J_n = U_n x V_n.
    U = np.array(sorted(pd.unique(z1[d1])), dtype=float)
    V = np.array(sorted(pd.unique(z2[d2])), dtype=float)

    columns = [
        "z1", "z2", "risk_w", "num10_w", "num01_w", "num11_w",
        "DeltaLambda10_w", "DeltaLambda01_w", "DeltaLambda11_w",
        "Gamma_w", "Q_w", "log1p_Q_w",
    ]

    margin1_table = _weighted_margin_table(z1, d1, w, eps)
    margin2_table = _weighted_margin_table(z2, d2, w, eps)

    if len(U) == 0 or len(V) == 0:
        empty = np.zeros((len(U), len(V)), dtype=float)
        return WeightedDabrowskaJumpEstimator(
            margin1_table=margin1_table,
            margin2_table=margin2_table,
            jump_table=pd.DataFrame(columns=columns),
            U=U,
            V=V,
            log1p_Q_grid=empty,
            prefix_log1p_Q=empty,
            eps=eps,
        )

    m1, m2 = len(U), len(V)

    # Eq. (weighted_bivariate_risk): R_w(u,v).
    # Each object contributes to all grid points with u <= Z_1i and v <= Z_2i.
    # We bin at the largest event-time indices not exceeding (Z_1i,Z_2i),
    # and then take a two-dimensional suffix sum.
    risk_atom = np.zeros((m1, m2), dtype=float)
    iu = np.searchsorted(U, z1, side="right") - 1
    iv = np.searchsorted(V, z2, side="right") - 1
    ok = (iu >= 0) & (iv >= 0)
    np.add.at(risk_atom, (iu[ok], iv[ok]), w[ok])
    risk_w = np.cumsum(np.cumsum(risk_atom[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]

    # Eq. (weighted_N11): N_11,w(u,v).
    num11 = np.zeros((m1, m2), dtype=float)
    mask11 = d1 & d2
    i11, j11, ok11 = _exact_event_indices(U, V, z1[mask11], z2[mask11])
    np.add.at(num11, (i11[ok11], j11[ok11]), w[mask11][ok11])

    # Eq. (weighted_N10): N_10,w(u,v).
    num10_atom = np.zeros((m1, m2), dtype=float)
    mask10 = d1
    z1_10, z2_10, w_10 = z1[mask10], z2[mask10], w[mask10]
    i10 = np.searchsorted(U, z1_10, side="left")
    j10 = np.searchsorted(V, z2_10, side="right") - 1
    ok10 = (i10 >= 0) & (i10 < m1) & (j10 >= 0) & (j10 < m2)
    valid10 = np.where(ok10)[0]
    ok10_exact = np.zeros_like(ok10, dtype=bool)
    ok10_exact[valid10] = U[i10[valid10]] == z1_10[valid10]
    np.add.at(num10_atom, (i10[ok10_exact], j10[ok10_exact]), w_10[ok10_exact])
    num10 = np.cumsum(num10_atom[:, ::-1], axis=1)[:, ::-1]

    # Eq. (weighted_N01): N_01,w(u,v).
    num01_atom = np.zeros((m1, m2), dtype=float)
    mask01 = d2
    z1_01, z2_01, w_01 = z1[mask01], z2[mask01], w[mask01]
    i01 = np.searchsorted(U, z1_01, side="right") - 1
    j01 = np.searchsorted(V, z2_01, side="left")
    ok01 = (i01 >= 0) & (i01 < m1) & (j01 >= 0) & (j01 < m2)
    valid01 = np.where(ok01)[0]
    ok01_exact = np.zeros_like(ok01, dtype=bool)
    ok01_exact[valid01] = V[j01[valid01]] == z2_01[valid01]
    np.add.at(num01_atom, (i01[ok01_exact], j01[ok01_exact]), w_01[ok01_exact])
    num01 = np.cumsum(num01_atom[::-1, :], axis=0)[::-1, :]

    # Eqs. (weighted_DeltaLambda11)--(weighted_DeltaLambda01).
    safe_risk = np.maximum(risk_w, eps)
    dL11 = np.where(risk_w > eps, num11 / safe_risk, 0.0)
    dL10 = np.where(risk_w > eps, num10 / safe_risk, 0.0)
    dL01 = np.where(risk_w > eps, num01 / safe_risk, 0.0)
    dL11 = np.clip(dL11, 0.0, 1.0 - eps)
    dL10 = np.clip(dL10, 0.0, 1.0 - eps)
    dL01 = np.clip(dL01, 0.0, 1.0 - eps)

    # Eq. (weighted_gamma).
    denom = np.maximum((1.0 - dL10) * (1.0 - dL01), eps)
    gamma = (dL11 - dL10 * dL01) / denom
    gamma = np.maximum(gamma, -1.0 + eps)
    log1p_Q_grid = _safe_log1p(gamma, floor=eps)
    prefix_log1p_Q = np.cumsum(np.cumsum(log1p_Q_grid, axis=0), axis=1)

    Z1, Z2 = np.meshgrid(U, V, indexing="ij")
    jump_table = pd.DataFrame(
        {
            "z1": Z1.ravel(),
            "z2": Z2.ravel(),
            "risk_w": risk_w.ravel(),
            "num10_w": num10.ravel(),
            "num01_w": num01.ravel(),
            "num11_w": num11.ravel(),
            "DeltaLambda10_w": dL10.ravel(),
            "DeltaLambda01_w": dL01.ravel(),
            "DeltaLambda11_w": dL11.ravel(),
            "Gamma_w": gamma.ravel(),
            "Q_w": gamma.ravel(),
            "log1p_Q_w": log1p_Q_grid.ravel(),
        }
    )

    return WeightedDabrowskaJumpEstimator(
        margin1_table=margin1_table,
        margin2_table=margin2_table,
        jump_table=jump_table,
        U=U,
        V=V,
        log1p_Q_grid=log1p_Q_grid,
        prefix_log1p_Q=prefix_log1p_Q,
        eps=eps,
    )


# =============================================================================
# Algorithm 3 wrapper
# =============================================================================

@dataclass
class CombinedEstimatorResult:
    estimator: WeightedDabrowskaJumpEstimator
    initial_cdf: StepCDF
    region_a_masses: pd.DataFrame
    weights: Array
    iterations: int
    convergence_diff: float
    summary: dict

    def cdf(self, x1, x2) -> Array:
        return self.estimator.cdf_x(x1, x2)


def fit_algorithm3(
    df: pd.DataFrame,
    eval_x1: Array,
    eval_x2: Array,
    epsilon: float = 5e-4,
    max_iter: int = 10,
    min_iter: int = 2,
    pi_floor: float = 1e-5,
    enforce_rectangle: bool = True,
    verbose: bool = True,
) -> CombinedEstimatorResult:
    """Fit the combined van der Laan--Dabrowska estimator."""
    _check_required_columns(df)
    abc = df[df["region"].isin(["A", "B", "C"])].copy().reset_index(drop=True)
    if len(abc) == 0:
        raise ValueError("A+B+C sample is empty.")

    initial_cdf, mass_df, init_info = van_der_laan_initial_estimator(abc)
    F_old = initial_cdf

    X1g, X2g = np.meshgrid(eval_x1, eval_x2, indexing="ij")
    old_grid = _cdf_grid_repair(F_old(X1g, X2g), enforce_rectangle=enforce_rectangle)

    weights = np.ones(len(abc), dtype=float) / len(abc)
    est: Optional[WeightedDabrowskaJumpEstimator] = None
    diff = np.inf

    for it in range(1, max_iter + 1):
        Fy = np.asarray(F_old(abc["y1"].to_numpy(float), abc["y2"].to_numpy(float)), dtype=float)
        pi = np.maximum(1.0 - Fy, pi_floor)
        weights = 1.0 / pi
        weights = weights / np.sum(weights)

        est = weighted_dabrowska_jump_estimator(abc, weights=weights)
        new_grid = _cdf_grid_repair(est.cdf_x(X1g, X2g), enforce_rectangle=enforce_rectangle)
        diff = float(np.max(np.abs(new_grid - old_grid)))

        F_old = lambda q1, q2, _est=est: _est.cdf_x(q1, q2)
        old_grid = new_grid

        if verbose:
            print({"iteration": it, "diff": diff, "n_jump_rows": int(len(est.jump_table))})

        if it >= min_iter and diff < epsilon:
            break

    if est is None:
        raise RuntimeError("Algorithm 3 did not construct an estimator.")

    n_U = int(len(est.U))
    n_V = int(len(est.V))

    summary = {
        **init_info,
        "algorithm3_iterations": int(it),
        "algorithm3_convergence_diff": float(diff),
        "n_ABC": int(len(abc)),
        "n_A": int((abc["region"] == "A").sum()),
        "n_B": int((abc["region"] == "B").sum()),
        "n_C": int((abc["region"] == "C").sum()),
        "weight_sum": float(np.sum(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "n_U_event_times": n_U,
        "n_V_event_times": n_V,
        "n_jump_rows": int(len(est.jump_table)),
        "jump_rows_match_UxV": bool(len(est.jump_table) == n_U * n_V),
        "enforce_rectangle_grid_repair": bool(enforce_rectangle),
    }

    return CombinedEstimatorResult(
        estimator=est,
        initial_cdf=initial_cdf,
        region_a_masses=mass_df,
        weights=weights,
        iterations=it,
        convergence_diff=diff,
        summary=summary,
    )


# =============================================================================
# Output helpers
# =============================================================================

def evaluate_on_grid(
    result: CombinedEstimatorResult,
    x1_grid: Array,
    x2_grid: Array,
    enforce_rectangle: bool = True,
) -> pd.DataFrame:
    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")
    F_raw = result.cdf(X1, X2)
    F = _cdf_grid_repair(F_raw, enforce_rectangle=enforce_rectangle)
    return pd.DataFrame({"x1": X1.ravel(), "x2": X2.ravel(), "F_TC": F.ravel()})



# =============================================================================
# Figure-generation helpers
# =============================================================================

def _require_matplotlib() -> None:
    """Raise a clear error if matplotlib is not available."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for figure generation. "
            "Install matplotlib or run with --no-plots."
        )


def _save_current_figure(path: Path, dpi: int = 180) -> None:
    """Save and close the current matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _plot_heatmap(
    matrix: Array,
    x_grid: Array,
    y_grid: Array,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    path: Path,
    title: Optional[str] = None,
    dpi: int = 180,
) -> None:
    """Write a rectangular heatmap for a gridded diagnostic quantity.

    If title is None or an empty string, no in-figure title is drawn.
    This is useful when figures are intended for manuscript or slide panels
    where captions or surrounding text provide the title.
    """
    _require_matplotlib()
    matrix = np.asarray(matrix, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        extent=[float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())],
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    _save_current_figure(path, dpi=dpi)


def _plot_step_margin(table: pd.DataFrame, title: str, path: Path, dpi: int = 180) -> None:
    """Write a one-dimensional weighted Kaplan--Meier step plot."""
    _require_matplotlib()
    plt.figure(figsize=(5.4, 3.8))
    if len(table) > 0:
        z = table["z"].to_numpy(float)
        S = np.exp(table["log_km"].to_numpy(float))
        plt.step(z, S, where="post")
    plt.xlabel("z")
    plt.ylabel("weighted marginal survival")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    _save_current_figure(path, dpi=dpi)


def _plot_weights(weights: Array, path: Path, dpi: int = 180) -> None:
    """Write a histogram of final Algorithm 3 weights."""
    _require_matplotlib()
    plt.figure(figsize=(5.4, 3.8))
    plt.hist(np.asarray(weights, dtype=float), bins=30)
    plt.xlabel("final normalized weight")
    plt.ylabel("count")
    plt.title("Algorithm 3 final weights")
    _save_current_figure(path, dpi=dpi)



def cdf_grid_to_cell_pdf(F: Array, x1_grid: Array, x2_grid: Array) -> tuple[Array, Array, Array, Array]:
    """Convert a gridded CDF to cell masses and cell-averaged PDF values.

    The returned cell arrays have shape (len(x1_grid)-1, len(x2_grid)-1) and
    live on the rectangular cells [x1_i,x1_{i+1}] x [x2_j,x2_{j+1}].
    """
    F = np.asarray(F, dtype=float)
    x1_grid = np.asarray(x1_grid, dtype=float)
    x2_grid = np.asarray(x2_grid, dtype=float)
    if F.shape != (len(x1_grid), len(x2_grid)):
        raise ValueError("F must have shape (len(x1_grid), len(x2_grid)).")
    dx1 = np.diff(x1_grid)
    dx2 = np.diff(x2_grid)
    if np.any(dx1 <= 0) or np.any(dx2 <= 0):
        raise ValueError("x1_grid and x2_grid must be strictly increasing.")

    cell_mass = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    cell_mass = np.maximum(cell_mass, 0.0)
    pdf = cell_mass / (dx1[:, None] * dx2[None, :])
    x1_mid = 0.5 * (x1_grid[:-1] + x1_grid[1:])
    x2_mid = 0.5 * (x2_grid[:-1] + x2_grid[1:])
    return cell_mass, pdf, x1_mid, x2_mid


def smooth_for_display(A: Array, sigma: float = 1.0) -> Array:
    """Smooth a two-dimensional diagnostic array for visualization only.

    This smoothing is not part of the estimator.  It is applied only to PDF
    figures, and the unsmoothed numerical PDF values are still written to CSV.
    If scipy is available, a Gaussian filter is used.  Otherwise, the function
    falls back to repeated nearest-neighbor averaging.
    """
    A = np.asarray(A, dtype=float)
    if sigma <= 0:
        return A.copy()

    try:
        from scipy.ndimage import gaussian_filter
        B = gaussian_filter(A, sigma=float(sigma), mode="nearest")
    except Exception:
        # Minimal dependency-free fallback.  The number of passes roughly grows
        # with sigma, but this branch is only for visualization.
        B = A.copy()
        n_pass = max(1, int(round(float(sigma))))
        for _ in range(n_pass):
            P = np.pad(B, 1, mode="edge")
            B = (
                P[1:-1, 1:-1]
                + P[:-2, 1:-1]
                + P[2:, 1:-1]
                + P[1:-1, :-2]
                + P[1:-1, 2:]
            ) / 5.0

    return np.maximum(B, 0.0)


def _log10_for_display(A: Array, floor_fraction: float = 1e-8) -> Array:
    """Return log10(A+floor) for display, with a data-dependent floor."""
    A = np.asarray(A, dtype=float)
    finite = A[np.isfinite(A)]
    if finite.size == 0:
        floor = 1e-12
    else:
        floor = max(float(np.nanmax(finite)) * floor_fraction, 1e-12)
    return np.log10(np.maximum(A, 0.0) + floor)


def empirical_cdf_grid_from_truth(df: pd.DataFrame, x1_grid: Array, x2_grid: Array) -> Optional[Array]:
    """Empirical truth CDF on a grid if x1_true and x2_true columns are present."""
    if not {"x1_true", "x2_true"}.issubset(df.columns):
        return None
    truth = df[["x1_true", "x2_true"]].dropna().to_numpy(float)
    if truth.size == 0:
        return None
    F = np.empty((len(x1_grid), len(x2_grid)), dtype=float)
    for i, x1 in enumerate(x1_grid):
        m1 = truth[:, 0] <= x1
        for j, x2 in enumerate(x2_grid):
            F[i, j] = np.mean(m1 & (truth[:, 1] <= x2))
    return _cdf_grid_repair(F, enforce_rectangle=True)


def _plot_heatmap_symmetric(
    matrix: Array,
    x_grid: Array,
    y_grid: Array,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    path: Path,
    title: Optional[str] = None,
    percentile: float = 99.0,
    dpi: int = 180,
) -> None:
    """Write a heatmap with symmetric color limits around zero.

    If title is None or an empty string, no in-figure title is drawn.
    """
    _require_matplotlib()
    matrix = np.asarray(matrix, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.percentile(np.abs(finite), percentile)) if finite.size else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        extent=[float(np.min(x_grid)), float(np.max(x_grid)), float(np.min(y_grid)), float(np.max(y_grid))],
        vmin=-vmax,
        vmax=vmax,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    _save_current_figure(path, dpi=dpi)


def _plot_pdf_products(
    F: Array,
    result: CombinedEstimatorResult,
    x1_grid: Array,
    x2_grid: Array,
    outdir: Path,
    input_df: Optional[pd.DataFrame] = None,
    dpi: int = 180,
    pdf_smooth_sigma: float = 0.0,
) -> dict:
    """Save estimated PDF, log-PDF, A-only comparison, and truth comparison if available.

    Raw PDF values are written to CSV.  If pdf_smooth_sigma > 0, additional
    smoothed PDF figures are written for visualization only.
    """
    _require_matplotlib()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    cell_mass, pdf, x1_mid, x2_mid = cdf_grid_to_cell_pdf(F, x1_grid, x2_grid)
    log_pdf = _log10_for_display(pdf)
    pdf_s = smooth_for_display(pdf, sigma=pdf_smooth_sigma)
    log_pdf_s = _log10_for_display(pdf_s)
    use_smoothing = bool(pdf_smooth_sigma > 0)

    _plot_heatmap(
        pdf,
        x1_mid,
        x2_mid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="estimated PDF",
#        title="Algorithm 3 estimated PDF (raw)",
        path=figdir / "tc_estimator_pdf.png",
        dpi=dpi,
    )
    _plot_heatmap(
        log_pdf,
        x1_mid,
        x2_mid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="log10 PDF",
#        title="Algorithm 3 estimated log PDF (raw)",
        path=figdir / "tc_estimator_log_pdf.png",
        dpi=dpi,
    )
    if use_smoothing:
        _plot_heatmap(
            pdf_s,
            x1_mid,
            x2_mid,
            xlabel="x1",
            ylabel="x2",
            colorbar_label="smoothed estimated PDF",
#            title="Algorithm 3 estimated PDF (smoothed for display)",
            path=figdir / "tc_estimator_pdf_smoothed.png",
            dpi=dpi,
        )
        _plot_heatmap(
            log_pdf_s,
            x1_mid,
            x2_mid,
            xlabel="x1",
            ylabel="x2",
            colorbar_label="log10 smoothed PDF",
#            title="Algorithm 3 estimated log PDF (smoothed for display)",
            path=figdir / "tc_estimator_log_pdf_smoothed.png",
            dpi=dpi,
        )

    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")
    F_a = _cdf_grid_repair(result.initial_cdf(X1, X2), enforce_rectangle=True)
    _, pdf_a, _, _ = cdf_grid_to_cell_pdf(F_a, x1_grid, x2_grid)
    diff_a = pdf - pdf_a
    pdf_a_s = smooth_for_display(pdf_a, sigma=pdf_smooth_sigma)
    diff_a_s = pdf_s - pdf_a_s
    _plot_heatmap(
        pdf_a,
        x1_mid,
        x2_mid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="A-only PDF",
#        title="Region-A initialization PDF (raw)",
        path=figdir / "a_only_pdf.png",
        dpi=dpi,
    )
    _plot_heatmap_symmetric(
        diff_a,
        x1_mid,
        x2_mid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="Algorithm 3 - A-only PDF",
#        title="PDF difference: Algorithm 3 minus A-only (raw)",
        path=figdir / "pdf_difference_algorithm3_minus_a_only.png",
        dpi=dpi,
    )
    if use_smoothing:
        _plot_heatmap(
            pdf_a_s,
            x1_mid,
            x2_mid,
            xlabel="x1",
            ylabel="x2",
            colorbar_label="smoothed A-only PDF",
#            title="Region-A initialization PDF (smoothed for display)",
            path=figdir / "a_only_pdf_smoothed.png",
            dpi=dpi,
        )
        _plot_heatmap_symmetric(
            diff_a_s,
            x1_mid,
            x2_mid,
            xlabel="x1",
            ylabel="x2",
            colorbar_label="Algorithm 3 - A-only PDF",
#            title="PDF difference: Algorithm 3 minus A-only (smoothed)",
            path=figdir / "pdf_difference_algorithm3_minus_a_only_smoothed.png",
            dpi=dpi,
        )

    pd.DataFrame({
        "x1": np.repeat(x1_mid, len(x2_mid)),
        "x2": np.tile(x2_mid, len(x1_mid)),
        "cell_mass": cell_mass.ravel(),
        "pdf": pdf.ravel(),
        "log10_pdf": log_pdf.ravel(),
        "pdf_smoothed_for_display": pdf_s.ravel(),
        "log10_pdf_smoothed_for_display": log_pdf_s.ravel(),
        "pdf_A_only": pdf_a.ravel(),
        "pdf_A_only_smoothed_for_display": pdf_a_s.ravel(),
        "pdf_minus_A_only": diff_a.ravel(),
        "pdf_minus_A_only_smoothed_for_display": diff_a_s.ravel(),
    }).to_csv(outdir / "tc_estimator_pdf_grid.csv", index=False)

    diagnostics = {
        "pdf_min": float(np.nanmin(pdf)),
        "pdf_max": float(np.nanmax(pdf)),
        "pdf_finite": bool(np.isfinite(pdf).all()),
        "pdf_cell_mass_sum": float(np.sum(cell_mass)),
        "pdf_A_only_min": float(np.nanmin(pdf_a)),
        "pdf_A_only_max": float(np.nanmax(pdf_a)),
        "pdf_smoothing_sigma": float(pdf_smooth_sigma),
        "pdf_smoothed_figures_written": bool(use_smoothing),
        "pdf_smoothed_min": float(np.nanmin(pdf_s)),
        "pdf_smoothed_max": float(np.nanmax(pdf_s)),
    }

    if input_df is not None:
        F_true = empirical_cdf_grid_from_truth(input_df, x1_grid, x2_grid)
        if F_true is not None:
            _, pdf_true, _, _ = cdf_grid_to_cell_pdf(F_true, x1_grid, x2_grid)
            diff_true = pdf - pdf_true
            pdf_true_s = smooth_for_display(pdf_true, sigma=pdf_smooth_sigma)
            diff_true_s = pdf_s - pdf_true_s
            _plot_heatmap(
                pdf_true,
                x1_mid,
                x2_mid,
                xlabel="x1",
                ylabel="x2",
                colorbar_label="true empirical PDF",
#                title="True empirical PDF (raw)",
                path=figdir / "true_empirical_pdf.png",
                dpi=dpi,
            )
            _plot_heatmap_symmetric(
                diff_true,
                x1_mid,
                x2_mid,
                xlabel="x1",
                ylabel="x2",
                colorbar_label="Estimated - true PDF",
#                title="PDF residual: estimate minus truth (raw)",
                path=figdir / "pdf_residual_estimate_minus_truth.png",
                dpi=dpi,
            )
            if use_smoothing:
                _plot_heatmap(
                    pdf_true_s,
                    x1_mid,
                    x2_mid,
                    xlabel="x1",
                    ylabel="x2",
                    colorbar_label="smoothed true empirical PDF",
#                    title="True empirical PDF (smoothed for display)",
                    path=figdir / "true_empirical_pdf_smoothed.png",
                    dpi=dpi,
                )
                _plot_heatmap_symmetric(
                    diff_true_s,
                    x1_mid,
                    x2_mid,
                    xlabel="x1",
                    ylabel="x2",
                    colorbar_label="Estimated - true PDF",
#                    title="PDF residual: estimate minus truth (smoothed)",
                    path=figdir / "pdf_residual_estimate_minus_truth_smoothed.png",
                    dpi=dpi,
                )
            pd.DataFrame({
                "x1": np.repeat(x1_mid, len(x2_mid)),
                "x2": np.tile(x2_mid, len(x1_mid)),
                "pdf_true_empirical": pdf_true.ravel(),
                "pdf_true_empirical_smoothed_for_display": pdf_true_s.ravel(),
                "pdf_minus_true": diff_true.ravel(),
                "pdf_minus_true_smoothed_for_display": diff_true_s.ravel(),
            }).to_csv(outdir / "truth_pdf_grid.csv", index=False)
            diagnostics.update({
                "truth_pdf_available": True,
                "pdf_true_min": float(np.nanmin(pdf_true)),
                "pdf_true_max": float(np.nanmax(pdf_true)),
                "pdf_true_smoothed_min": float(np.nanmin(pdf_true_s)),
                "pdf_true_smoothed_max": float(np.nanmax(pdf_true_s)),
                "pdf_rmse_vs_truth": float(np.sqrt(np.mean(diff_true ** 2))),
                "pdf_mae_vs_truth": float(np.mean(np.abs(diff_true))),
                "pdf_smoothed_rmse_vs_truth_smoothed": float(np.sqrt(np.mean(diff_true_s ** 2))),
                "pdf_smoothed_mae_vs_truth_smoothed": float(np.mean(np.abs(diff_true_s))),
            })
        else:
            diagnostics["truth_pdf_available"] = False
    else:
        diagnostics["truth_pdf_available"] = False

    return diagnostics

def save_figures(
    result: CombinedEstimatorResult,
    x1_grid: Array,
    x2_grid: Array,
    outdir: Path,
    enforce_rectangle: bool = True,
    input_df: Optional[pd.DataFrame] = None,
    dpi: int = 180,
    pdf_smooth_sigma: float = 0.0,
) -> dict:
    """Generate diagnostic and scientific figures from a fitted Algorithm 3 result.

    In addition to algorithm diagnostics, this function writes PDF estimates
    obtained by finite differencing the gridded CDF.
    """
    _require_matplotlib()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")
    F_raw = result.cdf(X1, X2)
    F = _cdf_grid_repair(F_raw, enforce_rectangle=enforce_rectangle)
    M = cdf_grid_to_cell_masses(F)

    _plot_heatmap(
        F,
        x1_grid,
        x2_grid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="F_TC(x1,x2)",
#        title="Algorithm 3 estimated CDF",
        path=figdir / "tc_estimator_cdf.png",
        dpi=dpi,
    )
    _plot_heatmap(
        M,
        x1_grid,
        x2_grid,
        xlabel="x1",
        ylabel="x2",
        colorbar_label="rectangle cell mass",
#        title="Gridded CDF rectangle masses",
        path=figdir / "tc_estimator_cell_mass.png",
        dpi=dpi,
    )

    figure_diagnostics = _plot_pdf_products(
        F,
        result,
        x1_grid,
        x2_grid,
        outdir,
        input_df=input_df,
        dpi=dpi,
        pdf_smooth_sigma=pdf_smooth_sigma,
    )

    est = result.estimator
    if len(est.U) > 0 and len(est.V) > 0:
        gamma_grid = np.expm1(est.log1p_Q_grid)
        _plot_heatmap_symmetric(
            gamma_grid,
            est.U,
            est.V,
            xlabel="z1",
            ylabel="z2",
            colorbar_label="Gamma",
#            title="Dabrowska bivariate correction",
            path=figdir / "dabrowska_gamma.png",
            dpi=dpi,
        )
        _plot_heatmap_symmetric(
            est.log1p_Q_grid,
            est.U,
            est.V,
            xlabel="z1",
            ylabel="z2",
            colorbar_label="log(1+Gamma)",
#            title="Dabrowska log correction",
            path=figdir / "dabrowska_log1p_Q.png",
            dpi=dpi,
        )
        figure_diagnostics.update({
            "gamma_min": float(np.nanmin(gamma_grid)),
            "gamma_max": float(np.nanmax(gamma_grid)),
            "gamma_abs_p99": float(np.nanpercentile(np.abs(gamma_grid), 99.0)),
            "log1p_Q_min": float(np.nanmin(est.log1p_Q_grid)),
            "log1p_Q_max": float(np.nanmax(est.log1p_Q_grid)),
        })

    _plot_step_margin(est.margin1_table, "Weighted marginal survival: coordinate 1", figdir / "margin1_survival.png", dpi=dpi)
    _plot_step_margin(est.margin2_table, "Weighted marginal survival: coordinate 2", figdir / "margin2_survival.png", dpi=dpi)
    _plot_weights(result.weights, figdir / "algorithm3_weights.png", dpi=dpi)
    return figure_diagnostics

def save_outputs(
    result: CombinedEstimatorResult,
    x1_grid: Array,
    x2_grid: Array,
    outdir: Path,
    enforce_rectangle: bool = True,
    make_plots: bool = True,
    plot_dpi: int = 180,
    input_df: Optional[pd.DataFrame] = None,
    pdf_smooth_sigma: float = 0.0,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result.estimator.jump_table.to_csv(outdir / "combined_jump_table.csv", index=False)
    result.estimator.margin1_table.to_csv(outdir / "margin1_table.csv", index=False)
    result.estimator.margin2_table.to_csv(outdir / "margin2_table.csv", index=False)
    result.region_a_masses.to_csv(outdir / "region_a_initial_masses.csv", index=False)
    pd.DataFrame({"weight": result.weights}).to_csv(outdir / "algorithm3_weights.csv", index=False)

    grid_df = evaluate_on_grid(result, x1_grid, x2_grid, enforce_rectangle=enforce_rectangle)
    grid_df.to_csv(outdir / "tc_estimator_grid.csv", index=False)

    F = grid_df["F_TC"].to_numpy().reshape(len(x1_grid), len(x2_grid))
    figure_status = "not_requested"
    figure_diagnostics: dict = {}
    if make_plots:
        if plt is None:
            figure_status = "skipped_matplotlib_unavailable"
        else:
            figure_diagnostics = save_figures(
                result,
                x1_grid,
                x2_grid,
                outdir,
                enforce_rectangle=enforce_rectangle,
                input_df=input_df,
                dpi=plot_dpi,
                pdf_smooth_sigma=pdf_smooth_sigma,
            )
            figure_status = "written"

    summary = {
        **result.summary,
        **_grid_diagnostics(F),
        **figure_diagnostics,
        "figures": figure_status,
    }
    pd.DataFrame([summary]).to_csv(outdir / "tc_estimator_summary.csv", index=False)
    with open(outdir / "tc_estimator_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Command line interface
# =============================================================================

def _make_grid_from_data(
    df: pd.DataFrame,
    grid_size: int,
    x1_min: Optional[float] = None,
    x1_max: Optional[float] = None,
    x2_min: Optional[float] = None,
    x2_max: Optional[float] = None,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> tuple[Array, Array]:
    """Construct the evaluation grid from explicit limits or data quantiles."""
    if grid_size < 2:
        raise ValueError("--grid must be at least 2.")

    x1 = df["x1_obs"].to_numpy(float)
    x2 = df["x2_obs"].to_numpy(float)

    lo1 = float(np.nanquantile(x1, q_low)) if x1_min is None else float(x1_min)
    hi1 = float(np.nanquantile(x1, q_high)) if x1_max is None else float(x1_max)
    lo2 = float(np.nanquantile(x2, q_low)) if x2_min is None else float(x2_min)
    hi2 = float(np.nanquantile(x2, q_high)) if x2_max is None else float(x2_max)

    if not (np.isfinite(lo1) and np.isfinite(hi1) and lo1 < hi1):
        raise ValueError("Invalid x1 grid limits.")
    if not (np.isfinite(lo2) and np.isfinite(hi2) and lo2 < hi2):
        raise ValueError("Invalid x2 grid limits.")

    return np.linspace(lo1, hi1, grid_size), np.linspace(lo2, hi2, grid_size)


def load_catalog_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a user-supplied Algorithm 3 catalog."""
    df = pd.read_csv(path)
    _check_required_columns(df)

    df = df.copy()
    df["region"] = df["region"].astype(str).str.upper().str.strip()
    for col in ["x1_obs", "x2_obs", "y1", "y2", "delta1", "delta2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["delta1"] = df["delta1"].astype(int)
    df["delta2"] = df["delta2"].astype(int)

    valid_region = df["region"].isin(["A", "B", "C", "D"])
    if not valid_region.all():
        bad = sorted(df.loc[~valid_region, "region"].unique().tolist())
        raise ValueError(f"Unknown region labels: {bad}. Expected A, B, C, or D.")

    used = df[df["region"].isin(["A", "B", "C"])].copy().reset_index(drop=True)
    finite = np.isfinite(used[["x1_obs", "x2_obs", "y1", "y2"]].to_numpy(float)).all(axis=1)
    if not finite.all():
        used = used.loc[finite].reset_index(drop=True)

    if len(used) == 0:
        raise ValueError("No finite A+B+C observations are available after loading the input CSV.")

    return used


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Algorithm 3 data-I/O implementation for a user-supplied catalog"
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV with x1_obs,x2_obs,y1,y2,delta1,delta2,region")
    parser.add_argument("--outdir", type=str, default="bivariate_survival_outputs", help="Output directory")
    parser.add_argument("--grid", type=int, default=30, help="Evaluation grid size per coordinate")
    parser.add_argument("--x1-min", type=float, default=None, help="Minimum x1 value for evaluation grid")
    parser.add_argument("--x1-max", type=float, default=None, help="Maximum x1 value for evaluation grid")
    parser.add_argument("--x2-min", type=float, default=None, help="Minimum x2 value for evaluation grid")
    parser.add_argument("--x2-max", type=float, default=None, help="Maximum x2 value for evaluation grid")
    parser.add_argument("--q-low", type=float, default=0.05, help="Lower quantile used for grid limits when explicit limits are omitted")
    parser.add_argument("--q-high", type=float, default=0.95, help="Upper quantile used for grid limits when explicit limits are omitted")
    parser.add_argument("--epsilon", type=float, default=5e-4, help="Algorithm 3 convergence threshold")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum Algorithm 3 iterations")
    parser.add_argument("--min-iter", type=int, default=2, help="Minimum Algorithm 3 iterations")
    parser.add_argument("--pi-floor", type=float, default=1e-5, help="Observation-probability floor")
    parser.add_argument("--quiet", action="store_true", help="Suppress iteration logs")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG diagnostic figures")
    parser.add_argument("--plot-dpi", type=int, default=180, help="DPI for diagnostic PNG figures")
    parser.add_argument("--pdf-smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma for PDF figures only; set 0 to disable")
    parser.add_argument(
        "--no-rectangle-repair",
        action="store_true",
        help="Disable nonnegative-cell-mass repair for diagnostic CDF grids",
    )
    args, _unknown = parser.parse_known_args(argv)

    enforce_rectangle = not args.no_rectangle_repair
    df = load_catalog_csv(args.input)

    x1_grid, x2_grid = _make_grid_from_data(
        df,
        grid_size=args.grid,
        x1_min=args.x1_min,
        x1_max=args.x1_max,
        x2_min=args.x2_min,
        x2_max=args.x2_max,
        q_low=args.q_low,
        q_high=args.q_high,
    )

    result = fit_algorithm3(
        df,
        eval_x1=x1_grid,
        eval_x2=x2_grid,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        pi_floor=args.pi_floor,
        enforce_rectangle=enforce_rectangle,
        verbose=not args.quiet,
    )

    outdir = Path(args.outdir)
    save_outputs(
        result,
        x1_grid,
        x2_grid,
        outdir,
        enforce_rectangle=enforce_rectangle,
        make_plots=not args.no_plots,
        plot_dpi=args.plot_dpi,
        input_df=df,
        pdf_smooth_sigma=args.pdf_smooth_sigma,
    )

    grid_df = pd.read_csv(outdir / "tc_estimator_grid.csv")
    F = grid_df["F_TC"].to_numpy().reshape(len(x1_grid), len(x2_grid))
    print(json.dumps({**result.summary, **_grid_diagnostics(F)}, indent=2))
    print(f"Wrote outputs to {outdir.resolve()}")


if __name__ == "__main__":
    # parse_known_args inside main() makes this safe both for command-line use
    # and for Jupyter/IPython, which injects arguments such as -f kernel.json.
    main()
