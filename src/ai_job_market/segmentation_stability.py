from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class ResampleStabilityConfig:
    """Configuration for subsample-based cluster stability.

    The procedure repeatedly draws rows without replacement, refits the same
    algorithm/K on the sampled rows, and compares the sampled solution with the
    reference full-DEV solution on the overlapping rows using Adjusted Rand
    Index (ARI).  ARI is permutation-invariant, so cluster label numbers do not
    need manual alignment.
    """

    n_resamples: int = 20
    sample_fraction: float = 0.85
    min_mean_ari: float = 0.75
    random_seed: int = 2026
    kmeans_n_init: int = 1
    gmm_n_init: int = 5
    gmm_max_iter: int = 1000
    gmm_tol: float = 1e-3
    gmm_reg_covar: float = 1e-5
    min_gmm_convergence_rate: float = 0.95


def _fit_labels_with_diagnostics(
    X: np.ndarray,
    algorithm: str,
    k: int,
    seed: int,
    *,
    kmeans_n_init: int = 1,
    gmm_n_init: int = 5,
    gmm_max_iter: int = 1000,
    gmm_tol: float = 1e-3,
    gmm_reg_covar: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one clustering solution and return labels plus convergence evidence.

    GMM ConvergenceWarning is captured rather than printed to the terminal.
    The warning is converted into explicit evidence so unstable GMM fits can be
    rejected by the robustness gate instead of being silently accepted.
    """
    if algorithm == "KMeans":
        model = KMeans(
            n_clusters=int(k),
            random_state=int(seed),
            n_init=int(kmeans_n_init),
        ).fit(X)
        return model.labels_.astype(int), {
            "converged": True,
            "convergence_warning": False,
            "n_iter": int(getattr(model, "n_iter_", 0)),
        }

    if algorithm == "GMM":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = GaussianMixture(
                n_components=int(k),
                random_state=int(seed),
                n_init=int(gmm_n_init),
                init_params="kmeans",
                reg_covar=float(gmm_reg_covar),
                max_iter=int(gmm_max_iter),
                tol=float(gmm_tol),
            ).fit(X)

        warning_seen = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        converged = bool(getattr(model, "converged_", False)) and not warning_seen
        return model.predict(X).astype(int), {
            "converged": converged,
            "convergence_warning": bool(warning_seen),
            "n_iter": int(getattr(model, "n_iter_", 0)),
        }

    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


def _fit_labels(
    X: np.ndarray,
    algorithm: str,
    k: int,
    seed: int,
    *,
    kmeans_n_init: int = 1,
    gmm_n_init: int = 5,
    gmm_max_iter: int = 1000,
    gmm_tol: float = 1e-3,
    gmm_reg_covar: float = 1e-5,
) -> np.ndarray:
    """Backward-compatible labels-only wrapper."""
    labels, _ = _fit_labels_with_diagnostics(
        X,
        algorithm,
        k,
        seed,
        kmeans_n_init=kmeans_n_init,
        gmm_n_init=gmm_n_init,
        gmm_max_iter=gmm_max_iter,
        gmm_tol=gmm_tol,
        gmm_reg_covar=gmm_reg_covar,
    )
    return labels


def subsample_stability(
    X: np.ndarray,
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    config: ResampleStabilityConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate structural stability under row subsampling.

    This is intentionally different from random-seed stability. Random-seed
    stability answers whether the optimizer converges to the same partition on
    the *same* rows. Subsample stability asks whether a similar partition is
    recovered when the observed sample changes.
    """

    X = np.asarray(X, dtype=float)
    reference_labels = np.asarray(reference_labels, dtype=int)
    n = len(X)
    if n != len(reference_labels):
        raise ValueError("X and reference_labels must have the same number of rows.")
    if n < max(10, int(k) * 3):
        raise ValueError("Too few rows for stable subsample clustering.")

    frac = float(config.sample_fraction)
    if not 0.5 <= frac < 1.0:
        raise ValueError("sample_fraction must be in [0.5, 1.0).")
    n_sample = max(int(np.floor(n * frac)), int(k) * 3)
    n_sample = min(n_sample, n - 1)

    rng = np.random.default_rng(int(config.random_seed) + int(k) * 1009 + (0 if algorithm == "KMeans" else 500_000))
    rows: list[dict[str, float | int | str | bool]] = []
    scores: list[float] = []
    convergence_flags: list[bool] = []
    warning_flags: list[bool] = []

    for i in range(int(config.n_resamples)):
        idx = np.sort(rng.choice(n, size=n_sample, replace=False))
        fit_seed = int(rng.integers(1, 2_147_000_000))
        sampled_labels, fit_diag = _fit_labels_with_diagnostics(
            X[idx],
            algorithm,
            int(k),
            fit_seed,
            kmeans_n_init=config.kmeans_n_init,
            gmm_n_init=config.gmm_n_init,
            gmm_max_iter=config.gmm_max_iter,
            gmm_tol=config.gmm_tol,
            gmm_reg_covar=config.gmm_reg_covar,
        )
        ari = float(adjusted_rand_score(reference_labels[idx], sampled_labels))
        scores.append(ari)
        convergence_flags.append(bool(fit_diag["converged"]))
        warning_flags.append(bool(fit_diag["convergence_warning"]))
        rows.append(
            {
                "algorithm": algorithm,
                "k": int(k),
                "resample": int(i + 1),
                "fit_seed": fit_seed,
                "sample_fraction": frac,
                "sample_rows": int(n_sample),
                "ari_vs_full_reference": ari,
                "converged": bool(fit_diag["converged"]),
                "convergence_warning": bool(fit_diag["convergence_warning"]),
                "n_iter": int(fit_diag["n_iter"]),
            }
        )

    arr = np.asarray(scores, dtype=float)
    conv_arr = np.asarray(convergence_flags, dtype=bool)
    convergence_rate = float(np.mean(conv_arr)) if len(conv_arr) else 1.0
    convergence_gate = (
        True
        if algorithm != "GMM"
        else bool(convergence_rate >= float(config.min_gmm_convergence_rate))
    )
    summary = {
        "resample_stability_ari_mean": float(np.mean(arr)),
        "resample_stability_ari_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "resample_stability_ari_p10": float(np.quantile(arr, 0.10)),
        "resample_stability_ari_min": float(np.min(arr)),
        "resample_stability_runs": int(len(arr)),
        "resample_stability_gate": bool(float(np.mean(arr)) >= float(config.min_mean_ari)),
        "resample_convergence_rate": convergence_rate,
        "resample_convergence_gate": bool(convergence_gate),
        "resample_convergence_warning_runs": int(np.sum(np.asarray(warning_flags, dtype=bool))),
    }
    return summary, pd.DataFrame(rows)
