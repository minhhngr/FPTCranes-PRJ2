from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .segmentation_stability import ResampleStabilityConfig, subsample_stability


@dataclass
class RepresentationSpec:
    representation_id: str
    label: str
    purpose: str
    excluded_raw_features: list[str]
    selected_input_features: list[str]
    dev_matrix: np.ndarray
    all_matrix: np.ndarray
    visual_dev: np.ndarray
    visual_all: np.ndarray
    transformer: dict[str, Any]
    component_summary: pd.DataFrame
    loadings: pd.DataFrame


def _component_count(
    ratios: np.ndarray,
    threshold: float,
    min_components: int = 2,
    max_components: int | None = None,
) -> int:
    ratios = np.asarray(ratios, dtype=float)
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
    cumulative = np.cumsum(ratios)
    n = int(np.searchsorted(cumulative, float(threshold), side="left") + 1)
    n = max(1, int(min_components), n)
    if max_components is not None:
        n = min(n, int(max_components))
    return min(n, len(ratios))


def _safe_pca_fit(X: np.ndarray, seed: int) -> PCA:
    X = np.asarray(X, dtype=float)
    n_components = max(1, min(X.shape[0], X.shape[1]))
    return PCA(n_components=n_components, random_state=seed).fit(X)


def _global_representation(
    *,
    representation_id: str,
    label: str,
    purpose: str,
    encoder: Any,
    X_dev: pd.DataFrame,
    X_all: pd.DataFrame,
    selected_input_features: list[str],
    excluded_raw_features: list[str],
    excluded_encoded_substrings: list[str],
    threshold: float,
    seed: int,
    min_components: int = 2,
    max_components: int | None = 25,
) -> RepresentationSpec:
    # Reconstruct the encoded matrix family-by-family from *unbalanced* blocks.
    # This is important for R2/R3/R4: after an ablated raw feature is removed,
    # the surviving dimensions are rebalanced using the NEW family dimension,
    # so the ablation is applied before the global PCA geometry is formed.
    dev_blocks = encoder.transform_family_blocks(X_dev, balanced=False)
    all_blocks = encoder.transform_family_blocks(X_all, balanced=False)
    all_names = list(map(str, encoder.get_feature_names_out()))

    dev_parts: list[np.ndarray] = []
    all_parts: list[np.ndarray] = []
    kept_name_parts: list[str] = []
    family_dims_after_ablation: dict[str, int] = {}

    for fam, Xd in dev_blocks.items():
        Xa = np.asarray(all_blocks[fam], dtype=float)
        Xd = np.asarray(Xd, dtype=float)
        prefix = f"{fam}::"
        fam_names = [n for n in all_names if n.startswith(prefix)]
        if len(fam_names) != Xd.shape[1]:
            fam_names = [f"{fam}::feature_{i}" for i in range(Xd.shape[1])]

        keep_f = np.ones(len(fam_names), dtype=bool)
        for pattern in excluded_encoded_substrings:
            keep_f &= np.array([pattern not in str(x) for x in fam_names], dtype=bool)
        if not keep_f.any():
            continue

        Xd = Xd[:, keep_f]
        Xa = Xa[:, keep_f]
        kept_fam_names = np.asarray(fam_names, dtype=object)[keep_f]

        # Rebalance after feature ablation, not before.
        d_f = int(Xd.shape[1])
        w_f = 1.0 / math.sqrt(max(1, d_f))
        dev_parts.append(Xd * w_f)
        all_parts.append(Xa * w_f)
        kept_name_parts.extend(map(str, kept_fam_names))
        family_dims_after_ablation[fam] = d_f

    if not dev_parts:
        raise ValueError(f"{representation_id}: all encoded columns were removed.")

    zdev_use = np.hstack(dev_parts)
    zall_use = np.hstack(all_parts)
    kept_names = np.asarray(kept_name_parts, dtype=object)
    keep = np.ones(len(kept_names), dtype=bool)

    pca = _safe_pca_fit(zdev_use, seed)
    pdev_full = pca.transform(zdev_use)
    pall_full = pca.transform(zall_use)
    n = _component_count(
        pca.explained_variance_ratio_,
        threshold,
        min_components=min_components,
        max_components=max_components,
    )
    dev_matrix = pdev_full[:, :n]
    all_matrix = pall_full[:, :n]
    visual_dev = pdev_full[:, : min(2, pdev_full.shape[1])]
    visual_all = pall_full[:, : min(2, pall_full.shape[1])]
    if visual_dev.shape[1] == 1:
        visual_dev = np.column_stack([visual_dev[:, 0], np.zeros(len(visual_dev))])
        visual_all = np.column_stack([visual_all[:, 0], np.zeros(len(visual_all))])

    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    comp_summary = pd.DataFrame(
        {
            "representation_id": representation_id,
            "family": "GLOBAL",
            "component_number": np.arange(1, len(ratios) + 1),
            "component": [f"PC{i}" for i in range(1, len(ratios) + 1)],
            "explained_variance_ratio": ratios,
            "cumulative_variance": np.cumsum(ratios),
            "used_for_clustering": np.arange(1, len(ratios) + 1) <= n,
            "used_for_visualization": np.arange(1, len(ratios) + 1) <= 2,
        }
    )
    comp_summary["variance_threshold"] = float(threshold)
    comp_summary["retained_component_count"] = int(n)
    comp_summary["variance_target_met"] = bool(np.sum(ratios[:n]) >= float(threshold) - 1e-12)

    load_rows: list[dict[str, Any]] = []
    for pc_idx in range(n):
        vals = pca.components_[pc_idx]
        order = np.argsort(np.abs(vals))[::-1][:10]
        for rank, idx in enumerate(order, 1):
            load_rows.append(
                {
                    "representation_id": representation_id,
                    "family": "GLOBAL",
                    "component": f"PC{pc_idx + 1}",
                    "rank_within_component": rank,
                    "encoded_feature": str(kept_names[idx]),
                    "signed_loading": float(vals[idx]),
                    "absolute_loading": float(abs(vals[idx])),
                    "explained_variance_ratio": float(ratios[pc_idx]),
                }
            )
    loadings = pd.DataFrame(load_rows)

    transformer = {
        "kind": "global_pca",
        "representation_id": representation_id,
        "encoder": encoder,
        "excluded_encoded_substrings": list(excluded_encoded_substrings),
        "kept_encoded_feature_names": list(map(str, kept_names)),
        "family_dims_after_ablation": family_dims_after_ablation,
        "pca": pca,
        "n_components": int(n),
        "variance_threshold": float(threshold),
        "max_components": None if max_components is None else int(max_components),
    }
    return RepresentationSpec(
        representation_id=representation_id,
        label=label,
        purpose=purpose,
        excluded_raw_features=excluded_raw_features,
        selected_input_features=selected_input_features,
        dev_matrix=dev_matrix,
        all_matrix=all_matrix,
        visual_dev=visual_dev,
        visual_all=visual_all,
        transformer=transformer,
        component_summary=comp_summary,
        loadings=loadings,
    )


def _familywise_representation(
    *,
    encoder: Any,
    X_dev: pd.DataFrame,
    X_all: pd.DataFrame,
    selected_input_features: list[str],
    threshold: float,
    caps: dict[str, int],
    seed: int,
) -> RepresentationSpec:
    dev_blocks = encoder.transform_family_blocks(X_dev, balanced=False)
    all_blocks = encoder.transform_family_blocks(X_all, balanced=False)

    family_models: dict[str, PCA] = {}
    family_counts: dict[str, int] = {}
    dev_latent: list[np.ndarray] = []
    all_latent: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []

    names = list(map(str, encoder.get_feature_names_out()))
    family_name_map: dict[str, list[str]] = {}
    for fam in dev_blocks:
        prefix = f"{fam}::"
        family_name_map[fam] = [n for n in names if n.startswith(prefix)]

    for fam in dev_blocks:
        Xd = np.asarray(dev_blocks[fam], dtype=float)
        Xa = np.asarray(all_blocks[fam], dtype=float)
        if Xd.shape[1] == 0:
            continue
        pca = _safe_pca_fit(Xd, seed)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
        cap = int(caps.get(fam, Xd.shape[1]))
        n = _component_count(ratios, threshold, min_components=1, max_components=cap)
        Zd = pca.transform(Xd)[:, :n]
        Za = pca.transform(Xa)[:, :n]

        # Balance latent families by their retained dimensionality so a family
        # does not dominate only because it retained more components.
        w = 1.0 / math.sqrt(max(1, n))
        dev_latent.append(Zd * w)
        all_latent.append(Za * w)
        family_models[fam] = pca
        family_counts[fam] = int(n)

        for i, ratio in enumerate(ratios, 1):
            component_rows.append(
                {
                    "representation_id": "R1_FAMILYWISE_PCA",
                    "family": fam,
                    "component_number": i,
                    "component": f"{fam}::PC{i}",
                    "explained_variance_ratio": float(ratio),
                    "cumulative_variance": float(np.sum(ratios[:i])),
                    "used_for_clustering": bool(i <= n),
                    "used_for_visualization": bool(i <= 2),
                    "family_latent_weight": float(w),
                    "variance_threshold": float(threshold),
                    "retained_component_count": int(n),
                    "variance_target_met": bool(np.sum(ratios[:n]) >= float(threshold) - 1e-12),
                    "family_component_cap": int(cap),
                }
            )

        fam_names = family_name_map.get(fam, [])
        if len(fam_names) != Xd.shape[1]:
            fam_names = [f"{fam}::feature_{i}" for i in range(Xd.shape[1])]
        for pc_idx in range(n):
            vals = pca.components_[pc_idx]
            order = np.argsort(np.abs(vals))[::-1][:10]
            for rank, idx in enumerate(order, 1):
                loading_rows.append(
                    {
                        "representation_id": "R1_FAMILYWISE_PCA",
                        "family": fam,
                        "component": f"{fam}::PC{pc_idx + 1}",
                        "rank_within_component": rank,
                        "encoded_feature": str(fam_names[idx]),
                        "signed_loading": float(vals[idx]),
                        "absolute_loading": float(abs(vals[idx])),
                        "explained_variance_ratio": float(ratios[pc_idx]),
                    }
                )

    dev_matrix = np.hstack(dev_latent)
    all_matrix = np.hstack(all_latent)

    # A separate 2D visualizer is fitted only for display; clustering continues
    # to use the concatenated family-wise latent matrix above.
    visualizer = PCA(n_components=2, random_state=seed).fit(dev_matrix)
    visual_dev = visualizer.transform(dev_matrix)
    visual_all = visualizer.transform(all_matrix)

    transformer = {
        "kind": "familywise_pca",
        "representation_id": "R1_FAMILYWISE_PCA",
        "encoder": encoder,
        "family_pcas": family_models,
        "family_component_counts": family_counts,
        "family_latent_weights": {k: 1.0 / math.sqrt(max(1, v)) for k, v in family_counts.items()},
        "visualizer": visualizer,
    }
    return RepresentationSpec(
        representation_id="R1_FAMILYWISE_PCA",
        label="R1 — Family-wise PCA",
        purpose="Compress each feature family separately, cap family latent dimensions, then concatenate balanced latent blocks.",
        excluded_raw_features=[],
        selected_input_features=selected_input_features,
        dev_matrix=dev_matrix,
        all_matrix=all_matrix,
        visual_dev=visual_dev,
        visual_all=visual_all,
        transformer=transformer,
        component_summary=pd.DataFrame(component_rows),
        loadings=pd.DataFrame(loading_rows),
    )


def build_representation_specs(
    *,
    encoder: Any,
    X_dev: pd.DataFrame,
    X_all: pd.DataFrame,
    all_raw_features: list[str],
    threshold: float,
    family_caps: dict[str, int],
    seed: int,
    global_max_components: int | None = 25,
) -> dict[str, RepresentationSpec]:
    specs: dict[str, RepresentationSpec] = {}
    specs["R0_GLOBAL_PCA"] = _global_representation(
        representation_id="R0_GLOBAL_PCA",
        label="R0 — Global family-balanced PCA",
        purpose="Baseline global PCA on the complete family-balanced encoded matrix.",
        encoder=encoder,
        X_dev=X_dev,
        X_all=X_all,
        selected_input_features=list(all_raw_features),
        excluded_raw_features=[],
        excluded_encoded_substrings=[],
        threshold=threshold,
        seed=seed,
        max_components=global_max_components,
    )
    specs["R1_FAMILYWISE_PCA"] = _familywise_representation(
        encoder=encoder,
        X_dev=X_dev,
        X_all=X_all,
        selected_input_features=list(all_raw_features),
        threshold=threshold,
        caps=family_caps,
        seed=seed,
    )
    specs["R2_NO_JOB_CATEGORY"] = _global_representation(
        representation_id="R2_NO_JOB_CATEGORY",
        label="R2 — Global PCA without job_category",
        purpose="Ablation to test whether job_category dominates the segmentation structure.",
        encoder=encoder,
        X_dev=X_dev,
        X_all=X_all,
        selected_input_features=[x for x in all_raw_features if x != "job_category"],
        excluded_raw_features=["job_category"],
        excluded_encoded_substrings=["Job Domain::cat__job_category_"],
        threshold=threshold,
        seed=seed,
        max_components=global_max_components,
    )
    specs["R3_NO_YEARS_EXPERIENCE"] = _global_representation(
        representation_id="R3_NO_YEARS_EXPERIENCE",
        label="R3 — Global PCA without years_of_experience",
        purpose="Ablation to test whether numeric years_of_experience dominates the segmentation structure.",
        encoder=encoder,
        X_dev=X_dev,
        X_all=X_all,
        selected_input_features=[x for x in all_raw_features if x != "years_of_experience"],
        excluded_raw_features=["years_of_experience"],
        excluded_encoded_substrings=["Experience & Education::num__years_of_experience"],
        threshold=threshold,
        seed=seed,
        max_components=global_max_components,
    )
    specs["R4_NO_JOB_CATEGORY_NO_YEARS"] = _global_representation(
        representation_id="R4_NO_JOB_CATEGORY_NO_YEARS",
        label="R4 — Global PCA without job_category + years_of_experience",
        purpose=(
            "Joint ablation to test whether the segmentation structure survives when both "
            "dominant signals are removed at the same time."
        ),
        encoder=encoder,
        X_dev=X_dev,
        X_all=X_all,
        selected_input_features=[
            x for x in all_raw_features if x not in {"job_category", "years_of_experience"}
        ],
        excluded_raw_features=["job_category", "years_of_experience"],
        excluded_encoded_substrings=[
            "Job Domain::cat__job_category_",
            "Experience & Education::num__years_of_experience",
        ],
        threshold=threshold,
        seed=seed,
        max_components=global_max_components,
    )
    return specs


def _pairwise_seed_stability(label_sets: list[np.ndarray]) -> float:
    vals: list[float] = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            vals.append(float(adjusted_rand_score(label_sets[i], label_sets[j])))
    return float(np.mean(vals)) if vals else 1.0


def _fit_gmm_with_diagnostics(
    X: np.ndarray,
    *,
    k: int,
    seed: int,
    n_init: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-3,
    reg_covar: float = 1e-5,
) -> tuple[GaussianMixture, dict[str, Any]]:
    """Fit GMM while converting convergence warnings into explicit evidence."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = GaussianMixture(
            n_components=int(k),
            random_state=int(seed),
            n_init=int(n_init),
            init_params="kmeans",
            reg_covar=float(reg_covar),
            max_iter=int(max_iter),
            tol=float(tol),
        ).fit(X)

    warning_seen = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged = bool(getattr(model, "converged_", False)) and not warning_seen
    return model, {
        "converged": converged,
        "convergence_warning": bool(warning_seen),
        "n_iter": int(getattr(model, "n_iter_", 0)),
    }


def evaluate_candidates(
    X_dev: np.ndarray,
    X_all: np.ndarray,
    *,
    representation_id: str,
    k_min: int,
    k_max: int,
    seed: int,
    stability_seeds: list[int],
    stability_min: float,
    min_cluster_share: float,
    silhouette_tolerance: float,
    resample_config: ResampleStabilityConfig,
    max_stability_candidates_per_algorithm: int = 3,
    full_stability: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Any, np.ndarray, np.ndarray]:
    """Evaluate KMeans/GMM candidates with staged robustness testing.

    In addition to separation, balance, optimizer-seed stability and row-subsample
    stability, GMM candidates must now pass an explicit convergence gate. Any
    ``ConvergenceWarning`` is captured and persisted as evidence rather than
    printed repeatedly to the terminal.
    """
    X_dev = np.asarray(X_dev, dtype=float)
    X_all = np.asarray(X_all, dtype=float)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], Any] = {}
    dev_labels_by_key: dict[tuple[str, int], np.ndarray] = {}
    n_dev = len(X_dev)

    # Pass 1: all Algorithm×K candidates.
    for algorithm in ["KMeans", "GMM"]:
        for k in range(int(k_min), int(k_max) + 1):
            if algorithm == "KMeans":
                main = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_dev)
                labels = main.labels_.astype(int)
                inertia = float(main.inertia_)
                bic = aic = np.nan
                main_diag = {
                    "converged": True,
                    "convergence_warning": False,
                    "n_iter": int(getattr(main, "n_iter_", 0)),
                }
            else:
                main, main_diag = _fit_gmm_with_diagnostics(
                    X_dev,
                    k=k,
                    seed=seed,
                    n_init=max(5, int(resample_config.gmm_n_init)),
                    max_iter=int(resample_config.gmm_max_iter),
                    tol=float(resample_config.gmm_tol),
                    reg_covar=float(resample_config.gmm_reg_covar),
                )
                labels = main.predict(X_dev).astype(int)
                inertia = np.nan
                bic = float(main.bic(X_dev))
                aic = float(main.aic(X_dev))

            if len(np.unique(labels)) < 2:
                continue

            counts = pd.Series(labels).value_counts()
            shares = counts / n_dev
            min_share = float(shares.min())
            entropy = float(-(shares * np.log(shares)).sum() / np.log(k)) if k > 1 else 1.0
            sil = float(silhouette_score(X_dev, labels))
            ch = float(calinski_harabasz_score(X_dev, labels))
            db = float(davies_bouldin_score(X_dev, labels))

            convergence_gate = bool(main_diag["converged"]) if algorithm == "GMM" else True
            rows.append(
                {
                    "representation_id": representation_id,
                    "algorithm": algorithm,
                    "k": int(k),
                    "silhouette": sil,
                    "stability_ari": np.nan,
                    "stability_evaluated": False,
                    "seed_convergence_rate": np.nan,
                    "seed_convergence_gate": True if algorithm == "KMeans" else False,
                    "resample_stability_ari_mean": np.nan,
                    "resample_stability_ari_std": np.nan,
                    "resample_stability_ari_p10": np.nan,
                    "resample_stability_ari_min": np.nan,
                    "resample_stability_runs": 0,
                    "resample_stability_gate": False,
                    "resample_evaluated": False,
                    "resample_convergence_rate": np.nan,
                    "resample_convergence_gate": True if algorithm == "KMeans" else False,
                    "resample_convergence_warning_runs": 0,
                    "min_cluster_share": min_share,
                    "balance_entropy": entropy,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "inertia": inertia,
                    "gmm_bic": bic,
                    "gmm_aic": aic,
                    "gmm_converged": bool(main_diag["converged"]) if algorithm == "GMM" else True,
                    "gmm_n_iter": int(main_diag["n_iter"]),
                    "gmm_convergence_warning": bool(main_diag["convergence_warning"])
                    if algorithm == "GMM"
                    else False,
                    "convergence_gate": convergence_gate,
                    "balance_gate": bool(min_share >= min_cluster_share),
                    "stability_gate": False,
                    "eligible": False,
                }
            )
            models[(algorithm, k)] = main
            dev_labels_by_key[(algorithm, k)] = labels

            labels_all = main.predict(X_all).astype(int)
            for rid, cl in enumerate(labels_all, 1):
                candidate_rows.append(
                    {
                        "representation_id": representation_id,
                        "record_id": rid,
                        "algorithm": algorithm,
                        "k": int(k),
                        "cluster": int(cl),
                    }
                )

    ev = pd.DataFrame(rows)
    if ev.empty:
        raise ValueError(f"{representation_id}: no valid clustering candidates.")

    # Pass 2: optimizer-seed and row-subsample stability.
    run_tables: list[pd.DataFrame] = []
    for algorithm in ["KMeans", "GMM"]:
        if full_stability:
            cand = ev[ev["algorithm"] == algorithm].sort_values("k")
        else:
            cand = ev[
                (ev["algorithm"] == algorithm) & ev["balance_gate"] & ev["convergence_gate"]
            ].sort_values("silhouette", ascending=False)
            if cand.empty:
                cand = ev[(ev["algorithm"] == algorithm) & ev["convergence_gate"]].sort_values(
                    "silhouette", ascending=False
                )
            if cand.empty:
                # Preserve full evidence even if every GMM candidate failed the
                # main-fit convergence gate; those candidates cannot be eligible.
                cand = ev[ev["algorithm"] == algorithm].sort_values("silhouette", ascending=False)

        attempts = 0
        for row in cand.itertuples(index=True):
            if (not full_stability) and attempts >= max(
                1, int(max_stability_candidates_per_algorithm)
            ):
                break
            attempts += 1
            key = (algorithm, int(row.k))
            label_sets: list[np.ndarray] = []
            seed_convergence_flags: list[bool] = []

            for s in stability_seeds:
                if algorithm == "KMeans":
                    m = KMeans(
                        n_clusters=int(row.k),
                        random_state=int(s),
                        n_init=2,
                    ).fit(X_dev)
                    lab = m.labels_.astype(int)
                    seed_convergence_flags.append(True)
                else:
                    m, diag = _fit_gmm_with_diagnostics(
                        X_dev,
                        k=int(row.k),
                        seed=int(s),
                        n_init=max(2, int(resample_config.gmm_n_init)),
                        max_iter=int(resample_config.gmm_max_iter),
                        tol=float(resample_config.gmm_tol),
                        reg_covar=float(resample_config.gmm_reg_covar),
                    )
                    lab = m.predict(X_dev).astype(int)
                    seed_convergence_flags.append(bool(diag["converged"]))
                label_sets.append(lab)

            seed_ari = _pairwise_seed_stability(label_sets)
            seed_convergence_rate = (
                float(np.mean(seed_convergence_flags)) if seed_convergence_flags else 1.0
            )
            seed_convergence_gate = (
                True
                if algorithm != "GMM"
                else bool(seed_convergence_rate >= float(resample_config.min_gmm_convergence_rate))
            )

            ev.loc[row.Index, "stability_evaluated"] = True
            ev.loc[row.Index, "stability_ari"] = seed_ari
            ev.loc[row.Index, "seed_convergence_rate"] = seed_convergence_rate
            ev.loc[row.Index, "seed_convergence_gate"] = bool(seed_convergence_gate)
            ev.loc[row.Index, "stability_gate"] = bool(
                seed_ari >= stability_min and seed_convergence_gate
            )

            # In full-evidence mode evaluate resampling even when an earlier
            # gate fails, so the diagnostic table remains complete.
            if full_stability or (seed_ari >= stability_min and seed_convergence_gate):
                res_summary, res_runs = subsample_stability(
                    X_dev,
                    dev_labels_by_key[key],
                    algorithm,
                    int(row.k),
                    resample_config,
                )
                res_runs.insert(0, "representation_id", representation_id)
                run_tables.append(res_runs)
                for col, val in res_summary.items():
                    ev.loc[row.Index, col] = val
                ev.loc[row.Index, "resample_evaluated"] = True
                ev.loc[row.Index, "resample_stability_gate"] = bool(
                    res_summary["resample_stability_gate"]
                )
                ev.loc[row.Index, "resample_convergence_gate"] = bool(
                    res_summary.get("resample_convergence_gate", True)
                )

            ev.loc[row.Index, "eligible"] = bool(
                bool(row.balance_gate)
                and bool(ev.loc[row.Index, "convergence_gate"])
                and seed_ari >= stability_min
                and bool(seed_convergence_gate)
                and bool(ev.loc[row.Index, "resample_stability_gate"])
                and bool(ev.loc[row.Index, "resample_convergence_gate"])
            )

            if (not full_stability) and bool(ev.loc[row.Index, "eligible"]):
                break

    # Candidate fallback never prefers a non-converged GMM while a converged
    # candidate (including KMeans) is available.
    pool = ev[ev["eligible"]].copy()
    fallback = "none"
    if pool.empty:
        pool = ev[
            ev["stability_evaluated"]
            & ev["balance_gate"]
            & ev["convergence_gate"]
            & ev["seed_convergence_gate"]
        ].copy()
        fallback = "no_fully_eligible_candidate_use_best_tested_converged"
    if pool.empty:
        pool = ev[ev["balance_gate"] & ev["convergence_gate"]].copy()
        fallback = "stability_unavailable_use_best_balanced_converged"
    if pool.empty:
        pool = ev[ev["convergence_gate"]].copy()
        fallback = "balance_unavailable_use_best_converged"
    if pool.empty:
        pool = ev.copy()
        fallback = "no_converged_candidate_available"

    best = float(pool["silhouette"].max())
    short = pool[pool["silhouette"] >= best - float(silhouette_tolerance)].copy()
    short["alg_pref"] = short["algorithm"].map({"KMeans": 0, "GMM": 1}).fillna(9)
    short = short.sort_values(
        ["k", "alg_pref", "silhouette"],
        ascending=[True, True, False],
    )
    selected = short.iloc[0]
    key = (str(selected.algorithm), int(selected.k))
    model = models[key]
    labels_dev = dev_labels_by_key[key]
    labels_all = model.predict(X_all).astype(int)

    ev["selected_within_representation"] = ev["algorithm"].eq(key[0]) & ev["k"].eq(key[1])
    selected_summary = {
        "representation_id": representation_id,
        "algorithm": key[0],
        "k": key[1],
        "silhouette": float(selected.silhouette),
        "stability_ari": float(selected.stability_ari)
        if pd.notna(selected.stability_ari)
        else np.nan,
        "seed_convergence_rate": float(selected.seed_convergence_rate)
        if pd.notna(selected.seed_convergence_rate)
        else np.nan,
        "resample_stability_ari_mean": float(selected.resample_stability_ari_mean)
        if pd.notna(selected.resample_stability_ari_mean)
        else np.nan,
        "resample_stability_ari_p10": float(selected.resample_stability_ari_p10)
        if pd.notna(selected.resample_stability_ari_p10)
        else np.nan,
        "resample_convergence_rate": float(selected.resample_convergence_rate)
        if pd.notna(selected.resample_convergence_rate)
        else np.nan,
        "gmm_converged": bool(selected.gmm_converged),
        "gmm_n_iter": int(selected.gmm_n_iter),
        "gmm_convergence_warning": bool(selected.gmm_convergence_warning),
        "convergence_gate": bool(selected.convergence_gate),
        "min_cluster_share": float(selected.min_cluster_share),
        "calinski_harabasz": float(selected.calinski_harabasz),
        "davies_bouldin": float(selected.davies_bouldin),
        "eligible": bool(selected.eligible),
        "fallback_mode": fallback,
    }
    return (
        ev,
        pd.concat(run_tables, ignore_index=True) if run_tables else pd.DataFrame(),
        pd.DataFrame(candidate_rows),
        selected_summary,
        model,
        labels_dev,
        labels_all,
    )


def pairwise_representation_ari(selected_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    ids = list(selected_labels)
    rows: list[dict[str, Any]] = []
    for a in ids:
        for b in ids:
            ari = float(adjusted_rand_score(selected_labels[a], selected_labels[b]))
            rows.append({"representation_a": a, "representation_b": b, "ari": ari})
    return pd.DataFrame(rows)


def choose_official_representation(
    summary: pd.DataFrame,
    pairwise_ari: pd.DataFrame,
    *,
    silhouette_tolerance: float = 0.02,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    df = summary.copy()
    other_mean = (
        pairwise_ari[pairwise_ari["representation_a"] != pairwise_ari["representation_b"]]
        .groupby("representation_a")["ari"]
        .mean()
        .rename("mean_cross_representation_ari")
    )
    df = df.merge(other_mean, left_on="representation_id", right_index=True, how="left")
    df["mean_cross_representation_ari"] = df["mean_cross_representation_ari"].fillna(1.0)

    pool = df[df["eligible"]].copy()
    fallback = "none"
    if pool.empty:
        pool = df.copy()
        fallback = "representation_eligibility_relaxed"

    best_sil = float(pool["silhouette"].max())
    pool["within_representation_tolerance"] = pool["silhouette"] >= best_sil - float(
        silhouette_tolerance
    )
    shortlist = pool[pool["within_representation_tolerance"]].copy()
    # Robustness first within a near-best separation band, then subsample
    # stability, then lower latent dimensionality / fewer raw inputs.
    shortlist = shortlist.sort_values(
        [
            "mean_cross_representation_ari",
            "resample_stability_ari_mean",
            "latent_dimensions",
            "raw_input_count",
        ],
        ascending=[False, False, True, True],
    )
    selected_id = str(shortlist.iloc[0]["representation_id"])
    df["selected_representation"] = df["representation_id"].eq(selected_id)
    df["within_representation_tolerance"] = df["silhouette"] >= best_sil - float(
        silhouette_tolerance
    )

    def dep(base: str, ablated: str, feature: str) -> dict[str, Any]:
        q = pairwise_ari[
            (pairwise_ari["representation_a"] == base)
            & (pairwise_ari["representation_b"] == ablated)
        ]
        ari = float(q.iloc[0]["ari"]) if len(q) else np.nan
        if np.isnan(ari):
            level = "unknown"
        elif ari >= 0.80:
            level = "low"
        elif ari >= 0.50:
            level = "moderate"
        else:
            level = "high"
        return {"feature": feature, "ari_vs_baseline": ari, "dependence_level": level}

    job_dep = dep("R0_GLOBAL_PCA", "R2_NO_JOB_CATEGORY", "job_category")
    exp_dep = dep("R0_GLOBAL_PCA", "R3_NO_YEARS_EXPERIENCE", "years_of_experience")
    joint_dep = dep(
        "R0_GLOBAL_PCA",
        "R4_NO_JOB_CATEGORY_NO_YEARS",
        "job_category + years_of_experience",
    )
    selected_row = df[df["selected_representation"]].iloc[0]

    obs = (
        f"Selected {selected_id}: {selected_row['algorithm']} K={int(selected_row['k'])}, "
        f"silhouette {float(selected_row['silhouette']):.3f}, mean subsample ARI "
        f"{float(selected_row['resample_stability_ari_mean']):.3f}, mean cross-representation ARI "
        f"{float(selected_row['mean_cross_representation_ari']):.3f}."
    )
    interp = (
        f"Ablation dependence is {job_dep['dependence_level']} for job_category "
        f"(R0↔R2 ARI={job_dep['ari_vs_baseline']:.3f}) and {exp_dep['dependence_level']} for "
        f"years_of_experience (R0↔R3 ARI={exp_dep['ari_vs_baseline']:.3f}), and "
        f"{joint_dep['dependence_level']} when both are removed "
        f"(R0↔R4 ARI={joint_dep['ari_vs_baseline']:.3f}). "
        "The official representation is chosen only among near-best separation candidates, "
        "with robustness and stability used before parsimony."
    )
    action = (
        "Use the selected representation's input feature contract for the official clustering model. "
        "If an ablated representation is selected, keep the excluded dominant feature out of Branch A "
        "until a later data refresh demonstrates stable incremental structural value."
    )
    insight = {
        "selected_representation": selected_id,
        "fallback_mode": fallback,
        "silhouette_tolerance": float(silhouette_tolerance),
        "job_category_dependence": job_dep,
        "years_of_experience_dependence": exp_dep,
        "joint_dominant_feature_dependence": joint_dep,
        "observed": obs,
        "interpretation": interp,
        "action": action,
    }
    return selected_id, df, insight


def transform_representation(transformer: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    kind = transformer["kind"]
    encoder = transformer["encoder"]
    if kind == "global_pca":
        blocks = encoder.transform_family_blocks(X, balanced=False)
        all_names = list(map(str, encoder.get_feature_names_out()))
        patterns = list(transformer.get("excluded_encoded_substrings", []))
        parts: list[np.ndarray] = []
        for fam, arr in blocks.items():
            arr = np.asarray(arr, dtype=float)
            prefix = f"{fam}::"
            fam_names = [n for n in all_names if n.startswith(prefix)]
            if len(fam_names) != arr.shape[1]:
                fam_names = [f"{fam}::feature_{i}" for i in range(arr.shape[1])]
            keep_f = np.ones(len(fam_names), dtype=bool)
            for pattern in patterns:
                keep_f &= np.array([pattern not in str(x) for x in fam_names], dtype=bool)
            if not keep_f.any():
                continue
            arr = arr[:, keep_f]
            arr = arr / math.sqrt(max(1, arr.shape[1]))
            parts.append(arr)
        z = np.hstack(parts)
        p = transformer["pca"].transform(z)
        return p[:, : int(transformer["n_components"])]
    if kind == "familywise_pca":
        blocks = encoder.transform_family_blocks(X, balanced=False)
        out: list[np.ndarray] = []
        for fam, pca in transformer["family_pcas"].items():
            n = int(transformer["family_component_counts"][fam])
            w = float(transformer["family_latent_weights"][fam])
            out.append(pca.transform(np.asarray(blocks[fam], dtype=float))[:, :n] * w)
        return np.hstack(out)
    raise ValueError(f"Unsupported representation transformer kind: {kind}")
