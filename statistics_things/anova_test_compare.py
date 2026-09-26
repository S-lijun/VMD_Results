"""
anova_test.py workflow as a reusable API (same usage style as anova_tukey_compare).

1) Shapiro-Wilk on within-user pairwise EER differences
2) If ANY difference is non-normal:
      Friedman (omnibus) + pairwise Wilcoxon + Holm
   Else:
      RM-ANOVA (omnibus) + pairwise paired t + Holm
3) Summarize which representations differ from ALL others
4) Optional pairwise p-value map (Holm-adjusted)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, ttest_rel, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from statistics_things.anova_tukey_compare import (
    build_wide_eer_table,
    plot_pairwise_map,
)


def _strip_per_user(name: str) -> str:
    return name.removeprefix("per-user")


def collect_per_user_reps(
    base: str | Path,
    tag: str | None = None,
) -> dict[str, Path]:
    """
    Collect all per-user* dirs under base.
    If tag is set, labels become "{tag}:{rep_name}" e.g. "LA:XYPlot".
    """
    base = Path(base)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    reps: dict[str, Path] = {}
    for d in sorted(base.glob("per-user*")):
        if not d.is_dir():
            continue
        name = _strip_per_user(d.name)
        label = f"{tag}:{name}" if tag else name
        if label in reps:
            raise ValueError(f"Duplicate representation label: {label}")
        reps[label] = d

    if not reps:
        raise ValueError(f"No per-user* directories under {base}")
    return reps


def collect_location_aware_free_reps(
    location_aware_base: str | Path,
    location_free_base: str | Path,
    aware_tag: str = "LA",
    free_tag: str = "LF",
) -> dict[str, Path]:
    """
    Merge per-user reps from two roots into one labeled dict.

    Results_uc/...  -> location-aware  (default tag LA)
    Results/...     -> location-free   (default tag LF)
    """
    aware = collect_per_user_reps(location_aware_base, tag=aware_tag)
    free = collect_per_user_reps(location_free_base, tag=free_tag)
    merged = {**aware, **free}
    if len(merged) != len(aware) + len(free):
        raise ValueError("Label collision between LA and LF representations")
    return merged


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    reps = [c for c in wide.columns if c != "User"]
    return wide.melt(
        id_vars="User",
        value_vars=reps,
        var_name="Representation",
        value_name="EER",
    )


def _shapiro_pairwise_differences(
    wide: pd.DataFrame,
    labels: list[str],
    normality_alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for a, b in combinations(labels, 2):
        diffs = (wide[a] - wide[b]).to_numpy(dtype=float)
        # Shapiro needs n>=3; tiny samples fall back to "not normal"
        if len(diffs) < 3 or np.allclose(diffs, diffs[0]):
            W, p = float("nan"), 0.0
            normal = False
        else:
            W, p = shapiro(diffs)
            W, p = float(W), float(p)
            normal = bool(p >= normality_alpha)
        rows.append(
            {
                "Comparison": f"{a} - {b}",
                "W": W,
                "p": p,
                "Normal": normal,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_to_matrices(
    pairwise: pd.DataFrame,
    labels: list[str],
    p_col: str = "p_corr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_mat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    sig_mat = pd.DataFrame(False, index=labels, columns=labels, dtype=bool)
    for _, r in pairwise.iterrows():
        a, b = str(r["A"]), str(r["B"])
        p = float(r[p_col])
        sig = bool(r["Significant"])
        p_mat.loc[a, b] = p
        p_mat.loc[b, a] = p
        sig_mat.loc[a, b] = sig
        sig_mat.loc[b, a] = sig
    return p_mat, sig_mat


def _per_rep_summary(
    pairwise: pd.DataFrame,
    labels: list[str],
    means: pd.Series,
) -> pd.DataFrame:
    rows = []
    n_others = len(labels) - 1
    for focal in labels:
        comps = pairwise[(pairwise["A"] == focal) | (pairwise["B"] == focal)]
        n_sig = int(comps["Significant"].sum())
        fail = []
        for _, r in comps.iterrows():
            other = r["B"] if r["A"] == focal else r["A"]
            if not bool(r["Significant"]):
                fail.append(str(other))
        rows.append(
            {
                "Representation": focal,
                "Mean EER": round(float(means[focal]), 4),
                "Significant_vs": f"{n_sig}/{n_others}",
                "All_significant": n_sig == n_others,
                "Fail_vs": ", ".join(fail) or "-",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["All_significant", "Mean EER"], ascending=[False, True])
        .reset_index(drop=True)
    )


def anova_test_compare(
    representations: dict[str, str | Path] | list[str | Path] | None = None,
    n: int = 1,
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    eer_as_percent: bool = True,
    plot: bool = True,
    verbose: bool = True,
    base: str | Path | None = None,
    location_aware_base: str | Path | None = None,
    location_free_base: str | Path | None = None,
    aware_tag: str = "LA",
    free_tag: str = "LF",
) -> dict[str, Any]:
    """
    Run the anova_test.py pipeline on per-user representation JSONs.

    Usage A — one folder link (all per-user* under it, typically 6):
        anova_test_compare(base=r"Results_uc/TWOS/ViT", n=1)

    Usage B — two folder links (LA + LF, typically 12):
        anova_test_compare(
            location_aware_base=r"Results_uc/TWOS/ViT",
            location_free_base=r"Results/TWOS/ViT",
            n=1,
        )

    Usage C — explicit dict/list of paths:
        anova_test_compare(reps, n=1)
    """
    if base is not None:
        if location_aware_base is not None or location_free_base is not None:
            raise ValueError("Use either base=... OR the two LA/LF bases, not both")
        representations = collect_per_user_reps(base, tag=None)
        if verbose:
            print(f"Loaded {len(representations)} representations from {base}")
            print(" ", list(representations.keys()))

    elif location_aware_base is not None or location_free_base is not None:
        if location_aware_base is None or location_free_base is None:
            raise ValueError(
                "Provide BOTH location_aware_base (Results_uc) "
                "and location_free_base (Results)"
            )
        representations = collect_location_aware_free_reps(
            location_aware_base,
            location_free_base,
            aware_tag=aware_tag,
            free_tag=free_tag,
        )
        if verbose:
            la = [k for k in representations if k.startswith(f"{aware_tag}:")]
            lf = [k for k in representations if k.startswith(f"{free_tag}:")]
            print(
                f"Loaded {len(representations)} representations "
                f"({aware_tag}={len(la)}, {free_tag}={len(lf)})"
            )
            print(f"  {aware_tag}:", la)
            print(f"  {free_tag}:", lf)

    if representations is None:
        raise ValueError(
            "Pass base=... OR location_aware_base+location_free_base "
            "OR representations=..."
        )

    wide = build_wide_eer_table(
        representations,
        n=n,
        eer_as_percent=eer_as_percent,
        verbose=verbose,
    )
    labels = [c for c in wide.columns if c != "User"]
    long_df = _wide_to_long(wide)
    means = wide[labels].mean()

    normality_df = _shapiro_pairwise_differences(
        wide, labels, normality_alpha=normality_alpha
    )
    all_normal = bool(normality_df["Normal"].all())

    if verbose:
        unit = "%" if eer_as_percent else ""
        print(f"n={n} | users={wide.shape[0]} | alpha={alpha}")
        print("Mean EER:")
        for lab in labels:
            print(f"  {lab}: {means[lab]:.4f}{unit}")

        print("\n=== Shapiro-Wilk (pairwise within-user differences) ===")
        for _, r in normality_df.iterrows():
            print(
                f"  {r['Comparison']}: W={r['W']:.3f}, p={r['p']:.4g}, "
                f"{'NORMAL' if r['Normal'] else 'NOT NORMAL'}"
            )
        print(
            "\nOverall normality: "
            + (
                "all pairwise differences approximately normal "
                "-> RM-ANOVA + paired t + Holm"
                if all_normal
                else "at least one non-normal "
                "-> Friedman + Wilcoxon + Holm"
            )
        )

    result: dict[str, Any] = {
        "n": n,
        "alpha": alpha,
        "n_users": int(wide.shape[0]),
        "labels": labels,
        "wide": wide,
        "long": long_df,
        "means": means,
        "normality_df": normality_df,
        "all_normal": all_normal,
        "method": None,
        "omnibus_name": None,
        "omnibus_stat": None,
        "omnibus_p": None,
        "omnibus_significant": None,
        "omnibus_table": None,
        "pairwise_df": None,
        "p_matrix": None,
        "sig_matrix": None,
        "per_rep_summary": None,
        "figure": None,
    }

    # -------------------- nonparametric branch --------------------
    if not all_normal:
        method = "friedman_wilcoxon_holm"
        stat, omnibus_p = friedmanchisquare(*[wide[lab].to_numpy() for lab in labels])
        omnibus_p = float(omnibus_p)
        omnibus_sig = bool(omnibus_p <= alpha)

        if verbose:
            print("\n=== Friedman test ===")
            print(
                f"Friedman (df={len(labels) - 1}) = {float(stat):.3f}, "
                f"p = {omnibus_p:.6g} -> significant = {omnibus_sig}"
            )

        rows = []
        for a, b in combinations(labels, 2):
            # zero-differences can occur; zero_method="wilcox" is scipy default
            W, p_unc = wilcoxon(
                wide[a].to_numpy(),
                wide[b].to_numpy(),
                alternative="two-sided",
            )
            rows.append({"A": a, "B": b, "stat": float(W), "p_unc": float(p_unc)})
        pairwise = pd.DataFrame(rows)
        reject, p_corr, _, _ = multipletests(
            pairwise["p_unc"], alpha=alpha, method="holm"
        )
        pairwise["p_corr"] = p_corr
        pairwise["Significant"] = reject
        pairwise = pairwise.rename(columns={"stat": "W"})

        if verbose:
            print("\n=== Pairwise Wilcoxon (Holm corrected) ===")
            for _, r in pairwise.iterrows():
                print(
                    f"  {r['A']} vs {r['B']}: "
                    f"W={r['W']:.1f}, p={r['p_unc']:.4g}, "
                    f"Holm p={r['p_corr']:.4g}, "
                    f"{'YES' if r['Significant'] else 'ns'}"
                )

        result.update(
            {
                "method": method,
                "omnibus_name": "Friedman",
                "omnibus_stat": float(stat),
                "omnibus_p": omnibus_p,
                "omnibus_significant": omnibus_sig,
            }
        )

    # -------------------- parametric branch --------------------
    else:
        method = "rm_anova_ttest_holm"
        anova = AnovaRM(
            data=long_df,
            depvar="EER",
            subject="User",
            within=["Representation"],
        ).fit()
        anova_table = anova.anova_table.copy()
        omnibus_p = float(anova_table.loc["Representation", "Pr > F"])
        omnibus_stat = float(anova_table.loc["Representation", "F Value"])
        omnibus_sig = bool(omnibus_p <= alpha)

        if verbose:
            print("\n=== Repeated-measures ANOVA ===")
            print(anova_table.to_string())
            print(
                f"\nANOVA p = {omnibus_p:.6g} -> significant = {omnibus_sig}"
            )

        rows = []
        for a, b in combinations(labels, 2):
            t_stat, p_unc = ttest_rel(wide[a].to_numpy(), wide[b].to_numpy())
            rows.append(
                {"A": a, "B": b, "stat": float(t_stat), "p_unc": float(p_unc)}
            )
        pairwise = pd.DataFrame(rows)
        reject, p_corr, _, _ = multipletests(
            pairwise["p_unc"], alpha=alpha, method="holm"
        )
        pairwise["p_corr"] = p_corr
        pairwise["Significant"] = reject
        pairwise = pairwise.rename(columns={"stat": "t"})

        if verbose:
            print("\n=== Pairwise paired t-tests (Holm corrected) ===")
            for _, r in pairwise.iterrows():
                print(
                    f"  {r['A']} vs {r['B']}: "
                    f"t={r['t']:.3f}, p={r['p_unc']:.4g}, "
                    f"Holm p={r['p_corr']:.4g}, "
                    f"{'YES' if r['Significant'] else 'ns'}"
                )

        result.update(
            {
                "method": method,
                "omnibus_name": "RM-ANOVA",
                "omnibus_stat": omnibus_stat,
                "omnibus_p": omnibus_p,
                "omnibus_significant": omnibus_sig,
                "omnibus_table": anova_table,
            }
        )

    p_mat, sig_mat = _pairwise_to_matrices(pairwise, labels, p_col="p_corr")
    per_rep = _per_rep_summary(pairwise, labels, means)

    result["pairwise_df"] = pairwise
    result["p_matrix"] = p_mat
    result["sig_matrix"] = sig_mat
    result["per_rep_summary"] = per_rep

    if verbose:
        print("\n=== Significant vs ALL other representations? ===")
        print(per_rep.to_string(index=False))

    if plot:
        title = (
            f"{result['omnibus_name']} post-hoc Holm map | n={n} | alpha={alpha}"
        )
        fig, _ = plot_pairwise_map(
            p_mat,
            sig_mat=sig_mat,
            alpha=alpha,
            title=title,
            cbar_label="Holm-adjusted p",
            p_digits=5,
        )
        # colorbar label is Tukey by default in helper; leave as generic enough
        result["figure"] = fig
        if verbose:
            import matplotlib.pyplot as plt

            plt.show()

    return result
