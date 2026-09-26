"""
RM-ANOVA + Tukey's HSD for per-user representation EER curves.

Email workflow:
  1) Repeated-measures ANOVA across representations (same users).
  2) If not significant -> stop.
  3) If significant -> Tukey's HSD on all pairs, print per-rep alphas,
     and draw a pairwise p-value map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statistics_things.wilcoxon_curve_compare import (
    BALABIT_INDEX_TO_USER,
    align_many_eer_dicts,
    collect_eer_by_n,
    load_json,
    _is_balabit_enum_keys,
)


def _label_from_path(path: str | Path) -> str:
    p = Path(path)
    name = p.name if p.is_dir() else p.parent.name
    return name.removeprefix("per-user")


def _normalize_reps(
    representations: dict[str, str | Path] | list[str | Path],
) -> dict[str, Path]:
    if isinstance(representations, list):
        reps = {_label_from_path(p): Path(p) for p in representations}
    else:
        reps = {str(k): Path(v) for k, v in representations.items()}
    if len(reps) < 2:
        raise ValueError("Need at least 2 representations")
    return reps


def build_wide_eer_table(
    representations: dict[str, str | Path] | list[str | Path],
    n: int = 1,
    eer_as_percent: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load per-user JSONs at fusion size n and return a wide table:
    columns = User + one column per representation (aligned common users).

    Balabit files that use indices 0..9 are remapped with:
      0->user7, 1->user9, ..., 9->user35
    so they align with files that already use "user7"/"user9"/...
    """
    reps = _normalize_reps(representations)
    raw: dict[str, dict[str, float]] = {}
    remapped_labels = []
    for lab, path in reps.items():
        data = load_json(path)
        eer = collect_eer_by_n(data, n)
        if not eer:
            raise ValueError(f"No EER found for n={n} in {path}")
        raw[lab] = eer
        if _is_balabit_enum_keys(eer.keys()):
            remapped_labels.append(lab)

    users, normed = align_many_eer_dicts(raw)
    if len(users) < 2:
        raise ValueError(
            f"Need >=2 common users across representations; got {len(users)}"
        )

    if verbose and remapped_labels:
        print("[Info] Balabit numeric index -> user mapping applied for:")
        print("      ", ", ".join(remapped_labels))
        print("[Info] mapping (0->user7, ..., 9->user35):")
        for idx, user in sorted(BALABIT_INDEX_TO_USER.items(), key=lambda x: int(x[0])):
            print(f"  {idx} -> {user}")
        print(f"[Info] paired common users ({len(users)}): {users}")

    scale = 100.0 if eer_as_percent else 1.0
    rows = []
    for u in users:
        row = {"User": u}
        for lab in reps:
            row[lab] = normed[lab][u] * scale
        rows.append(row)
    return pd.DataFrame(rows)


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    reps = [c for c in wide.columns if c != "User"]
    return wide.melt(
        id_vars="User",
        value_vars=reps,
        var_name="Representation",
        value_name="EER",
    )


def _tukey_to_matrices(
    tukey_df: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build symmetric p / reject / mean-diff matrices from Tukey rows."""
    p_mat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    sig_mat = pd.DataFrame(False, index=labels, columns=labels, dtype=bool)
    diff_mat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    for _, r in tukey_df.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        p = float(r["p-adj"])
        reject = bool(r["reject"])
        meandiff = float(r["meandiff"])
        p_mat.loc[a, b] = p
        p_mat.loc[b, a] = p
        sig_mat.loc[a, b] = reject
        sig_mat.loc[b, a] = reject
        # meandiff is group2 - group1 in statsmodels; store A->B as B-A style
        diff_mat.loc[a, b] = meandiff
        diff_mat.loc[b, a] = -meandiff

    for lab in labels:
        p_mat.loc[lab, lab] = np.nan
        sig_mat.loc[lab, lab] = False
        diff_mat.loc[lab, lab] = 0.0

    return p_mat, sig_mat, diff_mat


def _per_rep_alpha_table(
    p_mat: pd.DataFrame,
    sig_mat: pd.DataFrame,
    means: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    labels = list(p_mat.index)
    rows = []
    for focal in labels:
        others = [o for o in labels if o != focal]
        ps = [p_mat.loc[focal, o] for o in others]
        n_sig = int(sum(bool(sig_mat.loc[focal, o]) for o in others))
        rows.append(
            {
                "Representation": focal,
                "Mean EER": round(float(means[focal]), 4),
                "Significant_vs": f"{n_sig}/{len(others)}",
                "All_significant": n_sig == len(others),
                "Min p vs others": round(float(np.nanmin(ps)), 6),
                "Max p vs others": round(float(np.nanmax(ps)), 6),
                "Fail_vs": ", ".join(
                    o for o in others if not bool(sig_mat.loc[focal, o])
                )
                or "-",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["All_significant", "Mean EER"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _tukey_result_to_dataframe(tukey, alpha: float = 0.05) -> pd.DataFrame:
    """
    Normalize Tukey HSD output across statsmodels versions.

    Older versions use group1/group2; 0.15+ summary_frame uses group_t/group_c.
    Always return columns: group1, group2, meandiff, p-adj, lower, upper, reject.
    """
    df = None
    if hasattr(tukey, "summary_frame"):
        df = tukey.summary_frame().reset_index(drop=True).copy()
        rename = {}
        if "group_t" in df.columns and "group1" not in df.columns:
            # statsmodels 0.15: meandiff ~= group_t - group_c
            rename["group_c"] = "group1"
            rename["group_t"] = "group2"
        if "p_adj" in df.columns and "p-adj" not in df.columns:
            rename["p_adj"] = "p-adj"
        df = df.rename(columns=rename)

    if df is None or "group1" not in df.columns:
        raw = tukey._results_table.data
        df = pd.DataFrame(raw[1:], columns=list(raw[0])).copy()

    df = df.rename(columns={"p_adj": "p-adj"})
    required = ["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Unexpected Tukey result columns {list(df.columns)}; missing {missing}"
        )

    out = df[required].copy()
    out["group1"] = out["group1"].astype(str)
    out["group2"] = out["group2"].astype(str)
    for col in ("meandiff", "p-adj", "lower", "upper"):
        out[col] = out[col].astype(float)
    out["reject"] = out["reject"].map(
        lambda x: bool(x)
        if isinstance(x, (bool, np.bool_))
        else str(x).strip().lower() in {"true", "t", "1", "yes"}
    )
    # Safety: keep reject consistent with alpha if needed
    out.loc[out["p-adj"] <= alpha, "reject"] = True
    out.loc[out["p-adj"] > alpha, "reject"] = False
    return out.reset_index(drop=True)


def _format_pvalue(p: float, digits: int = 5) -> str:
    """Format p for heatmap cells; use sci-notation when tiny."""
    if p != p:  # NaN
        return ""
    if p < 10 ** (-digits):
        return f"{p:.2e}"
    return f"{p:.{digits}f}"


def plot_pairwise_map(
    p_mat: pd.DataFrame,
    sig_mat: pd.DataFrame | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    ax: plt.Axes | None = None,
    cbar_label: str = "p-adj",
    p_digits: int = 5,
):
    """
    Heatmap of pairwise p-values. Both axes = representations.
    Significant cells (p <= alpha) get a * annotation.
    """
    labels = list(p_mat.index)
    data = p_mat.to_numpy(dtype=float)
    display = np.array(data, copy=True)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6.5, 0.9 * len(labels) + 2), max(5.5, 0.9 * len(labels) + 2))
        )
    else:
        fig = ax.figure

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=max(alpha * 4, 0.2))

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Representation")
    ax.set_ylabel("Representation")
    ax.set_title(title or f"Pairwise p-values (alpha={alpha})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#666", fontsize=9)
                continue
            p = display[i, j]
            if np.isnan(p):
                continue
            sig = bool(sig_mat.iloc[i, j]) if sig_mat is not None else (p <= alpha)
            txt = _format_pvalue(p, digits=p_digits) + ("*" if sig else "")
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="black" if p > alpha else "white",
                fontsize=6.5 if len(labels) >= 10 else 7.5,
                fontweight="bold" if sig else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax


def anova_tukey_compare(
    representations: dict[str, str | Path] | list[str | Path],
    n: int = 1,
    alpha: float = 0.05,
    eer_as_percent: bool = True,
    plot: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Email-style pipeline: RM-ANOVA, then Tukey's HSD only if ANOVA is significant.

    Parameters
    ----------
    representations :
        {label: per-user json/dir} or list of paths (labels from folder names).
    n :
        Fusion size / score-count key in per-user JSON.
    alpha :
        Significance level for ANOVA gate and Tukey (default 0.05).
    eer_as_percent :
        If True, analyze EER * 100.
    plot :
        If True and ANOVA significant, draw pairwise p-value map.
    verbose :
        Print summary to stdout.

    Returns
    -------
    dict with keys:
      wide, long, means, anova_table, anova_p, anova_significant,
      tukey_df, p_matrix, sig_matrix, diff_matrix, per_rep_summary, figure
    """
    wide = build_wide_eer_table(
        representations,
        n=n,
        eer_as_percent=eer_as_percent,
        verbose=verbose,
    )
    labels = [c for c in wide.columns if c != "User"]
    long_df = _wide_to_long(wide)
    means = wide[labels].mean()

    anova = AnovaRM(
        data=long_df,
        depvar="EER",
        subject="User",
        within=["Representation"],
    ).fit()
    anova_table = anova.anova_table.copy()
    anova_p = float(anova_table.loc["Representation", "Pr > F"])
    anova_significant = bool(anova_p <= alpha)

    result: dict[str, Any] = {
        "n": n,
        "alpha": alpha,
        "n_users": int(wide.shape[0]),
        "labels": labels,
        "wide": wide,
        "long": long_df,
        "means": means,
        "anova_table": anova_table,
        "anova_p": anova_p,
        "anova_significant": anova_significant,
        "tukey_df": None,
        "p_matrix": None,
        "sig_matrix": None,
        "diff_matrix": None,
        "per_rep_summary": None,
        "figure": None,
    }

    if verbose:
        unit = "%" if eer_as_percent else ""
        print(f"n={n} | users={result['n_users']} | alpha={alpha}")
        print("Mean EER:")
        for lab in labels:
            print(f"  {lab}: {means[lab]:.4f}{unit}")
        print("\n=== Repeated-measures ANOVA ===")
        print(anova_table.to_string())
        print(f"\nANOVA p = {anova_p:.6g}  ->  significant = {anova_significant}")

    if not anova_significant:
        if verbose:
            print(
                "\nANOVA not significant: insufficient evidence that the "
                "representations differ. Skipping Tukey's HSD."
            )
        return result

    tukey = pairwise_tukeyhsd(
        endog=long_df["EER"].to_numpy(),
        groups=long_df["Representation"].to_numpy(),
        alpha=alpha,
    )
    tukey_df = _tukey_result_to_dataframe(tukey, alpha=alpha)
    p_mat, sig_mat, diff_mat = _tukey_to_matrices(tukey_df, labels)
    per_rep = _per_rep_alpha_table(p_mat, sig_mat, means, alpha)

    result["tukey_df"] = tukey_df
    result["p_matrix"] = p_mat
    result["sig_matrix"] = sig_mat
    result["diff_matrix"] = diff_mat
    result["per_rep_summary"] = per_rep

    if verbose:
        print("\n=== Tukey's HSD (all pairs) ===")
        show = tukey_df.copy()
        if "p-adj" in show.columns:
            show["p-adj"] = show["p-adj"].map(lambda x: f"{float(x):.6g}")
        print(show.to_string(index=False))

        print("\n=== Per representation vs all others (Tukey p-adj) ===")
        for focal in labels:
            print(f"\n{focal}  (mean={means[focal]:.4f}):")
            for other in labels:
                if other == focal:
                    continue
                p = p_mat.loc[focal, other]
                sig = bool(sig_mat.loc[focal, other])
                print(
                    f"  vs {other}: p-adj={p:.6g}  "
                    f"{'SIGNIFICANT' if sig else 'ns'} (alpha={alpha})"
                )

        print("\n=== Summary: significant vs remaining? ===")
        print(per_rep.to_string(index=False))

    if plot:
        fig, _ = plot_pairwise_map(
            p_mat,
            sig_mat=sig_mat,
            alpha=alpha,
            title=f"Tukey HSD pairwise map | n={n} | alpha={alpha}",
            cbar_label="p-adj (Tukey HSD)",
        )
        result["figure"] = fig
        if verbose:
            plt.show()

    return result
