import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter


# ============================================================
# Publication-style defaults
# ============================================================
_PUB_COLORS = [
    "#2F6F4E",  # forest green
    "#C45C26",  # burnt orange
    "#2F5D8C",  # steel blue
    "#6B4C9A",  # muted violet
    "#B33A3A",  # soft crimson
    "#5A7A3A",  # olive
]
_PUB_LINESTYLES = ["-", "--", "-.", "-", "--", "-."]
_PUB_MARKERS = ["o", "s", "^", "D", "P", "X"]
_STYLE = {
    "fig_facecolor": "#FFFFFF",
    "ax_facecolor": "#FAFAF8",
    "spine": "#4A4A4A",
    "tick": "#3D3D3D",
    "label": "#2C2C2C",
    "title": "#1F1F1F",
    "grid": "#D0D0D0",
    "legend_edge": "#E2E2E2",
    "legend_face": "#FFFFFF",
}


def generate_vivid_colors(n):
    """Return n distinct vivid hex colors (deterministic)."""
    colors = []
    for i in range(int(n)):
        h = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.85)
        colors.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return colors


def _assign_distinct_colors(n_dirs, color_list=None):
    """Return n_dirs colors. color_list is used exactly if provided."""
    if n_dirs <= 0:
        return []
    if color_list is not None:
        if len(color_list) != n_dirs:
            raise ValueError(
                f"color_list length {len(color_list)} != number of curves {n_dirs}"
            )
        return list(color_list)
    palette = list(_PUB_COLORS)
    if len(palette) < n_dirs:
        for c in generate_vivid_colors(max(n_dirs * 3, 8)):
            if c not in palette:
                palette.append(c)
            if len(palette) >= n_dirs:
                break
    return (palette * ((n_dirs // max(len(palette), 1)) + 1))[:n_dirs]

def _parse_result_json(data):
    """Return n array, mean EER, and optional per-n user EER lists."""
    if isinstance(data, dict) and ("n" in data and "avg_eer" in data):
        n = np.array(data["n"], dtype=int)
        avg_eer = np.array(data["avg_eer"], dtype=float)
        return n, avg_eer, None

    bucket = {}
    for user_block in data.values() if isinstance(data, dict) else []:
        if not isinstance(user_block, dict):
            continue
        for rec in user_block.values():
            if not isinstance(rec, dict):
                continue
            n_val = rec.get("n")
            eer_val = rec.get("EER", rec.get("avg_eer"))
            if n_val is None or eer_val is None:
                continue
            bucket.setdefault(int(n_val), []).append(float(eer_val))

    if not bucket:
        raise ValueError("Unsupported JSON structure for plotting")

    sorted_n = sorted(bucket.keys())
    n = np.array(sorted_n, dtype=int)
    avg_eer = np.array([np.mean(bucket[k]) for k in sorted_n])
    return n, avg_eer, bucket


def _nearest_n_for_event(target_event, n_values, seq_length, shift):
    events = n_values * seq_length + shift
    idx = int(np.argmin(np.abs(events - target_event)))
    return int(n_values[idx])


def _drop_max_eer_outlier(values):
    """Remove one maximum EER value (for box-plot outlier trimming)."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return values
    return np.delete(values, int(np.argmax(values)))


def _draw_box_at(ax, x, values, color, offset=0, box_width=22, alpha=0.22):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None

    q1, med, q3 = np.percentile(values, [25, 50, 75])
    vmin, vmax = values.min(), values.max()
    cx = x + offset
    half = box_width / 2

    rect = Rectangle(
        (cx - half, q1),
        box_width,
        q3 - q1,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.35,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.plot([cx - half, cx + half], [med, med], color=color, linewidth=1.8, solid_capstyle="round", zorder=4)
    ax.plot([cx, cx], [vmin, q1], color=color, linewidth=1.15, solid_capstyle="round", zorder=4, alpha=0.9)
    ax.plot([cx, cx], [q3, vmax], color=color, linewidth=1.15, solid_capstyle="round", zorder=4, alpha=0.9)

    cap = box_width * 0.28
    ax.plot([cx - cap, cx + cap], [vmin, vmin], color=color, linewidth=1.15, solid_capstyle="round", zorder=4, alpha=0.9)
    ax.plot([cx - cap, cx + cap], [vmax, vmax], color=color, linewidth=1.15, solid_capstyle="round", zorder=4, alpha=0.9)

    return float(np.mean(values)), float(np.std(values))


def _style_publication_axes(ax, fig):
    """Visual-only axis polish; does not change data/logic."""
    fig.patch.set_facecolor(_STYLE["fig_facecolor"])
    ax.set_facecolor(_STYLE["ax_facecolor"])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_STYLE["spine"])
        ax.spines[side].set_linewidth(1.15)

    ax.tick_params(
        axis="both",
        which="major",
        colors=_STYLE["tick"],
        labelsize=10,
        length=4.5,
        width=1.0,
        direction="out",
    )
    ax.grid(
        which="major",
        axis="both",
        linestyle=(0, (3, 3)),
        color=_STYLE["grid"],
        alpha=0.55,
        linewidth=0.85,
        zorder=0,
    )
    ax.set_axisbelow(True)


# ============================================================
# Publication-style EER comparison plot
# ============================================================
def plot_eer_publication_style(
    dir_list,
    max_event=650,
    seq_length=120,
    title=None,
    legend_names=None,
    shift_list=None,
    per_user_dir_list=None,
    box_event_positions=(300, 450, 600),
    linestyle_list=None,
    marker_list=None,
    color_list=None,
    show_boxplots=True,
    show_errorbars=True,
    drop_max_eer_outlier=False,
    panel_label=None,
    dataset_name="Balabit",
    figsize=(7.8, 5.2),
    plot_n_set={1, 5, 10, 15},
    x_axis_start_event=0,
):
    """
    Publication-style Mean EER vs Number of Events plot.

    Supports dense mean curves, per-user box plots, and error bars
    at selected event positions (similar to paper figures).

    If drop_max_eer_outlier is True, each box plot drops its largest
    per-user EER before drawing (and before error-bar statistics).

    x_axis_start_event controls x-axis left bound after applying shifts.
    """
    dir_list = [Path(d) for d in dir_list]
    n_dirs = len(dir_list)

    if legend_names is not None and len(legend_names) != n_dirs:
        raise ValueError("legend_names must match number of directories")

    if shift_list is None:
        shift_list = [0] * n_dirs
    if len(shift_list) != n_dirs:
        raise ValueError("shift_list must match number of directories")

    if per_user_dir_list is None:
        per_user_dir_list = dir_list
    else:
        if len(per_user_dir_list) != n_dirs:
            raise ValueError("per_user_dir_list must match number of directories")
        per_user_dir_list = [
            Path(d) if d is not None else None
            for d in per_user_dir_list
        ]

    color_list = _assign_distinct_colors(n_dirs, color_list=color_list)
    if linestyle_list is None:
        linestyle_list = ["-"] * n_dirs
    else:
        linestyle_list = list(linestyle_list)[:n_dirs]
        if len(linestyle_list) < n_dirs:
            linestyle_list = linestyle_list + ["-"] * (n_dirs - len(linestyle_list))
    marker_list = (marker_list or _PUB_MARKERS[:n_dirs])[:n_dirs]
    if len(marker_list) < n_dirs:
        marker_list = marker_list + ["o"] * (n_dirs - len(marker_list))

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    _style_publication_axes(ax, fig)
    has_any_curve = False
    all_eer_max = 0.0
    series_meta = []

    for dir_idx, result_dir in enumerate(dir_list):
        color = color_list[dir_idx]
        linestyle = linestyle_list[dir_idx]
        marker = marker_list[dir_idx]
        shift = shift_list[dir_idx]
        legend_label = legend_names[dir_idx] if legend_names else result_dir.name

        per_user_dir = per_user_dir_list[dir_idx]
        first_plot = True

        for json_file in result_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                n, avg_eer, user_bucket = _parse_result_json(data)

                if per_user_dir is not None:
                    per_user_file = per_user_dir / json_file.name
                    if per_user_file.exists() and per_user_file != json_file:
                        with open(per_user_file, "r") as f:
                            _, _, user_bucket = _parse_result_json(json.load(f))

                if plot_n_set is not None:
                    mask = np.isin(n, list(plot_n_set))
                    n = n[mask]
                    avg_eer = avg_eer[mask]
                    if user_bucket is not None:
                        user_bucket = {k: v for k, v in user_bucket.items() if k in set(n.tolist())}

                events = n * seq_length + shift
                mask = events <= max_event
                events = events[mask]
                avg_eer = avg_eer[mask]
                n = n[mask]

                if len(events) == 0:
                    continue

                ax.plot(
                    events,
                    avg_eer,
                    linestyle=linestyle,
                    linewidth=2.35,
                    marker=marker,
                    markersize=5.6,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.55,
                    color=color,
                    alpha=0.96,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=legend_label if first_plot else None,
                    zorder=5,
                )

                has_any_curve = True
                all_eer_max = max(all_eer_max, float(np.nanmax(avg_eer)))
                series_meta.append({
                    "color": color,
                    "n": n,
                    "events": events,
                    "avg_eer": avg_eer,
                    "user_bucket": user_bucket,
                    "shift": shift,
                })
                first_plot = False

            except Exception as e:
                print(f"[ERROR] {json_file}: {e}")

    if not has_any_curve:
        raise ValueError("No valid curves were plotted. Check directory path and JSON format.")

    if show_boxplots or show_errorbars:
        n_series = len(series_meta)
        box_width = 16
        box_gap = 10
        if n_series > 1:
            step = box_width + box_gap
            total_span = step * (n_series - 1)
            box_offsets = np.linspace(-total_span / 2, total_span / 2, n_series)
        else:
            box_offsets = np.array([0.0])

        for box_event in box_event_positions:
            if box_event > max_event:
                continue

            for s_idx, meta in enumerate(series_meta):
                if meta["user_bucket"] is None:
                    continue

                n_key = _nearest_n_for_event(
                    box_event, meta["n"], seq_length, meta["shift"]
                )
                if n_key not in meta["user_bucket"]:
                    continue

                user_eers = meta["user_bucket"][n_key]
                if drop_max_eer_outlier:
                    user_eers = _drop_max_eer_outlier(user_eers)
                x_pos = n_key * seq_length + meta["shift"]
                offset = box_offsets[s_idx] if show_boxplots else 0.0

                if show_boxplots:
                    mean_val, std_val = _draw_box_at(
                        ax,
                        x_pos,
                        user_eers,
                        meta["color"],
                        offset=offset,
                        box_width=box_width,
                    )
                else:
                    mean_val = float(np.mean(user_eers))
                    std_val = float(np.std(user_eers))

                if show_errorbars and mean_val is not None and std_val is not None:
                    ax.errorbar(
                        x_pos + offset,
                        mean_val,
                        yerr=std_val,
                        fmt="none",
                        ecolor=meta["color"],
                        elinewidth=1.5,
                        capsize=3.8,
                        capthick=1.5,
                        alpha=0.92,
                        zorder=6,
                    )

    # ---- axes ----
    ax.set_xlim(left=float(x_axis_start_event))
    ax.set_ylim(bottom=0)

    if all_eer_max <= 1.0:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))  # 1% ticks
    else:
        ax.yaxis.set_major_formatter(PercentFormatter(100.0, decimals=2))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))  # 1% ticks

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("Number of Events", fontsize=12.5, color=_STYLE["label"], labelpad=8)
    ax.set_ylabel("Mean EER (%)", fontsize=12.5, color=_STYLE["label"], labelpad=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

    if title:
        ax.set_title(title, fontsize=14, weight="bold", color=_STYLE["title"], pad=12)
    else:
        ax.set_title(
            f"{dataset_name}: Mean EER vs Mean Number of Events",
            fontsize=14,
            weight="bold",
            color=_STYLE["title"],
            pad=12,
        )

    legend = ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor=_STYLE["legend_edge"],
        facecolor=_STYLE["legend_face"],
        framealpha=0.96,
        markerscale=0.95,
        handlelength=2.0,
        handletextpad=0.6,
        borderpad=0.7,
        labelspacing=0.45,
    )
    legend.get_frame().set_linewidth(1.0)

    if panel_label:
        fig.text(
            0.5,
            -0.02,
            panel_label,
            ha="center",
            va="top",
            fontsize=12,
            color=_STYLE["label"],
            weight="medium",
        )

    # Re-apply after all artists so spines/grid cannot be overridden.
    _style_publication_axes(ax, fig)
    fig.tight_layout()
    plt.show()
