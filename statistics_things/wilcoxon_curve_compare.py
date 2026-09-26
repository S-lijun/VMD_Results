# statistics/wilcoxon_curve_compare.py

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path


def load_json(path):
    path = Path(path)
    if path.is_dir():
        preferred = path / "MultiCNN.json"
        if preferred.exists():
            path = preferred
        else:
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON found in directory: {path}")
            path = json_files[0]
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def collect_eer_by_n(data, n):
    """
    data: dict loaded from JSON
    n: int
    return: dict {user: eer}
    """
    eer_dict = {}
    for user, records in data.items():
        if not isinstance(records, dict):
            continue
        if str(n) in records:
            eer = records[str(n)].get("EER", None)
            if eer is not None:
                eer_dict[str(user)] = float(eer)
    return eer_dict


# Numeric-key Balabit files use enumeration index (NOT the raw user id).
# Contrasts with named keys like "user7" / "user9" / ...
# 0->user7, 1->user9, 2->user12, 3->user15, 4->user16,
# 5->user20, 6->user21, 7->user23, 8->user29, 9->user35
BALABIT_USER_ORDER = [
    "user7",
    "user9",
    "user12",
    "user15",
    "user16",
    "user20",
    "user21",
    "user23",
    "user29",
    "user35",
]

BALABIT_INDEX_TO_USER = {str(i): u for i, u in enumerate(BALABIT_USER_ORDER)}


def _all_numeric_keys(keys):
    return bool(keys) and all(str(k).isdigit() for k in keys)


def _is_balabit_enum_keys(keys):
    """True iff keys are exactly the Balabit enumeration indices 0..9."""
    if not _all_numeric_keys(keys):
        return False
    return set(str(k) for k in keys) == set(BALABIT_INDEX_TO_USER.keys())


def normalize_eer_user_keys(eer_dict):
    """
    Canonicalize per-user EER keys.

    - Named Balabit keys ("user7", ...) stay as-is
    - Balabit enumeration indices 0..9 -> user7..user35 (对照表)
    - Other numeric keys (e.g. TWOS 0..23) stay as digit strings
    """
    keys = list(eer_dict.keys())
    if not keys:
        return {}

    # Already named (user7 / user_7 style)
    if not _all_numeric_keys(keys):
        out = {}
        for k, v in eer_dict.items():
            s = str(k).strip()
            if s.lower().startswith("user"):
                # accept "user7" or "user_7"
                tail = s[4:].lstrip("_")
                out[f"user{int(tail)}"] = float(v)
            else:
                out[s] = float(v)
        return out

    # Exact Balabit 0..9 enumeration -> named users
    if _is_balabit_enum_keys(keys):
        return {
            BALABIT_INDEX_TO_USER[str(int(k))]: float(eer_dict[k])
            for k in keys
        }

    # Other datasets with numeric ids
    return {str(k): float(v) for k, v in eer_dict.items()}


def align_eer_dicts(eer_a, eer_b):
    """
    Pair users across two EER dicts after canonicalizing keys.

    Always applies Balabit 0..9 -> user* mapping when present, so
    named files ("user7") and index files ("0") line up on the对照表.
    """
    eer_a = normalize_eer_user_keys(eer_a)
    eer_b = normalize_eer_user_keys(eer_b)
    order_index = {u: i for i, u in enumerate(BALABIT_USER_ORDER)}

    def _sort_key(u):
        if u in order_index:
            return (0, order_index[u])
        if str(u).isdigit():
            return (1, int(u))
        return (2, str(u))

    common = sorted(set(eer_a) & set(eer_b), key=_sort_key)
    return common, eer_a, eer_b


def align_many_eer_dicts(eer_dicts: dict):
    """
    Canonicalize and intersect many {label: {user: eer}} dicts.
    Returns (common_users, normalized_dicts).
    """
    normed = {lab: normalize_eer_user_keys(d) for lab, d in eer_dicts.items()}
    common = set.intersection(*(set(d.keys()) for d in normed.values()))
    order_index = {u: i for i, u in enumerate(BALABIT_USER_ORDER)}

    def _sort_key(u):
        if u in order_index:
            return (0, order_index[u])
        if str(u).isdigit():
            return (1, int(u))
        return (2, str(u))

    return sorted(common, key=_sort_key), normed


def wilcoxon_curve_test(
    json_a_path,
    json_b_path,
    n_targets,
    label_a="Method A",
    label_b="Method B",
    alpha=0.1,
):
    """
    Replicates the exact logic of sig_diffs but for n-based score fusion curves.

    Parameters
    ----------
    json_a_path : str
        Path to first JSON (e.g. XYPlot)
    json_b_path : str
        Path to second JSON (e.g. CDF)
    n_targets : list[int]
        List of n values to test
    label_a, label_b : str
        Names shown in result table
    alpha : float
        Significance level
    """

    data_a = load_json(json_a_path)
    data_b = load_json(json_b_path)

    results = []
    warned_align = False

    for n in n_targets:
        eer_a = collect_eer_by_n(data_a, n)
        eer_b = collect_eer_by_n(data_b, n)

        a_was_numeric = _all_numeric_keys(eer_a.keys())
        b_was_numeric = _all_numeric_keys(eer_b.keys())
        common_users, eer_a, eer_b = align_eer_dicts(eer_a, eer_b)

        if not warned_align and (a_was_numeric or b_was_numeric):
            print("[Info] numeric index -> user (0->user7, ..., 9->user35):")
            for idx, user in enumerate(BALABIT_USER_ORDER):
                print(f"  {idx} -> {user}")
            print(f"[Info] paired EERs at n={n}:")
            for user in common_users:
                ea = eer_a.get(user)
                eb = eer_b.get(user)
                src_a = (
                    str(BALABIT_USER_ORDER.index(user))
                    if a_was_numeric and user in BALABIT_USER_ORDER
                    else user
                )
                src_b = (
                    str(BALABIT_USER_ORDER.index(user))
                    if b_was_numeric and user in BALABIT_USER_ORDER
                    else user
                )
                print(
                    f"  {user}: A[{src_a}]={ea*100:.2f}%  B[{src_b}]={eb*100:.2f}%"
                )
            warned_align = True

        if len(common_users) < 2:
            print(
                f"[Skip] n={n}: not enough paired users "
                f"(A={len(eer_a)}, B={len(eer_b)}, common={len(common_users)})"
            )
            continue

        vec_a = np.array([eer_a[u] for u in common_users])
        vec_b = np.array([eer_b[u] for u in common_users])

        mean_a = np.mean(vec_a)
        mean_b = np.mean(vec_b)
        abs_diff = abs(mean_b - mean_a)

        w_stat, p_value = wilcoxon(vec_a, vec_b)

        results.append({
            "Comparison": f"{label_a} vs {label_b}",
            "n": n,
            "Users": len(common_users),
            f"Mean EER ({label_a})": round(mean_a * 100, 2),
            f"Mean EER ({label_b})": round(mean_b * 100, 2),
            "Absolute Diff (%)": round(abs_diff * 100, 2),
            "Wilcoxon p-value": p_value,
            "Statistically Significant": p_value <= alpha,
        })

    df = pd.DataFrame(results)
    return df
