# ============================================================
# compare_sota.py (FINAL UNIVERSAL VERSION WITH OFFSET)
# ============================================================

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# USER NORMALIZATION（支持 offset）
# ============================================================

def normalize_user(u, offset=0):
    """
    Normalize ANY format → 'userX'

    support:
    - 'user7'
    - '7'
    - 7

    offset:
        0  → 
        +1 → ours 0 → user1
        -1 → ours 1 → user0
    """

    u = str(u).strip()

    if u.lower().startswith("user"):
        num = int(u[4:])
    else:
        num = int(u)

    num = num + offset

    return f"user{num}"


# ============================================================
# LOAD JSON (OURS)
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_eer_json(data, n, offset=0):
    eer_dict = {}

    for user, records in data.items():

        try:
            uid = normalize_user(user, offset)
        except:
            continue

        if str(n) not in records:
            continue

        eer = records[str(n)].get("EER", None)
        if eer is None:
            continue

        eer_dict[uid] = float(eer)

    return eer_dict


# ============================================================
# LOAD CSV (SOTA)
# ============================================================

def load_sota_csv(path, sota_n=1, offset=0):
    df = pd.read_csv(path)

    df = df[df["User"].astype(str) != "Mean"]
    df = df[df["n"] == sota_n]

    eer_dict = {}

    for _, row in df.iterrows():
        try:
            uid = normalize_user(row["User"], offset)
        except:
            continue

        eer_dict[uid] = float(row["EER"])

    return eer_dict


# ============================================================
# MAIN FUNCTION
# ============================================================

def compare_sota_vs_ours(
    json_path,
    sota_csv_path,
    ours_n=4,
    sota_n=1,
    ours_offset=0,  
    sota_offset=0,   
    alpha=0.05,
    alternative="less",
    debug=True,
):

    data = load_json(json_path)

    eer_ours = extract_eer_json(data, ours_n, offset=ours_offset)
    eer_sota = load_sota_csv(sota_csv_path, sota_n, offset=sota_offset)

    # ========================================================
    # DEBUG
    # ========================================================

    if debug:
        print("\n==== DEBUG ====")
        print("OURS count:", len(eer_ours))
        print("SOTA count:", len(eer_sota))

        print("OURS users:", sorted(eer_ours.keys())[:])
        print("SOTA users:", sorted(eer_sota.keys())[:])

    # ========================================================
    # INTERSECTION
    # ========================================================

    users = sorted(set(eer_ours.keys()) & set(eer_sota.keys()))

    if debug:
        print("\nPaired users:", users)
        print("Paired count:", len(users))

    if len(users) < 3:
        raise ValueError(f"Too few paired users: {len(users)}")

    vec_ours = np.array([eer_ours[u] for u in users])
    vec_sota = np.array([eer_sota[u] for u in users])

    # ========================================================
    # STATS
    # ========================================================

    mean_ours = np.mean(vec_ours)
    mean_sota = np.mean(vec_sota)

    w_stat, p_value = wilcoxon(vec_ours, vec_sota, alternative=alternative)

    return {
        "Users": len(vec_ours),
        "Mean_Ours (%)": round(mean_ours * 100, 2),
        "Mean_SOTA (%)": round(mean_sota * 100, 2),
        "p-value": float(p_value),
        "Significant": bool(p_value < alpha),
        "Ours Better": bool(mean_ours < mean_sota),
    }


# ============================================================
# PRETTY PRINT
# ============================================================

def pretty_print(res):

    print("\n==============================")
    print("SOTA vs Ours (600-event)")
    print("==============================")

    print(f"Users: {res['Users']}")
    print(f"Ours : {res['Mean_Ours (%)']}%")
    print(f"SOTA : {res['Mean_SOTA (%)']}%")

    print(f"\np-value: {res['p-value']:.6f}")
    print(f"Significant: {res['Significant']}")