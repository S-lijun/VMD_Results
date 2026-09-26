"""Summarize aggregate result JSONs into a small comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_aggregate_json(result_dir: str | Path, json_name: str = "MultiCNN.json") -> dict:
    result_dir = Path(result_dir)
    if result_dir.is_file():
        path = result_dir
    else:
        path = result_dir / json_name
        if not path.exists():
            json_files = sorted(result_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON found in: {result_dir}")
            path = json_files[0]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "avg_eer" not in data:
        raise ValueError(f"Expected aggregate JSON with avg_eer: {path}")
    return data


def _parse_dataset_and_model(result_dir: str | Path) -> tuple[str, str]:
    """
    Expect .../{Dataset}/{Model}/{RepresentationDir}
    e.g. Results_uc/Balabit/CNN/PairWise -> Balabit, CNN
    """
    p = Path(result_dir)
    if p.is_file():
        p = p.parent
    model = p.parent.name
    dataset = p.parent.parent.name
    return dataset, model


def summarize_result(
    result_dir: str | Path,
    representation: str,
    json_name: str = "MultiCNN.json",
    eer_as_percent: bool = True,
) -> pd.DataFrame:
    """
    Build a one-row table from an aggregate result directory/file.

    Parameters
    ----------
    result_dir :
        Path like Results_uc/Balabit/CNN/PairWise (dir or JSON file).
    representation :
        Display name you choose, e.g. "PairWise", "XYPlot Velocity".
    json_name :
        JSON filename inside the directory (default MultiCNN.json).
    eer_as_percent :
        If True, report EER and AUC as percent (x100).

    Returns
    -------
    DataFrame with columns: Dataset, Representation, Model, EER, AUC
    """
    data = _load_aggregate_json(result_dir, json_name=json_name)
    dataset, model = _parse_dataset_and_model(result_dir)

    eer = float(data["avg_eer"][0])
    if "avg_auc" in data:
        auc = float(data["avg_auc"][0])
    else:
        auc = float("nan")

    if eer_as_percent:
        eer = eer * 100.0
        auc = auc * 100.0

    decimals = 4 if eer_as_percent else 6
    return pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Representation": representation,
                "Model": model,
                "EER": round(eer, decimals),
                "AUC": round(auc, decimals),
            }
        ]
    )


def summarize_results(
    entries: list[tuple[str | Path, str]],
    json_name: str = "MultiCNN.json",
    eer_as_percent: bool = True,
) -> pd.DataFrame:
    """
    Batch version.

    entries : list of (result_dir, representation_name)
    """
    frames = [
        summarize_result(
            path,
            representation,
            json_name=json_name,
            eer_as_percent=eer_as_percent,
        )
        for path, representation in entries
    ]
    return pd.concat(frames, ignore_index=True)
