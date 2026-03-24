"""Bug-type confusion matrix analysis for thesis experiments."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.common.types import canonical_fine_bug_type

LOGGER = logging.getLogger("thesis_exp.analysis.confusion_matrix")


def _method_from_record(record: dict[str, Any]) -> str:
    """Return method/condition label: direct | ef_leaky | ef_no_answer | other."""
    name = str(record.get("diagnosis_output", {}).get("prompt_template_name", "unknown"))
    if name == "direct_diagnosis":
        return "direct"
    if name == "diagnosis_with_execution_feedback":
        return "ef_leaky"
    if name == "diagnosis_with_execution_feedback_no_leakage":
        return "ef_no_answer"
    return "other"


def _ground_truth_bug_type(record: dict[str, Any]) -> str:
    raw = record.get("evaluation_result", {}).get("ground_truth_bug_type")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "unknown"
    merged = canonical_fine_bug_type(str(raw).strip())
    return merged if merged else "unknown"


def _predicted_bug_type(record: dict[str, Any]) -> str | None:
    val = record.get("evaluation_result", {}).get("predicted_bug_type")
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    merged = canonical_fine_bug_type(str(val).strip())
    return merged if merged else None


def _filter_valid_parsed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only records with valid parsed bug type."""
    return [r for r in records if _predicted_bug_type(r) is not None]


def _safe_method_name(method: str) -> str:
    """Sanitize method name for filenames."""
    return method.replace("/", "_").replace("\\", "_")


def build_confusion_matrices_with_pandas(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, "pd.DataFrame"], dict[str, "pd.DataFrame"]]:
    """
    Build raw and row-normalized confusion matrices per method using pandas.

    Returns:
        - summary_data: dict for confusion_matrix_summary.json
        - counts_dfs: method -> DataFrame (rows=ground_truth, cols=predicted)
        - normalized_dfs: method -> row-normalized DataFrame
    """
    import pandas as pd

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[_method_from_record(r)].append(r)

    summary_data: dict[str, Any] = {"methods": {}}
    counts_dfs: dict[str, pd.DataFrame] = {}
    normalized_dfs: dict[str, pd.DataFrame] = {}

    for method, group_records in sorted(grouped.items()):
        filtered = _filter_valid_parsed(group_records)
        if not filtered:
            LOGGER.info("No valid parsed bug types for method %s, skipping", method)
            continue

        gt_labels = [_ground_truth_bug_type(r) for r in filtered]
        pred_labels = [_predicted_bug_type(r) for r in filtered]
        assert all(p is not None for p in pred_labels)
        pred_labels = [p for p in pred_labels if p is not None]

        all_labels = sorted(set(gt_labels) | set(pred_labels))
        if not all_labels:
            continue

        df = pd.crosstab(
            pd.Series(gt_labels, name="ground_truth"),
            pd.Series(pred_labels, name="predicted"),
            rownames=["ground_truth"],
            colnames=["predicted"],
        )

        df = df.reindex(index=all_labels, columns=all_labels, fill_value=0).astype(int)

        row_sums = df.sum(axis=1)
        row_sums = row_sums.replace(0, 1.0)
        df_norm = df.div(row_sums, axis=0)
        df_norm = df_norm.fillna(0.0)

        counts_dfs[method] = df
        normalized_dfs[method] = df_norm

        diagonal = sum(df.loc[label, label] for label in all_labels if label in df.index and label in df.columns)
        total = int(df.values.sum())
        diagonal_accuracy = diagonal / total if total > 0 else 0.0

        off_diagonal: list[tuple[str, str, int, float]] = []
        for gt in all_labels:
            for pred in all_labels:
                if gt != pred:
                    c = int(df.loc[gt, pred])
                    if c > 0:
                        rate = float(df_norm.loc[gt, pred])
                        off_diagonal.append((gt, pred, c, rate))

        top_confusion = None
        if off_diagonal:
            top = max(off_diagonal, key=lambda x: x[2])
            top_confusion = {
                "ground_truth": top[0],
                "predicted": top[1],
                "count": top[2],
                "normalized_rate": round(top[3], 4),
            }

        summary_data["methods"][method] = {
            "n_valid_records": len(filtered),
            "diagonal_accuracy": round(diagonal_accuracy, 4),
            "top_off_diagonal_confusion": top_confusion,
        }

    return summary_data, counts_dfs, normalized_dfs


def _write_dataframe_csv(df: "pd.DataFrame", output_path: Path) -> None:
    """Write DataFrame to CSV with ground_truth as first column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

def _create_heatmap_png(
    df: "pd.DataFrame",
    *,
    title: str,
    output_path: Path,
    is_counts: bool = True,
) -> None:
    """Create matplotlib heatmap PNG. Rows=ground_truth, Cols=predicted."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        LOGGER.warning("matplotlib not available, skipping heatmap: %s", exc)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = df.values
    row_labels = list(df.index)
    col_labels = list(df.columns)
    n_rows, n_cols = matrix.shape

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(4, n_rows * 0.6)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1.0 if not is_counts else None)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    max_val = float(np.nanmax(matrix)) if matrix.size else 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if is_counts:
                text_val = str(int(val))
            else:
                text_val = f"{val:.2f}" if val > 0 else "0.00"
            color = "white" if max_val > 0 and val > max_val / 2 else "black"
            ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=10)

    ax.set_xlabel("Predicted bug type")
    ax.set_ylabel("Ground-truth bug type")
    ax.set_title(title)
    cbar_label = "Count" if is_counts else "Row-normalized rate"
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved heatmap to %s", output_path)


def generate_confusion_matrix_artifacts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[str]:
    """
    Build confusion matrices per method and export CSVs, heatmaps, and summary.

    Outputs per method:
    - confusion_matrix_<method>_counts.csv
    - confusion_matrix_<method>_normalized.csv
    - confusion_matrix_<method>_heatmap_counts.png
    - confusion_matrix_<method>_heatmap_normalized.png

    Plus: confusion_matrix_summary.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        summary_data, counts_dfs, normalized_dfs = build_confusion_matrices_with_pandas(records)
    except ImportError as exc:
        LOGGER.warning("pandas not available, skipping confusion matrix analysis: %s", exc)
        return []

    created: list[str] = []

    for method in sorted(counts_dfs.keys()):
        safe = _safe_method_name(method)
        df_counts = counts_dfs[method]
        df_norm = normalized_dfs[method]

        counts_csv = output_path / f"confusion_matrix_{safe}_counts.csv"
        norm_csv = output_path / f"confusion_matrix_{safe}_normalized.csv"
        heatmap_counts_png = output_path / f"confusion_matrix_{safe}_heatmap_counts.png"
        heatmap_norm_png = output_path / f"confusion_matrix_{safe}_heatmap_normalized.png"

        _write_dataframe_csv(df_counts, counts_csv)
        _write_dataframe_csv(df_norm, norm_csv)
        _create_heatmap_png(
            df_counts,
            title=f"Bug-type confusion matrix ({method}) — counts",
            output_path=heatmap_counts_png,
            is_counts=True,
        )
        _create_heatmap_png(
            df_norm,
            title=f"Bug-type confusion matrix ({method}) — row-normalized",
            output_path=heatmap_norm_png,
            is_counts=False,
        )

        created.extend([str(counts_csv), str(norm_csv), str(heatmap_counts_png), str(heatmap_norm_png)])

    summary_json_path = output_path / "confusion_matrix_summary.json"
    summary_json_path.write_text(json.dumps(summary_data, indent=2, sort_keys=True), encoding="utf-8")
    created.append(str(summary_json_path))

    return created
