"""Summarize experiment outputs into tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from thesis_exp.analysis.confusion_matrix import generate_confusion_matrix_artifacts
from thesis_exp.common.types import canonical_fine_bug_type
from thesis_exp.schemas.sample import EvaluationResult


LOGGER = logging.getLogger("thesis_exp.analysis.summarize")

MAIN_RESULT_METRICS = (
    "bug_type_accuracy",
    "bug_type_accuracy_coarse",
    "localization_accuracy",
    "localization_accuracy_exact",
    "patch_passes_selected_tests",
    "repair_success",
    "diagnosis_hallucination_rate",
    "narrative_hallucination_rate",
    "hallucination_rate",
)

BUG_TYPE_BREAKDOWN_METRICS = (
    "bug_type_accuracy",
    "bug_type_accuracy_coarse",
    "localization_accuracy",
    "localization_accuracy_exact",
    "patch_passes_selected_tests",
    "repair_success",
)


@dataclass(slots=True)
class AnalysisArtifacts:
    """Paths for generated analysis outputs."""

    main_table_csv: str
    bug_type_table_csv: str
    execution_feedback_table_csv: str
    hallucination_rate_figure_svg: str
    two_layer_repair_figure_svg: str
    repair_success_vs_bug_type_figure_svg: str
    summary_json: str


def summarize_results(results: Iterable[EvaluationResult]) -> dict[str, float]:
    """Compute dataset-level means for core metrics (incl. two-layer repair)."""

    materialized_results = list(results)
    if not materialized_results:
        return {metric_name: 0.0 for metric_name in MAIN_RESULT_METRICS}

    return {
        metric_name: mean(float(getattr(result, metric_name)) for result in materialized_results)
        for metric_name in MAIN_RESULT_METRICS
    }


def load_result_records(results_jsonl_path: str | list[str]) -> list[dict[str, Any]]:
    """Load runner result records from one or more JSONL files."""

    paths = [results_jsonl_path] if isinstance(results_jsonl_path, str) else results_jsonl_path
    records: list[dict[str, Any]] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            LOGGER.warning("Results path does not exist: %s", path)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _method_name(record: dict[str, Any]) -> str:
    diagnosis_output = record.get("diagnosis_output", {})
    return str(diagnosis_output.get("prompt_template_name", "unknown_method"))


def _model_provider_name(record: dict[str, Any]) -> str:
    diagnosis_output = record.get("diagnosis_output", {})
    return str(diagnosis_output.get("model_provider_name", "unknown_provider"))


def _model_name(record: dict[str, Any]) -> str:
    diagnosis_output = record.get("diagnosis_output", {})
    return str(diagnosis_output.get("model_name", "unknown_model"))


def _model_label(record: dict[str, Any]) -> str:
    return f"{_model_provider_name(record)}/{_model_name(record)}"


def _ground_truth_bug_type(record: dict[str, Any]) -> str:
    evaluation_result = record.get("evaluation_result", {})
    raw = evaluation_result.get("ground_truth_bug_type")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "unknown_bug_type"
    merged = canonical_fine_bug_type(str(raw).strip())
    return merged if merged else "unknown_bug_type"


def _uses_execution_feedback(record: dict[str, Any]) -> bool:
    prompt_template_name = _method_name(record)
    return "execution_feedback" in prompt_template_name


def _ef_condition(record: dict[str, Any]) -> str:
    """Return condition label for three-way comparison: direct | ef_leaky | ef_no_answer | diagnosis_only | diagnosis_then_repair."""
    name = _method_name(record)
    if name == "direct_diagnosis":
        return "direct"
    if name == "diagnosis_with_execution_feedback":
        return "ef_leaky"
    if name == "diagnosis_with_execution_feedback_no_leakage":
        return "ef_no_answer"
    if name == "diagnosis_only":
        return "diagnosis_only"
    if name == "diagnosis_then_repair":
        return "diagnosis_then_repair"
    return "other"


def _paired_key(record: dict[str, Any]) -> tuple[str, str]:
    """Canonical key for joining: (sample_id, variant id or 'original').

    New result records omit ``transformed_sample_id``; missing key is treated as ``original``.
    """
    sample_id = str(record.get("sample_id", ""))
    tid = record.get("transformed_sample_id")
    transformed_id = str(tid) if tid else "original"
    return (sample_id, transformed_id)


def filter_to_paired_subset(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Keep only records where the same (sample_id, transformed_sample_id) exists in all three conditions.
    Returns (filtered_records, alignment_info).
    """
    by_condition: dict[str, list[dict[str, Any]]] = {
        "direct": [],
        "ef_leaky": [],
        "ef_no_answer": [],
    }
    for r in records:
        cond = _ef_condition(r)
        if cond in by_condition:
            by_condition[cond].append(r)

    keys_direct = {_paired_key(r) for r in by_condition["direct"]}
    keys_ef_leaky = {_paired_key(r) for r in by_condition["ef_leaky"]}
    keys_ef_no_answer = {_paired_key(r) for r in by_condition["ef_no_answer"]}

    # Paired = present in all three
    paired_keys = keys_direct & keys_ef_leaky & keys_ef_no_answer

    # If any condition is empty, no paired subset; return all records
    if not paired_keys or not by_condition["direct"] or not by_condition["ef_leaky"] or not by_condition["ef_no_answer"]:
        alignment_info = {
            "use_paired_subset": False,
            "n_paired": 0,
            "n_direct": len(by_condition["direct"]),
            "n_ef_leaky": len(by_condition["ef_leaky"]),
            "n_ef_no_answer": len(by_condition["ef_no_answer"]),
            "reason": "Paired subset empty or not all three conditions present.",
        }
        return records, alignment_info

    # Keep one record per (paired_key, condition) to avoid double-counting duplicates
    seen: set[tuple[tuple[str, str], str]] = set()
    filtered: list[dict[str, Any]] = []
    for r in records:
        key = _paired_key(r)
        if key not in paired_keys:
            continue
        cond = _ef_condition(r)
        if (key, cond) in seen:
            continue
        seen.add((key, cond))
        filtered.append(r)
    alignment_info = {
        "use_paired_subset": True,
        "n_paired": len(paired_keys),
        "n_direct": len(by_condition["direct"]),
        "n_ef_leaky": len(by_condition["ef_leaky"]),
        "n_ef_no_answer": len(by_condition["ef_no_answer"]),
        "n_direct_only": len(keys_direct - keys_ef_leaky - keys_ef_no_answer),
        "n_ef_leaky_only": len(keys_ef_leaky - keys_direct - keys_ef_no_answer),
        "n_ef_no_answer_only": len(keys_ef_no_answer - keys_direct - keys_ef_leaky),
    }
    return filtered, alignment_info


def _infer_patch_passes_from_notes(evaluation_result: dict[str, Any]) -> float | None:
    """Best-effort for JSONL written before ``patch_passes_selected_tests`` existed."""

    if evaluation_result.get("patch_passes_selected_tests") is not None:
        return None
    notes = str(evaluation_result.get("evaluation_notes", "") or "")
    if "patch_passes_selected_tests=" in notes:
        frag = notes.split("patch_passes_selected_tests=")[1].split("|")[0].strip()
        try:
            return 1.0 if int(frag) == 1 else 0.0
        except ValueError:
            return None
    if "repair_tests_passed=" in notes:
        frag = notes.split("repair_tests_passed=")[1].split("|")[0].strip()
        if "/" in frag:
            a, b = frag.split("/", 1)
            try:
                ai, bi = int(a.strip()), int(b.strip())
                if bi == 0:
                    return 0.0
                return 1.0 if ai == bi else 0.0
            except ValueError:
                return None
    return None


def _metric_value(record: dict[str, Any], metric_name: str) -> float:
    evaluation_result = record.get("evaluation_result", {})
    value = evaluation_result.get(metric_name)
    if value is None and metric_name == "localization_accuracy_exact":
        value = evaluation_result.get("localization_accuracy")
    if value is None and metric_name == "patch_passes_selected_tests":
        inferred = _infer_patch_passes_from_notes(evaluation_result)
        if inferred is not None:
            return inferred
    return _safe_float(value)


def _derived_rate(records: list[dict[str, Any]], field_name: str) -> float:
    if not records:
        return 0.0
    return mean(float(bool(record.get("evaluation_result", {}).get(field_name, False))) for record in records)


def _group_mean(records: list[dict[str, Any]], metric_name: str) -> float:
    if not records:
        return 0.0
    return mean(_metric_value(record, metric_name) for record in records)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _create_bar_chart_svg(
    values: list[tuple[str, float]],
    *,
    title: str,
    y_axis_label: str,
    output_path: Path,
) -> None:
    width = 960
    height = 540
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max((value for _, value in values), default=1.0)
    max_value = max(max_value, 1.0)
    bar_width = plot_width / max(len(values), 1)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{_escape_xml(title)}</text>',
        f'<text x="24" y="{margin_top + plot_height / 2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 24 {margin_top + plot_height / 2})">{_escape_xml(y_axis_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
    ]

    for tick_index in range(6):
        tick_value = tick_index / 5
        y = margin_top + plot_height - (tick_value * plot_height)
        elements.append(f'<line x1="{margin_left - 6}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="black"/>')
        elements.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial">{tick_value:.1f}</text>'
        )

    for index, (label, value) in enumerate(values):
        x = margin_left + index * bar_width + 10
        normalized_height = (value / max_value) * plot_height if max_value > 0 else 0
        y = margin_top + plot_height - normalized_height
        elements.append(
            f'<rect x="{x}" y="{y}" width="{max(bar_width - 20, 10)}" height="{normalized_height}" fill="#4f81bd"/>'
        )
        elements.append(
            f'<text x="{x + max(bar_width - 20, 10) / 2}" y="{y - 6}" text-anchor="middle" font-size="12" font-family="Arial">{value:.3f}</text>'
        )
        label_x = x + max(bar_width - 20, 10) / 2
        label_y = margin_top + plot_height + 20
        elements.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="end" font-size="12" font-family="Arial" transform="rotate(-45 {label_x} {label_y})">{_escape_xml(label)}</text>'
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def _create_grouped_bar_chart_svg(
    values: list[tuple[str, float, float]],
    *,
    title: str,
    left_metric_name: str,
    right_metric_name: str,
    output_path: Path,
) -> None:
    width = 960
    height = 540
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / max(len(values), 1)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{_escape_xml(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<rect x="{width - 250}" y="50" width="14" height="14" fill="#4f81bd"/>',
        f'<text x="{width - 230}" y="62" font-size="12" font-family="Arial">{_escape_xml(left_metric_name)}</text>',
        f'<rect x="{width - 130}" y="50" width="14" height="14" fill="#c0504d"/>',
        f'<text x="{width - 110}" y="62" font-size="12" font-family="Arial">{_escape_xml(right_metric_name)}</text>',
    ]

    for tick_index in range(6):
        tick_value = tick_index / 5
        y = margin_top + plot_height - (tick_value * plot_height)
        elements.append(f'<line x1="{margin_left - 6}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="black"/>')
        elements.append(
            f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial">{tick_value:.1f}</text>'
        )

    for index, (label, left_value, right_value) in enumerate(values):
        group_x = margin_left + index * group_width
        inner_width = max(group_width - 20, 20)
        bar_width = inner_width / 2 - 6
        left_height = left_value * plot_height
        right_height = right_value * plot_height
        left_x = group_x + 10
        right_x = left_x + bar_width + 12
        left_y = margin_top + plot_height - left_height
        right_y = margin_top + plot_height - right_height

        elements.append(f'<rect x="{left_x}" y="{left_y}" width="{bar_width}" height="{left_height}" fill="#4f81bd"/>')
        elements.append(f'<rect x="{right_x}" y="{right_y}" width="{bar_width}" height="{right_height}" fill="#c0504d"/>')
        label_x = group_x + inner_width / 2 + 10
        label_y = margin_top + plot_height + 20
        elements.append(
            f'<text x="{label_x}" y="{label_y}" text-anchor="end" font-size="12" font-family="Arial" transform="rotate(-45 {label_x} {label_y})">{_escape_xml(label)}</text>'
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def build_main_result_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build table 1 grouped by method and model."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (_method_name(record), _model_provider_name(record), _model_name(record))
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (method_name, provider_name, model_name), group_records in sorted(grouped.items()):
        row = {
            "method": method_name,
            "model_provider": provider_name,
            "model_name": model_name,
            "n_records": len(group_records),
        }
        for metric_name in MAIN_RESULT_METRICS:
            row[metric_name] = round(_group_mean(group_records, metric_name), 6)
        rows.append(row)
    return rows


def build_bug_type_breakdown_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build table 2 grouped by ground-truth bug type."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_ground_truth_bug_type(record), []).append(record)

    rows: list[dict[str, Any]] = []
    for bug_type, group_records in sorted(grouped.items()):
        row = {
            "bug_type": bug_type,
            "n_records": len(group_records),
        }
        for metric_name in BUG_TYPE_BREAKDOWN_METRICS:
            row[metric_name] = round(_group_mean(group_records, metric_name), 6)
        rows.append(row)
    return rows


def build_execution_feedback_comparison_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build table 3: Direct vs EF-leaky vs EF-no-answer (three-way when all present)."""

    grouped: dict[str, list[dict[str, Any]]] = {
        "direct": [],
        "ef_leaky": [],
        "ef_no_answer": [],
    }
    for record in records:
        cond = _ef_condition(record)
        if cond in grouped:
            grouped[cond].append(record)

    # Fallback to two-way if only direct + one EF type
    if not grouped["ef_no_answer"] and (grouped["direct"] or grouped["ef_leaky"]):
        legacy_grouped: dict[str, list[dict[str, Any]]] = {
            "without_execution_feedback": grouped["direct"],
            "with_execution_feedback": grouped["ef_leaky"],
        }
        rows: list[dict[str, Any]] = []
        for group_name, group_records in legacy_grouped.items():
            if not group_records:
                continue
            row = {"condition": group_name, "n_records": len(group_records)}
            for metric_name in MAIN_RESULT_METRICS:
                row[metric_name] = round(_group_mean(group_records, metric_name), 6)
            rows.append(row)
        return rows

    rows = []
    for group_name in ("direct", "ef_leaky", "ef_no_answer"):
        group_records = grouped[group_name]
        if not group_records:
            continue
        row = {"condition": group_name, "n_records": len(group_records)}
        for metric_name in MAIN_RESULT_METRICS:
            row[metric_name] = round(_group_mean(group_records, metric_name), 6)
        rows.append(row)
    return rows


def generate_analysis_artifacts(
    results_jsonl_path: str | list[str],
    output_dir: str,
    *,
    use_paired_subset: bool = True,
) -> AnalysisArtifacts:
    """Generate tables and figures from runner JSONL outputs.

    When use_paired_subset=True (default), table_1, table_3, and confusion matrices
    use only the paired subset: samples present in all three conditions (direct,
    ef_leaky, ef_no_answer). This ensures n_records is identical across conditions
    for a fair comparison.
    """

    records = load_result_records(results_jsonl_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records_for_tables = records
    alignment_info: dict[str, Any] = {}
    if use_paired_subset:
        records_for_tables, alignment_info = filter_to_paired_subset(records)
        if alignment_info.get("use_paired_subset"):
            LOGGER.info(
                "Using paired subset: n_paired=%s (direct=%s, ef_leaky=%s, ef_no_answer=%s)",
                alignment_info["n_paired"],
                alignment_info["n_direct"],
                alignment_info["n_ef_leaky"],
                alignment_info["n_ef_no_answer"],
            )
        else:
            LOGGER.info("Paired subset not applicable: %s", alignment_info.get("reason", ""))

    main_table_rows = build_main_result_table(records_for_tables)
    bug_type_rows = build_bug_type_breakdown_table(records_for_tables)
    feedback_rows = build_execution_feedback_comparison_table(records_for_tables)

    main_table_csv = output_path / "table_1_main_results.csv"
    bug_type_table_csv = output_path / "table_2_bug_type_breakdown.csv"
    feedback_table_csv = output_path / "table_3_execution_feedback_comparison.csv"
    hallucination_svg = output_path / "figure_1_hallucination_rate.svg"
    two_layer_repair_svg = output_path / "figure_2_two_layer_repair.svg"
    repair_vs_bug_type_svg = output_path / "figure_3_repair_success_vs_bug_type_accuracy.svg"
    summary_json = output_path / "analysis_summary.json"

    _write_csv(main_table_rows, main_table_csv)
    _write_csv(bug_type_rows, bug_type_table_csv)
    _write_csv(feedback_rows, feedback_table_csv)

    hallucination_grouped_values = [
        (
            f"{row['method']} | {row['model_name']}",
            float(row.get("diagnosis_hallucination_rate", 0.0)),
            float(row.get("narrative_hallucination_rate", row.get("hallucination_rate", 0.0))),
        )
        for row in main_table_rows
    ]
    _create_grouped_bar_chart_svg(
        hallucination_grouped_values,
        title="Diagnosis vs Narrative Hallucination by Method and Model",
        left_metric_name="diagnosis_hallucination_rate",
        right_metric_name="narrative_hallucination_rate",
        output_path=hallucination_svg,
    )

    two_layer_values = [
        (
            f"{row['method']} | {row['model_name']}",
            float(row["patch_passes_selected_tests"]),
            float(row["repair_success"]),
        )
        for row in main_table_rows
    ]
    _create_grouped_bar_chart_svg(
        two_layer_values,
        title="Two-layer repair: patch passes selected tests vs strict success (mutation adequacy)",
        left_metric_name="patch_passes_selected_tests",
        right_metric_name="repair_success",
        output_path=two_layer_repair_svg,
    )

    grouped_bar_values = [
        (
            f"{row['method']} | {row['model_name']}",
            float(row["repair_success"]),
            float(row["bug_type_accuracy"]),
        )
        for row in main_table_rows
    ]
    _create_grouped_bar_chart_svg(
        grouped_bar_values,
        title="Strict repair success vs bug type accuracy",
        left_metric_name="repair_success",
        right_metric_name="bug_type_accuracy",
        output_path=repair_vs_bug_type_svg,
    )

    summary_payload = {
        "n_records": len(records_for_tables),
        "n_records_full": len(records),
        "paired_subset_alignment": alignment_info,
        "repair_without_true_diagnosis_rate_overall": _derived_rate(records_for_tables, "repair_without_true_diagnosis"),
        "main_result_table_rows": main_table_rows,
        "bug_type_breakdown_rows": bug_type_rows,
        "execution_feedback_rows": feedback_rows,
        "repair_without_true_diagnosis_rate_by_method_model": [
            {
                "method": row["method"],
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
                "repair_without_true_diagnosis_rate": round(
                    _derived_rate(
                        [
                            record
                            for record in records_for_tables
                            if _method_name(record) == row["method"]
                            and _model_provider_name(record) == row["model_provider"]
                            and _model_name(record) == row["model_name"]
                        ],
                        "repair_without_true_diagnosis",
                    ),
                    6,
                ),
            }
            for row in main_table_rows
        ],
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    confusion_paths = generate_confusion_matrix_artifacts(records_for_tables, output_path)
    for p in confusion_paths:
        LOGGER.info("Generated confusion matrix artifact: %s", p)

    LOGGER.info("Generated analysis artifacts in %s", output_path)
    return AnalysisArtifacts(
        main_table_csv=str(main_table_csv),
        bug_type_table_csv=str(bug_type_table_csv),
        execution_feedback_table_csv=str(feedback_table_csv),
        hallucination_rate_figure_svg=str(hallucination_svg),
        two_layer_repair_figure_svg=str(two_layer_repair_svg),
        repair_success_vs_bug_type_figure_svg=str(repair_vs_bug_type_svg),
        summary_json=str(summary_json),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize experiment results into tables and figures.")
    parser.add_argument("--results-jsonl", required=True, help="Path to runner results JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated tables and figures.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_argument_parser().parse_args()
    generate_analysis_artifacts(args.results_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
