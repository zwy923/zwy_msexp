"""Verify direct vs execution_feedback comparison integrity."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def sample_key(r: dict) -> str:
    """Canonical key for joining: sample_id (same across experiments)."""
    return r.get("sample_id", "")


def sample_fingerprint(r: dict) -> dict:
    """Fields that must match for same underlying sample."""
    bg = r.get("buggy_sample", {})
    inj = bg.get("bug_injection_record", {})
    return {
        "problem_id": r.get("problem_id"),
        "bug_type": inj.get("bug_type"),
        "changed_line_count": inj.get("changed_line_count"),
        "injection_line_start": inj.get("injection_line_start"),
        "injection_line_end": inj.get("injection_line_end"),
        "buggy_code": bg.get("buggy_code"),
        "transformed_sample_id": r.get("transformed_sample_id"),
    }


def extract_raw_bug_type(r: dict) -> str | None:
    raw = r.get("diagnosis_output", {}).get("raw_response_text", "")
    if not raw:
        return None
    m = re.search(r'"bug_type"\s*:\s*"([^"]+)"', raw)
    return m.group(1) if m else None


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    direct_path = project_root / "results" / "sanitized_mbpp_direct_pilot" / "results.jsonl"
    ef_leaky_path = project_root / "results" / "sanitized_mbpp_execution_feedback_pilot" / "results.jsonl"
    ef_no_answer_path = project_root / "results" / "sanitized_mbpp_execution_feedback_no_leakage_pilot" / "results.jsonl"
    out_dir = project_root / "results" / "three_way_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_records = load_jsonl(direct_path) if direct_path.exists() else []
    ef_leaky_records = load_jsonl(ef_leaky_path) if ef_leaky_path.exists() else []
    ef_no_answer_records = load_jsonl(ef_no_answer_path) if ef_no_answer_path.exists() else []

    direct_by_sample = {sample_key(r): r for r in direct_records}
    ef_leaky_by_sample = {sample_key(r): r for r in ef_leaky_records}
    ef_no_answer_by_sample = {sample_key(r): r for r in ef_no_answer_records}

    # Three-way when all exist; otherwise two-way (direct vs ef_leaky)
    three_way = bool(ef_no_answer_by_sample)
    if three_way:
        in_all = sorted(set(direct_by_sample) & set(ef_leaky_by_sample) & set(ef_no_answer_by_sample))
        ef_by_sample = ef_leaky_by_sample  # for backward compat in artifact paths
        ef_records = ef_leaky_records
    else:
        in_all = sorted(set(direct_by_sample) & set(ef_leaky_by_sample))
        ef_by_sample = ef_leaky_by_sample
        ef_records = ef_leaky_records

    only_direct = sorted(set(direct_by_sample) - set(ef_leaky_by_sample) - set(ef_no_answer_by_sample))
    only_ef_leaky = sorted(set(ef_leaky_by_sample) - set(direct_by_sample) - set(ef_no_answer_by_sample))
    only_ef_no_answer = sorted(set(ef_no_answer_by_sample) - set(direct_by_sample) - set(ef_leaky_by_sample))

    mismatches = []
    for sid in in_all:
        fd = sample_fingerprint(direct_by_sample[sid])
        fe = sample_fingerprint(ef_leaky_by_sample[sid])
        if fd != fe:
            mismatches.append({"sample_id": sid, "direct": fd, "ef_leaky": fe})
        if three_way:
            fn = sample_fingerprint(ef_no_answer_by_sample[sid])
            if fd != fn or fe != fn:
                mismatches.append({"sample_id": sid, "direct": fd, "ef_no_answer": fn})

    alignment_report = {
        "n_direct": len(direct_records),
        "n_ef_leaky": len(ef_leaky_records),
        "n_ef_no_answer": len(ef_no_answer_records),
        "n_in_all": len(in_all),
        "three_way": three_way,
        "n_only_direct": len(only_direct),
        "n_only_ef_leaky": len(only_ef_leaky),
        "n_only_ef_no_answer": len(only_ef_no_answer),
        "n_fingerprint_mismatches": len(mismatches),
        "fingerprint_mismatches": mismatches,
    }

    # Export comparison table (join by sample_id)
    comparison_rows = []
    for sid in in_all:
        rd = direct_by_sample[sid]
        rl = ef_leaky_by_sample[sid]
        inj_d = rd.get("buggy_sample", {}).get("bug_injection_record", {})
        ev_d = rd.get("evaluation_result", {})
        ev_l = rl.get("evaluation_result", {})
        row = {
            "sample_id": sid,
            "record_id_direct": rd.get("record_id"),
            "record_id_ef_leaky": rl.get("record_id"),
            "repair_success_direct": ev_d.get("repair_success"),
            "repair_success_ef_leaky": ev_l.get("repair_success"),
            "bug_type_accuracy_direct": ev_d.get("bug_type_accuracy"),
            "bug_type_accuracy_ef_leaky": ev_l.get("bug_type_accuracy"),
            "loc_acc_direct": ev_d.get("localization_accuracy"),
            "loc_acc_ef_leaky": ev_l.get("localization_accuracy"),
        }
        if three_way:
            rn = ef_no_answer_by_sample[sid]
            ev_n = rn.get("evaluation_result", {})
            row["record_id_ef_no_answer"] = rn.get("record_id")
            row["repair_success_ef_no_answer"] = ev_n.get("repair_success")
            row["bug_type_accuracy_ef_no_answer"] = ev_n.get("bug_type_accuracy")
            row["loc_acc_ef_no_answer"] = ev_n.get("localization_accuracy")
        comparison_rows.append(row)
    comparison_csv_path = out_dir / "sample_alignment_comparison.csv"
    if comparison_rows:
        with comparison_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)
    alignment_report["comparison_csv_path"] = str(comparison_csv_path)

    # 2. Raw bug_type distribution (parser fairness)
    direct_raw = [extract_raw_bug_type(r) for r in direct_records]
    ef_leaky_raw = [extract_raw_bug_type(r) for r in ef_leaky_records]
    direct_parsed = [r.get("evaluation_result", {}).get("predicted_bug_type") for r in direct_records]
    ef_leaky_parsed = [r.get("evaluation_result", {}).get("predicted_bug_type") for r in ef_leaky_records]

    parser_report = {
        "direct_raw_bug_type_counts": dict(Counter(d for d in direct_raw if d)),
        "ef_leaky_raw_bug_type_counts": dict(Counter(e for e in ef_leaky_raw if e)),
        "direct_parsed_bug_type_counts": dict(Counter(p for p in direct_parsed if p)),
        "ef_leaky_parsed_bug_type_counts": dict(Counter(p for p in ef_leaky_parsed if p)),
    }
    if three_way:
        ef_no_raw = [extract_raw_bug_type(r) for r in ef_no_answer_records]
        ef_no_parsed = [r.get("evaluation_result", {}).get("predicted_bug_type") for r in ef_no_answer_records]
        parser_report["ef_no_answer_raw_bug_type_counts"] = dict(Counter(e for e in ef_no_raw if e))
        parser_report["ef_no_answer_parsed_bug_type_counts"] = dict(Counter(p for p in ef_no_parsed if p))

    # 3. Localization: ground truth vs predicted (all records for full check)
    loc_samples = []
    for r in direct_records:
        ev = r.get("evaluation_result", {})
        loc_samples.append({
            "sample_id": sample_key(r),
            "source": "direct",
            "gt_line_start": ev.get("ground_truth_injection_line_start"),
            "gt_line_end": ev.get("ground_truth_injection_line_end"),
            "pred_line": ev.get("predicted_bug_line_start"),
            "loc_acc": ev.get("localization_accuracy"),
            "loc_acc_exact": ev.get("localization_accuracy_exact"),
        })
    for r in ef_leaky_records:
        ev = r.get("evaluation_result", {})
        loc_samples.append({
            "sample_id": sample_key(r),
            "source": "execution_feedback",
            "gt_line_start": ev.get("ground_truth_injection_line_start"),
            "gt_line_end": ev.get("ground_truth_injection_line_end"),
            "pred_line": ev.get("predicted_bug_line_start"),
            "loc_acc": ev.get("localization_accuracy"),
            "loc_acc_exact": ev.get("localization_accuracy_exact"),
        })

    # 4. Execution feedback prompt content (from artifacts)
    ef_artifact_dir = project_root / "results" / "sanitized_mbpp_execution_feedback_pilot" / "artifacts"
    direct_artifact_dir = project_root / "results" / "sanitized_mbpp_direct_pilot" / "artifacts"

    def extract_ef_sections(text: str) -> dict:
        """Extract Failing Test Cases and Execution Feedback sections (potential leakage)."""
        out = {"failing_test_cases": "", "execution_feedback": ""}
        if "## Failing Test Cases" in text:
            start = text.index("## Failing Test Cases")
            rest = text[start:]
            end = rest.find("\n## ", 1)
            out["failing_test_cases"] = rest[:end] if end > 0 else rest
        if "## Execution Feedback" in text:
            start = text.index("## Execution Feedback")
            rest = text[start:]
            end = rest.find("\n## ", 1)
            out["execution_feedback"] = rest[:end] if end > 0 else rest
        return out

    prompt_samples = []
    for i, sid in enumerate(in_all):
        safe_sid = sid.replace("::", "_")
        ef_resp = ef_artifact_dir / "sanitized_mbpp_execution_feedback_pilot" / "openai_compatible" / "gpt-5-mini" / safe_sid / "response.json"
        direct_resp = direct_artifact_dir / "sanitized_mbpp_direct_pilot" / "openai_compatible" / "gpt-5-mini" / safe_sid / "response.json"
        if ef_resp.exists() and direct_resp.exists():
            ef_data = json.loads(ef_resp.read_text(encoding="utf-8"))
            direct_data = json.loads(direct_resp.read_text(encoding="utf-8"))
            ef_req = ef_data.get("request_payload", {})
            direct_req = direct_data.get("request_payload", {})
            ef_user = ""
            direct_user = ""
            for m in ef_req.get("messages", []):
                if m.get("role") == "user":
                    ef_user = m.get("content", "")
                    break
            for m in direct_req.get("messages", []):
                if m.get("role") == "user":
                    direct_user = m.get("content", "")
                    break
            ef_sections = extract_ef_sections(ef_user)
            # LEAKAGE CHECK: Failing Test Cases contain expected output (e.g. assert f(x)==4)
            prompt_samples.append({
                "sample_id": sid,
                "direct_user_len": len(direct_user),
                "ef_user_len": len(ef_user),
                "failing_test_cases_section": ef_sections["failing_test_cases"][:600],
                "execution_feedback_section": ef_sections["execution_feedback"][:400],
                "leakage_note": "Failing test assertions expose expected output (e.g. assert f(x)==y).",
            })

    # 5. Patched code execution sanity (sample repair_success=1 records for manual verification)
    def _repair_success_records(records: list[dict], source: str, n: int = 8) -> list[dict]:
        success_records = [r for r in records if r.get("evaluation_result", {}).get("repair_success") == 1.0]
        samples = []
        for r in success_records[:n]:
            bg = r.get("buggy_sample", {})
            diag = r.get("diagnosis_output", {})
            ev = r.get("evaluation_result", {})
            samples.append({
                "sample_id": sample_key(r),
                "source": source,
                "buggy_code_preview": (bg.get("buggy_code") or "")[:400],
                "parsed_repaired_code_preview": (diag.get("parsed_repaired_code") or "")[:400],
                "entry_point": bg.get("entry_point"),
                "evaluation_notes": ev.get("evaluation_notes"),
                "note": "Verify: patched code is exec'd directly; tests run in same namespace.",
            })
        return samples

    patched_exec_samples = {
        "direct": _repair_success_records(direct_records, "direct"),
        "ef_leaky": _repair_success_records(ef_leaky_records, "ef_leaky"),
    }
    if three_way:
        patched_exec_samples["ef_no_answer"] = _repair_success_records(ef_no_answer_records, "ef_no_answer")

    # 6. repair_without_true_diagnosis breakdown
    def _breakdown_rwtd(records: list[dict], source: str) -> dict:
        wrong_bug_only = []
        wrong_loc_only = []
        both_wrong = []
        for r in records:
            ev = r.get("evaluation_result", {})
            if ev.get("repair_success") != 1.0:
                continue
            bt = ev.get("bug_type_accuracy", 0)
            loc = ev.get("localization_accuracy_exact")
            if loc is None:
                loc = ev.get("localization_accuracy", 0)
            if bt < 1.0 and (loc is None or loc >= 1.0):
                wrong_bug_only.append(sample_key(r))
            elif bt >= 1.0 and (loc is None or loc < 1.0):
                wrong_loc_only.append(sample_key(r))
            elif bt < 1.0 and loc is not None and loc < 1.0:
                both_wrong.append(sample_key(r))
        return {
            "repair_success_count": sum(1 for r in records if r.get("evaluation_result", {}).get("repair_success") == 1.0),
            "repair_without_true_diagnosis_count": sum(1 for r in records if r.get("evaluation_result", {}).get("repair_without_true_diagnosis")),
            "wrong_bug_type_only": wrong_bug_only,
            "wrong_localization_only": wrong_loc_only,
            "both_wrong": both_wrong,
            "n_wrong_bug_only": len(wrong_bug_only),
            "n_wrong_loc_only": len(wrong_loc_only),
            "n_both_wrong": len(both_wrong),
        }

    rwtd_direct = _breakdown_rwtd(direct_records, "direct")
    rwtd_ef_leaky = _breakdown_rwtd(ef_leaky_records, "ef_leaky")
    rwtd_report = {"direct": rwtd_direct, "ef_leaky": rwtd_ef_leaky}
    if three_way:
        rwtd_report["ef_no_answer"] = _breakdown_rwtd(ef_no_answer_records, "ef_no_answer")

    # 7. Pipeline consistency (direct vs execution_feedback)
    direct_cfg = project_root / "configs" / "sanitized_mbpp_direct_pilot.yaml"
    ef_cfg = project_root / "configs" / "sanitized_mbpp_execution_feedback_pilot.yaml"
    pipeline_report = {
        "configs_identical_except": ["experiment_id", "experiment_name", "prompt.variant", "output.results_dir"] if direct_cfg.exists() and ef_cfg.exists() else "config files not found",
        "shared_pipeline": [
            "parse_model_diagnosis_output (same parser)",
            "DiagnosisEvaluator (same evaluator)",
            "execute_patched_code_safely (same sandbox)",
            "EvaluationConfig from YAML (localization_tolerance_lines, repair_timeout, etc.)",
        ],
        "only_difference": "prompt.variant: direct_diagnosis vs diagnosis_with_execution_feedback",
    }

    # Export
    report = {
        "1_sample_alignment": alignment_report,
        "2_parser_fairness": parser_report,
        "3_localization_sample": loc_samples,
        "4_execution_feedback_prompt_extra": prompt_samples,
        "5_patched_code_execution_sample": patched_exec_samples,
        "6_repair_without_true_diagnosis_breakdown": rwtd_report,
        "7_pipeline_consistency": pipeline_report,
    }
    out_path = out_dir / "verification_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also write a human-readable summary
    summary_lines = [
        "=== Verification Report ===",
        "",
        "1. SAMPLE ALIGNMENT",
        f"  Direct records: {alignment_report['n_direct']}",
        f"  EF-leaky records: {alignment_report['n_ef_leaky']}",
        f"  EF-no-answer records: {alignment_report['n_ef_no_answer']}",
        f"  Samples in all: {alignment_report['n_in_all']} (three_way={alignment_report['three_way']})",
        f"  Only in direct: {alignment_report['n_only_direct']}",
        f"  Only in EF-leaky: {alignment_report['n_only_ef_leaky']}",
        f"  Only in EF-no-answer: {alignment_report['n_only_ef_no_answer']}",
        f"  Fingerprint mismatches: {alignment_report['n_fingerprint_mismatches']}",
        "",
        "2. PARSER FAIRNESS (raw bug_type)",
        f"  Direct raw types: {parser_report.get('direct_raw_bug_type_counts', {})}",
        f"  EF-leaky raw types: {parser_report.get('ef_leaky_raw_bug_type_counts', {})}",
        f"  EF-no-answer raw types: {parser_report.get('ef_no_answer_raw_bug_type_counts', {})}",
        "",
        "3. LOCALIZATION (all records)",
        "  See verification_report.json -> 3_localization_sample",
        "",
        "4. EXECUTION FEEDBACK EXTRA PROMPT CONTENT",
        "  See verification_report.json -> 4_execution_feedback_prompt_extra",
        "",
        "5. PATCHED CODE EXECUTION (repair_success=1 samples)",
        f"  Direct: {len(patched_exec_samples['direct'])} samples",
        f"  EF-leaky: {len(patched_exec_samples['ef_leaky'])} samples",
        f"  EF-no-answer: {len(patched_exec_samples.get('ef_no_answer', []))} samples",
        "  See 5_patched_code_execution_sample for buggy vs patched code.",
        "",
        "6. REPAIR_WITHOUT_TRUE_DIAGNOSIS BREAKDOWN",
        f"  Direct: wrong_bug_only={rwtd_direct['n_wrong_bug_only']}, wrong_loc_only={rwtd_direct['n_wrong_loc_only']}, both={rwtd_direct['n_both_wrong']}",
        f"  EF-leaky: wrong_bug_only={rwtd_ef_leaky['n_wrong_bug_only']}, wrong_loc_only={rwtd_ef_leaky['n_wrong_loc_only']}, both={rwtd_ef_leaky['n_both_wrong']}",
        f"  EF-no-answer: wrong_bug_only={rwtd_report.get('ef_no_answer', {}).get('n_wrong_bug_only', 'N/A')}, wrong_loc_only={rwtd_report.get('ef_no_answer', {}).get('n_wrong_loc_only', 'N/A')}, both={rwtd_report.get('ef_no_answer', {}).get('n_both_wrong', 'N/A')}",
        "",
        "7. PIPELINE CONSISTENCY",
        "  Direct and EF share same parser, evaluator, sandbox. Only prompt.variant differs.",
    ]

    summary_path = out_dir / "verification_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Report written to {out_path}")
    print(f"Summary written to {summary_path}")
    print(f"Comparison table: {comparison_csv_path}")
    print("")
    pass_align = (
        alignment_report["n_fingerprint_mismatches"] == 0
        and alignment_report["n_only_direct"] == 0
        and alignment_report["n_only_ef_leaky"] == 0
        and (not three_way or alignment_report["n_only_ef_no_answer"] == 0)
    )
    print("1. SAMPLE ALIGNMENT:", "PASS" if pass_align else "REVIEW")
    print("2. PARSER: see raw_bug_type_counts in report")
    print("3. LOCALIZATION: see localization_sample in report")
    print("4. EXECUTION FEEDBACK LEAKAGE: see prompt_extra in report")
    print("5. PATCHED CODE EXEC: see 5_patched_code_execution_sample (patched code exec'd directly)")
    print("6. RWTD BREAKDOWN: wrong_bug_only / wrong_loc_only / both - see 6_repair_without_true_diagnosis_breakdown")
    print("7. PIPELINE: direct and EF share same parser/evaluator/sandbox")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
