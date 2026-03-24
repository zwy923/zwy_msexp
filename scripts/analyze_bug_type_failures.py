"""List fine-grained bug_type failures from results.jsonl (dedupe by sample_id)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dedupe(path: Path) -> list[dict]:
    by_sid: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r.get("sample_id") or r.get("buggy_sample", {}).get("sample_id")
        by_sid[sid] = r
    return list(by_sid.values())


def fails(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("evaluation_result", {}).get("bug_type_accuracy", 0) < 1.0]


def main() -> int:
    sets = [
        ("DIRECT", ROOT / "results/sanitized_mbpp_direct_pilot/results.jsonl"),
        ("EF_LEAKY", ROOT / "results/sanitized_mbpp_execution_feedback_pilot/results.jsonl"),
        ("EF_NO_LEAK", ROOT / "results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl"),
    ]
    for label, path in sets:
        if not path.exists():
            print(f"{label}: missing {path}")
            continue
        rows = load_dedupe(path)
        bad = fails(rows)
        conf: Counter[tuple[str, str]] = Counter()
        for r in bad:
            ev = r["evaluation_result"]
            conf[(str(ev["ground_truth_bug_type"]), str(ev.get("predicted_bug_type")))] += 1
        print(f"==== {label} unique={len(rows)} fine_bug_type_fails={len(bad)} ====")
        for (gt, pr), n in sorted(conf.items(), key=lambda x: -x[1]):
            print(f"  {gt:28} -> {pr:28}  {n}")
        print()
        for i, r in enumerate(sorted(bad, key=lambda x: x.get("sample_id", "")), 1):
            ev = r["evaluation_result"]
            diag = r.get("diagnosis_output", {})
            bg = r.get("buggy_sample", {})
            sid = r.get("sample_id") or bg.get("sample_id")
            pid = r.get("problem_id")
            expl = (diag.get("parsed_bug_explanation") or "").replace("\n", " ")[:280]
            print(f"{i:2}. {sid}")
            print(
                f"    problem={pid}  gt={ev['ground_truth_bug_type']}  pred={ev.get('predicted_bug_type')}  "
                f"loc_exact={ev.get('localization_accuracy_exact')}  repair={ev.get('repair_success')}  "
                f"mut_status={ev.get('mutation_adequacy_status', '')}"
            )
            print(f"    expl: {expl}")
        print("\n" + "=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
