#!/usr/bin/env python3
"""Print detailed rows where fine bug_type does not match (deduped by sample_id)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-jsonl",
        default="results/sanitized_mbpp_direct_pilot/results.jsonl",
        help="Experiment results.jsonl",
    )
    args = ap.parse_args()
    p = _ROOT / args.results_jsonl
    if not p.is_file():
        print(f"Not found: {p}", file=sys.stderr)
        return 1

    by: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r.get("sample_id") or r["buggy_sample"]["sample_id"]
        by[sid] = r

    fails = [r for r in by.values() if r.get("evaluation_result", {}).get("bug_type_accuracy", 0) < 1]
    print(f"fine bug_type fails: {len(fails)} of {len(by)} (deduped)\n")
    print("=" * 100)

    for i, r in enumerate(sorted(fails, key=lambda x: x.get("sample_id", "")), 1):
        ev = r["evaluation_result"]
        bg = r["buggy_sample"]
        inj = bg.get("bug_injection_record", {})
        diag = r.get("diagnosis_output", {})
        print(f"### {i}. {r.get('sample_id')}")
        print(f"problem_id: {r.get('problem_id')}  |  gt: {ev.get('ground_truth_bug_type')}  |  pred: {ev.get('predicted_bug_type')}")
        print(f"loc_exact: {ev.get('localization_accuracy_exact')}  repair: {ev.get('repair_success')}  mut: {ev.get('mutation_adequacy_status', '')}")
        print(f"injector: {inj.get('injection_operator_name', '')}  lines {inj.get('injection_line_start')}-{inj.get('injection_line_end')}")
        og = inj.get("original_code_snippet", "") or ""
        bg_snip = inj.get("buggy_code_snippet", "") or ""
        print(f"GT snippet:   {og[:140]!r}")
        print(f"Buggy snip:   {bg_snip[:140]!r}")
        expl = (diag.get("parsed_bug_explanation") or "").replace("\n", " ")
        print(f"model says:   {expl[:600]}{'...' if len(expl) > 600 else ''}")
        print("-" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
