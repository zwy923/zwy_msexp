from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from thesis_exp.datasets.loader import (  # noqa: E402
    dataset_record_to_problem,
    validate_dataset_problem_record,
)
from thesis_exp.datasets.mbpp import (  # noqa: E402
    load_sanitized_mbpp_raw_payloads,
    validate_sanitized_mbpp_record,
)
from thesis_exp.evaluators.diagnosis_evaluator import (  # noqa: E402
    EvaluationConfig,
    execute_patched_code_safely,
)
from thesis_exp.evaluators.mutation import _collect_mutants  # noqa: E402
from thesis_exp.schemas.sample import ProgrammingProblem  # noqa: E402


@dataclass(frozen=True, slots=True)
class Row:
    problem_id: str
    n_tests: int
    n_code_lines: int
    ref_ok: bool
    n_mutants: int
    killed: int
    kill_rate: str
    risk: str
    note: str


def _load_problems(dataset_path: str, fmt: str) -> list[ProgrammingProblem]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(dataset_path)

    if fmt == "sanitized_mbpp":
        payloads = load_sanitized_mbpp_raw_payloads(str(path))
        return [
            dataset_record_to_problem(validate_sanitized_mbpp_record(p)) for p in payloads
        ]

    if fmt == "jsonl":
        problems: list[ProgrammingProblem] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rec = validate_dataset_problem_record(payload)
            problems.append(dataset_record_to_problem(rec))
        return problems

    raise ValueError(f"Unknown format: {fmt}")


def _analyze_one(
    problem: ProgrammingProblem,
    eval_cfg: EvaluationConfig,
    max_mutants: int,
) -> Row:
    ref = problem.reference_solution.reference_code
    tests = problem.test_cases
    n_tests = len(tests)
    n_lines = len(ref.splitlines())

    ref_exec = execute_patched_code_safely(ref, tests, eval_cfg)
    ref_ok = (
        ref_exec.syntax_valid
        and not ref_exec.timed_out
        and ref_exec.total_test_count > 0
        and ref_exec.passed_test_count == ref_exec.total_test_count
    )

    if not ref_ok:
        return Row(
            problem_id=problem.problem_id,
            n_tests=n_tests,
            n_code_lines=n_lines,
            ref_ok=False,
            n_mutants=0,
            killed=0,
            kill_rate="n/a",
            risk="skip",
            note="reference did not pass all tests under eval config",
        )

    mutants = _collect_mutants(ref, max_mutants)
    n_mut = len(mutants)
    if n_mut == 0:
        return Row(
            problem_id=problem.problem_id,
            n_tests=n_tests,
            n_code_lines=n_lines,
            ref_ok=True,
            n_mutants=0,
            killed=0,
            kill_rate="n/a",
            risk="unknown",
            note="no Compare/BinOp/BoolOp mutants generated (code may be too simple for this mutator)",
        )

    killed = 0
    for msrc in mutants:
        r = execute_patched_code_safely(msrc, tests, eval_cfg)
        dead = (
            r.syntax_valid
            and not r.timed_out
            and r.total_test_count > 0
            and r.passed_test_count < r.total_test_count
        )
        if dead:
            killed += 1

    rate = killed / n_mut
    kill_rate_str = f"{rate:.2f}"

    # 启发式分级（可调）
    if killed == 0:
        risk = "high"
        note = "tests kill no sampled mutants — very thin oracle for these local faults"
    elif rate < 0.25:
        risk = "medium"
        note = "low kill rate; many wrong nearby programs may still pass asserts"
    else:
        risk = "low"
        note = "mutants often fail asserts (necessary but not sufficient for semantic correctness)"

    return Row(
        problem_id=problem.problem_id,
        n_tests=n_tests,
        n_code_lines=n_lines,
        ref_ok=ref_ok,
        n_mutants=n_mut,
        killed=killed,
        kill_rate=kill_rate_str,
        risk=risk,
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot-check reference solutions vs AST mutants (thin-test heuristic).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sanitized-mbpp.json",
        help="Path to sanitized_mbpp JSON or experiment jsonl (e.g. accepted_samples.jsonl).",
    )
    parser.add_argument(
        "--format",
        choices=("sanitized_mbpp", "jsonl"),
        default="sanitized_mbpp",
        help="sanitized_mbpp: single JSON array file; jsonl: one problem dict per line.",
    )
    parser.add_argument("--sample", type=int, default=12, help="Number of problems to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument(
        "--max-mutants",
        type=int,
        default=16,
        help="Cap on mutants per problem (same family as evaluation mutation adequacy).",
    )
    parser.add_argument(
        "--repair-timeout",
        type=float,
        default=3.0,
        help="Per-execution timeout (seconds), aligned with typical eval.",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default="",
        help="Comma-separated problem_id list; if set, ignores --sample and analyzes exactly these.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional path to write CSV rows.",
    )
    args = parser.parse_args()

    eval_cfg = EvaluationConfig(
        repair_timeout_seconds=args.repair_timeout,
        use_hidden_tests_for_repair=True,
        use_public_tests_for_repair=True,
    )

    all_problems = _load_problems(args.dataset, args.format)
    if args.only_ids.strip():
        want = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        selected = [p for p in all_problems if p.problem_id in want]
        missing = want - {p.problem_id for p in selected}
        if missing:
            print("WARNING: problem_id not found:", ", ".join(sorted(missing)), file=sys.stderr)
    else:
        rng = random.Random(args.seed)
        k = min(args.sample, len(all_problems))
        selected = rng.sample(all_problems, k=k)

    rows = [_analyze_one(p, eval_cfg, args.max_mutants) for p in selected]

    # 简要汇总
    by_risk: dict[str, int] = {}
    for r in rows:
        by_risk[r.risk] = by_risk.get(r.risk, 0) + 1

    print("=== spotcheck_reference_test_thinness ===")
    print(f"dataset={args.dataset} format={args.format} analyzed={len(rows)}")
    print(f"max_mutants={args.max_mutants} timeout={args.repair_timeout}s")
    print("risk counts:", json.dumps(by_risk, ensure_ascii=False, sort_keys=True))
    print()

    headers = [
        "problem_id",
        "n_tests",
        "n_lines",
        "ref_ok",
        "n_mut",
        "killed",
        "kill_rate",
        "risk",
        "note",
    ]
    col_widths = [14, 7, 7, 6, 5, 6, 8, 8, 50]

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, c in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 20
            parts.append(str(c)[:w].ljust(w))
        return " ".join(parts)

    print(fmt_row(headers))
    print("-" * 120)
    for r in rows:
        print(
            fmt_row(
                [
                    r.problem_id,
                    str(r.n_tests),
                    str(r.n_code_lines),
                    str(r.ref_ok),
                    str(r.n_mutants),
                    str(r.killed),
                    r.kill_rate,
                    r.risk,
                    r.note,
                ]
            )
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow(
                    [
                        r.problem_id,
                        r.n_tests,
                        r.n_code_lines,
                        r.ref_ok,
                        r.n_mutants,
                        r.killed,
                        r.kill_rate,
                        r.risk,
                        r.note,
                    ]
                )
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
