#!/usr/bin/env python3
"""Run the full diagnosis pipeline for ONE problem with staged artifacts (real API).

支持：
  - 从 `sanitized-mbpp.json`（与 YAML dataset 一致）读题，或
  - 从 `accepted_samples.jsonl` 读题（`--accepted-jsonl`）。
  - `--all-prompt-variants`：一次写出 direct / EF-leaky / EF-no-leakage 三份完整 prompt。
  - `--api-mode`：`config`（按 YAML 只打一次）、`none`（只写 prompt 不调 API）、`all_three`（三种 prompt 各打一次）。

用法见 `--help` 与仓库 README。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from thesis_exp.common.env import load_env_file  # noqa: E402
from thesis_exp.datasets.loader import (  # noqa: E402
    dataset_record_to_problem,
    validate_dataset_problem_record,
)
from thesis_exp.datasets.mbpp import load_sanitized_mbpp_raw_payloads, validate_sanitized_mbpp_record  # noqa: E402
from thesis_exp.datasets.mbpp_filter import (  # noqa: E402
    SanitizedMbppFilterConfig,
    decide_sanitized_mbpp_problem,
)
from thesis_exp.evaluators import DiagnosisEvaluator  # noqa: E402
from thesis_exp.llm import DiagnosisArtifactStore, DiagnosisInferenceEngine, ModelConfig  # noqa: E402
from thesis_exp.llm.registry import create_adapter  # noqa: E402
from thesis_exp.parsers import parse_model_diagnosis_output, parse_repair_response  # noqa: E402
from thesis_exp.prompts import (  # noqa: E402
    PromptBuilderOptions,
    RepairPromptContext,
    build_diagnosis_prompt,
    build_repair_prompt,
)
from thesis_exp.quality import SampleQualityConfig, validate_buggy_sample  # noqa: E402
from thesis_exp.runners.experiment_runner import (  # noqa: E402
    RunnerExperimentConfig,
    _build_execution_feedback,
    _build_prompt_context,
    build_buggy_sample,
)
from thesis_exp.schemas.sample import BuggyProgramSample, ProgrammingProblem  # noqa: E402

THREE_WAY_VARIANTS: tuple[str, str, str] = (
    "direct_diagnosis",
    "diagnosis_with_execution_feedback",
    "diagnosis_with_execution_feedback_no_leakage",
)

VARIANT_SHORT: dict[str, str] = {
    "direct_diagnosis": "direct",
    "diagnosis_with_execution_feedback": "ef_leaky",
    "diagnosis_with_execution_feedback_no_leakage": "ef_no_leakage",
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _problem_id_to_task_id(problem_id: str) -> str:
    pid = problem_id.strip()
    if pid.startswith("mbpp_"):
        return pid[5:]
    return pid


def _find_raw_payload(dataset_path: Path, problem_id: str) -> dict[str, Any] | None:
    want = problem_id.strip()
    tid = _problem_id_to_task_id(want)
    for payload in load_sanitized_mbpp_raw_payloads(str(dataset_path)):
        p_task = str(payload.get("task_id", "")).strip()
        if p_task == tid or f"mbpp_{p_task}" == want:
            return payload
    return None


def _load_problem_from_sanitized_config(config_path: Path, problem_id: str) -> tuple[ProgrammingProblem, Path]:
    cfg = RunnerExperimentConfig.from_file(str(config_path))
    ds_path = Path(cfg.dataset.path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    if cfg.dataset.format != "sanitized_mbpp":
        raise ValueError("sanitized_mbpp path requires dataset.format == sanitized_mbpp in YAML.")

    from thesis_exp.datasets.mbpp import load_sanitized_mbpp_dataset

    problems = load_sanitized_mbpp_dataset(str(ds_path))
    for p in problems:
        if p.problem_id == problem_id:
            return p, ds_path
    raise KeyError(f"problem_id not in dataset: {problem_id}")


def _find_accepted_line(accepted_path: Path, problem_id: str) -> dict[str, Any]:
    for line in accepted_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("problem_id") == problem_id:
            return rec
    raise KeyError(f"problem_id not found in {accepted_path}: {problem_id}")


def _accepted_line_to_problem(line: dict[str, Any]) -> ProgrammingProblem:
    tests = line.get("tests")
    if not tests and isinstance(line.get("raw_record"), dict):
        tr = line["raw_record"].get("test_list") or []
        ti = line["raw_record"].get("test_imports") or []
        prefix = "\n".join(x for x in ti if isinstance(x, str) and x.strip())
        tests = [f"{prefix}\n{t}" if prefix else t for t in tr if isinstance(t, str)]
    if not isinstance(tests, list) or not tests:
        raise ValueError("accepted_samples line must have non-empty tests (or raw_record.test_list).")

    payload: dict[str, Any] = {
        "problem_id": line["problem_id"],
        "prompt": (line.get("prompt") or "").strip() or "(empty)",
        "entry_point": (line.get("entry_point") or "").strip(),
        "reference_code": (line.get("reference_code") or "").strip(),
        "tests": tests,
        "programming_language": line.get("programming_language") or "python",
        "starter_code": line.get("starter_code") or "",
        "problem_title": line.get("problem_title") or line["problem_id"],
        "difficulty": line.get("difficulty") or "",
        "topic": line.get("topic") or "",
        "source_dataset_name": line.get("source_dataset_name") or "",
        "source_split_name": line.get("source_split_name") or "",
    }
    if not payload["entry_point"] or not payload["reference_code"]:
        raise ValueError("accepted_samples line missing entry_point or reference_code.")
    return dataset_record_to_problem(validate_dataset_problem_record(payload))


def _filter_decision_from_accepted_or_raw(
    accepted_line: dict[str, Any] | None,
    raw_payload: dict[str, Any] | None,
    filter_cfg: SanitizedMbppFilterConfig,
) -> Any | None:
    if raw_payload is not None:
        normalized = validate_sanitized_mbpp_record(raw_payload)
        return decide_sanitized_mbpp_problem(normalized, filter_cfg)
    if accepted_line and isinstance(accepted_line.get("raw_record"), dict):
        normalized = validate_sanitized_mbpp_record(accepted_line["raw_record"])
        return decide_sanitized_mbpp_problem(normalized, filter_cfg)
    return None


def _forbid_mock_provider(models: list[Any], allow_mock: bool) -> None:
    for m in models:
        pn = getattr(m, "provider_name", "")
        if pn == "mock" and not allow_mock:
            raise SystemExit(
                "Config uses provider 'mock' (no real API). Use openai_compatible or pass --allow-mock for dry runs."
            )


def _write_prompt_bundle(
    out_dir: Path,
    prompt_context: Any,
    prompt_options: PromptBuilderOptions,
    chosen: BuggyProgramSample,
    *,
    all_variants: bool,
    config_variant: str,
) -> None:
    if all_variants:
        prompts_dir = out_dir / "06_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, str]] = []
        for v in THREE_WAY_VARIANTS:
            tmpl = build_diagnosis_prompt(prompt_context, v, options=prompt_options)
            rendered = tmpl.render(chosen)
            fname = f"{v}.json"
            _write_json(
                prompts_dir / fname,
                {
                    "prompt_variant": v,
                    "short_label": VARIANT_SHORT[v],
                    "template_name": tmpl.template_name,
                    "response_schema_name": tmpl.response_schema_name,
                    "system_prompt": rendered["system_prompt"],
                    "user_prompt": rendered["user_prompt"],
                },
            )
            index.append({"variant": v, "short_label": VARIANT_SHORT[v], "file": fname})
        _write_json(
            prompts_dir / "index.json",
            {
                "description": "direct = no failing tests in prompt; ef_leaky = full asserts + traceback; ef_no_leakage = sanitized feedback",
                "variants": index,
            },
        )

    tmpl_cfg = build_diagnosis_prompt(prompt_context, config_variant, options=prompt_options)
    rendered_cfg = tmpl_cfg.render(chosen)
    _write_json(
        out_dir / "06_prompt_diagnosis.json",
        {
            "prompt_variant": config_variant,
            "note": "matches YAML prompt.variant (for api_mode=config)",
            "template_name": tmpl_cfg.template_name,
            "response_schema_name": tmpl_cfg.response_schema_name,
            "system_prompt": rendered_cfg["system_prompt"],
            "user_prompt": rendered_cfg["user_prompt"],
        },
    )


def _run_one_diagnosis_call(
    *,
    engine: DiagnosisInferenceEngine,
    chosen: BuggyProgramSample,
    variant: str,
    prompt_options: PromptBuilderOptions,
    prompt_context: Any,
    run_id_suffix: str,
    exp_cfg: RunnerExperimentConfig,
    model_run: Any,
    llm_root: Path,
) -> tuple[Any, str]:
    prompt_template = build_diagnosis_prompt(prompt_context, variant, options=prompt_options)
    model_config = model_run.to_model_config(str(llm_root))
    record_id = f"{exp_cfg.experiment_id}::{chosen.sample_id}::{model_run.provider_name}::{model_run.model_name}::{variant}"
    raw_response = engine.diagnose(chosen, prompt_template, model_config, run_id=run_id_suffix)
    diagnosis_output = parse_model_diagnosis_output(
        raw_response,
        diagnosis_output_id=f"diag::{record_id}",
        response_schema_name=prompt_template.response_schema_name,
    )
    return diagnosis_output, record_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Full single-problem pipeline with staged artifacts (real API).")
    parser.add_argument("--config", required=True, help="Experiment YAML/JSON (same as run_experiment.py).")
    parser.add_argument("--problem-id", required=True, help="e.g. mbpp_103")
    parser.add_argument("--out-dir", required=True, help="Directory to write all artifacts.")
    parser.add_argument(
        "--accepted-jsonl",
        default="",
        help="Load problem from this accepted_samples.jsonl (must contain problem_id). Overrides YAML dataset path.",
    )
    parser.add_argument(
        "--sanitized-json",
        default="",
        help="Optional sanitized-mbpp.json path for 03a raw MBPP row + filter. Default: try ./sanitized-mbpp.json then config dataset.path.",
    )
    parser.add_argument(
        "--injector",
        default="",
        help="Force one injector type; default: first applicable + quality-pass from config list.",
    )
    parser.add_argument("--run-id", default="", help="LLM artifact run_id prefix (default: experiment_id + _single_trace).")
    parser.add_argument("--allow-mock", action="store_true", help="Allow provider mock (no network).")
    parser.add_argument(
        "--all-prompt-variants",
        action="store_true",
        help="Write 06_prompts/ with direct_diagnosis, diagnosis_with_execution_feedback, diagnosis_with_execution_feedback_no_leakage.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("config", "none", "all_three"),
        default="config",
        help="config: one call using YAML prompt.variant; none: skip LLM; all_three: three calls (three-way prompts).",
    )
    args = parser.parse_args()

    load_env_file(_ROOT)

    config_path = Path(args.config)
    exp_cfg = RunnerExperimentConfig.from_file(str(config_path))
    _forbid_mock_provider(exp_cfg.models, args.allow_mock)

    pv = exp_cfg.prompt.variant
    if args.api_mode == "all_three" and pv not in THREE_WAY_VARIANTS:
        raise SystemExit(
            f"api_mode=all_three only runs the 3-way direct/EF-leaky/EF-no-leakage prompts; YAML has variant={pv!r}. "
            "Use api_mode=config for diagnosis_only / diagnosis_then_repair."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id_base = args.run_id.strip() or f"{exp_cfg.experiment_id}_single_trace"
    llm_root = out_dir / "08_llm_artifacts"
    llm_root.mkdir(parents=True, exist_ok=True)

    problem_id = args.problem_id.strip()
    accepted_line: dict[str, Any] | None = None
    ds_path: Path | None = None

    if args.accepted_jsonl.strip():
        ap = Path(args.accepted_jsonl.strip())
        if not ap.is_file():
            raise FileNotFoundError(ap)
        accepted_line = _find_accepted_line(ap, problem_id)
        problem = _accepted_line_to_problem(accepted_line)
        manifest_dataset = str(ap.resolve())
    else:
        problem, ds_path = _load_problem_from_sanitized_config(config_path, problem_id)
        manifest_dataset = str(ds_path.resolve())

    # 03a raw MBPP + 03b filter
    sanitized_path: Path | None = None
    if args.sanitized_json.strip():
        sanitized_path = Path(args.sanitized_json.strip())
    elif ds_path is not None:
        sanitized_path = ds_path
    else:
        cand = _ROOT / "sanitized-mbpp.json"
        sanitized_path = cand if cand.is_file() else None

    raw_payload: dict[str, Any] | None = None
    if sanitized_path and sanitized_path.is_file():
        raw_payload = _find_raw_payload(sanitized_path, problem.problem_id)

    if raw_payload is not None:
        _write_json(out_dir / "03a_sanitized_raw_record.json", raw_payload)
        _write_json(out_dir / "03a_source.json", {"source": "sanitized_mbpp", "path": str(sanitized_path.resolve())})
    elif accepted_line is not None:
        _write_json(out_dir / "03a_sanitized_raw_record.json", accepted_line.get("raw_record") or {})
        _write_json(
            out_dir / "03a_source.json",
            {"source": "accepted_jsonl", "path": str(Path(args.accepted_jsonl).resolve())},
        )
    else:
        _write_json(
            out_dir / "03a_source.json",
            {"source": "none", "note": "No sanitized json found; filter decision may be missing."},
        )

    filter_cfg = SanitizedMbppFilterConfig(evaluation_config=exp_cfg.evaluation)
    decision = _filter_decision_from_accepted_or_raw(accepted_line, raw_payload, filter_cfg)
    if decision is not None:
        _write_json(
            out_dir / "03b_filter_decision.json",
            {
                "accepted": decision.accepted,
                "rejection_reasons": decision.rejection_reasons,
                "filter_metadata": asdict(decision.filter_metadata),
                "normalized_record": asdict(decision.normalized_record) if decision.normalized_record else {},
            },
        )
    else:
        _write_json(
            out_dir / "03b_filter_decision.json",
            {"note": "Could not compute filter decision (missing raw MBPP / raw_record)."},
        )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "dataset_source": "accepted_jsonl" if accepted_line else "sanitized_mbpp",
        "dataset_path": manifest_dataset,
        "problem_id": problem.problem_id,
        "run_id_base": run_id_base,
        "llm_artifact_root": str(llm_root.resolve()),
        "injector_requested": args.injector or None,
        "all_prompt_variants": args.all_prompt_variants,
        "api_mode": args.api_mode,
    }
    _write_json(out_dir / "00_manifest.json", manifest)

    def cfg_to_dict() -> dict[str, Any]:
        return {
            "experiment_id": exp_cfg.experiment_id,
            "experiment_name": exp_cfg.experiment_name,
            "dataset": asdict(exp_cfg.dataset),
            "generation": asdict(exp_cfg.generation),
            "prompt": asdict(exp_cfg.prompt),
            "models": [asdict(m) for m in exp_cfg.models],
            "evaluation": asdict(exp_cfg.evaluation),
            "output": asdict(exp_cfg.output),
        }

    _write_json(out_dir / "01_experiment_config.json", cfg_to_dict())
    _write_json(out_dir / "02_programming_problem.json", problem.to_dict())

    injector_list = (
        [args.injector.strip()]
        if args.injector.strip()
        else list(exp_cfg.generation.injector_types)
    )

    quality_cfg = SampleQualityConfig(
        max_changed_lines=exp_cfg.generation.max_changed_lines,
        require_reference_pass_all_tests=exp_cfg.generation.require_reference_pass_all_tests,
        require_buggy_fail_some_tests=exp_cfg.generation.require_buggy_fail_some_tests,
        require_buggy_syntax_valid=exp_cfg.generation.require_buggy_syntax_valid,
        require_buggy_executable=exp_cfg.generation.require_buggy_executable,
    )
    prompt_options = PromptBuilderOptions(
        include_language_tag=exp_cfg.prompt.include_language_tag,
        include_output_schema_example=exp_cfg.prompt.include_output_schema_example,
        require_json_only_response=exp_cfg.prompt.require_json_only_response,
    )

    attempts_path = out_dir / "04_injection_attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    chosen: BuggyProgramSample | None = None
    chosen_injector: str | None = None

    with attempts_path.open("w", encoding="utf-8") as att_f:
        for inj in injector_list:
            sample = build_buggy_sample(problem, inj)
            row: dict[str, Any] = {
                "injector_type": inj,
                "injector_applicable": sample is not None,
            }
            if sample is None:
                att_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                continue
            report = validate_buggy_sample(sample, quality_cfg, exp_cfg.evaluation)
            row["quality_accepted"] = report.accepted
            row["quality_failure_reasons"] = report.failure_reasons
            row["sample_id"] = sample.sample_id
            if report.accepted:
                row["bug_type"] = sample.bug_injection_record.bug_type
                row["injection_lines"] = [
                    sample.bug_injection_record.injection_line_start,
                    sample.bug_injection_record.injection_line_end,
                ]
            att_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if report.accepted and chosen is None:
                chosen = sample
                chosen_injector = inj

    if chosen is None:
        logging.error("No injector produced an accepted buggy sample. See 04_injection_attempts.jsonl")
        manifest["status"] = "failed_no_sample"
        _write_json(out_dir / "00_manifest.json", manifest)
        return 2

    manifest["sample_id"] = chosen.sample_id
    manifest["injector_used"] = chosen_injector
    _write_json(out_dir / "00_manifest.json", manifest)
    _write_json(out_dir / "05_buggy_sample.json", chosen.to_dict())
    q_report = validate_buggy_sample(chosen, quality_cfg, exp_cfg.evaluation)
    _write_json(out_dir / "05b_sample_quality.json", asdict(q_report))

    prompt_context = _build_prompt_context(chosen)
    _write_prompt_bundle(
        out_dir,
        prompt_context,
        prompt_options,
        chosen,
        all_variants=args.all_prompt_variants,
        config_variant=pv,
    )

    ef = _build_execution_feedback(chosen)
    _write_json(
        out_dir / "06b_execution_feedback.json",
        {
            "feedback_summary": ef.feedback_summary if ef else None,
            "traceback_text": ef.traceback_text if ef else None,
            "failing_stdout": ef.failing_stdout if ef else None,
            "failing_stderr": ef.failing_stderr if ef else None,
        }
        if ef
        else None,
    )

    model_run = exp_cfg.models[0]
    if args.api_mode != "none":
        model_config = model_run.to_model_config(str(llm_root))
        if not model_config.resolve_api_key() and model_run.api_key_env_var:
            raise SystemExit(
                f"Missing API key: set environment variable {model_run.api_key_env_var} or put it in .env"
            )

    adapters = {model_run.provider_name: create_adapter(model_run.provider_name)}
    engine = DiagnosisInferenceEngine(
        adapters=adapters,
        artifact_store=DiagnosisArtifactStore(str(llm_root)),
    )

    evaluator = DiagnosisEvaluator(exp_cfg.evaluation)
    eval_dir = out_dir / "10_evaluations"
    diag_dir = out_dir / "09_diagnoses"

    if args.api_mode == "none":
        manifest["status"] = "ok_prompts_only"
        _write_json(out_dir / "00_manifest.json", manifest)
        print(f"Done (api_mode=none). Artifacts: {out_dir.resolve()}")
        return 0

    if args.api_mode == "config":
        if pv == "diagnosis_then_repair":
            # keep previous behavior: use yaml variant for first stage then repair
            diagnosis_output, record_id = _run_one_diagnosis_call(
                engine=engine,
                chosen=chosen,
                variant=pv,
                prompt_options=prompt_options,
                prompt_context=prompt_context,
                run_id_suffix=run_id_base,
                exp_cfg=exp_cfg,
                model_run=model_run,
                llm_root=llm_root,
            )
            repair_context = RepairPromptContext(
                sample_id=prompt_context.sample_id,
                problem_statement=prompt_context.problem_statement,
                buggy_code=prompt_context.buggy_student_code,
                diagnosis_bug_type=diagnosis_output.parsed_bug_type or "unknown",
                diagnosis_bug_line=diagnosis_output.parsed_bug_line_start,
                diagnosis_explanation=diagnosis_output.parsed_bug_explanation or "",
                programming_language=chosen.programming_language,
            )
            repair_prompt = build_repair_prompt(repair_context, options=prompt_options)
            repair_response = engine.diagnose(
                repair_context,
                repair_prompt,
                model_run.to_model_config(str(llm_root)),
                run_id=run_id_base,
            )
            patched_code, repair_error = parse_repair_response(repair_response.response_text)
            diagnosis_output.parsed_repaired_code = patched_code
            if repair_error:
                diagnosis_output.parsing_error_message = (
                    f"{diagnosis_output.parsing_error_message} | repair: {repair_error}"
                ).strip(" |")
            _write_json(
                out_dir / "11_repair_prompt.json",
                {
                    "template_name": repair_prompt.template_name,
                    "system_prompt": repair_prompt.render(repair_context)["system_prompt"],
                    "user_prompt": repair_prompt.render(repair_context)["user_prompt"],
                },
            )
            _write_json(
                out_dir / "11b_repair_raw_response.json",
                {
                    "response_text": repair_response.response_text,
                    "artifact_directory": repair_response.artifact_directory,
                },
            )
        else:
            diagnosis_output, record_id = _run_one_diagnosis_call(
                engine=engine,
                chosen=chosen,
                variant=pv,
                prompt_options=prompt_options,
                prompt_context=prompt_context,
                run_id_suffix=run_id_base,
                exp_cfg=exp_cfg,
                model_run=model_run,
                llm_root=llm_root,
            )

        diag_dir.mkdir(parents=True, exist_ok=True)
        _write_json(diag_dir / f"{pv}.json", diagnosis_output.to_dict())
        _write_json(out_dir / "09_parsed_diagnosis.json", diagnosis_output.to_dict())
        evaluation = evaluator.evaluate_single(chosen, diagnosis_output)
        eval_dir.mkdir(parents=True, exist_ok=True)
        _write_json(eval_dir / f"{pv}.json", evaluation.to_dict())
        _write_json(out_dir / "10_evaluation.json", evaluation.to_dict())
        manifest["evaluation_notes"] = evaluation.evaluation_notes

    else:
        # all_three
        diag_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if pv == "diagnosis_then_repair":
            raise SystemExit("api_mode=all_three is incompatible with diagnosis_then_repair; use api_mode=config.")
        notes_list: list[str] = []
        for v in THREE_WAY_VARIANTS:
            rid = f"{run_id_base}__{VARIANT_SHORT[v]}"
            diagnosis_output, _ = _run_one_diagnosis_call(
                engine=engine,
                chosen=chosen,
                variant=v,
                prompt_options=prompt_options,
                prompt_context=prompt_context,
                run_id_suffix=rid,
                exp_cfg=exp_cfg,
                model_run=model_run,
                llm_root=llm_root,
            )
            short = VARIANT_SHORT[v]
            _write_json(diag_dir / f"{short}.json", diagnosis_output.to_dict())
            evaluation = evaluator.evaluate_single(chosen, diagnosis_output)
            _write_json(eval_dir / f"{short}.json", evaluation.to_dict())
            notes_list.append(f"{short}: {evaluation.evaluation_notes}")
        # Legacy single files: mirror YAML variant if it is one of the three
        if pv in THREE_WAY_VARIANTS:
            mid = VARIANT_SHORT[pv]
            _write_json(
                out_dir / "09_parsed_diagnosis.json",
                json.loads((diag_dir / f"{mid}.json").read_text(encoding="utf-8")),
            )
            _write_json(
                out_dir / "10_evaluation.json",
                json.loads((eval_dir / f"{mid}.json").read_text(encoding="utf-8")),
            )
        manifest["evaluation_notes"] = " | ".join(notes_list)

    readme = out_dir / "08_llm_artifacts_README.txt"
    readme_lines = [
        f"Diagnosis request/response under: {llm_root}",
        f"run_id_base = {run_id_base}",
        f"sample_id = {chosen.sample_id}",
        f"provider = {model_run.provider_name}",
        f"model = {model_run.model_name}",
        "",
    ]
    if args.api_mode == "all_three":
        for v in THREE_WAY_VARIANTS:
            readme_lines.append(
                f"  {VARIANT_SHORT[v]}: .../{run_id_base}__{VARIANT_SHORT[v]}/"
                f"{model_run.provider_name}/{model_run.model_name}/{chosen.sample_id}/"
            )
    else:
        readme_lines.append(
            f"  .../{run_id_base}/{model_run.provider_name}/{model_run.model_name}/{chosen.sample_id}/"
        )
    readme_lines.append("Files: request.json, response.json, raw_response.txt")
    readme.write_text("\n".join(readme_lines), encoding="utf-8")

    manifest["status"] = "ok"
    _write_json(out_dir / "00_manifest.json", manifest)
    print(f"Done. Artifacts: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
