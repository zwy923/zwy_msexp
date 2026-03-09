"""Full experiment runner for the diagnosis pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_exp.datasets import (
    SanitizedMbppFilterConfig,
    filter_sanitized_mbpp_samples,
    load_programming_problem_dataset,
    load_sanitized_mbpp_dataset,
)
from thesis_exp.evaluators import DiagnosisEvaluator, EvaluationConfig, execute_patched_code_safely
from thesis_exp.injectors.registry import create_injector
from thesis_exp.llm import DiagnosisArtifactStore, DiagnosisInferenceEngine, ModelConfig
from thesis_exp.llm.registry import create_adapter
from thesis_exp.parsers import parse_model_diagnosis_output
from thesis_exp.prompts import (
    ExecutionFeedback,
    InputOutputSpecification,
    PromptBuilderOptions,
    PromptContext,
    build_diagnosis_prompt,
)
from thesis_exp.quality import SampleQualityConfig, validate_buggy_sample, validate_transformed_sample
from thesis_exp.schemas.sample import BugInjectionRecord, BuggyProgramSample, EvaluationResult, ModelDiagnosisOutput, ProgrammingProblem, TransformedSample
from thesis_exp.transforms.registry import create_transformer

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class DatasetConfig:
    """Dataset loading configuration."""

    path: str
    format: str = "jsonl"
    max_problems: int | None = None
    run_filtering: bool = False
    filter_output_dir: str = ""
    use_accepted_subset: bool = False


@dataclass(slots=True)
class GenerationConfig:
    """Bug generation and transformation configuration."""

    injector_types: list[str]
    transformation_names: list[str] = field(default_factory=list)
    enable_transformations: bool = True
    include_original_sample: bool = True
    random_seed: int = 0
    max_changed_lines: int = 3
    require_reference_pass_all_tests: bool = True
    require_buggy_fail_some_tests: bool = True
    require_buggy_syntax_valid: bool = True
    require_buggy_executable: bool = True
    require_transformed_behavior_preserved: bool = True


@dataclass(slots=True)
class PromptConfig:
    """Prompt-building configuration."""

    variant: str = "direct_diagnosis"
    include_language_tag: bool = True
    include_output_schema_example: bool = True
    require_json_only_response: bool = True


@dataclass(slots=True)
class ModelRunConfig:
    """One model run specification."""

    provider_name: str
    model_name: str
    api_base_url: str = ""
    api_key_env_var: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    enable_json_output: bool = True
    extra_settings: dict[str, Any] = field(default_factory=dict)

    def to_model_config(self, artifact_root_dir: str) -> ModelConfig:
        """Convert to provider-neutral model config."""

        return ModelConfig(
            provider_name=self.provider_name,
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            api_key_env_var=self.api_key_env_var,
            api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            backoff_initial_seconds=self.backoff_initial_seconds,
            backoff_multiplier=self.backoff_multiplier,
            enable_json_output=self.enable_json_output,
            artifact_root_dir=artifact_root_dir,
            extra_settings=self.extra_settings,
        )


@dataclass(slots=True)
class OutputConfig:
    """Output configuration."""

    results_dir: str
    jsonl_filename: str = "results.jsonl"
    csv_filename: str = "results.csv"
    artifact_root_dir: str = ""
    log_level: str = "INFO"


@dataclass(slots=True)
class RunnerExperimentConfig:
    """Top-level experiment runner configuration."""

    experiment_id: str
    experiment_name: str
    dataset: DatasetConfig
    generation: GenerationConfig
    prompt: PromptConfig
    models: list[ModelRunConfig]
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(results_dir="results"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunnerExperimentConfig":
        """Build config from a plain dictionary."""

        output_payload = dict(payload.get("output", {}))
        output_payload.pop("resume", None)
        return cls(
            experiment_id=payload["experiment_id"],
            experiment_name=payload["experiment_name"],
            dataset=DatasetConfig(**payload["dataset"]),
            generation=GenerationConfig(**payload["generation"]),
            prompt=PromptConfig(**payload["prompt"]),
            models=[ModelRunConfig(**item) for item in payload["models"]],
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
            output=OutputConfig(**output_payload),
        )

    @classmethod
    def from_file(cls, config_path: str) -> "RunnerExperimentConfig":
        """Load config from JSON or YAML."""

        path = Path(config_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML experiment configs.")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        if not isinstance(payload, dict):
            raise ValueError("Experiment config must deserialize to a dictionary.")
        return cls.from_dict(payload)


def _stable_id(*parts: str) -> str:
    joined = "||".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:16]


def _ensure_output_directory(output_config: OutputConfig) -> tuple[Path, Path, Path]:
    results_dir = Path(output_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / output_config.jsonl_filename
    csv_path = results_dir / output_config.csv_filename
    artifact_root_dir = Path(output_config.artifact_root_dir or (results_dir / "artifacts"))
    artifact_root_dir.mkdir(parents=True, exist_ok=True)
    return jsonl_path, csv_path, artifact_root_dir


def load_programming_problems(dataset_config: DatasetConfig) -> list[ProgrammingProblem]:
    """Load a dataset of programming problems from JSON or JSONL."""
    if dataset_config.format == "sanitized_mbpp":
        problems = load_sanitized_mbpp_dataset(dataset_config.path)
    else:
        problems = load_programming_problem_dataset(dataset_config.path, dataset_config.format)

    if dataset_config.max_problems is not None:
        return problems[: dataset_config.max_problems]
    return problems


def _resolve_dataset_for_runner(
    dataset_config: DatasetConfig,
    output_config: OutputConfig,
) -> DatasetConfig:
    """Resolve dataset input, optionally running sanitized MBPP filtering first."""

    if dataset_config.format != "sanitized_mbpp" or not dataset_config.run_filtering:
        return dataset_config

    filter_output_dir = dataset_config.filter_output_dir or str(Path(output_config.results_dir) / "filtered_sanitized_mbpp")
    filter_sanitized_mbpp_samples(
        dataset_config.path,
        filter_output_dir,
        SanitizedMbppFilterConfig(),
    )

    if not dataset_config.use_accepted_subset:
        return dataset_config

    accepted_subset_path = str(Path(filter_output_dir) / "accepted_samples.jsonl")
    return DatasetConfig(
        path=accepted_subset_path,
        format="jsonl",
        max_problems=dataset_config.max_problems,
        run_filtering=False,
        filter_output_dir=filter_output_dir,
        use_accepted_subset=False,
    )


def _extract_line_snippet(source_code: str, changed_lines: list[int]) -> str:
    if not changed_lines:
        return ""
    lines = source_code.splitlines()
    selected = [lines[line_number - 1] for line_number in changed_lines if 1 <= line_number <= len(lines)]
    return "\n".join(selected)


def build_buggy_sample(
    problem: ProgrammingProblem,
    injector_type: str,
) -> BuggyProgramSample | None:
    """Generate one buggy sample using a configured injector."""

    injector = create_injector(injector_type)
    injection_result = injector.inject(
        problem.reference_solution.reference_code,
        function_name=problem.entry_point,
    )
    if injection_result is None:
        return None

    sample_id = f"{problem.problem_id}::{injector_type}::{_stable_id(problem.problem_id, injector_type)}"
    injection_id = f"{sample_id}::injection"
    original_snippet = _extract_line_snippet(problem.reference_solution.reference_code, injection_result.changed_lines)
    buggy_snippet = _extract_line_snippet(injection_result.modified_code, injection_result.changed_lines)

    return BuggyProgramSample(
        sample_id=sample_id,
        problem_id=problem.problem_id,
        programming_language=problem.programming_language,
        problem_statement=problem.problem_statement,
        starter_code=problem.starter_code,
        reference_solution_id=problem.reference_solution.solution_id,
        reference_code=problem.reference_solution.reference_code,
        test_cases=problem.test_cases,
        bug_injection_record=BugInjectionRecord(
            injection_id=injection_id,
            problem_id=problem.problem_id,
            source_solution_id=problem.reference_solution.solution_id,
            bug_type=str(injection_result.bug_type),
            bug_description=injection_result.transformation_description,
            injection_operator_name=injection_result.injector_name,
            injection_line_start=min(injection_result.changed_lines),
            injection_line_end=max(injection_result.changed_lines),
            changed_line_count=len(injection_result.changed_lines),
            original_code_snippet=original_snippet,
            buggy_code_snippet=buggy_snippet,
        ),
        buggy_code=injection_result.modified_code,
        sample_version="original",
        entry_point=problem.entry_point,
    )


def generate_transformed_variants(
    sample: BuggyProgramSample,
    transformation_names: list[str],
    quality_config: SampleQualityConfig,
    logger: logging.Logger | None = None,
) -> list[TransformedSample]:
    """Generate transformed variants for one buggy sample."""

    transformed_samples: list[TransformedSample] = []
    for transformation_name in transformation_names:
        transformer = create_transformer(transformation_name)
        transformation_result = transformer.transform(sample, validate_behavior=True)
        if transformation_result is None:
            continue
        transformation_quality_report = validate_transformed_sample(transformation_result, quality_config)
        if not transformation_quality_report.accepted:
            if logger is not None:
                logger.info(
                    "Rejected transformed sample %s (%s): %s",
                    transformation_result.transformed_sample.transformed_sample_id,
                    transformation_name,
                    "; ".join(transformation_quality_report.failure_reasons),
                )
            continue
        transformed_samples.append(transformation_result.transformed_sample)
    return transformed_samples


def _build_execution_feedback(sample: BuggyProgramSample) -> ExecutionFeedback | None:
    public_test_cases = [test_case for test_case in sample.test_cases if not test_case.is_hidden]
    if not public_test_cases:
        return None

    repair_result = execute_patched_code_safely(
        sample.buggy_code,
        public_test_cases,
        EvaluationConfig(use_hidden_tests_for_repair=False, use_public_tests_for_repair=True),
    )
    if repair_result.total_test_count == 0:
        return None

    failed_tests = [outcome for outcome in repair_result.test_outcomes if not outcome.passed]
    if not failed_tests and not repair_result.execution_error_message:
        return None

    summary = repair_result.execution_error_message or (
        f"{len(failed_tests)} public test(s) failed out of {repair_result.total_test_count}."
    )
    traceback_text = "\n".join(
        f"{outcome.test_case_id}: {outcome.exception_type}: {outcome.exception_message}"
        for outcome in failed_tests
        if outcome.exception_type or outcome.exception_message
    )
    failing_stdout = "\n".join(outcome.stdout_text for outcome in failed_tests if outcome.stdout_text).strip()
    failing_stderr = "\n".join(outcome.stderr_text for outcome in failed_tests if outcome.stderr_text).strip()
    return ExecutionFeedback(
        feedback_summary=summary,
        traceback_text=traceback_text,
        failing_stdout=failing_stdout,
        failing_stderr=failing_stderr,
    )


def _build_prompt_context(
    sample: BuggyProgramSample,
    transformed_sample: TransformedSample | None,
) -> PromptContext:
    buggy_code = transformed_sample.transformed_buggy_code if transformed_sample else sample.buggy_code
    problem_statement = (
        transformed_sample.transformed_problem_statement if transformed_sample else sample.problem_statement
    )
    starter_code = transformed_sample.transformed_starter_code if transformed_sample else sample.starter_code
    reference_code = transformed_sample.transformed_reference_code if transformed_sample else sample.reference_code
    _ = starter_code, reference_code

    failing_test_cases = [test_case.test_code for test_case in sample.test_cases if test_case.test_code and not test_case.is_hidden]
    execution_feedback = _build_execution_feedback(sample)
    return PromptContext(
        sample_id=transformed_sample.transformed_sample_id if transformed_sample else sample.sample_id,
        problem_statement=problem_statement,
        buggy_student_code=buggy_code,
        programming_language=sample.programming_language,
        input_output_specification=InputOutputSpecification(),
        failing_test_cases=failing_test_cases,
        execution_feedback=execution_feedback,
        metadata={
            "sample_id": sample.sample_id,
            "transformed_sample_id": transformed_sample.transformed_sample_id if transformed_sample else "",
        },
    )


def _flatten_result_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                nested_prefix = f"{prefix}.{key}" if prefix else key
                visit(nested_prefix, item)
            return
        if isinstance(value, list):
            flattened[prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            return
        flattened[prefix] = value

    visit("", record)
    return flattened


class IncrementalResultWriter:
    """Append experiment records to JSONL and CSV files."""

    def __init__(self, jsonl_path: Path, csv_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self._csv_header: list[str] | None = None

    def append(self, record: dict[str, Any]) -> None:
        """Append one record to JSONL and CSV outputs."""

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        flat_record = _flatten_result_record(record)
        if self._csv_header is None:
            existing_header = None
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    existing_header = next(reader, None)
            self._csv_header = existing_header or sorted(flat_record)

        missing_columns = [column for column in flat_record if column not in self._csv_header]
        if missing_columns:
            self._csv_header.extend(sorted(missing_columns))
            self._rewrite_csv_with_new_header()

        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_header)
            if not file_exists:
                writer.writeheader()
            writer.writerow({column: flat_record.get(column, "") for column in self._csv_header})

    def _rewrite_csv_with_new_header(self) -> None:
        if not self.csv_path.exists():
            return
        rows: list[dict[str, str]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_header)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in self._csv_header})


def _variant_name(transformed_sample: TransformedSample | None) -> str:
    return transformed_sample.transformation_name if transformed_sample else "original"


def _build_record_id(
    config: RunnerExperimentConfig,
    sample: BuggyProgramSample,
    transformed_sample: TransformedSample | None,
    model_config: ModelRunConfig,
) -> str:
    variant_name = _variant_name(transformed_sample)
    transformed_id = transformed_sample.transformed_sample_id if transformed_sample else "original"
    return "::".join(
        [
            config.experiment_id,
            sample.sample_id,
            transformed_id,
            variant_name,
            model_config.provider_name,
            model_config.model_name,
            config.prompt.variant,
        ]
    )


def _serialize_record(
    *,
    record_id: str,
    config: RunnerExperimentConfig,
    sample: BuggyProgramSample,
    transformed_sample: TransformedSample | None,
    diagnosis_output: Any,
    evaluation_result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "problem_id": sample.problem_id,
        "sample_id": sample.sample_id,
        "transformed_sample_id": transformed_sample.transformed_sample_id if transformed_sample else None,
        "variant_name": _variant_name(transformed_sample),
        "buggy_sample": sample.to_dict(),
        "transformed_sample": transformed_sample.to_dict() if transformed_sample else None,
        "diagnosis_output": diagnosis_output.to_dict(),
        "evaluation_result": evaluation_result.to_dict(),
    }


def _model_group_key(model_run: ModelRunConfig) -> str:
    return f"{model_run.provider_name}::{model_run.model_name}"


class ExperimentRunner:
    """Run the full experiment pipeline with resume support."""

    def __init__(self, config: RunnerExperimentConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"thesis_exp.runner.{config.experiment_id}")
        self._configure_logging()

        jsonl_path, csv_path, artifact_root_dir = _ensure_output_directory(config.output)
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.artifact_root_dir = artifact_root_dir
        self.result_writer = IncrementalResultWriter(jsonl_path, csv_path)
        self.evaluator = DiagnosisEvaluator(config.evaluation)

        adapters = {
            model_run.provider_name: create_adapter(model_run.provider_name)
            for model_run in config.models
        }
        self.inference_engine = DiagnosisInferenceEngine(
            adapters=adapters,
            artifact_store=DiagnosisArtifactStore(str(self.artifact_root_dir)),
        )

    @classmethod
    def from_file(cls, config_path: str) -> "ExperimentRunner":
        """Create a runner from a YAML or JSON config file."""

        return cls(RunnerExperimentConfig.from_file(config_path))

    def run(self) -> list[EvaluationResult]:
        """Run the configured experiment end-to-end."""

        random.seed(self.config.generation.random_seed)
        effective_dataset_config = _resolve_dataset_for_runner(self.config.dataset, self.config.output)
        problems = load_programming_problems(effective_dataset_config)
        self.logger.info("Loaded %s programming problems.", len(problems))
        quality_config = SampleQualityConfig(
            max_changed_lines=self.config.generation.max_changed_lines,
            require_reference_pass_all_tests=self.config.generation.require_reference_pass_all_tests,
            require_buggy_fail_some_tests=self.config.generation.require_buggy_fail_some_tests,
            require_buggy_syntax_valid=self.config.generation.require_buggy_syntax_valid,
            require_buggy_executable=self.config.generation.require_buggy_executable,
            require_transformed_behavior_preserved=self.config.generation.require_transformed_behavior_preserved,
        )

        prompt_options = PromptBuilderOptions(
            include_language_tag=self.config.prompt.include_language_tag,
            include_output_schema_example=self.config.prompt.include_output_schema_example,
            require_json_only_response=self.config.prompt.require_json_only_response,
        )

        all_results: list[EvaluationResult] = []
        total_buggy_samples = 0

        for problem_index, problem in enumerate(problems, start=1):
            self.logger.info("Processing problem %s/%s: %s", problem_index, len(problems), problem.problem_id)
            for injector_type in self.config.generation.injector_types:
                sample = build_buggy_sample(problem, injector_type)
                if sample is None:
                    self.logger.info("Injector '%s' was not applicable for %s.", injector_type, problem.problem_id)
                    continue

                sample_quality_report = validate_buggy_sample(
                    sample,
                    quality_config,
                    self.config.evaluation,
                )
                if not sample_quality_report.accepted:
                    self.logger.info(
                        "Rejected buggy sample %s (%s): %s",
                        sample.sample_id,
                        injector_type,
                        "; ".join(sample_quality_report.failure_reasons),
                    )
                    continue

                total_buggy_samples += 1
                transformed_variants = (
                    generate_transformed_variants(
                        sample,
                        self.config.generation.transformation_names,
                        quality_config,
                        logger=self.logger,
                    )
                    if self.config.generation.enable_transformations
                    else []
                )
                sample_variants: list[TransformedSample | None] = []
                if self.config.generation.include_original_sample:
                    sample_variants.append(None)
                sample_variants.extend(transformed_variants)

                pending_group_records: dict[
                    str,
                    list[tuple[str, TransformedSample | None, ModelDiagnosisOutput, EvaluationResult]],
                ] = {}
                for transformed_sample in sample_variants:
                    prompt_context = _build_prompt_context(sample, transformed_sample)
                    prompt_template = build_diagnosis_prompt(
                        prompt_context,
                        self.config.prompt.variant,
                        options=prompt_options,
                    )

                    for model_run in self.config.models:
                        record_id = _build_record_id(self.config, sample, transformed_sample, model_run)
                        model_config = model_run.to_model_config(str(self.artifact_root_dir))
                        raw_response = self.inference_engine.diagnose(
                            transformed_sample or sample,
                            prompt_template,
                            model_config,
                            run_id=self.config.experiment_id,
                        )
                        diagnosis_output = parse_model_diagnosis_output(
                            raw_response,
                            diagnosis_output_id=f"diag::{record_id}",
                            transformed_sample_id=transformed_sample.transformed_sample_id if transformed_sample else None,
                            response_schema_name=prompt_template.response_schema_name,
                        )
                        evaluation_result = self.evaluator.evaluate_single(sample, diagnosis_output)
                        all_results.append(evaluation_result)
                        pending_group_records.setdefault(_model_group_key(model_run), []).append(
                            (
                                record_id,
                                transformed_sample,
                                diagnosis_output,
                                evaluation_result,
                            )
                        )

                for grouped_records in pending_group_records.values():
                    grouped_results = [item[3] for item in grouped_records]
                    self.evaluator.evaluate_consistency_group(grouped_results)
                    for record_id, transformed_sample, diagnosis_output, evaluation_result in grouped_records:
                        record = _serialize_record(
                            record_id=record_id,
                            config=self.config,
                            sample=sample,
                            transformed_sample=transformed_sample,
                            diagnosis_output=diagnosis_output,
                            evaluation_result=evaluation_result,
                        )
                        self.result_writer.append(record)

        self.logger.info(
            "Experiment finished: %s buggy samples, %s records.",
            total_buggy_samples,
            len(all_results),
        )
        return all_results

    def _configure_logging(self) -> None:
        if self.logger.handlers:
            return
        log_level = getattr(logging, self.config.output.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False
