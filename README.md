# Thesis Experiment: LLM Bug Diagnosis & Hallucination Evaluation

Evaluates hallucinated bug diagnosis in LLM feedback for programming education. Compares three conditions: direct diagnosis, execution feedback with answer leakage (EF-leaky), and execution feedback without expected outputs (EF-no-answer).

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

**Run full experiment** (direct + EF-leaky + EF-no-answer, then summarize + verify):

```bash
# Any OS (recommended)
set PYTHONPATH=src
python scripts/run_three_way_experiments.py

# Wrappers (same as above)
.\scripts\run_formal_experiments.ps1   # Windows
./scripts/run_formal_experiments.sh    # Linux / macOS
```

Options: `--no-clean`, `--skip-summarize`, `--skip-verify`.

**Run single config:**

```bash
python scripts/run_experiment.py --config configs/sanitized_mbpp_direct_pilot.yaml
```

**Single problem — full pipeline + staged artifacts (real API):**

Requires the same API key as in YAML (e.g. `OPENAI_API_KEY`) when calling the model. Writes filter metadata, injection attempts, prompts, `08_llm_artifacts/…/request.json` & `response.json`, parsed diagnosis, and evaluation JSON.

```bash
set PYTHONPATH=src
python scripts/run_single_problem_pipeline.py ^
  --config configs/sanitized_mbpp_direct_pilot.yaml ^
  --problem-id mbpp_103 ^
  --out-dir results/single_problem_trace/mbpp_103 ^
  --injector condition_inversion
```

**Load from `accepted_samples.jsonl`** (ignores YAML `dataset.path` for the problem body; still uses YAML for models, injectors, evaluation):

```bash
python scripts/run_single_problem_pipeline.py ^
  --config configs/sanitized_mbpp_direct_pilot.yaml ^
  --problem-id mbpp_103 ^
  --accepted-jsonl data/filtered/sanitized_mbpp_direct_pilot/accepted_samples.jsonl ^
  --out-dir results/single_problem_trace/mbpp_103_acc
```

**Write all three prompt variants** (direct / EF-leaky / EF-no-leakage) under `06_prompts/`:

```bash
python scripts/run_single_problem_pipeline.py ... --all-prompt-variants
```

**API modes:** `--api-mode config` (default: one call using YAML `prompt.variant`), `--api-mode none` (only artifacts + prompts, no HTTP), `--api-mode all_three` (three LLM calls for the three-way comparison; incompatible with `diagnosis_then_repair` / `diagnosis_only` in YAML).

Optional `--sanitized-json path/to/sanitized-mbpp.json` for `03a`/`03b` when using `--accepted-jsonl`.

Use `--allow-mock` only for offline debugging (uses `mock` provider in config).

**Summarize results:**

```bash
python scripts/summarize_results.py \
  --results-jsonl results/sanitized_mbpp_direct_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl \
  --output-dir results/three_way_comparison
```

## Reference vs thin tests (spot-check)

To estimate whether official asserts catch simple wrong variants of the reference (AST mutants aligned with `mutation.py`), run:

```bash
set PYTHONPATH=src
python scripts/spotcheck_reference_test_thinness.py --dataset sanitized-mbpp.json --format sanitized_mbpp --sample 15 --seed 0
```

Target specific `problem_id`s (e.g. `mbpp_605`):

```bash
python scripts/spotcheck_reference_test_thinness.py --dataset sanitized-mbpp.json --format sanitized_mbpp --only-ids mbpp_605,mbpp_103
```

This does **not** prove semantic correctness; high `kill_rate` is only a weak sanity check. `risk=high` means sampled mutants all passed tests — strong sign of an overly thin oracle for those fault shapes.

## Data

- **Input:** `sanitized-mbpp.json` (sanitized MBPP benchmark)
- **Bug types (fine-grained):** `loop_boundary_error` (from injectors `off_by_one` / `wrong_loop_bound`), `accumulator_init_error`, `conditional_logic_error` (from injectors `condition_inversion` / `wrong_comparison_operator`), `premature_return` (see `configs/sanitized_mbpp.common.yaml` → `generation.injector_types`)
- **Config:** Shared defaults in `configs/sanitized_mbpp.common.yaml`; each `sanitized_mbpp_*_pilot.yaml` sets `extends`, `experiment_id`, `prompt.variant`, and `output` paths. Edit `max_problems` etc. in the common file.
- **Re-run filtering only** (writes `dataset.filter_output_dir` + `data/sanitized_mbpp_filter_latest`, same `evaluation` as YAML):

  ```bash
  set PYTHONPATH=src
  python scripts/rerun_mbpp_filter.py
  ```

## Pipeline

1. Load problems → filter → inject bugs

**Unified reference validation:** Whether the reference solution “passes tests” uses one code path: `validate_reference_solution` → `execute_patched_code_safely` with `EvaluationConfig` (same test subset as `select_tests_for_evaluation`, same subprocess sandbox as repair/quality). Sanitized-MBPP **filter**, **`validate_buggy_sample`**, and **batch runner** therefore agree when they share the same `evaluation` settings. Standalone `mbpp_filter` CLI keeps `evaluation_config=None` and derives timeout from `execution_timeout_seconds`; experiment **`run_filtering`** passes the YAML `evaluation` block into `SanitizedMbppFilterConfig(evaluation_config=...)`.
2. Build prompts (direct / EF-leaky / EF-no-answer)
3. Call LLM → parse JSON diagnosis
4. Evaluate: bug_type_accuracy (fine), bug_type_accuracy_coarse, localization_accuracy (see `evaluation.localization_tolerance_lines`; default ±1 line around the injected span so “condition line” vs adjacent “effect” line still counts), **localization_accuracy_exact** (tolerance 0), **patch_passes_selected_tests**, **repair_success** (strict: tests + mutation gate when enabled), diagnosis_hallucination_rate, narrative_hallucination_rate
5. Summarize → tables, figures, confusion matrices

## Outputs

| Path | Contents |
|------|----------|
| `results/<experiment_id>/results.jsonl` | Raw records |
| `results/<experiment_id>/results.csv` | Flattened CSV |
| `results/three_way_comparison/` | Tables, figures, confusion matrices (`figure_2_two_layer_repair.svg`, `figure_3_repair_success_vs_bug_type_accuracy.svg`, …) |
| `results/three_way_comparison/table_3_execution_feedback_comparison.csv` | Main comparison |
| `results/three_way_comparison/confusion_matrix_*.csv` | Bug-type confusion per method |

## Project Structure

```
configs/
  sanitized_mbpp.common.yaml   # shared dataset / generation / models / evaluation
  sanitized_mbpp_*_pilot.yaml  # thin variants (extends common)
scripts/
  run_experiment.py              # one config
  run_three_way_experiments.py   # three-way batch + summarize + verify
  run_single_problem_pipeline.py # single problem + artifacts
  summarize_results.py, verify_comparison.py, export_paired_sample_ids.py
  spotcheck_reference_test_thinness.py
src/thesis_exp/
  analysis/       # Summarize, confusion matrices
  datasets/       # MBPP loader, filter
  evaluators/     # Diagnosis evaluation, repair execution
  injectors/      # Bug injection (registry keys: off_by_one, wrong_loop_bound, … → fine type loop_boundary_error)
  llm/            # Inference engine, adapters
  parsers/        # JSON diagnosis parser
  prompts/        # Prompt builders
  runners/        # Experiment runner
  schemas/        # Data models
```

## Hallucination: Diagnosis vs Narrative

Two hallucination metrics are reported separately:

- **diagnosis_hallucination_rate**: Predicted bug type ≠ ground truth (wrong diagnosis).
- **narrative_hallucination_rate**: Text claims multiple bugs or unsupported error causes (text-pattern based).

This separation better aligns with the research question: diagnosis hallucination reflects model correctness; narrative hallucination reflects extra/unsupported claims in the explanation.

## Bug Type: Fine vs Coarse Accuracy

Both fine-grained and coarse-grained bug type accuracy are reported. Coarse mapping groups structurally similar fine types (e.g. `loop_boundary_error` → `loop_bound_error`). Legacy result files may still list old labels; they are normalized to the current taxonomy (e.g. `condition_inversion` / `wrong_comparison_operator` → `conditional_logic_error`).

## Phased Tasks (Diagnosis vs Repair)

Two variants separate diagnosis from repair to test task entanglement:

- **diagnosis_only**: Output only `bug_type`, `bug_line`, `explanation`. No patched code. Use to measure diagnosis accuracy without patch-induced optimization.
- **diagnosis_then_repair**: Stage 1 outputs diagnosis JSON; stage 2 separately requests patched code given the diagnosis. Enables comparison: does diagnosis-only yield better bug_type accuracy?

Configs: `configs/sanitized_mbpp_diagnosis_only_pilot.yaml`, `configs/sanitized_mbpp_diagnosis_then_repair_pilot.yaml`.

## Repair Evaluation: Two Layers + Mutation Adequacy

Results report **two repair layers** so readers do not conflate “patch wrong” with “tests passed but mutation gate failed”:

1. **`patch_passes_selected_tests`** — 1.0 iff the patched code is syntactically valid, does not time out, and passes **every selected test** (public/hidden per `EvaluationConfig`). No mutation check.
2. **`repair_success`** (strict) — Same as layer (1), and when `use_mutation_adequacy` is on and a reference is available, the suite must also kill at least `min_mutants_killed` mutants of the reference—**unless** the mutator produces no mutants (`empty_reason=no_mutants_generated`), in which case mutation adequacy is **not assessable** and the patch is not penalized for that.

`mutation_adequacy_status` and `evaluation_notes` (e.g. `patch_passes_selected_tests=1`, `repair_tests_passed=3/3`, `tests_adequate=false(...)`) disambiguate failures. Summaries and `figure_2_two_layer_repair.svg` plot layer (1) vs layer (2). Configure via `evaluation.use_mutation_adequacy`, `min_mutants_killed`, `max_mutants_for_adequacy`, `repair_timeout_seconds` in YAML.

**Sandbox:** Patched and reference code run under `build_safe_exec_builtins` (restricted `__builtins__` + allowlisted imports). Common builtins such as `map`, `filter`, `pow`, `round`, etc. are included so idiomatic student solutions are not rejected spuriously.

## Reference

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., & Sutton, C. (2021). *Program synthesis with large language models*. arXiv preprint arXiv:2108.07732. MBPP sanitized benchmark: https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json


