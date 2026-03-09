# Thesis Experiment: LLM Bug Diagnosis & Hallucination Evaluation

Evaluates hallucinated bug diagnosis in LLM feedback for programming education. Compares three conditions: direct diagnosis, execution feedback with answer leakage (EF-leaky), and execution feedback without expected outputs (EF-no-answer).

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

**Run full experiment** (direct + EF-leaky + EF-no-answer):

```bash
# Windows
.\scripts\run_formal_experiments.ps1

# Linux / macOS
./scripts/run_formal_experiments.sh
```

**Run single config:**

```bash
python scripts/run_experiment.py --config configs/sanitized_mbpp_direct_pilot.yaml
```

**Summarize results:**

```bash
python scripts/summarize_results.py \
  --results-jsonl results/sanitized_mbpp_direct_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl \
  --output-dir results/three_way_comparison
```

## Data

- **Input:** `sanitized-mbpp.json` (sanitized MBPP benchmark)
- **Bug types:** `condition_inversion`, `wrong_comparison_operator`
- **Config:** `configs/*.yaml` — set `max_problems` to control sample size

## Pipeline

1. Load problems → filter → inject bugs
2. Build prompts (direct / EF-leaky / EF-no-answer)
3. Call LLM → parse JSON diagnosis
4. Evaluate: bug_type_accuracy, localization_accuracy, repair_success, hallucination_rate
5. Summarize → tables, figures, confusion matrices

## Outputs

| Path | Contents |
|------|----------|
| `results/<experiment_id>/results.jsonl` | Raw records |
| `results/<experiment_id>/results.csv` | Flattened CSV |
| `results/three_way_comparison/` | Tables, figures, confusion matrices |
| `results/three_way_comparison/table_3_execution_feedback_comparison.csv` | Main comparison |
| `results/three_way_comparison/confusion_matrix_*.csv` | Bug-type confusion per method |

## Project Structure

```
configs/          # Experiment YAML configs
scripts/          # run_experiment, summarize_results, verify_comparison
src/thesis_exp/
  analysis/       # Summarize, confusion matrices
  datasets/       # MBPP loader, filter
  evaluators/     # Diagnosis evaluation, repair execution
  injectors/      # Bug injection (condition_inversion, wrong_comparison_operator)
  llm/            # Inference engine, adapters
  parsers/        # JSON diagnosis parser
  prompts/        # Prompt builders
  runners/        # Experiment runner
  schemas/        # Data models
```

## Reference

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., & Sutton, C. (2021). *Program synthesis with large language models*. arXiv preprint arXiv:2108.07732. MBPP sanitized benchmark: https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json


