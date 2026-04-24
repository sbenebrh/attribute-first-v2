# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for the ACL 2024 paper "Attribute First, then Generate: Locally-attributable Grounded Text Generation". The system produces grounded text summaries where each output sentence is attributable to specific source document spans ("highlights").

Two main experimental tracks:
- **`few_shot_experiments/`** — LLM-based (GPT/Gemini) few-shot pipeline
- **`fine_tuned/`** — Fine-tuned transformer models (LED/PRIMERA-based)

Two task settings: **MDS** (Multi-Document Summarization) and **LFQA** (Long-form Question Answering).

## Environment Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

Python 3.11 is used (`.venv/`). A Python 3.9 backup venv is at `.venv_py39_backup/`.

## Few-shot Experiments

All commands run from `few_shot_experiments/`:

```bash
# Run a single subtask
python run_script.py --config-file configs/<SPLIT>/<SETTING>/<SUBTASK>.json

# Run iterative sentence generation
python run_iterative_sentence_generation.py --config-file configs/<SPLIT>/<SETTING>/iterative_sentence_generation.json

# Run a full pipeline
python run_full_pipeline.py --config-file configs/<SPLIT>/<SETTING>/<PIPELINE_TYPE>.json
```

- `SPLIT`: `dev` or `test`
- `SETTING`: `MDS` or `LFQA`
- `SUBTASK`: `content_selection`, `clustering`, `fusion_in_context`, `e2e_only_setting`, `ALCE`
- `PIPELINE_TYPE`: `full_pipeline` or `full_CoT_pipeline`

**Pipeline chaining**: Each subtask outputs `pipeline_format_results.json` in its outdir. Pass it to the next subtask via `--indir-alignments /path/to/pipeline_format_results.json`.

**Key flags**: `--debugging` (small subset), `--CoT` (chain-of-thought for FiC/clustering), `--n-demos N` (ICL shots, default 2), `--model-name` (default `gemini-pro`).

### Evaluation

```bash
# From few_shot_experiments/
python evaluation/calc_rouge_l.py --results-dir results/<SPLIT>/<SETTING>/<EXP>/
python evaluation/calc_repetition.py --results-dir results/<SPLIT>/<SETTING>/<EXP>/
python evaluation/calc_llm_repetition.py --results-dir results/<SPLIT>/<SETTING>/<EXP>/
python evaluation/calc_longcite.py --results-dir results/<SPLIT>/<SETTING>/<EXP>/
```

## Fine-tuned Models

All commands run from `fine_tuned/`:

```bash
# Training
python train.py <HF_TRAINER_ARGS> --config-file configs/train_<MODEL_TYPE>.json

# Inference (First-attribute pipeline)
python inference.py --config-file configs/inference.json

# Inference (vanilla PRIMERA baseline)
python inference_e2e.py --config-file configs/e2e_inference.json
```

Config files under `fine_tuned/configs/`: `train_highlights_detection.json`, `train_generative_clustering.json`, `train_attributable_fusion.json`.

## Architecture

### Few-shot Pipeline Flow

```
Input data (data/{MDS,LFQA}/{split}.json)
  → content_selection  (select relevant spans from docs)
  → clustering         (group highlights by output sentence)
  → FiC (fusion-in-context) / iterative sentence generation
  → final summary with per-sentence attributions
```

`utils.py` handles: data loading, prompt construction (template substitution with `{INST}`, `{D}`, `{A}`, `{Q}`, `{HLIST}`, etc.), model calls (OpenAI/Gemini), and result saving. `subtask_specific_utils.py` has per-subtask response parsers and pipeline format converters.

### Fine-tuned Pipeline Flow

```
Input data
  → Highlights detection (token classification on LED encoder)
  → Generative clustering (FiC model groups highlights)
  → Attributable fusion (seq2seq generation conditioned on clusters)
  → final summary
```

`fine_tuned/src/train/` has preprocessors for each model stage. `fine_tuned/src/inference/` has the runtime runners (`run_highlights_detection.py`, `run_fic_model.py`, `run_attributable_fusion.py`).

### Data Format

Input files are JSONL. Each line has `unique_id`, `topic`, `documents` (list with `documentFile`, `rawDocumentText`, `documentText`, `docSentCharIdxToSentIdx`), `set_of_highlights_in_context` (highlight alignments between doc spans and summary spans), and `response`/`question`.

Results are saved as `results.json`, `results.csv`, `used_demonstrations.json`, and optionally `pipeline_format_results.json` (for pipeline chaining).

## Agent Workflow

### Use Subagents for Independent Work

Spawn subagents in parallel whenever tasks are independent. Do not run them sequentially if they don't depend on each other.

| Situation | Agent to use |
|-----------|-------------|
| Modifying any Python file | `code-reviewer` immediately after |
| New feature or bug fix | `tdd-guide` before writing code |
| Complex design question | `architect` or `planner` |
| Build or import error | `build-error-resolver` |
| Codebase-wide search | `Explore` subagent |

**Typical parallel pattern for this repo** — e.g., when adding a new evaluation metric:
1. Agent 1: `tdd-guide` — write tests for the new metric
2. Agent 2: `Explore` — find where existing metrics are integrated
3. Agent 3: `code-reviewer` — review any files already changed

### Enter Plan Mode for Complex Tasks

Activate Plan Mode (via `EnterPlanMode`) — without using extended thinking — before starting any task that meets one or more of these conditions:

- Touches more than one experimental track (`few_shot_experiments/` **and** `fine_tuned/`)
- Requires changes to the pipeline chaining contract (`pipeline_format_results.json` schema)
- Involves adding a new subtask or a new evaluation metric end-to-end
- Requires modifying the data format (`Highlight`, `Document`, `Example` dataclasses)
- Is ambiguous about which setting (MDS vs LFQA), split (dev vs test), or model family (GPT vs Gemini vs fine-tuned) should be targeted

In Plan Mode, produce a step-by-step plan and **stop to ask the user** before executing. Do not proceed autonomously when the scope is unclear.

### When to Ask Before Acting

Always pause and ask the user for clarification when:

- The target split (`dev` / `test`) or setting (`MDS` / `LFQA`) is not specified
- It is unclear whether the change should apply to the few-shot track, the fine-tuned track, or both
- A config file would be overwritten or a results directory would be deleted
- The requested model name is not one of the supported families (`gpt-*`, `gemini-*`, or a local checkpoint path)
- GPU/CPU memory requirements cannot be inferred from context (fine-tuned inference requires ≥60 GiB GPU + ≥100 GiB CPU RAM)
