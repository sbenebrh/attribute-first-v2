# Few-shot Pipeline — Dev Results Status

**Date:** 2026-04-25  
**Branch:** main

---

## Summary of fixes applied

### 1. FiC CoT regex bug (non-structured experiments)

The regex that extracted highlight-to-sentence alignments in `response_parsers.py` used `[\n ]+`
(one-or-more whitespace), causing it to miss alignments when the model placed no gap character
between the colon and the sentence text.  Fixed to `[ \n]*` (zero-or-more).

Effect: 39/44 LFQA baseline instances had `alignments=[]` before the fix.
After reparsing from stored `full_model_response`, 39 were recovered.

### 2. Structured output pipeline — three root-cause bugs

All four structured variants (LFQA + MDS × CoT + decontex) produced 100% empty output:

| Component | Bug | Fix |
|-----------|-----|-----|
| **CS** (content_selection) | Parser caught all exceptions → always fell back to text parser; JSON truncated mid-array even with 4096 tokens | Added `_extract_partial_highlights()` for partial JSON recovery; narrowed `except Exception` → `except json.JSONDecodeError`; set `output_max_length: 8192` in all structured configs |
| **AH** (ambiguity_highlight) | Schema had `non_ambiguous_highlights` as `list[string]` with no `doc_id` → model returned empty lists 43/44 times | Redesigned schema to `{"highlights": [{"doc_id": "...", "span_text": "..."}]}`; added structured instruction prompt (no SPAN_DELIM) |
| **FiC** (fusion_in_context) | `abstain` field in `required` list of `FIC_COT_SCHEMA` → model set `abstain: true` for 100% of instances | Removed `abstain` from both `properties` and `required` |

---

## LongCite F1 results — post-reparse (task b7xnkli37, Apr 24 20:42)

> These numbers were computed by task b7xnkli37 immediately after reparsing each
> `pipeline_format_results.json`, making them the authoritative post-fix values.
> The on-disk `avg_results.json` files reflect earlier runs and may differ.

### LFQA (n=44)

| Experiment | Instances fixed by reparse | None documentFile | Citation Recall | Citation Precision | LongCite F1 |
|---|---|---|---|---|---|
| `full_CoT_pipeline` (baseline) | **39** | 0 / 393 (0%) | — | — | **0.9425** ✅ |
| `full_CoT_pipeline_zeroshot` | 2 | 8 / 277 (3%) | — | — | 0.7630 |
| `full_CoT_pipeline_dialogue` | 3 | 8 / 242 (3%) | — | — | 0.7130 |
| `full_decontextualization_CoT_pipeline` | 0 | 5 / 200 (2%) | — | — | 0.2664 ⚠️ |
| `full_decontextualization_CoT_pipeline_zeroshot` | 2 | 3 / 257 (1%) | — | — | 0.5627 |
| `full_decontextualization_CoT_pipeline_dialogue` | 11 | 14 / 197 (7%) | — | — | 0.4306 |

### MDS (n=20)

| Experiment | Instances fixed by reparse | None documentFile | LongCite F1 |
|---|---|---|---|
| `full_CoT_pipeline_1` (baseline) | n/a (not reparsed) | — | **0.9750** ✅ |
| `full_CoT_pipeline_dialogue` | n/a | — | 0.8697 |
| `full_decontextualization_CoT_pipeline` | n/a | — | **0.9794** ✅ |
| `full_decontextualization_CoT_pipeline_dialogue` | n/a | — | 0.9392 |
| `full_CoT_pipeline_zeroshot` | 0 | **29 / 96 (30%)** | 0.1328 ⚠️ |
| `full_decontextualization_CoT_pipeline_zeroshot` | 0 | **42 / 110 (38%)** | 0.1059 ⚠️ |

---

## Open issues

### HIGH: MDS zeroshot experiments — broken pipeline_format (30–38% None) [ROOT CAUSE FOUND]

Both `MDS/full_CoT_pipeline_zeroshot` and `MDS/full_decontextualization_CoT_pipeline_zeroshot`
have 30–38% of highlights with `documentFile=None`, yielding LongCite F1 ≈ 0.10–0.13.

**Root cause (2026-04-25):** The "None highlights" are not real predictions — they are
**rate-limit error strings** (`"ERROR - 429 Resource exhausted. Please try again later."`)
that leak through `convert_FiC_CoT_results_to_pipeline_format`. The converter splits
the error string with spaCy's sentence splitter and emits each sentence as a `scuSentence`
with `documentFile=None`. So the F1 ≈ 0.13 is *metric-on-error-strings*, not a real F1.

**Fix:** See IMPROVEMENTS.md §1.1 — drop ERROR instances before pipeline_format conversion.
Then `--rerun` the 5 affected MDS zeroshot instances.

### HIGH: LFQA decontex pipeline — unexpectedly low F1 (0.2664)

`full_decontextualization_CoT_pipeline` has 5 highlights with `documentFile=None` (2%) and
LongCite F1 = 0.2664, despite the baseline reaching 0.9425.
The reparse fixed 0 instances, so the issue is not the FiC regex.
Possible causes: AH → FiC alignment mismatch, or AH output quality degradation.
Compare against `full_decontextualization_CoT_pipeline_zeroshot` (F1=0.5627) to isolate.

### MEDIUM: Dialogue variants — 5 unfixable ERROR instances

In `full_CoT_pipeline_dialogue` and `full_decontextualization_CoT_pipeline_dialogue`,
5 instances have truncated `full_model_response` (never reached the CoT planning section).
These cannot be recovered by reparse; require `--rerun` to re-call the model.

### MEDIUM: Structured pipelines — not yet run at full scale

The structured pipeline fixes (CS/AH/FiC schemas + parsers + `output_max_length: 8192`)
have been validated on small tests (3 instances) but the full dev runs have not been launched.
Four full runs needed: LFQA+MDS × CoT+decontex.

### LOW: ROUGE-L and llm_repetition — not yet computed for zeroshot/dialogue variants

Standard non-LongCite metrics (`calc_rouge_l.py`, `calc_llm_repetition.py`) are missing
for the zeroshot and dialogue experiment variants.

---

## File changes in this fix cycle

| File | Change |
|------|--------|
| `response_parsers.py` | Added `_extract_partial_highlights()`; narrowed exception handling in structured parsers |
| `schemas.py` | Redesigned AH schema; removed `abstain` from FiC CoT schema |
| `subtask_specific_utils.py` | Dispatch to structured parsers when `structured_output=True` |
| `prompt_utils.py` | Use structured instruction keys when `structured_output=True` |
| `run_script.py` | Forward `output_max_length` config key to `prompt_model()` |
| `prompts/LFQA.json`, `prompts/MDS.json` | Added `instruction-content-selection-structured` and `instruction-ambiguity-highlight-structured` keys |
| `configs/dev/*/[cs\|ah\|fic]_structured.json` | Added `"output_max_length": 8192` to all 6 structured configs |
| `results/dev/LFQA/full_CoT_pipeline*/` | Reparsed `results.json`, `pipeline_format_results.json`, updated LongCite metrics |
