# Improvements — Code & Concepts

**Date:** 2026-04-25
**Scope:** few_shot_experiments/ (LLM pipeline). Findings apply mostly to this track; fine_tuned/ noted where relevant.

This is a deep audit of what should be improved in the codebase and the methodology
behind the "Attribute First, then Generate" few-shot pipeline. Items are grouped by area
and ranked **P0 (critical) → P1 (important) → P2 (polish)** within each section.

---

## 1. Critical bugs (P0)

### 1.1 Rate-limit errors leak into `pipeline_format_results.json` as fake predictions
**Location:** `pipeline_converters.py:get_set_of_highlights_in_context_FiC_CoT` + `convert_FiC_CoT_results_to_pipeline_format`.

When `gemini_call` exhausts retries on a 429, `model_call_wrapper` returns
`{"final_output": "ERROR - 429 Resource exhausted. Please try again later.", "alignments": []}`.
The downstream converter then runs `nlp(final_output).sents` on the error string,
splits it into "ERROR - 429 Resource exhausted." and "Please try again later.",
and emits each as a **`scuSentence` with `documentFile=None`**.

This is the root cause of the **30–38% None highlights in MDS zeroshot**.
The reported F1 ≈ 0.13 is *not* a real metric — it's metric-on-error-strings.

**Fix:**
- Drop instances whose `final_output` starts with `"ERROR"` or whose `alignments=[]` *before* converting to pipeline format.
- Emit one explicit row per dropped instance: `{"unique_id": ..., "skipped_reason": "model_error", "set_of_highlights_in_context": []}`.
- LongCite scorer should treat skipped instances either as recall=0/excluded; we should report both numbers.

### 1.2 Single-threaded `prompt_model` makes 429s the default behavior
**Location:** `utils.py:prompt_model`, `utils.py:model_call_wrapper`.

Each instance is processed sequentially with `num_retries=5` and a hardcoded `sleep(60)` on 429.
A single bad minute can burn 5 minutes of the run. With 44+20 instances and re-runs across 6 experiment variants,
this is the dominant cost.

**Fix:**
- Replace the for-loop with `asyncio.gather` + a token-bucket rate limiter (e.g. `aiolimiter`).
- Implement exponential backoff with jitter: `wait = min(60, 2**attempt + uniform(0, 1))`.
- Respect Gemini's `retry-after` header where available.
- When still 429 after backoff, mark the instance "deferred" rather than "ERROR" so we can retry later in a separate pass.

### 1.3 `parse_FiC_response` regex is fragile and silently produces empty alignments
**Location:** `response_parsers.py:139`.

Pattern: `r"highlight(?:s)? ([\d, and]+) (?:is|are) combined to form sentence (\d+):[ \n]*(.*?)(?=[\n]+|\Z)"`.

Issues:
- `[\d, and]+` is a character class — it matches any of `d`, ` `, `,`, `a`, `n`, etc. Not the words "and". The intent doesn't match the implementation.
- `re.DOTALL` is **not** set, but `.*?` is supposed to span newlines for multi-line sentences. Currently it stops at first newline.
- The look-ahead `(?=[\n]+|\Z)` requires at least one newline; if the model's output has only one line, the match terminates only at EOF.
- We just fixed `[\n ]+` → `[ \n]*` (one→zero), but the character class is still semantically wrong.

**Fix:** Replace regex parsing with **structured JSON output** as the default for FiC-CoT.
Keep the regex parser only as a fallback for non-structured runs. Add a unit-test corpus
of representative model outputs (10–20 cases) so regression is caught.

### 1.4 `parse_FiC_structured_response` swallows all exceptions
**Location:** `response_parsers.py:304`.

```python
except Exception:
    return parse_FiC_response(response, prompt)
```

When JSON mode is requested but Gemini truncates, returns malformed JSON, or returns a
schema-violating shape, we silently fall back to the text parser — which returns
`alignments=[]` because there's no "So the final summary is:" sentinel in JSON output.

**Fix:**
- Narrow to `except (json.JSONDecodeError, KeyError, TypeError)`.
- Implement a `_extract_partial_sentences()` helper analogous to `_extract_partial_highlights()`.
- Surface a clear error when JSON mode was on but parsing failed → triggers retry with a different demo set, not silent fallback to text parser.

### 1.5 `MIN_CONTEXT_SPAN_CHARS = 30` discards short but high-value highlights
**Location:** `pipeline_converters.py:170`.

This filter was added (Fix A) to avoid single-token fragments like "He" or "Rome." polluting FiC.
But it also drops dates, numerals, named entities, and short factual quotes — all of which are
*good* attribution targets.

**Fix:** Type-aware filter. Keep spans that are ≥30 chars **OR** contain at least one
NER-tagged entity / numeric / date token (use spaCy `nlp(span).ents`).

### 1.6 Decontextualized spans are matched against source as if verbatim
**Location:** `response_parsers.py:218–222` (CS) and `:270–274` (AH).

The assertion `rmv_spaces_and_punct(s) in rmv_spaces_and_punct(docs_texts_clean[doc_name])`
requires the span text to literally appear in the source. But AH (decontextualization) by
definition rewrites spans (e.g. "He visited Rome" → "Donald Trump visited Rome").
So the assert *should* fail for properly decontextualized spans — and it currently
"works" only because the model often returns the original span unchanged.

**Fix:** Schema must carry both fields:
```json
{"doc_id": "Document [1]", "original_span": "He visited Rome", "decontextualized_span": "Donald Trump visited Rome"}
```
Use `original_span` for offset matching; pass `decontextualized_span` to FiC.
This requires updating the AH schema, the AH parser, the AH prompt, and the AH→FiC converter.

---

## 2. Architecture & code quality (P1)

### 2.1 `utils.py` is a 654-line kitchen sink
Split into:
- `api_clients.py` — gemini_call, openai_call, gemini_chat_call, model_call_wrapper, prompt_model
- `prompt_construction.py` — make_demo, make_doc_prompt, make_*_prompt
- `span_utils.py` — find_substring, merge_spans, rmv_spaces_and_punct, get_consecutive_subspans
- `highlights.py` — get_highlighted_doc, add_highlights, extract_highlights
- `io.py` — save_results, get_data, update_args
- `gpu_utils.py` — get_max_memory (only used by fine_tuned/)
- `tokenization.py` — TokenCounter, get_token_counter, _normalize_model_name
- `config.py` — typed `RunConfig` dataclass replacing `args.x` getattrs

### 2.2 Wildcard imports break tooling
`from utils import *`, `from prompt_utils import *`, `from response_parsers import *` are
used everywhere. This:
- Hides which names each module actually depends on.
- Defeats `ruff`/`pyright`/`mypy` unused-import detection.
- Makes refactoring dangerous (renames silently break).

**Fix:** Replace with explicit imports. Add `__all__` to each module.

### 2.3 No tests, no CI
There are zero unit tests. Critical regex parsers, span-merging logic, and JSON-recovery
helpers have no regression coverage. The recent `[\n ]+` → `[ \n]*` regex bug would have
been caught by a 5-line pytest.

**Fix (minimum viable):**
- `tests/test_response_parsers.py`: 10–15 fixture model responses (CoT, non-CoT, structured, truncated, partial-JSON, with various model phrasings) → assert expected `alignments`.
- `tests/test_pipeline_converters.py`: 5 instance fixtures → assert `documentFile` is never `None` for non-error inputs.
- `tests/test_span_utils.py`: edge cases for `find_substring`, `merge_spans`, `_extract_partial_highlights`.
- GitHub Actions: `pytest`, `ruff check`, `pyright` on every PR.

### 2.4 Module-level side effects slow startup
`pipeline_converters.py` loads sentence-transformer + tokenizer + torch at import time.
`utils.py` calls `genai.configure()` and creates module-level singletons.
`response_parsers.py` lazily-loads spaCy (good) but still imports `from utils import *`.

**Fix:** Wrap heavy imports in `_get_*` accessor functions (already done for spaCy). Apply
the same pattern to sentence-transformer.

### 2.5 Reproducibility holes
- `np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False)` in `construct_prompts`
  has no seed parameter — different demos every run.
- No `requirements.txt` exact-pinning; `google-generativeai`, `transformers`, `spacy` are all
  semver-volatile.
- Two venvs (`.venv`, `.venv_py39_backup`) — pick one.

**Fix:**
- Add `--seed` CLI arg, propagated to `np.random.default_rng(seed)`.
- Pin every dep in `requirements.txt` with `==`, regenerate via `pip-compile`.
- Document the canonical Python version (3.11) and remove the backup venv.

### 2.6 Silent reads of `args.x` via `getattr(...)` everywhere
`structured_output`, `output_max_length`, `rerun`, `rerun_path`, `rerun_n_demos`, etc. — all
read defensively from a dict-shaped `argparse.Namespace`. This is fragile because:
- A typo in the JSON config produces `False` silently.
- No type coercion (`"true"` string ≠ `True` boolean).
- Adding a new arg requires touching 4 files.

**Fix:** Define a `RunConfig` Pydantic dataclass / Pydantic-Settings model. Validate at load.
Reject unknown keys. Coerce types.

### 2.7 Duplicate / dead code
- `rmv_spaces_and_punct` (utils.py:221) and `remove_spaces_and_punctuation` (utils.py:224) — same intent, different impls. Delete one.
- `_extract_partial_highlights` lives in `response_parsers.py` but is general-purpose JSON salvage. Move to `span_utils.py`.
- `parse_FiC_response` mixes CoT and non-CoT in one function — split into two.
- `evaluation/calc_repetition.py` and `evaluation/calc_llm_repetition.py` share substantial code → factor out `evaluation/_repetition_common.py`.

### 2.8 Magic numbers
| Const | File | Should be |
|-------|------|-----------|
| `COSINE_SIMILARITY_THR = 0.6` | pipeline_converters.py | Configurable; recalibrate for current sentence-transformer |
| `MIN_CONTEXT_SPAN_CHARS = 30` | pipeline_converters.py | NER-aware (see §1.5) |
| `output_max_length = 4096` (default) | utils.py | Per-subtask config (we now do 8192 for structured) |
| `tkn_max_limit = 30000` (fallback) | utils.py | Per-model lookup |
| `prct_surplus_lst = [0.5, 0.6, 0.7]` | prompt_utils.py | Configurable |
| `wait = 60` (429 backoff) | utils.py | Exponential backoff |
| `safety_settings = BLOCK_NONE × 4` | utils.py | Document why; may not be needed for benign prompts |

---

## 3. Conceptual / methodological issues (P0–P1)

### 3.1 No supervision signal between stages — pure error compounding
**P0.** CS → AH → FiC has no joint optimization. Errors in CS (over- or under-selection)
cannot be corrected by AH; AH errors flow uncorrected into FiC.

**Possible fixes:**
- Run all three steps in a single structured prompt that returns
  `{cs_highlights, decontextualized_spans, sentences}` simultaneously. One LLM call instead of three.
- Add a "reflect" step: after FiC generates, ask the model whether each sentence's cited
  highlights actually support the claim; iterate until convergence (max 2 rounds).

### 3.2 The "abstain" mechanism was designed wrong, and is currently absent
**P1.** The original FiC schema had `abstain` in `required` → model abstained 100%.
We removed it. But conceptually we still need *per-sentence* abstain: when a generated
sentence cannot be backed by any highlight cluster, mark it `documentFile=None` *explicitly*
(not as an artifact of pipeline conversion failures, see §1.1).

**Fix:** Add `confidence` (float) and `abstain` (bool) to the per-sentence schema. Treat
`abstain=true` sentences as either (a) excluded from the prediction or (b) flagged in the UI.

### 3.3 Substring matching for span→source linking misses paraphrases
**P1.** `rmv_spaces_and_punct(s) in rmv_spaces_and_punct(doc)` handles whitespace and
punctuation differences, but breaks on:
- Quote normalization ("said it" vs "said it")
- Word reordering
- Synonym substitution ("said" vs "stated")
- Truncation ("the President of the United States" vs "the President")

**Fix:** Use **fuzzy span matching** as fallback:
1. Try exact substring.
2. Try `rapidfuzz.fuzz.partial_ratio(span, doc) >= 90`.
3. Use embedding-based search (sentence-transformer cosine ≥ 0.85) as last resort.
Report a "match_method" field per highlight so we can audit fallback rates.

### 3.4 Ground-truth highlights are never compared to predictions
**P1.** The dataset has gold `set_of_highlights_in_context`, but our pipeline only emits
predicted highlights. We never compute span-IoU between predictions and gold.

**Fix:** Add `evaluation/calc_highlight_overlap.py`:
- Span-level IoU between predicted offsets and gold offsets (per document).
- Sentence-level recall: what fraction of gold-highlighted sentences were touched by any prediction?
- Document-level coverage: do we cite from all documents that the gold cites from?

This is independent of LongCite (which uses an LLM judge) — the difference between IoU and
LongCite F1 is informative.

### 3.5 LongCite citation pool fallback inflates recall artificially
**P1, location:** `evaluation/calc_longcite.py:87–129`.

When `set_of_highlights_in_context` is empty, we send the **top-5 documents (300-char raw snippets)**
as the citation pool. LongCite then has lots of text to attribute against, so recall trends up
*even when our pipeline produced no real highlights*.

**Fix:** Report two F1 numbers per experiment:
- **Strict F1**: only our extracted highlights as the citation pool. If empty → recall=0.
- **Generous F1**: with document-text fallback (current behavior).
The gap between them quantifies how much our pipeline contributes vs the document fallback.

### 3.6 AH (decontextualization) is currently retrieval-only, not generation-enhancing
**P1.** Decontextualized spans are added to the candidate pool but never appear in the FiC
prompt as decontextualized text — FiC sees them as raw spans. So the model never benefits
from the disambiguation.

**Fix:** Pass decontextualized text to FiC alongside the original. Either:
- Replace the original span text in the FiC context with the decontextualized version.
- Or annotate: `<span doc=1 original="He visited Rome">Donald Trump visited Rome</span>`.

### 3.7 No prompt-sensitivity / seed-variance numbers
**P2.** Every reported F1 is a single-seed point estimate with n=20 (MDS) or n=44 (LFQA).
A single bad demo set can swing F1 by 5–10 points.

**Fix:** Run each experiment with k=5 random seeds, report mean ± std, and use bootstrap
CIs over instances. Pre-allocate the seeds in a `seeds.json` file for reproducibility.

### 3.8 No comparison to "post-hoc attribution" baseline
**P1.** A cheap baseline: have the model generate the answer freely (no CS/AH/FiC), then
post-hoc attribute each output sentence to the most-similar source span via embedding search.
If post-hoc attribution achieves 80% of FiC's F1 at 1/3 the cost, the whole pipeline is
hard to justify.

**Fix:** Add `e2e_post_hoc_attribution.json` config. Generate, then attribute via
sentence-transformer cosine. Compare LongCite F1.

### 3.9 Dataset size too small for the conclusions we want to draw
**P1.** n=20 (MDS) and n=44 (LFQA) are small enough that ±5 points on F1 is well within
sampling noise. We're claiming differences (decontex vs baseline, structured vs non-structured)
that require larger samples to be significant.

**Fix:**
- Run on the test split (typically larger).
- Report 95% bootstrap CIs and Wilcoxon signed-rank for paired comparisons.
- For LFQA, the original ASQA dataset is much larger — sample 200 instances if compute allows.

---

## 4. Pipeline / experimental design (P1)

### 4.1 Dialogue mode does not actually save compute on Gemini
The implementation assumes ChatSession reuses KV-cache across turns. Gemini's API does not
expose KV-cache reuse — every turn re-encodes the full conversation. So "savings" are only
in network round-trips and our own demo-construction code.

**Fix:** Either drop dialogue mode, or measure actual token cost per instance (already
tracked via `prompt_tokens_total`). If equal cost, document that dialogue mode is for
*UX*, not throughput.

### 4.2 Decontextualization could be a single span-rewriting pass over CS output
The current 2-LLM-call structure (CS, then AH, then FiC) is overkill. AH could be a
deterministic post-process if we use a simple coreference resolver (e.g. `fastcoref`,
or spaCy's neuralcoref). Or it could be merged into the CS prompt as
"return spans + their decontextualized form".

### 4.3 No ablation on schema design
Why `{"highlights": [{"doc_id, span_text"}]}` over `{"by_doc": {"doc_id": ["span1", "span2"]}}`
or `{"spans": [{"doc_id", "start_offset", "end_offset"}]}`?
- Offsets directly would skip the entire substring-matching layer (§3.3).
- By-doc nesting might reduce schema-violation rates.
- Ask the model to commit to a specific span via offsets: less ambiguity, no matching step.

Run all three and report token cost, parse-failure rate, and downstream F1.

### 4.4 No pipeline-level cost tracking
We track LongCite's `prompt_tokens_total` and `completion_tokens_total`, but not the
upstream CS/AH/FiC LLM calls. Without per-stage cost numbers, we can't reason about
quality/cost tradeoffs.

**Fix:** Wrap each `gemini_call` to record `input_tokens, output_tokens, latency_ms` to
a per-instance log. Aggregate per-experiment.

### 4.5 Evaluation runs are not idempotent
LongCite uses Gemini as the judge. Even at temperature=0, model outputs vary. Two runs of
the same experiment can produce different F1.

**Fix:** Cache LongCite per-instance scores keyed by `(prediction_hash, citation_hash)`.
On rerun, only re-judge instances whose inputs changed.

---

## 5. Documentation, repo hygiene (P2)

### 5.1 No data-schema reference
`unique_id`, `documents`, `set_of_highlights_in_context`, `new_set_of_highlights_in_context`,
`context_set_of_highlights_in_context`, `documentFile`, `documentText`, `rawDocumentText`,
`docSentCharIdx`, `scuSentCharIdx`, `docSpanText`, `docSpanOffsets`, `sent_idx` — all of these
are touched by the code but not defined anywhere.

**Fix:** `docs/SCHEMAS.md` with one paragraph + JSON example per type.

### 5.2 README is anemic
The README only links to the paper. A new collaborator would have to read CLAUDE.md to know
how to run anything. CLAUDE.md is for Claude Code, not for humans.

**Fix:** Move the "running an experiment" section from CLAUDE.md → README.md. Keep CLAUDE.md
focused on agent-specific guidance.

### 5.3 `multi-text-fic/` is a 906MB sub-directory in the repo root
It has its own `.git`, isn't a proper submodule (`.gitmodules` doesn't exist), and shows up
as `m multi-text-fic` in `git status`. It's an external repo someone cloned in here.

**Fix:** Either make it a proper git submodule (and pin a commit), or delete it and document
the URL in the README. As-is, it bloats backups and confuses tooling.

### 5.4 `FUTURE_WORK_PLAN.md` exists but is gitignored / untracked
**Status check:** it's actually tracked. But it's 334 lines and overlaps with this doc.
Consolidate or archive.

### 5.5 No `.python-version` or pyproject.toml
Python 3.11 is documented in CLAUDE.md but not enforced. Add `.python-version` for
pyenv users and `[project] requires-python = ">=3.11,<3.12"` to a `pyproject.toml`.

---

## 6. Priority-ordered action list

### Sprint 1 — Fix the data quality (so metrics are trustworthy)
1. **§1.1**: Drop ERROR instances before pipeline_format conversion. Add unit test.
2. **§1.6**: Redesign AH schema with `original_span` + `decontextualized_span`. Update parser, prompt, converter.
3. **§3.5**: Add strict-F1 LongCite alongside generous-F1.
4. **§3.4**: Add `calc_highlight_overlap.py` for span-IoU vs gold.
5. Re-run all six broken experiments (LFQA + MDS × CoT/decontex × baseline/zeroshot/dialogue) with the fixes.

### Sprint 2 — Make experiments reproducible
1. **§2.5**: Pin requirements, add `--seed`, remove `.venv_py39_backup/`.
2. **§3.7**: Run k=5 seeds per experiment, bootstrap CIs.
3. **§4.4**: Per-stage token / latency tracking.
4. **§4.5**: Cache LongCite by `(prediction_hash, citation_hash)`.

### Sprint 3 — Refactor for maintainability
1. **§2.3**: Add tests for response_parsers, pipeline_converters, span_utils.
2. **§2.1, §2.2**: Split utils.py, replace wildcard imports.
3. **§2.6**: Pydantic `RunConfig`.
4. **§1.2**: Async prompt_model with rate limiter.

### Sprint 4 — Methodological improvements
1. **§3.1**: Single-prompt CS+AH+FiC variant. Measure quality/cost.
2. **§3.8**: Post-hoc attribution baseline.
3. **§4.3**: Schema-design ablation (3 schemas × 4 experiments).
4. **§3.6**: Plumb decontextualized text into FiC prompt.

### Sprint 5 — Polish
1. §2.7 dead code, §2.8 magic numbers, §5.1 schema docs, §5.2 README rewrite.
