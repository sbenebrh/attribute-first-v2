# Future Work Plan — Attribute-First-Then-Generate

**Date:** 2026-04-22  
**Project:** Few-Shot Grounded Text Generation (MDS + LFQA)  
**Author:** Samuel Benibu-Gui

---

## 1. Current State and Key Findings

### 1.1 What Works

The full decontextualization CoT pipeline is functional end-to-end for MDS:

| Pipeline | Setting | ROUGE-L F1 | Rep-2 | Notes |
|---|---|---|---|---|
| full_CoT_pipeline | LFQA | 0.3671 | — | Baseline (no decontextualization) |
| full_decontextualization_CoT_pipeline | LFQA | **0.3396** | — | **-2.8pp vs baseline** |
| full_decontextualization_CoT_pipeline | MDS | 0.2046 | 0.0535 | Only complete MDS run |
| full_CoT_pipeline_zeroshot | MDS | 0.1452 | 0.127 | 5/20 errors (rate limit) |

### 1.2 The Core Problem: LFQA Decontextualization Hurts

The decontextualization step **lowers** LFQA ROUGE-L by 2.8 percentage points.  
The same step is expected to **help** (it was designed to add context to ambiguous highlights like
pronouns referring to antecedents outside the selected span).

Likely causes (to diagnose before further experiments):

1. **Format mismatch in the FiC step.** After decontextualization, the highlights contain
   rewritten text that may not match the `docSpanText` tokens the FiC prompt template expects.
   The LFQA prompts may reference the original span differently than the MDS prompts do.

2. **`query` vs `question` field inconsistency.** LFQA instances use `"query"` as the key,
   but prompt templates and the `always_with_question` logic may inject it under `"question"`.
   If the AH step receives a blank question, its decontextualization is context-free and
   produces generic rewrites.

3. **Over-rewriting.** The AH step may expand every highlight (even unambiguous ones),
   making spans longer and less precise, which reduces lexical overlap with gold summaries.

4. **Prompt calibration gap.** The AH prompt was tuned/demonstrated on MDS examples. LFQA
   highlights are typically shorter and more fragmentary; the same prompt may overreact.

---

## 2. Immediate Fixes (Before Next Experiments)

### Fix A: Debug LFQA format mismatch in FiC after decontextualization

**Goal:** Understand exactly how `pipeline_format_results.json` from the AH step differs from
the one produced by content selection, and whether the FiC prompt template handles both correctly.

**Action:**
1. Inspect two intermediate files side-by-side:
   ```
   results/dev/LFQA/full_decontextualization_CoT_pipeline/itermediate_results/ambiguity_highlight/pipeline_format_results.json
   results/dev/LFQA/full_decontextualization_CoT_pipeline/itermediate_results/content_selection/pipeline_format_results.json
   ```
2. Confirm the FiC prompt rendering uses the rewritten span text correctly (check
   `construct_non_demo_part` in `subtask_specific_utils.py`).
3. If the FiC template still references `docSpanText` (original) instead of the rewritten text
   stored by AH, patch `convert_ambiguity_highlight_results_to_pipeline_format` to overwrite
   `docSpanText` with the rewritten version.

### Fix B: Align `query` → `question` for LFQA in the AH step

**Goal:** Ensure the AH step receives the question text so it can decontextualize
with respect to what the user is asking.

**Action:**
- In `subtask_specific_utils.py`, when building the AH prompt for LFQA, fall back to
  `instance.get("query", "")` if `instance.get("question", "")` is empty.
- Add a unit test: create a mock LFQA instance with `query` but no `question`, verify the
  rendered AH prompt contains the query text.

### Fix C: Selective decontextualization

**Goal:** Only rewrite highlights that actually contain an unresolved reference, not all of them.

**Action:**
- The AH model already labels each highlight; parse its output to identify which spans
  are marked as ambiguous (`problematic_instance == "yes"` in the gold data provides ground truth).
- Add a `skip_unambiguous` flag to the AH step: if the model says the span is clear, pass it
  through unchanged.
- Expected benefit: preserves lexical overlap for clear spans, reduces ROUGE-L degradation.

---

## 3. Future Experiment Directions

### 3.1 Outline-Based Clustering (High Priority)

**Motivation:** The current pipeline does content selection → (decontextualization) → FiC.
The FiC step receives all selected highlights at once and must simultaneously cluster them
AND write the final text. This conflation of two tasks raises the difficulty.
An outline first separates "what to cover" from "how to write it".

**Proposed pipeline:**
```
content_selection
  → ambiguity_highlight (optional)
  → outline_generation        ← NEW
  → cluster_to_outline        ← NEW (or reuse clustering with outline as constraint)
  → iterative_sentence_generation (per outline point)
```

**Outline generation step:**
- Input: selected highlights + (for LFQA) the question.
- Task prompt: "Given these highlights, produce a numbered outline of the key points
  the final summary should cover. Each outline point should correspond to one or two
  summary sentences."
- Output schema (structured mode):
  ```json
  {"outline": [{"point_id": 1, "point_text": "..."}, ...]}
  ```

**Cluster-to-outline step:**
- Input: highlights + generated outline.
- Task prompt: "Assign each highlight to the outline point it best supports."
- Output: `{"assignments": [{"highlight_id": ..., "point_id": ...}]}`

**Benefit for LFQA:** The question can be injected into the outline generation prompt,
guiding the outline toward the answer structure rather than topic coverage — naturally
improving coherence for QA.

**Benefit over current clustering:** The current clustering model has no explicit notion
of summary structure. The outline acts as an inductive bias that forces each cluster
to map to a communicative goal.

**Implementation cost:** Medium. Requires:
- New prompt in `prompts/MDS.json` and `prompts/LFQA.json` for `outline_generation`.
- New subtask `outline_generation` in `subtask_specific_utils.py`.
- New `run_full_pipeline.py` allowed combination:
  `{"content_selection", "ambiguity_highlight", "outline_generation", "cluster_to_outline", "iterative_sentence_generation"}`.
- New config files under `configs/dev/`.

---

### 3.2 Iterative Fixing / Self-Correction Loop

**Motivation:** The pipeline produces outputs in a single forward pass. Errors (broken
citations, repeated information, incomplete coverage) are only detected at evaluation time.
Self-correction — asking the model to critique and rewrite its own output — is well-supported
by the Gemini generation API and can be done within the existing `gemini_chat_call` framework.

**Proposed extension to the dialogue pipeline:**
```
[Turn 1] content_selection
[Turn 2] (optional) ambiguity_highlight
[Turn 3] fusion_in_context  → draft summary
[Turn 4] citation_check     ← NEW: "For each sentence, verify the assigned highlights support it."
[Turn 5] fix_pass           ← NEW: "Rewrite any sentence flagged as unsupported."
```

**Turn 4 citation check prompt:**
```
Review your summary. For each sentence, check whether the highlights you assigned truly
support the sentence text.
Format: [{"sent_id": 1, "is_supported": true/false, "issue": "...or null"}]
```

**Turn 5 fix prompt:**
```
For the sentences you marked as unsupported, rewrite them so they are fully supported by
their assigned highlights. Keep all other sentences unchanged.
```

**Why dialogue mode is the right vehicle:** The draft summary + highlight assignments are
already in the chat session context. No re-sending of documents is needed — the fix pass
costs only the token length of the correction instruction + the revised sentences.

**Measurable benefit:** Track citation recall (LongCite) before and after the fix pass.
Preliminary MDS citation recall: 0.959 (baseline) / 0.983 (decontextualization).
LFQA citation recall: 0.118 / 0.180 — there is significant room to improve LFQA citation recall.

**Implementation cost:** Low. Requires:
- Two new continuation strings in `run_dialogue_pipeline` (`citation_check_continuation`,
  `fix_continuation`).
- New `--iterative-fix` flag on `run_full_pipeline.py`.
- New evaluation: compare LongCite before/after fix within same run.

---

### 3.3 Coherence-Aware Clustering

**Motivation:** The current clustering step groups highlights by semantic similarity (cosine
similarity with a threshold of 0.6 in `subtask_specific_utils.py`, `COSINE_SIMILARITY_THR`).
Semantic similarity does not guarantee discourse coherence: two highlights that are
topically similar may discuss the same event from conflicting perspectives, leading to
incoherent sentences.

**Proposed approach: Discourse Relation–Guided Clustering**

Step 1 — pairwise relation classification (lightweight, can reuse the cosine similarity pass):
- For each pair of candidate highlight groupings, ask the model (or a small classifier):
  "Do these two spans contrast, elaborate, or cause-and-effect each other?"
- Use the predicted relation to adjust the cosine distance:
  - ELABORATION: keep pairs together (lower distance).
  - CONTRAST: keep apart unless explicitly flagged for contrastive sentence generation.
  - CAUSAL: group in temporal/causal order.

Step 2 — coherence-constrained assignment:
- Modify the clustering objective to maximize within-cluster coherence rather than
  just within-cluster similarity.
- Can be implemented as a post-processing step on top of the existing cosine clustering:
  split clusters where a CONTRAST relation is detected between sub-groups.

**Simpler baseline (immediate experiment):**
- Add an `ordering_pass` after clustering: given the cluster-to-summary-sentence assignments,
  ask the model: "Reorder these groups so the final summary flows coherently from beginning
  to end." This is a reranking task, not a generation task, and is cheap.

**Measurable benefit:** Add a topic coherence metric (already implemented as P1:
`evaluation/calc_topic_coherence.py`) to the evaluation suite. Compare topic coherence
scores for clustering-based pipelines vs CoT pipelines vs outline-based pipelines.

**Implementation cost:** Medium-High for full discourse relation clustering, Low for the
ordering_pass baseline.

---

### 3.4 Structured Output + Abstain Refinement

**Motivation:** The structured output mode (P4) uses JSON schema responses to improve
instruction following. The abstain mechanism was designed so the model can signal "I cannot
produce an attributable output" rather than hallucinating. However, abstain rate and
downstream behavior have not yet been systematically measured.

**Proposed experiments:**

1. **Abstain rate analysis:** Count how many FiC outputs are `"ABSTAIN"` across the
   structured output variants. If abstain rate > 5%, investigate whether the instruction
   is too strict or the highlights are too sparse.

2. **Partial abstain:** Instead of whole-output abstain, allow sentence-level abstain:
   ```json
   {"sentences": [{"sentence_id": 1, "sentence_text": "...", "highlight_ids": [2,3], "abstain": false},
                  {"sentence_id": 2, "sentence_text": null, "highlight_ids": [], "abstain": true}]}
   ```
   Sentences with `"abstain": true` are dropped from the final summary. This reduces
   hallucinations for individual sentences without discarding the entire output.

3. **Schema-driven decontextualization output:** The current AH step outputs free text.
   Wrapping it in a structured schema (`{"rewrites": [{"highlight_id": int, "rewritten_text": str}]}`)
   would eliminate the format mismatch (Fix A above) systematically.

---

### 3.5 Model-Agnostic Pipeline and OpenAI Comparison

**Motivation:** All current experiments use Gemini (gemini-2.0-flash for MDS,
gemini-pro-latest for LFQA). The `utils.py` file already supports OpenAI via `openai_call`.
A direct GPT-4o comparison would strengthen the experimental story.

**Proposed experiments:**
- Re-run `full_CoT_pipeline_zeroshot` with `gpt-4o` on both MDS and LFQA dev sets.
- Compare ROUGE-L, repetition, and citation recall across Gemini vs GPT-4o.
- Expected finding: GPT-4o likely has higher ROUGE-L but higher cost; Gemini flash is
  faster and cheaper. Document the tradeoff.

**Config change needed:** Only the `"model_name"` field in each config JSON. No code change.

---

### 3.6 Error Analysis Pipeline

**Motivation:** 5/20 MDS instances and some LFQA instances produced ERROR outputs due to
rate limits. These are not random failures — they tend to cluster on instances where the
model took more retries (longer documents, more highlights). A systematic error analysis
would reveal whether failures correlate with input characteristics.

**Proposed tooling:**
- Add `analysis/error_analysis.py` that:
  - Joins `results.json` with the input data.
  - Computes per-instance stats: num_highlights, total_doc_length, num_sentences_in_gold.
  - Plots ROUGE-L vs num_highlights and vs doc_length.
  - Identifies outlier instances (very low ROUGE-L despite no error).

---

## 4. Experimental Matrix for Next Phase

Priority order based on expected gain vs implementation cost:

| Priority | Experiment | Setting | Expected Δ ROUGE-L | Cost |
|---|---|---|---|---|
| P0 | Fix A+B: LFQA format mismatch + query field | LFQA | +2-4pp | Low |
| P1 | Iterative fix pass in dialogue mode | Both | +1-3pp citation recall | Low |
| P2 | Selective decontextualization (skip unambiguous) | LFQA | +1-2pp | Low |
| P3 | Outline-based clustering | Both | +2-5pp | Medium |
| P4 | Coherence ordering pass (reranking baseline) | Both | +0-1pp coherence | Low |
| P5 | Structured schema for AH output | Both | eliminates format bugs | Low |
| P6 | GPT-4o comparison | Both | — (comparative) | Low ($$) |
| P7 | Full discourse coherence clustering | Both | +1-3pp coherence | High |

---

## 5. Evaluation Suite Completion

The following metrics are implemented but not yet run on all variants:

| Metric | Script | Missing runs |
|---|---|---|
| ROUGE-L | `evaluation/calc_rouge_l.py` | All zeroshot/dialogue/structured variants |
| N-gram repetition | `evaluation/calc_repetition.py` | Same |
| LLM repetition | `evaluation/calc_llm_repetition.py` | All (requires GOOGLE_API_KEY) |
| LongCite citation recall | `evaluation/calc_longcite.py` | All new variants |
| Topic coherence | `evaluation/calc_topic_coherence.py` | All variants |

**Recommendation:** After all 12 variants complete, run all 5 metrics in a single sweep using
a wrapper script `evaluation/run_all_metrics.sh` that iterates over `results/dev/` and calls
each eval script.

---

## 6. Open Questions

1. **Why does decontextualization hurt LFQA but (likely) help MDS?**
   Hypothesis: LFQA highlights are already answer-focused; rewriting them adds noise.
   MDS highlights are topic-breadth selections; rewriting to add referential context helps.
   Experiment: measure AH output quality by comparing rewritten spans to gold `docSpanText`.

2. **Is the cosine similarity threshold (0.6) optimal for both settings?**
   MDS topics are broader; LFQA answers are more focused. LFQA may benefit from a
   higher threshold (more fine-grained clusters) to avoid merging answer points.

3. **Does dialogue mode actually save tokens vs independent calls?**
   The implementation sends the full document in turn 1 and only instructions in subsequent
   turns. Measure actual token counts per instance across dialogue vs non-dialogue runs.

4. **Is abstain correlated with low-quality highlights?**
   If the model abstains on instances with sparse or low-quality content selection outputs,
   the abstain signal could be used as a quality filter or trigger a re-selection pass.
