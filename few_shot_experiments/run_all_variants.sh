#!/bin/bash
# Run all pipeline variants on MDS and LFQA dev sets.
# Skips already-completed runs (where results.json exists with 0 errors).
# Usage: cd few_shot_experiments && bash run_all_variants.sh

set -e
source ../.venv/bin/activate

EVAL_SCRIPT_ROUGE="python3 evaluation/calc_rouge_l.py"
EVAL_SCRIPT_REP="python3 evaluation/calc_repetition.py"

is_complete() {
    local results_json=$1
    [[ -f "$results_json" ]] || return 1
    local errors
    errors=$(python3 -c "
import json, sys
with open('$results_json') as f:
    d = json.load(f)
n = len(d)
errors = sum(1 for v in d.values() if isinstance(v,dict) and str(v.get('final_output','')).startswith('ERROR'))
print(errors)
" 2>/dev/null || echo "999")
    [[ "$errors" == "0" ]]
}

run_and_eval() {
    local config=$1
    local extra_flags=$2
    local stem
    stem=$(basename "$config" .json)
    local setting
    setting=$(python3 -c "import json; d=json.load(open('$config')); print(d[0]['config_file'].split('/')[2])" 2>/dev/null || echo "unknown")
    local split="dev"
    local outdir="results/${split}/${setting}/${stem}"

    if is_complete "${outdir}/results.json"; then
        echo "SKIP (already complete): $stem ($setting)"
        return
    fi

    echo "========================================="
    echo "Running: $stem ($setting) flags='$extra_flags'"
    echo "========================================="
    python3 run_full_pipeline.py --config-file "$config" $extra_flags

    echo "--- ROUGE-L ---"
    $EVAL_SCRIPT_ROUGE --results_json "${outdir}/results.json" --output_dir "${outdir}/" || true
    echo "--- Repetition ---"
    $EVAL_SCRIPT_REP --results_json "${outdir}/results.json" --output_dir "${outdir}/" || true
}

# MDS variants
run_and_eval configs/dev/MDS/full_CoT_pipeline_zeroshot.json ""
run_and_eval configs/dev/MDS/full_decontextualization_CoT_pipeline_zeroshot.json ""
run_and_eval configs/dev/MDS/full_CoT_pipeline_dialogue.json "--dialogue-mode"
run_and_eval configs/dev/MDS/full_decontextualization_CoT_pipeline_dialogue.json "--dialogue-mode"
run_and_eval configs/dev/MDS/full_CoT_pipeline_structured.json ""
run_and_eval configs/dev/MDS/full_decontextualization_CoT_pipeline_structured.json ""

# LFQA variants
run_and_eval configs/dev/LFQA/full_CoT_pipeline_zeroshot.json ""
run_and_eval configs/dev/LFQA/full_decontextualization_CoT_pipeline_zeroshot.json ""
run_and_eval configs/dev/LFQA/full_CoT_pipeline_dialogue.json "--dialogue-mode"
run_and_eval configs/dev/LFQA/full_decontextualization_CoT_pipeline_dialogue.json "--dialogue-mode"
run_and_eval configs/dev/LFQA/full_CoT_pipeline_structured.json ""
run_and_eval configs/dev/LFQA/full_decontextualization_CoT_pipeline_structured.json ""

echo "All variants completed!"
