"""
Command‑line tool to compute simple repetition metrics over model outputs.

This script mirrors the interface of ``calc_rouge_l.py`` so that it can be
plugged into the same evaluation pipeline. It reads a JSON file containing
model predictions (or a dictionary keyed by identifiers) and extracts the
generated text from the ``final_output`` field.  It then computes a token
``n``‑gram repetition rate for each example and produces per‑example and
aggregate summaries.

Repetition rate for an ``n``‑gram is defined as:

``Rep@n = 1.0 - (number of unique n‑grams) / (total number of n‑grams)``

with the edge case that if the input contains fewer than ``n`` tokens the
value is zero.  A higher Rep@n means more repetition.  By default, the script
computes bigram, trigram and four‑gram repetition (``n=2,3,4``).

Usage example::

    python calc_repetition.py --results_json path/to/results.json \
                              --output_dir evaluation_results

This will write two files into ``evaluation_results``:

* ``repetition_per_example.jsonl`` – one JSON object per line with per‑example
  repetition rates and metadata (id, counts etc.).
* ``repetition_summary.json`` – an aggregate summary of macro repetition rates
  across all examples.

The script is intentionally conservative about assumptions on the structure of
``results.json``.  It follows the heuristics used in ``calc_rouge_l.py`` to
find either a list of instances or a dictionary of instances keyed by ids.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Regular expression for tokenization – loosely matches words and contractions.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def _normalize(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def _tokenize(text: str) -> List[str]:
    """Tokenize the input into a list of word tokens."""
    return _TOKEN_RE.findall(_normalize(text))

def repetition_ngram_rate(text: str, n: int = 3) -> float:
    """Compute the repetition rate for token n‑grams in ``text``.

    The repetition rate is defined as 1 − (unique n‑grams / total n‑grams).

    If ``text`` contains fewer than ``n`` tokens or if ``n`` is less than 1,
    the function returns 0.0.
    """
    tokens = _tokenize(text)
    if n <= 0 or len(tokens) < n:
        return 0.0
    total = len(tokens) - n + 1
    # Compute all n‑grams; for small n this is inexpensive.
    ngrams = [tuple(tokens[i:i + n]) for i in range(total)]
    unique_count = len(set(ngrams))
    return 1.0 - (unique_count / total)

def repetition_bundle(text: str, ns: Tuple[int, ...] = (2, 3, 4)) -> Dict[str, float]:
    """Return a dictionary of repetition rates for multiple n values."""
    return {f"rep_{n}": repetition_ngram_rate(text, n) for n in ns}

def extract_candidate_reference(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract the candidate (prediction) and reference fields from an instance.

    This mirrors the logic used in ``calc_rouge_l.py``: the candidate text
    comes from the ``final_output`` field (or any key ending with
    ``final_output``) and the reference comes from ``gold_summary``.  If no
    candidate is found the function returns ``None``.
    """
    # Reference (unused in repetition, but returned for interface compatibility)
    ref = obj.get("gold_summary")

    # Candidate: prefer 'final_output', but fall back to any key ending with that
    cand = None
    if "final_output" in obj:
        cand = obj.get("final_output")
    else:
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith("final_output"):
                cand = v
                break
    # Skip pipeline errors
    if isinstance(cand, str) and cand.startswith("ERROR"):
        cand = None
    return cand, ref

# Typing alias for arbitrary JSON structures
JsonLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

def _looks_like_instance_list(lst: List[Any]) -> bool:
    """Heuristic to decide whether a list contains prediction instances.

    An instance list is a list of dictionaries where at least one element has
    either a ``gold_summary`` or ``final_output`` key.  This logic mirrors
    ``calc_rouge_l.py``.
    """
    if not lst:
        return False
    dict_elems = [x for x in lst if isinstance(x, dict)]
    if not dict_elems:
        return False
    for d in dict_elems[:5]:  # inspect a few
        if "gold_summary" in d or "final_output" in d:
            return True
        if any(isinstance(k, str) and k.endswith("final_output") for k in d.keys()):
            return True
    return False

def _find_instance_list(data: JsonLike) -> Optional[List[Dict[str, Any]]]:
    """Recursively search for a list of instance dictionaries inside arbitrary JSON."""
    # If data is a list, and looks like a list of instances, return it.
    if isinstance(data, list):
        if _looks_like_instance_list(data):
            return [x for x in data if isinstance(x, dict)]
        for x in data:
            found = _find_instance_list(x)
            if found is not None:
                return found
        return None
    # If data is a dict, check common container keys or search nested structures.
    if isinstance(data, dict):
        for key in ("results", "instances", "data", "items", "examples", "outputs"):
            v = data.get(key)
            if isinstance(v, list) and _looks_like_instance_list(v):
                return [x for x in v if isinstance(x, dict)]
        for v in data.values():
            if isinstance(v, list) and _looks_like_instance_list(v):
                return [x for x in v if isinstance(x, dict)]
        for v in data.values():
            if isinstance(v, (dict, list)):
                found = _find_instance_list(v)
                if found is not None:
                    return found
        return None
    return None

def load_instances(path: str) -> List[Dict[str, Any]]:
    """Load prediction instances from a JSON file using robust heuristics.

    The function supports three formats:

    * A list of instance dictionaries at the top level.
    * An object with a key (e.g., ``results``) whose value is a list of
      instances.
    * A dictionary keyed by instance identifiers, where each value is an
      instance dictionary.

    In the last case, all dictionary values will be returned as the list of
    instances.  If none of these structures are found, an error is raised.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: JsonLike = json.load(f)
    # Direct list at top level
    if isinstance(data, list) and _looks_like_instance_list(data):
        return [x for x in data if isinstance(x, dict)]
    # Search for a nested list
    found = _find_instance_list(data)
    if found is not None:
        return found
    # Special case: dictionary keyed by id
    if isinstance(data, dict):
        values = list(data.values())
        dict_values = [v for v in values if isinstance(v, dict)]
        if dict_values:
            for d in dict_values[:5]:
                if "gold_summary" in d or "final_output" in d:
                    return dict_values
                if any(isinstance(k, str) and k.endswith("final_output") for k in d.keys()):
                    return dict_values
        raise ValueError(
            "Unrecognized results.json structure: dict without an instance list. "
            f"Top‑level keys: {sorted(list(data.keys()))}"
        )
    raise ValueError(f"Unrecognized results.json structure: {type(data)}")

@dataclass(frozen=True)
class RepetitionScore:
    """Container for repetition scores for a single example."""
    rep_2: float
    rep_3: float
    rep_4: float

def aggregate_repetition(scores: Iterable[RepetitionScore]) -> Dict[str, float]:
    """Compute macro averages of repetition metrics across all examples."""
    scores_list = list(scores)
    n = len(scores_list)
    if n == 0:
        return {
            "n": 0,
            "rep_2_macro": 0.0,
            "rep_3_macro": 0.0,
            "rep_4_macro": 0.0,
        }
    rep2_mean = sum(s.rep_2 for s in scores_list) / n
    rep3_mean = sum(s.rep_3 for s in scores_list) / n
    rep4_mean = sum(s.rep_4 for s in scores_list) / n
    return {
        "n": n,
        "rep_2_macro": rep2_mean,
        "rep_3_macro": rep3_mean,
        "rep_4_macro": rep4_mean,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute repetition metrics for model outputs.")
    parser.add_argument("--results_json", required=True, help="Path to results.json with predictions.")
    parser.add_argument("--output_dir", default=None, help="Directory to write outputs (defaults to dirname of results_json).")
    parser.add_argument("--max_examples", type=int, default=None, help="Optionally limit the number of examples processed.")
    args = parser.parse_args()

    # Determine output directory.  If not provided, default to the directory
    # containing the results file.  ``os.path.dirname`` returns '' if there is
    # no directory component, so we handle that case by using '.'.
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.results_json) or ".", "metrics", "repetition")
    os.makedirs(out_dir, exist_ok=True)

    # Load instances
    instances = load_instances(args.results_json)
    total_instances = len(instances)
    if args.max_examples is not None:
        instances = instances[: args.max_examples]
    # For reporting and debugging
    print(f"Detected {total_instances} instances in {args.results_json}; processing {len(instances)}")

    # Prepare output paths
    per_example_path = os.path.join(out_dir, "repetition_per_example.jsonl")
    summary_path = os.path.join(out_dir, "repetition_summary.json")

    scores: List[RepetitionScore] = []

    # Determine if input is a dict keyed by id – we need to preserve keys for per‑example output.
    # Read raw JSON to iterate in deterministic order when using dict input.
    with open(args.results_json, "r", encoding="utf-8") as rf:
        raw = json.load(rf)

    skipped = 0
    with open(per_example_path, "w", encoding="utf-8") as f_out:
        if isinstance(raw, dict):
            items = list(raw.items())
            if args.max_examples is not None:
                items = items[: args.max_examples]
            for idx, (top_key, obj) in enumerate(items):
                if not isinstance(obj, dict):
                    continue
                cand, _ = extract_candidate_reference(obj)
                if cand is None:
                    skipped += 1
                    continue
                bundle = repetition_bundle(cand, ns=(2, 3, 4))
                s = RepetitionScore(rep_2=bundle["rep_2"], rep_3=bundle["rep_3"], rep_4=bundle["rep_4"])
                scores.append(s)
                uid = obj.get("unique_id", obj.get("id", top_key))
                rec = {
                    "id": uid,
                    "rep_2": s.rep_2,
                    "rep_3": s.rep_3,
                    "rep_4": s.rep_4,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            # Raw is a list or nested, use loaded instances and enumerate
            for idx, obj in enumerate(instances):
                cand, _ = extract_candidate_reference(obj)
                if cand is None:
                    skipped += 1
                    continue
                bundle = repetition_bundle(cand, ns=(2, 3, 4))
                s = RepetitionScore(rep_2=bundle["rep_2"], rep_3=bundle["rep_3"], rep_4=bundle["rep_4"])
                scores.append(s)
                uid = obj.get("unique_id", obj.get("id", idx))
                rec = {
                    "id": uid,
                    "rep_2": s.rep_2,
                    "rep_3": s.rep_3,
                    "rep_4": s.rep_4,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if skipped:
        print(f"Skipped {skipped} instances with ERROR/None output.")

    # Compute aggregate metrics and write summary
    summary = aggregate_repetition(scores)
    summary["n_skipped"] = skipped
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    print("Wrote:", per_example_path)
    print("Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()