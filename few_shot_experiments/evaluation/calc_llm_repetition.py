"""
calc_llm_repetition.py
======================

This script provides an example of how to compute a repetition metric for
generated answers or summaries using a large language model (LLM) as the
evaluator.  Instead of counting repeated n‑grams, the model is asked to
score how repetitive a given piece of text is according to a rubric.  The
rubric is inspired by published LLM–as‑a‑judge guidelines, which state that
concise writing should avoid repeating ideas or phrases【116462658301640†L390-L398】【837838729122035†L749-L761】.

The script accepts a JSON‐formatted input file where each entry contains the
model output under the key `final_output`.  It iterates over these
instances, sends each generated text to the Gemini model and parses the
rating returned by the LLM.  At the end, it writes per‑example ratings and
macro/micro averages to the specified output directory.

Usage:
    python calc_llm_repetition.py --results_json path/to/results.json \
        --output_dir path/to/outdir --model gemini-2.0-flash

Requirements:
    - Install the Google Generative AI SDK:
        pip install google-generativeai
    - Export your API key as an environment variable `GOOGLE_API_KEY`.

Notes:
    - This script does **not** call the LLM automatically in the
      environment used by this exercise because the API key is not
      provided here.  It is intended as a template for running locally
      with the appropriate credentials.  See the README for details.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # If the package is not available, we handle this in main().


def _load_instances(path: str) -> List[Dict[str, Any]]:
    """Load instances from a JSON file.

    The input file can be either a list of instance dicts or a dict
    mapping identifiers to instance dicts.  The function returns a list of
    instance dicts in both cases.  Each instance is expected to contain
    a `final_output` field containing the generated text to be evaluated.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    raise ValueError("Unexpected JSON structure in results file.")


def _build_prompt(candidate: str) -> str:
    """Construct the evaluation prompt for the LLM.

    The prompt instructs the judge model to assess how repetitive the
    candidate text is on an ordinal scale.  The rubric follows the
    conciseness guidelines described in research on LLM evaluation, where
    a fully concise answer avoids repeating ideas and includes only
    essential information【837838729122035†L749-L761】.  A minor repetition still counts
    as a conciseness issue【837838729122035†L759-L761】.
    """
    return f"""
            You are a repetition evaluator.  You will be given a piece of generated text.
            
            Rate how repetitive the text is on a scale from 0 to 3 according to this
            rubric:
            
            * 3 – Not repetitive: The text does not repeat words, phrases or ideas
              unnecessarily.  It is concise and includes only essential information
              without redundancy【837838729122035†L749-L757】.
            * 2 – Slightly repetitive: The text contains minor repetitions of words or
              ideas but remains mostly concise.  There may be one or two
              unnecessary restatements【837838729122035†L759-L761】.
            * 1 – Highly repetitive: The text repeats information or phrases multiple
              times, making it verbose or redundant.
            * 0 – Extremely repetitive: The text is dominated by repeated phrases or
              sentences, significantly detracting from clarity and conciseness.
            
            Return only the numeric score (0, 1, 2, or 3) and nothing else.
            
            Generated text:
            
    """.strip() + candidate.strip() + "\n"  # Ensures the candidate is appended on new line


def _parse_rating(response_text: str) -> Optional[int]:
    """Parse the numeric rating from the LLM response.

    The LLM is instructed to return only a number between 0 and 3.  This
    function extracts the first integer in the response.  If no valid
    integer is found, it returns None.
    """
    match = re.search(r"\b([0-3])\b", response_text)
    if match:
        return int(match.group(1))
    return None


def evaluate_repetition(
    candidate: str,
    model: Any,
    num_retries: int = 10,
    base_sleep: float = 30.0,
) -> int:
    """Send the candidate text to the LLM and return the repetition score.

    Retries with exponential backoff on 429 rate-limit errors.
    """
    prompt = _build_prompt(candidate)
    last_err: Optional[BaseException] = None
    for attempt in range(num_retries):
        try:
            result = model.generate_content(prompt, stream=False)
            full_text = ""
            if hasattr(result, "text") and isinstance(result.text, str):
                full_text = result.text.strip()
            else:
                try:
                    full_text = result.candidates[0].content.parts[0].text.strip()  # type: ignore
                except Exception:
                    pass
            score = _parse_rating(full_text)
            if score is None:
                raise RuntimeError(f"Could not parse LLM response: {full_text}")
            return score
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt)
            print(f"[calc_llm_repetition] attempt {attempt+1}/{num_retries} failed: {e}. Retrying in {sleep_s:.0f}s…")
            time.sleep(sleep_s)
    raise RuntimeError(f"evaluate_repetition failed after {num_retries} retries: {last_err}")


def aggregate_scores(scores: Iterable[int]) -> Dict[str, float]:
    """Compute macro average for the repetition scores.

    The scores are ordinal values in [0, 3].  We normalise by dividing by
    3 to get a 0–1 scale when computing averages.  The function returns
    the macro mean of the raw scores and the normalised scores.
    """
    scores_list = list(scores)
    if not scores_list:
        return {"n": 0, "macro_mean": 0.0, "macro_mean_normalised": 0.0}
    n = len(scores_list)
    mean_raw = sum(scores_list) / n
    mean_norm = mean_raw / 3
    return {"n": n, "macro_mean": mean_raw, "macro_mean_normalised": mean_norm}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repetition using an LLM judge")
    parser.add_argument("--results_json", required=True, help="Path to the model outputs JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to write per-example and summary results")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name to use for evaluation")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit the number of examples (for testing)")
    args = parser.parse_args()

    if genai is None:
        raise ImportError("google-generativeai is not installed.  Please install it via pip and try again.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable to your API key.")

    # Configure the generative AI client
    genai.configure(api_key=api_key)
    model_name = args.model if args.model.startswith("models/") else f"models/{args.model}"
    model = genai.GenerativeModel(model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    instances = _load_instances(args.results_json)
    if args.max_examples is not None:
        instances = instances[: args.max_examples]

    per_example_path = os.path.join(args.output_dir, "llm_repetition_per_example.jsonl")
    summary_path = os.path.join(args.output_dir, "llm_repetition_summary.json")

    scores: List[int] = []
    skipped = 0
    with open(per_example_path, "w", encoding="utf-8") as f_out:
        for idx, inst in enumerate(instances):
            candidate = inst.get("final_output") or inst.get("prediction") or ""
            if isinstance(candidate, str) and candidate.startswith("ERROR"):
                skipped += 1
                continue
            # Evaluate the candidate with the LLM
            score = evaluate_repetition(candidate, model)
            scores.append(score)
            record = {
                "id": inst.get("unique_id", inst.get("id", idx)),
                "repetition_score": score,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    if skipped:
        print(f"Skipped {skipped} instances with ERROR output.")

    summary = aggregate_scores(scores)
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    print(f"Processed {len(scores)} instances.  Summary saved to {summary_path}.")


if __name__ == "__main__":
    main()