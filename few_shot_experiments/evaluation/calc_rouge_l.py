import re
from dataclasses import dataclass
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize(text))

def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # keep b shortest to reduce memory
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0]
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]

@dataclass(frozen=True)
class RougeL:
    lcs_len: int
    cand_len: int
    ref_len: int
    precision: float
    recall: float
    f1: float

def rouge_l(candidate: str, reference: str) -> RougeL:
    c = _tokenize(candidate)
    r = _tokenize(reference)
    lcs = _lcs_len(c, r)

    p = (lcs / len(c)) if c else 0.0
    rec = (lcs / len(r)) if r else 0.0
    f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
    return RougeL(lcs, len(c), len(r), p, rec, f1)



def extract_candidate_reference(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ref = obj.get("gold_summary")

    if "final_output" in obj:
        cand = obj.get("final_output")
    else:
        # defensive fallback: first key ending with 'final_output'
        cand = None
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith("final_output"):
                cand = v
                break

    # Treat abstained instances as empty (score 0) rather than crashing.
    if isinstance(cand, str) and cand.strip().upper() == "ABSTAIN":
        cand = ""

    # Skip pipeline errors — return None so callers can filter them out.
    if isinstance(cand, str) and cand.startswith("ERROR"):
        cand = None

    return cand, ref


JsonLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

def _looks_like_instance_list(lst: List[Any]) -> bool:
    """Heuristic: list of dicts where at least one element has expected keys."""
    if not lst:
        return False
    dict_elems = [x for x in lst if isinstance(x, dict)]
    if not dict_elems:
        return False

    # if any of first few dicts has keys we expect, treat it as instance list
    for d in dict_elems[:5]:
        if "gold_summary" in d or "final_output" in d:
            return True
        if any(isinstance(k, str) and k.endswith("final_output") for k in d.keys()):
            return True
    return False

def _find_instance_list(data: JsonLike) -> Optional[List[Dict[str, Any]]]:
    """Search recursively for a list of dict instances inside arbitrary JSON."""
    if isinstance(data, list):
        if _looks_like_instance_list(data):
            return [x for x in data if isinstance(x, dict)]
        for x in data:
            found = _find_instance_list(x)
            if found is not None:
                return found
        return None

    if isinstance(data, dict):
        # Common container keys (repo-dependent, so include more)
        for key in ("results", "instances", "data", "items", "examples", "outputs"):
            v = data.get(key)
            if isinstance(v, list) and _looks_like_instance_list(v):
                return [x for x in v if isinstance(x, dict)]

        # Any top-level list value that looks right
        for v in data.values():
            if isinstance(v, list) and _looks_like_instance_list(v):
                return [x for x in v if isinstance(x, dict)]

        # Recurse into nested dict/list values
        for v in data.values():
            if isinstance(v, (dict, list)):
                found = _find_instance_list(v)
                if found is not None:
                    return found
        return None

    return None

def load_instances(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data: JsonLike = json.load(f)

    if isinstance(data, list) and _looks_like_instance_list(data):
        return [x for x in data if isinstance(x, dict)]

    found = _find_instance_list(data)
    if found is not None:
        return found
    # Special-case: top-level dict keyed by ids (e.g., test0/test1/...) where values are instances.
    if isinstance(data, dict):
        values = list(data.values())
        dict_values = [v for v in values if isinstance(v, dict)]
        if dict_values:
            # If at least one value dict looks like an instance, treat all dict values as instances.
            for d in dict_values[:5]:
                if "gold_summary" in d or "final_output" in d:
                    return dict_values
                if any(isinstance(k, str) and k.endswith("final_output") for k in d.keys()):
                    return dict_values

        raise ValueError(
            "Unrecognized results.json structure: dict without an instance list. "
            f"Top-level keys: {sorted(list(data.keys()))}"
        )

    raise ValueError(f"Unrecognized results.json structure: {type(data)}")


def aggregate(scores: Iterable[RougeL]) -> Dict[str, float]:
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return {"n": 0, "macro_f1": 0.0, "micro_f1": 0.0}

    macro_p = sum(s.precision for s in scores) / n
    macro_r = sum(s.recall for s in scores) / n
    macro_f1 = sum(s.f1 for s in scores) / n

    sum_lcs = sum(s.lcs_len for s in scores)
    sum_c = sum(s.cand_len for s in scores)
    sum_r = sum(s.ref_len for s in scores)

    micro_p = (sum_lcs / sum_c) if sum_c else 0.0
    micro_r = (sum_lcs / sum_r) if sum_r else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    return {
        "n": n,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.results_json), "metrics", "rouge_l")
    os.makedirs(out_dir, exist_ok=True)

    instances = load_instances(args.results_json)
    print(f"Detected {len(instances)} instances in {args.results_json}")

    if args.max_examples is not None:
        instances = instances[:args.max_examples]

    per_example_path = os.path.join(out_dir, "rouge_l_per_example.jsonl")
    scores = []

    with open(args.results_json, "r", encoding="utf-8") as rf:
        raw = json.load(rf)

    skipped = 0
    with open(per_example_path, "w", encoding="utf-8") as f:
        if isinstance(raw, dict):
            items = list(raw.items())
            if args.max_examples is not None:
                items = items[:args.max_examples]

            for idx, (top_key, obj) in enumerate(items):
                if not isinstance(obj, dict):
                    continue
                cand, ref = extract_candidate_reference(obj)
                if cand is None:
                    skipped += 1
                    continue
                s = rouge_l(cand, ref or "")
                scores.append(s)

                uid = obj.get("unique_id", obj.get("id", top_key))
                rec = {
                    "id": uid,
                    "rougeL_precision": s.precision,
                    "rougeL_recall": s.recall,
                    "rougeL_f1": s.f1,
                    "lcs_len": s.lcs_len,
                    "cand_len": s.cand_len,
                    "ref_len": s.ref_len,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for idx, obj in enumerate(instances):
                cand, ref = extract_candidate_reference(obj)
                if cand is None:
                    skipped += 1
                    continue
                s = rouge_l(cand, ref or "")
                scores.append(s)

                uid = obj.get("unique_id", obj.get("id", idx))
                rec = {
                    "id": uid,
                    "rougeL_precision": s.precision,
                    "rougeL_recall": s.recall,
                    "rougeL_f1": s.f1,
                    "lcs_len": s.lcs_len,
                    "cand_len": s.cand_len,
                    "ref_len": s.ref_len,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if skipped:
        print(f"Skipped {skipped} instances with ERROR/None output.")

    summary = aggregate(scores)
    summary["n_skipped"] = skipped
    with open(os.path.join(out_dir, "rouge_l_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", per_example_path)
    print("Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()


