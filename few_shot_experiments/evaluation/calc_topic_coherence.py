"""
calc_topic_coherence.py
=======================
Topic-coherence evaluation using Gemini-as-judge.

For each generated summary, the LLM assigns a topic label to every sentence.
We then compute:
  - interleaving_score: fraction of triplets (i, i+1, i+2) where topic[i]==topic[i+2]
    and topic[i]!=topic[i+1]  (e.g. politics→sport→politics = 1 violation)
  - topic_switches: total number of adjacent sentence pairs with different topics
  - unique_topics: number of distinct topics in the summary

A lower interleaving_score means the model groups related content together.

Usage:
    python calc_topic_coherence.py \
        --results_json results/dev/MDS/full_CoT_pipeline_1/results.json \
        --output_dir results/dev/MDS/full_CoT_pipeline_1/topic_coherence/
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import llm_api

logging.basicConfig(level=logging.INFO)


def _load_instances(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    raise ValueError(f"Unrecognised results file structure: {type(data)}")


def _get_summary(instance: Dict[str, Any]) -> str:
    for key in ("final_output", "response", "prediction", "summary"):
        val = instance.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter – good enough for evaluation."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


_TOPIC_PROMPT = """\
You are a topic-labelling assistant. Given a multi-sentence text, assign a
concise TOPIC LABEL (2-4 words, ALL CAPS) to each sentence. Return the labels
as a numbered list, one per line, in the same order as the sentences.

Use the SAME label for sentences that discuss the same theme. Prefer broad
labels (e.g. "ECONOMIC POLICY", "PUBLIC HEALTH", "SPORTS RESULTS") over
sentence-specific paraphrases.

Example output:
1. ECONOMIC POLICY
2. ECONOMIC POLICY
3. PUBLIC HEALTH
4. SPORTS RESULTS
5. ECONOMIC POLICY

Text:
{sentences}

Return ONLY the numbered list.
"""


def _label_sentences(sentences: List[str], model: str) -> Optional[List[str]]:
    if not sentences:
        return []
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = _TOPIC_PROMPT.format(sentences=numbered)
    response = llm_api.query_llm(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_new_tokens=512,
    )
    if not response:
        return None
    labels: List[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+)$", line)
        if m:
            labels.append(m.group(1).strip().upper())
    if len(labels) != len(sentences):
        return None
    return labels


def _compute_metrics(labels: List[str]) -> Dict[str, Any]:
    n = len(labels)
    if n == 0:
        return {"n_sentences": 0, "unique_topics": 0, "topic_switches": 0,
                "interleaving_score": 0.0, "labels": []}
    unique_topics = len(set(labels))
    switches = sum(1 for i in range(n - 1) if labels[i] != labels[i + 1])
    # Interleaving: A→B→A pattern count / max possible
    interleaving = 0
    max_triplets = max(n - 2, 0)
    for i in range(n - 2):
        if labels[i] == labels[i + 2] and labels[i] != labels[i + 1]:
            interleaving += 1
    interleaving_score = interleaving / max_triplets if max_triplets > 0 else 0.0
    return {
        "n_sentences": n,
        "unique_topics": unique_topics,
        "topic_switches": switches,
        "interleaving_count": interleaving,
        "interleaving_score": round(interleaving_score, 4),
        "labels": labels,
    }


def safe_mean(values: List[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0


def main(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    per_example_path = os.path.join(args.output_dir, "topic_coherence_per_example.jsonl")
    summary_path = os.path.join(args.output_dir, "topic_coherence_summary.json")

    if not args.force_rerun and os.path.exists(summary_path):
        logging.info(f"Results already exist in {args.output_dir}. Use --force-rerun to recompute.")
        return

    instances = _load_instances(args.results_json)
    max_n = min(args.max_examples, len(instances)) if args.max_examples else len(instances)
    logging.info(f"Scoring {max_n} instances from {args.results_json}")

    rows = []
    interleaving_scores, switch_counts, unique_topic_counts = [], [], []

    with open(per_example_path, "w") as fout:
        for i, inst in enumerate(instances[:max_n]):
            uid = inst.get("unique_id") or inst.get("id") or str(i)
            summary = _get_summary(inst)
            if not summary:
                logging.warning(f"[{uid}] empty summary, skipping")
                continue

            sentences = _split_sentences(summary)
            labels = _label_sentences(sentences, model=args.model)

            if labels is None:
                logging.warning(f"[{uid}] label parse failed, skipping")
                continue

            metrics = _compute_metrics(labels)
            metrics["unique_id"] = uid

            fout.write(json.dumps(metrics) + "\n")
            rows.append(metrics)
            interleaving_scores.append(metrics["interleaving_score"])
            switch_counts.append(metrics["topic_switches"])
            unique_topic_counts.append(metrics["unique_topics"])

            if (i + 1) % 5 == 0:
                logging.info(f"  scored {i+1}/{max_n}")

    avg = {
        "n": len(rows),
        "macro_interleaving_score": round(safe_mean(interleaving_scores), 4),
        "macro_topic_switches": round(safe_mean(switch_counts), 4),
        "macro_unique_topics": round(safe_mean(unique_topic_counts), 4),
        "model": args.model,
    }
    with open(summary_path, "w") as f:
        json.dump(avg, f, indent=2)

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "topic_coherence_per_example.csv"), index=False)
    logging.info(f"Done. Summary: {avg}")
    logging.info(f"Wrote: {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Topic coherence evaluation using LLM-as-judge.")
    ap.add_argument("--results_json", required=True, help="Path to results.json")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--model", default="models/gemini-2.0-flash", help="Gemini model id")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--force_rerun", action="store_true", default=False)
    args = ap.parse_args()
    main(args)
