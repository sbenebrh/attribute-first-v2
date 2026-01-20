
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_data
import auto_scorer as longcite_scorer


logging.basicConfig(level=logging.INFO)


def load_instances_jsonl(indir: str) -> list[dict]:
    """Load pipeline_format_results.json which is jsonl (one instance per line)."""
    with open(indir, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_longcite_js(raw_instance: dict, per_sent_inputs: list[dict]) -> dict:
    """Adapt our pipeline instance + get_data() output to LongCite auto_scorer input schema."""
    query = raw_instance.get("query") or raw_instance.get("topic") or ""

    # Our pipeline commonly stores the final text summary in `response`.
    prediction = (
        raw_instance.get("response")
        or raw_instance.get("final_output")
        or raw_instance.get("prediction")
        or ""
    )

    # Make sure we never pass None to auto_scorer (it does .strip()).
    query = "" if query is None else str(query)
    prediction = "" if prediction is None else str(prediction)

    statements = []
    for elem in per_sent_inputs:
        statement = elem.get("sentence") or ""
        attribution = elem.get("attribution") or {}
        # LongCite expects: citation = [{"cite": "..."}, ...]
        citations = [
            {"cite": txt}
            for txt in attribution.values()
            if isinstance(txt, str) and txt.strip()
        ]
        statements.append({"statement": str(statement), "citation": citations})

    return {"query": query, "prediction": prediction, "statements": statements}


def safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0


def main(args):
    if (not args.indir or not args.outdir) and not args.indir_outdir_json:
        raise Exception("must pass either --indir and --outdir or --indir-outdir-json")

    if not args.indir_outdir_json:
        indir_outdir_list = [{"indir": args.indir, "outdir": args.outdir}]
    else:
        with open(args.indir_outdir_json, "r") as f:
            indir_outdir_list = json.loads(f.read())

    # IMPORTANT: LongCite auto_scorer expects a global `GPT_MODEL`.
    # We set it dynamically so the scorer works with Gemini.
    longcite_scorer.GPT_MODEL = args.model

    for run in indir_outdir_list:
        curr_indir = run["indir"]
        curr_outdir = run["outdir"]

        results_csv = os.path.join(curr_outdir, "results.csv")
        avg_json = os.path.join(curr_outdir, "avg_results.json")
        if not args.force_rerun and os.path.exists(results_csv) and os.path.exists(avg_json):
            logging.info(
                f"LongCite scores already exist in {curr_outdir}. "
                f"Skipping (use --force-rerun to recompute)."
            )
            continue

        logging.info(f"Calculating LongCite for {curr_indir}")
        logging.info(f"Saving to {curr_outdir}")

        # create outdir if doesn't exist
        Path(curr_outdir).mkdir(parents=True, exist_ok=True)

        # Load raw instances (for query/prediction) and aligned per-sentence inputs (for citations).
        raw_instances = load_instances_jsonl(curr_indir)
        inputs, instances_unique_ids, _full_instance_stats = get_data(
            curr_indir,
            debug=bool(args.debug_get_data),
            max_debug_instances=int(args.debug_max_instances),
            max_debug_sents=int(args.debug_max_sents),
        )

        if len(raw_instances) != len(inputs) or len(instances_unique_ids) != len(inputs):
            raise RuntimeError(
                f"Size mismatch: raw_instances={len(raw_instances)} inputs={len(inputs)} "
                f"unique_ids={len(instances_unique_ids)}"
            )

        max_n = min(args.max_examples, len(inputs)) if args.max_examples is not None else len(inputs)

        rows = []
        recalls, precisions, f1s = [], [], []
        prompt_tokens_total = 0
        completion_tokens_total = 0

        for i in range(max_n):
            uid = instances_unique_ids[i]
            raw = raw_instances[i]
            per_sent = inputs[i]

            js = to_longcite_js(raw, per_sent)

            try:
                scored = longcite_scorer.get_citation_score(js, max_statement_num=args.max_statement_num)
                rec = float(scored.get("citation_recall", 0.0))
                prec = float(scored.get("citation_precision", 0.0))
                f1 = float(scored.get("citation_f1", 0.0))

                usage = scored.get("gpt_usage", {}) or {}
                p_tok = int(usage.get("prompt_tokens", 0))
                c_tok = int(usage.get("completion_tokens", 0))

                prompt_tokens_total += p_tok
                completion_tokens_total += c_tok

                recalls.append(rec)
                precisions.append(prec)
                f1s.append(f1)

                rows.append(
                    {
                        "unique_id": uid,
                        "citation_recall": rec,
                        "citation_precision": prec,
                        "citation_f1": f1,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "n_statements": len(js.get("statements", [])),
                    }
                )
            except Exception as e:
                logging.exception(f"LongCite scoring failed for unique_id={uid}")
                rows.append(
                    {
                        "unique_id": uid,
                        "citation_recall": np.nan,
                        "citation_precision": np.nan,
                        "citation_f1": np.nan,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "n_statements": len(js.get("statements", [])),
                        "error": str(e),
                    }
                )

        avg_results = {
            "citation_recall": round(safe_mean(recalls), 4),
            "citation_precision": round(safe_mean(precisions), 4),
            "citation_f1": round(safe_mean(f1s), 4),
            "prompt_tokens_total": int(prompt_tokens_total),
            "completion_tokens_total": int(completion_tokens_total),
            "num_instances_scored": int(max_n),
            "model": args.model,
            "max_statement_num": args.max_statement_num,
        }

        with open(avg_json, "w") as f:
            f.write(json.dumps(avg_results, indent=2))

        pd.DataFrame(rows).to_csv(results_csv, index=False)

        logging.info(f"Done. Wrote: {results_csv}")
        logging.info(f"Done. Wrote: {avg_json}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="LongCite evaluation over pipeline_format_results.json (jsonl).")
    argparser.add_argument("-i", "--indir", type=str, default=None, help="path to pipeline_format_results.json (jsonl)")
    argparser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="output directory (writes results.csv + avg_results.json)",
    )
    argparser.add_argument(
        "--indir-outdir-json",
        type=str,
        default=None,
        help="path to json file to enable running several variants (list of {indir,outdir})",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="models/gemini-2.0-flash",
        help="LLM model id for the LongCite scorer (used by llm_api.query_llm)",
    )
    argparser.add_argument(
        "--max-statement-num",
        type=int,
        default=None,
        help="If set, evaluate only the first N statements per instance (for cost control)",
    )
    argparser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="If set, evaluate only the first N instances in the file (for smoke tests)",
    )
    argparser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="recalculate even if results already exist in outdir",
    )
    argparser.add_argument(
        "--debug-get-data",
        action="store_true",
        default=False,
        help="Enable verbose debug prints inside utils.get_data()",
    )
    argparser.add_argument(
        "--debug-max-instances",
        type=int,
        default=1,
        help="When --debug-get-data is set, print debug for at most N instances",
    )
    argparser.add_argument(
        "--debug-max-sents",
        type=int,
        default=3,
        help="When --debug-get-data is set, print debug for at most N sentences per instance",
    )

    args = argparser.parse_args()
    main(args)