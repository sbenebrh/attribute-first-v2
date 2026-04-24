import os
import argparse
from utils import *
from run_script import main as main_func
from run_iterative_sentence_generation import main as iterative_sent_gen_main
from subtask_specific_utils import (
    get_data, get_subtask_funcs, get_subtask_prompt_structures,
    construct_non_demo_part, construct_prompts,
    parse_ambiguity_highlight_response,
    convert_ambiguity_highlight_results_to_pipeline_format,
    parse_FiC_response,
    convert_FiC_CoT_results_to_pipeline_format,
)
import logging
from copy import deepcopy
from pathlib import Path
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def run_subtask(full_configs, subtask_name, curr_outdir, original_args_dict, indir_alignments=None):
    """
    full_configs: full pipeline configs
    subtask_name: curr subtask name ("content_selection", "clustering", "iterative_sentence_generation", or "fusion_in_context"). Note: "ambiguity_highlight" is handled inside this file as a passthrough step (no model call).
    curr_outdir: curr subtask's outdir
    original_args_dict: full otiginal args
    indir_alignments: path to previous subtask's alignments (for content_selection this should be None or a pre-defined alignment path)
    """
    curr_configs = [elem for elem in full_configs if elem['subtask']==subtask_name][0]
    curr_configs.update({"outdir":curr_outdir,
                         "indir_alignments":indir_alignments})
    func_args = deepcopy(original_args_dict) # initialize args that didn't appear in the subask's configs file to default values
    func_args.update(curr_configs)
    if subtask_name!="iterative_sentence_generation":
        main_func(argparse.Namespace(**func_args))
    else:
        iterative_sent_gen_main(argparse.Namespace(**func_args))


def main(args):
    original_args_dict = deepcopy(args.__dict__) 
    with open(args.config_file, 'r') as f1:
        full_configs= json.loads(f1.read())
    
    # make sure all configs for all subtasks are supplied
    allowed = [
        {"content_selection", "clustering", "iterative_sentence_generation"},
        {"content_selection", "fusion_in_context"},
        {"content_selection", "topic_outline_fusion"},
        {"content_selection", "topic_cluster_fusion"},
        {"content_selection", "ambiguity_highlight", "clustering", "iterative_sentence_generation"},
        {"content_selection", "ambiguity_highlight", "fusion_in_context"},
        {"content_selection", "ambiguity_highlight", "topic_outline_fusion"},
        {"content_selection", "ambiguity_highlight", "topic_cluster_fusion"},
        {"content_selection", "fusion_in_context_v2"},
        {"content_selection", "ambiguity_highlight", "fusion_in_context_v2"},
    ]
    if set([elem['subtask'] for elem in full_configs]) not in allowed:
        raise Exception(
            'configs must be one of: '
            '(1) "content_selection", "clustering", "iterative_sentence_generation"; '
            '(2) "content_selection", "fusion_in_context"; '
            '(3) "content_selection", "topic_outline_fusion"; '
            '(4) "content_selection", "topic_cluster_fusion"; '
            '(5) add "ambiguity_highlight" between content_selection and any downstream step.'
        )
    # make sure all config files share the same split and setting
    all_splits, all_settings = [], []
    for elem in full_configs:
        with open(elem['config_file'], 'r') as f1:
            curr_configs = json.loads(f1.read())
            all_splits.append(curr_configs['split'])
            all_settings.append(curr_configs['setting'])
    if len(set(all_splits))!=1 or len(set(all_settings))!=1:
        raise Exception("all subtasks must have the same split (test/dev) and the same setting (MDS/LFQA)")

    # define and create outdir
    pipeline_subdir = Path(args.config_file).stem
    outdir = args.outdir if args.outdir else  f"results/{all_splits[0]}/{all_settings[0]}/{pipeline_subdir}"
    logging.info(f"saving results to {outdir}")
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    intermediate_outdir = os.path.join(outdir, "itermediate_results") # subdir with results of intermediate subtasks
    path = Path(intermediate_outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    # --- Dialogue mode: run the whole CoT pipeline as a single multi-turn chat ---
    if getattr(args, "dialogue_mode", False):
        subtasks = set(e["subtask"] for e in full_configs)
        if "clustering" in subtasks or "iterative_sentence_generation" in subtasks:
            raise ValueError("--dialogue-mode only supports CoT (fusion_in_context) pipelines.")
        run_dialogue_pipeline(args, full_configs, original_args_dict, outdir, intermediate_outdir)
        return

    # content selection
    content_selection_outdir = os.path.join(intermediate_outdir, "content_selection")
    logging.info("running content seletion:")
    run_subtask(full_configs=full_configs, 
                subtask_name="content_selection", 
                curr_outdir=content_selection_outdir, 
                original_args_dict=original_args_dict, 
                indir_alignments=args.indir_alignments)

    # By default, downstream steps consume the alignments produced by content_selection
    prev_alignments_path = os.path.join(content_selection_outdir, "pipeline_format_results.json")

    if "ambiguity_highlight" in [elem["subtask"] for elem in full_configs]:
        ambiguity_highlight_outdir = os.path.join(intermediate_outdir, "ambiguity_highlight")
        logging.info("running ambiguity highlight:")
        run_subtask(
            full_configs=full_configs,
            subtask_name="ambiguity_highlight",
            curr_outdir=ambiguity_highlight_outdir,
            original_args_dict=original_args_dict,
            indir_alignments=prev_alignments_path,
        )
        # Downstream steps should now consume the updated alignments
        prev_alignments_path = os.path.join(ambiguity_highlight_outdir, "pipeline_format_results.json")

    # fully-decomposted pipeline (not CoT)
    if "clustering" in [elem['subtask'] for elem in full_configs]:
        # clustering
        clustering_outdir = os.path.join(intermediate_outdir, "clustering")
        logging.info("running clustering:")
        run_subtask(full_configs=full_configs,
                    subtask_name="clustering",
                    curr_outdir=clustering_outdir,
                    original_args_dict=original_args_dict,
                    indir_alignments=prev_alignments_path)
        # iterative_sentence_generation
        logging.info("running final iterative sentence generation:")
        run_subtask(full_configs=full_configs,
                    subtask_name="iterative_sentence_generation",
                    curr_outdir=outdir,
                    original_args_dict=original_args_dict,
                    indir_alignments=os.path.join(clustering_outdir, "pipeline_format_results.json")) # the alignments are the outputs of the previous subtask (clustering)

    # CoT approach pipeline (fusion_in_context, topic_outline_fusion, or topic_cluster_fusion)
    else:
        fic_subtasks = {"fusion_in_context", "fusion_in_context_v2", "topic_outline_fusion", "topic_cluster_fusion"}
        fic_subtask = next(
            s for s in [e["subtask"] for e in full_configs] if s in fic_subtasks
        )
        logging.info(f"running CoT-style fusion ({fic_subtask}):")
        run_subtask(full_configs=full_configs,
                    subtask_name=fic_subtask,
                    curr_outdir=outdir,
                    original_args_dict=original_args_dict,
                    indir_alignments=prev_alignments_path)







def _subtask_cfg(full_configs, subtask_name):
    matches = [e for e in full_configs if e["subtask"] == subtask_name]
    return matches[0] if matches else None


def _build_subtask_args(full_configs, subtask_name, original_args_dict, curr_outdir, indir_alignments=None):
    cfg = _subtask_cfg(full_configs, subtask_name)
    func_args = deepcopy(original_args_dict)
    func_args.update(cfg)
    func_args["outdir"] = curr_outdir
    func_args["indir_alignments"] = indir_alignments
    return argparse.Namespace(**func_args)


def _load_subtask_prompt_dict(args):
    indir_prompt = args.indir_prompt if hasattr(args, "indir_prompt") and args.indir_prompt else f"prompts/{args.setting}.json"
    with open(indir_prompt, "r") as f:
        return json.loads(f.read())


def run_dialogue_pipeline(args, full_configs, original_args_dict, outdir, intermediate_outdir):
    """Run the CoT pipeline as a multi-turn dialogue per instance.

    Instead of N independent LLM calls (one per subtask), each instance shares
    a single ChatSession.  The first turn sends the full content-selection
    prompt (with documents + demos).  Subsequent turns send only the task
    instruction, relying on the model's conversation context — dramatically
    reducing repeated token costs.

    Supports pipelines with or without the ambiguity_highlight step.
    """
    has_ah = "ambiguity_highlight" in [e["subtask"] for e in full_configs]
    cs_outdir = os.path.join(intermediate_outdir, "content_selection")
    ah_outdir = os.path.join(intermediate_outdir, "ambiguity_highlight") if has_ah else None
    Path(cs_outdir).mkdir(parents=True, exist_ok=True)
    if ah_outdir:
        Path(ah_outdir).mkdir(parents=True, exist_ok=True)

    # --- Build args for each subtask ---
    cs_args = _build_subtask_args(full_configs, "content_selection", original_args_dict,
                                   curr_outdir=cs_outdir,
                                   indir_alignments=args.indir_alignments)
    cs_args = update_args(cs_args)

    fic_args_ns = _build_subtask_args(full_configs, "fusion_in_context", original_args_dict,
                                       curr_outdir=outdir, indir_alignments=None)
    fic_args_ns = update_args(fic_args_ns)

    if has_ah:
        ah_args = _build_subtask_args(full_configs, "ambiguity_highlight", original_args_dict,
                                       curr_outdir=ah_outdir, indir_alignments=None)
        ah_args = update_args(ah_args)

    # --- Load prompt dicts for all subtasks ---
    cs_prompt_dict = _load_subtask_prompt_dict(cs_args)
    fic_prompt_dict = _load_subtask_prompt_dict(fic_args_ns)

    # --- Build CS prompts (full: demos + docs + instruction) ---
    cs_prompt_structures = get_subtask_prompt_structures(
        cs_prompt_dict, cs_args.setting, "content_selection", cs_args.CoT,
        cs_args.always_with_question if hasattr(cs_args, "always_with_question") else False
    )
    cs_data_args = deepcopy(cs_args)
    cs_data_args.indir_alignments = args.indir_alignments
    _, cs_alignments_dict = get_data(cs_data_args)

    cs_used_demos, cs_prompts, cs_additional = construct_prompts(
        prompt_dict=cs_prompt_dict,
        alignments_dict=cs_alignments_dict,
        n_demos=cs_args.n_demos,
        debugging=cs_args.debugging if hasattr(cs_args, "debugging") else False,
        merge_cross_sents_highlights=cs_args.merge_cross_sents_highlights if hasattr(cs_args, "merge_cross_sents_highlights") else False,
        specific_prompt_details=cs_prompt_structures,
        tkn_counter=get_token_counter(cs_args.model_name),
        no_highlights=True,
        cut_surplus=cs_args.cut_surplus if hasattr(cs_args, "cut_surplus") else False,
        prct_surplus=cs_args.prct_surplus if hasattr(cs_args, "prct_surplus") else None,
    )

    # Continuation prompts for AH and FiC (instruction-only, no document context)
    ah_continuation = None
    if has_ah:
        ah_prompt_dict = _load_subtask_prompt_dict(ah_args)
        ah_instruction = ah_prompt_dict.get("instruction-ambiguity-highlight", "")
        ah_answer_prefix = ah_prompt_dict.get("answer_ambiguity_highlight_prompt", "")
        ah_answer_format = ah_prompt_dict.get("answer_ambiguity_highlight_format", "")
        ah_continuation = (
            "Now, based on the highlights you identified above, please perform the following task:\n\n"
            f"{ah_instruction}\n\n{ah_answer_prefix}{ah_answer_format}"
        )

    fic_instruction = fic_prompt_dict.get("instruction-FiC-CoT", fic_prompt_dict.get("instruction-FiC", ""))
    fic_answer_prefix = fic_prompt_dict.get("answer_FiC-CoT_prompt", fic_prompt_dict.get("answer_FiC_prompt", ""))
    fic_continuation = (
        "Now, using the highlights from the previous step(s), please perform the following task:\n\n"
        f"{fic_instruction}\n\n{fic_answer_prefix}"
    )

    model_name = cs_args.model_name
    temperature = cs_args.temperature if hasattr(cs_args, "temperature") else 0.1
    num_retries = cs_args.num_retries if hasattr(cs_args, "num_retries") else 3

    cs_parse_fn, cs_pipeline_fn = get_subtask_funcs("content_selection")
    ah_parse_fn = parse_ambiguity_highlight_response if has_ah else None
    fic_parse_fn, fic_pipeline_fn = get_subtask_funcs("FiC")

    cs_results, ah_results, fic_results = {}, {}, {}

    logging.info("[dialogue] running per-instance chat sessions ...")
    for uid, cs_prompt in tqdm(cs_prompts.items()):
        try:
            session = create_chat_session(model_name)

            # Turn 1: content selection
            cs_raw = None
            for _ in range(num_retries):
                try:
                    cs_raw = gemini_chat_call(session, cs_prompt,
                                              temperature=temperature)
                    cs_parsed = cs_parse_fn(cs_raw, cs_prompt)
                    if cs_parsed is not None:
                        break
                except Exception:
                    cs_raw = None
            if cs_raw is None:
                logging.warning(f"[dialogue] CS failed for {uid}")
                continue
            cs_results[uid] = cs_parsed

            if has_ah:
                # Turn 2: ambiguity highlight
                ah_raw = None
                for _ in range(num_retries):
                    try:
                        ah_raw = gemini_chat_call(session, ah_continuation,
                                                   temperature=temperature)
                        ah_parsed = ah_parse_fn(ah_raw, ah_continuation)
                        if ah_parsed is not None:
                            break
                    except Exception:
                        ah_raw = None
                ah_results[uid] = ah_parsed if ah_raw is not None else {}

            # Turn 3: FiC
            fic_raw = None
            for _ in range(num_retries):
                try:
                    fic_raw = gemini_chat_call(session, fic_continuation,
                                               temperature=temperature)
                    fic_parsed = fic_parse_fn(fic_raw, fic_continuation)
                    if fic_parsed is not None:
                        break
                except Exception:
                    fic_raw = None
            if fic_raw is not None:
                fic_results[uid] = fic_parsed
            else:
                logging.warning(f"[dialogue] FiC failed for {uid}")

        except Exception as exc:
            logging.error(f"[dialogue] instance {uid} failed: {exc}")

    # Convert and save CS intermediate results
    cs_pipeline = cs_pipeline_fn(cs_results, cs_alignments_dict)
    save_results(cs_outdir, cs_used_demos, cs_results, pipeline_format_results=cs_pipeline)

    # If AH present: convert using the CS pipeline as base, then save
    if has_ah and ah_results:
        ah_pipeline = convert_ambiguity_highlight_results_to_pipeline_format(
            ah_results, cs_pipeline)
        save_results(ah_outdir, [], ah_results, pipeline_format_results=ah_pipeline)
        base_for_fic = ah_pipeline
    else:
        base_for_fic = cs_pipeline

    # Convert and save FiC final results
    if fic_results:
        fic_pipeline = fic_pipeline_fn(fic_results, base_for_fic)
        save_results(outdir, cs_used_demos, fic_results, pipeline_format_results=fic_pipeline)
    else:
        logging.error("[dialogue] No FiC results to save!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--config-file', type=str, required=True, help='path to json config file.')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('--model-name', type=str, default="gemini-pro", help='model name')
    argparser.add_argument('--n-demos', type=int, default=2, help='number of ICL examples (default 2)')
    argparser.add_argument('--num-retries', type=int, default=1, help='number of retries of running the model.')
    argparser.add_argument('--temperature', type=float, default=0.2, help='temperature of generation')
    argparser.add_argument('--debugging', action='store_true', default=False, help='if debugging mode.')
    argparser.add_argument('--merge-cross-sents-highlights', action='store_true', default=False, help='whether to merge consecutive highlights that span across several sentences.')    
    argparser.add_argument('--CoT', action='store_true', default=False, help='whether to use a CoT approach (relevant for FiC and clustering).')    
    argparser.add_argument('--cut-surplus', action='store_true', default=False, help='whether to cut surplus text from prompts (in subtask with given highlights - everything after last highlight, and in tasks without - last prct_surplus sentences).')
    argparser.add_argument('--prct-surplus', type=float, default=None, help='for subtasks without given highlights (e.g. content_selection, e2e_only_setting, or ALCE) - what percentage of top document sents to drop in cases when the prompts are too long.')
    argparser.add_argument('--always-with-question', action='store_true', default=False, help='relevant for LFQA - whether to always add the question (also to clustering and FiC)')
    argparser.add_argument('--num-demo-changes', type=int, default=4, help='number of changing demos when the currently-chosen set of demos returns an ERROR.')
    argparser.add_argument('--rerun', action='store_true', default=False, help='if need to rerun on instances that had errors')
    argparser.add_argument('--rerun-path', type=str, default=None, help='path to rerun on (where the results are)')
    argparser.add_argument('--rerun-n-demos', type=int, default=None, help='new n_demos for rerun in cases when the current n_demos doesnt work.')
    argparser.add_argument('--rerun-temperature', type=float, default=None, help='new temperature for rerun in cases when the current temperature doesnt work.')
    argparser.add_argument('--no-prefix', action='store_true', default=False, help='ablation study where the prefix is not add.')
    argparser.add_argument('--structured-output', action='store_true', default=False,
                           help='Use Gemini JSON mode (response_schema) for each subtask. '
                                'Improves instruction following and enables abstain in FiC.')
    argparser.add_argument('--dialogue-mode', action='store_true', default=False,
                           help='Run the CoT pipeline as a multi-turn dialogue (prefix-caching style). '
                                'Each instance shares one ChatSession: turn 1 = full CS prompt, '
                                'subsequent turns = instruction-only (no document re-send). '
                                'Only compatible with CoT (fusion_in_context) pipelines.')
    args = argparser.parse_args()
    main(args)