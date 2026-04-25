from utils import *
from subtask_specific_utils import *
from schemas import SUBTASK_SCHEMAS
import logging
from pathlib import Path
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def main(args):

    if not args.config_file and (not args.setting or not args.subtask or not args.split):
        raise Exception("If no config file is passed, then must explicitly determine setting, subtask, and split.")
    
    # if config_file is passed - load its arguments
    if args.config_file:
        args = update_args(args)

    # get the outdir
    CoT_suffix = "_CoT" if args.CoT else ""
    merged_cross_sent_highlights_suffix = "_merged_cross_sents_sep" if args.merge_cross_sents_highlights else ""
    outdir = args.outdir if args.outdir else  f"results/{args.split}/{args.setting}/{args.subtask}{CoT_suffix}{merged_cross_sent_highlights_suffix}"
    logging.info(f"saving results to {outdir}")

    # create outdir if doesn't exist
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)

    # save as args to json file
    with open(os.path.join(outdir, "args.json"), 'w') as f1:
        f1.write(json.dumps(args.__dict__, indent=2))
    
    prompt_dict, alignments_dict = get_data(args)

    # get subtask related functions
    structured_output = getattr(args, "structured_output", False)
    parse_response_func, convert_to_pipeline_style_func = get_subtask_funcs(
        args.subtask, structured_output=structured_output)

    if args.cut_surplus and args.subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS and not args.prct_surplus:
        logging.error(f"when passing --cut-surplus with the subtask {args.subtask} - you need to also pass --prct-surplus")
        exit(1)

    # get subtask related prompt structures (instructions and answer-related)
    specific_prompt_details = get_subtask_prompt_structures(prompt_dict=prompt_dict, setting=args.setting, subtask=args.subtask, CoT=args.CoT, always_with_question=args.always_with_question, structured_output=getattr(args, "structured_output", False))
    
    used_demos, prompts, additional_data = construct_prompts(prompt_dict=prompt_dict, 
                                                             alignments_dict=alignments_dict, 
                                                             n_demos=args.n_demos, 
                                                             debugging=args.debugging, 
                                                             merge_cross_sents_highlights=args.merge_cross_sents_highlights, 
                                                             specific_prompt_details=specific_prompt_details,
                                                             tkn_counter=get_token_counter(args.model_name),
                                                             no_highlights=args.subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS,
                                                             cut_surplus=args.cut_surplus,
                                                             prct_surplus=args.prct_surplus)

    structured_output = getattr(args, "structured_output", False)
    response_schema = SUBTASK_SCHEMAS.get(args.subtask) if structured_output else None

    # Rerun mode: only re-prompt instances that previously had ERROR outputs.
    existing_results = {}
    rerun = getattr(args, "rerun", False)
    rerun_path = getattr(args, "rerun_path", None)
    if rerun:
        path_to_load = rerun_path or os.path.join(outdir, "results.json")
        if os.path.exists(path_to_load):
            with open(path_to_load, 'r') as f_existing:
                existing_results = json.load(f_existing)
            error_ids = {k for k, v in existing_results.items()
                         if str(v.get("final_output", "")).startswith("ERROR")}
            prompts = {k: v for k, v in prompts.items() if k in error_ids}
            logging.info(f"Rerun mode: {len(prompts)} instances with ERROR outputs to retry.")
        else:
            logging.warning(f"Rerun path not found ({path_to_load}), running all instances.")

    n_demos_to_use = getattr(args, "rerun_n_demos", None) or args.n_demos
    temperature_to_use = getattr(args, "rerun_temperature", None) or args.temperature

    responses = prompt_model(prompts=prompts,
                             model_name=args.model_name,
                             parse_response_fn=parse_response_func,
                             num_retries=args.num_retries,
                             temperature=temperature_to_use,
                             response_schema=response_schema,
                             output_max_length=getattr(args, "output_max_length", 4096))

    ############# SAVE #############
    # combine results with all instances' details
    final_results = dict(existing_results)  # seed with existing results (empty if not rerun)
    final_results.update({key: dict() for key in responses.keys()})
    for instance_name, resp in responses.items():
        final_results[instance_name].update(additional_data[instance_name])
        final_results[instance_name]['gold_summary'] = [elem['response'] for elem in alignments_dict if elem['unique_id']==instance_name][0]
        if args.subtask=="FiC" and args.CoT and not "alignments" in resp.keys(): # when there is an ERROR in the FiC-CoT
            final_results[instance_name]["alignments"] = []
        final_results[instance_name].update(resp)

    pipeline_format_results = None
    if convert_to_pipeline_style_func:
        try:
            pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict)
        except:
            logging.info("The coversion to the pipeline format wasn't successful - please check.")

    # save
    save_results(outdir, used_demos, final_results, pipeline_format_results)        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--config-file', type=str, default=None, help='path to json config file. Should come instead of all the other parameters')
    argparser.add_argument('--split', type=str, default=None, help='data split (test or dev)')
    argparser.add_argument('--setting', type=str, default=None, help='setting (MDS or LFQA)')
    argparser.add_argument('--subtask', type=str, default=None, help='subtask to run (content_selection, clustering, FiC, e2e_only_setting, ALCE)')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
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
    argparser.add_argument('--structured-output', action='store_true', default=False,
                           help='Use Gemini JSON mode (response_schema) for each subtask. '
                                'Improves instruction following. For FiC, also enables abstain.')
    argparser.add_argument('--rerun', action='store_true', default=False,
                           help='Re-run only instances that had ERROR outputs.')
    argparser.add_argument('--rerun-path', type=str, default=None,
                           help='Path to existing results.json to patch (defaults to outdir/results.json).')
    argparser.add_argument('--rerun-n-demos', type=int, default=None,
                           help='Override n_demos for the rerun.')
    argparser.add_argument('--rerun-temperature', type=float, default=None,
                           help='Override temperature for the rerun.')
    argparser.add_argument('--num-demo-changes', type=int, default=4,
                           help='Number of demo changes on ERROR before giving up.')
    argparser.add_argument('--max-examples', type=int, default=None,
                           help='Cap total examples processed.')
    argparser.add_argument('--no-prefix', action='store_true', default=False,
                           help='Ablation: omit the prefix.')
    argparser.add_argument('--dialogue-mode', action='store_true', default=False,
                           help='Run pipeline as multi-turn chat (FiC only).')
    args = argparser.parse_args()
    main(args)