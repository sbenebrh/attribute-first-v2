"""Thin compatibility shim — re-exports everything from the split modules."""
from prompt_utils import *
from response_parsers import *
from pipeline_converters import *


def get_subtask_funcs(subtask, structured_output: bool = False):
    """returns the relevant functions for each task - parse_response, convert_to_pipeline_style"""
    if subtask == "FiC":
        parse_fn = parse_FiC_structured_response if structured_output else parse_FiC_response
        return parse_fn, convert_FiC_CoT_results_to_pipeline_format
    elif subtask == "content_selection":
        parse_fn = parse_content_selection_structured_response if structured_output else parse_content_selection_response
        return parse_fn, convert_content_selection_results_to_pipeline_format
    elif subtask == "clustering":
        return parse_clustering_response, convert_clustering_results_to_pipeline_format
    elif subtask == "e2e_only_setting":
        return parse_e2e_only_setting_response, convert_e2e_only_setting_to_pipeline_format
    elif subtask == "ALCE":
        return parse_ALCE_response, convert_ALCE_to_pipeline_format
    elif subtask == "ambiguity_highlight":
        parse_fn = parse_ambiguity_highlight_structured_response if structured_output else parse_ambiguity_highlight_response
        return parse_fn, convert_ambiguity_highlight_results_to_pipeline_format
    elif subtask in ("topic_outline_fusion", "topic_cluster_fusion", "FiC_v2"):
        parse_fn = parse_FiC_structured_response if structured_output else parse_FiC_response
        return parse_fn, convert_FiC_CoT_results_to_pipeline_format
    else:
        raise Exception(f"{subtask} is not yet supported")
