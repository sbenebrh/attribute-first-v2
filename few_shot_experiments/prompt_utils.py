from utils import *
import numpy as np
from typing import List, Dict


def get_indir_paths(args):
    if type(args)!=dict: # i.e. argparse
        args = args.__dict__
    indir_alignments = args['indir_alignments'] if args['indir_alignments'] else f"../data/{args['setting']}/{args['split']}.json"
    indir_prompt = args['indir_prompt'] if args['indir_prompt'] else f"prompts/{args['setting']}.json"
    return indir_prompt, indir_alignments

def get_data(args):
    indir_prompt, indir_alignments = get_indir_paths((args))

    alignments_dict = []
    with open(indir_alignments, 'r') as f1:
        alignments_dict = [json.loads(line) for line in f1.readlines()]

    with open(indir_prompt, 'r') as f1:
        prompt_dict = json.loads(f1.read())

    max_examples = getattr(args, "max_examples", None)
    if max_examples is not None:
        alignments_dict = alignments_dict[:max_examples]

        if isinstance(prompt_dict, list):
            prompt_dict = prompt_dict[:max_examples]

        elif isinstance(prompt_dict, dict) and len(alignments_dict) > 0 and "id" in alignments_dict[0]:
            keep_ids = {a["id"] for a in alignments_dict}
            prompt_dict = {k: v for k, v in prompt_dict.items() if k in keep_ids}

    return prompt_dict, alignments_dict

def get_subtask_prompt_structures(prompt_dict : Dict, setting: str, subtask : str, CoT : bool, always_with_question : bool, structured_output: bool = False) -> Dict:
    """returns the subtask relevant prompt structures (instruction and answer-related and demo_prompt)"""

    demo_prompt = prompt_dict["demo_prompt_content_selection"] if setting=="LFQA" and (subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS or always_with_question) else prompt_dict["demo_prompt"]

    with_question_suffix = "-with-question" if always_with_question and setting=="LFQA" else ""

    if subtask == "FiC":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_FiC{CoT_suffix}_prompt"],
                                  "answer_FiC_planning_prompt":prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt":prompt_dict["answer_highlights_fusion_prompt"]}
        instruction_prompt = prompt_dict[f"instruction-FiC{CoT_suffix}{with_question_suffix}"]
    elif subtask == "content_selection":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_content_selection_prompt"],
                                  "answer_content_selection_format":prompt_dict["answer_content_selection_format"]}
        instr_key = "instruction-content-selection-structured" if structured_output and "instruction-content-selection-structured" in prompt_dict else "instruction-content-selection"
        instruction_prompt = prompt_dict[instr_key]
    elif subtask == "clustering":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_clustering{CoT_suffix}_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_clustering_CoT_prompt_intermediate":prompt_dict["answer_clustering-CoT_prompt_intermediate"],
                                  "answer_clustering_format":prompt_dict["answer_clustering_format"],
                                  "answer_clustering_CoT_format":prompt_dict["answer_clustering-CoT_format"]}
        instruction_prompt = prompt_dict[f"instruction-clustering{with_question_suffix}"]
    elif subtask == "e2e_only_setting":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_e2e_only_setting_prompt"]}
        instruction_prompt = prompt_dict["instruction-e2e-only-setting"]
    elif subtask == "ALCE":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_ALCE_prompt"],
                                  "answer_ALCE_format":prompt_dict["answer_ALCE_format"]}
        instruction_prompt = prompt_dict["instruction-ALCE"]
    elif subtask == "ambiguity_highlight" :
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_ambiguity_highlight_prompt"],
                                  "answer_ambiguity_highlight_format":prompt_dict["answer_ambiguity_highlight_format"]}
        instr_key = "instruction-ambiguity-highlight-structured" if structured_output and "instruction-ambiguity-highlight-structured" in prompt_dict else "instruction-ambiguity-highlight"
        instruction_prompt = prompt_dict[instr_key]
    elif subtask in ("topic_outline_fusion", "topic_cluster_fusion"):
        answer_related_prompts = {"answer_prompt": prompt_dict["answer_FiC-CoT_prompt"],
                                  "answer_FiC_planning_prompt": prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt": prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt": prompt_dict["answer_highlights_fusion_prompt"]}
        instr_key = "topic-outline" if subtask == "topic_outline_fusion" else "topic-cluster"
        instruction_prompt = prompt_dict[f"instruction-{instr_key}-CoT{with_question_suffix}"]
    elif subtask == "FiC_v2":
        answer_related_prompts = {"answer_prompt": prompt_dict["answer_FiC-CoT_prompt"],
                                  "answer_FiC_planning_prompt": prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt": prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt": prompt_dict["answer_highlights_fusion_prompt"]}
        if always_with_question and setting == "LFQA":
            instruction_prompt = prompt_dict["instruction-FiC-CoT-with-question-v2"]
        else:
            instruction_prompt = prompt_dict["instruction-FiC-CoT"]
    else:
        raise Exception(f"{subtask} is not yet supported")

    return {"answer_related_prompts" : answer_related_prompts,
            "instruction_prompt" : instruction_prompt,
            "demo_prompt" : demo_prompt}

def construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus: bool = False, prct_surplus: float = 0.25):
        """prct_surplus: the percentage of last sentences to consider as surplus and remove in subtasks without given highlights (e.g., content_selection or end-to-end or ALCE)"""
        highlight_start_tkn = "{HS}"
        highlight_end_tkn="{HE}"
        topic_name = instance['unique_id']
        highlighted_texts = get_highlighted_doc(docs={elem['documentFile']:elem['rawDocumentText'] for elem in instance['documents'] if elem['documentFile']},
                                                highlights=instance['set_of_highlights_in_context'],
                                                highlight_start_tkn = highlight_start_tkn,
                                                highlight_end_tkn=highlight_end_tkn,
                                                merge_cross_sents_highlights=merge_cross_sents_highlights,
                                                doc_sents={elem['documentFile']:elem['documentText'] for elem in instance['documents'] if elem['documentFile']})

        if cut_surplus:
            if no_highlights:
                highlighted_texts = {elem['documentFile']:"".join(elem['documentText'][:max(int(len(elem['documentText'])*(1-prct_surplus)), 5)]) for elem in instance['documents'] if elem['documentFile']}
            else:
                highlighted_texts = {doc_name:rmv_txt_after_last_highlight(doc_text, highlight_end_tkn) for doc_name,doc_text in highlighted_texts.items()}
                highlighted_texts = {key:value for key,value in highlighted_texts.items() if value}

        docs_order = [{"doc_name":doc_name, "doc_text":doc_text} for doc_name,doc_text in highlighted_texts.items()]
        eval_item = {"docs":[{'text':dct["doc_text"]} for dct in docs_order]}

        _q = instance.get("query") or instance.get("question") or ""
        if _q:
            eval_item["question"] = _q

        curr_prompt, curr_highlight_list = make_demo(
            item=eval_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights,
            test=True
        )

        return curr_prompt, curr_highlight_list, topic_name, docs_order

def construct_prompts(prompt_dict : Dict, alignments_dict : List[Dict], n_demos : int, debugging : bool, merge_cross_sents_highlights : bool, specific_prompt_details : Dict, tkn_counter: Dict, no_highlights : bool = False, cut_surplus : bool = False, prct_surplus: float = None):
    DEMO_HEADER = "### DEMO EXAMPLES (DO NOT ANSWER) ###\n"
    TARGET_HEADER = "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"

    train_ids = np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False) if not debugging else [0,2]
    head_prompt = DEMO_HEADER if n_demos and n_demos > 0 else ""
    head_prompt_shorter = DEMO_HEADER if n_demos and n_demos > 0 else ""
    used_demos = []
    for train_id in train_ids:
        train_item = prompt_dict["demos"][train_id]
        used_demos.append(train_item)

        curr_prompt_demo, _ = make_demo(
            item=train_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt += curr_prompt_demo
        head_prompt += prompt_dict["demo_sep"]

        train_item_shorter = {key:[{doc_key:elem['shorter_text'] if doc_key=="text" else doc_value
                                    for doc_key,doc_value in elem.items()} for elem in value] if key=="docs" else value for key,value in train_item.items()}

        curr_prompt_demo_shorter, _ = make_demo(
            item=train_item_shorter, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt_shorter += curr_prompt_demo_shorter
        head_prompt_shorter += prompt_dict["demo_sep"]

    if debugging:
        alignments_dict = alignments_dict[:3]

    final_prompts, additional_data = {}, {}
    for instance in alignments_dict:
        curr_prompt, curr_highlight_list, topic_name, docs_order = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights)

        if cut_surplus:
            curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=prct_surplus)
            final_prompts[topic_name] = head_prompt_shorter + TARGET_HEADER + curr_prompt
        elif tkn_counter["tkn_counter"].token_count(head_prompt + curr_prompt)>=tkn_counter["tkn_max_limit"]:
            prct_surplus_lst = [0.5, 0.6, 0.7]
            for curr_prct_surplus in prct_surplus_lst:
                curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=curr_prct_surplus)
                if tkn_counter["tkn_counter"].token_count(head_prompt_shorter + curr_prompt)<tkn_counter["tkn_max_limit"]:
                    break
            final_prompts[topic_name] = head_prompt_shorter + TARGET_HEADER + curr_prompt
        else:
            curr_highlight_list_shorter, docs_order_shorter = [], []
            final_prompts[topic_name] = head_prompt + TARGET_HEADER + curr_prompt
        highlighted_docs = [{"doc_name":elem["doc_name"],
                             "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order]
        non_highlighted_docs = [{"doc_name":elem["doc_name"],
                                 "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order]
        highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                     "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order_shorter]
        non_highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                         "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order_shorter]

        no_highlights_prfx = "gold_" if no_highlights else ""

        additional_data[topic_name] = {"non_highlighted_docs":non_highlighted_docs,
                                       f"{no_highlights_prfx}highlighted_docs":highlighted_docs,
                                       f"{no_highlights_prfx}highlights":curr_highlight_list,
                                       f"non_highlighted_docs_shorter":non_highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlighted_docs_shorter":highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlights_shorter":curr_highlight_list_shorter}

        _q = instance.get("query") or instance.get("question") or ""
        if _q:
            additional_data[topic_name].update({"question": _q})
    return used_demos, final_prompts, additional_data
