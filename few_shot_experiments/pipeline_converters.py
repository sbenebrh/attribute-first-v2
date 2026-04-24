from utils import *
from copy import deepcopy
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from response_parsers import _get_spacy_nlp, adapt_highlights_to_doc_alignments

COSINE_SIMILARITY_THR = 0.6
sent_transformer_model_name = "sentence-transformers/paraphrase-distilroberta-base-v1"
sent_transformer_tokenizer = AutoTokenizer.from_pretrained(sent_transformer_model_name)
sent_transformer_model = AutoModel.from_pretrained(sent_transformer_model_name)


def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def _highlights_in_context_from_spans(doc_name, doc_text, spans, nlp, doc_sents=None):
    """Convert a list[str] spans (verbatim) into the pipeline highlight objects."""
    if not spans:
        return []
    return get_set_of_highlights_in_context_content_selection(
        doc_name=doc_name,
        doc_text=doc_text,
        highlights=spans,
        nlp=nlp,
        doc_sents=doc_sents,
    )


def _docspan_ranges(hic_item: dict):
    """Return list[(start,end)] ranges for a single highlight-in-context item."""
    offsets = hic_item.get("docSpanOffsets")
    if not offsets:
        return []
    ranges = []
    for off in offsets:
        if isinstance(off, (list, tuple)) and len(off) == 2:
            try:
                s, e = int(off[0]), int(off[1])
            except Exception:
                continue
            if s < e:
                ranges.append((s, e))
    return ranges


def _ranges_overlap(a, b):
    """True iff half-open intervals a=[s1,e1), b=[s2,e2) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def _merge_set_of_highlights_in_context(existing_list, new_list):
    """Merge HA highlights into existing HS highlights, deduplicating and removing overlaps."""
    existing_list = existing_list if isinstance(existing_list, list) else []
    new_list = new_list if isinstance(new_list, list) else []

    def _key(h):
        try:
            return json.dumps(h, sort_keys=True)
        except Exception:
            return str(h)

    merged = []
    seen = set()
    ranges_by_doc = {}

    for h in existing_list:
        if not isinstance(h, dict):
            continue
        k = _key(h)
        if k in seen:
            continue
        seen.add(k)
        merged.append(h)
        doc = h.get("documentFile")
        if doc:
            ranges_by_doc.setdefault(doc, []).extend(_docspan_ranges(h))

    kept_new = []
    for h in new_list:
        if not isinstance(h, dict):
            continue
        k = _key(h)
        if k in seen:
            continue

        doc = h.get("documentFile")
        new_ranges = _docspan_ranges(h)

        if doc and new_ranges:
            existing_ranges = ranges_by_doc.get(doc, [])
            if any(_ranges_overlap(r, er) for r in new_ranges for er in existing_ranges):
                continue
            ranges_by_doc.setdefault(doc, []).extend(new_ranges)

        seen.add(k)
        kept_new.append(h)
        merged.append(h)

    return kept_new, merged


def convert_ambiguity_highlight_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    """Merge ambiguity-added highlights (HA) into the existing HS highlights."""
    nlp = _get_spacy_nlp()
    pipeline_style_data = []

    for unique_id, value in results.items():
        curr_original_inst = deepcopy([e for e in alignments_dict if e.get("unique_id") == unique_id][0])

        carry_gold_fields = [
            "gold_highlights",
            "gold_highlighted_docs",
            "gold_highlights_shorter",
            "gold_highlighted_docs_shorter",
            "gold_summary",
        ]
        for k in carry_gold_fields:
            if k in value:
                curr_original_inst[k] = value[k]

        non_highlighted_docs = value.get("non_highlighted_docs")
        if not isinstance(non_highlighted_docs, list):
            non_highlighted_docs = []

        doc_file_to_raw = {}
        doc_file_to_sents = {}
        for d in curr_original_inst.get("documents", []):
            if not isinstance(d, dict):
                continue
            df = d.get("documentFile")
            if not df:
                continue
            doc_file_to_raw[df] = d.get("rawDocumentText")
            if "documentText" in d:
                doc_file_to_sents[df] = d.get("documentText")

        ha_final = value.get("final_output", {})
        ha_hic_all = []

        if isinstance(ha_final, dict):
            for doc_key, spans in ha_final.items():
                if not isinstance(spans, list) or not doc_key.startswith("Document ["):
                    continue
                try:
                    doc_i = int(doc_key.replace("Document [", "").replace("]", "")) - 1
                except Exception:
                    continue
                if doc_i < 0 or doc_i >= len(non_highlighted_docs):
                    continue

                doc_name = non_highlighted_docs[doc_i].get("doc_name") if isinstance(non_highlighted_docs[doc_i], dict) else None
                if not doc_name:
                    continue

                raw = doc_file_to_raw.get(doc_name)
                if not raw:
                    continue

                doc_sents = doc_file_to_sents.get(doc_name)
                ha_hic_all.extend(_highlights_in_context_from_spans(doc_name, raw, spans, nlp, doc_sents=doc_sents))

        existing_hic = curr_original_inst.get("set_of_highlights_in_context", [])
        # Filter out context spans that are too short to be useful (avoids single-token
        # fragments like "He" or "Rome." polluting the FiC highlight list).
        MIN_CONTEXT_SPAN_CHARS = 30
        ha_hic_filtered = [h for h in ha_hic_all if len(h.get("docSpanText", "")) >= MIN_CONTEXT_SPAN_CHARS]
        kept_new_hic, merged_hic = _merge_set_of_highlights_in_context(existing_hic, ha_hic_filtered)

        # Merge filtered context spans into set_of_highlights_in_context so FiC can use them.
        # Previously kept separate (Fix A) to avoid tiny fragments; length filter above prevents that.
        curr_original_inst["set_of_highlights_in_context"] = merged_hic
        curr_original_inst["context_set_of_highlights_in_context"] = kept_new_hic
        curr_original_inst["new_set_of_highlights_in_context"] = kept_new_hic

        if "full_model_response" in value:
            curr_original_inst["ambiguity_full_model_response"] = value["full_model_response"]
        if "removed_nested_or_duplicate" in value:
            curr_original_inst["removed_nested_or_duplicate"] = value["removed_nested_or_duplicate"]

        pipeline_style_data.append(curr_original_inst)

    return pipeline_style_data


def get_set_of_highlights_in_context_content_selection(doc_name, doc_text, highlights, nlp, doc_sents, *args, **kwargs):
    if not doc_sents:
        doc_sents = [sent.text for sent in nlp(doc_text).sents]
    sents_idx_limits = [find_substring(doc_text, sent) for sent in doc_sents]

    highlights_in_context_list = []
    for h in highlights:
        if not rmv_spaces_and_punct(h):
            continue
        h = h.strip()
        h_idx_limits = find_substring(doc_text, h)

        assert h_idx_limits[0]!=-1, "didn't find highlight"

        relevant_sents_i = sorted([i for i,lims in enumerate(sents_idx_limits) if set(range(lims[0],lims[1])).intersection(set(range(h_idx_limits[0], h_idx_limits[1]))) and rmv_spaces_and_punct(doc_sents[i])])
        relevant_sents_i = sorted([i for i in relevant_sents_i if not any(rmv_spaces_and_punct(doc_sents[i]) in rmv_spaces_and_punct(doc_sents[j]) and rmv_spaces_and_punct(doc_sents[i])!=rmv_spaces_and_punct(doc_sents[j]) for j in relevant_sents_i)])

        if len(relevant_sents_i)>1:
            for sentence_index in relevant_sents_i:
                curr_sents_idx_limits = sents_idx_limits[sentence_index]
                curr_h_idx_limits = get_consecutive_subspans(sorted(list(set(range(h_idx_limits[0],h_idx_limits[1]+1)).intersection(range(curr_sents_idx_limits[0], curr_sents_idx_limits[1]+1)))))
                docSentCharIdx = str(curr_sents_idx_limits[0])
                docSentText = doc_text[curr_sents_idx_limits[0]:curr_sents_idx_limits[1]]
                docSpanOffsets = [list(subspan) for subspan in curr_h_idx_limits]
                docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in curr_h_idx_limits])

                highlights_in_context_list.append({"documentFile" : doc_name,
                                                "scuSentCharIdx" : None,
                                                "scuSentence" : None,
                                                "docSentCharIdx" : docSentCharIdx,
                                                "docSentText" : docSentText,
                                                "docSpanText" : docSpanText,
                                                "docSpanOffsets" : docSpanOffsets,
                                                "sent_idx" : sentence_index})

        elif len(relevant_sents_i)==0:
            potentially_containing_sents_i = [i for i,sent in enumerate(doc_sents) if rmv_spaces_and_punct(h) in rmv_spaces_and_punct(sent)]
            assert potentially_containing_sents_i, "highlight wasn't found"
            relevant_sents_i = potentially_containing_sents_i[0]
            docSentCharIdx = str(sents_idx_limits[relevant_sents_i][0])
            docSentText = doc_text[sents_idx_limits[relevant_sents_i][0]:sents_idx_limits[relevant_sents_i][1]]

            lcs_details = longest_common_subsequence(doc_sents[relevant_sents_i], h)
            in_sent_spans = get_consecutive_subspans(sorted(lcs_details[1]))
            docSpanOffsets = [[int(docSentCharIdx)+elem for elem in span] for span in in_sent_spans]
            docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in docSpanOffsets])

            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})
        else:
            relevant_sents_i = relevant_sents_i[0]
            docSentCharIdx = [str(sent[0]) for sent_i,sent in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSentText = [doc_text[sent_span[0]:sent_span[1]] for sent_i,sent_span in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSpanOffsets = [list(h_idx_limits)]
            docSpanText = doc_text[h_idx_limits[0]:h_idx_limits[1]]
            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})

    json_highlights_in_context_list = set([json.dumps(elem) for elem in highlights_in_context_list])
    highlights_in_context_list = [json.loads(elem) for elem in json_highlights_in_context_list]
    return highlights_in_context_list


def convert_content_selection_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = _get_spacy_nlp()
    pipeline_style_data = []

    for unique_id, value in results.items():
        curr_original_inst = deepcopy([elem for elem in alignments_dict if elem.get("unique_id") == unique_id][0])
        curr_documents = curr_original_inst.get("documents", [])

        doc_by_file = {}
        for d in curr_documents:
            if not isinstance(d, dict):
                continue
            df = d.get("documentFile") or d.get("documentUrl")
            if df:
                doc_by_file[str(df)] = d

        non_highlighted_docs = value.get("non_highlighted_docs")
        if not isinstance(non_highlighted_docs, list):
            non_highlighted_docs = []

        final_output = value.get("final_output", {})
        if not isinstance(final_output, dict):
            final_output = {}

        highlights_in_context = []

        for i, doc in enumerate(non_highlighted_docs):
            doc_key = f"Document [{str(i+1)}]"
            if doc_key not in final_output:
                continue

            if not isinstance(doc, dict):
                continue

            doc_name = doc.get("doc_name") or doc.get("documentFile") or doc.get("url") or doc.get("source_url")
            if not doc_name:
                continue
            doc_name = str(doc_name)

            matching_doc = doc_by_file.get(doc_name)
            if matching_doc is None:
                matching_doc = next(
                    (
                        dd
                        for dd in curr_documents
                        if isinstance(dd, dict)
                        and (
                            str(dd.get("documentUrl")) == doc_name
                            or str(dd.get("documentFile")) == doc_name
                        )
                    ),
                    None,
                )

            if matching_doc is None:
                continue

            doc_text = matching_doc.get("rawDocumentText") or matching_doc.get("source_raw_text")
            if not doc_text:
                continue

            doc_sents = None
            if isinstance(matching_doc.get("documentText"), list):
                doc_sents = matching_doc.get("documentText")

            spans = final_output.get(doc_key)
            if not isinstance(spans, list):
                continue

            hic = get_set_of_highlights_in_context_content_selection(
                doc_name=doc_name,
                doc_text=doc_text,
                highlights=spans,
                nlp=nlp,
                doc_sents=doc_sents,
            )
            highlights_in_context.extend(hic)

        curr_original_inst["set_of_highlights_in_context"] = highlights_in_context
        pipeline_style_data.append(curr_original_inst)

    return pipeline_style_data


def get_set_of_highlights_in_context_clustering(curr_instance, nlp, doc_sents, *args, **kwargs):
    highlights_in_context_list = []
    highlight_global_index = 1
    for doc_i,doc_highlights in enumerate(curr_instance['highlights']):
        curr_doc_name = curr_instance['highlighted_docs'][doc_i]['doc_name']
        curr_doc_text = next(iter([elem['rawDocumentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)
        curr_doc_sents = next(iter([elem['documentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)

        for highlight in doc_highlights:
            curr_highlights_in_context = get_set_of_highlights_in_context_content_selection(doc_name=curr_doc_name,
                                                                                            doc_text=curr_doc_text,
                                                                                            highlights=[highlight],
                                                                                            nlp=nlp,
                                                                                            doc_sents=curr_doc_sents)

            relevant_clusters = [cluster_i for cluster_i,elem in enumerate(curr_instance['final_output']) if highlight_global_index in elem['cluster']]

            for cluster_index in relevant_clusters:
                highlights_in_context_list+=[{k: cluster_index if k == 'scuSentCharIdx' else v for k, v in d.items()} for d in curr_highlights_in_context]

            highlight_global_index+=1

    return highlights_in_context_list


def get_set_of_highlights_in_context_ALCE(curr_instance):
    highlights_in_context_list = []
    final_output = ""
    curr_scuSentCharIdx = 0
    for sentwise_results in curr_instance["final_output"]:
        if sentwise_results['cited_docs']:
            for doc_i in sentwise_results['cited_docs']:
                highlights_in_context_list.append({"documentFile" : curr_instance['non_highlighted_docs'][doc_i-1]['doc_name'],
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})
        else:
                highlights_in_context_list.append({"documentFile" : None,
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})

        curr_scuSentCharIdx = curr_scuSentCharIdx + len(sentwise_results['sent']) + 1
        final_output = final_output + sentwise_results['sent'] + " "

    assert all(final_output[elem['scuSentCharIdx']:].startswith(elem['scuSentence']) for elem in highlights_in_context_list), "scuSentence doesn't match scuSentCharIdx"
    return highlights_in_context_list, final_output.strip()


def convert_clustering_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = _get_spacy_nlp()
    pipeline_style_data = []
    for key,value in results.items():
        curr_pipeline_style_data = [elem for elem in alignments_dict if elem["unique_id"]==key][0]

        curr_documents = curr_pipeline_style_data["documents"]
        highlights_in_context = get_set_of_highlights_in_context_clustering(curr_instance=value,
                                                                            nlp=nlp,
                                                                            doc_sents=curr_documents)
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["gold_summary"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    return pipeline_style_data


def convert_e2e_only_setting_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            original_alignments_dict.update({"set_of_highlights_in_context":[],
                                             "response" : value["final_output"],
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data


def convert_ALCE_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            curr_documents = [elem["documents"] for elem in alignments_dict if elem["unique_id"]==key][0]
            highlights_in_context, final_output = get_set_of_highlights_in_context_ALCE(curr_instance=value)
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            original_alignments_dict.update({"set_of_highlights_in_context" : highlights_in_context,
                                             "response" : final_output,
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data


def get_set_of_highlights_in_context_FiC_CoT(curr_instance, nlp, doc_sents, *args, **kwargs):
    clustering_style_instance_format = {"highlights":curr_instance["highlights"],
                                        "highlighted_docs":curr_instance["highlighted_docs"],
                                        "final_output":[{"cluster":elem['highlights']} for elem in curr_instance['alignments']]}
    clustered_set_of_highlights = get_set_of_highlights_in_context_clustering(curr_instance=clustering_style_instance_format,
                                                                              nlp=nlp,
                                                                              doc_sents=doc_sents)
    highlights_in_context_list = []
    alignments_sents_embedding = None
    for sent in nlp(curr_instance['final_output']).sents:
        relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text)==rmv_spaces_and_punct(elem['sent_text'])]
        if not relevant_sent_id:
            relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if (rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text) in rmv_spaces_and_punct(elem['sent_text'])) or (rmv_spaces_and_punct(elem['sent_text']) and rmv_spaces_and_punct(elem['sent_text']) in rmv_spaces_and_punct(sent.text))]

        if not relevant_sent_id:
            curr_sent_embedding = get_sentence_embedding(sent.text, sent_transformer_model, sent_transformer_tokenizer)
            if not alignments_sents_embedding:
                alignments_sents_embedding = [get_sentence_embedding(elem["sent_text"], sent_transformer_model, sent_transformer_tokenizer) for elem in curr_instance['alignments']]
            similarities = [1 - cosine(curr_sent_embedding , s_embedding) for s_embedding in alignments_sents_embedding]
            relevant_sent_id = [curr_instance['alignments'][i]['sent_id'] for i,scr in enumerate(similarities) if scr>=COSINE_SIMILARITY_THR]

        if not relevant_sent_id:
            highlights_in_context_list.append({"documentFile" : None,
                                               "scuSentCharIdx" : sent.start_char,
                                               "scuSentence" : sent.text,
                                               "docSentCharIdx" : None,
                                               "docSentText" : None,
                                               "docSpanText" : None,
                                               "docSpanOffsets" : None,
                                               "sent_idx" : None})
        else:
            relevant_clustered_highlights_in_context = deepcopy([elem for elem in clustered_set_of_highlights if elem['scuSentCharIdx']+1 in relevant_sent_id])
            relevant_clustered_highlights_in_context = [{key:value if key!="scuSentence" else sent.text for key,value in elem.items()} for elem in relevant_clustered_highlights_in_context]
            relevant_clustered_highlights_in_context = [{key:value if key!="scuSentCharIdx" else sent.start_char for key,value in elem.items()} for elem in relevant_clustered_highlights_in_context]
            highlights_in_context_list += relevant_clustered_highlights_in_context
    return highlights_in_context_list


def convert_FiC_CoT_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = _get_spacy_nlp()
    pipeline_style_data = []
    for key,value in results.items():
        curr_pipeline_style_data = deepcopy([elem for elem in alignments_dict if elem["unique_id"]==key][0])

        curr_documents = curr_pipeline_style_data["documents"]
        highlights_in_context = get_set_of_highlights_in_context_FiC_CoT(curr_instance=value,
                                                                         nlp=nlp,
                                                                         doc_sents=curr_documents)
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["final_output"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    return pipeline_style_data
