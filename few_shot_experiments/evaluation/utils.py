import json

# NOTE: attribution_metrics is intentionally not imported here.
# LongCite evaluation only needs get_data(), and we want to avoid extra deps.

import spacy
from string import punctuation
import numpy as np
from copy import deepcopy

nlp = spacy.load("en_core_web_sm")
SPAN_SEP = "<HIGHLIGHT_SEP>"
IGNORE_POS = ['PUNCT', 'SPACE']


# Helper for debug prints: truncate and flatten previews
def _dbg_trunc(x, n: int = 240):
    """Best-effort short preview for debug prints."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")

def adapt_attribution(curr_inputs, new_attribution_key):
    adapted_inputs = []
    for inst in curr_inputs:
        updated_inst = deepcopy(inst)
        updated_inst = [{key:value if key!="attribution" else elem[new_attribution_key] for key,value in elem.items()} for elem in updated_inst]
        assert all(elem['attribution']==elem[new_attribution_key] for elem in updated_inst), f"{new_attribution_key} not adapted properly"
        assert all(elem[new_attribution_key]==inst[i][new_attribution_key] for i,elem in enumerate(updated_inst)), f"{new_attribution_key} changed"
        adapted_inputs.append(updated_inst)
    return adapted_inputs

def get_data(indir, debug: bool = False, max_debug_instances: int = 1, max_debug_sents: int = 3):
    with open(indir, 'r') as f1:
        data_dict = [json.loads(line) for line in f1.readlines()]
    if debug:
        print(f"[get_data] loaded {len(data_dict)} instances from: {indir}")
        if len(data_dict) == 0:
            print("[get_data] WARNING: file is empty (no jsonl lines).")
        else:
            sample_keys = list(data_dict[0].keys())
            print(f"[get_data] sample instance keys: {sample_keys[:30]}")
            print(f"[get_data] sample unique_id: {data_dict[0].get('unique_id')}")
            # quick schema sanity checks
            print(f"[get_data] has 'documents': {'documents' in data_dict[0]} | has 'set_of_highlights_in_context': {'set_of_highlights_in_context' in data_dict[0]}")
            if isinstance(data_dict[0].get('documents'), list) and len(data_dict[0]['documents']) > 0:
                d0 = data_dict[0]['documents'][0]
                print(f"[get_data] first document keys: {list(d0.keys())[:30]}")
                print(f"[get_data] first documentFile: {d0.get('documentFile')}")
                print(f"[get_data] first rawDocumentText preview: {_dbg_trunc(d0.get('rawDocumentText', ''), 200)}")
            if isinstance(data_dict[0].get('set_of_highlights_in_context'), list) and len(data_dict[0]['set_of_highlights_in_context']) > 0:
                h0 = data_dict[0]['set_of_highlights_in_context'][0]
                print(f"[get_data] first highlight keys: {list(h0.keys())[:30]}")
                print(f"[get_data] first highlight documentFile={h0.get('documentFile')} docSpanOffsets={h0.get('docSpanOffsets')}")
                print(f"[get_data] first highlight scuSentence preview: {_dbg_trunc(h0.get('scuSentence', ''), 200)}")

    inputs = []
    full_instance_stats = []
    instances_unique_ids = []
    for instance in data_dict:
        missing_attribution = 0
        if debug and len(inputs) < max_debug_instances:
            print("\n[get_data] ------------------------------")
            print(f"[get_data] instance unique_id={instance.get('unique_id')}")
            print(f"[get_data] documents={len(instance.get('documents', []))} highlights_in_context={len(instance.get('set_of_highlights_in_context', []))}")
        if debug and len(inputs) < max_debug_instances:
            doc_files = [d.get('documentFile') for d in instance.get('documents', [])]
            print(f"[get_data] documentFiles: {doc_files}")
            if len(doc_files) == 0:
                print("[get_data] WARNING: instance has 0 documents")
            # show a couple highlight previews
            hic = instance.get('set_of_highlights_in_context', [])
            for j, h in enumerate(hic[:min(3, len(hic))]):
                print(f"[get_data] highlight[{j}] docFile={h.get('documentFile')} docSpanOffsets={h.get('docSpanOffsets')} docSentCharIdx={h.get('docSentCharIdx')}")
                print(f"[get_data] highlight[{j}] docSpanText preview: {_dbg_trunc(h.get('docSpanText',''), 180)}")
        all_attributing_tkns = list()
        curr_inputs = []
        curr_tokenized_docs = {elem["documentFile"]:nlp(elem["rawDocumentText"]) for elem in instance['documents']}
        # ignore punctuation and spaces
        curr_relevant_tkns_docs = {key:[tkn for tkn in value if not tkn.pos_ in IGNORE_POS] for key,value in curr_tokenized_docs.items()}
        if debug and len(inputs) < max_debug_instances:
            try:
                # Show token counts per doc (before/after filters)
                raw_counts = {k: len(v) for k, v in curr_tokenized_docs.items()}
                rel_counts = {k: len(v) for k, v in curr_relevant_tkns_docs.items()}
                print(f"[get_data] token_counts(raw) per doc: {raw_counts}")
                print(f"[get_data] token_counts(relevant) per doc: {rel_counts}")
            except Exception as _e:
                print(f"[get_data] DEBUG token count print failed: {_e}")
        curr_content_tkns_docs = {key:[tkn for tkn in value if not tkn.is_stop] for key,value in curr_relevant_tkns_docs.items()}
        doc_tkn_cnt = {key:len(value) for key,value in curr_relevant_tkns_docs.items()}
        doc_content_tkn_cnt = {key:len(value) for key,value in curr_content_tkns_docs.items()}

        _dbg_sents_printed = 0
        for scuSentence in set([elem['scuSentence'] for elem in instance['set_of_highlights_in_context']]):
            if debug and len(inputs) < max_debug_instances and _dbg_sents_printed < max_debug_sents:
                print(f"[get_data] scuSentence preview: {str(scuSentence)[:160]}")
            attributing_tkns = {}
            attributing_content_tkns = {}
            curr_scuSentence_alignments = [elem for elem in instance['set_of_highlights_in_context'] if elem['scuSentence']==scuSentence]
            if debug and len(inputs) < max_debug_instances and _dbg_sents_printed < max_debug_sents:
                try:
                    df_set = list(set([e.get('documentFile') for e in curr_scuSentence_alignments]))
                    print(f"[get_data] alignments for this sentence: {len(curr_scuSentence_alignments)} | documentFiles={df_set}")
                except Exception as _e:
                    print(f"[get_data] DEBUG alignment print failed: {_e}")
            assert all(elem['documentFile'] for elem in curr_scuSentence_alignments) or len(set([elem['documentFile'] for elem in curr_scuSentence_alignments]))==1, "there is one highlight_in_context where documentFile is None but there is also attribution elsewhere"
            total_attribution = {}
            total_sent_attribution = {} # also get a variant that includes the sentences within which the highlights are - to avoid "penalty" caused by incoherencies resulting from concatenating disparate spans
            total_doc_attribution = {} # also get a variant that includes the full docs within which the highlights are - to avoid "penalty" caused by incoherencies resulting from concatenating disparate spans
            for documentFile in set([elem['documentFile'] for elem in curr_scuSentence_alignments]):
                if not documentFile: # in cases of missing attribution (ALCE and RARR)
                    missing_attribution += 1
                    if debug and len(inputs) < max_debug_instances:
                        print("[get_data] WARNING: missing documentFile for a highlight; counting as missing attribution and skipping this scuSentence.")
                        print(f"[get_data] offending alignment preview: {_dbg_trunc(curr_scuSentence_alignments[:1], 260)}")
                    break
                curr_documentFile_alignments = [elem for elem in curr_scuSentence_alignments if elem['documentFile']==documentFile]
                # when all doc is attributing - docSpanOffsets is None (ALCE)
                if not curr_documentFile_alignments[0]['docSpanOffsets']:
                    total_attribution[documentFile] = [elem['rawDocumentText'] for elem in instance['documents'] if elem['documentFile']==documentFile][0]
                    attributing_tkns[documentFile] = curr_relevant_tkns_docs[documentFile]
                    attributing_content_tkns[documentFile] = curr_content_tkns_docs[documentFile]
                    total_sent_attribution, total_doc_attribution = None, None
                else:
                    # order by docSpanOffset's first subspan
                    curr_documentFile_alignments = sorted(curr_documentFile_alignments, key=lambda x: x['docSpanOffsets'][0][0])
                    doc_spans = [" ".join(elem['docSpanText'].split(SPAN_SEP)).strip() for elem in curr_documentFile_alignments] # split disparate highlights and join again with a space
                    doc_spans = [elem for elem in doc_spans if elem] # remove highlights that are empty lines
                    doc_spans = [elem if elem[-1] in punctuation else f"{elem}." for elem in doc_spans] # add period to spans without punctuation in their end (as each such span comes from separate sentences)
                    # doc_spans = [elem['docSpanText'].split(SPAN_SEP) for elem in curr_documentFile_alignments]
                    # doc_spans = list(itertools.chain.from_iterable(doc_spans)) # flatten
                    curr_attribution = " ".join(doc_spans)
                    total_attribution[documentFile] = " ".join(curr_attribution.split()) # replace consecutive spaces/new lines with a single space

                    # get attributing tokens
                    # 1st - find all "attributing" char indices
                    attributing_idxs = [idx for elem in curr_documentFile_alignments for span in elem["docSpanOffsets"] for idx in range(span[0], span[1])]
                    # 2nd - find tkns whose idx is in attributing_idxs
                    attributing_tkns[documentFile] = [tkn for tkn in curr_tokenized_docs[documentFile] if tkn.idx in attributing_idxs and not tkn.pos_ in IGNORE_POS]
                    attributing_content_tkns[documentFile] = [tkn for tkn in attributing_tkns[documentFile] if not tkn.is_stop]

                    # get attributing sentences
                    attributing_sents = sorted(list(set((int(elem["docSentCharIdx"]), elem["docSentText"]) for elem in curr_documentFile_alignments)), key=lambda x: x[0])
                    total_sent_attribution[documentFile] = " ".join([elem[1] for elem in attributing_sents])

                    # get attributing documents
                    total_doc_attribution[documentFile] = [elem['rawDocumentText'] for elem in instance['documents'] if elem['documentFile']==documentFile][0]

            # ignore punctuation and spaces
            all_curr_tkns = [tkn for tkn_list in attributing_tkns.values() for tkn in tkn_list]
            all_doc_tkns = [tkn for tkn_list in curr_relevant_tkns_docs.values() for tkn in tkn_list]
            all_curr_content_tkns = [tkn for tkn in all_curr_tkns if not tkn.is_stop]
            all_doc_content_tkns = [tkn for tkn in all_doc_tkns if not tkn.is_stop]

            docwise_curr_tkns_cnt = {key:len(value) for key,value in attributing_tkns.items()}
            docwise_curr_content_tkns_cnt = {key:len([tkn for tkn in value if not tkn.is_stop]) for key,value in attributing_tkns.items()}


            docwise_curr_tkns_percent = {key:100*value/doc_tkn_cnt[key] for key,value in docwise_curr_tkns_cnt.items()}
            docwise_curr_content_tkns_percent = {key:100*value/doc_content_tkn_cnt[key] for key,value in docwise_curr_content_tkns_cnt.items()}


            curr_inputs.append({"sentence":scuSentence,
                                "attribution":total_attribution,
                                "full_sent_attribution": total_sent_attribution,
                                "full_doc_attribution" : total_doc_attribution,
                                "tkns":{"absolute":len(all_curr_tkns),
                                            "percent":round(100*len(all_curr_tkns)/len(all_doc_tkns), 4)},
                                "content_tkns":{"absolute":len(all_curr_content_tkns),
                                                    "percent":round(100*len(all_curr_content_tkns)/len(all_doc_content_tkns), 4)},
                                "avg_docwise_tkns":{"absolute":round(np.mean(list(docwise_curr_tkns_cnt.values())), 4) if docwise_curr_tkns_cnt else None,
                                                         "percent":round(np.mean(list(docwise_curr_tkns_percent.values())), 4) if docwise_curr_tkns_percent else None},
                                "avg_docwise_content_tkns":{"abs":round(np.mean(list(docwise_curr_content_tkns_cnt.values())), 4) if docwise_curr_content_tkns_cnt else None,
                                                                "percent":round(np.mean(list(docwise_curr_content_tkns_percent.values())), 4) if docwise_curr_content_tkns_percent else None}})
            if debug and len(inputs) < max_debug_instances and _dbg_sents_printed < max_debug_sents:
                try:
                    df_keys = list(total_attribution.keys())
                    print(f"[get_data] attribution docs for this sentence: {df_keys}")
                    # show short attribution lengths
                    lens = {k: (len(v) if isinstance(v, str) else None) for k, v in total_attribution.items()}
                    print(f"[get_data] attribution text lengths: {lens}")
                    print(f"[get_data] tkns.abs={curr_inputs[-1]['tkns']['absolute']} content_tkns.abs={curr_inputs[-1]['content_tkns']['absolute']}")
                except Exception as _e:
                    print(f"[get_data] DEBUG print failed: {_e}")
                try:
                    last = curr_inputs[-1]
                    attrib = last.get('attribution') or {}
                    print(f"[get_data] produced per-sent entry: sentence_preview={_dbg_trunc(last.get('sentence',''), 160)}")
                    print(f"[get_data] attribution keys: {list(attrib.keys())}")
                    # short preview of first attribution value (if any)
                    if isinstance(attrib, dict) and len(attrib) > 0:
                        k0 = list(attrib.keys())[0]
                        print(f"[get_data] attribution[{k0}] preview: {_dbg_trunc(attrib.get(k0,''), 200)}")
                except Exception as _e:
                    print(f"[get_data] DEBUG per-sent output print failed: {_e}")
                _dbg_sents_printed += 1
            all_attributing_tkns.append(attributing_tkns)

        inputs.append(curr_inputs)
        instances_unique_ids.append(instance['unique_id'])
        if debug and len(inputs) <= max_debug_instances:
            print(f"[get_data] instance done: unique_id={instance.get('unique_id')} per_sent_entries={len(curr_inputs)} missing_attribution_sents={missing_attribution}")

        # get full instance statistics
        aggregated_attrib_tkns_i = {key:set([tkn.i for elem in all_attributing_tkns if elem and key in elem for tkn in elem[key]]) for key,value in curr_tokenized_docs.items()}
        aggregated_attrib_tkns = {key:[tkn for tkn in curr_tokenized_docs[key] if tkn.i in value] for key,value in aggregated_attrib_tkns_i.items()}
        aggregated_attrib_content_tkns = {key:[tkn for tkn in value if not tkn.is_stop] for key,value in aggregated_attrib_tkns.items()}
        aggregated_attrib_tkns_cnt = {key:len(value) for key,value in aggregated_attrib_tkns.items()}
        aggregated_attrib_content_tkns_cnt = {key:len(value) for key,value in aggregated_attrib_content_tkns.items()}
        aggregated_docwise_curr_tkns_percent = {key:100*value/doc_tkn_cnt[key] for key,value in aggregated_attrib_tkns_cnt.items()}
        aggregated_docwise_curr_content_tkns_percent = {key:100*value/doc_content_tkn_cnt[key] for key,value in aggregated_attrib_content_tkns_cnt.items()}



        full_instance_stats.append({"non_attributed_sent_cnt" : missing_attribution,
                                    "total_sent_cnt" : len(set([elem['scuSentence'] for elem in instance['set_of_highlights_in_context']])),
                                    "tkns" : {"absolute":sum(list(aggregated_attrib_tkns_cnt.values())),
                                                     "percent":round(100*sum(list(aggregated_attrib_tkns_cnt.values()))/sum(list(doc_tkn_cnt.values())), 4)},
                                    "content_tkns" : {"absolute":sum(list(aggregated_attrib_content_tkns_cnt.values())),
                                                             "percent":round(100*sum(list(aggregated_attrib_content_tkns_cnt.values()))/sum(list(doc_content_tkn_cnt.values())), 4)},
                                    "avg_docwise_tkns" : {"absolute":round(np.mean(list(aggregated_attrib_tkns_cnt.values())), 4),
                                                                 "percent":round(np.mean(list(aggregated_docwise_curr_tkns_percent.values())), 4)},
                                    "avg_docwise_content_tkns" : {"absolute":round(np.mean(list(aggregated_attrib_content_tkns_cnt.values())), 4),
                                                                         "percent":round(np.mean(list(aggregated_docwise_curr_content_tkns_percent.values())), 4)}})

    if debug:
        print(f"\n[get_data] done. Produced inputs for {len(inputs)} instances; unique_ids={len(instances_unique_ids)}; stats={len(full_instance_stats)}")
        if len(inputs) > 0:
            print(f"[get_data] first instance has {len(inputs[0])} per-sentence entries.")
            if len(inputs[0]) > 0:
                print(f"[get_data] first per-sent entry keys: {list(inputs[0][0].keys())}")
                print(f"[get_data] first per-sent sentence preview: {str(inputs[0][0].get('sentence'))[:160]}")
        try:
            print(f"[get_data] RETURN types: inputs={type(inputs)} instances_unique_ids={type(instances_unique_ids)} full_instance_stats={type(full_instance_stats)}")
            if len(inputs) > 0:
                print(f"[get_data] RETURN lens: inputs={len(inputs)} inputs[0]={len(inputs[0]) if inputs[0] is not None else None} unique_ids={len(instances_unique_ids)} stats={len(full_instance_stats)}")
                print(f"[get_data] RETURN preview inputs[0][0]: {_dbg_trunc(inputs[0][0] if len(inputs[0])>0 else None, 420)}")
        except Exception as _e:
            print(f"[get_data] DEBUG return preview failed: {_e}")

    return inputs, instances_unique_ids, full_instance_stats