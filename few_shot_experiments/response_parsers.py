from utils import *
from copy import deepcopy
import re
import spacy

_SPACY_NLP = None


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def adapt_highlights_to_doc_alignments(doc_texts_dict, salience_dict):
    salience_dict_adapted = deepcopy(salience_dict)
    for doc_name, salience_list in salience_dict.items():
        for salient_span in salience_list:
            if rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(doc_texts_dict[doc_name]):
                continue
            alternative_docs = [key for key,value in doc_texts_dict.items() if rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(value)]
            assert len(alternative_docs)>0, "not all selected content is traceable"

            alternative_doc = alternative_docs[0]
            if not alternative_doc in salience_dict_adapted.keys():
                salience_dict_adapted[alternative_doc] = []

            salience_dict_adapted[alternative_doc].append(salient_span)
            salience_dict_adapted[doc_name] = [h for h in salience_dict_adapted[doc_name] if h != salient_span]

    return salience_dict_adapted


# --- Helper functions for parsing content selection and ambiguity highlight ---
def _split_response_into_doc_blocks(response: str):
    """Return list[("Document [i]", text_after_colon)] in appearance order."""
    return [
        (m.group(1), (m.group(2) or "").strip())
        for m in re.finditer(r"(Document \[\d+\]):(.*?)(?=Document \[|\Z)", response, re.DOTALL)
    ]


def _clean_highlight_markers(text: str) -> str:
    if not text:
        return ""
    for tok in ("<highlight_start>", "<highlight_end>", "{HS}", "{HE}"):
        text = text.replace(tok, "")
    return text


def _extract_last_instance_docs_from_prompt(prompt: str):
    """Extract the last instance's document block from the prompt."""
    if not prompt:
        return {}, [], ""

    m_all = list(re.finditer(r"\bAnswer:\b", prompt))
    pre_answer = prompt[: m_all[-1].start()] if m_all else prompt

    start_idx = pre_answer.rfind("Document [1]:")
    docs_block = pre_answer[start_idx:] if start_idx != -1 else pre_answer

    docs_texts_pairs = re.findall(
        r"(Document \[\d+\]):\s+([^\n]*(?:\n(?!Document \[\d+\]:)[^\n]*)*)",
        docs_block,
    )

    docs_names = [name for name, _ in docs_texts_pairs]
    docs_texts = {name: text for name, text in docs_texts_pairs}

    return docs_texts, docs_names, docs_block


def _extract_hs_spans_from_docs_block(docs_block: str):
    """Extract spans that are already highlighted in the prompt docs block."""
    if not docs_block:
        return []

    spans = []
    for m in re.finditer(r"<highlight_start>(.*?)<highlight_end>", docs_block, re.DOTALL):
        spans.append(m.group(1))
    for m in re.finditer(r"\{HS\}(.*?)\{HE\}", docs_block, re.DOTALL):
        spans.append(m.group(1))

    spans = [s.strip() for s in spans if rmv_spaces_and_punct(_clean_highlight_markers(s))]
    return spans


def _dedupe_preserve_order(items: list):
    seen = set()
    out = []
    for x in items:
        cleaned = _clean_highlight_markers(x or "").strip()
        key = rmv_spaces_and_punct(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _filter_out_nested_or_duplicate_against_hs(ha_spans: list, hs_spans: list):
    """Remove HA spans that are *true duplicates* of existing HS spans."""
    if not ha_spans:
        return [], []
    if not hs_spans:
        return ha_spans, []

    hs_norm_set = {
        rmv_spaces_and_punct(_clean_highlight_markers(s))
        for s in hs_spans
        if rmv_spaces_and_punct(_clean_highlight_markers(s))
    }

    kept, removed = [], []
    for s in ha_spans:
        ns = rmv_spaces_and_punct(_clean_highlight_markers(s))
        if not ns:
            removed.append(s)
            continue

        if ns in hs_norm_set:
            removed.append(s)
        else:
            kept.append(s)

    return kept, removed


def parse_FiC_response(response, prompt):
    if "where each time you cluster highlights to build the next sentence." in prompt:
        assert "So the final summary is:" in response or "So the final answer is:" in response, "cannot find the final summary in the model's response"
        if "So the final summary is:" in response:
            final_summary = response[response.index("So the final summary is:"):].replace("So the final summary is:", "").strip()
            planning_response = response[:response.index("So the final summary is:")]
        else:
            final_summary = response[response.index("So the final answer is:"):].replace("So the final answer is:", "").strip()
            planning_response = response[:response.index("So the final answer is:")]
        pattern = r"highlight(?:s)? ([\d, and]+) (?:is|are) combined to form sentence (\d+):[ \n]*(.*?)(?=[\n]+|\Z)"
        alignment_matches = re.findall(pattern, planning_response, re.IGNORECASE)
        def _parse_highlight_ids(raw: str):
            tokens = re.split(r"[,\s]+", raw)
            ids = []
            for t in tokens:
                t = t.replace("and", "").strip()
                if t.isdigit():
                    ids.append(int(t))
            return sorted(ids)
        alignments = [{"sent_id" : int(elem[1]),
                    "highlights" : _parse_highlight_ids(elem[0]),
                    "sent_text" : elem[2]} for elem in alignment_matches]
        alignments = sorted(alignments, key=lambda x: x['sent_id'])
        if not "".join([elem['sent_text'] for elem in alignments]).replace(" ", "").replace("\n", "")==final_summary.replace(" ", "").replace("\n", ""):
            logging.info(f"separate sentences don't match final summary. Separate sentences combined:\n {' '.join([elem['sent_text'] for elem in alignments])}.\n\n Final summary:\n{final_summary}")
        return {"alignments":alignments,
                "final_output":final_summary,
                "full_model_response":response}
    else:
        if response.startswith("So the summary is:"):
            response = response[len("So the summary is:"):]

        return {"final_output":response,
                "full_model_response":response}


def _extract_partial_highlights(text: str) -> list:
    """Recover complete JSON highlight items from a truncated response."""
    items = []
    start = text.find("[")
    if start == -1:
        return items
    decoder = json.JSONDecoder()
    pos = start + 1
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            obj, end_pos = decoder.raw_decode(text, pos)
            if isinstance(obj, dict) and "doc_id" in obj and "span_text" in obj:
                items.append(obj)
            pos = end_pos
        except json.JSONDecodeError:
            break
    return items


def parse_content_selection_structured_response(response, prompt):
    """Parse CS response when structured output (JSON mode) was used."""
    try:
        data = json.loads(response)
        highlights = data.get("highlights", [])
    except json.JSONDecodeError:
        highlights = _extract_partial_highlights(response)
        if not highlights:
            return parse_content_selection_response(response, prompt)

    salience_dict = {}
    for item in highlights:
        doc_id_raw = str(item.get("doc_id", "")).strip()
        span_text = item.get("span_text", "")
        if re.match(r"^\d+$", doc_id_raw):
            doc_id = f"Document [{doc_id_raw}]"
        else:
            doc_id = doc_id_raw
        spans = [s.strip() for s in span_text.split("<SPAN_DELIM>") if rmv_spaces_and_punct(s)]
        salience_dict.setdefault(doc_id, []).extend(spans)

    if not any(salience_dict.values()):
        raise Exception("no content was found")

    docs_texts, docs_names, _docs_block = _extract_last_instance_docs_from_prompt(prompt or "")
    if docs_names:
        assert all(key in docs_names for key in salience_dict.keys()), "not only relevant documents are included"
        docs_texts_clean = {k: _clean_highlight_markers(v) for k, v in docs_texts.items()}
        salience_dict = adapt_highlights_to_doc_alignments(docs_texts_clean, salience_dict)
        assert all(
            all(rmv_spaces_and_punct(s) in rmv_spaces_and_punct(docs_texts_clean[doc_name])
                for s in spans)
            for doc_name, spans in salience_dict.items()
        ), "not all selected content is traceable"

    return {"final_output": salience_dict, "full_model_response": response}


def parse_ambiguity_highlight_structured_response(response, prompt=None):
    """Parse AH response when structured output (JSON mode) was used.

    Expects schema: {"highlights": [{"doc_id": "...", "span_text": "..."}]}
    where span_text is the decontextualized (or original) highlight text.
    """
    try:
        data = json.loads(response)
        highlights = data.get("highlights", [])
    except json.JSONDecodeError:
        highlights = _extract_partial_highlights(response)
        if not highlights:
            return parse_ambiguity_highlight_response(response, prompt)

    ha_dict = {}
    for item in highlights:
        doc_id_raw = str(item.get("doc_id", "")).strip()
        span_text = (item.get("span_text") or "").strip()
        if re.match(r"^\d+$", doc_id_raw):
            doc_id = f"Document [{doc_id_raw}]"
        else:
            doc_id = doc_id_raw
        if rmv_spaces_and_punct(span_text):
            ha_dict.setdefault(doc_id, []).append(span_text)

    docs_texts, docs_names, docs_block = _extract_last_instance_docs_from_prompt(prompt or "")
    if docs_names:
        expected_set = set(docs_names)
        returned_set = set(ha_dict.keys())
        intersect = returned_set & expected_set
        if returned_set and not intersect:
            raise Exception("not only relevant documents are included")

        ha_dict = {k: v for k, v in ha_dict.items() if k in expected_set}
        for dn in docs_names:
            ha_dict.setdefault(dn, [])

        docs_texts_clean = {k: _clean_highlight_markers(v) for k, v in docs_texts.items()}
        ha_dict = adapt_highlights_to_doc_alignments(docs_texts_clean, ha_dict)

        for dn in docs_names:
            ha_dict[dn] = _dedupe_preserve_order(ha_dict.get(dn, []))

        assert all(
            all(rmv_spaces_and_punct(_clean_highlight_markers(s)) in rmv_spaces_and_punct(docs_texts_clean[doc_name])
                for s in spans)
            for doc_name, spans in ha_dict.items()
        ), "not all selected content is traceable"

        hs_spans = _extract_hs_spans_from_docs_block(docs_block)
        if hs_spans:
            for doc_name, spans in list(ha_dict.items()):
                kept, _ = _filter_out_nested_or_duplicate_against_hs(spans, hs_spans)
                ha_dict[doc_name] = kept
            for dn in docs_names:
                ha_dict.setdefault(dn, [])

    return {"final_output": ha_dict, "full_model_response": response}


def parse_FiC_structured_response(response, prompt=None):
    """Parse FiC response when structured output (JSON mode) was used."""
    try:
        data = json.loads(response)
        if data.get("abstain"):
            return {"alignments": [], "final_output": "ABSTAIN",
                    "full_model_response": response, "abstained": True}
        sentences = data.get("sentences", [])
        alignments = [
            {"sent_id": s["sentence_id"],
             "highlights": s.get("highlight_ids", []),
             "sent_text": s["sentence_text"]}
            for s in sentences
        ]
        final_output = " ".join(s["sentence_text"] for s in sentences)
        return {"alignments": alignments, "final_output": final_output,
                "full_model_response": response, "abstained": False}
    except Exception:
        return parse_FiC_response(response, prompt)


def parse_content_selection_response(response, prompt):
    doc_spans_pairs = _split_response_into_doc_blocks(response)
    if not doc_spans_pairs:
        raise Exception("no content was found")

    salience_dict = {}
    for doc_id, spans_blob in doc_spans_pairs:
        spans = spans_blob.split("<SPAN_DELIM>") if spans_blob else []
        spans = [s for s in spans if rmv_spaces_and_punct(s)]
        spans = [s.strip() for s in spans if rmv_spaces_and_punct(s)]
        salience_dict[doc_id] = spans

    docs_texts, docs_names, _docs_block = _extract_last_instance_docs_from_prompt(prompt or "")
    assert docs_names, "could not extract documents from prompt"
    assert all(key in docs_names for key in salience_dict.keys()), "not only relevant documents are included"

    docs_texts_clean = {k: _clean_highlight_markers(v) for k, v in docs_texts.items()}
    salience_dict = adapt_highlights_to_doc_alignments(docs_texts_clean, salience_dict)

    assert all(
        all(rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(docs_texts_clean[doc_name])
            for salient_span in salient_spans)
        for doc_name, salient_spans in salience_dict.items()
    ), "not all selected content is traceable"

    return {"final_output": salience_dict, "full_model_response": response}


def parse_clustering_response(response, prompt):
    original_response = deepcopy(response)
    if "So, the highlighted spans are clustered as follows:" in response:
        response = response.split("So, the highlighted spans are clustered as follows:")[-1].strip()
    clusters = json.loads(response)
    assert all(type(elem)==dict for elem in clusters), "not all elems in jsoned response are dictionaries"
    assert all(len(elem)==1 for elem in clusters), "not all dicts in jsoned response are of length 1"
    assert all("cluster" in elem.keys() for elem in clusters), "not all keys are named \"cluster\""
    assert all(type(list(elem.values())[0])==list for elem in clusters), "not all values are lists"
    assert all(all(type(elem)==int for elem in list(cluster_elem.values())[0]) for cluster_elem in clusters), "not all elems of lists are integers"

    curr_instance_part = prompt.split("The highlighted spans are:")[-1]
    highlights_indices = re.findall(r'\n\d+\. ', curr_instance_part)
    prompt_indices = [int(elem.replace("\n", "").replace(".", "")) for elem in highlights_indices]
    response_indices = [i for elem in clusters for i in elem['cluster']]
    assert all(i in prompt_indices for i in response_indices), "not all indices in the lists are actual highlights"

    return {"final_output":clusters,
            "full_model_response":original_response}


def parse_e2e_only_setting_response(response, *args, **kwargs):
    original_response = deepcopy(response)
    if response.strip().lower().startswith("answer:"):
        response = response[len("answer:"):].strip()

    return {"final_output":response,
            "full_model_response":original_response}


def parse_ALCE_response(response, prompt):
    original_response = deepcopy(response)
    if response.lower().startswith("answer:"):
        response = response[len("answer:"):].strip()

    all_citations = []
    for match in re.finditer(r"\[\d+\]", response):
        start = match.start()
        end = match.end()
        matched_text = match.group()
        all_citations.append((matched_text, start, end))
    assert all(tpl[0]==response[tpl[1]:tpl[2]] for tpl in all_citations), "misalignment in the citation extraction"

    curr_instance_part = prompt.split("If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.")[-1]
    docs_names = re.findall(r'Document (\[\d+\]):', curr_instance_part)
    assert all(key in docs_names for key in [tpl[0] for tpl in all_citations]), "not only relevant documents are included"

    response_no_citations = re.sub(r"\[\d+\]", "", response)
    nlp = _get_spacy_nlp()
    response_sents = [sent.text.strip() for sent in nlp(response_no_citations).sents]

    if any(len(find_substring_indices(response, sent))>1 and rmv_spaces_and_punct(sent) for sent in response_sents):
        logging.warning(f"at least one sentence was found in more than one mention was found in the generated text so ensure citations are properly paired: {sent}")

    sent_to_citation_mapping = []
    prev_sent_end_idx = -1
    for sent in response_sents:
        if not rmv_spaces_and_punct(sent):
            sent_to_citation_mapping.append({"sent":sent,
                                             "spans":None,
                                             "cited_docs":[]})
            continue
        sent_start_idx = find_substring_indices(response, sent)
        if len(sent_start_idx)>1:
            sent_start_idx = min([elem for elem in sent_start_idx if elem>prev_sent_end_idx])
        else:
            sent_start_idx = sent_start_idx[0]
        sent_end_idx = sent_start_idx + len(sent)
        sent_to_citation_mapping.append({"sent":sent,
                                         "spans":(sent_start_idx, sent_end_idx),
                                         "cited_docs":[]})

        prev_sent_end_idx = sent_end_idx

    for citation in all_citations:
        curr_doc_num = int(citation[0].replace("[", "").replace("]", ""))
        citation_start_idx = citation[1]
        relevant_sents_i = [i for i,elem in enumerate(sent_to_citation_mapping) if elem['spans'] and elem['spans'][1]<citation_start_idx]
        if not relevant_sents_i:
            logging.info(f"in the following model's response the citation {citation[0]} came before any sentence - so it is ignored:\n{response}")
            continue
        cited_sent_i = max(relevant_sents_i)
        sent_to_citation_mapping[cited_sent_i]['cited_docs'].append(curr_doc_num)

    final_response = [{'sent':elem['sent'],
                       'cited_docs':sorted(list(set(elem['cited_docs'])))} for elem in sent_to_citation_mapping]

    return {"final_output":final_response,
            "response_with_citations" : response,
            "response_without_citations" : response_no_citations,
            "full_model_response" : original_response}


def parse_ambiguity_highlight_response(response, prompt=None):
    doc_spans_pairs = _split_response_into_doc_blocks(response)
    if not doc_spans_pairs:
        raise Exception("no documents were found in ambiguity_highlight output")

    ha_dict = {}
    for doc_id, spans_blob in doc_spans_pairs:
        if spans_blob == "":
            spans = []
        else:
            if "<SPAN_DELIM>" in spans_blob:
                spans = [s.strip() for s in spans_blob.split("<SPAN_DELIM>")]
            else:
                spans = [s.strip() for s in re.split(r"\s*-\s*|\n+", spans_blob)]

            spans = [s for s in spans if rmv_spaces_and_punct(s)]

        spans = [_clean_highlight_markers(s).strip() for s in spans]
        spans = [s for s in spans if rmv_spaces_and_punct(s)]
        spans = _dedupe_preserve_order(spans)
        ha_dict[doc_id] = spans

    docs_texts, docs_names, docs_block = _extract_last_instance_docs_from_prompt(prompt or "")
    ignored_non_target_docs = []
    if docs_names:
        expected_set = set(docs_names)
        returned_set = set(ha_dict.keys())
        intersect = returned_set & expected_set

        ignored_non_target_docs = sorted(list(returned_set - expected_set))

        if returned_set and not intersect:
            raise Exception("not only relevant documents are included")

        ha_dict = {k: v for k, v in ha_dict.items() if k in expected_set}

        for dn in docs_names:
            ha_dict.setdefault(dn, [])

        docs_texts_clean = {k: _clean_highlight_markers(v) for k, v in docs_texts.items()}
        ha_dict = adapt_highlights_to_doc_alignments(docs_texts_clean, ha_dict)

        assert all(
            all(rmv_spaces_and_punct(_clean_highlight_markers(s)) in rmv_spaces_and_punct(docs_texts_clean[doc_name])
                for s in spans)
            for doc_name, spans in ha_dict.items()
        ), "not all selected content is traceable"

        hs_spans = _extract_hs_spans_from_docs_block(docs_block)
        removed_nested = {}
        if hs_spans:
            for doc_name, spans in list(ha_dict.items()):
                kept, removed = _filter_out_nested_or_duplicate_against_hs(spans, hs_spans)
                ha_dict[doc_name] = kept
                if removed:
                    removed_nested[doc_name] = removed

            for dn in docs_names:
                ha_dict.setdefault(dn, [])

        out = {
            "final_output": ha_dict,
            "full_model_response": response,
            "removed_nested_or_duplicate": removed_nested,
        }
        if ignored_non_target_docs:
            out["ignored_non_target_docs"] = ignored_non_target_docs
        return out

    return {"final_output": ha_dict, "full_model_response": response}
