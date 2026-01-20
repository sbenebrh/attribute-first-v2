def _as_str(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return None


def _normalize_llm_return(ret):
    """Normalize llm_api.query_llm return value.

    Expected patterns in this repo:
      - (output_text, usage_dict)
      - output_text
      - None
    """
    if isinstance(ret, tuple) and len(ret) == 2:
        out, usage = ret
        return _as_str(out), usage
    return _as_str(ret), None
import os, sys
import json
import re
import numpy as np

sys.path.append('../')
try:
    # if run as a package module
    from ..llm_api import query_llm  # type: ignore
except Exception:
    # if run from repo root where llm_api.py is importable
    from llm_api import query_llm


need_citation_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation. 
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".\n\n{}
"""


def cat_qa_and_statement(question, answer, statement):
    prompt = f"<question>\n{question.strip()}\n</question>\n\n<response>\n{answer.strip()}\n</response>\n\n<sentence>\n{statement.strip()}\n</sentence>"
    return prompt


def need_citation_to_score(s):
    if s is None:
        return None
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "yes".lower() in l.lower():
            return 1
        else:
            return 0
    else:
        return None


def need_citation(question, answer, sentence):
    prompt = need_citation_prompt_template.format(cat_qa_and_statement(question, answer, sentence))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        ret = query_llm(
            msg,
            model=GPT_MODEL,
            temperature=0 if t == 0 else 1,
            max_new_tokens=10,
            stop="Analysis:",
            return_usage=True,
        )
        output, usage = _normalize_llm_return(ret)
        score = need_citation_to_score(output)

        if score is None:
            print("Unexcept need_citation output: ", output)
            # If the backend returned nothing, just retry.
            if output is None:
                continue
            # Some providers return a safety trigger string.
            if isinstance(output, str) and 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage


support_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical. 
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating. 
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""


def cat_question_statement_context(question, statement, context):
    prompt = f"<question>\n{question.strip()}\n</question>\n\n<statement>\n{statement.strip()}\n</statement>\n\n<snippet>\n{context.strip()}\n</snippet>\n\n"
    return prompt


def support_level_to_score(s):
    if s is None:
        return None
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "fully".lower() in l.lower():
            return 1
        elif "partially".lower() in l.lower():
            return 0.5
        else:
            return 0
    else:
        return None


def is_support(question, statement, context):
    if context == "":
        return 0, "No matched citation", None
    prompt = support_prompt_template.format(cat_question_statement_context(question, statement, context))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        ret = query_llm(
            msg,
            model=GPT_MODEL,
            temperature=0 if t == 0 else 1,
            max_new_tokens=10,
            stop="Analysis:",
            return_usage=True,
        )
        output, usage = _normalize_llm_return(ret)
        score = support_level_to_score(output)

        if score is None:
            print("Unexcept support output: ", output)
            if output is None:
                continue
            if isinstance(output, str) and 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage


def score_recall(question, answer, statements_with_citations):
    # Recall: for each statement, either:
    #  - it has citations -> check support against concatenated cited snippets
    #  - it has no citations -> check whether it *needs* citation; if it does, penalize
    # This function must NEVER crash due to LLM/API flakiness; when uncertain, be conservative.
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js.get('statement', ''), js.get('citation', [])
        matched_citations = [c.get('cite', '') for c in citations if isinstance(c, dict)]
        matched_citations = [c for c in matched_citations if isinstance(c, str) and c.strip()]

        if len(matched_citations) > 0:
            context = '\n\n'.join(matched_citations).strip()
            score, output, usage = is_support(question, statement, context)
            usages.append(usage)
            js.update({
                "support_output": output,
                "support_score": score,
            })

            if score is None:
                # Conservative fallback: treat as unsupported.
                js["support_error"] = f"support_score=None; output={output!r}"
                scores.append(0.0)
            else:
                try:
                    scores.append(float(score))
                except Exception:
                    js["support_error"] = f"support_score_not_float={score!r}; output={output!r}"
                    scores.append(0.0)
        else:
            score, output, usage = need_citation(question, answer, statement)
            usages.append(usage)
            # If the sentence needs citation (score==1), then recall credit is 0.
            # If it does not need citation (score==0), then recall credit is 1.
            js.update({
                "support_output": output,
                "support_score": (1 - score) if score is not None else None,
            })

            if score is None:
                # Conservative fallback: assume it needs citation -> credit 0.
                js["support_error"] = f"need_citation_score=None; output={output!r}"
                scores.append(0.0)
            else:
                try:
                    scores.append(float(1 - score))
                except Exception:
                    js["support_error"] = f"need_citation_score_not_float={score!r}; output={output!r}"
                    scores.append(0.0)

    if len(scores) == 0:
        return 0, statements_with_citations, usages
    return float(np.mean(scores)), statements_with_citations, usages


relevant_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almostly unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating. 
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""


def relevant_level_to_score(s):
    if s is None:
        return None
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "unrelevant".lower() in l.lower():
            return 0
        else:
            return 1
    else:
        return None


def is_relevant(question, statement, citation):
    prompt = relevant_prompt_template.format(cat_question_statement_context(question, statement, citation))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        ret = query_llm(
            msg,
            model=GPT_MODEL,
            temperature=0 if t == 0 else 1,
            max_new_tokens=10,
            stop="Analysis:",
            return_usage=True,
        )
        output, usage = _normalize_llm_return(ret)
        score = relevant_level_to_score(output)

        if score is None:
            print("Unexcept relevant output: ", output)
            if output is None:
                continue
            if isinstance(output, str) and 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage


def score_precision(question, answer, statements_with_citations):
    # Precision: for each citation attached to a statement, check whether the cited snippet is relevant.
    # This function must NEVER crash due to LLM/API flakiness; when uncertain, be conservative.
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js.get('statement', ''), js.get('citation', [])
        if not isinstance(citations, list):
            continue
        for c in citations:
            if not isinstance(c, dict):
                continue
            cite_text = c.get('cite', '')
            score, output, usage = is_relevant(question, statement, cite_text)
            usages.append(usage)
            c.update({
                "relevant_output": output,
                "relevant_score": score,
            })

            if score is None:
                # Conservative fallback: treat as not relevant.
                c["relevant_error"] = f"relevant_score=None; output={output!r}"
                scores.append(0.0)
            else:
                try:
                    scores.append(float(score))
                except Exception:
                    c["relevant_error"] = f"relevant_score_not_float={score!r}; output={output!r}"
                    scores.append(0.0)

    if len(scores) == 0:
        return 0, statements_with_citations, usages
    return float(np.mean(scores)), statements_with_citations, usages


def get_citation_score(js, max_statement_num=None):
    question, answer, statements_with_citations = js['query'], js['prediction'], js['statements']
    answer = re.sub(r"<cite>.*?</cite>", "", answer, flags=re.DOTALL)
    answer = answer.replace('<statement>', '').replace('</statement>', '')
    if max_statement_num and len(statements_with_citations) > max_statement_num:
        print(f"Too many statments, only evaluate {max_statement_num} of {len(statements_with_citations)} statements.")
        statements_with_citations = statements_with_citations[:max_statement_num]
    recall, _, usages1 = score_recall(question, answer, statements_with_citations)
    precision, _, usages2 = score_precision(question, answer, statements_with_citations)
    js['citation_recall'] = recall
    js['citation_precision'] = precision
    js['citation_f1'] = 0.0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)
    js['gpt_usage'] = {
        'prompt_tokens': sum((x or {}).get('prompt_tokens', 0) for x in (usages1 + usages2)),
        'completion_tokens': sum((x or {}).get('completion_tokens', 0) for x in (usages1 + usages2)),
    }
    return js
