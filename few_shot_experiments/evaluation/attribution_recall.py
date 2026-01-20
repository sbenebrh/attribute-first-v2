import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from typing import List, Dict
import numpy as np
from string import punctuation
from tqdm import tqdm
from .utils import *
import logging
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

class AttributionRecall:
    def __init__(self, nli_model: str="google/t5_xxl_true_nli_mixture"):
        """
        nli_model: the name of the nli model
        probability_score: whether to use the likelihood score of the model for the generation of "1" or just count number of generated "1"s
        """
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model, 
                                                           use_fast=False)
        
        self.nli_model = AutoModelForSeq2SeqLM.from_pretrained(nli_model, 
                                                               max_memory=get_max_memory(), 
                                                               device_map="auto")

        
    def _run_nli_prompt_wrapper(self, passage, claim, separated_attributions):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """
        def run_nli_prompt(curr_passage, curr_claim):

            input_text = "premise: {} hypothesis: {}".format(curr_passage, curr_claim)
            input_ids = self.nli_tokenizer(input_text, return_tensors="pt").input_ids.to(self.nli_model.device)
            with torch.inference_mode():
                outputs = self.nli_model.generate(input_ids, max_new_tokens=2)
            result = self.nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference = 1 if result == "1" else 0
            return inference
    
        try:
            return run_nli_prompt(passage, claim)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            logging.warning("current instance was too big for the NLI model - will check each attribution separately")
        
        at_least_one_passed = False # make sure that not all attributing texts were too long
        for attrib in separated_attributions:
            try:
                curr_inference = run_nli_prompt(attrib, claim)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                continue
            at_least_one_passed = True
            if curr_inference == 1: # enough that at least one of the attributing texts entails
                return 1
        # if not at_least_one_passed:
        #     raise Exception("All attributed texts were to big for the nli model to handle.")
        return 0



    def _run_nli_prompt_prob_wrapper(self, passage, claim, separated_attributions):
        """
        Run inference for assessing AIS between a premise and hypothesis using the likelihood function.
        Taken from https://huggingface.co/google/t5_11b_trueteacher_and_anli
        """
        def run_nli_prompt_prob(curr_passage, curr_claim):
            input_text = "premise: {} hypothesis: {}".format(curr_passage, curr_claim)
            input_ids = self.nli_tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=2048).input_ids.to(self.nli_model.device)
            decoder_input_ids = torch.tensor([[self.nli_tokenizer.pad_token_id]]).to(self.nli_model.device)
            outputs = self.nli_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits[0], dim=-1)
            one_token_id = self.nli_tokenizer('1').input_ids[0]
            entailment_prob = probs[0, one_token_id].item()
            return entailment_prob

        try:
            return run_nli_prompt_prob(passage, claim)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            logging.warning("current instance was too big for the NLI model - will check each attribution separately")
        
        entailment_prob = -1
        for attrib in separated_attributions:
            try:
                curr_entailment_prob = run_nli_prompt_prob(attrib, claim)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                continue
            if curr_entailment_prob>entailment_prob:
                entailment_prob = curr_entailment_prob

        if entailment_prob==-1:
            # raise Exception("All attributed texts were to big for the nli model to handle.")
            return 0.0 
        
        return entailment_prob

    
    def get_sent_score(self, sent_attributions, probability_score=False):
        if len(sent_attributions['attribution'])==0: # no attribution
            return 0
        adapted_attributions = [a.replace("\n", " ") for a in sent_attributions['attribution'].values()] # replace all "\n" with spaces (because "\n" will be used as delim - similar to the paper "Enabling Large Language Models to Generate Text with Citation")
        adapted_attributions = [f"{a}." if not a[-1] in punctuation else a for a in adapted_attributions] # add period if attribution doesn't end with some punctuation
        concat_attributions = "\n ".join(adapted_attributions)
        if probability_score:
            return self._run_nli_prompt_prob_wrapper(passage=concat_attributions, claim=sent_attributions['sentence'], separated_attributions=adapted_attributions)
        else:
            return self._run_nli_prompt_wrapper(passage=concat_attributions, claim=sent_attributions['sentence'], separated_attributions=adapted_attributions)

    def get_instance_score(self, curr_attributions, probability_score=False):
        sents_scores = [self.get_sent_score(sent_attributions, probability_score=probability_score) for sent_attributions in curr_attributions]
        return np.mean(sents_scores)

    def evaluate(self, attributions: List[List[Dict]], only_entail_cnt_scores: bool = False, only_likelihood_scores: bool = False) -> List:
        """
        attributions: a list of a list of dictionaries with keys "sentence" (the output sentence) and "attribution" (a dictionary where the key is the name of the input document, and the value is the concatentation of the attributing spans within the document)
        """
        non_probability_scores, probability_scores = None, None
        if not only_likelihood_scores:
            non_probability_scores = [100*round(self.get_instance_score(instance, probability_score=False), 4) for instance in tqdm(attributions)]
        if not only_entail_cnt_scores:
            probability_scores = [100*round(self.get_instance_score(instance, probability_score=True), 4) for instance in tqdm(attributions)]
        
        return {"entail_cnt_scores" : non_probability_scores,
                "likelihood_scores" : probability_scores}
        # return {"entail_cnt_scores" : non_probability_scores}