import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from typing import List, Dict
import numpy as np
from string import punctuation
from tqdm import tqdm
import re
from .utils import *


class AttributionRecallFlan:
    def __init__(self, model_name: str="google/flan-t5-xxl", batch_size: int = 5, entailment_word: str = "Entailment", contradiction_word: str = "Contradiction", neutral_word: str = "Neutral", verbose: bool=True):
        """
        model_name: the name of the model
        batch_size: size of inference batch
        entailment_word: word used by model to indicate entailment
        contradiction_word: word used by model to indicate contradiction
        neutral_word: word used by model to indicate neutral
        verbose: whether to add a progress bar or not
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                           use_fast=False)
        
        self.nli_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,  
                                                            max_memory=get_max_memory(), 
                                                            device_map="auto")
        self.batch_size = batch_size
        self.entailment = entailment_word
        self.contradiction = contradiction_word
        self.neutral = neutral_word
        self.entailment_tkn_ids = self.tokenizer.encode(self.entailment, add_special_tokens=False)
        self.contradiction_tkn_ids = self.tokenizer.encode(self.contradiction, add_special_tokens=False)
        self.neutral_tkn_ids = self.tokenizer.encode(self.neutral, add_special_tokens=False)
        self.max_special_tkn_len = max(len(self.entailment_tkn_ids), len(self.contradiction_tkn_ids), len(self.neutral_tkn_ids))
        self.verbose = verbose

    def get_prompt(self, premise, hypothesis):
        prompt = f"""
                        ### Instruction: Read the following and determine if the hypothesis can be inferred from the premise. 
                        Options: Entailment, Contradiction, or Neutral 

                        ### Input: 
                        Premise: {premise}
                        Hypothesis: {hypothesis} 

                        ### Response (choose only one of the options from above):
                    """
        return re.sub(' +', ' ', prompt.strip()) # replace consecutive spaces with a single space


    def get_scores(self, curr_inputs):
        n_batches = int(np.ceil(len(curr_inputs) / self.batch_size))
        all_outputs = []
        for batch_i in range(n_batches):
            input_ids = self.tokenizer.batch_encode_plus(curr_inputs[batch_i*self.batch_size:(batch_i+1)*self.batch_size], 
                                                            padding=True, 
                                                            truncation=True,
                                                            return_tensors="pt")["input_ids"].to(self.nli_model.device)
            
            outputs = self.nli_model.generate(input_ids,
                                              min_length=1,
                                              max_new_tokens=self.max_special_tkn_len,
                                              output_scores=True,
                                              return_dict_in_generate=True,
                                              early_stopping=True,
                                              num_beams=1)
            all_outputs.append(outputs)
        all_scores = [torch.cat([curr_output.scores[i] for curr_output in all_outputs]) for i in range(self.max_special_tkn_len)]
        outputs_scores = [F.softmax(all_scores[i], dim=1) for i in range(self.max_special_tkn_len)]
        outputs_scores_entailment = torch.prod(torch.stack([outputs_scores[i][:, tkn_id] for i,tkn_id in enumerate(self.entailment_tkn_ids)]), dim=0)
        outputs_scores_contradiction = torch.prod(torch.stack([outputs_scores[i][:, tkn_id] for i,tkn_id in enumerate(self.contradiction_tkn_ids)]), dim=0)
        outputs_scores_neutral = torch.prod(torch.stack([outputs_scores[i][:, tkn_id] for i,tkn_id in enumerate(self.neutral_tkn_ids)]), dim=0)

        
        return all_outputs, outputs_scores_entailment, outputs_scores_contradiction, outputs_scores_neutral

    def get_instance_score(self, curr_attributions):
        curr_prompts = [self.get_prompt(premise="\n ".join([f"{a}.".replace("\n", " ") if not a[-1] in punctuation else a.replace("\n", " ") for a in elem['attribution'].values()]), hypothesis=elem['sentence']) for elem in curr_attributions]
        curr_scores = self.get_scores(curr_prompts)

        # number of "Entailment"
        curr_outputs = [self.tokenizer.batch_decode(elem.sequences, skip_special_tokens=True) for elem in curr_scores[0]]
        curr_outputs = [output for elem in curr_outputs for output in elem]
        abs_score = len([elem for elem in curr_outputs if self.entailment in elem])/len(curr_outputs) if curr_outputs else 0

        # likelihood score
        prob_score = torch.mean(curr_scores[1])
        
        return abs_score, prob_score.item()

    def evaluate(self, attributions: List[List[Dict]], *args, **kwargs) -> List:
        """
        attributions: a list of a list of dictionaries with keys "sentence" (the output sentence) and "attribution" (a dictionary where the key is the name of the input document, and the value is the concatentation of the attributing spans within the document)
        """
        if self.verbose:
            scores = [self.get_instance_score(instance)  for instance in tqdm(attributions)]
        else:
            scores = [self.get_instance_score(instance)  for instance in attributions]
        return {"entail_cnt_scores" : [round(100*score[0], 4) for score in scores],
                "likelihood_scores" : [round(100*score[1], 4) for score in scores]}