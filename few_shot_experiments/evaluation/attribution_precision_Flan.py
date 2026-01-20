from typing import List, Dict
from tqdm import tqdm
from .attribution_recall_Flan import AttributionRecallFlan

class AttributionPrecisionFlan:
    def __init__(self, model_name: str="google/flan-t5-xxl"):
        self.model = AttributionRecallFlan(verbose=False, model_name=model_name)
    
    def get_instance_score(self, curr_attributions):
        recall_scores = [self.model.get_instance_score([attrib])[0] for attrib in curr_attributions]
        citation_precisions = []
        for i,sent_instance in enumerate(curr_attributions):
            for doc_name,attrib in sent_instance['attribution'].items():
                if recall_scores[i]==0: # the entire instance has a recall=0
                    citation_precisions.append(0)
                else:
                    only_attrib_score = self.model.get_instance_score([{'sentence':sent_instance['sentence'],
                                                                        'attribution':{doc_name:attrib}}])[0]
                    if only_attrib_score == 1: # the entire proposition is relevant
                        citation_precisions.append(1)
                    else:
                        exclude_attrib_score = self.model.get_instance_score([{'sentence':sent_instance['sentence'],
                                                                        'attribution':{key:value for key,value in sent_instance['attribution'].items() if key!=doc_name}}])[0]
                        citation_precisions.append(1-exclude_attrib_score)
        return len([elem for elem in citation_precisions if elem==1])/len(citation_precisions)
    
    def evaluate(self, attributions: List[List[Dict]]) -> List:
        """
        attributions: a list of a list of dictionaries with keys "sentence" (the output sentence) and "attribution" (a dictionary where the key is the name of the input document, and the value is the concatentation of the attributing spans within the document)
        """

        return [round(100*self.get_instance_score(instance), 4)  for instance in tqdm(attributions)]
