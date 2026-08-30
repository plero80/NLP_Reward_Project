from typing import Protocol
from transformers import PreTrainedModel


class GenerateModel(Protocol):
    
    
    
    def generate(self, prompt: str) -> str:
        ...
    
    def generate_with_question(self, prompt: str) -> str:
        ...
        
    

class ScoreModel(Protocol):
    
    base_model_prefix: str
    model: PreTrainedModel
       
    def score(self, input:str) -> float:
        ...