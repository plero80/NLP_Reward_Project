from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from Models.models import GenerateModel

import logging
logger = logging.getLogger(__name__)

NAME = "Policy"



class PolicyModel(GenerateModel):
    
    def __init__(self, model_name ) -> None:
        
        self.model_name: str = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        
        
    def generate(self, prompt: str) -> str:
        logger.info(NAME + " generating text")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )

        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return generated_text
    
    
    def generate_with_question(self, prompt: str) -> str:
        answer = self.generate(prompt)
        
        full_text = f"""
            Request: {prompt}
            Model answer: {answer}
        """
        
        return full_text