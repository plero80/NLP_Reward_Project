from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from Models.models import GenerateModel
import logging


logger = logging.getLogger(__name__)
NAME = "Policy Model"


class PolicyModel(GenerateModel):
    
    def __init__(self, model_name ) -> None:
        
        self.model_name: str = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        self.lock_log = True
        
        
    def generate(self, prompt: str) -> str:
        logger.info("%s: Generating text", NAME)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]


        logger.info("%s: Stop generating text", NAME)
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
    
    
    def generate_with_question(self, prompt: str) -> str:
        answer = self.generate(prompt)
        
        full_text = f"""
            Request: {prompt}
            Model answer: {answer}
        """
        
        logger.debug("%s: Policy model output: %s", NAME, full_text)
        
        return full_text