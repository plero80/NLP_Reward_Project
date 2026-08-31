import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer




logger = logging.getLogger(__name__)

class RequestDataset(Dataset):
    
    def __init__(self, requests: list[str], tokenizer_name: str) -> None:
        
        if not isinstance(requests, list):
            raise TypeError(f"Expected list but got: {type(requests)}")
        if not all(isinstance(request, str) for request in requests):
            raise TypeError("all requests must be strings")

        self.ds: list[str] = list(requests)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        
    
    
    @classmethod
    def from_raw(cls, ds: Dataset , tokenizer_name: str):
        requests = []
        ds_dict = ds["train"] if isinstance(ds, dict) else ds

        logger.info("We have %s examples", len(ds_dict["chosen"]))
        # Find the human request in dataset.
        for train_example in ds_dict["chosen"]:
            
            request = cls._find_human_request(train_example)
            

            if request != "":
                requests.append(request)

        
        logging.info("After filtering, we have : %s", len(requests))
        return cls(requests, tokenizer_name)
        
        
    @classmethod
    def from_processed(cls, requests: list[str], tokenizer_name: str):
        return cls(requests, tokenizer_name)
        
    
    @staticmethod
    def _find_human_request(text):
        start = text.find("Human:")
        
        if start == -1:
            return ""
        
        end = text.find("?", start)
        
        if end == -1:
            return ""
        
        
        text = text[start: end + 1]
        return text
    
    @classmethod
    def load(cls, path, tokenizer_name: str):
        logger.info("PATH: %s", path)

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File: {path} not found")

        loaded = torch.load(path, weights_only=False)
        logger.debug("Expecting list: %s", type(loaded))

        if not isinstance(loaded, list):
            raise TypeError(f"RequestDataset accepts only list, got: {type(loaded)}")

        return cls.from_processed(loaded, tokenizer_name)

    def save(self, path):
        torch.save(self.ds, path)
        
        
    def get(self, start:int, end:int) -> list[str]:
        return self.ds[start:end]



    
    def truncate(self, start:int, end:int) -> None:
        self.ds = self.ds[start: end]
    

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self):
        return len(self.ds)
    
    
    def __getitem__(self, index) -> Any:
        
        prompt = self.ds[index]
        
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
