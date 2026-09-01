import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from typing import Sequence
from transformers import AutoTokenizer




logger = logging.getLogger(__name__)

class RequestDataset(Dataset):
    
    def __init__(self, requests: list[str], tokenizer_name: str, dic:dict = {}, do_dict: bool = False) -> None:

        if do_dict is True:

            if "prompts" not in dic:
                raise ValueError("Invalid dictionary to init the class")
            self.columns = dic

        else:
            if not isinstance(requests, list):
                raise TypeError(f"Expected list but got: {type(requests)}")
            if not all(isinstance(request, str) for request in requests):
                raise TypeError("all requests must be strings")

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.columns = dic
            self.columns = {
                "prompts" : [],
            }

            self.columns["prompts"] = list(requests)
    
    
    
    
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
        torch.save(self.columns["prompts"], path)
        
        
    def get(self, start:int, end:int) -> list[str]:
        return self.columns["prompts"][start:end]

    @classmethod
    def from_dict(cls, dic: dict, tokenizer_name: str):
        """Assuming that list given already filtered"""

        if "prompts" not in dic:
            raise ValueError(""" Have to contain "prompts" as keys  """)

        if not isinstance(dic["prompts"], list):
            raise ValueError("the values for prompts must be a list")

        return cls([], tokenizer_name, dic, True)
        
        
        
    def get(self, name: str) -> list[str]:
        if name not in self.columns:
            raise ValueError("Invalid query")
        
        return self.columns[name]
    
    
    def column_name_exists(self, name: str) -> bool:
        if name in self.columns:
                return True   
            
        return False
        

    def add_column(self, name: str, prompts: Sequence[str], tokenizer_name):

        new_dict = self.columns.copy()
        new_dict[name] = prompts
        return RequestDataset.from_dict(new_dict, tokenizer_name)

    
    def truncate(self, start:int, end:int) -> None:
        self.columns["prompts"] = self.columns["prompts"][start: end]
    

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self):
        return len(self.columns["prompts"])
    
    
    def __getitem__(self, index) -> Any:

        if isinstance(index, str):
            if index not in self.columns:
                raise ValueError("Key isnt't exists")

            return self.columns[index]
        
        prompt = self.columns["prompts"][index]
        
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
