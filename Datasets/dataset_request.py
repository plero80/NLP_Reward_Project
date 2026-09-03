import logging
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, DatasetDict



logger = logging.getLogger(__name__)

class RequestDataset(Dataset):
    
    def __init__(
        self,
        requests: list[str],
        tokenizer_name: str,
        dic: dict[str, list[Any]] | None = None,
        do_dict: bool = False,
        _tokenizer=None,
    ) -> None:
        self.tokenizer = (
            _tokenizer
            if _tokenizer is not None
            else AutoTokenizer.from_pretrained(tokenizer_name)
        )
        self.tokenizer_name = tokenizer_name

        if do_dict is True:

            if dic is None:
                raise ValueError("Invalid dictionary to init the class")
            if "prompts" not in dic:
                raise ValueError("Invalid dictionary to init the class")
            self.columns = {name: list(values) for name, values in dic.items()}

        else:
            if not isinstance(requests, list):
                raise TypeError(f"Expected list but got: {type(requests)}")
            if not all(isinstance(request, str) for request in requests):
                raise TypeError("all requests must be strings")

            self.columns = {
                "prompts" : [],
            }
            
            self.columns["prompts"] = list(requests)
    
    
    
    @classmethod
    def from_raw(
        cls,
        ds: Dataset,
        tokenizer_name: str,
        start: int = 0,
        end: int | None = None,
    ):
        requests = []
        ds_dict = ds["train"] if isinstance(ds, dict) else ds
        chosen = ds_dict["chosen"]
        if end is None:
            end = len(chosen)
        if start < 0 or end < start:
            raise ValueError("end must be greater than or equal to start")

        logger.info("We have %s examples", end - start)
        # Find the human request in dataset.
        for train_example in chosen[start:end]:
            
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

    def save_full(self, path: str | Path) -> Path:
        """Save every dataset column in a safe, versioned JSON artifact."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "tokenizer_name": self.tokenizer_name,
            "columns": self.columns,
        }
        try:
            serialized = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "RequestDataset columns must contain JSON-compatible values"
            ) from error
        destination.write_text(serialized + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load_full(
        cls,
        path: str | Path,
        *,
        tokenizer=None,
    ) -> "RequestDataset":
        """Load a dataset created by :meth:`save_full`."""
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Policy dataset is not valid JSON: {source}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("Policy dataset JSON must contain an object")
        required_fields = {"format_version", "tokenizer_name", "columns"}
        if set(payload) != required_fields:
            raise ValueError(
                "Policy dataset fields must be exactly: "
                f"{sorted(required_fields)}"
            )
        if payload["format_version"] != 1:
            raise ValueError(
                "Unsupported policy dataset format_version: "
                f"{payload['format_version']!r}"
            )
        tokenizer_name = payload["tokenizer_name"]
        columns = payload["columns"]
        if not isinstance(tokenizer_name, str) or not tokenizer_name:
            raise ValueError("Policy dataset tokenizer_name must be a string")
        if not isinstance(columns, dict):
            raise ValueError("Policy dataset columns must be an object")

        return cls.from_dict(columns, tokenizer_name, tokenizer=tokenizer)
        
        
    def get_list(self,name: str, start:int, end:int) -> list[str]:
        
        if self.column_name_exists(name):
            return self.columns[name][start : end]

        else:
            raise ValueError("Column name doesn't exists")


    @classmethod
    def from_dict(cls, dic: dict, tokenizer_name: str, *, tokenizer=None):
        """Assuming that list given already filtered"""

        if "prompts" not in dic:
            raise ValueError(""" Have to contain "prompts" as keys  """)

        if not isinstance(dic["prompts"], list):
            raise ValueError("the values for prompts must be a list")

        prompt_count = len(dic["prompts"])
        for name, values in dic.items():
            if not isinstance(values, list):
                raise ValueError(f"the values for {name} must be a list")
            if len(values) != prompt_count:
                raise ValueError(
                    f"column {name} has {len(values)} items, expected {prompt_count}"
                )

        return cls([], tokenizer_name, dic, True, _tokenizer=tokenizer)
        
        
    @classmethod
    def reset(cls, dataset):
        if not isinstance(dataset, RequestDataset):
            raise TypeError("Invalid dataset. Needs to be with type RequestDataset")
        
        if not dataset.column_name_exists("prompts") :
            raise ValueError("Have to contrain column prompts")
        
        
        return cls.from_processed(dataset.columns["prompts"], dataset.tokenizer_name)
        
        
    def get(self, name: str) -> list[str]:
        if name not in self.columns:
            raise ValueError("Invalid query")
        
        return self.columns[name]
    
    
    def column_name_exists(self, name: str) -> bool:
        if name in self.columns:
                return True   
            
        return False
        

    def add_column(self, name: str, items: list, tokenizer_name):
        if len(items) != len(self):
            raise ValueError(
                f"column {name} has {len(items)} items, expected {len(self)}"
            )

        new_dict = self.columns.copy()
        new_dict[name] = list(items)
        return RequestDataset.from_dict(new_dict, tokenizer_name)

    
    def truncate(self, start:int, end:int) -> None:
        self.columns["prompts"] = self.columns["prompts"][start: end]
    

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self):
        return len(self.columns["prompts"])

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.columns


    def __setitem__(self, key, value):
        if key not in self.columns and not isinstance(value, list):
            raise ValueError("Key isn't exist or value isn't a list")

        self.columns[self] = value
    
    
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
