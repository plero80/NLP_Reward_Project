from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Any, Protocol, cast
import torch
from datasets import Dataset as HFDataset, DatasetDict
from peft import TaskType, get_peft_model


from Datasets.dataset_request import RequestDataset
from Models.models import GenerateModel
from Models.lora import LoRASettings
from Models.runtime import best_dtype, current_device
import logging
from collections.abc import Sequence



logger = logging.getLogger(__name__)
NAME = "Policy Model"


class _CausalLanguageModel(Protocol):
    """The model surface used by ``PolicyModel``.

    This avoids leaking Transformers' internal generation protocol into this
    module.  In Transformers 5, that protocol is currently incompatible with
    Pylance's view of ``PreTrainedModel.device``.
    """

    @property
    def device(self) -> torch.device: ...

    def generate(self, **kwargs: Any) -> torch.LongTensor: ...

    def save_pretrained(self, path: str) -> None: ...

    def to(self, device: torch.device | str) -> Any: ...


class PolicyModel(GenerateModel):
    
    def __init__(
        self,
        model_name: str,
        lora_settings: LoRASettings | None = None,
    ) -> None:
        
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decoder-only models should generally use left padding for generation.
        self.tokenizer.padding_side = "left"
        
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=best_dtype(),
        )
        if lora_settings is not None:
            model = get_peft_model(
                base_model,
                lora_settings.build(TaskType.CAUSAL_LM),
            )
            model.print_trainable_parameters()
        else:
            model = base_model

        # AutoModelForCausalLM and the CAUSAL_LM PEFT wrapper both implement
        # this interface, but their third-party annotations do not express a
        # Pylance-compatible common type.
        self.model = cast(_CausalLanguageModel, model)
        self.model.to(current_device())
        self.dataset: RequestDataset | HFDataset | None = None


    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        
    def generate(self, prompt: str) -> str:
        logger.info("%s: Generating text", NAME)

        generation_model: Any = self.model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = generation_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]


        logger.info("%s: Stop generating text", NAME)
        
        model_answer = PolicyModel._find_model_answer(self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ))
        
        return model_answer
    
    
    def generate_batch(self, prompts: Sequence[str]) -> Sequence[str]:
        logger.info("%s: Generating a batch of %d answers", NAME, len(prompts))

        if not prompts:
            return []

        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generation_model: Any = self.model

        with torch.inference_mode():
            output_ids = generation_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Remove the input prompt tokens from every generated sequence.
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]

        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        answers = [
            PolicyModel._find_model_answer(generated_text)
            for generated_text in generated_texts
        ]

        logger.debug("%s: Generated answers: %s", NAME, answers)

        return answers
    
    
    def set_dataset(self, dataset, batch_size = 8) -> None:
        """Update the answers of the model inside the dataset"""
        
        PolicyModel._check_valid_dataset(dataset)
        self.generate_new_dataset(dataset, batch_size)
        
    
    def add_scores(self, scores: list[float], reward_name: str) -> None:
        if self.dataset is None:
            raise ValueError("The policy doesn't contain a dataset")
        if PolicyModel._is_request_dataset_like(self.dataset):
            self.dataset = self.dataset.add_column(
                reward_name,
                scores,
                self.model_name,
            )
        else:
            if reward_name in self.dataset.column_names:
                self.dataset = self.dataset.remove_columns(reward_name)
            self.dataset = self.dataset.add_column(reward_name, scores)
    
        
    def get_dataset_col(self, name) -> list:
        if self.dataset is None:
            raise ValueError("The policy doesn't contain a dataset")
        if PolicyModel._is_request_dataset_like(self.dataset):
            return self.dataset.get(name)
        return list(self.dataset[name])
        
        
    @staticmethod
    def _is_request_dataset_like(dataset: object) -> bool:
        return all(
            hasattr(dataset, name)
            for name in ("columns", "column_name_exists", "get", "add_column")
        )

    @staticmethod
    def _prompt_column(dataset: object) -> str:
        if PolicyModel._is_request_dataset_like(dataset):
            if dataset.column_name_exists("prompts"):
                return "prompts"
            raise ValueError("The dataset must contain column: prompts")

        if isinstance(dataset, HFDataset):
            if "prompts" in dataset.column_names:
                return "prompts"
            if "prompt" in dataset.column_names:
                return "prompt"
            raise ValueError("The dataset must contain column: prompt or prompts")

        raise TypeError(
            "Invalid dataset type. Type needed: RequestDataset, "
            "datasets.Dataset, or datasets.DatasetDict with a train split"
        )

    @staticmethod
    def _normalize_dataset(dataset: object) -> object:
        if isinstance(dataset, DatasetDict):
            if "train" not in dataset:
                raise ValueError("DatasetDict must contain a train split")
            return dataset["train"]
        return dataset
        
        
    @classmethod
    def _check_valid_dataset(cls, dataset) -> None:
        cls._prompt_column(cls._normalize_dataset(dataset))
        
    
    
    def generate_new_dataset(
        self,
        dataset: RequestDataset | HFDataset | DatasetDict,
        batch_size: int = 8,
    ) -> RequestDataset | HFDataset:
        """Put a new updated intance of dataset inside dataset field"""
        dataset = PolicyModel._normalize_dataset(dataset)
        PolicyModel._check_valid_dataset(dataset)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        logger.info("Starting to generate new answers to the prompts")
        prompt_column = PolicyModel._prompt_column(dataset)
        prompts = dataset[prompt_column]
        answers: list[str] = []

        total_batches = (len(prompts) + batch_size - 1) // batch_size
        progress_interval = max(1, total_batches // 4)
        count_batch = 0
        
        for start in range(0, len(prompts), batch_size):
            raw_prompt_batch = list(prompts[start : start + batch_size])
            if not all(isinstance(prompt, str) for prompt in raw_prompt_batch):
                raise TypeError("all prompts must be strings")
            prompt_batch = cast(list[str], raw_prompt_batch)
            answers.extend(self.generate_batch(prompt_batch))
            
            
            count_batch += 1
            if count_batch % progress_interval == 0 or count_batch == total_batches:
                logger.debug(
                    "Policy batch generated %s / %s",
                    count_batch,
                    total_batches,
                )
            
        if len(answers) != len(dataset):
            raise RuntimeError(
                "generation must return exactly one answer for every prompt"
            )

        if PolicyModel._is_request_dataset_like(dataset):
            self.dataset = dataset.add_column("answers", answers, self.model_name)
        else:
            if "answers" in dataset.column_names:
                dataset = dataset.remove_columns("answers")
            self.dataset = dataset.add_column("answers", answers)
        return self.dataset
        
        
    @staticmethod
    def _find_model_answer(text):
        #start = text.rfind("Answer:")
        start = -1
        if start == -1:
            return text
               
        text = text[start: ]
        return text
    
    
    def generate_with_question(self, prompt: str) -> tuple:
        answer = self.generate(prompt)
        
        full_text = f"""
            Request: {prompt}
            Model answer: {answer}
        """
        
        logger.debug("%s: Policy model output: %s", NAME, full_text)
        
        return prompt, answer
