from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Any, Protocol, TypeGuard, cast
from pathlib import Path
import torch
from datasets import Dataset as HFDataset, DatasetDict
from peft import PeftConfig, PeftModel, TaskType, get_peft_model


from Datasets.dataset_request import RequestDataset
from Datasets.dataset_classifier import DatasetClassifier


from Models.models import GenerateModel
from Models.lora import LoRASettings
from Models.runtime import best_dtype, current_device
import logging
from collections.abc import Sequence



logger = logging.getLogger(__name__)
NAME = "Policy Model"

DatasetInput = RequestDataset | HFDataset | DatasetDict
NormalizedDataset = RequestDataset | HFDataset


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

    DATASET_FILE_NAME = "policy_dataset.json"

    @staticmethod
    def _configure_tokenizer(tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    def __init__(
        self,
        model_name: str,
        lora_settings: LoRASettings | None = None,
    ) -> None:
        
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._configure_tokenizer(self.tokenizer)
        
        
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
        self.dataset: RequestDataset | None = None


    def save(self, path: str | Path) -> None:
        destination = Path(path).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(destination)
        self.save_dataset(destination)

    def save_dataset(self, checkpoint_directory: str | Path) -> Path | None:
        """Save the policy's complete in-memory dataset beside its weights."""
        destination = (
            Path(checkpoint_directory).expanduser() / self.DATASET_FILE_NAME
        )
        if self.dataset is None:
            if destination.is_file():
                destination.unlink()
            return None
        return self.dataset.save_full(destination)


    def move_to_current_device(self) -> torch.device:
        """Restore the policy to the device selected for active computation."""
        device = current_device()
        self.model.to(device)
        return device
        
        
    def generate(self, prompt: str) -> str:
        logger.info("%s: Generating text", NAME)

        device = self.move_to_current_device()
        generation_model: Any = self.model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

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

        device = self.move_to_current_device()
        inputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
        ).to(device)

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
    
    
    def set_dataset(self, dataset: DatasetInput, batch_size: int = 8) -> None:
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
        
    
    def offload(self) -> None:
        self.model.to("cpu")
    
    
    
    def move_to_gpu(self, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.model.to(device)
        
        
    def return_rows(self) -> tuple[str,str,float,float]:
        if self.dataset is not None:
            return self.dataset["prompts"], self.dataset["answers"], self.dataset["proxy"], self.dataset["judge"]  
        
        else:
            raise ValueError("The dataset doesn't contain all the necessary data to create dataset for training classifier")

        
    def get_dataset_col(self, name) -> list:
        if self.dataset is None:
            raise ValueError("The policy doesn't contain a dataset")
        if PolicyModel._is_request_dataset_like(self.dataset):
            return self.dataset.get(name)
        return list(self.dataset[name])
        
        
    @staticmethod
    def _is_request_dataset_like(dataset: object) -> TypeGuard[RequestDataset]:
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
    def _normalize_dataset(dataset: DatasetInput) -> NormalizedDataset:
        if isinstance(dataset, DatasetDict):
            if "train" not in dataset:
                raise ValueError("DatasetDict must contain a train split")
            return dataset["train"]
        return dataset
        
        
    @classmethod
    def _check_valid_dataset(cls, dataset: DatasetInput) -> None:
        cls._prompt_column(cls._normalize_dataset(dataset))
        
        
    
    def generate_dataset_classifier(self, theta) -> DatasetClassifier:
        
        if self.dataset is None:
            raise ValueError("dataset is None")
        
        
        if not all(list(map(self.dataset.column_name_exists, ["proxy","judge","prompts","answers"]))):
            raise ValueError("Invalid dataset for creating dataset for classifier")
        

        else:
            prompts = self.get_dataset_col("prompts")
            answers = self.get_dataset_col("answers")
            proxy_scores = self.get_dataset_col("proxy")
            judge_scores = self.get_dataset_col("judge")
            
            dataset_classifier = DatasetClassifier(theta)
            dataset_classifier.add(prompts, answers, proxy_scores, judge_scores)
        
        return dataset_classifier
    
    
    
    def generate_new_dataset(
        self,
        dataset: DatasetInput | None = None,
        batch_size: int = 8,
    ) -> RequestDataset | HFDataset | None:
        """Put a new updated intance of dataset inside dataset field"""
        if dataset is None:
            dataset = self.dataset
        if dataset is None:
            raise ValueError("The policy doesn't contain a dataset")

        dataset = PolicyModel._normalize_dataset(dataset)
        dataset = RequestDataset.reset(dataset) # Delete all scores because of new answers.
        
        
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
        
        return self.dataset
        
        
    @staticmethod
    def _find_model_answer(text):
        #start = text.rfind("Answer:")
        start = -1
        if start == -1:
            return text
               
        text = text[start: ]
        return text
    


    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        is_trainable: bool = False,
        device: torch.device | str | None = None,
    ) -> "PolicyModel":
        """Load policy weights saved by PPO or :meth:`save`.

        Full-model checkpoints contain ``config.json`` and
        ``model.safetensors``. PEFT checkpoints instead contain
        ``adapter_config.json`` and adapter weights; for those checkpoints the
        base model is loaded first and the trained adapter is attached.

        If the checkpoint contains a policy dataset artifact, all of its
        columns are restored. PPO optimizer, scheduler, critic, RNG, and
        dataloader state are not restored.
        """
        checkpoint = Path(path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")
        if not checkpoint.is_dir():
            raise NotADirectoryError(
                f"Policy checkpoint must be a directory: {checkpoint}"
            )

        checkpoint_source = str(checkpoint)
        adapter_checkpoint = (checkpoint / "adapter_config.json").is_file()

        if adapter_checkpoint:
            peft_config = PeftConfig.from_pretrained(checkpoint_source)
            base_model_name = peft_config.base_model_name_or_path
            if not base_model_name:
                raise ValueError(
                    "PEFT checkpoint does not identify its base model: "
                    f"{checkpoint}"
                )

            tokenizer_source = (
                checkpoint_source
                if (checkpoint / "tokenizer_config.json").is_file()
                else base_model_name
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=best_dtype(),
            )
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_source,
                is_trainable=is_trainable,
            )
            model_name = base_model_name
        else:
            tokenizer_source = checkpoint_source
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_source,
                dtype=best_dtype(),
            )
            model_name = checkpoint_source

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        cls._configure_tokenizer(tokenizer)

        loaded = cls.__new__(cls)
        loaded.model_name = model_name
        loaded.tokenizer = tokenizer
        loaded.model = cast(_CausalLanguageModel, model)
        loaded.model.to(device if device is not None else current_device())
        dataset_path = checkpoint / cls.DATASET_FILE_NAME
        loaded.dataset = (
            RequestDataset.load_full(dataset_path, tokenizer=tokenizer)
            if dataset_path.is_file()
            else None
        )

        if is_trainable:
            model.train()
        else:
            model.eval()

        return loaded


    def normalize_score_col(self, name, mean, std) -> None:
        if std <= 0:
            raise ValueError("std must be greater than zero")

        scores = torch.as_tensor(
            self.dataset[name],
            dtype=torch.float32,
            device="cpu",
        )

        normalized_scores = (scores - mean) / std
        self.dataset[name] = normalized_scores.tolist()

