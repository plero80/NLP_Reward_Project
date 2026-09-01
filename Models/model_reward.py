from Models.models import ScoreModel, BinaryClassifier
from Models.model_classifier import Classifier
from Models.model_policy import PolicyModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import math
import torch

from concurrent.futures import ThreadPoolExecutor



from collections.abc import Sequence
import logging
from types import SimpleNamespace
from typing import Any

from Models.runtime import best_dtype, current_device



logger = logging.getLogger(__name__)


class _CompositeRewardBackbone(torch.nn.Module):
    def __init__(
        self,
        reward_model: torch.nn.Module,
        classifiers: list[torch.nn.Module],
        state: SimpleNamespace,
    ) -> None:
        super().__init__()
        self.reward_model = reward_model
        self.classifiers = torch.nn.ModuleList(classifiers)
        self.state = state

    def forward(self, **kwargs: Any) -> Any:
        reward_backbone = getattr(
            self.reward_model,
            self.reward_model.base_model_prefix,
        )
        output = reward_backbone(**kwargs)

        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        penalties = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=output.hidden_states[-1].dtype,
        )
        with torch.no_grad():
            for classifier in self.classifiers:
                logits = classifier(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
                if logits.shape[-1] == 2:
                    probability = torch.softmax(logits.float(), dim=-1)[:, 1]
                elif logits.shape[-1] == 1:
                    probability = torch.sigmoid(logits.float().squeeze(-1))
                else:
                    raise ValueError(
                        "Reward classifiers must output one or two logits"
                    )
                probability = probability.clamp(0.0, 1.0 - 1e-12)
                penalties += torch.log1p(-probability).to(penalties.dtype)

        self.state.penalties = penalties
        return output







class CompositeRewardModel(torch.nn.Module):
    """Expose classifier-adjusted rewards through TRL PPO's model contract."""

    base_model_prefix = "backbone"

    def __init__(
        self,
        reward_model: torch.nn.Module,
        classifiers: list[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.config = reward_model.config
        self._state = SimpleNamespace(penalties=None)
        self.backbone = _CompositeRewardBackbone(
            reward_model,
            classifiers,
            self._state,
        )

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reward_logits = self.backbone.reward_model.score(hidden_states)
        penalties = self._state.penalties
        if penalties is None:
            raise RuntimeError("Composite reward backbone must run before score")
        return reward_logits + penalties[:, None, None]


class RewardModel(ScoreModel):
    
    def __init__(
        self,
        model_name: str,
        model_mode: str,
        load: bool = False,
        checkpoint: str | None = None,
    ) -> None:
        if load and not checkpoint:
            raise ValueError("checkpoint is required when load=True")
        model_source = checkpoint if load and checkpoint is not None else model_name
        self.base_model_prefix = model_name
        self.model_mode = model_mode
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            dtype=best_dtype(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(current_device())
        self.model.eval()
        self.classifiers: list[dict] = []
        self.std = None
        self.mean = None
        
        
    def init_normalization(self, prompts: Sequence[str], answers: Sequence[str]) -> None:
        
        scores = self.score(prompts, answers)
        scores = torch.as_tensor(scores, dtype=torch.float32)

        self.mean = scores.mean()
        self.std = scores.std(correction=0)
        
        
        
        
    def add_classifier(
        self,
        name: str,
        classifier: BinaryClassifier,
        tokenizer=None,
    ) -> None:
        logger.info("%s: Added the classifier: %s", self.model_mode, name)
        classifier_model = getattr(classifier, "model", None)
        if classifier_model is not None:
            classifier_model.eval()
        self.classifiers.append({
                                "name" : name,
                                "classifier" : classifier,
                                "tokenizer" : tokenizer
                            })
        
        
        
    def load_classifier(self, name: str, path: str) -> None:
        classifier = Classifier(name=name, model_name=path)
        classifier.model.to(self.model.device)
        classifier.model.eval()

        self.add_classifier(name, classifier, classifier.tokenizer)

        logger.info(
            "%s: Loaded classifier '%s' from %s",
            self.model_mode,
            name,
            path,
        )

    def for_ppo(self) -> torch.nn.Module:
        """Return the base or classifier-adjusted model expected by TRL PPO."""
        if not self.classifiers:
            self.model.eval()
            return self.model
        classifier_models = [
            entry["classifier"].model
            for entry in self.classifiers
        ]
        composite_model = CompositeRewardModel(self.model, classifier_models)
        composite_model.eval()
        return composite_model
        
        

    def _score_reward_model(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:
        logger.info("%s: Starting to calculate the score", self.model_mode)
        
        conversations = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            for prompt, answer in zip(prompts, answers)
        ]

        formatted_texts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
            )
            for conversation in conversations
        ]

        # Remove a duplicated BOS token when present.
        if self.tokenizer.bos_token is not None:
            formatted_texts = [
                text[len(self.tokenizer.bos_token):]
                if text.startswith(self.tokenizer.bos_token)
                else text
                for text in formatted_texts
            ]

        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16_384,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        
        scores = outputs.logits.squeeze(-1).float().cpu().tolist()
        logger.debug("%s: Reward scores: %s", self.model_mode, scores)

        return scores

    def score(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
    ) -> list[float]:
        if len(prompts) != len(answers):
            logger.error(
                "%s: prompts and answers must have the same length",
                self.model_mode,
            )
            raise ValueError("prompts and answers must have the same length")

        if len(prompts) == 0:
            return []

        prompt_batch = list(prompts)
        answer_batch = list(answers)

        with ThreadPoolExecutor(
            max_workers=len(self.classifiers) + 1,
            thread_name_prefix="reward-inference",
        ) as executor:
            reward_future = executor.submit(
                self._score_reward_model,
                prompt_batch,
                answer_batch,
            )
            classifier_futures = [
                (
                    entry["name"],
                    executor.submit(
                        entry["classifier"].predict_proba,
                        prompt_batch,
                        answer_batch,
                    ),
                )
                for entry in self.classifiers
            ]

            scores = reward_future.result()
            classifier_probabilities = [
                (name, future.result())
                for name, future in classifier_futures
            ]

        combined_scores = [float(score) for score in scores]
        epsilon = 1e-12

        # Penalize high undesirable-class probabilities across the batch.
        for classifier_name, probabilities in classifier_probabilities:
            if len(probabilities) != len(combined_scores):
                raise ValueError(
                    f"Classifier '{classifier_name}' returned "
                    f"{len(probabilities)} probabilities for "
                    f"{len(combined_scores)} inputs"
                )

            for index, probability in enumerate(probabilities):
                probability = float(probability)
                if (
                    not math.isfinite(probability)
                    or probability < 0.0
                    or probability > 1.0
                ):
                    raise ValueError(
                        f"Classifier '{classifier_name}' returned invalid "
                        f"probability {probability} at index {index}"
                    )

                safe_probability = min(probability, 1.0 - epsilon)
                combined_scores[index] += math.log1p(-safe_probability)

        logger.debug(
            "%s: Combined reward scores: %s",
            self.model_mode,
            combined_scores,
        )
        
        if self.std is not None and self.mean is not None:
            normalized_scores = (
                torch.as_tensor(combined_scores, dtype=torch.float32)
                - self.mean.cpu()
            ) / (self.std.cpu() + 1e-8)
            combined_scores = normalized_scores.tolist()
            
            
        return combined_scores



    def score_policy(self, policy:PolicyModel) -> None:
        
        prompts = policy.get_dataset_col("prompts")
        answers = policy.get_dataset_col("answers")
        
        scores = self.score(prompts, answers)
        
        policy.add_scores(scores, self.model_mode)
        
        assert self.model_mode in policy.dataset
        assert len(policy.get_dataset_col(self.model_mode)) > 0