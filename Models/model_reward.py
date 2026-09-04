from Models.models import ScoreModel, BinaryClassifier
from Models.model_classifier import Classifier
from Models.model_policy import PolicyModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import math
import torch

from collections.abc import Sequence
import logging
from types import SimpleNamespace
from typing import Any

from Models.runtime import best_dtype, current_device
from Models.reward_adjustment import (
    ClassifierProbabilityPenalty,
    ProbabilityPenaltyHead,
    RewardAdjustment,
)
from collections.abc import Callable


logger = logging.getLogger(__name__)


class _CompositeRewardBackbone(torch.nn.Module):
    def __init__(
        self,
        reward_model: torch.nn.Module,
        adjustment_heads: list[torch.nn.Module],
        state: SimpleNamespace,
    ) -> None:
        super().__init__()
        self.reward_model = reward_model
        self.adjustment_heads = torch.nn.ModuleList(adjustment_heads)
        self.state = state

    def forward(self, **kwargs: Any) -> Any:
        reward_backbone = getattr(
            self.reward_model,
            self.reward_model.base_model_prefix,
        )
        output = reward_backbone(**kwargs)
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        adjustments = torch.zeros(
            input_ids.shape[0],
            device=input_ids.device,
            dtype=output.hidden_states[-1].dtype,
        )
        with torch.no_grad():
            for head in self.adjustment_heads:
                values = head(input_ids, attention_mask)
                if values.shape != adjustments.shape:
                    raise ValueError(
                        "Reward adjustment returned shape "
                        f"{tuple(values.shape)}; expected {tuple(adjustments.shape)}"
                    )
                adjustments += values.to(adjustments.dtype)
        self.state.adjustments = adjustments
        return output







class CompositeRewardModel(torch.nn.Module):
    """Apply generic adjustment heads through TRL's reward-model contract."""

    base_model_prefix = "backbone"

    def __init__(
        self,
        reward_model: torch.nn.Module,
        adjustment_heads: list[torch.nn.Module],
        mean: float | None = None,
        std: float | None = None,
    ) -> None:
        super().__init__()
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together")
        if mean is not None and std is not None:
            mean = float(mean)
            std = float(std)
            if not math.isfinite(mean) or not math.isfinite(std):
                raise ValueError("Reward normalization values must be finite")
            if std <= 1e-8:
                raise ValueError("std must be greater than zero")
        self.config = reward_model.config
        self.mean = mean
        self.std = std
        self._state = SimpleNamespace(adjustments=None)
        heads = [
            head
            if hasattr(head, "classifier_model") or hasattr(head, "gap_model")
            else ProbabilityPenaltyHead(head)
            for head in adjustment_heads
        ]
        self.backbone = _CompositeRewardBackbone(
            reward_model,
            heads,
            self._state,
        )

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reward_logits = self.backbone.reward_model.score(hidden_states)
        adjustments = self._state.adjustments
        if adjustments is None:
            raise RuntimeError("Composite reward backbone must run before score")
        reward_logits = reward_logits + adjustments[:, None, None]
        if self.mean is not None and self.std is not None:
            reward_logits = (reward_logits.float() - self.mean) / self.std
        return reward_logits


class RewardModel(ScoreModel):
    
    def __init__(
        self,
        model_name: str,
        model_mode: str,
        load: bool = False,
        checkpoint: str | None = None,
        ref_mean: float | None = None,
        ref_std: float | None = None,
    ) -> None:
        if load and not checkpoint:
            raise ValueError("checkpoint is required when load=True")
        if (ref_mean is None) != (ref_std is None):
            raise ValueError("ref_mean and ref_std must be provided together")
        if ref_mean is not None and ref_std is not None:
            ref_mean = float(ref_mean)
            ref_std = float(ref_std)
            if not math.isfinite(ref_mean) or not math.isfinite(ref_std):
                raise ValueError("Reference normalization values must be finite")
            if ref_std <= 1e-8:
                raise ValueError("ref_std must be greater than zero")
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
        # Chat templates place the assistant response near the end. Preserve
        # that response if an unusually long conversation must be truncated.
        self.tokenizer.truncation_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(current_device())
        self.model.eval()
        self.classifiers: list[dict] = []
        self.adjustments: list[RewardAdjustment] = []
        self.mean = ref_mean
        self.std = ref_std
        
        
    def init_normalization(
        self,
        policy: PolicyModel,
        batch_size: int = 1,
        max_length: int = 2_048,
    ) -> None:
        
        prompts = policy.get_dataset_col("prompts")
        answers = policy.get_dataset_col("answers")
        
        scores = self.score(
            prompts,
            answers,
            batch_size=batch_size,
            max_length=max_length,
        )
        scores = torch.as_tensor(scores, dtype=torch.float32)

        mean = float(scores.mean().item())
        std = float(scores.std(correction=0).item())
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("Normalization statistics must be finite")
        if std <= 1e-8:
            raise ValueError(
                "Cannot normalize reward scores with near-zero variance"
            )
        self.mean = mean
        self.std = std
        
        
        
        
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
        self.add_adjustment(ClassifierProbabilityPenalty(classifier))

    def add_adjustment(self, adjustment: RewardAdjustment) -> None:
        """Attach a callable additive policy without knowing its internals."""
        model = getattr(adjustment, "model", None)
        if isinstance(model, torch.nn.Module):
            model.eval()
        self.adjustments.append(adjustment)

    def _active_adjustments(self) -> list[RewardAdjustment]:
        if hasattr(self, "adjustments"):
            return self.adjustments
        # Compatibility for previously constructed/pickled RewardModel objects.
        return [
            ClassifierProbabilityPenalty(entry["classifier"])
            for entry in getattr(self, "classifiers", ())
        ]
        
        
        
    def load_classifier(self, name: str, path: str) -> None:
        classifier = Classifier.load(path)
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
        adjustments = self._active_adjustments()
        if not adjustments and self.mean is None:
            self.model.eval()
            return self.model
        adjustment_heads = [entry.ppo_head() for entry in adjustments]
        composite_model = CompositeRewardModel(
            self.model,
            adjustment_heads,
            mean=self.mean,
            std=self.std,
        )
        composite_model.eval()
        return composite_model
        
        

    def _score_reward_model(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        max_length: int,
    ) -> list[float]:
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
            max_length=max_length,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        
        score_tensor = outputs.logits.reshape(-1)
        if score_tensor.numel() != len(prompts):
            raise ValueError(
                f"{self.model_mode} returned {score_tensor.numel()} scores "
                f"for {len(prompts)} inputs"
            )
        scores = score_tensor.float().cpu().tolist()
        logger.debug("%s: Reward scores: %s", self.model_mode, scores)

        return scores

    def score(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        batch_size: int = 1,
        max_length: int = 2_048,
        normalize_score: bool = False,
    ) -> list[float]:
        if len(prompts) != len(answers):
            logger.error(
                "%s: prompts and answers must have the same length",
                self.model_mode,
            )
            raise ValueError("prompts and answers must have the same length")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if len(prompts) == 0:
            return []

        logger.info(
            "%s: Calculating %d scores in batches of %d",
            self.model_mode,
            len(prompts),
            batch_size,
        )

        prompt_batch = list(prompts)
        answer_batch = list(answers)
        combined_scores: list[float] = []
        # Keep GPU activation memory bounded by adjusting each micro-batch.
        for start in range(0, len(prompt_batch), batch_size):
            batch_prompts = prompt_batch[start : start + batch_size]
            batch_answers = answer_batch[start : start + batch_size]
            reward_scores = self._score_reward_model(
                batch_prompts,
                batch_answers,
                max_length=max_length,
            )
            batch_scores = [float(score) for score in reward_scores]

            for adjustment in self._active_adjustments():
                values = adjustment(batch_prompts, batch_answers)
                if len(values) != len(batch_scores):
                    raise ValueError(
                        f"Reward adjustment '{adjustment.id}' returned "
                        f"{len(values)} values for "
                        f"{len(batch_scores)} inputs"
                    )
                for index, value in enumerate(values):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Reward adjustment '{adjustment.id}' returned "
                            f"non-finite value at index "
                            f"{start + index}"
                        )
                    batch_scores[index] += value

            combined_scores.extend(batch_scores)

        logger.debug(
            "%s: Combined reward scores: %s",
            self.model_mode,
            combined_scores,
        )
        
        if normalize_score:
            if self.std is None or self.mean is None:
                raise RuntimeError(
                    "Normalization requested before mean/std were configured"
                )
            normalized_scores = (
                torch.as_tensor(combined_scores, dtype=torch.float32)
                - float(self.mean)
            ) / (float(self.std) + 1e-8)
            combined_scores = normalized_scores.tolist()
            
            
        return combined_scores



    def score_policy(
        self,
        policy: PolicyModel,
        batch_size: int = 1,
        max_length: int = 2_048,
        normalize_score: bool = False,
    ) -> None:
        if policy.dataset is None:
            raise ValueError("The policy doesn't contain a dataset")
        column_names = (
            policy.dataset.column_names
            if hasattr(policy.dataset, "column_names")
            else list(policy.dataset.columns)
        )
        prompt_column = (
            "prompts"
            if "prompts" in column_names
            else "prompt"
            if "prompt" in column_names
            else None
        )
        if prompt_column is None:
            raise ValueError("The policy dataset must contain prompt or prompts")
        
        prompts = policy.get_dataset_col(prompt_column)
        answers = policy.get_dataset_col("answers")
        
        scores = self.score(
            prompts,
            answers,
            batch_size=batch_size,
            max_length=max_length,
            normalize_score=normalize_score,
        )
        
        policy.add_scores(scores, self.model_mode)
        
        if hasattr(policy.dataset, "column_names"):
            assert self.model_mode in policy.dataset.column_names
        else:
            assert self.model_mode in policy.dataset.columns
        assert len(policy.get_dataset_col(self.model_mode)) > 0







from types import SimpleNamespace

from transformers import PreTrainedTokenizerBase, PretrainedConfig


class DeterministicBackbone(torch.nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        embedding: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.embedding.requires_grad_(False)
        good_input_ids = tokenizer(
            "good",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        bad_input_ids = tokenizer(
            "bad",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        if good_input_ids.numel() == 0 or bad_input_ids.numel() == 0:
            raise ValueError("Reward anchor words must produce at least one token")
        self.register_buffer("good_input_ids", good_input_ids, persistent=False)
        self.register_buffer("bad_input_ids", bad_input_ids, persistent=False)
        self.last_rewards: torch.Tensor | None = None

    def _mean_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        if attention_mask is None:
            return token_embeddings.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        token_sum = (token_embeddings * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1.0)
        return token_sum / token_count

    def calculate_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            text_embeddings = self._mean_embedding(input_ids, attention_mask)
            good_embedding = self._mean_embedding(self.good_input_ids)
            bad_embedding = self._mean_embedding(self.bad_input_ids)
            good_similarity = torch.nn.functional.cosine_similarity(
                text_embeddings.float(),
                good_embedding.float(),
                dim=-1,
            )
            bad_similarity = torch.nn.functional.cosine_similarity(
                text_embeddings.float(),
                bad_embedding.float(),
                dim=-1,
            )
            return torch.where(
                good_similarity > bad_similarity,
                torch.ones_like(good_similarity),
                -torch.ones_like(bad_similarity),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        attention_mask = kwargs.get("attention_mask")
        self.last_rewards = self.calculate_rewards(
            input_ids,
            attention_mask,
        )

        # TRL expects hidden states shaped [batch, sequence, hidden].
        dummy_hidden_states = torch.zeros(
            (*input_ids.shape, 1),
            dtype=torch.float32,
            device=input_ids.device,
        )

        return SimpleNamespace(
            hidden_states=(dummy_hidden_states,),
        )


class DeterministicPPORewardModule(torch.nn.Module):
    base_model_prefix = "backbone"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        embedding: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.config = PretrainedConfig(num_labels=1)
        self.backbone = DeterministicBackbone(
            tokenizer,
            embedding,
        )

    def score(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        rewards = self.backbone.last_rewards

        if rewards is None:
            raise RuntimeError("Backbone must run before score()")

        # TRL expects [batch, sequence_length, 1].
        return rewards[:, None, None].expand(
            -1,
            hidden_states.shape[1],
            1,
        )
      
        
class DeterministicReward:
    def __init__(
        self,
        model_name: str,
        ref_mean: float | None = None,
        ref_std: float | None = None,
    ) -> None:
        if (ref_mean is None) != (ref_std is None):
            raise ValueError("ref_mean and ref_std must be provided together")
        if ref_mean is not None and ref_std is not None:
            ref_mean = float(ref_mean)
            ref_std = float(ref_std)
            if not math.isfinite(ref_mean) or not math.isfinite(ref_std):
                raise ValueError("Reference normalization values must be finite")
            if ref_std <= 1e-8:
                raise ValueError("ref_std must be greater than zero")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=best_dtype(),
        )
        embedding = language_model.get_input_embeddings()
        self.classifiers: list[dict] = []
        self.adjustments: list[RewardAdjustment] = []

        self.model = DeterministicPPORewardModule(
            self.tokenizer,
            embedding,
        )
        self.model.to(current_device())
        self.mean = ref_mean
        self.std = ref_std

    def calculate_reward(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        rewards = self.model.backbone.calculate_rewards(
            inputs["input_ids"],
            inputs.get("attention_mask"),
        )
        return float(rewards[0].item())

    def add_classifier(
        self,
        name: str,
        classifier: BinaryClassifier,
        tokenizer=None,
    ) -> None:
        classifier_model = getattr(classifier, "model", None)
        if classifier_model is not None:
            classifier_model.eval()
        self.classifiers.append(
            {
                "name": name,
                "classifier": classifier,
                "tokenizer": tokenizer,
            }
        )
        self.add_adjustment(ClassifierProbabilityPenalty(classifier))

    def add_adjustment(self, adjustment: RewardAdjustment) -> None:
        model = getattr(adjustment, "model", None)
        if isinstance(model, torch.nn.Module):
            model.eval()
        self.adjustments.append(adjustment)

    def for_ppo(self) -> torch.nn.Module:
        if not self.adjustments and self.mean is None:
            self.model.eval()
            return self.model
        adjustment_heads = [entry.ppo_head() for entry in self.adjustments]
        composite_model = CompositeRewardModel(
            self.model,
            adjustment_heads,
            mean=self.mean,
            std=self.std,
        )
        composite_model.eval()
        return composite_model








        
