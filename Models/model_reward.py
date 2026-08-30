from Models.models import ScoreModel, BinaryClassifier
from Models.model_classifier import Classifier


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import math
import torch

from concurrent.futures import ThreadPoolExecutor



from collections.abc import Sequence
import logging



logger = logging.getLogger(__name__)


class RewardModel(ScoreModel):
    
    def __init__(self, model_name, model_mode, load = False, checkpoint = None) -> None:
        self.base_model_prefix = model_name
        self.model_mode = model_mode
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifiers: list[dict] = []
        
        
        
    def add_classifier(
        self,
        name: str,
        classifier: BinaryClassifier,
        tokenizer=None,
    ) -> None:
        logger.info("%s: Added the classifier: %s", self.model_mode, name)
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
        return combined_scores
