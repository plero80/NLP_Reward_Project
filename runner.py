from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
from typing import Any, cast

from datasets import Dataset
import torch

from Datasets.dataset_classifier import DatasetClassifier
from Models.model_classifier import Classifier
from Models.model_evaluator import EvaluatorModel
from Models.model_policy import PolicyModel
from Models.model_reward import RewardModel
from Models.runtime import current_device
from Models.model_value import ValueModel
from Trainers.trainer_classifier import (
    ClassifierTrainer,
    ClassifierTrainingConfig,
)
from Trainers.trainer_ppo import PolicyPPOTrainer, PPOTrainingConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyEvaluation:
    stage: str
    dataset: Dataset
    total: int
    appropriate_count: int
    appropriate_rate: float
    proxy_reward_mean: float
    proxy_reward_std: float
    judge_reward_mean: float
    judge_reward_std: float
    reward_gap_mean: float


@dataclass(frozen=True)
class RunnerResult:
    baseline: PolicyEvaluation
    post_ppo: PolicyEvaluation
    final_evaluation_dataset: Dataset
    final_proxy_reward_mean: float
    classifier_dataset: DatasetClassifier
    classifier_train_dataset: DatasetClassifier
    classifier_test_dataset: DatasetClassifier
    classifier_metrics: dict[str, Any]
    classifier: Classifier
    evaluation_report_path: str

    @property
    def evaluation_dataset(self) -> Dataset:
        """Backward-compatible access to the post-PPO evaluation dataset."""
        return self.post_ppo.dataset


class Runner:
    def __init__(self, dataset: Dataset, config: Any) -> None:
        self.config = config
        self.policy = PolicyModel(
            config.policy_name,
            getattr(config, "policy_lora_settings", None),
        )
        self.reward = RewardModel(
            config.reward_name,
            "proxy_reward",
            getattr(config, "reward_model_load", False),
            getattr(config, "reward_model_checkpoint", None),
        )
        for classifier_name, checkpoint in getattr(
            config,
            "initial_classifier_checkpoints",
            (),
        ):
            self.reward.load_classifier(classifier_name, checkpoint)
        self.judge = RewardModel(config.judge_name, "judge_reward")
        self.value = ValueModel(
            getattr(config, "value_name", None) or config.reward_name
        )
        self.evaluator = EvaluatorModel(
            getattr(config, "evaluator_name", "gpt-5.4-mini")
        )

        training_length = min(config.dataset_limit, len(dataset))
        static_length = config.static_dataset_length
        if training_length + static_length > len(dataset):
            raise ValueError(
                "dataset must have enough rows for disjoint PPO/classifier "
                "training and static evaluation splits"
            )

        self.dataset = dataset.select(range(training_length))
        self.static_dataset = dataset.select(
            range(training_length, training_length + static_length)
        )

    def _ppo_config(self) -> PPOTrainingConfig:
        config = self.config
        return PPOTrainingConfig(
            output_dir=getattr(
                config,
                "ppo_output_dir",
                "checkpoints/policy",
            ),
            epochs=getattr(config, "ppo_epochs", 1.0),
            batch_size=getattr(config, "ppo_batch_size", 1),
            gradient_accumulation_steps=getattr(
                config,
                "ppo_gradient_accumulation_steps",
                1,
            ),
            learning_rate=getattr(config, "ppo_learning_rate", 3e-6),
            response_length=getattr(config, "ppo_response_length", 128),
            num_ppo_epochs=getattr(config, "ppo_optimization_epochs", 4),
            num_mini_batches=getattr(config, "ppo_num_mini_batches", 1),
            save_steps=getattr(config, "ppo_save_steps", 100),
            logging_steps=getattr(config, "ppo_logging_steps", 10),
        )

    def _classifier_config(self) -> ClassifierTrainingConfig:
        config = self.config
        return ClassifierTrainingConfig(
            output_dir=getattr(
                config,
                "classifier_output_dir",
                "checkpoints/classifier",
            ),
            epochs=getattr(config, "classifier_epochs", 3.0),
            batch_size=getattr(config, "classifier_batch_size", 8),
            learning_rate=getattr(
                config,
                "classifier_learning_rate",
                2e-5,
            ),
            max_length=getattr(config, "classifier_max_length", 512),
            lora_settings=getattr(config, "classifier_lora_settings", None),
        )

    @staticmethod
    def _text_columns(dataset: Dataset) -> tuple[list[str], list[str]]:
        prompt_column = "prompt" if "prompt" in dataset.column_names else "prompts"
        prompts = list(dataset[prompt_column])
        answers = list(dataset["answers"])
        if not all(isinstance(value, str) for value in prompts + answers):
            raise TypeError("all prompts and answers must be strings")
        return cast(list[str], prompts), cast(list[str], answers)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    @staticmethod
    def _std(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    def _set_evaluation_seed(self) -> None:
        seed = getattr(self.config, "evaluation_seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _place_models_for_run(self) -> None:
        device = current_device()
        self.policy.model.to(device)
        self.reward.model.to(device)
        self.judge.model.to(device)
        self.value.model.to(device)
        for entry in self.reward.classifiers:
            entry["classifier"].model.to(device)

    def _offload_before_classifier_training(self) -> None:
        if not torch.cuda.is_available():
            return
        self.policy.model.to("cpu")
        self.judge.model.to("cpu")
        self.value.model.to("cpu")
        torch.cuda.empty_cache()

    def _evaluate_policy(self, stage: str) -> PolicyEvaluation:
        self._set_evaluation_seed()
        generated = self.policy.generate_new_dataset(
            self.static_dataset,
            batch_size=getattr(self.config, "generation_batch_size", 8),
        )
        evaluated = self.evaluator.evaluate(
            generated,
            concurrency=getattr(self.config, "evaluator_concurrency", 5),
        )
        prompts, answers = self._text_columns(evaluated)
        proxy_rewards = [float(value) for value in self.reward.score(prompts, answers)]
        judge_rewards = [float(value) for value in self.judge.score(prompts, answers)]
        reward_gaps = [
            proxy - judge
            for proxy, judge in zip(proxy_rewards, judge_rewards)
        ]
        evaluated = evaluated.add_column("proxy_reward", proxy_rewards)
        evaluated = evaluated.add_column("judge_reward", judge_rewards)
        evaluated = evaluated.add_column("reward_gap", reward_gaps)

        appropriate_count = sum(int(value) for value in evaluated["appropriate"])
        total = len(evaluated)
        result = PolicyEvaluation(
            stage=stage,
            dataset=evaluated,
            total=total,
            appropriate_count=appropriate_count,
            appropriate_rate=appropriate_count / total if total else 0.0,
            proxy_reward_mean=self._mean(proxy_rewards),
            proxy_reward_std=self._std(proxy_rewards),
            judge_reward_mean=self._mean(judge_rewards),
            judge_reward_std=self._std(judge_rewards),
            reward_gap_mean=self._mean(reward_gaps),
        )
        logger.info(
            "%s evaluation: appropriate=%d/%d (%.2f%%), proxy=%.4f±%.4f, "
            "judge=%.4f±%.4f, gap=%.4f",
            stage,
            result.appropriate_count,
            result.total,
            result.appropriate_rate * 100.0,
            result.proxy_reward_mean,
            result.proxy_reward_std,
            result.judge_reward_mean,
            result.judge_reward_std,
            result.reward_gap_mean,
        )
        return result

    @staticmethod
    def _policy_summary(evaluation: PolicyEvaluation) -> dict[str, Any]:
        return {
            "stage": evaluation.stage,
            "total": evaluation.total,
            "appropriate_count": evaluation.appropriate_count,
            "appropriate_rate": evaluation.appropriate_rate,
            "proxy_reward_mean": evaluation.proxy_reward_mean,
            "proxy_reward_std": evaluation.proxy_reward_std,
            "judge_reward_mean": evaluation.judge_reward_mean,
            "judge_reward_std": evaluation.judge_reward_std,
            "reward_gap_mean": evaluation.reward_gap_mean,
        }

    def _save_evaluation_artifacts(
        self,
        baseline: PolicyEvaluation,
        post_ppo: PolicyEvaluation,
        final_evaluation_dataset: Dataset,
        final_proxy_reward_mean: float,
        classifier_dataset: DatasetClassifier,
        classifier_train_dataset: DatasetClassifier,
        classifier_test_dataset: DatasetClassifier,
        classifier_metrics: dict[str, Any],
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = Path(
            getattr(
                self.config,
                "evaluation_output_dir",
                "checkpoints/evaluations",
            )
        ) / timestamp
        root.mkdir(parents=True, exist_ok=False)

        baseline.dataset.to_json(root / "baseline.jsonl")
        post_ppo.dataset.to_json(root / "post_ppo.jsonl")
        final_evaluation_dataset.to_json(root / "final.jsonl")
        Dataset.from_list(classifier_dataset.dataset).to_json(
            root / "classifier_all.jsonl"
        )
        Dataset.from_list(classifier_train_dataset.dataset).to_json(
            root / "classifier_train.jsonl"
        )
        Dataset.from_list(classifier_test_dataset.dataset).to_json(
            root / "classifier_test.jsonl"
        )

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline": self._policy_summary(baseline),
            "post_ppo": self._policy_summary(post_ppo),
            "policy_change": {
                "appropriate_percentage_points": (
                    post_ppo.appropriate_rate - baseline.appropriate_rate
                )
                * 100.0,
                "judge_reward_mean": (
                    post_ppo.judge_reward_mean - baseline.judge_reward_mean
                ),
            },
            "classifier": {
                "all_rows": len(classifier_dataset),
                "train_rows": len(classifier_train_dataset),
                "test_rows": len(classifier_test_dataset),
                "class_counts": classifier_dataset.class_counts(),
                "metrics": classifier_metrics,
            },
            "final": {
                "reward_classifier_count": len(self.reward.classifiers),
                "proxy_reward_mean": final_proxy_reward_mean,
                "proxy_reward_change_after_classifier": (
                    final_proxy_reward_mean - post_ppo.proxy_reward_mean
                ),
            },
        }
        report_path = root / "summary.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=float),
            encoding="utf-8",
        )
        logger.info("Saved evaluation artifacts to %s", root)
        return str(report_path)

    def run(self) -> RunnerResult:
        self._place_models_for_run()
        classifier_names = [entry["name"] for entry in self.reward.classifiers]
        logger.info(
            "Starting run with %d reward classifier(s): %s",
            len(classifier_names),
            classifier_names or "none",
        )

        # Establish a policy baseline before any PPO updates.
        baseline = self._evaluate_policy("baseline")

        # PPO owns rollouts, optimization, and policy checkpointing.
        ppo_trainer = PolicyPPOTrainer(
            policy=self.policy,
            reward=self.reward,
            value=self.value,
            train_dataset=self.dataset,
            eval_dataset=self.static_dataset,
            config=self._ppo_config(),
        )
        ppo_trainer.train()

        # Evaluate the updated policy on exactly the same static prompts.
        post_ppo = self._evaluate_policy("post_ppo")
        logger.info(
            "Policy change: appropriate=%+.2f percentage points, judge=%+.4f",
            (post_ppo.appropriate_rate - baseline.appropriate_rate) * 100.0,
            post_ppo.judge_reward_mean - baseline.judge_reward_mean,
        )

        # Build labels from the gap between proxy and independent judge scores.
        generated_training_dataset = self.policy.generate_new_dataset(
            self.dataset,
            batch_size=getattr(self.config, "generation_batch_size", 8),
        )
        prompts, answers = self._text_columns(generated_training_dataset)
        reward_scores = self.reward.score(prompts, answers)
        judge_scores = self.judge.score(prompts, answers)
        classifier_dataset = DatasetClassifier(
            theta=getattr(self.config, "classifier_theta", 2.0)
        )
        classifier_dataset.add(
            prompts=prompts,
            answers=answers,
            reward_scores=reward_scores,
            judge_scores=judge_scores,
        )
        class_counts = classifier_dataset.class_counts()
        if len(class_counts) < 2:
            raise ValueError(
                "Classifier labels contain only one class. Adjust "
                "classifier_theta or collect more examples before training."
            )
        classifier_train_dataset, classifier_test_dataset = (
            classifier_dataset.split(
                test_size=getattr(self.config, "classifier_test_size", 0.2),
                random_state=getattr(self.config, "evaluation_seed", 42),
            )
        )
        logger.info(
            "Classifier data: total=%d classes=%s train=%d test=%d",
            len(classifier_dataset),
            class_counts,
            len(classifier_train_dataset),
            len(classifier_test_dataset),
        )
        self._offload_before_classifier_training()

        # Train with validation and measure generalization on held-out rows.
        classifier_name = getattr(
            self.config,
            "classifier_name",
            "reward_gap_classifier",
        )
        classifier = Classifier(
            name=classifier_name,
            model_name=(
                getattr(self.config, "classifier_model_name", None)
                or self.config.reward_name
            ),
        )
        classifier_trainer = ClassifierTrainer(
            classifier,
            self._classifier_config(),
        )
        trained_classifier = classifier_trainer.train(
            classifier_train_dataset,
            classifier_test_dataset,
        )
        classifier_metrics = classifier_trainer.evaluate(
            trained_classifier,
            classifier_test_dataset,
        )
        self.reward.add_classifier(
            classifier_name,
            classifier,
            classifier.tokenizer,
        )

        # Re-score the post-PPO answers to measure the classifier's effect on
        # the composite proxy reward. The policy itself is unchanged here.
        final_prompts, final_answers = self._text_columns(post_ppo.dataset)
        final_proxy_rewards = [
            float(value)
            for value in self.reward.score(final_prompts, final_answers)
        ]
        final_evaluation_dataset = post_ppo.dataset.add_column(
            "proxy_reward_after_classifier",
            final_proxy_rewards,
        )
        final_proxy_reward_mean = self._mean(final_proxy_rewards)
        logger.info(
            "Run complete with %d reward classifier(s). Held-out F1=%.4f, "
            "ROC-AUC=%s, final proxy reward=%.4f (change=%+.4f)",
            len(self.reward.classifiers),
            float(classifier_metrics.get("test_f1", 0.0)),
            classifier_metrics.get("test_roc_auc", "not available"),
            final_proxy_reward_mean,
            final_proxy_reward_mean - post_ppo.proxy_reward_mean,
        )
        evaluation_report_path = self._save_evaluation_artifacts(
            baseline=baseline,
            post_ppo=post_ppo,
            final_evaluation_dataset=final_evaluation_dataset,
            final_proxy_reward_mean=final_proxy_reward_mean,
            classifier_dataset=classifier_dataset,
            classifier_train_dataset=classifier_train_dataset,
            classifier_test_dataset=classifier_test_dataset,
            classifier_metrics=classifier_metrics,
        )

        return RunnerResult(
            baseline=baseline,
            post_ppo=post_ppo,
            final_evaluation_dataset=final_evaluation_dataset,
            final_proxy_reward_mean=final_proxy_reward_mean,
            classifier_dataset=classifier_dataset,
            classifier_train_dataset=classifier_train_dataset,
            classifier_test_dataset=classifier_test_dataset,
            classifier_metrics=classifier_metrics,
            classifier=classifier,
            evaluation_report_path=evaluation_report_path,
        )
