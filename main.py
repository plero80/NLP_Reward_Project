from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from Datasets.dataset_classifier import DatasetClassifier
from Models.models import GenerateModel, ScoreModel
from Trainers.trainer_classifier import ClassifierTrainer
from Trainers.trainer_ppo import PolicyPPOTrainer


@dataclass(frozen=True)
class BatchResult:
    prompts: list[str]
    answers: list[str]
    reward_scores: list[float]
    judge_scores: list[float]


def process_prompt_batch(
    prompts: Sequence[str],
    policy_model: GenerateModel,
    reward_model: ScoreModel,
    judge_model: ScoreModel,
    dataset: DatasetClassifier,
) -> BatchResult:
    prompt_batch = list(prompts)
    if not prompt_batch:
        return BatchResult([], [], [], [])

    answers = policy_model.generate_batch(prompt_batch)
    if len(answers) != len(prompt_batch):
        raise ValueError(
            "The policy model must return one answer for every prompt"
        )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="batch-scoring",
    ) as executor:
        reward_future = executor.submit(
            reward_model.score,
            prompt_batch,
            answers,
        )
        judge_future = executor.submit(
            judge_model.score,
            prompt_batch,
            answers,
        )

        reward_scores = reward_future.result()
        judge_scores = judge_future.result()

    dataset.add(
        prompts=prompt_batch,
        answers=answers,
        reward_scores=reward_scores,
        judge_scores=judge_scores,
    )

    return BatchResult(
        prompts=prompt_batch,
        answers=list(answers),
        reward_scores=[float(score) for score in reward_scores],
        judge_scores=[float(score) for score in judge_scores],
    )


@dataclass
class TrainingPipeline:
    policy_model: GenerateModel
    reward_model: ScoreModel
    judge_model: ScoreModel
    ppo_trainer: PolicyPPOTrainer
    classifier_dataset: DatasetClassifier
    classifier_trainers: dict[str, ClassifierTrainer] = field(
        default_factory=dict
    )

    def run_batch(self, prompts: Sequence[str]) -> BatchResult:
        result = process_prompt_batch(
            prompts=prompts,
            policy_model=self.policy_model,
            reward_model=self.reward_model,
            judge_model=self.judge_model,
            dataset=self.classifier_dataset,
        )

        self.ppo_trainer.train_step(
            prompts=result.prompts,
            answers=result.answers,
            rewards=result.reward_scores,
        )
        return result

    def run(self, prompt_batches: Sequence[Sequence[str]]) -> None:
        for prompts in prompt_batches:
            self.run_batch(prompts)

    def train_classifiers(
        self,
        eval_datasets: dict[str, DatasetClassifier] | None = None,
    ) -> None:
        eval_datasets = eval_datasets or {}

        for name, trainer in self.classifier_trainers.items():
            trainer.train(
                train_dataset=self.classifier_dataset,
                eval_dataset=eval_datasets.get(name),
            )
