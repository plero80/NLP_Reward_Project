from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from Datasets.dataset_classifier import DatasetClassifier
from Models.models import GenerateModel, ScoreModel


def process_prompt_batch(
    prompts: Sequence[str],
    policy_model: GenerateModel,
    reward_model: ScoreModel,
    judge_model: ScoreModel,
    dataset: DatasetClassifier,
) -> None:
    prompt_batch = list(prompts)
    if not prompt_batch:
        return

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
