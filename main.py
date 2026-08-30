import logging

from datasets import Dataset, load_dataset

from config import RunnerConfig
from runner import Runner


logger = logging.getLogger(__name__)


def _extract_prompt(chosen_conversation: str) -> str:
    marker = "\n\nAssistant:"
    if marker not in chosen_conversation:
        raise ValueError("HH-RLHF conversation has no final Assistant turn")
    conversation, _ = chosen_conversation.rsplit(marker, maxsplit=1)
    return f"{conversation}{marker}"


def load_prompt_dataset(config: RunnerConfig) -> Dataset:
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    required_rows = config.dataset_limit + config.static_dataset_length
    if len(dataset) < required_rows:
        raise ValueError(
            f"Dataset contains {len(dataset)} rows, but {required_rows} are required"
        )
    dataset = dataset.select(range(required_rows))
    return dataset.map(
        lambda row: {"prompt": _extract_prompt(row["chosen"])},
        remove_columns=dataset.column_names,
        desc="Extracting HH-RLHF prompts",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = RunnerConfig()
    dataset = load_prompt_dataset(config)
    result = Runner(dataset, config).run()
    logger.info("Final classifier metrics: %s", result.classifier_metrics)


if __name__ == "__main__":
    main()
