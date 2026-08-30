from collections.abc import Sequence
from typing import Any, Protocol, cast

import torch
from transformers import PreTrainedTokenizerBase


class PPOStepEngine(Protocol):
    def step(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
        rewards: list[torch.Tensor],
    ) -> Any:
        ...


class PolicyPPOTrainer:
    def __init__(
        self,
        engine: PPOStepEngine,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device | str,
    ) -> None:
        if not callable(getattr(engine, "step", None)):
            raise TypeError(
                "The PPO engine must provide step(queries, responses, rewards)"
            )

        self.engine = engine
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = cast(torch.Tensor, encoded["input_ids"])
        return input_ids.squeeze(0).to(self.device)

    def train_step(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        rewards: Sequence[float],
    ) -> Any:
        lengths = {len(prompts), len(answers), len(rewards)}
        if len(lengths) != 1:
            raise ValueError(
                "prompts, answers, and rewards must have the same length"
            )

        if len(prompts) == 0:
            return None

        query_tensors = [
            self._tokenize(prompt)
            for prompt in prompts
        ]
        response_tensors = [
            self._tokenize(answer)
            for answer in answers
        ]
        reward_tensors = [
            torch.tensor(float(reward), device=self.device)
            for reward in rewards
        ]

        return self.engine.step(
            query_tensors,
            response_tensors,
            reward_tensors,
        )
