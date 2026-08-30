import asyncio
import logging
import os
import random

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal


logger = logging.getLogger(__name__)

class EvaluatorOutput(BaseModel):
    appropriate: Literal[0, 1]


class EvaluatorModel:
    def __init__(self, model_name: str):
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in .env or the environment")

        self.model_name = model_name

        self.system_prompt = (
            "Evaluate whether the candidate assistant response is appropriate "
            "for the given user prompt."
        )

        self.model = AsyncOpenAI()

    async def response(self, question: str, answer: str):

        evaluator_input = f"""
USER PROMPT:
{question}

CANDIDATE ASSISTANT RESPONSE:
{answer}

Determine whether the candidate assistant response is appropriate.
"""

        response = await self.model.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "developer",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": evaluator_input,
                },
            ],
            text_format=EvaluatorOutput,
        )

        result = response.output_parsed

        if result is None:
            raise ValueError("Evaluator failed to return structured output.")

        return result.appropriate

    async def _call_with_retry(self, question, answer, max_retries=8):
        delay = 1.0

        for attempt in range(max_retries):
            try:
                return await self.response(question, answer)
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise

                wait = delay + random.uniform(0, 0.5)
                print(
                    f"Evaluator rate limit hit. Retrying in {wait:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, 20)

    async def response_many(
        self,
        pairs,
        concurrency: int = 5,
        semaphore: asyncio.Semaphore | None = None,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        if semaphore is None:
            semaphore = asyncio.Semaphore(concurrency)

        async def worker(question, answer):
            async with semaphore:
                return await self._call_with_retry(
                    question,
                    answer,
                )

        return await asyncio.gather(
            *(worker(question, answer) for question, answer in pairs)
        )

    async def evaluate_async(
        self,
        dataset: Dataset,
        concurrency: int = 5,
    ) -> Dataset:
        """Return the dataset with an ``appropriate`` evaluator column."""
        prompt_column = (
            "prompt"
            if "prompt" in dataset.column_names
            else "prompts"
            if "prompts" in dataset.column_names
            else None
        )
        if prompt_column is None:
            raise ValueError("dataset must contain a 'prompt' or 'prompts' column")
        if "answers" not in dataset.column_names:
            raise ValueError("dataset must contain an 'answers' column")
        if "appropriate" in dataset.column_names:
            raise ValueError("dataset already contains an 'appropriate' column")

        pairs = zip(dataset[prompt_column], dataset["answers"])
        outcomes = await self.response_many(pairs, concurrency=concurrency)
        appropriate_count = sum(outcomes)
        total = len(outcomes)
        logger.info(
            "Evaluator marked %d/%d responses appropriate (%.1f%%)",
            appropriate_count,
            total,
            100.0 * appropriate_count / total if total else 0.0,
        )
        print(
            f"Evaluation: {appropriate_count}/{total} responses appropriate "
            f"({100.0 * appropriate_count / total if total else 0.0:.1f}%)"
        )
        return dataset.add_column("appropriate", outcomes)

    def evaluate(self, dataset: Dataset, concurrency: int = 5) -> Dataset:
        """Synchronous entry point for scripts such as ``Runner.run``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(dataset, concurrency))
        raise RuntimeError(
            "evaluate() cannot run inside an active event loop; "
            "await evaluate_async() instead"
        )

    async def __call__(self, *args, **kwargs):
        return await self.response(*args, **kwargs)
