import asyncio
import os
import random

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal

class EvaluatorOutput(BaseModel):
    appropriate: Literal[0, 1]


class EvaluatorModel:
    def __init__(self, model_name: str, safety_identifier: str):
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in .env or the environment")

        self.model_name = model_name
        self.safety_identifier = str(safety_identifier)

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
            safety_identifier=self.safety_identifier,
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

    async def __call__(self, *args, **kwargs):
        return await self.response(*args, **kwargs)
