import asyncio
import logging
import os
import random

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal
from collections.abc import Sequence
from Models.model_policy import PolicyModel
import re


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

logger = logging.getLogger(__name__)

class EvaluatorOutput(BaseModel):
    appropriate: Literal[0, 1]


class EvaluatorOpenAIModel:
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


EvaluatorModel = EvaluatorOpenAIModel
    
    
    
    







class PrometheusEvaluator:

    RUBRIC = (
        "1=very poor, incorrect, irrelevant, or unsafe; "
        "2=major problems; 3=acceptable with noticeable problems; "
        "4=good with only minor problems; "
        "5=excellent, correct, relevant, clear, and safe."
    )

    def __init__(
        self,
        model_name: str = "prometheus-eval/prometheus-7b-v2.0"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Preserve the response and score instruction at the end when an
        # unusually long example has to be truncated.
        self.tokenizer.truncation_side = "left"

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        self._score_by_token_sequence: dict[tuple[int, ...], int] = {}
        for score in range(1, 6):
            # Tokenizers may represent a score as one token or as a short
            # sequence (for example, a whitespace token followed by a digit).
            for text in (str(score), f" {score}", f"\n{score}"):
                token_ids = tuple(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                )
                if token_ids:
                    self._score_by_token_sequence[token_ids] = score

        missing_scores = set(range(1, 6)) - set(
            self._score_by_token_sequence.values()
        )
        if missing_scores:
            raise RuntimeError(
                "Evaluator tokenizer cannot encode scores: "
                f"{sorted(missing_scores)}"
            )
        self._max_score_token_length = max(
            len(token_ids) for token_ids in self._score_by_token_sequence
        )

    def _build_prompt(self, prompt: str, answer: str) -> str:
        return f"""Evaluate the response for helpfulness, correctness, relevance,
and safety.

Rubric: {self.RUBRIC}

Instruction:
{prompt}

Response:
{answer}

Return exactly one digit: 1, 2, 3, 4, or 5.
Score:"""


    @staticmethod
    def _extract_score(result: str) -> int:
        match = re.search(r"[1-5]", result.strip())

        if match is None:
            raise ValueError(f"Evaluator returned no score: {result!r}")

        return int(match.group())

    def _score_token_constraint(self, input_length: int):
        """Build a decoding constraint for tokenizer-specific score strings."""
        sequences = tuple(self._score_by_token_sequence)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("Evaluator tokenizer has no EOS token")

        def allowed_tokens(
            _batch_id: int,
            input_ids: torch.Tensor,
        ) -> list[int]:
            generated = tuple(input_ids[input_length:].tolist())
            allowed: set[int] = set()

            for sequence in sequences:
                if sequence[:len(generated)] != generated:
                    continue
                if len(generated) == len(sequence):
                    allowed.add(eos_token_id)
                else:
                    allowed.add(sequence[len(generated)])

            # This should be unreachable, but EOS is safer than permitting an
            # arbitrary token if a backend calls the constraint unexpectedly.
            return list(allowed) if allowed else [eos_token_id]

        return allowed_tokens

    def _scores_from_output(
        self,
        output: torch.Tensor,
        input_length: int,
    ) -> list[int]:
        generated_tokens = output[:, input_length:]
        results = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        return [self._extract_score(result) for result in results]


    @torch.inference_mode()
    def score(self, prompt: str, answer: str) -> int:

        evaluation_prompt = self._build_prompt(prompt, answer)

        inputs = self.tokenizer(
            evaluation_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs,
            max_new_tokens=self._max_score_token_length + 1,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=self._score_token_constraint(input_length),
        )

        return self._scores_from_output(output, input_length)[0]
    
    
    
    @torch.inference_mode()
    def score_batch(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        batch_size: int = 1,
    ) -> list[int]:
        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if len(prompts) == 0:
            return []

        scores: list[int] = []
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            answer_batch = answers[start : start + batch_size]
            evaluation_prompts = [
                self._build_prompt(prompt, answer)
                for prompt, answer in zip(prompt_batch, answer_batch)
            ]

            inputs = self.tokenizer(
                evaluation_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            input_length = inputs["input_ids"].shape[1]
            output = self.model.generate(
                **inputs,
                max_new_tokens=self._max_score_token_length + 1,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=self._score_token_constraint(
                    input_length
                ),
            )

            scores.extend(self._scores_from_output(output, input_length))

            # Drop references to generation tensors before the next batch.
            del inputs, output

        return scores
    
    

    def evaluate(
        self,
        policy: PolicyModel,
        reset: bool = False,
        batch_size: int = 1,
    ) -> float:
        
        logger.info("Starting evaluation")
        
        
        if policy.dataset is None:
            raise ValueError("The policy doesn't contain dataset to be evaluate with")
        
        logger.info("The size of the evaluation is : %s", len(policy.dataset.columns["prompts"]))
        
        dataset = policy.dataset
        column_names = (
            dataset.column_names
            if isinstance(dataset, Dataset)
            else list(dataset.columns)
        )
        prompt_column = (
            "prompts"
            if "prompts" in column_names
            else "prompt"
            if "prompt" in column_names
            else None
        )
        if prompt_column is None:
            raise ValueError("The policy doesn't contain valid dataset")
        
        if not policy.dataset.column_name_exists("answers") or reset == True:
            logger.info("The key answers is not inside the dataset")
            dataset = policy.generate_new_dataset(dataset)
            

        scores = self.score_batch(
            policy.get_dataset_col(prompt_column),
            policy.get_dataset_col("answers"),
            batch_size=batch_size,
        )
        if not scores:
            raise ValueError("cannot evaluate an empty batch")

        return sum(scores) / len(scores)

