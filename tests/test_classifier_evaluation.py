import unittest
import json
from pathlib import Path
import tempfile
from uuid import UUID

import numpy as np
from datasets import Dataset
import torch
from transformers import EvalPrediction

from Datasets.dataset_classifier import DatasetClassifier
from Trainers.trainer_classifier import compute_classifier_metrics
from main import _extract_prompt
from Models.model_reward import CompositeRewardModel
from runner import Runner


class ClassifierMetricTests(unittest.TestCase):
    def test_perfect_binary_predictions(self) -> None:
        evaluation = EvalPrediction(
            predictions=np.array(
                [
                    [4.0, -4.0],
                    [-4.0, 4.0],
                    [3.0, -3.0],
                    [-3.0, 3.0],
                ]
            ),
            label_ids=np.array([0, 1, 0, 1]),
        )

        metrics = compute_classifier_metrics(evaluation)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["false_positives"], 0.0)
        self.assertEqual(metrics["false_negatives"], 0.0)

    def test_single_logit_predictions(self) -> None:
        evaluation = EvalPrediction(
            predictions=np.array([[-5.0], [5.0]]),
            label_ids=np.array([0, 1]),
        )

        metrics = compute_classifier_metrics(evaluation)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


class ClassifierDatasetTests(unittest.TestCase):
    def test_save_load_round_trip_preserves_id_theta_and_rows(self) -> None:
        dataset = DatasetClassifier(theta=0.5)
        dataset.add(
            ["Unicode prompt: שלום\nnext line"],
            ["answer"],
            [2.0],
            [1.0],
        )

        self.assertEqual(UUID(dataset.id).version, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "dataset.json"
            saved_path = dataset.save(path)
            loaded = DatasetClassifier.load(saved_path)

        self.assertEqual(saved_path, path)
        self.assertEqual(loaded.id, dataset.id)
        self.assertEqual(loaded.theta, dataset.theta)
        self.assertEqual(loaded.dataset, dataset.dataset)

    def test_split_is_disjoint_and_stratified(self) -> None:
        dataset = DatasetClassifier(theta=0.0)
        prompts = [f"prompt-{index}" for index in range(20)]
        answers = [f"answer-{index}" for index in range(20)]
        rewards = [1.0 if index % 2 else -1.0 for index in range(20)]
        judges = [0.0] * 20
        dataset.add(prompts, answers, rewards, judges)

        train_dataset, test_dataset = dataset.split(
            test_size=0.2,
            random_state=7,
        )

        self.assertEqual(len(train_dataset), 16)
        self.assertEqual(len(test_dataset), 4)
        self.assertEqual(train_dataset.class_counts(), {0: 8, 1: 8})
        self.assertEqual(test_dataset.class_counts(), {0: 2, 1: 2})
        self.assertEqual(train_dataset.id, dataset.id)
        self.assertEqual(test_dataset.id, dataset.id)
        train_prompts = {row["prompt"] for row in train_dataset.dataset}
        test_prompts = {row["prompt"] for row in test_dataset.dataset}
        self.assertTrue(train_prompts.isdisjoint(test_prompts))


class EntrypointTests(unittest.TestCase):
    def test_extracts_final_hh_rlhf_prompt(self) -> None:
        conversation = (
            "\n\nHuman: Hello"
            "\n\nAssistant: Hi"
            "\n\nHuman: Help me"
            "\n\nAssistant: Certainly"
        )

        prompt = _extract_prompt(conversation)

        self.assertEqual(
            prompt,
            "\n\nHuman: Hello\n\nAssistant: Hi"
            "\n\nHuman: Help me\n\nAssistant:",
        )


class PolicyEvaluationTests(unittest.TestCase):
    def test_policy_evaluation_aggregates_all_metrics(self) -> None:
        class FakePolicy:
            def generate_new_dataset(self, dataset, batch_size=8):
                return dataset.add_column("answers", ["a1", "a2"])

        class FakeEvaluator:
            def evaluate(self, dataset, concurrency=5):
                return dataset.add_column("appropriate", [1, 0])

        class FakeScorer:
            def __init__(self, scores):
                self.scores = scores

            def score(self, prompts, answers):
                return self.scores

        runner = object.__new__(Runner)
        runner.config = type(
            "Config",
            (),
            {
                "generation_batch_size": 2,
                "evaluator_concurrency": 1,
                "evaluation_seed": 7,
            },
        )()
        runner.static_dataset = Dataset.from_dict({"prompt": ["p1", "p2"]})
        runner.policy = FakePolicy()
        runner.evaluator = FakeEvaluator()
        runner.reward = FakeScorer([3.0, 1.0])
        runner.judge = FakeScorer([2.0, 2.0])

        evaluation = runner._evaluate_policy("test")

        self.assertEqual(evaluation.total, 2)
        self.assertEqual(evaluation.appropriate_count, 1)
        self.assertEqual(evaluation.appropriate_rate, 0.5)
        self.assertEqual(evaluation.proxy_reward_mean, 2.0)
        self.assertEqual(evaluation.judge_reward_mean, 2.0)
        self.assertEqual(evaluation.reward_gap_mean, 0.0)
        self.assertEqual(
            evaluation.dataset.column_names,
            [
                "prompt",
                "answers",
                "appropriate",
                "proxy_reward",
                "judge_reward",
                "reward_gap",
            ],
        )

    def test_evaluation_artifacts_are_persisted(self) -> None:
        dataset = Dataset.from_dict(
            {
                "prompt": ["p"],
                "answers": ["a"],
                "appropriate": [1],
                "proxy_reward": [2.0],
                "judge_reward": [1.0],
                "reward_gap": [1.0],
            }
        )
        from runner import PolicyEvaluation

        evaluation = PolicyEvaluation(
            stage="test",
            dataset=dataset,
            total=1,
            appropriate_count=1,
            appropriate_rate=1.0,
            proxy_reward_mean=2.0,
            proxy_reward_std=0.0,
            judge_reward_mean=1.0,
            judge_reward_std=0.0,
            reward_gap_mean=1.0,
        )
        classifier_dataset = DatasetClassifier(theta=0.0)
        classifier_dataset.add(["p"], ["a"], [2.0], [1.0])

        with tempfile.TemporaryDirectory() as directory:
            runner = object.__new__(Runner)
            runner.config = type(
                "Config",
                (),
                {"evaluation_output_dir": directory},
            )()
            runner.reward = type("Reward", (), {"classifiers": []})()

            report_path = runner._save_evaluation_artifacts(
                baseline=evaluation,
                post_ppo=evaluation,
                final_evaluation_dataset=dataset,
                final_proxy_reward_mean=2.0,
                classifier_dataset=classifier_dataset,
                classifier_train_dataset=classifier_dataset,
                classifier_test_dataset=classifier_dataset,
                classifier_metrics={"test_f1": 1.0},
            )

            report_file = Path(report_path)
            self.assertTrue(report_file.exists())
            report = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(report["classifier"]["metrics"]["test_f1"], 1.0)
            self.assertTrue((report_file.parent / "baseline.jsonl").exists())


class CompositeRewardTests(unittest.TestCase):
    def test_classifier_penalty_is_applied_to_ppo_reward_logits(self) -> None:
        class Backbone(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                hidden = torch.full((*input_ids.shape, 1), 2.0)
                return type("Output", (), {"hidden_states": [hidden]})()

        class Reward(torch.nn.Module):
            base_model_prefix = "core"

            def __init__(self):
                super().__init__()
                self.core = Backbone()
                self.config = type("Config", (), {})()

            def score(self, hidden_states):
                return hidden_states

        class Classifier(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                logits = torch.zeros((input_ids.shape[0], 1))
                return type("Output", (), {"logits": logits})()

        model = CompositeRewardModel(Reward(), [Classifier()])
        output = model.backbone(input_ids=torch.tensor([[1, 2, 3]]))
        adjusted = model.score(output.hidden_states[-1])

        expected = 2.0 + np.log(0.5)
        self.assertTrue(
            torch.allclose(adjusted, torch.full_like(adjusted, expected))
        )

    def test_normalization_is_applied_to_ppo_reward_logits(self) -> None:
        class Backbone(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                hidden = torch.full((*input_ids.shape, 1), 2.0)
                return type("Output", (), {"hidden_states": [hidden]})()

        class Reward(torch.nn.Module):
            base_model_prefix = "core"

            def __init__(self):
                super().__init__()
                self.core = Backbone()
                self.config = type("Config", (), {})()

            def score(self, hidden_states):
                return hidden_states

        model = CompositeRewardModel(Reward(), [], mean=1.0, std=0.5)
        output = model.backbone(input_ids=torch.tensor([[1, 2, 3]]))
        normalized = model.score(output.hidden_states[-1])

        self.assertTrue(
            torch.allclose(normalized, torch.full_like(normalized, 2.0))
        )


if __name__ == "__main__":
    unittest.main()
