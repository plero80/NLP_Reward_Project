from importlib import reload

from Trainers.trainer_classifier import ClassifierTrainer, ClassifierTrainingConfig
from Trainers.trainer_ppo import PPOTrainingConfig, PolicyPPOTrainer
from Models.model_policy import PolicyModel
from Models.model_value import ValueModel
from Models.model_reward import RewardModel
from Models.model_classifier import Classifier
import Models.model_evaluator as model_evaluator
import Datasets.dataset_request as dataset_request
from datasets import load_dataset
import torch
import gc
import asyncio


dataset_request = reload(dataset_request)
RequestDataset = dataset_request.RequestDataset



def eval_policy_with_reward(
    dataset_name: str,
    start: int,
    end: int,
    policy_name: str,
    reward_model_name: str,
    reward_mode_name: str,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    rollout_forward_batch_size: int = 16,
    policy_batch_size: int = 16,
    response_length: int = 128,
) -> float:

    # 1.) Create dataset
    dataset = load_dataset(dataset_name)
    dataset = RequestDataset.from_raw(dataset, "Qwen/Qwen3-0.6B")
    dataset.truncate(start, end)
    
    
    # 2.) Add reward and policy and value
    policy = PolicyModel(policy_name)
    value = ValueModel(policy_name)
    reward_model = RewardModel(reward_model_name, reward_mode_name)


    # 3.) PPO training 
    config = PPOTrainingConfig(
        output_dir=(
            f"outputs/ppo_policy/"
            f"{reward_mode_name}_{start}_{end}"
        ),
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rollout_forward_batch_size=rollout_forward_batch_size,
        response_length=response_length,
        num_ppo_epochs=4,
        num_mini_batches=1,
        learning_rate=3e-6,
        save_steps=100,
        save_total_limit=1,
    )
    trainer = PolicyPPOTrainer(policy, reward_model, value, dataset, config)
    trainer.train()


    # 4.) Delete reward and value
    del trainer
    del reward_model
    del value


    gc.collect()
    torch.cuda.empty_cache()


    # 5.) generate answers from policy.
    policy.generate_new_dataset(dataset, policy_batch_size)


    # 6.) Move policy to cpu.
    policy.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    
    
    # 7.) Create evaluator and evaluate policy
    PrometheusEvaluator = model_evaluator.PrometheusEvaluator
    evaluator = PrometheusEvaluator()
    value = evaluator.evaluate(policy)


    # 8.) Delete evaluator
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()


    return value


