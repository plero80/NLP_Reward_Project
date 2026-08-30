from Models.model_policy import PolicyModel
from Models.model_reward import RewardModel
from Trainers.trainer_ppo import PolicyPPOTrainer
from Trainers.trainer_classifier import ClassifierTrainer
from Datasets import dataset_classifier


class Runner:
    
    
    def __init__(self, dataset, config):
        
        
        self.policy = PolicyModel(config.policy_name)
        self.reward = RewardModel(config.reward_name, "proxy_reward", config.reward_model_load, config.reward_model_checkpoint)
        self.judge = RewardModel(config.judge_name, "judge_reward")
        self.dataset  = dataset
        
        
        self.static_dataset = dataset[:config.static_dataset_length]
        
        
    
    def run(self):
        
        # 1.) Evaluate the policy.
        generated_answer_dataset = self.policy.generate_new_dataset(self.static_dataset)