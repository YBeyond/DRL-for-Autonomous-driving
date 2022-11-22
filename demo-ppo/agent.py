import logging
import logging.config

import torch
import numpy as np

from smarts.core.agent import Agent
from smarts.core.sensors import Observation
from preprocess import EnvPostProcessor
from network import PPONet


class PPOContinuousAgent(Agent):
    def __init__(self, running_mode="train") -> None:
        self.model = PPONet(2)
        self.env_post_processor = EnvPostProcessor()
        self.training = True if running_mode == "train" else False

    def act(self, obs: Observation):
        new_env_obs = self.env_post_processor.AssembleEnvState(obs)
        env_state = torch.Tensor(
            np.array(list(new_env_obs["env_info"]))).float().unsqueeze(0)
        vec_state = torch.Tensor(
            np.array(list(new_env_obs["vec_info"]))).float().unsqueeze(0)

        action_mean, action_logstd, critic_value = self.model.forward(
            env_state, vec_state)
        action, logproba = self.model.select_action(action_mean, action_logstd)
        action = action.data.cpu().numpy()[0]
        # [throttle, brake, steer]
        desire_action = [max(0, action[1]), min(
            0, action[1]), action[0]]
        transition = zip(env_state, vec_state, desire_action, critic_value)
        if self.training:
            return transition, desire_action
        else:
            return desire_action

    # 该函数基于数据进行训练
    def update(self, transition):
        pass
