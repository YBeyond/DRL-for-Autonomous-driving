import logging

import torch
import torch.nn as nn
import numpy as np

from smarts.core.agent import Agent
from smarts.core.sensors import Observation
from preprocess import EnvPostProcessor
from network import PPONet
from param import PolicyParam


class PPOContinuousAgent(Agent):
    def __init__(self) -> None:
        self.args = PolicyParam
        self.model = PPONet(2)
        self.env_post_processor = EnvPostProcessor()
        if self.args.model_path and self.args.mode == "eval":
            self.model.load_state_dict(torch.load(
                self.args.model_path), strict=False)

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
        return desire_action

        # if self.args.mode == "train":
        #     transition = zip(env_state, vec_state, desire_action, critic_value)
        #     return transition, desire_action
        # else:
        #     return desire_action

    # 该函数基于数据进行训练
    def update(self, transition):
        pass
