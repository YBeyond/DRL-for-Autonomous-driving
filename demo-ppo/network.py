import logging
from typing import OrderedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 环境信息输入为图片，基于图片提取特征
class ImgFeatureNet(nn.Module):
    def __init__(self) -> None:
        super(ImgFeatureNet, self).__init__()
        self.act_cnn1 = nn.Conv2d(5, 16, 3)  # H*W = 256*256, 5 input channel
        self.act_cv1_pool = nn.MaxPool2d(3, stride=2)  # 126 * 126
        self.act_cnn2 = nn.Conv2d(16, 16, 3)
        self.act_cv2_pool = nn.MaxPool2d(3, stride=2)  # 61 * 61
        self.act_cnn3 = nn.Conv2d(16, 16, 3)
        self.act_cv3_pool = nn.MaxPool2d(3, stride=2)  # 29 * 29
        self.act_cnn4 = nn.Conv2d(16, 16, 3)
        self.act_cv4_pool = nn.MaxPool2d(3, stride=2)  # 13 * 13
        self.act_cnn5 = nn.Conv2d(16, 16, 3)  # 11 * 11
        self.output_size = 1936  # 121

    def forward(self, img_state):
        mt_tmp = F.relu(self.act_cnn1(img_state))
        mt_tmp = self.act_cv1_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn2(mt_tmp))
        mt_tmp = self.act_cv2_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn3(mt_tmp))
        mt_tmp = self.act_cv3_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn4(mt_tmp))
        mt_tmp = self.act_cv4_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn5(mt_tmp))
        mt_tmp = torch.flatten(mt_tmp, 1)  # 16 * 121 = 1936
        return mt_tmp


# 环境信息输入为语义结果，基于语义进行特征提取
class VecFeatureNet(nn.Module):
    def __init__(self) -> None:
        super(VecFeatureNet, self).__init__()
        self.act_cnn1 = nn.Conv2d(1, 1, (5, 1))
        self.output_size = 70

    def forward(self, vec_state):
        mt_tmp = F.relu(self.act_cnn1(vec_state))
        mt_tmp = torch.flatten(mt_tmp, 1)
        return mt_tmp


# 使用ActorNet替换PPONet的Actor部分
class ActorNet(nn.Module):
    def __init__(self) -> None:
        super(ActorNet, self).__init__()

    def forward(self):
        pass


# 使用CriticNet替换PPONet的Critic部分
class CriticNet(nn.Module):
    def __init__(self) -> None:
        super(CriticNet, self).__init__()

    def forward(self):
        pass


# PPO的策略网络，输入为env_state和vec_state
# TODO：优化为一个输入
class PPONet(nn.Module):
    def __init__(self, num_outputs, obs_type="img") -> None:
        super(PPONet, self).__init__()
        self.feature_net = ImgFeatureNet() if obs_type == "img" else VecFeatureNet()
        self.actor_logit = nn.Sequential(
            OrderedDict(
                [
                    ("actor_1", nn.Linear(self.feature_net.output_size + 6, 64)),
                    ("actor_relu_1", nn.ReLU()),
                    ("actor_2", nn.Linear(64, 64)),
                    ("actor_relu_2", nn.ReLU()),
                    ("actor_3", nn.Linear(64, 64)),
                    ("actor_relu_3", nn.ReLU()),
                ]
            )
        )

        self.critic_value = nn.Sequential(
            OrderedDict(
                [
                    ("critic_1", nn.Linear(self.feature_net.output_size + 6, 64)),
                    ("critic_relu_1", nn.ReLU()),
                    ("critic_2", nn.Linear(64, 64)),
                    ("critic_relu_2", nn.ReLU()),
                    ("critic_3", nn.Linear(64, 64)),
                    ("critic_relu_3", nn.ReLU()),
                    ("critic_output", nn.Linear(64, 1)),
                ]
            )
        )

        self.logit = nn.Sequential(nn.Linear(64, num_outputs * 2), nn.Tanh())

        # 初始化模型权重
        # initialize_weights

    # actor的forward函数
    def _forward_actor(self, env_obs, vec_obs):  # env_obs:来自于环境信息，vec_obs:来自于自车信息
        env_feature = self.feature_net(env_obs)
        env_feature = env_feature.reshape(
            env_obs.shape[0], -1)  # 主要是提取batch size
        complex_feature = torch.cat(
            (env_feature, vec_obs.reshape(env_feature.shape[0], -1)), 1)
        logits = self.actor_logit(complex_feature)
        action_logit = self.logit(logits)
        action_mean, action_logstd = torch.chunk(action_logit, 2, dim=-1)
        return action_mean, action_logstd

    # critic的forward函数，与actor有冗余代码
    def _forward_critic(self, env_obs, vec_obs):
        env_feature = self.feature_net(env_obs)
        env_feature = env_feature.reshape(env_obs.shape[0], -1)
        complex_feature = torch.cat(
            (env_feature, vec_obs.reshape(env_feature.shape[0], -1)), 1)
        values = self.critic_value(complex_feature)
        return values

    def forward(self, env_obs, vec_obs):
        action_mean, action_logstd = self._forward_actor(env_obs, vec_obs)
        critic_value = self._forward_critic(env_obs, vec_obs)
        # 只在这里体现critic_value,为啥不把actor和critic分开成两个网络
        return action_mean, action_logstd, critic_value

    def select_action(self, action_mean, action_logstd):
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        # 计算选择该动作的概率
        logproba = self._normal_logproba(action, action_mean, action_logstd)
        return action, logproba

    def _normal_logproba(self, x, mean, logstd, std=None):
        if std == None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = -0.5 * math.log(2*math.pi) - \
            logstd - (x-mean).pow(2)/(2*std_sq)
        return logproba.sum(1)

    # 获取动作概率和熵，与之前有冗余代码
    def get_logproba(self, env_obs, vec_obs, actions):
        action_mean, action_logstd = self._forward_actor(env_obs, vec_obs)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        entropy = torch.distributions.Normal(
            action_mean, torch.exp(action_logstd)).entropy()
        return logproba, entropy
