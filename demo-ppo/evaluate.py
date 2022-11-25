import logging
import pathlib
import os

import torch

import gym
from argument_parser import default_argument_parser
from policy import PPOPolicy, ImgFeatureNet
from preprocess import EnvPostProcessor
import numpy as np

from smarts import sstudio
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.zoo.agent_spec import AgentSpec

from reward import new_reward
from agent import PPOContinuousAgent

# from tensorboardX import SummaryWriter


# import numpy as np
# np.set_printoptions(threshold=np.inf)


def DealWithImage(env_obs: Observation):
    rgb_image = env_obs.top_down_rgb.data
    gray_image = np.dot(rgb_image, (0.1, 0.8, 0.1))
    gray_image = np.divide(gray_image, 255.0)
    # gray_image = gray_image[-4:]  # Keep the _STACK most recent frames.
    print("shape:{} data:{}".format(gray_image.shape, gray_image))

    return gray_image


class ContinuousAgent(Agent):
    def __init__(self, model_path=None) -> None:
        self.model = PPOPolicy(2)
        self.env_post_processor = EnvPostProcessor()
        self.feature_net = ImgFeatureNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def act(self, obs: Observation):
        new_env_obs = self.env_post_processor.AssembleEnvState(env_obs=obs)
        env_state = torch.Tensor(
            np.array(list(new_env_obs["env_info"]))).float().unsqueeze(0)
        vec_state = torch.Tensor(
            np.array(list(new_env_obs["vec_info"]))).float().unsqueeze(0)
        # print("cnn obs:{}".format(env_state.shape))
        # print("cnn obs:{}".format(vec_state.shape))
        # print("env state-env info:{}".format(len(new_env_obs["env_info"])))
        action_mean, action_logstd, critic_value = self.model.forward(
            env_state, vec_state)
        # print("action mean shape :{}".format(action_mean.shape))
        action, logproba = self.model.select_action(action_mean, action_logstd)
        # print("action shape :{}".format(action.shape))
        # print("logproba : {}".format(logproba))
        action = action.data.cpu().numpy()[0]
        agent_action = [max(
            0, action[1]), min(0, action[1]), action[0]]
        # return agent_action
        return [0.05, 0.0, 2]


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # Step 1:定义Agent的输入和输出接口
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Full, max_episode_steps=max_episode_steps
        ),
        agent_builder=PPOContinuousAgent,
        reward_adapter=new_reward,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=True,
        visdom=False,
    )
    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    env = SingleAgent(env=env)

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            # print("===action:{}".format(agent_action))
            # agent_action = agent_action.from_numpy(np.array(agent_action))
            observation, reward, done, info = env.step(agent_action)
            # print("===observation:{}".format(observation))
            # print("===reward:{}".format(reward))
            # print("info:{}".format(info))
            episode.record_step(observation, reward, done, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "sumo"
                / "loop"
            )
        ]
    root_dir = "/src/smarts/scenarios"
    args.scenarios = [root_dir+"/intersection/1_to_2lane_left_turn_c"]
    # args.scenarios = [root_dir+"/intersection/1_to_2lane_left_turn_t"]
    # args.scenarios = [root_dir+"/merge/3lane_multi_agent"]
    # args.scenarios = [root_dir+"/merge/3lane_single_agent"]
    args.episodes = 50
    # sstudio.build_scenario(scenario=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
