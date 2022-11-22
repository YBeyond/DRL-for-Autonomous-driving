import logging

import gym

# from smarts import sstudio
# from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
# from smarts.core.sensors import Observation
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.zoo.agent_spec import AgentSpec

from agent import PPOContinuousAgent
from reward import new_reward


def make_env(scenarios, headless, max_episode_steps=None):
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
    return env, agent_spec
