import logging
import pathlib
import multiprocessing as mp
from collections import namedtuple

import numpy as np
import torch

from env import make_env
from preprocess import EnvPostProcessor

from argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes
from smarts import sstudio


logger = logging.getLogger("PPO")

Transition = namedtuple("Transition", ("env_state", "vec_state",
                        "value", "action", "logproba", "mask", "reward", "info"))

Get_Enough_Batch = mp.Value("i", 0)


class Episode(object):
    def __init__(self) -> None:
        self.episode = []

    def push(self, *args):
        self.episode.append(Transition(*args))

    def __len__(self):
        return len(self.episode)


class Memory(object):
    def __init__(self) -> None:
        self.memory = []
        self.num_episode = 0
        self.arrive_goal_num = 0

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1
        # TODO：需要根据新的info修改
        # if epi.episode[-1][-1]["reached_stoparea"]:
        #     self.arrive_goal_num += 1

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class EnvWorker(mp.Process):
    def __init__(self, remote, queue, lock, seed, worker_index) -> None:
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.worker_index = worker_index
        self.queue = queue
        self.lock = lock
        self.env_post_processor = EnvPostProcessor()

        torch.manual_seed(seed)
        np.random.seed(seed)

    def do_process(self, scenarios, headless, num_episodes, max_episode_steps=None):
        # Step 1:定义Agent的输入和输出接口
        env, agent_spec = make_env(scenarios, headless, max_episode_steps)
        episode_counter = 0
        while True:
            command, policy = self.remote.recv()
            if command == "sample":
                # for _ in range(num_episodes):
                while Get_Enough_Batch.value == 0:
                    episode_counter = episode_counter+1

                    # for episode in episodes(n=num_episodes):
                    agent = agent_spec.build_agent()
                    observation = env.reset()
                    # episode.record_scenario(env.scenario_log)
                    print("---worker: {} episode : {}---".format(
                        self.worker_index, episode_counter))
                    logger.debug("in function do_process")
                    done = False
                    while not done:
                        epi = Episode()
                        with torch.no_grad():
                            new_obs = self.env_post_processor.AssembleEnvState(
                                env_obs=observation)
                            env_state = torch.Tensor(
                                np.array(list(new_obs["env_info"]))).float().unsqueeze(0)
                            vec_state = torch.Tensor(
                                np.array(list(new_obs["vec_info"]))).float().unsqueeze(0)
                            action_mean, action_logstd, critic_value = policy.forward(
                                env_state, vec_state)
                            action, logproba = policy.select_action(
                                action_mean, action_logstd)
                            action = action.data.cpu().numpy()[0]
                            logproba = logproba.data.cpu().numpy()[0]
                            value = critic_value.data.cpu().numpy()[0]
                            env_state = env_state.data.cpu().numpy()[0]
                            vec_state = vec_state.data.cpu().numpy()[0]
                        self.env_post_processor.RecordLastAction(action=action)
                        # steer:action[0],accelerate:action[1]-> [throttle,brake,steer]
                        agent_action = [max(
                            0, action[1]), min(0, action[1]), action[0]]
                        # agent_action = agent.act(observation)
                        observation, reward, done, info = env.step(
                            agent_action)
                        # episode.record_step(observation, reward, done, info)
                        mask = 0 if done else 1
                        epi.push(env_state, vec_state, value, action,
                                 logproba, mask, reward, info)
                        if done:
                            with self.lock:
                                self.queue.put(epi)
                            break
            elif command == "close":
                self.remote.close()
                self.env.close()
                break

        env.close()

    def run(self):
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
        # args.scenarios = [root_dir+"/intersection/1_to_2lane_left_turn_c",
        #                   root_dir+"/intersection/1_to_2lane_left_turn_t",
        #                   root_dir+"/merge/3lane_multi_agent",
        #                   root_dir+"/merge/3lane_single_agent",
        #                   ]
        args.scenarios = [root_dir+"/intersection/1_to_2lane_left_turn_c"]
        # args.scenarios = [root_dir+"/intersection/1_to_2lane_left_turn_t"]
        # args.scenarios = [root_dir+"/merge/3lane_multi_agent"]
        # args.scenarios = [root_dir+"/merge/3lane_single_agent"]
        args.headless = True

        # sstudio.build_scenario(scenario=args.scenarios)
        self.do_process(
            scenarios=args.scenarios,
            headless=args.headless,
            num_episodes=15,
        )


class MemorySampler(object):
    def __init__(self, args) -> None:
        self.args = args
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.device = args.device

        self.queue = mp.Queue()
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_workers)])
        self.workers = [EnvWorker(remote, self.queue, self.lock, args.seed + index, index)
                        for index, remote in enumerate(self.work_remotes)]

        for worker in self.workers:
            # worker.daemon = True
            worker.start()

        for remote in self.work_remotes:
            remote.close()

    def smaple(self, policy):
        policy.to("cpu")
        memory = Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(("sample", policy))

        while len(memory) < self.batch_size:
            episode = self.queue.get(True)
            memory.push(episode)

        Get_Enough_Batch.value = 1

        while self.queue.qsize() > 0:
            self.queue.get()

        policy.to(self.device)
        print("worker num : {} queue size :{} memory len : {}".format(
            self.num_workers, self.queue.qsize(), len(memory)))
        return memory

    def close(self):
        Get_Enough_Batch.value = 1
        for remote in self.remotes:
            remote.send(("close", None))
        for worker in self.workers:
            worker.join()
