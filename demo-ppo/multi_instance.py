import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt


# from workers import MemorySampler
from new_worker import MemorySampler
# from policy import PPOPolicy
from network import PPONet
from param import PolicyParam
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='./log')
TENSORBOARD_LOG = False


# 定义logger
logger = logging.getLogger("PPO")
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
fh = logging.FileHandler('myapp.log')
sh.setLevel(logging.DEBUG)
formater = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formater)
fh.setFormatter(formater)
logger.addHandler(sh)
# logger.addHandler(fh)


# TODO : 1. 保存模型  2.加载模型 3.logger 4.tensorboard可视化 5.生成log和模型保存路径


class MulProPPO:
    def __init__(self) -> None:
        self.args = PolicyParam
        self.global_sample_size = 0
        self._make_dir()
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.args.device == "cuda":
            torch.cuda.manual_seed(self.args.seed)
        print("current device is {}".format(self.args.device))
        self.clip_now = self.args.clip

        self.sampler = MemorySampler(self.args)
        self.model = PPONet(2)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.args.lr)
        self.start_episode = 0
        self._load_model(self.args.model_path)

    def _load_model(self, model_path: str = None):
        if not model_path:
            return
        pretrained_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if self._check_keys(self.model, pretrained_dict):
            self.model.load_state_dict(pretrained_dict, strict=False)

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict.keys())
        used_pretrained_keys = model_keys & ckpt_keys
        missing_keys = ckpt_keys - model_keys
        missing_keys = [x for x in missing_keys if not x.endswith(
            "num_batches_tracked")]
        assert len(
            used_pretrained_keys) > 0, "check_key load NONE from pretrained checkpoint"
        return True

    def _make_dir(self):
        current_dir = os.path.abspath(".")
        self.exp_dir = current_dir + \
            "/results/{}/exp_{}/".format(time.strftime("%Y-%m-%d"),
                                         time.time())
        self.model_dir = current_dir + \
            "/results/{}/model_{}/".format(time.strftime("%Y-%m-%d"),
                                           time.time())
        try:
            # os.makedirs(self.exp_dir)
            os.makedirs(self.model_dir)
        except:
            print("file is existed")

    def train(self):
        for i_episode in range(self.args.num_episode):
            print("--------"+str(i_episode)+"--------")
            logger.info("in function train")
            memory = self.sampler.smaple(self.model)
            batch = memory.sample()
            batch_size = len(memory)
            rewards = torch.from_numpy(np.array(batch.reward))
            values = torch.from_numpy(np.array(batch.value))
            masks = torch.from_numpy(np.array(batch.mask))
            actions = torch.from_numpy(np.array(batch.action))
            env_states = torch.from_numpy(np.array(batch.env_state))
            vec_states = torch.from_numpy(np.array(batch.vec_state))
            oldlogprobas = torch.from_numpy(np.array(batch.logproba))

            returns = torch.Tensor(batch_size)
            deltas = torch.Tensor(batch_size)
            advantages = torch.Tensor(batch_size)
            prev_return = 0
            prev_value = 0
            prev_advantage = 0

            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + \
                    self.args.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.args.gamma * \
                    prev_value * masks[i] - values[i]
                advantages[i] = deltas[i] + self.args.gamma + \
                    self.args.lamda * prev_advantage * masks[i]

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]

            if self.args.advantage_norm:
                advantages = (advantages - advantages.mean()
                              )/(advantages.std() + 1e-6)

            env_states = env_states.to(self.device)
            values = values.to(self.device)
            vec_states = vec_states.to(self.device)
            actions = actions.to(self.device)
            oldlogprobas = oldlogprobas.to(self.device)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)
            for i_epoch in range(int(self.args.num_epoch * batch_size / self.args.minibatch_size)):
                print("i_epoch : {}".format(i_epoch))
                minibatch_ind = np.random.choice(
                    batch_size, self.args.minibatch_size, replace=False)
                minibatch_env_state = env_states[minibatch_ind]
                minibatch_vec_state = vec_states[minibatch_ind]
                minibatch_actions = actions[minibatch_ind]
                minibatch_values = values[minibatch_ind]
                minibatch_oldlogproba = oldlogprobas[minibatch_ind]
                minibatch_newlogproba, entropy = self.model.get_logproba(
                    minibatch_env_state, minibatch_vec_state, minibatch_actions)
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newvalues = self.model._forward_critic(
                    minibatch_env_state, minibatch_vec_state).flatten()
                assert minibatch_oldlogproba.shape == minibatch_newlogproba.shape
                ratio = torch.exp(
                    minibatch_newlogproba - minibatch_oldlogproba)
                assert ratio.shape == minibatch_advantages.shape
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1-self.clip_now, 1 +
                                    self.clip_now) * minibatch_advantages
                loss_surr = -torch.mean(torch.min(surr1, surr2))

                if self.args.use_clipped_value_loss:
                    value_pred_clipped = minibatch_values + \
                        (minibatch_newvalues -
                         minibatch_values).clamp(-self.args.vf_clip_param, self.args.vf_clip_param)
                    value_losses = (minibatch_newvalues -
                                    minibatch_returns).pow(2)
                    value_loss_clip = (value_pred_clipped -
                                       minibatch_returns).pow(2)
                    loss_value = torch.max(
                        value_losses, value_loss_clip).mean()
                else:
                    loss_value = torch.mean(
                        (minibatch_newvalues - minibatch_returns).pow(2))

                if self.args.lossvalue_norm:
                    minibatch_return_6std = 6*minibatch_returns.std()
                    loss_value = (torch.mean(
                        (minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std)
                else:
                    loss_value = torch.mean(
                        (minibatch_newvalues - minibatch_returns).pow(2))

                loss_entropy = -torch.mean(entropy)

                total_loss = (loss_surr + self.args.loss_coeff_value *
                              loss_value + self.args.loss_coeff_entropy * loss_entropy)
                if TENSORBOARD_LOG:
                    writer.add_scalars('ppo_loss', {'loss_surr': loss_surr, 'loss_value': loss_value,
                                                    'loss_entropy': loss_entropy, 'total_loss': total_loss}, i_epoch)
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                if i_episode % self.args.save_num_episode == 0:
                    torch.save(self.model.state_dict(), self.model_dir +
                               "network_{}.pth".format(i_episode))

        self.sampler.close()


if __name__ == '__main__':
    torch.set_num_threads(1)

    mpp = MulProPPO()
    mpp.train()
