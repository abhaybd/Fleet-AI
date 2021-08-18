import numpy as np
import scipy.signal

import torch
from torch import nn

from .AgentBase import AgentBase

"""
Implements PPO, with some tricks
Reference: 
    https://github.com/openai/baselines/tree/master/baselines/ppo2
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/
    https://openreview.net/forum?id=r1etN1rtPB (PPO tricks paper)
"""


class PPO(AgentBase):
    def __init__(self,
                 device,
                 actor_fn,
                 critic_fn,
                 target_kl=0.015,
                 clip_ratio=0.2,
                 discount=0.99,
                 gae_lam=0.97,
                 entropy_coeff=0.01,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-4,
                 do_assert=False
                 ):
        super().__init__(device, actor_fn, critic_fn,
                         target_kl, clip_ratio, discount, gae_lam, entropy_coeff,
                         actor_learning_rate, critic_learning_rate, do_assert)
        self.device = device

        self.actor = actor_fn(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic = critic_fn(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # initialize networks
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.discount = discount
        self.gae_lam = gae_lam
        self.entropy_coeff = entropy_coeff
        self.do_assert = do_assert

        self.vf_loss_fn = nn.MSELoss()

        self._shared_memory.extend([self.actor, self.critic])

    def copy_from(self, other):
        super().copy_from(other)

        self.actor.load_state_dict(other.actor.state_dict())
        self.actor_optimizer.load_state_dict(other.actor_optimizer.state_dict())
        self.critic.load_state_dict(other.critic.state_dict())
        self.critic_optimizer.load_state_dict(other.critic_optimizer.state_dict())

    def copy_on(self, device):
        args = (device,) + self.init_args[1:]
        other = PPO(*args)
        other.copy_from(self)
        return other

    def clamp(self, a, lo, hi):
        if type(lo) != torch.Tensor:
            lo = torch.tensor(lo, dtype=torch.float32, device=self.device).expand_as(a)
        if type(hi) != torch.Tensor:
            hi = torch.tensor(hi, dtype=torch.float32, device=self.device).expand_as(a)
        return torch.min(torch.max(a, lo), hi)

    def _discount_cumsum(self, arr, discount):
        """shape of arr is (step+1, 1)"""
        # Weirdly enough, despite the moving of the tensors back and forth, this is much faster than
        # naively doing it with pytorch operations. Go figure.
        if type(arr) == torch.Tensor:
            arr = arr.cpu().numpy()
        # taken from OpenAI Spinning Up (which took it from rllab)
        ret_np = scipy.signal.lfilter([1], [1, float(-discount)], arr[::-1], axis=0)[::-1]
        # copy is necessary because of negative stride
        return torch.tensor(ret_np.copy(), dtype=torch.float32, device=self.device)

    def _gae(self, rewards, values, last_val):
        """Shape is (traj, step, 1), except for traj_last_vals which is (traj, 1)"""
        rewards = torch.cat((rewards, torch.tensor([[last_val]], device=self.device, dtype=rewards.dtype)))
        values = torch.cat((values, torch.tensor([[last_val]], device=self.device, dtype=values.dtype)))

        deltas = rewards[:-1] + self.discount * values[1:] - values[:-1]
        advantages = self._discount_cumsum(deltas, self.discount * self.gae_lam)

        rewards_to_go = self._discount_cumsum(rewards, self.discount)[:-1]

        return advantages, rewards_to_go

    def _policy_loss(self, states, actions, old_log_probs, advantages):
        log_probs, entropy = self.actor.eval_actions(states, actions)
        policy_ratios = torch.exp(log_probs - old_log_probs)

        if self.do_assert:
            assert log_probs.shape == old_log_probs.shape == policy_ratios.shape == advantages.shape
            assert entropy.shape == ()

        # policy clipping
        policy_objective1 = policy_ratios * advantages
        policy_objective2 = advantages * self.clamp(policy_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_objective = torch.min(policy_objective1, policy_objective2)
        policy_loss = -policy_objective.mean() - entropy * self.entropy_coeff
        info = {
            "log_probs": log_probs,
            "entropy": entropy,
            "policy_ratios": policy_ratios
        }
        return policy_loss, info

    def train(self, sample, actor_steps=80, critic_steps=80):
        """
        sample is assumed to be sequential states
        sample is (states, actions, log_probs, rewards, traj_last_vals)
        each has shape (traj, step, dim), except traj_last_vals and traj_lens, which are (traj, 1)
        traj_last_val is 0 if trajectory terminated, or critic value estimate if horizon reached
        """
        self.total_it += 1
        states, actions, old_log_probs, rewards, traj_last_vals = sample

        # flatten trajectories
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)

        with torch.no_grad():
            old_values = self.critic(states)

        advantages, rewards_to_go = [], []
        for i in range(len(rewards)): # iterating over trajectories
            # old_values is flattened, and we can't reshape since trajectories aren't all the same length
            # so we calculate which values correspond to this trajectory
            idx_start = sum(map(len, rewards[:i]))
            idx_end = idx_start + len(rewards[i])
            vs = old_values[idx_start:idx_end]
            adv, r2g = self._gae(rewards[i], vs, traj_last_vals[i])
            advantages.append(adv)
            rewards_to_go.append(r2g)

        # flatten
        rewards = torch.cat(rewards, dim=0)
        advantages = torch.cat(advantages, dim=0)
        rewards_to_go = torch.cat(rewards_to_go, dim=0)

        if self.do_assert:
            assert all(len(x.shape) == 2 for x in [states, actions, old_log_probs, rewards, advantages, rewards_to_go])
            assert all(x.shape[0] == states.shape[0] for x in [states, actions, old_log_probs, advantages, rewards_to_go])
            assert all(x.shape[1] == 1 for x in [old_log_probs, rewards, advantages, rewards_to_go])

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.actor.train()
        self.critic.train()

        info = {}

        def incr(d, key, val):
            d[key] = d.get(key, 0) + val

        def div(d, key, denominator):
            if key in d.keys():
                d[key] /= denominator

        with torch.no_grad():
            pre_policy_loss, _ = self._policy_loss(states, actions, old_log_probs, advantages)

        actor_step = 0
        while actor_step < actor_steps:
            actor_step += 1

            self.actor_optimizer.zero_grad()
            policy_loss, loss_info = self._policy_loss(states, actions, old_log_probs, advantages)
            policy_loss.backward()
            self.actor_optimizer.step()

            # useful info
            approx_kl = (old_log_probs - loss_info["log_probs"]).mean()
            clipped = (loss_info["policy_ratios"] > (1+self.clip_ratio)) | (loss_info["policy_ratios"] < (1-self.clip_ratio))
            clip_frac = clipped.float().mean()

            incr(info, "actor_loss", policy_loss.item())
            incr(info, "entropy", loss_info["entropy"].item())
            info["approx_kl"] = approx_kl.item() # don't increment
            incr(info, "policy_clip_frac", clip_frac.item())

            # prevent actor from diverging too much
            if approx_kl > self.target_kl:
                break

        post_policy_loss, _ = self._policy_loss(states, actions, old_log_probs, advantages)
        info["actor_loss_delta"] = (post_policy_loss - pre_policy_loss).item()
        info["actor_train_steps"] = actor_step
        div(info, "actor_loss", actor_step)
        div(info, "entropy", actor_step)
        div(info, "policy_clip_frac", actor_step)


        for _ in range(critic_steps):
            # value function clipping
            value_pred = self.critic(states)
            value_pred_clipped = self.clamp(value_pred, old_values - self.clip_ratio, old_values + self.clip_ratio)
            if self.do_assert:
                assert value_pred.shape == value_pred_clipped.shape == rewards_to_go.shape
            vf_loss1 = self.vf_loss_fn(value_pred, rewards_to_go)
            vf_loss2 = self.vf_loss_fn(value_pred_clipped, rewards_to_go)
            vf_loss = torch.min(vf_loss1, vf_loss2)

            self.critic_optimizer.zero_grad()
            vf_loss.backward()
            self.critic_optimizer.step()

            incr(info, "critic_loss", vf_loss.item() / critic_steps)

        return info

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1,-1)
            pd = self.actor(state)
            sample = pd.sample().flatten()
            log_prob = pd.log_prob(sample).sum()
            return sample.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def select_action_greedy(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1,-1)
            self.actor.eval()
            return self.actor.greedy(state).cpu().numpy().squeeze()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1,-1)
            self.critic.eval()
            return self.critic(state).item()

    def save(self, filename):
        state = super()._save_dict()
        state.update({
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic": self.critic.state_dict()
        })
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename, map_location=self.device)
        super()._load_save_dict(state)
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
