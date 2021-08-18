import torch
from torch import nn
from torch.distributions import Categorical, Normal
import numpy as np

from .MultiCategorical import MultiCategorical


def _create_network(layer_sizes, activation=nn.Tanh, end_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes)-2 else end_activation
        layers += (nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act())
    return nn.Sequential(*layers)


class ActorBase(nn.Module):
    def _distribution(self, state):
        raise NotImplementedError

    def forward(self, state):
        return self._distribution(state)

    def _log_prob(self, dist, actions):
        raise NotImplementedError

    def eval_actions(self, states, actions):
        dist = self._distribution(states)
        return self._log_prob(dist, actions), dist.entropy().mean()

    def greedy(self, state):
        raise NotImplementedError


class MultiDiscActor(ActorBase):
    def __init__(self, device, state_dim, action_dims, layers=(64,64)):
        super().__init__()
        self.device = device
        self.base_net = _create_network((state_dim,) + layers, end_activation=nn.Tanh).to(device)
        self.outputs = nn.ModuleList()
        for n in action_dims:
            self.outputs.append(nn.Linear(layers[-1], n).to(device))

    def _distribution(self, state):
        base_out = self.base_net(state)
        dists = [Categorical(logits=layer(base_out)) for layer in self.outputs]
        return MultiCategorical(dists)

    def _log_prob(self, dist, actions):
        return dist.log_prob(actions).unsqueeze(-1)

    def greedy(self, state):
        return self._distribution(state).argmax()

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class DiscActor(ActorBase):
    def __init__(self, device, state_dim, action_dim, layers=(64,64)):
        super().__init__()
        self.logits_net = _create_network((state_dim,) + layers + (action_dim,)).to(device)

    def _distribution(self, state):
        output = self.logits_net(state)
        return Categorical(logits=output)

    def _log_prob(self, dist, actions):
        """Assumes actions is column vector, returns column vector"""
        return dist.log_prob(actions.squeeze()).unsqueeze(-1)

    def greedy(self, state):
        return self._distribution(state).probs.argmax()


class ContScaledActor(ActorBase):
    def __init__(self, device, state_dim, min_action, max_action, std=0.6, layers=(64,64)):
        super().__init__()
        assert min_action.shape == max_action.shape

        action_dim = max_action.shape[0]
        self.layers = _create_network((state_dim,) + layers + (action_dim,), end_activation=nn.Tanh)
        self.min_action = torch.tensor(min_action, dtype=torch.float32, device=device)
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)
        log_std = torch.tensor(np.log(std), dtype=torch.float32, device=device).expand((action_dim,)).clone()
        self.log_std = nn.Parameter(log_std)

    def _distribution(self, state):
        output = self.layers(state)
        min_act = self.min_action.expand_as(output)
        max_act = self.max_action.expand_as(output)
        mean = (output + 1) / 2.0 # scale to [0,1]
        mean = (mean * (max_act - min_act)) + min_act # scale to [min, max]
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob(self, dist, actions):
        """Assumes dimension of actions is (batch, action_dim)"""
        return dist.log_prob(actions).sum(axis=-1).unsqueeze(-1) # needs to return a column vector

    def greedy(self, state):
        return self._distribution(state).mean

    def to(self, device):
        s = super().to(device)
        s.min_action = s.min_action.to(device)
        s.max_action = s.max_action.to(device)
        return s


class ContActor(ActorBase):
    def __init__(self, device, state_dim, action_dim, std=0.6, layers=(64, 64)):
        super(ContActor, self).__init__()

        self.layers = _create_network((state_dim,) + layers + (action_dim,)).to(device)

        log_std = torch.tensor(np.log(std), dtype=torch.float32, device=device).expand((action_dim,)).clone()
        self.log_std = nn.Parameter(log_std)

    def _distribution(self, state):
        output = self.layers(state)
        mean = output
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob(self, dist, actions):
        """Assumes dimension of actions is (batch, action_dim)"""
        return dist.log_prob(actions).sum(axis=-1).unsqueeze(-1) # needs to return a column vector

    def greedy(self, state):
        return self._distribution(state).mean


class Critic(nn.Module):
    def __init__(self, device, state_dim, layers=(64, 64)):
        super().__init__()

        self.v_net = _create_network((state_dim,) + layers + (1,)).to(device)

    def forward(self, state):
        return self.v_net(state)
