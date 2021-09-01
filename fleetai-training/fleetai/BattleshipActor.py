import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

from .actor_critic import ActorBase, create_network


class BattleshipActor(ActorBase):
    def __init__(self, device, state_dim, action_dim, layers=(256, 256)):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.logits_net = create_network((state_dim,) + layers + (action_dim,)).to(device)
        self.forward_probs = False # required for torchscript tracing

    def forward(self, state):
        if self.forward_probs:
            return self.probs(state)
        else:
            return self._distribution(state)

    def _distribution(self, state):
        output = self.logits_net(state)
        return Categorical(logits=output)

    def _log_prob(self, dist, actions):
        """Assumes actions is column vector, returns column vector"""
        return dist.log_prob(actions.squeeze()).unsqueeze(-1)

    def greedy(self, state):
        return self._distribution(state).probs.argmax()

    def probs(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        logits = self.logits_net(state)
        probs = logits_to_probs(logits)
        return probs
