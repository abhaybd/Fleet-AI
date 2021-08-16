import torch

class AgentBase(object):
    def __init__(self, *args):
        self.init_args = args
        self._total_it = torch.tensor([0], dtype=torch.int64)
        self._shared_memory = [self._total_it]

    @property
    def total_it(self):
        return self._total_it.item()

    @total_it.setter
    def total_it(self, value):
        self._total_it[0] = value

    def share_memory(self):
        for x in self._shared_memory:
            if hasattr(x, "share_memory"):
                x.share_memory()
            elif hasattr(x, "share_memory_"):
                x.share_memory_()

    def reset(self):
        pass

    def select_action(self, state):
        raise Exception("Not implemented!")

    def copy(self):
        constructor = type(self)
        other = constructor(*self.init_args)
        other.copy_from(self)
        return other

    def copy_from(self, other):
        self.total_it = other.total_it

    def copy_on(self, device):
        raise Exception("Not implemented!")

    def save(self, filename):
        raise Exception("Not implemented!")

    def load(self, filename):
        raise Exception("Not implemented!")

    def train(self, sample):
        raise NotImplementedError

    def _save_dict(self):
        return {"total_it": self.total_it}

    def _load_save_dict(self, save_dict):
        self.total_it = save_dict["total_it"]
