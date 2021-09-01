from functools import partial
import yaml

from .train_base import train
from .util import save_agent, load_agent


def read_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    train(partial(save_agent, None), partial(load_agent, None), read_config)