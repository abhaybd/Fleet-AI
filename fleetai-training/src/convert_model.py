import argparse
import os

import torch
import yaml

from battleship_util import load_agent_from_args, create_env_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to model directory")
    parser.add_argument("output_path", type=str, help="Path to place the output file")
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return args, config

def main():
    args, config = parse_args()

    env = create_env_fn(config)()
    agent = load_agent_from_args(torch.device("cpu"), args.model_dir, config)

    state = torch.tensor(env.observation_space.sample(), dtype=torch.float32)
    traced_actor = torch.jit.trace(agent.actor, (state,torch.tensor(True)))
    print(traced_actor.code)
    traced_actor.save(args.output_path)

if __name__ == "__main__":
    main()