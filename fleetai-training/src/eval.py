import os
import argparse
from time import sleep

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from util import pretty_dict
from battleship_util import create_agent_from_args, create_env_fn, run_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to directory containing config and model file")
    parser.add_argument("-n", "--num_eval", type=int, default=4, help="Number of episodes to run")
    parser.add_argument("-ms", "--max_steps", type=int, default=500, help="Max episode length")
    parser.add_argument("-ts", "--timestep", type=int, default=1000, help="Timestep, in ms. Only relevant if -r defined.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--render", action="store_true",
                       help="Render to screen (mutually exclusive with -hg)")
    group.add_argument("-hg", "--histogram", action="store_true",
                       help="Plot histograms of data (use larger values of -n)")
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return args, config


def main():
    args, config = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_fn = create_env_fn(config)
    env = env_fn()
    device = torch.device("cpu")
    agent = create_agent_from_args(device, config, env)

    model_path = os.path.join(args.model_dir, f"{config['agent']['algo']}.pt")
    if os.path.isfile(model_path):
        agent.load(model_path)
    else:
        raise Exception(f"{model_path} does not exist!")

    render_callback = None
    cleanup_callback = None
    if args.render:
        try:
            env.render()

            def render(e):
                e.render()
                sleep(args.timestep / 1000)

            render_callback = render
            cleanup_callback = lambda: None
        except:
            print("Render flag specified, but rendering not possible! Running without rendering...")
        finally:
            env.close()

    if not args.histogram:
        eval_info = run_eval(env_fn, agent.actor, args.num_eval, args.max_steps, render_callback=render_callback)
        print("\n\n" + pretty_dict(eval_info, float_fmt="%.2f") + "\n\n")
    else:
        hist_info = run_eval(env_fn, agent.actor, args.num_eval, args.max_steps, render_callback=render_callback,
                             reduce_info=False)
        traj_lens = hist_info["traj_lens"]
        traj_rews = hist_info["traj_rews"]
        fig, (len_ax, rew_ax) = plt.subplots(2)
        fig.suptitle(f"Evaluation Metrics: {config['agent']['model_name']}\nSeed={args.seed}")
        len_ax.set_title("Episode Lengths")
        len_ax.set(xlabel="Game Length", ylabel="% Games")
        len_ax.hist(traj_lens, bins=20, density=True)
        rew_ax.set_title("Total Rewards")
        rew_ax.set(xlabel="Total Reward", ylabel="% Games")
        rew_ax.hist(traj_rews, bins=20, density=True)
        fig.tight_layout()
        plt.show()

        save_path = os.path.join(args.model_dir, "eval.png")
        fig.savefig(save_path)

    if cleanup_callback is not None:
        cleanup_callback()


if __name__ == "__main__":
    main()