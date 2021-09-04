import os
import argparse
from time import sleep

from tensorboardX import SummaryWriter
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from .util import pretty_dict, reduce_eval
from .battleship_util import create_agent_from_args, create_env_fn, run_eval


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

    return args


def eval(agent, config, num_eval=4, max_steps=500, timestep=100, seed=0, render=False, histogram=False,
         fig_callback=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_fn = create_env_fn(config)
    env = env_fn()

    render_callback = None
    cleanup_callback = None
    if render:
        try:
            env.render()

            def render(e):
                e.render()
                sleep(timestep / 1000)

            render_callback = render
            cleanup_callback = lambda: None
        except:
            print("Render flag specified, but rendering not possible! Running without rendering...")
        finally:
            env.close()

    eval_info = run_eval(env_fn, agent.actor, num_eval, max_steps, render_callback=render_callback,
                         reduce_info=False)
    traj_lens = eval_info["traj_lens"]
    traj_rews = eval_info["traj_rews"]
    last_rews = eval_info["last_rews"]
    reduced_info = reduce_eval(traj_lens, traj_rews, last_rews)

    if histogram:
        fig, (len_ax, rew_ax) = plt.subplots(2)
        fig.suptitle(f"Evaluation Metrics: {config['agent']['model_name']}\nSeed={seed}")
        def plot_hist(ax, title, x_label, y_label, data, bins=20):
            mean = np.mean(data)
            ax.set_title(title)
            ax.set(xlabel=x_label, ylabel=y_label)
            ax.hist(data, bins=bins, density=True)
            ax.axvline(mean, color="black", linestyle="dashed")
            min_ylim, max_ylim = ax.get_ylim()
            min_xlim, max_xlim = ax.get_xlim()
            text_x = mean + (max_xlim - min_xlim) * 0.02
            text_y = min_ylim + (max_ylim - min_ylim) * 0.1
            ax.text(text_x, text_y, f"Avg: {mean:.1f}")
        plot_hist(len_ax, "Episode Lengths", "Game Length", "% Games", traj_lens)
        plot_hist(rew_ax, "Total Rewards", "Total Reward", "% Games", traj_rews)
        fig.tight_layout()
        if fig_callback is not None:
            fig_callback(fig)

    if cleanup_callback is not None:
        cleanup_callback()
    return reduced_info

def main():
    args = parse_args()
    device = torch.device("cpu")
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env = create_env_fn(config)()
    agent = create_agent_from_args(device, config, env)
    model_path = os.path.join(args.model_dir, f"{config['agent']['algo']}.pt")
    if os.path.isfile(model_path):
        agent.load(model_path)
    else:
        raise Exception(f"{model_path} does not exist!")

    def fig_callback(fig):
        save_path = os.path.join(args.model_dir, "eval.png")
        fig.savefig(save_path)
        fig.show()

    info = eval(agent, config,
                num_eval=args.num_eval,
                max_steps=args.max_steps,
                timestep=args.timestep,
                seed=args.seed,
                render=args.render,
                histogram=args.histogram,
                fig_callback=fig_callback)
    print(pretty_dict(info, "%.2f"))

if __name__ == "__main__":
    main()
