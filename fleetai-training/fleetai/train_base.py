import sys
import argparse
from itertools import chain
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter

from .vec_env import DummyVecEnv, SubprocVecEnv

from .util import collect_trajectories_vec_env, pretty_dict, get_or_else
from .battleship_util import create_agent_from_args, create_env_fn, run_eval


def parse_args(load_config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("-r", "--resume", action='store_true',
                        help="Enable resume training from a saved model.")
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    config["resume"] = args.resume
    return config


def train(save_agent, load_agent, load_config):
    args = parse_args(load_config)

    np.random.seed(args["training"]["seed"])
    torch.manual_seed(args["training"]["seed"])

    env_fn = create_env_fn(args)

    if torch.cuda.is_available() and args["training"]["use_gpu"]:
        device = torch.device("cuda", args["training"]["gpu_idx"])
    else:
        if args["training"]["use_gpu"] and not torch.cuda.is_available():
            print("WARN: use_gpu specified but CUDA not available! Using CPU instead...", file=sys.stderr)
        device = torch.device("cpu")

    # create environment (optionally using subprocesses)
    if args["training"]["num_procs"] == 1:
        env = DummyVecEnv([env_fn])
    else:
        env = SubprocVecEnv([env_fn] * args["training"]["num_procs"])

    agent = create_agent_from_args(device, args, env)
    load_agent(args, agent)

    if "log_dir" in args:
        writer = SummaryWriter(log_dir=args["log_dir"])
    else:
        log_base_dir = get_or_else(args["training"], "log_base_dir", "runs")
        log_dir_name = datetime.now().strftime("%b%d_%H%M%S") + "_" + args["agent"]["model_name"]
        log_dir = log_base_dir + "/" + log_dir_name # don't use join since windows can do / but GCP can't do \
        writer = SummaryWriter(log_dir=log_dir)
        args["log_dir"] = log_dir

    save_interval = args["training"]["save_interval"]
    eval_interval = args["eval"]["eval_interval"]

    save_agent(args, agent)
    while agent.total_it < args["training"]["total_steps"]:
        sample, rollout_info = collect_trajectories_vec_env(env, args["training"]["train_samples"], device,
                                                            agent.select_action, agent.get_value,
                                                            max_steps=args["env"]["max_steps"],
                                                            policy_accepts_batch=False)
        train_info = agent.train(sample, actor_steps=args["training"]["ppo"]["actor_steps"],
                                 critic_steps=args["training"]["ppo"]["critic_steps"])

        for name, val in chain(train_info.items(), rollout_info.items()):
            writer.add_scalar(f"Train/{name}", val, agent.total_it)
        print(f"{agent.total_it} - {pretty_dict({**train_info, **rollout_info})}")

        # launch eval
        if eval_interval != -1 and agent.total_it % eval_interval == 0:
            eval_info = run_eval(env_fn, agent.actor, args["eval"]["num_ep"], args["env"]["max_steps"])
            for name, val in eval_info.items():
                writer.add_scalar(f"Eval/{name}", val, agent.total_it)
            print(f"Evaluation - {pretty_dict(eval_info)}")

        if save_interval != -1 and agent.total_it % save_interval == 0:
            save_agent(args, agent)
    save_agent(args, agent)
    env.close()
    print("Finished training!")
