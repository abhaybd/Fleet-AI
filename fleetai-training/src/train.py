import sys
import yaml
import argparse
import os
from itertools import chain

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vec_env import DummyVecEnv, SubprocVecEnv

from util import collect_trajectories_vec_env, pretty_dict
from battleship_util import create_agent_from_args, create_env_fn, run_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("-r", "--resume", action='store_true',
                        help="Enable resume training from a saved model.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["resume"] = args.resume
    return config


def get_save_paths(args):
    dir_name = os.path.join(args["agent"]["save_dir"], args["agent"]["model_name"])
    return dir_name, f"{os.path.join(dir_name, args['agent']['algo'])}.pt"


def save_agent(args, agent):
    dir_name, agent_path = get_save_paths(args)
    os.makedirs(dir_name, exist_ok=True)
    args_path = os.path.join(dir_name, "config.yaml")
    agent.save(agent_path)
    with open(args_path, "w") as f:
        f.write(yaml.dump(args, default_flow_style=False))


def load_agent(args, agent):
    _, agent_path = get_save_paths(args)
    file_exists = os.path.isfile(agent_path)
    resume = ("resume" in args) and args["resume"]
    if resume and file_exists:
        agent.load(agent_path)
    elif file_exists and not resume:
        raise Exception("A model exists at the save path. Use -r to resume training.")
    elif resume and not file_exists:
        raise Exception("Resume flag specified, but no model found.")


def main():
    args = parse_args()

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
        writer = SummaryWriter(comment=f"_{args['agent']['model_name']}")
        args["log_dir"] = writer.log_dir

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
        if args["eval"]["eval_interval"] != -1 and agent.total_it % args["eval"]["eval_interval"] == 0:
            eval_info = run_eval(env_fn, agent.actor, args["eval"]["num_ep"], args["env"]["max_steps"])
            for name, val in eval_info.items():
                writer.add_scalar(f"Eval/{name}", val, agent.total_it)
            print(f"Evaluation - {pretty_dict(eval_info)}")

        if agent.total_it % args["training"]["save_interval"] == 0:
            save_agent(args, agent)
    save_agent(args, agent)
    env.close()
    print("Finished training!")


if __name__ == "__main__":
    main()
