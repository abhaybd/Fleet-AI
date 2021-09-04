import sys
import argparse
from itertools import chain

import numpy as np
from tensorboardX import SummaryWriter

from .vec_env import DummyVecEnv, SubprocVecEnv

from .dummy_writer import DummyWriter
from .util import collect_trajectories_vec_env, pretty_dict, get_or_else, \
    create_writer as create_writer_default
from .battleship_util import create_agent_from_args, create_env_fn, run_eval
from .eval import eval

# import last so comet stuff can happen beforehand
import torch

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


def log_metrics(writer: SummaryWriter, args, eval_metrics):
    def flatten_dict(source_dict, dest_dict):
        for key, value in source_dict.items():
            if isinstance(value, dict):
                flatten_dict(value, dest_dict)
            else:
                dest_dict[key] = value
    hparams = {}
    flatten_dict(args["training"], hparams)
    hparams.pop("gpu_idx", None)
    hparams.pop("use_gpu", None)
    hparams.pop("seed", None)
    hparams.pop("save_interval", None)
    hparams["latent_var_precision"] = args["env"]["latent_var_precision"]
    writer.add_hparams(hparam_dict=hparams, metric_dict=eval_metrics)


def train(save_agent, load_agent, load_config, create_writer=create_writer_default):
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

    with (DummyWriter() if get_or_else(args["logging"], "disabled", False) else create_writer(args)) as writer:
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
        if get_or_else(args["eval"], "num_eval_after", -1) > 0:
            print("Running evaluation...")
            final_eval_info = eval(agent, args,
                                   num_eval=args["eval"]["num_eval_after"],
                                   max_steps=args["env"]["max_steps"],
                                   seed=args["eval"]["seed"],
                                   histogram=True,
                                   fig_callback=lambda fig: writer.add_figure("Metrics Histogram", fig))
            log_metrics(writer, args, final_eval_info)
            print("Finished evaluating!")
        else:
            log_metrics(writer, args, {})
