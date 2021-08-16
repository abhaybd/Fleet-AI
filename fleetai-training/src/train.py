import yaml
import argparse
import os
from functools import partial
from itertools import chain

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vec_env import DummyVecEnv, SubprocVecEnv

from PPO import PPO
from env import BattleshipEnv
from util import collect_trajectories_vec_env, run_evaluation, pretty_dict

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
    dir_name = os.path.join(args.agent.save_dir, args.agent.model_name)
    return dir_name, f"{os.path.join(dir_name, args.agent.algo)}.pt"


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
    if args.resume and file_exists:
        agent.load(agent_path)
    elif file_exists and not args.resume:
        raise Exception("A model exists at the save path. Use -r to resume training.")
    elif args.resume and not file_exists:
        raise Exception("Resume flag specified, but no model found.")

def create_env_fn(args):
    return lambda: BattleshipEnv(observation_space=args["env"]["state_space"])

def create_agent_from_args(device, args, env):
    actor_kwargs = dict(layers=(256,256), action_dims=env.action_space.nvec)
    args_training = args["training"]
    args_ppo = args_training["ppo"]
    kwargs = dict(discount=args_training["discount"],
                  gae_lam=args_ppo["gae_lam"],
                  clip_ratio=args_ppo["clip_ratio"],
                  actor_learning_rate=args_ppo["actor_learning_rate"],
                  critic_learning_rate=args_ppo["critic_learning_rate"],
                  entropy_coeff=args_ppo["entropy_coeff"],
                  target_kl=args_ppo["target_kl"],
                  actor_type=args["agent"]["actor_type"])
    return PPO(device, env.observation_space.shape[0], actor_kwargs=actor_kwargs, **kwargs), False


def main():
    args = parse_args()

    np.random.seed(args.training.seed)
    torch.manual_seed(args.training.seed)

    env_fn = create_env_fn(args)

    if torch.cuda.is_available() and args.training.use_gpu:
        device = torch.device("cuda", args.training.gpu_idx)
    else:
        device = torch.device("cpu")

    # create environment (optionally using subprocesses)
    if args.training.num_procs == 1:
        env = DummyVecEnv([env_fn])
    else:
        env = SubprocVecEnv([env_fn] * args.training.num_procs)

    agent, continuous = create_agent_from_args(device, args, env)
    load_agent(args, agent)

    writer = SummaryWriter(comment=f"_{args.agent.model_name}")

    save_agent(args, agent)
    while agent.total_it < args.training.total_steps:
        def select_action(greedy, s):
            if greedy:
                return agent.select_action_greedy(s)
            else:
                return agent.select_action(s)

        sample, rollout_info = collect_trajectories_vec_env(env, args.training.train_samples, device,
                                                            partial(select_action, False), agent.get_value,
                                                            max_steps=args.env.max_steps, policy_accepts_batch=False)
        train_info = agent.train(sample, actor_steps=args.training.ppo.actor_steps,
                                 critic_steps=args.training.ppo.critic_steps)

        for name, val in chain(train_info.items(), rollout_info.items()):
            writer.add_scalar(f"Train/{name}", val, agent.total_it)
        print(f"{agent.total_it} - {pretty_dict({**train_info, **rollout_info})}")

        # launch eval
        if args.eval.eval_interval != -1 and agent.total_it % args.eval.eval_interval == 0:
            eval_info = run_evaluation(env_fn, args.eval.num_traj,
                                       partial(select_action, True), max_steps=args.env.max_steps)
            for name, val in eval_info.items():
                writer.add_scalar(f"Eval/{name}", val, agent.total_it)
            print(f"Evaluation - {pretty_dict(eval_info)}")

        if agent.total_it % args.training.save_interval == 0:
            save_agent(args, agent)
    save_agent(args, agent)
    env.close() # cleanup env
    print("Finished training!")


if __name__ == "__main__":
    main()
