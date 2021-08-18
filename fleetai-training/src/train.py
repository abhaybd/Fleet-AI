import yaml
import argparse
import os
from itertools import chain

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vec_env import DummyVecEnv, SubprocVecEnv
from actor_critic import MultiDiscActor, Critic
from ppo import PPO

from BattleshipActor import BattleshipActor
from env import BattleshipEnv
from util import collect_trajectories_vec_env, run_evaluation_seq, pretty_dict


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


def create_env_fn(args):
    return lambda: BattleshipEnv(observation_space=args["env"]["state_space"],
                                 action_space=args["env"]["action_space"])


def create_agent_from_args(device, args, env):
    args_training = args["training"]
    args_ppo = args_training["ppo"]
    act_space = args["env"]["action_space"]
    actor_type = args["agent"]["actor_type"]
    if act_space == "coords":
        assert actor_type == "multi-disc", "coords action space requires multi-disc agent"
        actor_fn = lambda dev: MultiDiscActor(dev, env.observation_space.shape[0],
                                              env.action_space.nvec, layers=(256, 256))
    elif act_space == "flat":
        assert actor_type == "disc", "flat action space requires disc agent"
        actor_fn = lambda dev: BattleshipActor(dev, env.observation_space.shape[0], env.action_space.n)
    else:
        raise AssertionError
    critic_fn = lambda dev: Critic(dev, env.observation_space.shape[0], layers=(256, 256))
    ppo = PPO(device=device, actor_fn=actor_fn, critic_fn=critic_fn,
              discount=args_training["discount"],
              gae_lam=args_ppo["gae_lam"],
              clip_ratio=args_ppo["clip_ratio"],
              actor_learning_rate=args_ppo["actor_learning_rate"],
              critic_learning_rate=args_ppo["critic_learning_rate"],
              entropy_coeff=args_ppo["entropy_coeff"],
              target_kl=args_ppo["target_kl"])
    return ppo


def run_eval(env_fn, actor: BattleshipActor, n_ep, max_steps):
    env = env_fn()

    def select_action(state):
        probs = actor.probs(state).flatten().detach().cpu().numpy()
        options = reversed(sorted(list(enumerate(probs)), key=lambda e: e[1]))
        for action, prob in options:
            if env.can_move(action):
                return action
        raise AssertionError

    return run_evaluation_seq(lambda: env, n_ep, select_action, max_steps)


def main():
    args = parse_args()

    np.random.seed(args["training"]["seed"])
    torch.manual_seed(args["training"]["seed"])

    env_fn = create_env_fn(args)

    if torch.cuda.is_available() and args["training"]["use_gpu"]:
        device = torch.device("cuda", args["training"]["gpu_idx"])
    else:
        device = torch.device("cpu")

    # create environment (optionally using subprocesses)
    if args["training"]["num_procs"] == 1:
        env = DummyVecEnv([env_fn])
    else:
        env = SubprocVecEnv([env_fn] * args["training"]["num_procs"])

    agent = create_agent_from_args(device, args, env)
    load_agent(args, agent)

    writer = SummaryWriter(comment=f"_{args['agent']['model_name']}")

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
