from functools import partial
import os
import yaml
import json
from datetime import datetime

from tensorboardX import SummaryWriter
import numpy as np

from .ppo import PPOBuffer
from .vec_env import DummyVecEnv, SubprocVecEnv

def get_or_else(d, key, default):
    return d[key] if key in d else default

def create_writer(args):
    if get_or_else(args["logging"], "log_to_comet", False):
        return create_comet_writer(args)
    else:
        return create_disk_or_gcp_writer(args)

def create_disk_or_gcp_writer(args):
    if "log_dir" in args:
        writer = SummaryWriter(log_dir=args["log_dir"])
    else:
        log_base_dir = get_or_else(args["logging"], "log_base_dir", "runs")
        log_dir_name = datetime.now().strftime("%b%d_%H%M%S") + "_" + args["agent"]["model_name"]
        log_dir = log_base_dir + "/" + log_dir_name # don't use join since windows can do / but GCP can't do \
        writer = SummaryWriter(log_dir=log_dir)
        args["log_dir"] = log_dir
    return writer

def create_comet_writer(args):
    comet_env_var = "COMET_APPLICATION_CREDENTIALS"
    assert comet_env_var in os.environ, "Comet Application Credentials not set!"
    with open(os.environ[comet_env_var]) as f:
        comet_config = json.load(f)
        comet_config["disabled"] = False
        return SummaryWriter(comet_config=comet_config, write_to_disk=False)

def get_save_paths(args):
    dir_name = os.path.join(args["agent"]["save_dir"], args["agent"]["model_name"])
    return dir_name, f"{os.path.join(dir_name, args['agent']['algo'])}.pt"

def save_agent(dir_name, args, agent):
    if dir_name is None:
        dir_name, agent_path = get_save_paths(args)
    else:
        agent_path = f"{os.path.join(dir_name, args['agent']['algo'])}.pt"
    os.makedirs(dir_name, exist_ok=True)
    args_path = os.path.join(dir_name, "config.yaml")
    agent.save(agent_path)
    with open(args_path, "w") as f:
        f.write(yaml.dump(args, default_flow_style=False))

def load_agent(dir_name, args, agent):
    if dir_name is None:
        _, agent_path = get_save_paths(args)
    else:
        agent_path = os.path.join(dir_name, args["agent"]["algo"] + ".pt")
    file_exists = os.path.isfile(agent_path)
    resume = ("resume" in args) and args["resume"]
    if resume and file_exists:
        agent.load(agent_path)
    elif file_exists and not resume:
        raise Exception("A model exists at the save path. Use -r to resume training.")
    elif resume and not file_exists:
        raise Exception("Resume flag specified, but no model found.")


def pretty_dict(d, float_fmt="%.6f"):
    return ', '.join(
        [f'{k}={float_fmt % v}' if isinstance(v, float) else f'{k}={v:.0f}'
         for k, v in sorted(d.items(), key=lambda x: x[0])])


def collect_trajectories(env, n_samples, device, policy, value_fn, max_steps=500):
    buf = PPOBuffer()
    state, done = env.reset(), False
    steps = 0
    traj_id = buf.create_traj()
    sum_rews = []
    total_reward = 0
    traj_lens = []
    last_rews = []
    while buf.size() < n_samples:
        action, log_prob = policy(state)
        next_state, reward, done, _ = env.step(action)
        buf.put_single_data(traj_id, state, action, log_prob, reward)
        state = next_state
        total_reward += reward
        steps += 1
        if done or steps == max_steps or buf.size() == n_samples:
            buf.finish_traj(traj_id, 0.0 if done else value_fn(state))
            if buf.size() < n_samples:
                sum_rews.append(total_reward)
                total_reward = 0
                traj_lens.append(steps)
                last_rews.append(reward)
                state, done, steps = env.reset(), False, 0
                traj_id = buf.create_traj()
    rollout_info = {
        "sum_rew_avg": np.mean(sum_rews),
        "traj_len_avg": np.mean(traj_lens),
        "last_rew_avg": np.mean(last_rews)
    }
    return buf.get(device), rollout_info


def collect_trajectories_vec_env(vec_env, n_samples, device, policy, value_fn, max_steps=500,
                                 policy_accepts_batch=False):
    states = vec_env.reset()
    vec_dim = states.shape[0]
    if n_samples % vec_dim != 0:
        raise Exception("The number of samples should be divisible by the number of parallel environments!")
    buf = PPOBuffer()
    traj_ids = [buf.create_traj() for _ in range(vec_dim)]
    steps = np.zeros((vec_dim,))
    # ensure that the policy can accept batches
    if not policy_accepts_batch:
        def batch_policy(p, sts):
            acts, lps = [], []
            for s in sts:
                a, lp = p(s)
                acts.append(a)
                lps.append(lp)
            return np.array(acts), np.array(lps)
        policy = partial(batch_policy, policy)
    # do parallel rollout
    sum_rew_tracker = np.zeros(vec_dim)
    sum_rews = []
    traj_lens = []
    last_rews = []
    while buf.size() < n_samples:
        # get actions and step in environment
        actions, log_probs = policy(states)
        next_states, rewards, dones, _ = vec_env.step(actions)
        sum_rew_tracker += rewards.flatten()
        # put data into buffer
        for i, traj_id in enumerate(traj_ids):
            buf.put_single_data(traj_id, states[i], actions[i], log_probs[i], rewards[i])
        states = next_states
        steps += 1
        # if the buffer is full, finish all trajectories
        if buf.size() >= n_samples:
            assert buf.size() == n_samples, "Number of samples should already have been checked??"
            for i, traj_id in enumerate(traj_ids):
                buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
        else:
            # otherwise, finish the trajectories that are done or have hit the max number of steps
            for i, traj_id in enumerate(traj_ids):
                if dones[i] or steps[i] == max_steps:
                    sum_rews.append(sum_rew_tracker[i])
                    sum_rew_tracker[i] = 0
                    traj_lens.append(steps[i])
                    last_rews.append(rewards[i])
                    buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
                    steps[i] = 0
                    traj_ids[i] = buf.create_traj()
    rollout_info = {
        "sum_rew_avg": np.mean(sum_rews),
        "traj_len_avg": np.mean(traj_lens),
        "last_rew_avg": np.mean(last_rews)
    }
    return buf.get(device), rollout_info


def run_evaluation(env_fn, n_trajectories, policy, max_steps=500, render_callback=None, policy_accepts_batch=False):
    if n_trajectories == 1:
        env = DummyVecEnv([env_fn])
    else:
        env = SubprocVecEnv([env_fn] * n_trajectories)
    states = env.reset()
    vec_dim = states.shape[0]
    # ensure that the policy can accept batches
    if not policy_accepts_batch:
        def batch_policy(p, sts):
            return np.array([p(s) for s in sts])
        policy = partial(batch_policy, policy)
    # do parallel rollout
    steps = np.zeros(vec_dim)
    sum_rews = np.zeros(vec_dim)
    # traj_lens = np.zeros(vec_dim)
    last_rews = np.zeros(vec_dim)
    finished = np.array([False] * vec_dim)
    if render_callback is not None:
        render_callback(env)
    while not finished.all():
        # get actions and step in environment
        actions = policy(states)
        states, rewards, dones, _ = env.step(actions)
        mask = ~finished
        sum_rews[mask] += rewards[mask]  # add rewards to unfinished trajectories
        steps[mask] += 1
        last_rews[mask] = rewards[mask]
        finished = finished | dones | (steps == max_steps)
        if render_callback is not None:
            render_callback(env)
    return dict(sum_rew_avg=np.mean(sum_rews), sum_rew_std=np.std(sum_rews),
                sum_rew_min=np.min(sum_rews), sum_rew_max=np.max(sum_rews),
                traj_len_avg=np.mean(steps), last_rew_avg=np.mean(last_rews),
                norm_rew_avg=np.mean(np.array(sum_rews)/np.array(steps)))


def run_evaluation_seq(env_fn, n_trajectories, policy, max_steps=500, render_callback=None, reduce_info=True):
    env = env_fn()
    traj_rews = []
    traj_lens = []
    last_rews = []
    for _ in range(n_trajectories):
        state, done, steps = env.reset(), False, 0
        if render_callback is not None:
            render_callback(env)
        rewards = []
        while not done and steps < max_steps:
            action = policy(state)
            state, reward, done, _ = env.step(action)
            if render_callback is not None:
                render_callback(env)
            rewards.append(reward)
            steps += 1
        last_rews.append(rewards[-1])
        traj_lens.append(steps)
        traj_rews.append(np.sum(rewards))
    if render_callback is not None:
        env.close()
    if reduce_info:
        return dict(sum_rew_avg=np.mean(traj_rews), sum_rew_std=np.std(traj_rews),
                    sum_rew_min=np.min(traj_rews), sum_rew_max=np.max(traj_rews),
                    traj_len_avg=np.mean(traj_lens), last_rew_avg=np.mean(last_rews),
                    norm_rew_avg=np.mean(np.array(traj_rews)/np.array(traj_lens)))
    else:
        return dict(traj_lens=traj_lens, traj_rews=traj_rews, last_rews=last_rews)
