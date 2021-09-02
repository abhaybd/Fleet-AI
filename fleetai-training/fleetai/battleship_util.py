import os

from .ppo import PPO
from .actor_critic import MultiDiscActor, Critic

from .BattleshipActor import BattleshipActor
from .env import BattleshipEnv
from .util import run_evaluation_seq, get_or_else


def run_eval(env_fn, actor: BattleshipActor, n_ep, max_steps, render_callback=None, reduce_info=True):
    env = env_fn()

    def select_action(state):
        probs = actor.probs(state).flatten().detach().cpu().numpy()
        options = reversed(sorted(list(enumerate(probs)), key=lambda e: e[1]))
        for action, prob in options:
            if env.can_move(action):
                return action
        raise AssertionError

    return run_evaluation_seq(lambda: env, n_ep, select_action, max_steps=max_steps,
                              render_callback=render_callback, reduce_info=reduce_info)


def create_env_fn(args):
    sizes = args["env"]["ship_sizes"] if "ship_sizes" in args["env"] else (1,2,3,4,5)
    board_width = get_or_else(args["env"], "board_width", 10)
    board_height = get_or_else(args["env"], "board_height", 10)
    # only relevant when using latent variables
    latent_var_precision = get_or_else(args["env"], "latent_var_precision", 16)
    return lambda: BattleshipEnv(observation_space=args["env"]["state_space"],
                                 action_space=args["env"]["action_space"],
                                 latent_var_precision=latent_var_precision,
                                 ships=sizes, board_height=board_height, board_width=board_width)

def create_agent_from_args(device, args, env):
    args_training = args["training"]
    args_ppo = args_training["ppo"]
    act_space = args["env"]["action_space"]
    actor_type = args["agent"]["actor_type"]
    layers = tuple(get_or_else(args["agent"], "layers", (256, 256)))
    if act_space == "coords":
        assert actor_type == "multi-disc", "coords action space requires multi-disc agent"
        actor_fn = lambda dev: MultiDiscActor(dev, env.observation_space.shape[0],
                                              env.action_space.nvec, layers=layers)
    elif act_space == "flat":
        assert actor_type == "disc", "flat action space requires disc agent"
        actor_fn = lambda dev: BattleshipActor(dev, env.observation_space.shape[0], env.action_space.n)
    else:
        raise AssertionError
    critic_fn = lambda dev: Critic(dev, env.observation_space.shape[0], layers=layers)
    ppo = PPO(device=device, actor_fn=actor_fn, critic_fn=critic_fn,
              discount=args_training["discount"],
              gae_lam=args_ppo["gae_lam"],
              clip_ratio=args_ppo["clip_ratio"],
              actor_learning_rate=args_ppo["actor_learning_rate"],
              critic_learning_rate=args_ppo["critic_learning_rate"],
              entropy_coeff=args_ppo["entropy_coeff"],
              target_kl=args_ppo["target_kl"])
    return ppo

def load_agent_from_args(device, model_dir, args):
    env_fn = create_env_fn(args)
    env = env_fn()
    agent = create_agent_from_args(device, args, env)

    model_path = os.path.join(model_dir, f"{args['agent']['algo']}.pt")
    if os.path.isfile(model_path):
        agent.load(model_path)
    else:
        raise Exception(f"{model_path} does not exist!")
    return agent
