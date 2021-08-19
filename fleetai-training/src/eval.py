import os
import argparse
from time import sleep

import torch
import yaml

from util import pretty_dict, run_evaluation
from battleship_util import create_agent_from_args, create_env_fn, run_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to directory containing config and model file")
    parser.add_argument("-n", "--num_eval", type=int, default=4)
    parser.add_argument("-ms", "--max_steps", type=int, default=500)
    parser.add_argument("-ts", "--timestep", type=int, default=1000, help="Timestep, in ms")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--render", action="store_true",
                       help="Render to screen (mutually exclusive with -v)")
    # group.add_argument("-v", "--video_path", type=str,
    #                    help="Render to a .webm video and save to this path (mutually exlusive with -r)")
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.yaml")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return args, config


def main():
    args, config = parse_args()

    env_fn = create_env_fn(config)
    env = env_fn()
    device = torch.device("cpu")
    agent = create_agent_from_args(device, config, env)

    model_path = os.path.join(args.model_dir, f"{config['agent']['algo']}.pt")
    if os.path.isfile(model_path):
        agent.load(model_path)
    else:
        raise Exception(f"{args.model_path} does not exist!")

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
    # elif args.video_path is not None:
    #     import cv2
    #     size = (1080, 720)
    #     fps = round(1 / env.control_dt)
    #     out = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    #     render_callback = lambda e: out.write(
    #         cv2.resize(cv2.cvtColor(e.render(mode="rgb_array"), cv2.COLOR_RGB2BGR), size))
    #     def end():
    #         out.release()
    #         vid_dir = os.path.dirname(args.video_path)
    #         vid_name = os.path.basename(args.video_path)
    #         hostname = socket.gethostbyaddr(socket.gethostname())[0]
    #         port = 8000
    #         print(f"You can view the created video at http://{hostname}:{port}/{vid_name}")
    #         subproc = subprocess.Popen(["python", "-m", "http.server", "--directory", vid_dir, str(port)],
    #                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #         def kill_handler(*_):
    #             print("\b\bQuitting...")
    #             subproc.kill()
    #         signal.signal(signal.SIGINT, kill_handler)
    #         subproc.wait()
    #     cleanup_callback = end

    eval_info = run_eval(env_fn, agent.actor, args.num_eval, args.max_steps, render_callback=render_callback)
    # eval_info = run_evaluation_seq(env_fn, n_trajectories=args.num_eval, )
    # eval_info = run_evaluation(env_fn, args.num_eval, policy, max_steps=args.max_steps,
    #                            render_callback=render_callback)
    print("\n\n" + pretty_dict(eval_info, float_fmt="%.2f") + "\n\n")
    if cleanup_callback is not None:
        cleanup_callback()


if __name__ == "__main__":
    main()