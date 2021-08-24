# Fleet-AI

Available at https://fleet-ai.web.app/

Fleet-AI is a Battleship AI powered by deep reinforcement learning. There are two parts, `fleetai-training`, which trains the neural network, and `fleetai-webapp`, a webapp written in React.js where users can play against the AI.

## Training

Training code is located in the `fleetai-training` directory.

Install dependencies with `pip install -r requirements.txt`. If you want to use CUDA for GPU acceleration, you should set that up as well. Note that if you are not using CUDA/GPU, that should be reflected in the config file.

Model/environment parameters are specified in a config file. An example config file is located in `fleetai-training/config/config.yaml`. When a model is saved, its associated config file is saved along with the parameters. Both the parameter file and the config file can be used to load a model configuration from disk.

The deep RL algorithm used is a custom implementation of [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), with a few [tricks](https://openreview.net/forum?id=r1etN1rtPB) incorporated, such as value function clipping, orthogonal initialization, [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), etc.

You can train a model with `python src/train.py -c config/config.yaml`. Progress can be monitored with `tensorboard ---logdir runs`.

After a model is trained, you can evaluate it using `python src/eval.py [path to model dir]`. Run `python src/eval.py -h` to view the available options.

When the model is finalized, you should convert it to ONNX so it can be served over the web. You can do so with `python src/convert_model.py [path to model dir] [path to output file]`. For deployment, the converted model file should be placed at `fleetai-webapp/public/converted_actor.onnx`.

## Webapp

Code for the webapp, written in React.js, is located in the `fleetai-webapp` directory.

Install dependencies with `npm install`, run locally with `npm run start`, build with `npm run build`, and deploy with `npm run deploy`.

The UI/UX is usable, but not very polished. I'm happy to look at PRs if you feel like contributing! Mobile platforms are also supported, but Firefox may not work.