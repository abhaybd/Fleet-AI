# Fleet-AI

Available at https://fleet-ai.web.app/

Fleet-AI is a Battleship AI powered by deep reinforcement learning. There are two parts, `fleetai-training`, which trains the neural network, and `fleetai-webapp`, a webapp written in React.js where users can play against the AI.

## Training

Training code is located in the `fleetai-training` directory. You can either train locally, or submit training jobs to Google Cloud AI-Platform.

### Train Locally

Install dependencies with `pip install -r requirements.txt`. If you want to use CUDA for GPU acceleration, you should set that up as well. Note that if you are not using CUDA/GPU, that should be reflected in the config file. Note that you should not use the `setup.py` file in any way, as that is intended for packing for GCP.

Model/environment parameters are specified in a config file. An example config file is located in `fleetai-training/config/config.yaml`. When a model is saved, its associated config file is saved along with the parameters. Both the parameter file and the config file can be used to load a model configuration from disk.

The deep RL algorithm used is a custom implementation of [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), with a few [tricks](https://openreview.net/forum?id=r1etN1rtPB) incorporated, such as value function clipping, orthogonal initialization, [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), etc.

Unless otherwise specified, the following commands are run from the `fleetai-training` folder.

You can train a model with `python -m fleetai.train -c config/config.yaml`. Progress can be monitored with `tensorboard ---logdir runs`.

After a model is trained, you can evaluate it using `python -m fleetai.eval [path to model dir]`. Run `python -m fleetai.eval -h` to view the available options.

When the model is finalized, you should convert it to ONNX so it can be served over the web. You can do so with `python -m fleetai.convert_model [path to model dir] [path to output file]`. For deployment, the converted model file should be placed at `fleetai-webapp/public/converted_actor.onnx`.

### Train Remotely

The following instructions are for Windows systems, but can be adapted for other systems pretty easily.

First, create a new Google Cloud Project. Then, create a Cloud Storage Bucket. Take note of the name of the bucket, and change `train_remote.bat` to reflect the new bucket name. (Change the line `set BUCKET=fleetai-storage`)

Then, install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), marking the new project as the default.

Now, you can submit training jobs with `train_remote.bat [JOB NAME] [CONFIG PATH]`. Note that the job name must be unique for each invocation. Also, in the config file, your `agent.save_dir` attribute must be a path to a folder in a Bucket. (i.e. of the form `gs://BUCKET_NAME/PATH_TO_DIR`)

The current setup uses CPU to train, but if you want to enable GPU acceleration you should change the `IMAGE_URI` variable in `train_remote.bat` to `gcr.io/cloud-ml-public/training/pytorch-gpu.1-7`.

Additionally, resuming training of a pretrained model is possible. To do so, simply add `-r` to the end of the parameter list of the `gcloud ai-platform jobs submit training` line.

## Webapp

Code for the webapp, written in React.js, is located in the `fleetai-webapp` directory.

Install dependencies with `npm install`, run locally with `npm run start`, build with `npm run build`, and deploy with `npm run deploy`.

The UI/UX is usable, but not very polished. I'm happy to look at PRs if you feel like contributing! Mobile platforms are also supported, but Firefox may not work.