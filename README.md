# Fleet-AI

Available at https://fleet-ai.web.app/

Fleet-AI is a Battleship AI powered by deep reinforcement learning. There are two parts, `fleetai-training`, which trains the neural network, and `fleetai-webapp`, a webapp written in React.js where users can play against the AI.

## Training + Evaluating Models

Training code is located in the `fleetai-training` directory. You can either train locally, or submit training jobs to Google Cloud AI-Platform.

### Training

#### Train Locally

Install dependencies with `pip install -r requirements.txt`. If you want to use CUDA for GPU acceleration, you should set that up as well. Note that if you are not using CUDA/GPU, that should be reflected in the config file. Note that you should not use the `setup.py` file in any way, as that is intended for packing for GCP.

Model/environment parameters are specified in a config file. An example config file is located in `fleetai-training/config/config.yaml`. When a model is saved, its associated config file is saved along with the parameters. Both the parameter file and the config file can be used to load a model configuration from disk.

The deep RL algorithm used is a custom implementation of [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), with a few [tricks](https://openreview.net/forum?id=r1etN1rtPB) incorporated, such as value function clipping, orthogonal initialization, [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), etc.

Unless otherwise specified, the following commands are run from the `fleetai-training` folder.

You can train a model with `python -m fleetai.train -c config/config.yaml`. Progress can be monitored with `tensorboard --logdir runs`. 

If you wish to resume a previously stopped train job, you can add the `-r` flag. It will load the model stored at the model save location and resume training it. The easiest way to do so is `python -m fleetai.train -c models/MODEL_DIR/config.yaml -r`.

#### Train Remotely

The following instructions are for Windows systems, but can be adapted for other systems pretty easily.

First, create a new Google Cloud Project. Then, create a Cloud Storage Bucket. Take note of the name of the bucket, and change `train_remote.bat` to reflect the new bucket name. (Change the line `set BUCKET=fleetai-storage`)

Then, install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), marking the new project as the default.

Now, you can submit training jobs with `train_remote.bat [JOB NAME] [CONFIG PATH]`. Note that the job name must be unique for each invocation. Also, in the config file, your `agent.save_dir` attribute must be a path to a folder in a Bucket. (i.e. of the form `gs://BUCKET_NAME/PATH_TO_DIR`)

Due to some issue, GPU acceleration is not supported when training remotely.

Additionally, resuming training of a pretrained model is possible. To do so, simply add `-r` to the end of the parameter list of the `gcloud ai-platform jobs submit training` line.

When a job is finished, you can download all stored models from the bucket to the `models` folder with `download_models.bat models`.

### Monitoring Training Jobs

#### Writing logs

Logs can be written to a variety of places for local or remote monitoring. It's highly extensible as well, if you need to add your own. Currently, logging to the following places is supported:

- Local Directory
- Google Cloud Storage Bucket
- [Comet-ML](https://www.comet.ml/)

Technically, all three can be used for either local or remote training, but writing to a local directory isn't useful at all for remote training.

If, for some reason, you don't want any logs, simply set `disabled: true` in the `logging` section of the config file.

For writing to a local directory or a bucket, ensure that `log_to_comet: false` is set in the config file, and the `log_base_dir` is set either to a local path (for writing to a local directory) or the URI to the bucket location (`gs://BUCKET_NAME/PATH`)

When writing to a bucket, you must ensure that Google Cloud authentication is set up. When training remotely, this is automatically handled. However, when you're training locally you must provision keys to a service account and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to the key. Read more about this [here](https://cloud.google.com/docs/authentication/getting-started).

For writing to Comet, ensure that `log_to_comet: true` is set. The value of `log_base_dir` is irrelevant, and will not be used. Similarly to Google Cloud, you must set up authentication. Create a JSON file with the following keys:

- `api_key`
- `workspace` (typically your username)
- `project_name`

Then, set the `COMET_APPLICATION_CREDENTIALS` environment variable to point to the newly created file. Alternatively, you can also place the JSON file at `fleetai/static/comet.json`. If both are provided, the environment variable will be preferred.

#### Viewing logs

When viewing logs in a local directory, you can do so with `tensorboard --logdir [LOG_BASE_DIR]`. 

Viewing logs stored on a Bucket is mostly the same, and can either by done with a) `tensorboard --logdir gs://BUCKET_NAME/PATH` or b) setting up a Compute VM in the same region and running tensorboard on that VM. (a) may incur excessive bandwidth usage and has less platform support, while (b) is more work to set up and may incur resource usage charges.

Logs on Comet can be viewed through the web interface.

### Evaluation

After a model is trained, you can evaluate it using `python -m fleetai.eval [path to model dir]`. Run `python -m fleetai.eval -h` to view the available options.

When the model is finalized, you should convert it to ONNX so it can be served over the web. You can do so with `python -m fleetai.convert_model [path to model dir] [path to output file]`. For deployment, the converted model file should be placed at `fleetai-webapp/public/converted_actor.onnx`.

## Webapp

Code for the webapp, written in React.js, is located in the `fleetai-webapp` directory.

Install dependencies with `npm install`, run locally with `npm run start`, build with `npm run build`, and deploy with `npm run deploy`.

The UI/UX is usable, but not very polished. I'm happy to look at PRs if you feel like contributing! Mobile platforms are also supported, but Firefox may not work.