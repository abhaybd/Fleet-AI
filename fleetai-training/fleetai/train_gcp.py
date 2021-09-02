import shutil
import os

import yaml
from google.cloud import storage
from parse import parse

from .train_base import train
from . import util

LOCAL_MODEL_DIR = "model"
LOCAL_ZIP_DIR = "."

client = storage.Client()

def get_bucket_and_path(uri):
    bucket, path = parse("gs://{}/{}", uri)
    return bucket, path

def uri_to_blob(uri):
    return get_blob(*get_bucket_and_path(uri))

def get_blob(bucket, name):
    return client.bucket(bucket).blob(name)

def local_zip_path(args, base=False):
    name = args["agent"]["model_name"] + (".zip" if not base else "")
    return os.path.join(LOCAL_ZIP_DIR, name)

def save_agent(args, agent):
    util.save_agent(LOCAL_MODEL_DIR, args, agent)
    zip_file_local_base = local_zip_path(args, base=True)
    zip_name = os.path.basename(zip_file_local_base) + ".zip"
    zip_file_remote = os.path.join(args["agent"]["save_dir"], zip_name)
    shutil.make_archive(zip_file_local_base, "zip", LOCAL_MODEL_DIR)
    blob = uri_to_blob(zip_file_remote)
    blob.upload_from_filename(zip_file_local_base + ".zip")

def load_agent(args, agent):
    resume = "resume" in args and args["resume"]
    local_zip = local_zip_path(args, base=False)
    zip_name = os.path.basename(local_zip)
    remote_zip = os.path.join(args["agent"]["save_dir"], zip_name)
    blob = uri_to_blob(remote_zip)
    exists = blob.exists()
    if resume and exists:
        blob.download_to_filename(local_zip)
        shutil.unpack_archive(local_zip, "zip", LOCAL_MODEL_DIR)
        util.load_agent(LOCAL_MODEL_DIR, args, agent)
    elif exists and not resume:
        raise Exception("A model exists at the save path. Use -r to resume training.")
    elif resume and not exists:
        raise Exception("Resume flag specified, but no model found.")

def read_config(path):
    blob = uri_to_blob(path)
    blob.download_to_filename("tmp/config.yaml")
    with open("tmp/config.yaml") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    train(save_agent, load_agent, read_config)
