"""
This file uses Ray Serve's batch inference ability to perform
parallel inference on BERT model.

https://docs.ray.io/en/master/serve/tutorials/batch.html
"""
import hashlib
import json
import os
import subprocess

import torch
import numpy as np

import ray
from ray import serve
from transformers import pipeline


def _download_from_s3(s3_uri: str, is_dir: bool = False) -> str:
    assert s3_uri.startswith("s3://")

    cache_dir = f"/tmp/demo_s3_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = cache_dir + "/" + hashlib.sha256(s3_uri.encode()).hexdigest()

    recursive_flag = "--recursive" if is_dir else ""

    if not os.path.exists(cache_path):
        subprocess.run(
            f"aws s3 cp {recursive_flag} {s3_uri} {cache_path}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    return cache_path


class BertFeaturizer:
    def __init__(self, model_path):
        self.bert_model = pipeline(
            "feature-extraction",
            model_path,
            device=0 if torch.cuda.is_available() else -1,
        )

    def __call__(self, request):
        inp = request.args["script"]
        inp = " ".join(inp.split(" ")[:200])
        arr = np.array(self.bert_model(inp)).sum(axis=1).squeeze()
        return arr.tolist()


if __name__ == "__main__":
    # This is expected to be a huggingface model checkpoint
    MODEL_S3_PATH = "s3://summit-demo-dev-data/bert-feat-out"
    model_path = _download_from_s3(MODEL_S3_PATH, is_dir=True)

    client = serve.start()
    client.create_backend(
        "bert-feat",
        BertFeaturizer,
        model_path,
        config={
            "num_replicas": 8,
        },
        ray_actor_options={"num_gpus": 1})
    client.create_endpoint("bert-feat", backend="bert-feat")
    handle = client.get_handle("bert-feat")

    scripts_path = _download_from_s3(
        # Note: this data is not avaiable for public.
        # However, it has the format:
        # {"id_1": "plot 1", "id_2": "plot 2"}
        "/tmp/movie-plot.json")
    loaded = json.load(open(scripts_path))

    ids = json.load(open("/tmp/all-image-ids.json"))

    oids = []
    for i in ids:
        scripts = loaded[i]
        script = " ".join(scripts)
        oids.append(handle.remote(script=script))
    features = ray.get(oids)

    id_to_feat_vector = {i: f for i, f in zip(ids, features)}
    with open('./bert-plot-features.json', 'w') as f:
        json.dump(id_to_feat_vector, f)
