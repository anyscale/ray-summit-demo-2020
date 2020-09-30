import time

import ray
from ray import serve

import numpy as np

from movie_recs.serve import retrain_sklearn_lr_model
from movie_recs.serve.deploy_plot import PlotRecommender

@ray.remote
class PeriodicRetrainer:
    def __init__(self):
        self.impressions = ray.get_actor("impressions")
        self.prev_impressions = 0

    def should_retrain(self):
        total_impressions = ray.get(self.impressions.count_for_model.remote("plot"))
        new_impressions = total_impressions - self.prev_impressions
        print(f"Got {new_impressions} new impressions.")
        if new_impressions >= 5:
            self.prev_impressions = new_impressions
            return True
        return False

    def run_loop(self):
        while True:
            if self.should_retrain():
                # Retrain the model.
                print("Retraining model...")
                new_data_df = ray.get(self.impressions.get_model_clicks.remote("plot"))
                new_model = retrain_sklearn_lr_model(new_data_df)

                # Deploy the new model (using incremental rollout).
                client = serve.connect()
                backend_name = f"plot:{int(time.time())}"
                client.create_backend(backend_name, PlotRecommender, new_model)
                client.set_traffic("plot", {"plot:v0": 0.9, backend_name: 0.1})
                print(f"Deployed new backend {backend_name}.")

            time.sleep(1)


ray.init()
actor = PeriodicRetrainer.options(lifetime="detached", name="retrainer").remote()
actor.run_loop.remote()
