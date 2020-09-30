from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger

from movie_recs.train import huggingface_bert_trainer


def tune_huggingface_model():
    trainable, model_args = huggingface_bert_trainer()

    analysis = tune.run(
        trainable,
        config={
            "args": model_args,
            "learning_rate": tune.uniform(1e-5, 1e-2),
            "weight_decay": tune.uniform(0.0, 0.3),
            "wandb": {
                "project": "bert-movies",
            },
        },
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        stop={"training_iteration": 20},
        num_samples=8,
        loggers=DEFAULT_LOGGERS + (WandbLogger, ),
    )

    return analysis


if __name__ == "__main__":
    print(tune_huggingface_model())
