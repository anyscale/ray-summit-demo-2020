# Ray Summit 2020: Anyscale Product Demo

Welcome to the code repository for the Anyscale Product Demo. The demo illustrates how the Anyscale Platform enables you to build an end-to-end distributed AI application in a matter of minutes.

Unfortunately, we are not able to provide the data used in the demo, so the code here is not easily runnable. This is primarily here for transparency and to allow you to dive deeper into the code if you want to better understand how it works. Feel free to browse and post any questions in the [issues](https://github.com/anyscale/ray-summit-demo-2020/issues) page!

<!-- TODO: fill this in once the video is available. -->
<!-- You can re-watch the demo via this Youtube Link. -->

Overview of the contents of this repository:

```
.
├── movie_recs                 # This Python package is where all the action happens.
│   ├── image_processing       # Code used for converting images to image palettes.
│   │   ├── palettes.py          # Script that uses Ray to scale up image palette generation.
│   │   └── utils.py             # Utilities for loading images, storing the palettes, etc.
│   ├── serve                  # Code used to deploy models and services.
│   │   ├── __init__.py          # Utility functions and definitions to simplify the serving code.
│   │   ├── setup.py             # Starts up Ray Serve and deploys some basic services.
│   │   ├── deploy_color.py      # Color-based recommendation model.
│   │   ├── deploy_plot.py       # Plot-based recommendation model.
│   │   ├── deploy_ensemble.py   # Ensemble model that selects between color and plot models.
│   │   ├── retrain.py           # Deploys the online learning actor.
│   ├── train                  # Code used for hyperparameter tuning the NLP model.
│   │   ├── __init__.py          # Meat of the model definition used by `tune_bert.py`.
│   │   ├── _offline_batch.py    # Generates the embeddings for all of the movies in the dataset.
│   │   ├── _train_lr.py         # Trains the initial logistic regression ranking model.
│   │   └── tune_bert.py         # Entrypoint to use Ray Tune to tune BERT.
├── Dockerfile                 # Our demo runs in Anyscale using Docker.
├── requirements.txt           # Python dependencies.
└── setup.py                   # Python package file.
```
