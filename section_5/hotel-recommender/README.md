hotel-recommender
==============================

Hotel recommender based on Expedia data set using Word2Vec logic

Create Environment
------------

1. `make create_environment`
2. `source activate hotel-recommender`
3. `make requirements`

Download Data
------------

1. Download data (`all.zip` file) from https://www.kaggle.com/c/expedia-hotel-recommendations
2. Save `all.zip` under `./data/raw`
3. Unzip `all.zip`

Explore Data
------------

1. Run `make notebook` and choose `data_exploration.ipynb` notebook

Generate Input For ML Model
------------

1. Run `make data`

Train Hotel Cluster Embeddings Model
------------

1. Run `make train`

Explore ML Model
------------

1. Run `make notebook` and choose `model_exploration.ipynb` notebook

Build Docker Image
------------

1. Run `make build_image`

Train Hotel Cluster Embeddings Model in Docker Locally
------------

1. Run `make train_image_locally`

Deploy Hotel Cluster Embeddings Model in Docker Locally
------------

1. Run `make deploy_image_locally`

Push Docker Image
------------

1. Run `make push_image`

Train Hotel Cluster Embeddings Model on SageMaker
------------

1. Run `make train_on_sagemaker hyperparameters_json_file=<hyperparameters.json>`

Deploy Hotel Cluster Embeddings Model on SageMaker
------------

1. Run `make deploy_on_sagemaker model_location=<s3-model-location>`

Deploy Hotel Cluster Embeddings Endpoints for A/B Testing on SageMaker
------------

1. Run `make deploy_ab_test_on_sagemaker s3_model_location_a=<s3-model-a-location> s3_model_location_b=<s3-model-b-location> model_a_weight==<model-a-weight> model_b_weight=model-b-weight`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
