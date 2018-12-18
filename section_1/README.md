# Your First Machine Learning Model on SageMaker

This example shows how to train and deploy a scikit-learn based Machine Learning model on Amazon SageMaker.

It is inspired by [Sagify](https://github.com/Kenza-AI/sagify) and [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples).

## Requirements

1. Python 3.6 is installed
2. `mkvirtualenv packt-sagemaker`. Make sure the virtualenv is activated after you create it.
3. `pip install jupyter sagemaker numpy scipy scikit-learn pandas`

## How Amazon SageMaker Runs Training and Prediction

1. `docker run image train`: for training
2. `docker run image serve`: for prediction

## Training

### Data

The [iris](https://archive.ics.uci.edu/ml/datasets/iris) data set is used for training the model.

### Machine Learning Algorithm

[scikit-learn's Random Forest implementation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is chosen in order to train the iris classifier.

### Machine Learning Model Accuracy Metrics

Precision and Recall are used to evaluate the trained ML model.

## Prediction

The following technologies are used in order to build a RESTful prediction service:

1. [nginx](https://www.nginx.com/): a high-performance web server to handle and serve HTTP requests and responses, respectively.
2. [gunicorn](https://gunicorn.org/): a Python WSGI HTTP server responsible to run multiple copies of your application and load balance between them.
3. [flask](http://flask.pocoo.org/): a Python micro web framework that lets you implement the controllers for the two SageMaker endpoints `/ping` and `/invocations`. 

REST Endpoints:

1. `GET /ping`: health endpoint
2. `POST /invocations`: predict endpoint that expects a JSON body with the required features

## Code

1. `train_and_deploy_your_first_model_on_sagemaker.ipynb`: Jupyter notebook to train/deploy your first ML model on SageMaker
