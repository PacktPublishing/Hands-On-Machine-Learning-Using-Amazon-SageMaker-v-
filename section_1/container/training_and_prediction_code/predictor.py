# This is the file that implements a flask server to do inferences. It's
# the file that you will modify to implement the scoring for your own
# algorithm.

from __future__ import print_function

import json
import os
import pickle

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds
# it.
# It has a predict function that does a prediction based on the model and
# the input data.


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not
        already loaded."""
        if cls.model is None:
            with open(os.path.join(model_path, 'model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (numpy.ndarray): The data on which to do the
            predictions."""
        clf = cls.get_model()
        return clf.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample
    container, we declare it healthy if we can load the model
    successfully."""
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(
        response='\n', status=status, mimetype='application/json'
    )


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a JSON data. In this sample server, we take data
     as JSON, convert it to a pandas data frame for internal use and then
     convert the prediction(s) back to JSON. Example:

     {
        "feat_1": [6.3, 6.7, 7.2, 5.7, 5.7, 4.8, 4.6],
        "feat_2": [3.3, 3.1, 3.2, 4.4, 2.6, 3.1, 3.6],
        "feat_3": [4.7, 4.7, 6.0, 1.5, 3.5, 1.6, 1.0],
        "feat_4": [1.6, 1.5, 1.8, 0.4, 1.0, 0.2, 0.2]
     }
    """
    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
    else:
        return flask.Response(
            response='This predictor only supports JSON data',
            status=415,
            mimetype='application/json'
        )

    data_df = pd.DataFrame.from_dict(data)

    predictions = ScoringService.predict(data_df.values)

    resp_json = {
        'predictions': predictions.tolist()
    }

    return flask.Response(
        response=json.dumps(resp_json),
        status=200,
        mimetype='application/json'
    )
