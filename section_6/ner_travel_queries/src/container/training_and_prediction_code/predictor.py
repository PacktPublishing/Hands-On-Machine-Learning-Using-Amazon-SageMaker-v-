# This is the file that implements a flask server to do inferences. It's
# the file that you will modify to implement the scoring for your own
# algorithm.

from __future__ import print_function

import json
import os
import sys;sys.path.append("..")

import flask
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy

from src.models.data import get_raw_data

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds
# it.
# It has a predict function that does a prediction based on the model and
# the input data.


class ScoringService(object):
    model, w2idx, idx2label = None, None, None
    max_train_sentence_length = 50

    @classmethod
    def encode_sentence(cls, sentence_tokens, skip_unknown_words=False):
        X = []
        outer_X = []
        for word in sentence_tokens:
            if word in cls.get_w2idx():
                X.append(cls.get_w2idx()[word])

            elif skip_unknown_words:
                continue

            else:
                X.append(0)

        outer_X.append(X)
        return pad_sequences(outer_X, maxlen=cls.max_train_sentence_length, dtype=numpy.float64)

    @classmethod
    def decode_single_label(cls, encoded_label):
        return cls.get_idx2label()[numpy.argmax(encoded_label)]

    @classmethod
    def decode_labels(cls, encoded_labels):
        return [cls.decode_single_label(class_prs) for class_prs in encoded_labels]

    @classmethod
    def get_w2idx(cls):
        if cls.w2idx is None:
            input_file_path = os.path.join(model_path, 'w2idx.pkl')
            with open(input_file_path, 'rb') as _in:
                cls.w2idx = pickle.load(_in)

        return cls.w2idx

    @classmethod
    def get_idx2label(cls):
        if cls.idx2label is None:
            input_file_path = os.path.join(model_path, 'idx2label.pkl')
            with open(input_file_path, 'rb') as _in:
                cls.idx2label = pickle.load(_in)

        return cls.idx2label

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not
        already loaded."""
        if cls.model is None:
            import keras

            model_file_path = os.path.join(model_path, 'model.h5')
            cls.model = keras.models.load_model(model_file_path)
        return cls.model

    @classmethod
    def predict(cls, sentence_tokens):
        """For the input, do the predictions and return them.

        Args:
            sentence_tokens (list[str]): list of sentence tokens
        """
        model = cls.get_model()

        encoded_sentence = cls.encode_sentence(sentence_tokens)

        encoded_prediction = model.predict(encoded_sentence)

        return cls.decode_labels(
            encoded_prediction[0][-len(sentence_tokens):]
        )


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(
        response='\n', status=200, mimetype='application/json'
    )


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Example:

     {
        "sentence_tokens": [
            "show",
            "me",
            "the",
            "cheapest",
            "round",
            "trips",
            "from",
            "dallas",
            "to",
            "baltimore"
        ]
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

    predictions = ScoringService.predict(
        data['sentence_tokens']
    )

    resp_json = {
        'predictions': predictions
    }

    return flask.Response(
        response=json.dumps(resp_json),
        status=200,
        mimetype='application/json'
    )
