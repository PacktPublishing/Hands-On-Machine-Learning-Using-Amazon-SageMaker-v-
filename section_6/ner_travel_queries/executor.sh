#!/usr/bin/env bash

if [ $1 = "train" ]; then
    python ./container/training_and_prediction_code/train
else
    python ./container/training_and_prediction_code/serve
fi