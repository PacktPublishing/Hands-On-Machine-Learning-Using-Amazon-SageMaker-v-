#!/usr/bin/env bash

docker run -it -v ${PWD}/local_test:/opt/ml -p 8080:8080 --rm ner-travel-queries serve