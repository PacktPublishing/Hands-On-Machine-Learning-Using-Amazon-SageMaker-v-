# Build an image that can do training and inference in SageMaker
# This is an image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM python:3.6-slim-stretch

MAINTAINER Pavlos Mitsoulis Ntompos <p.mitsoulis@gmail.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
         g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip install numpy==1.14.5 scipy scikit-learn pandas flask gevent gunicorn future
RUN rm -rf /root/.cache

COPY training_and_prediction_code /opt/program
WORKDIR /opt/program
