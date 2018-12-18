# -*- coding: utf-8 -*-
import logging

import click

from src.models.data_reader import DataReader
from src.models.ner_model import NERModelTrainer


def train(input_path, output_path, **kwargs):
    data_reader = DataReader(input_path)
    ner_model_trainer = NERModelTrainer(data_reader)
    ner_model_trainer.train(**kwargs)
    ner_model_trainer.persist(output_path)
    ner_model_trainer.evaluate(output_path)


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(input_path, output_path):
    logger = logging.getLogger(__name__)
    logger.info('Training LSTM')

    train(
        input_path,
        output_path,
        epochs=30,
        batch=30,
        dropout=0.15,
        lstm_units=200
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
