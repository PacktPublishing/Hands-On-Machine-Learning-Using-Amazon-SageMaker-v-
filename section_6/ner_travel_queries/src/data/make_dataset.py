# -*- coding: utf-8 -*-
import gzip
import os
import logging
import shutil
import urllib.request

import click


_BASE_ATIS_DATA_URL = "https://s3-eu-west-1.amazonaws.com/atis/"
_ATIS_DATA_GZ_NAME_FORMAT = "atis.fold%s.pkl.gz"
_ATIS_DATA_NAME_FORMAT = "atis.fold%s.pkl"


def _download_atis_data(output_path):
    for i in range(5):
        file_name = _ATIS_DATA_GZ_NAME_FORMAT % i
        origin = _BASE_ATIS_DATA_URL + file_name

        save_file = os.path.join(output_path, file_name)
        print('Downloading data %s' % file_name)
        urllib.request.urlretrieve(origin, save_file)


def _decompress_atis_data(input_path, output_path):
    for i in range(5):
        gzip_file_name = _ATIS_DATA_GZ_NAME_FORMAT % i
        decompressed_file_name = _ATIS_DATA_NAME_FORMAT % i
        save_file = os.path.join(output_path, decompressed_file_name)
        print('Decompressing data %s' % gzip_file_name)
        with gzip.open(os.path.join(input_path, gzip_file_name), 'rb') as f_in, \
                open(save_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


@click.command()
@click.argument('interim_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def main(interim_path, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading ATIS Dataset')

    _download_atis_data(interim_path)
    _decompress_atis_data(interim_path, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
