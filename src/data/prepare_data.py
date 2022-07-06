import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import urllib.request

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    ## Fetch the Wine dataset from the URL and save it in the Raw format
    urllib.request.urlretrieve("https://www.dropbox.com/s/0vvhzsbf0tmjq2s/CarPrice_Dataset_cleaned.csv?dl=1", "./data/CarPrice_Dataset_cleaned.csv")

    logger = logging.getLogger(__name__)
    logger.info('Data set Fetched from the cloud.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()