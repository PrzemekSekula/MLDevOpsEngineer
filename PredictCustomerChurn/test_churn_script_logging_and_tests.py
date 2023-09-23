"""
Unit tests for churn_library.py
Author: Przemek Sekula
Date: September 2023
"""

import argparse
import logging
import pytest
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models

#### FIXTURES ####


@pytest.fixture(scope='module')
def args():
    """ Arguments for the script
    """
    args_parser = argparse.ArgumentParser()

    data_path = './data/bank_data.csv'
    test_output = './test_output'

    args_parser.add_argument('--data_path',
                             type=str,
                             default=data_path,
                             )
    args_parser.add_argument('--eda_path',
                             type=str,
                             default=test_output,
                             )
    args_parser.add_argument('--model_path',
                             type=str,
                             default=test_output,
                             )
    args_parser.add_argument('--results_path',
                             type=str,
                             default=test_output,
                             )

    # The code below passes only the arguments that are needed for the tests
    # to args. This is done to avoid passing all the arguments from the command
    # line. Useful, if you want to run pytest with additional arguments
    my_args = [
        '--data_path', data_path,
        '--eda_path', test_output,
        '--model_path', test_output,
        '--results_path', test_output,
    ]

    return args_parser.parse_args(args=my_args)


@pytest.fixture(scope='module')
def encoded_cols():
    """ List of categorical columns that need to be encoded by encoder_helper
    """
    return pytest.data.select_dtypes(include='object').columns.tolist()


#### UNIT TESTS ####

def test_import_data(args):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """

    try:
        data = import_data(args.data_path)
    except Exception as exc:
        logging.error('Import data failed %s', exc)
        raise exc

    try:
        assert isinstance(data, pd.DataFrame)
    except AssertionError as exc:
        logging.error('Import data failed - not a dataframe')
        raise AssertionError from exc

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as exc:
        logging.error('Import data failed - dataframe empty')
        raise AssertionError from exc

    try:
        assert 'Churn' in data.columns
    except AssertionError as exc:
        logging.error('Import data failed - label column Churn not in "  \
                "dataframe')
        raise AssertionError from exc

    # We are going to use only first 100 rows, to speed up tests
    pytest.data = data.head(100)
    logging.info('SUCCESS: Loading dataframe successful')


def test_eda(args):
    '''
    test perform eda function
    '''
    try:
        perform_eda(pytest.data, args.eda_path)
    except Exception as exc:
        logging.error('EDA failed %s', exc)
        raise exc
    logging.info('SUCCESS: EDA successful')


def test_encoder_helper(encoded_cols):
    '''
    test encoder helper
    '''
    data = pytest.data
    try:
        encoded_data = encoder_helper(data, encoded_cols)
    except Exception as exc:
        logging.error('Encoder helper failed %s', exc)
        raise exc

    try:
        assert encoded_data.shape[1] == data.shape[1] + len(encoded_cols)
    except AssertionError as exc:
        logging.error('Encoder helper failed - shape mismatch')
        raise AssertionError from exc

    logging.info('SUCCESS: Encoder helper successful')


def test_perform_feature_engineering1():
    '''
    test perform_feature_engineering
    '''
    try:
        train_test_data = perform_feature_engineering(pytest.data)
    except Exception as exc:
        logging.error('Feature engineering failed %s', exc)
        raise exc

    try:
        assert len(train_test_data) == 4
    except AssertionError as exc:
        logging.error('Feature engineering failed - Expected x_train, '
                      'x_test, y_train, y_test')
        raise AssertionError from exc

    x_train, x_test, y_train, y_test = train_test_data

    try:
        assert x_train.shape[1] == x_test.shape[1]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
    except AssertionError as exc:
        logging.error('Feature engineering failed - shapes of '
                      'x_train, x_test, y_train, and y_test do not match')
        raise AssertionError from exc

    pytest.train_test_data = train_test_data
    logging.info('SUCCESS: Feature engineering successful')


def test_train_models(args):
    '''
    test train_models
    '''
    x_train, x_test, y_train, y_test = pytest.train_test_data

    try:
        train_models(
            x_train, x_test, y_train, y_test, args
        )
    except Exception as err:
        logging.error('Train models failed %s', err)
        raise err

    logging.info('SUCCESS: Train models successful')


if __name__ == "__main__":
    print('Just run "pytest"')
